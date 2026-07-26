//===- ConvertIntentPaddedRowReductionToGPU.cpp ---------------------------===//
//
// Structural single-launch lowering for a masked short row dot followed by
// zero padding.  This family is intentionally keyed only by typed MLIR graph
// shape/dataflow; function names and frontend/source identities are absent.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <string>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTPADDEDROWREDUCTIONTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static bool isZeroSplat(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                        : DenseElementsAttr();
  if (!dense || !dense.isSplat())
    return false;
  if (isa<FloatType>(dense.getElementType()))
    return dense.getSplatValue<APFloat>().isZero();
  if (isa<IntegerType>(dense.getElementType()))
    return dense.getSplatValue<APInt>().isZero();
  return false;
}

static SmallVector<int64_t> integerArray(ArrayAttr attributes) {
  SmallVector<int64_t> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    if (!integer)
      return {};
    result.push_back(integer.getInt());
  }
  return result;
}

static Value stripUnitExpansion(Value value) {
  while (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto source = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    auto result = dyn_cast<RankedTensorType>(expand.getResult().getType());
    if (!source || !result || !source.hasStaticShape() ||
        !result.hasStaticShape() ||
        source.getElementType() != result.getElementType())
      return {};
    SmallVector<int64_t> sourceNonUnit;
    SmallVector<int64_t> resultNonUnit;
    for (int64_t extent : source.getShape())
      if (extent != 1)
        sourceNonUnit.push_back(extent);
    for (int64_t extent : result.getShape())
      if (extent != 1)
        resultNonUnit.push_back(extent);
    if (sourceNonUnit != resultNonUnit)
      return {};
    value = expand.getSrc();
  }
  return value;
}

struct PaddedRowDotLowering {
  func::FuncOp function;
  unsigned requestedThreads;
  bool requestedVectorizeFullFeatures;
  bool requestedPromoteRank0Scalars;

  int64_t rows = 0;
  int64_t physicalRows = 0;
  int64_t featureWidth = 0;
  int64_t paddedRows = 0;
  int64_t threads = 0;
  int64_t blocksPerRow = 0;
  Type storageType;
  unsigned validRowsArgument = 0;
  unsigned validFeaturesArgument = 0;

  PadOp pad;
  ReduceOp reduce;
  MapBinaryOp product;
  SelectOp leftSelect;
  SelectOp rightSelect;

  explicit PaddedRowDotLowering(func::FuncOp function,
                                unsigned requestedThreads,
                                bool requestedVectorizeFullFeatures,
                                bool requestedPromoteRank0Scalars)
      : function(function), requestedThreads(requestedThreads),
        requestedVectorizeFullFeatures(requestedVectorizeFullFeatures),
        requestedPromoteRank0Scalars(requestedPromoteRank0Scalars) {}

  bool matchBound(Value value, int64_t extent, unsigned &argument) {
    value = stripUnitExpansion(value);
    auto compare = value ? value.getDefiningOp<MapBinaryOp>() : MapBinaryOp();
    if (!compare || compare.getFn() != "lt")
      return false;
    auto iota = compare.getLhs().getDefiningOp<IotaOp>();
    auto bound = dyn_cast<BlockArgument>(compare.getRhs());
    auto iotaType = iota
                        ? dyn_cast<RankedTensorType>(iota.getResult().getType())
                        : RankedTensorType();
    auto boundType = bound
                         ? dyn_cast<RankedTensorType>(bound.getType())
                         : RankedTensorType();
    if (!iota || !bound || !iotaType || !boundType ||
        !iotaType.hasStaticShape() || iotaType.getRank() != 1 ||
        iotaType.getDimSize(0) != extent || boundType.getRank() != 0 ||
        !isa<IntegerType>(iotaType.getElementType()) ||
        iotaType.getElementType() != boundType.getElementType())
      return false;
    argument = bound.getArgNumber();
    return true;
  }

  LogicalResult match() {
    auto type = function.getFunctionType();
    if (type.getNumInputs() != 4 || type.getNumResults() != 1 ||
        !function.getBody().hasOneBlock())
      return failure();
    auto returned = dyn_cast<func::ReturnOp>(
        function.getBody().front().getTerminator());
    if (!returned || returned.getNumOperands() != 1)
      return failure();

    pad = returned.getOperand(0).getDefiningOp<PadOp>();
    reduce = pad ? pad.getInput().getDefiningOp<ReduceOp>() : ReduceOp();
    product = reduce ? reduce.getInput().getDefiningOp<MapBinaryOp>()
                     : MapBinaryOp();
    if (!pad || !reduce || !product || product.getFn() != "mul" ||
        reduce.getFn() != "sum" || reduce.getAxes().size() != 1 ||
        cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 3)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(pad.getResult().getType());
    auto reducedType = dyn_cast<RankedTensorType>(reduce.getResult().getType());
    auto leftCast = product.getLhs().getDefiningOp<CastOp>();
    auto rightCast = product.getRhs().getDefiningOp<CastOp>();
    leftSelect = leftCast ? leftCast.getInput().getDefiningOp<SelectOp>()
                          : SelectOp();
    rightSelect = rightCast ? rightCast.getInput().getDefiningOp<SelectOp>()
                            : SelectOp();
    if (!outputType || !reducedType || !outputType.hasStaticShape() ||
        !reducedType.hasStaticShape() || outputType.getRank() != 3 ||
        reducedType.getRank() != 3 || !isa<Float32Type>(outputType.getElementType()) ||
        !isa<Float32Type>(reducedType.getElementType()) || !leftCast ||
        !rightCast || !leftSelect || !rightSelect)
      return failure();

    auto leftInput = dyn_cast<BlockArgument>(leftSelect.getX());
    auto rightInput = dyn_cast<BlockArgument>(rightSelect.getX());
    if (!leftInput || !rightInput || leftInput.getArgNumber() == rightInput.getArgNumber() ||
        !isZeroSplat(leftSelect.getY()) || !isZeroSplat(rightSelect.getY()))
      return failure();
    auto leftType = dyn_cast<RankedTensorType>(leftInput.getType());
    auto rightType = dyn_cast<RankedTensorType>(rightInput.getType());
    if (!leftType || !rightType || !leftType.hasStaticShape() ||
        leftType != rightType || leftType.getRank() != 4 ||
        leftType.getDimSize(0) != 1 || leftType.getDimSize(1) <= 0 ||
        leftType.getDimSize(2) <= 0 || leftType.getDimSize(3) <= 0 ||
        !isa<Float16Type, BFloat16Type, Float32Type>(
            leftType.getElementType()))
      return failure();
    unsigned leftArgument = leftInput.getArgNumber();
    unsigned rightArgument = rightInput.getArgNumber();
    if (!((leftArgument == 0 && rightArgument == 1) ||
          (leftArgument == 1 && rightArgument == 0)))
      return failure();

    rows = leftType.getDimSize(1);
    physicalRows = leftType.getDimSize(2);
    featureWidth = leftType.getDimSize(3);
    paddedRows = outputType.getDimSize(2);
    storageType = leftType.getElementType();
    if (outputType.getDimSize(0) != 1 || outputType.getDimSize(1) != rows ||
        reducedType.getShape() !=
            ArrayRef<int64_t>({1, rows, physicalRows}) ||
        paddedRows < physicalRows || featureWidth > 32)
      return failure();

    auto low = integerArray(pad.getLo());
    auto high = integerArray(pad.getHi());
    if (low != SmallVector<int64_t>({0, 0, 0}) || high.size() != 3 ||
        high[0] != 0 || high[1] != 0 ||
        high[2] != paddedRows - physicalRows ||
        !pad.getValue().isZero())
      return failure();

    Value leftCondition = stripUnitExpansion(leftSelect.getCond());
    Value rightCondition = stripUnitExpansion(rightSelect.getCond());
    if (!leftCondition || !rightCondition || leftCondition != rightCondition)
      return failure();
    auto conjunction = leftCondition.getDefiningOp<MapBinaryOp>();
    if (!conjunction || conjunction.getFn() != "logical_and")
      return failure();
    unsigned firstArgument = 0;
    unsigned secondArgument = 0;
    bool firstRows = matchBound(conjunction.getLhs(), physicalRows,
                                firstArgument);
    bool firstFeatures = matchBound(conjunction.getLhs(), featureWidth,
                                    firstArgument);
    bool secondRows = matchBound(conjunction.getRhs(), physicalRows,
                                 secondArgument);
    bool secondFeatures = matchBound(conjunction.getRhs(), featureWidth,
                                     secondArgument);
    if (firstRows && secondFeatures) {
      validRowsArgument = firstArgument;
      validFeaturesArgument = secondArgument;
    } else if (firstFeatures && secondRows) {
      validFeaturesArgument = firstArgument;
      validRowsArgument = secondArgument;
    } else {
      return failure();
    }
    if (validRowsArgument < 2 || validFeaturesArgument < 2 ||
        validRowsArgument == validFeaturesArgument)
      return failure();

    llvm::DenseSet<Operation *> reachable;
    std::function<void(Value)> visit = [&](Value value) {
      Operation *definition = value.getDefiningOp();
      if (!definition || !reachable.insert(definition).second)
        return;
      for (Value operand : definition->getOperands())
        visit(operand);
    };
    visit(returned.getOperand(0));
    for (Operation &operation :
         function.getBody().front().without_terminator())
      if (!reachable.contains(&operation))
        return failure();

    threads = requestedThreads;
    if (threads == 0) {
      threads = 32;
      while (threads < paddedRows)
        threads *= 2;
    }
    if (threads < 32 || threads > 1024 || threads % 32 != 0)
      return failure();
    blocksPerRow = (paddedRows + threads - 1) / threads;
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *context = function.getContext();
    Location location = function.getLoc();
    OpBuilder builder(context);
    auto originalType = function.getFunctionType();
    SmallVector<Type> inputs;
    for (Type input : originalType.getInputs()) {
      auto tensor = cast<RankedTensorType>(input);
      inputs.push_back(MemRefType::get(tensor.getShape(),
                                       tensor.getElementType()));
    }
    auto output = cast<RankedTensorType>(originalType.getResult(0));
    inputs.push_back(MemRefType::get(output.getShape(),
                                     output.getElementType()));

    std::string name = function.getName().str();
    function.setName(name + "_intent_src");
    builder.setInsertionPoint(function);
    auto lowered = builder.create<func::FuncOp>(
        location, name, FunctionType::get(context, inputs, {}));
    Block *entry = lowered.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value grid = builder.create<arith::ConstantIndexOp>(
        location, rows * blocksPerRow);
    Value block = builder.create<arith::ConstantIndexOp>(location, threads);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    // Rank-0 invocation scalars are loaded in the host launch stub and captured
    // by value.  Loading them inside gpu.launch would issue one redundant
    // global-memory transaction per output thread and would not match the
    // scalar-by-value ABI used by the source frontend.
    Value validRowsRaw;
    Value validFeaturesRaw;
    if (requestedPromoteRank0Scalars) {
      validRowsRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validRowsArgument), ValueRange{});
      validFeaturesRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validFeaturesArgument), ValueRange{});
    }
    auto launch = builder.create<gpu::LaunchOp>(location, grid, one, one,
                                                 block, one, one);
    builder.setInsertionPointToStart(&launch.getBody().front());
    Value blockId = launch.getBlockIds().x;
    Value threadId = launch.getThreadIds().x;
    Value c0 = builder.create<arith::ConstantIndexOp>(location, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(location, 1);
    Value cBlocksPerRow = builder.create<arith::ConstantIndexOp>(
        location, blocksPerRow);
    Value cThreads =
        builder.create<arith::ConstantIndexOp>(location, threads);
    Value cPhysicalRows =
        builder.create<arith::ConstantIndexOp>(location, physicalRows);
    Value cPaddedRows =
        builder.create<arith::ConstantIndexOp>(location, paddedRows);
    Value cFeatureWidth =
        builder.create<arith::ConstantIndexOp>(location, featureWidth);
    Value row = builder.create<arith::DivUIOp>(location, blockId,
                                                cBlocksPerRow);
    Value rowBlock = builder.create<arith::RemUIOp>(location, blockId,
                                                     cBlocksPerRow);
    Value rowBase =
        builder.create<arith::MulIOp>(location, rowBlock, cThreads);
    Value paddedRow =
        builder.create<arith::AddIOp>(location, rowBase, threadId);
    Value zero = builder.create<arith::ConstantFloatOp>(
        location, APFloat(0.0f), builder.getF32Type());

    Value outputActive = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, paddedRow, cPaddedRows);
    auto outputIf = builder.create<scf::IfOp>(
        location, outputActive, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(outputIf.thenBlock());
    if (!requestedPromoteRank0Scalars) {
      validRowsRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validRowsArgument), ValueRange{});
      validFeaturesRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validFeaturesArgument), ValueRange{});
    }
    Value validRows = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), validRowsRaw);
    Value validFeatures = builder.create<arith::IndexCastOp>(
        location, builder.getIndexType(), validFeaturesRaw);
    Value inPhysicalRows = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, paddedRow, cPhysicalRows);
    Value inValidRows = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, paddedRow, validRows);
    Value rowActive = builder.create<arith::AndIOp>(
        location, inPhysicalRows, inValidRows);

    auto emitScalarProduct = [&](OpBuilder &nested, Value column) -> Value {
      SmallVector<Value> indices{c0, row, paddedRow, column};
      Value left = nested.create<memref::LoadOp>(
          location, entry->getArgument(0), indices);
      Value right = nested.create<memref::LoadOp>(
          location, entry->getArgument(1), indices);
      if (storageType != builder.getF32Type()) {
        left = nested.create<arith::ExtFOp>(location,
                                            builder.getF32Type(), left);
        right = nested.create<arith::ExtFOp>(location,
                                             builder.getF32Type(), right);
      }
      return nested.create<arith::MulFOp>(location, left, right);
    };

    auto valueIf = builder.create<scf::IfOp>(
        location, TypeRange{builder.getF32Type()}, rowActive,
        /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard rowGuard(builder);
      builder.setInsertionPointToStart(valueIf.thenBlock());
      Value allFeatures = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::sge, validFeatures,
          cFeatureWidth);
      auto featureIf = builder.create<scf::IfOp>(
          location, TypeRange{builder.getF32Type()}, allFeatures,
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard featureGuard(builder);
        builder.setInsertionPointToStart(featureIf.thenBlock());
        Value accumulator = zero;
        int64_t vectorWidth = isa<Float32Type>(storageType) ? 4 : 8;
        if (requestedVectorizeFullFeatures &&
            featureWidth % vectorWidth == 0) {
          auto vectorType = VectorType::get({vectorWidth}, storageType);
          for (int64_t base = 0; base < featureWidth; base += vectorWidth) {
            Value column = builder.create<arith::ConstantIndexOp>(
                location, base);
            SmallVector<Value> indices{c0, row, paddedRow, column};
            Value leftVector = builder.create<vector::LoadOp>(
                location, vectorType, entry->getArgument(0), indices);
            Value rightVector = builder.create<vector::LoadOp>(
                location, vectorType, entry->getArgument(1), indices);
            for (int64_t lane = 0; lane < vectorWidth; ++lane) {
              Value left = builder.create<vector::ExtractOp>(
                  location, leftVector, ArrayRef<int64_t>{lane});
              Value right = builder.create<vector::ExtractOp>(
                  location, rightVector, ArrayRef<int64_t>{lane});
              if (storageType != builder.getF32Type()) {
                left = builder.create<arith::ExtFOp>(
                    location, builder.getF32Type(), left);
                right = builder.create<arith::ExtFOp>(
                    location, builder.getF32Type(), right);
              }
              Value term =
                  builder.create<arith::MulFOp>(location, left, right);
              accumulator = builder.create<arith::AddFOp>(
                  location, accumulator, term);
            }
          }
        } else {
          for (int64_t column = 0; column < featureWidth; ++column) {
            Value columnIndex = builder.create<arith::ConstantIndexOp>(
                location, column);
            Value term = emitScalarProduct(builder, columnIndex);
            accumulator = builder.create<arith::AddFOp>(
                location, accumulator, term);
          }
        }
        builder.create<scf::YieldOp>(location, accumulator);
      }
      {
        OpBuilder::InsertionGuard featureGuard(builder);
        builder.setInsertionPointToStart(featureIf.elseBlock());
        auto loop = builder.create<scf::ForOp>(
            location, c0, cFeatureWidth, c1, ValueRange{zero});
        builder.setInsertionPointToStart(loop.getBody());
        Value column = loop.getInductionVar();
        Value accumulator = loop.getRegionIterArgs()[0];
        Value featureActive = builder.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::slt, column, validFeatures);
        auto termIf = builder.create<scf::IfOp>(
            location, TypeRange{builder.getF32Type()}, featureActive,
            /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard termGuard(builder);
          builder.setInsertionPointToStart(termIf.thenBlock());
          builder.create<scf::YieldOp>(
              location, emitScalarProduct(builder, column));
        }
        {
          OpBuilder::InsertionGuard termGuard(builder);
          builder.setInsertionPointToStart(termIf.elseBlock());
          builder.create<scf::YieldOp>(location, zero);
        }
        builder.setInsertionPointAfter(termIf);
        Value next = builder.create<arith::AddFOp>(
            location, accumulator, termIf.getResult(0));
        builder.create<scf::YieldOp>(location, next);
        builder.setInsertionPointAfter(loop);
        builder.create<scf::YieldOp>(location, loop.getResult(0));
      }
      builder.setInsertionPointAfter(featureIf);
      builder.create<scf::YieldOp>(location, featureIf.getResult(0));
    }
    {
      OpBuilder::InsertionGuard rowGuard(builder);
      builder.setInsertionPointToStart(valueIf.elseBlock());
      builder.create<scf::YieldOp>(location, zero);
    }
    builder.setInsertionPointAfter(valueIf);
    builder.create<memref::StoreOp>(
        location, valueIf.getResult(0), entry->getArgument(4),
        ValueRange{c0, row, paddedRow});
    builder.setInsertionPointAfter(outputIf);

    builder.create<gpu::TerminatorOp>(location);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(location);
    function.erase();
    return success();
  }
};

struct ConvertIntentPaddedRowReductionToGPUPass
    : ::intent::impl::ConvertIntentPaddedRowReductionToGPUBase<
          ConvertIntentPaddedRowReductionToGPUPass> {
  using ConvertIntentPaddedRowReductionToGPUBase::
      ConvertIntentPaddedRowReductionToGPUBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> targets;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      PaddedRowDotLowering probe(function, blockThreads,
                                 vectorizeFullFeatures,
                                 promoteRank0Scalars);
      if (succeeded(probe.match()))
        targets.push_back(function);
    }
    for (func::FuncOp function : targets) {
      PaddedRowDotLowering lowering(function, blockThreads,
                                    vectorizeFullFeatures,
                                    promoteRank0Scalars);
      if (failed(lowering.match()) || failed(lowering.rewrite())) {
        function.emitError("padded row reduction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
