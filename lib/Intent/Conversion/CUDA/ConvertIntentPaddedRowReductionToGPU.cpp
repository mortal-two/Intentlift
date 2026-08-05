//===- ConvertIntentPaddedRowReductionToGPU.cpp ---------------------------===//
//
// Structural single-launch lowering for a masked short row dot followed by
// zero padding.  This path is intentionally keyed only by typed MLIR graph
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

static void annotatePaddedRowState(func::FuncOp lowered, int64_t threads,
                                   StringRef topology, int64_t rowsPerBlock,
                                   int64_t rowsPerSubgroup,
                                   int64_t subgroupWidth, int64_t vectorWidth,
                                   unsigned hostValueCaptureCount) {
  MLIRContext *context = lowered.getContext();
  unsigned shuffleCount = 0;
  unsigned barrierCount = 0;
  unsigned allReduceCount = 0;
  lowered.walk([&](gpu::ShuffleOp) { ++shuffleCount; });
  lowered.walk([&](gpu::BarrierOp) { ++barrierCount; });
  lowered.walk([&](gpu::AllReduceOp) { ++allReduceCount; });
  SmallVector<Attribute> trace{
      StringAttr::get(context,
                      "cuda.padded_row_reduction.masked_short_dot.v1"),
      StringAttr::get(context,
                      ("cuda.reduction." + topology + ".v1").str())};
  if (hostValueCaptureCount != 0)
    trace.push_back(
        StringAttr::get(context, "cuda.b.rank0_host_value_capture.v1"));
  if (vectorWidth > 1)
    trace.push_back(StringAttr::get(
        context, "cuda.memory.packetized_feature_load.v1"));
  if (rowsPerSubgroup > 1)
    trace.push_back(StringAttr::get(
        context, "cuda.reduction.padded_subgroup_multirow.v1"));
  auto i64 = IntegerType::get(context, 64);
  lowered->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(context, "padded_row_reduction"));
  lowered->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(context, topology));
  lowered->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(context, trace));
  lowered->setAttr("intent.backend.fma_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.shuffle_count",
                   IntegerAttr::get(i64, shuffleCount));
  lowered->setAttr("intent.backend.barrier_count",
                   IntegerAttr::get(i64, barrierCount));
  lowered->setAttr("intent.backend.all_reduce_count",
                   IntegerAttr::get(i64, allReduceCount));
  lowered->setAttr("intent.backend.block_threads",
                   IntegerAttr::get(i64, threads));
  lowered->setAttr("intent.backend.rows_per_block",
                   IntegerAttr::get(i64, rowsPerBlock));
  lowered->setAttr("intent.backend.rows_per_subgroup",
                   IntegerAttr::get(i64, rowsPerSubgroup));
  lowered->setAttr("intent.backend.subgroup_width",
                   IntegerAttr::get(i64, subgroupWidth));
  lowered->setAttr("intent.backend.vector_width",
                   IntegerAttr::get(i64, vectorWidth));
  lowered->setAttr("intent.backend.predicated_value_if_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.folded_constexpr_select_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.host_value_capture_count",
                   IntegerAttr::get(i64, hostValueCaptureCount));
  lowered->setAttr("intent.backend.epilogue_load_motion_count",
                   IntegerAttr::get(i64, 0));
}

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
  std::string requestedReductionTopology;
  unsigned requestedRowsPerSubgroup;
  unsigned requestedLanesPerRow;

  int64_t rows = 0;
  int64_t batch = 0;
  int64_t physicalRows = 0;
  int64_t featureWidth = 0;
  int64_t paddedRows = 0;
  int64_t threads = 0;
  int64_t blocksPerRow = 0;
  int64_t subgroupsPerBlock = 0;
  int64_t outputRowsPerBlock = 0;
  Type storageType;
  unsigned validRowsArgument = 0;
  unsigned validFeaturesArgument = 0;
  bool staticBounds = false;

  PadOp pad;
  ReduceOp reduce;
  MapBinaryOp product;
  SelectOp leftSelect;
  SelectOp rightSelect;

  explicit PaddedRowDotLowering(func::FuncOp function,
                                unsigned requestedThreads,
                                bool requestedVectorizeFullFeatures,
                                bool requestedPromoteRank0Scalars,
                                StringRef requestedReductionTopology,
                                unsigned requestedRowsPerSubgroup,
                                unsigned requestedLanesPerRow)
      : function(function), requestedThreads(requestedThreads),
        requestedVectorizeFullFeatures(requestedVectorizeFullFeatures),
        requestedPromoteRank0Scalars(requestedPromoteRank0Scalars),
        requestedReductionTopology(requestedReductionTopology.str()),
        requestedRowsPerSubgroup(requestedRowsPerSubgroup),
        requestedLanesPerRow(requestedLanesPerRow) {}

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
    if ((type.getNumInputs() != 2 && type.getNumInputs() != 4) ||
        type.getNumResults() != 1 ||
        !function.getBody().hasOneBlock())
      return failure();
    staticBounds = type.getNumInputs() == 2;
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
    if (!outputType || !reducedType || !outputType.hasStaticShape() ||
        !reducedType.hasStaticShape() || outputType.getRank() != 3 ||
        reducedType.getRank() != 3 ||
        !isa<Float32Type>(outputType.getElementType()) ||
        !isa<Float32Type>(reducedType.getElementType()) || !leftCast ||
        !rightCast)
      return failure();

    BlockArgument leftInput;
    BlockArgument rightInput;
    if (staticBounds) {
      leftInput = dyn_cast<BlockArgument>(leftCast.getInput());
      rightInput = dyn_cast<BlockArgument>(rightCast.getInput());
    } else {
      leftSelect = leftCast.getInput().getDefiningOp<SelectOp>();
      rightSelect = rightCast.getInput().getDefiningOp<SelectOp>();
      leftInput = leftSelect ? dyn_cast<BlockArgument>(leftSelect.getX())
                             : BlockArgument();
      rightInput = rightSelect ? dyn_cast<BlockArgument>(rightSelect.getX())
                               : BlockArgument();
      if (!leftSelect || !rightSelect || !isZeroSplat(leftSelect.getY()) ||
          !isZeroSplat(rightSelect.getY()))
        return failure();
    }
    if (!leftInput || !rightInput ||
        leftInput.getArgNumber() == rightInput.getArgNumber())
      return failure();
    auto leftType = dyn_cast<RankedTensorType>(leftInput.getType());
    auto rightType = dyn_cast<RankedTensorType>(rightInput.getType());
    if (!leftType || !rightType || !leftType.hasStaticShape() ||
        leftType != rightType || leftType.getRank() != 4 ||
        leftType.getDimSize(0) <= 0 || leftType.getDimSize(1) <= 0 ||
        leftType.getDimSize(2) <= 0 || leftType.getDimSize(3) <= 0 ||
        !isa<Float16Type, BFloat16Type, Float32Type>(
            leftType.getElementType()))
      return failure();
    unsigned leftArgument = leftInput.getArgNumber();
    unsigned rightArgument = rightInput.getArgNumber();
    if (!((leftArgument == 0 && rightArgument == 1) ||
          (leftArgument == 1 && rightArgument == 0)))
      return failure();

    batch = leftType.getDimSize(0);
    rows = leftType.getDimSize(1);
    physicalRows = leftType.getDimSize(2);
    featureWidth = leftType.getDimSize(3);
    paddedRows = outputType.getDimSize(2);
    storageType = leftType.getElementType();
    if (outputType.getDimSize(0) != batch || outputType.getDimSize(1) != rows ||
        reducedType.getShape() !=
            ArrayRef<int64_t>({batch, rows, physicalRows}) ||
        paddedRows < physicalRows)
      return failure();

    auto low = integerArray(pad.getLo());
    auto high = integerArray(pad.getHi());
    if (low != SmallVector<int64_t>({0, 0, 0}) || high.size() != 3 ||
        high[0] != 0 || high[1] != 0 ||
        high[2] != paddedRows - physicalRows ||
        !pad.getValue().isZero())
      return failure();

    if (!staticBounds) {
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
    }

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
    if (requestedReductionTopology == "serial-row") {
      if (requestedRowsPerSubgroup != 1 || requestedLanesPerRow != 32)
        return failure();
      outputRowsPerBlock = threads;
      blocksPerRow = (paddedRows + threads - 1) / threads;
    } else if (requestedReductionTopology == "subgroup-feature") {
      if (featureWidth < 2 || featureWidth > 1024)
        return failure();
      if (requestedRowsPerSubgroup == 0 || requestedRowsPerSubgroup > 8 ||
          (requestedRowsPerSubgroup & (requestedRowsPerSubgroup - 1)) != 0)
        return failure();
      if (requestedLanesPerRow < 4 || requestedLanesPerRow > 32 ||
          (requestedLanesPerRow & (requestedLanesPerRow - 1)) != 0 ||
          32 % requestedLanesPerRow != 0)
        return failure();
      if (requestedVectorizeFullFeatures) {
        int64_t packetWidth = isa<Float32Type>(storageType) ? 4 : 8;
        if (!staticBounds ||
            featureWidth % (requestedLanesPerRow * packetWidth) != 0)
          return failure();
      }
      // Use complete hardware warps even when the logical feature strip is
      // shorter.  PTX sub-warp shuffle masks are fragile for multiple logical
      // groups in one warp; inactive feature lanes contribute zero instead.
      subgroupsPerBlock = threads / requestedLanesPerRow;
      outputRowsPerBlock = subgroupsPerBlock * requestedRowsPerSubgroup;
      blocksPerRow =
          (paddedRows + outputRowsPerBlock - 1) / outputRowsPerBlock;
    } else {
      return failure();
    }
    return success();
  }

  LogicalResult rewriteSubgroupFeature() {
    MLIRContext *context = function.getContext();
    Location location = function.getLoc();
    OpBuilder builder(context);
    auto originalType = function.getFunctionType();
    int64_t packetWidth = isa<Float32Type>(storageType) ? 4 : 8;
    Type packetType = VectorType::get({packetWidth}, storageType);
    SmallVector<Type> inputs;
    unsigned inputIndex = 0;
    for (Type input : originalType.getInputs()) {
      auto tensor = cast<RankedTensorType>(input);
      if (requestedVectorizeFullFeatures && inputIndex < 2) {
        // Keep the packet in the memref element type.  A vector.load from a
        // scalar-element memref is scalarized by GPU-to-NVVM, while this form
        // reaches one wide global-memory instruction under the same bare-pointer
        // kernel ABI.
        SmallVector<int64_t> packetShape(tensor.getShape());
        packetShape.back() /= packetWidth;
        inputs.push_back(MemRefType::get(packetShape, packetType));
      } else {
        inputs.push_back(MemRefType::get(tensor.getShape(),
                                         tensor.getElementType()));
      }
      ++inputIndex;
    }
    auto output = cast<RankedTensorType>(originalType.getResult(0));
    inputs.push_back(MemRefType::get(output.getShape(),
                                     output.getElementType()));
    unsigned outputArgument = originalType.getNumInputs();

    std::string name = function.getName().str();
    function.setName(name + "_intent_src");
    builder.setInsertionPoint(function);
    auto lowered = builder.create<func::FuncOp>(
        location, name, FunctionType::get(context, inputs, {}));
    Block *entry = lowered.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value grid = builder.create<arith::ConstantIndexOp>(
        location, batch * rows * blocksPerRow);
    Value block = builder.create<arith::ConstantIndexOp>(location, threads);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value validRowsRaw;
    Value validFeaturesRaw;
    if (!staticBounds && requestedPromoteRank0Scalars) {
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
    Value cBlocksPerRow = builder.create<arith::ConstantIndexOp>(
        location, blocksPerRow);
    Value cOutputRowsPerBlock = builder.create<arith::ConstantIndexOp>(
        location, outputRowsPerBlock);
    Value cSubgroupWidth = builder.create<arith::ConstantIndexOp>(
        location, requestedLanesPerRow);
    Value cFeatureWidth = builder.create<arith::ConstantIndexOp>(
        location, featureWidth);
    Value cPhysicalRows = builder.create<arith::ConstantIndexOp>(
        location, physicalRows);
    Value cPaddedRows = builder.create<arith::ConstantIndexOp>(
        location, paddedRows);
    Value cRowsPerBatch = builder.create<arith::ConstantIndexOp>(
        location, rows * blocksPerRow);
    Value batchId = builder.create<arith::DivUIOp>(location, blockId,
                                                   cRowsPerBatch);
    Value batchBlock = builder.create<arith::RemUIOp>(location, blockId,
                                                      cRowsPerBatch);
    Value row = builder.create<arith::DivUIOp>(location, batchBlock,
                                                cBlocksPerRow);
    Value rowBlock = builder.create<arith::RemUIOp>(location, batchBlock,
                                                     cBlocksPerRow);
    Value subgroup = builder.create<arith::DivUIOp>(
        location, threadId, cSubgroupWidth);
    Value lane = builder.create<arith::RemUIOp>(location, threadId,
                                                 cSubgroupWidth);
    Value subgroupBase = builder.create<arith::MulIOp>(
        location, rowBlock, cOutputRowsPerBlock);
    Value subgroupBaseRow = builder.create<arith::AddIOp>(
        location, subgroupBase, subgroup);
    Value zero = builder.create<arith::ConstantFloatOp>(
        location, APFloat(0.0f), builder.getF32Type());

    if (requestedRowsPerSubgroup > 1) {
      // Keep several independent rows live at once.  The former implementation
      // completed the entire reduction/shuffle/store chain for row N before it
      // issued any work for row N+1.  Interleaving the feature work below
      // exposes row-level memory and arithmetic ILP while preserving the exact
      // reduction tree within every row.
      Value validRows;
      Value validFeatures;
      if (staticBounds) {
        validRows =
            builder.create<arith::ConstantIndexOp>(location, physicalRows);
        validFeatures =
            builder.create<arith::ConstantIndexOp>(location, featureWidth);
      } else {
        if (!requestedPromoteRank0Scalars) {
          validRowsRaw = builder.create<memref::LoadOp>(
              location, entry->getArgument(validRowsArgument), ValueRange{});
          validFeaturesRaw = builder.create<memref::LoadOp>(
              location, entry->getArgument(validFeaturesArgument), ValueRange{});
        }
        validRows = builder.create<arith::IndexCastOp>(
            location, builder.getIndexType(), validRowsRaw);
        validFeatures = builder.create<arith::IndexCastOp>(
            location, builder.getIndexType(), validFeaturesRaw);
      }

      SmallVector<Value> paddedRowValues;
      SmallVector<Value> outputActiveValues;
      SmallVector<Value> rowActiveValues;
      paddedRowValues.reserve(requestedRowsPerSubgroup);
      outputActiveValues.reserve(requestedRowsPerSubgroup);
      rowActiveValues.reserve(requestedRowsPerSubgroup);
      for (unsigned rowSlot = 0; rowSlot < requestedRowsPerSubgroup;
           ++rowSlot) {
        Value paddedRow = subgroupBaseRow;
        if (rowSlot != 0) {
          Value cRowOffset = builder.create<arith::ConstantIndexOp>(
              location, rowSlot * subgroupsPerBlock);
          paddedRow = builder.create<arith::AddIOp>(location, subgroupBaseRow,
                                                    cRowOffset);
        }
        Value outputActive = builder.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::ult, paddedRow, cPaddedRows);
        Value inPhysicalRows = builder.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::ult, paddedRow, cPhysicalRows);
        Value inValidRows = builder.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::slt, paddedRow, validRows);
        Value inInput = builder.create<arith::AndIOp>(
            location, inPhysicalRows, inValidRows);
        Value rowActive = builder.create<arith::AndIOp>(
            location, outputActive, inInput);
        paddedRowValues.push_back(paddedRow);
        outputActiveValues.push_back(outputActive);
        rowActiveValues.push_back(rowActive);
      }

      auto emitScalarProduct = [&](OpBuilder &nested, Value paddedRow,
                                   Value column) -> Value {
        SmallVector<Value> indices{batchId, row, paddedRow, column};
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
      auto emitPacketProduct = [&](OpBuilder &nested, Value paddedRow,
                                   Value packetColumn) -> Value {
        SmallVector<Value> indices{batchId, row, paddedRow, packetColumn};
        Value leftVector = nested.create<memref::LoadOp>(
            location, entry->getArgument(0), indices);
        Value rightVector = nested.create<memref::LoadOp>(
            location, entry->getArgument(1), indices);
        Value sum = nested.create<arith::ConstantFloatOp>(
            location, APFloat(0.0f), nested.getF32Type());
        for (int64_t element = 0; element < packetWidth; ++element) {
          Value left = nested.create<vector::ExtractOp>(
              location, leftVector, ArrayRef<int64_t>{element});
          Value right = nested.create<vector::ExtractOp>(
              location, rightVector, ArrayRef<int64_t>{element});
          if (storageType != nested.getF32Type()) {
            left = nested.create<arith::ExtFOp>(location, nested.getF32Type(),
                                                left);
            right = nested.create<arith::ExtFOp>(location, nested.getF32Type(),
                                                 right);
          }
          Value product =
              nested.create<arith::MulFOp>(location, left, right);
          sum = nested.create<arith::AddFOp>(location, sum, product);
        }
        return sum;
      };

      SmallVector<Value> accumulators(requestedRowsPerSubgroup, zero);
      if (requestedVectorizeFullFeatures) {
        for (int64_t base = 0; base < featureWidth;
             base += requestedLanesPerRow * packetWidth) {
          Value cPacketBase = builder.create<arith::ConstantIndexOp>(
              location, base / packetWidth);
          Value packetColumn = builder.create<arith::AddIOp>(
              location, lane, cPacketBase);
          for (unsigned rowSlot = 0; rowSlot < requestedRowsPerSubgroup;
               ++rowSlot) {
            auto termIf = builder.create<scf::IfOp>(
                location, TypeRange{builder.getF32Type()},
                rowActiveValues[rowSlot], /*withElseRegion=*/true);
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(termIf.thenBlock());
              builder.create<scf::YieldOp>(
                  location,
                  emitPacketProduct(builder, paddedRowValues[rowSlot],
                                    packetColumn));
            }
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(termIf.elseBlock());
              builder.create<scf::YieldOp>(location, zero);
            }
            builder.setInsertionPointAfter(termIf);
            accumulators[rowSlot] = builder.create<arith::AddFOp>(
                location, accumulators[rowSlot], termIf.getResult(0));
          }
        }
      } else {
        for (int64_t base = 0; base < featureWidth;
             base += requestedLanesPerRow) {
          Value cBase = builder.create<arith::ConstantIndexOp>(location, base);
          Value column = builder.create<arith::AddIOp>(location, lane, cBase);
          Value inStaticFeatures = builder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::ult, column, cFeatureWidth);
          Value inValidFeatures = builder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::slt, column, validFeatures);
          Value featureActive = builder.create<arith::AndIOp>(
              location, inStaticFeatures, inValidFeatures);
          for (unsigned rowSlot = 0; rowSlot < requestedRowsPerSubgroup;
               ++rowSlot) {
            Value termActive = builder.create<arith::AndIOp>(
                location, rowActiveValues[rowSlot], featureActive);
            auto termIf = builder.create<scf::IfOp>(
                location, TypeRange{builder.getF32Type()}, termActive,
                /*withElseRegion=*/true);
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(termIf.thenBlock());
              builder.create<scf::YieldOp>(
                  location,
                  emitScalarProduct(builder, paddedRowValues[rowSlot], column));
            }
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(termIf.elseBlock());
              builder.create<scf::YieldOp>(location, zero);
            }
            builder.setInsertionPointAfter(termIf);
            accumulators[rowSlot] = builder.create<arith::AddFOp>(
                location, accumulators[rowSlot], termIf.getResult(0));
          }
        }
      }

      Value laneZero = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, lane, c0);
      for (unsigned rowSlot = 0; rowSlot < requestedRowsPerSubgroup;
           ++rowSlot) {
        Value accumulator = accumulators[rowSlot];
        for (int64_t delta = 1; delta < requestedLanesPerRow; delta <<= 1) {
          Value other = builder
                            .create<gpu::ShuffleOp>(location, accumulator, delta,
                                                    32,
                                                    gpu::ShuffleMode::XOR)
                            .getShuffleResult();
          accumulator = builder.create<arith::AddFOp>(location, accumulator,
                                                       other);
        }
        Value storeActive = builder.create<arith::AndIOp>(
            location, laneZero, outputActiveValues[rowSlot]);
        auto storeIf = builder.create<scf::IfOp>(
            location, storeActive, /*withElseRegion=*/false);
        builder.setInsertionPointToStart(storeIf.thenBlock());
        Value outputValue = builder.create<arith::SelectOp>(
            location, rowActiveValues[rowSlot], accumulator, zero);
        builder.create<memref::StoreOp>(
            location, outputValue, entry->getArgument(outputArgument),
            ValueRange{batchId, row, paddedRowValues[rowSlot]});
        builder.setInsertionPointAfter(storeIf);
      }

      builder.create<gpu::TerminatorOp>(location);
      builder.setInsertionPointAfter(launch);
      builder.create<func::ReturnOp>(location);
      annotatePaddedRowState(
          lowered, threads, "subgroup_feature", outputRowsPerBlock,
          requestedRowsPerSubgroup, requestedLanesPerRow,
          requestedVectorizeFullFeatures
              ? (isa<Float32Type>(storageType) ? 4 : 8)
              : 1,
          !staticBounds && requestedPromoteRank0Scalars ? 2 : 0);
      function.erase();
      return success();
    }

    // Preserve the compact one-row lowering byte-for-byte when multirow search
    // is disabled.  The multirow state above deliberately uses a different,
    // interleaved live-accumulator graph.
    for (unsigned rowSlot = 0; rowSlot < requestedRowsPerSubgroup; ++rowSlot) {
      Value paddedRow = subgroupBaseRow;
      if (rowSlot != 0) {
        Value cRowOffset = builder.create<arith::ConstantIndexOp>(
            location, rowSlot * subgroupsPerBlock);
        paddedRow = builder.create<arith::AddIOp>(location, subgroupBaseRow,
                                                  cRowOffset);
      }
      Value outputActive = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult, paddedRow, cPaddedRows);
      auto outputIf = builder.create<scf::IfOp>(
          location, outputActive, /*withElseRegion=*/false);
      builder.setInsertionPointToStart(outputIf.thenBlock());
      Value validRows;
      Value validFeatures;
      if (staticBounds) {
        validRows =
            builder.create<arith::ConstantIndexOp>(location, physicalRows);
        validFeatures =
            builder.create<arith::ConstantIndexOp>(location, featureWidth);
      } else if (!requestedPromoteRank0Scalars) {
        validRowsRaw = builder.create<memref::LoadOp>(
            location, entry->getArgument(validRowsArgument), ValueRange{});
        validFeaturesRaw = builder.create<memref::LoadOp>(
            location, entry->getArgument(validFeaturesArgument), ValueRange{});
      }
      if (!staticBounds) {
        validRows = builder.create<arith::IndexCastOp>(
            location, builder.getIndexType(), validRowsRaw);
        validFeatures = builder.create<arith::IndexCastOp>(
            location, builder.getIndexType(), validFeaturesRaw);
      }
      Value inPhysicalRows = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult, paddedRow, cPhysicalRows);
      Value inValidRows = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::slt, paddedRow, validRows);
      Value rowActive = builder.create<arith::AndIOp>(
          location, inPhysicalRows, inValidRows);
      auto emitScalarProduct = [&](OpBuilder &nested, Value column) -> Value {
        SmallVector<Value> indices{batchId, row, paddedRow, column};
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
      auto emitPacketProduct = [&](OpBuilder &nested,
                                   Value packetColumn) -> Value {
        SmallVector<Value> indices{batchId, row, paddedRow, packetColumn};
        Value leftVector = nested.create<memref::LoadOp>(
            location, entry->getArgument(0), indices);
        Value rightVector = nested.create<memref::LoadOp>(
            location, entry->getArgument(1), indices);
        Value sum = nested.create<arith::ConstantFloatOp>(
            location, APFloat(0.0f), nested.getF32Type());
        for (int64_t element = 0; element < packetWidth; ++element) {
          Value left = nested.create<vector::ExtractOp>(
              location, leftVector, ArrayRef<int64_t>{element});
          Value right = nested.create<vector::ExtractOp>(
              location, rightVector, ArrayRef<int64_t>{element});
          if (storageType != nested.getF32Type()) {
            left = nested.create<arith::ExtFOp>(location, nested.getF32Type(),
                                                left);
            right = nested.create<arith::ExtFOp>(location, nested.getF32Type(),
                                                 right);
          }
          Value product =
              nested.create<arith::MulFOp>(location, left, right);
          sum = nested.create<arith::AddFOp>(location, sum, product);
        }
        return sum;
      };

      Value accumulator = zero;
      if (requestedVectorizeFullFeatures) {
        for (int64_t base = 0; base < featureWidth;
             base += requestedLanesPerRow * packetWidth) {
          Value cPacketBase = builder.create<arith::ConstantIndexOp>(
              location, base / packetWidth);
          Value packetColumn = builder.create<arith::AddIOp>(
              location, lane, cPacketBase);
          auto termIf = builder.create<scf::IfOp>(
              location, TypeRange{builder.getF32Type()}, rowActive,
              /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(termIf.thenBlock());
            builder.create<scf::YieldOp>(
                location, emitPacketProduct(builder, packetColumn));
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(termIf.elseBlock());
            builder.create<scf::YieldOp>(location, zero);
          }
          builder.setInsertionPointAfter(termIf);
          accumulator = builder.create<arith::AddFOp>(
              location, accumulator, termIf.getResult(0));
        }
      } else {
        for (int64_t base = 0; base < featureWidth;
             base += requestedLanesPerRow) {
          Value cBase = builder.create<arith::ConstantIndexOp>(location, base);
          Value column = builder.create<arith::AddIOp>(location, lane, cBase);
          Value inStaticFeatures = builder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::ult, column, cFeatureWidth);
          Value inValidFeatures = builder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::slt, column, validFeatures);
          Value featureActive = builder.create<arith::AndIOp>(
              location, inStaticFeatures, inValidFeatures);
          Value termActive = builder.create<arith::AndIOp>(
              location, rowActive, featureActive);
          auto termIf = builder.create<scf::IfOp>(
              location, TypeRange{builder.getF32Type()}, termActive,
              /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(termIf.thenBlock());
            builder.create<scf::YieldOp>(
                location, emitScalarProduct(builder, column));
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(termIf.elseBlock());
            builder.create<scf::YieldOp>(location, zero);
          }
          builder.setInsertionPointAfter(termIf);
          accumulator = builder.create<arith::AddFOp>(
              location, accumulator, termIf.getResult(0));
        }
      }
      for (int64_t delta = 1; delta < requestedLanesPerRow; delta <<= 1) {
        Value other = builder
                          .create<gpu::ShuffleOp>(location, accumulator, delta,
                                                  32,
                                                  gpu::ShuffleMode::XOR)
                          .getShuffleResult();
        accumulator = builder.create<arith::AddFOp>(
            location, accumulator, other);
      }
      Value laneZero = builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, lane, c0);
      auto storeIf = builder.create<scf::IfOp>(location, laneZero,
                                               /*withElseRegion=*/false);
      builder.setInsertionPointToStart(storeIf.thenBlock());
      Value outputValue = builder.create<arith::SelectOp>(
          location, rowActive, accumulator, zero);
      builder.create<memref::StoreOp>(
          location, outputValue, entry->getArgument(outputArgument),
          ValueRange{batchId, row, paddedRow});
      builder.setInsertionPointAfter(outputIf);
    }
    builder.create<gpu::TerminatorOp>(location);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(location);
    annotatePaddedRowState(
        lowered, threads, "subgroup_feature", outputRowsPerBlock,
        requestedRowsPerSubgroup, requestedLanesPerRow,
        requestedVectorizeFullFeatures
            ? (isa<Float32Type>(storageType) ? 4 : 8)
            : 1,
        !staticBounds && requestedPromoteRank0Scalars ? 2 : 0);
    function.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (requestedReductionTopology == "subgroup-feature")
      return rewriteSubgroupFeature();
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
    unsigned outputArgument = originalType.getNumInputs();

    std::string name = function.getName().str();
    function.setName(name + "_intent_src");
    builder.setInsertionPoint(function);
    auto lowered = builder.create<func::FuncOp>(
        location, name, FunctionType::get(context, inputs, {}));
    Block *entry = lowered.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value grid = builder.create<arith::ConstantIndexOp>(
        location, batch * rows * blocksPerRow);
    Value block = builder.create<arith::ConstantIndexOp>(location, threads);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    // Rank-0 invocation scalars are loaded in the host launch stub and captured
    // by value.  Loading them inside gpu.launch would issue one redundant
    // global-memory transaction per output thread and would not match the
    // scalar-by-value ABI used by the source frontend.
    Value validRowsRaw;
    Value validFeaturesRaw;
    if (!staticBounds && requestedPromoteRank0Scalars) {
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
    Value cRowsPerBatch = builder.create<arith::ConstantIndexOp>(
        location, rows * blocksPerRow);
    Value batchId = builder.create<arith::DivUIOp>(location, blockId,
                                                   cRowsPerBatch);
    Value batchBlock = builder.create<arith::RemUIOp>(location, blockId,
                                                      cRowsPerBatch);
    Value row = builder.create<arith::DivUIOp>(location, batchBlock,
                                                cBlocksPerRow);
    Value rowBlock = builder.create<arith::RemUIOp>(location, batchBlock,
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
    Value validRows;
    Value validFeatures;
    if (staticBounds) {
      validRows = builder.create<arith::ConstantIndexOp>(location, physicalRows);
      validFeatures =
          builder.create<arith::ConstantIndexOp>(location, featureWidth);
    } else if (!requestedPromoteRank0Scalars) {
      validRowsRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validRowsArgument), ValueRange{});
      validFeaturesRaw = builder.create<memref::LoadOp>(
          location, entry->getArgument(validFeaturesArgument), ValueRange{});
    }
    if (!staticBounds) {
      validRows = builder.create<arith::IndexCastOp>(
          location, builder.getIndexType(), validRowsRaw);
      validFeatures = builder.create<arith::IndexCastOp>(
          location, builder.getIndexType(), validFeaturesRaw);
    }
    Value inPhysicalRows = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, paddedRow, cPhysicalRows);
    Value inValidRows = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, paddedRow, validRows);
    Value rowActive = builder.create<arith::AndIOp>(
        location, inPhysicalRows, inValidRows);

    auto emitScalarProduct = [&](OpBuilder &nested, Value column) -> Value {
      SmallVector<Value> indices{batchId, row, paddedRow, column};
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
            SmallVector<Value> indices{batchId, row, paddedRow, column};
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
        location, valueIf.getResult(0), entry->getArgument(outputArgument),
        ValueRange{batchId, row, paddedRow});
    builder.setInsertionPointAfter(outputIf);

    builder.create<gpu::TerminatorOp>(location);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(location);
    int64_t serialVectorWidth =
        requestedVectorizeFullFeatures &&
                featureWidth % (isa<Float32Type>(storageType) ? 4 : 8) == 0
            ? (isa<Float32Type>(storageType) ? 4 : 8)
            : 1;
    annotatePaddedRowState(
        lowered, threads, "serial_row", outputRowsPerBlock,
        /*rowsPerSubgroup=*/1, /*subgroupWidth=*/32, serialVectorWidth,
        !staticBounds && requestedPromoteRank0Scalars ? 2 : 0);
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
                                 promoteRank0Scalars, reductionTopology,
                                 rowsPerSubgroup, lanesPerRow);
      if (succeeded(probe.match()))
        targets.push_back(function);
    }
    for (func::FuncOp function : targets) {
      PaddedRowDotLowering lowering(function, blockThreads,
                                    vectorizeFullFeatures,
                                    promoteRank0Scalars, reductionTopology,
                                    rowsPerSubgroup, lanesPerRow);
      if (failed(lowering.match()) || failed(lowering.rewrite())) {
        function.emitError("padded row reduction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
