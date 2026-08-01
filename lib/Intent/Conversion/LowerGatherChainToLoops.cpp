//===- LowerGatherChainToLoops.cpp - direct indexed gather loops ----------===//
//
// Composes an exact typed two-level Intent gather without naming a source
// kernel.  Fixed-width vector mode intentionally emits vector.gather so LLVM's
// RISC-V backend can select vluxei rather than scalarizing data-dependent loads.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "Intent/IntentOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent {
#define GEN_PASS_DEF_LOWERGATHERCHAINTOLOOPS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct GatherChainMatch {
  func::FuncOp function;
  GatherOp first;
  tensor::ExpandShapeOp expandColumn;
  GatherOp second;
  tensor::CollapseShapeOp collapseResult;
  RankedTensorType dataType;
  RankedTensorType rowIndexType;
  RankedTensorType columnIndexType;
  RankedTensorType resultType;
};

static std::optional<GatherChainMatch> matchGatherChain(func::FuncOp function) {
  if (!function.getBody().hasOneBlock())
    return std::nullopt;
  Block &body = function.getBody().front();
  auto result = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!result || result.getNumOperands() != 1)
    return std::nullopt;

  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator())
    operations.push_back(&operation);
  if (operations.size() != 4)
    return std::nullopt;

  auto first = dyn_cast<GatherOp>(operations[0]);
  auto expand = dyn_cast<tensor::ExpandShapeOp>(operations[1]);
  auto second = dyn_cast<GatherOp>(operations[2]);
  auto collapse = dyn_cast<tensor::CollapseShapeOp>(operations[3]);
  if (!first || !expand || !second || !collapse ||
      first.getMode() != "take" || first.getAxis() != 0 ||
      second.getMode() != "along_axis" || second.getAxis() != 1 ||
      second.getData() != first.getResult() ||
      second.getIndex() != expand.getResult() ||
      collapse.getSrc() != second.getResult() ||
      result.getOperand(0) != collapse.getResult())
    return std::nullopt;

  auto dataType = dyn_cast<RankedTensorType>(first.getData().getType());
  auto rowIndexType = dyn_cast<RankedTensorType>(first.getIndex().getType());
  auto firstResultType = dyn_cast<RankedTensorType>(first.getResult().getType());
  auto columnIndexType = dyn_cast<RankedTensorType>(expand.getSrc().getType());
  auto expandedIndexType = dyn_cast<RankedTensorType>(expand.getResult().getType());
  auto secondResultType = dyn_cast<RankedTensorType>(second.getResult().getType());
  auto resultType = dyn_cast<RankedTensorType>(collapse.getResult().getType());
  if (!dataType || !rowIndexType || !firstResultType || !columnIndexType ||
      !expandedIndexType || !secondResultType || !resultType ||
      !dataType.hasStaticShape() || !rowIndexType.hasStaticShape() ||
      !firstResultType.hasStaticShape() || !columnIndexType.hasStaticShape() ||
      !expandedIndexType.hasStaticShape() ||
      !secondResultType.hasStaticShape() || !resultType.hasStaticShape() ||
      dataType.getRank() != 2 || rowIndexType.getRank() != 1 ||
      columnIndexType.getRank() != 1 || firstResultType.getRank() != 2 ||
      expandedIndexType.getRank() != 2 || secondResultType.getRank() != 2 ||
      resultType.getRank() != 1 || !dataType.getElementType().isF32() ||
      !rowIndexType.getElementType().isInteger(32) ||
      !columnIndexType.getElementType().isInteger(32))
    return std::nullopt;

  int64_t elements = rowIndexType.getDimSize(0);
  int64_t columns = dataType.getDimSize(1);
  if (elements <= 0 || columns <= 0 ||
      firstResultType.getShape() !=
          ArrayRef<int64_t>({elements, columns}) ||
      columnIndexType.getDimSize(0) != elements ||
      expandedIndexType.getShape() != ArrayRef<int64_t>({elements, 1}) ||
      secondResultType.getShape() != ArrayRef<int64_t>({elements, 1}) ||
      resultType.getDimSize(0) != elements ||
      firstResultType.getElementType() != dataType.getElementType() ||
      secondResultType.getElementType() != dataType.getElementType() ||
      resultType.getElementType() != dataType.getElementType())
    return std::nullopt;

  return GatherChainMatch{function, first, expand, second, collapse, dataType,
                          rowIndexType, columnIndexType, resultType};
}

static MemRefType asIdentityMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value loadIndex(OpBuilder &builder, Location location, Value memory,
                       Value position) {
  Value value =
      builder.create<memref::LoadOp>(location, memory, ValueRange{position});
  return builder.create<arith::IndexCastOp>(location, builder.getIndexType(),
                                            value);
}

static void rewriteGatherChain(const GatherChainMatch &match,
                               unsigned vectorWidth) {
  func::FuncOp source = match.function;
  MLIRContext *context = source.getContext();
  Location location = source.getLoc();
  OpBuilder builder(context);

  SmallVector<Type> arguments{
      asIdentityMemRef(match.dataType), asIdentityMemRef(match.rowIndexType),
      asIdentityMemRef(match.columnIndexType),
      asIdentityMemRef(match.resultType)};
  auto functionType = FunctionType::get(context, arguments, {});
  std::string name = source.getName().str();
  source.setName(name + "_intent_src");
  builder.setInsertionPoint(source);
  auto lowered = builder.create<func::FuncOp>(location, name, functionType);
  Block *entry = lowered.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value data = entry->getArgument(0);
  Value rowIndex = entry->getArgument(1);
  Value columnIndex = entry->getArgument(2);
  Value output = entry->getArgument(3);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value upper = builder.create<arith::ConstantIndexOp>(
      location, match.resultType.getDimSize(0));
  Value step =
      builder.create<arith::ConstantIndexOp>(location, vectorWidth);
  auto loop = builder.create<scf::ForOp>(location, zero, upper, step);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
  Value position = loop.getInductionVar();

  if (vectorWidth == 1) {
    Value row = loadIndex(bodyBuilder, location, rowIndex, position);
    Value column = loadIndex(bodyBuilder, location, columnIndex, position);
    Value value = bodyBuilder.create<memref::LoadOp>(
        location, data, ValueRange{row, column});
    bodyBuilder.create<memref::StoreOp>(location, value, output,
                                        ValueRange{position});
  } else {
    int64_t lanes = static_cast<int64_t>(vectorWidth);
    VectorType indexVectorType =
        VectorType::get({lanes}, bodyBuilder.getI32Type());
    VectorType valueVectorType =
        VectorType::get({lanes}, bodyBuilder.getF32Type());
    VectorType maskVectorType =
        VectorType::get({lanes}, bodyBuilder.getI1Type());
    Value rows = bodyBuilder.create<vector::LoadOp>(
        location, indexVectorType, rowIndex, ValueRange{position});
    Value columns = bodyBuilder.create<vector::LoadOp>(
        location, indexVectorType, columnIndex, ValueRange{position});
    Value rowWidth = bodyBuilder.create<arith::ConstantIntOp>(
        location, match.dataType.getDimSize(1), 32);
    Value rowWidths = bodyBuilder.create<vector::SplatOp>(
        location, indexVectorType, rowWidth);
    Value offsets = bodyBuilder.create<arith::AddIOp>(
        location,
        bodyBuilder.create<arith::MulIOp>(location, rows, rowWidths), columns);
    Value trueValue =
        bodyBuilder.create<arith::ConstantIntOp>(location, 1, 1);
    Value mask = bodyBuilder.create<vector::SplatOp>(
        location, maskVectorType, trueValue);
    Value zeroValue = bodyBuilder.create<arith::ConstantFloatOp>(
        location, APFloat(0.0f), bodyBuilder.getF32Type());
    Value passThrough = bodyBuilder.create<vector::SplatOp>(
        location, valueVectorType, zeroValue);
    Value gathered = bodyBuilder.create<vector::GatherOp>(
        location, valueVectorType, data, ValueRange{zero, zero}, offsets, mask,
        passThrough);
    bodyBuilder.create<vector::StoreOp>(location, gathered, output,
                                        ValueRange{position});
  }

  builder.setInsertionPointAfter(loop);
  builder.create<func::ReturnOp>(location);
  lowered->setAttr("intent.backend.lowering_path_kind",
                   builder.getStringAttr("composed_indexed_gather"));
  lowered->setAttr(
      "intent.backend.transformation_trace",
      builder.getArrayAttr(
          {builder.getStringAttr("rvv.gather.composed-indexed.v1")}));
  source.erase();
}

struct LowerGatherChainToLoopsPass
    : ::intent::impl::LowerGatherChainToLoopsBase<
          LowerGatherChainToLoopsPass> {
  using LowerGatherChainToLoopsBase::LowerGatherChainToLoopsBase;

  void runOnOperation() override {
    if (vectorWidth != 1 && vectorWidth != 2 && vectorWidth != 4 &&
        vectorWidth != 8) {
      getOperation().emitError(
          "gather-chain vector width must be 1, 2, 4, or 8");
      signalPassFailure();
      return;
    }
    SmallVector<GatherChainMatch> matches;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      if (std::optional<GatherChainMatch> match = matchGatherChain(function))
        matches.push_back(*match);
    if (matches.empty()) {
      if (requireLowering) {
        getOperation().emitError(
            "required typed two-level gather graph was not present");
        signalPassFailure();
      }
      return;
    }
    for (GatherChainMatch &match : matches) {
      if (match.resultType.getDimSize(0) % vectorWidth != 0) {
        match.function.emitError(
            "gather length is not divisible by the requested vector width");
        signalPassFailure();
        return;
      }
      rewriteGatherChain(match, vectorWidth);
    }
    Builder builder(getOperation().getContext());
    getOperation()->setAttr("intent.rvv.composed_gather_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.composed_gather_chain_depth",
                            builder.getI64IntegerAttr(2));
    getOperation()->setAttr("intent.rvv.composed_gather_vector_width",
                            builder.getI64IntegerAttr(vectorWidth));
  }
};

} // namespace
