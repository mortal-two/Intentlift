//===- FuseLinalgSoftmax.cpp - fused stable row Softmax -------------------===//
//
// Build a real vector-dialect implementation state for the canonical stable
// row Softmax graph. The pass is structural and never names a kernel or emits
// target source.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include <limits>
#include <optional>

namespace intent {
#define GEN_PASS_DEF_FUSELINALGSOFTMAX
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct SoftmaxMatch {
  linalg::GenericOp maximum;
  tensor::ExpandShapeOp expandedMaximum;
  linalg::GenericOp subtract;
  linalg::GenericOp exponent;
  linalg::GenericOp sum;
  tensor::ExpandShapeOp expandedSum;
  linalg::GenericOp divide;
  arith::AddFOp sumCombine;
  arith::SubFOp subtractScalar;
  arith::DivFOp divideScalar;
  Value input;
  RankedTensorType inputType;
};

static bool isRankTwoIdentity(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumSymbols() == 0 &&
         map.isIdentity();
}

static bool isRowProjection(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 &&
         map.getResult(0) == getAffineDimExpr(0, map.getContext());
}

static bool isRowBroadcast(AffineMap map) {
  if (map.getNumDims() != 2 || map.getNumSymbols() != 0 ||
      map.getNumResults() != 2 ||
      map.getResult(0) != getAffineDimExpr(0, map.getContext()))
    return false;
  auto constant = dyn_cast<AffineConstantExpr>(map.getResult(1));
  return constant && constant.getValue() == 0;
}

static bool hasIteratorKinds(linalg::GenericOp op,
                             ArrayRef<utils::IteratorType> expected) {
  return llvm::equal(op.getIteratorTypesArray(), expected);
}

static bool hasOnlyUsers(Value value, ArrayRef<Operation *> expected) {
  llvm::SmallPtrSet<Operation *, 4> expectedSet(expected.begin(), expected.end());
  llvm::SmallPtrSet<Operation *, 4> actual;
  for (Operation *user : value.getUsers())
    actual.insert(user);
  return actual == expectedSet;
}

static std::optional<APFloat> constantFloat(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  if (!attribute)
    return std::nullopt;
  return attribute.getValue();
}

static bool hasReductionIdentity(linalg::GenericOp op, bool maximum) {
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || fill->getNumOperands() != 2 || fill->getNumResults() != 1 ||
      op.getDpsInits()[0] != fill->getResult(0))
    return false;
  std::optional<APFloat> value = constantFloat(fill->getOperand(0));
  if (!value)
    return false;
  return maximum ? value->isInfinity() && value->isNegative()
                 : value->isZero();
}

template <typename ScalarOp>
static ScalarOp matchBinaryPointwise(linalg::GenericOp op) {
  if (!op || !op.hasPureTensorSemantics() || op.getNumLoops() != 2 ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1 ||
      !hasIteratorKinds(op, {utils::IteratorType::parallel,
                             utils::IteratorType::parallel}))
    return {};
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !isRankTwoIdentity(maps[0]) ||
      !isRowBroadcast(maps[1]) || !isRankTwoIdentity(maps[2]))
    return {};
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3 || !body.getArgument(2).use_empty() ||
      !llvm::hasSingleElement(body.without_terminator()))
    return {};
  auto scalar = dyn_cast<ScalarOp>(&body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!scalar || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != scalar.getResult() ||
      scalar.getLhs() != body.getArgument(0) ||
      scalar.getRhs() != body.getArgument(1))
    return {};
  return scalar;
}

static math::ExpOp matchExponent(linalg::GenericOp op) {
  if (!op || !op.hasPureTensorSemantics() || op.getNumLoops() != 2 ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1 ||
      !hasIteratorKinds(op, {utils::IteratorType::parallel,
                             utils::IteratorType::parallel}))
    return {};
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isRankTwoIdentity(maps[0]) ||
      !isRankTwoIdentity(maps[1]))
    return {};
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getArgument(1).use_empty() ||
      !llvm::hasSingleElement(body.without_terminator()))
    return {};
  auto exponent = dyn_cast<math::ExpOp>(&body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!exponent || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != exponent.getResult() ||
      exponent.getOperand() != body.getArgument(0))
    return {};
  return exponent;
}

static Operation *matchReduction(linalg::GenericOp op, bool maximum,
                                 arith::AddFOp &sumCombine) {
  if (!op || !op.hasPureTensorSemantics() || op.getNumLoops() != 2 ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1 ||
      !hasIteratorKinds(op, {utils::IteratorType::parallel,
                             utils::IteratorType::reduction}))
    return nullptr;
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isRankTwoIdentity(maps[0]) ||
      !isRowProjection(maps[1]) || !hasReductionIdentity(op, maximum))
    return nullptr;
  auto inputType = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto outputType = dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!inputType || !outputType || inputType.getRank() != 2 ||
      outputType.getRank() != 1 || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape() || !inputType.getElementType().isF32() ||
      outputType.getElementType() != inputType.getElementType() ||
      outputType.getDimSize(0) != inputType.getDimSize(0))
    return nullptr;
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2 ||
      !llvm::hasSingleElement(body.without_terminator()))
    return nullptr;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return nullptr;
  if (maximum) {
    auto combine = dyn_cast<arith::MaxNumFOp>(&body.front());
    if (!combine || yield.getValues().front() != combine.getResult() ||
        combine.getLhs() != body.getArgument(1) ||
        combine.getRhs() != body.getArgument(0))
      return nullptr;
    return combine.getOperation();
  }
  auto combine = dyn_cast<arith::AddFOp>(&body.front());
  if (!combine || yield.getValues().front() != combine.getResult() ||
      combine.getLhs() != body.getArgument(1) ||
      combine.getRhs() != body.getArgument(0) ||
      !arith::bitEnumContainsAll(combine.getFastmath(),
                                 arith::FastMathFlags::reassoc))
    return nullptr;
  sumCombine = combine;
  return combine.getOperation();
}

static tensor::ExpandShapeOp matchRowExpand(Value value, Value source,
                                            RankedTensorType inputType) {
  auto expand = value.getDefiningOp<tensor::ExpandShapeOp>();
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  if (!expand || expand->getNumOperands() != 1 ||
      expand->getOperand(0) != source || !resultType ||
      resultType.getRank() != 2 || resultType.getDimSize(0) !=
          inputType.getDimSize(0) || resultType.getDimSize(1) != 1)
    return {};
  return expand;
}

static std::optional<SoftmaxMatch> matchSoftmax(linalg::GenericOp divide) {
  arith::DivFOp divideScalar = matchBinaryPointwise<arith::DivFOp>(divide);
  if (!divideScalar)
    return std::nullopt;
  auto inputType = dyn_cast<RankedTensorType>(divide.getResult(0).getType());
  if (!inputType || inputType.getRank() != 2 || !inputType.hasStaticShape() ||
      !inputType.getElementType().isF32())
    return std::nullopt;

  Value exponentValue = divide.getDpsInputs()[0];
  auto exponent = exponentValue.getDefiningOp<linalg::GenericOp>();
  if (!matchExponent(exponent))
    return std::nullopt;
  auto expandedSum =
      divide.getDpsInputs()[1].getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandedSum)
    return std::nullopt;
  Value sumValue = expandedSum->getOperand(0);
  auto sum = sumValue.getDefiningOp<linalg::GenericOp>();
  arith::AddFOp sumCombine;
  if (!matchReduction(sum, false, sumCombine) ||
      sum.getDpsInputs()[0] != exponentValue ||
      !matchRowExpand(expandedSum.getResult(), sumValue, inputType))
    return std::nullopt;

  Value subtractValue = exponent.getDpsInputs()[0];
  auto subtract = subtractValue.getDefiningOp<linalg::GenericOp>();
  arith::SubFOp subtractScalar =
      matchBinaryPointwise<arith::SubFOp>(subtract);
  if (!subtractScalar)
    return std::nullopt;
  Value input = subtract.getDpsInputs()[0];
  if (input.getType() != inputType)
    return std::nullopt;
  auto expandedMaximum =
      subtract.getDpsInputs()[1].getDefiningOp<tensor::ExpandShapeOp>();
  if (!expandedMaximum)
    return std::nullopt;
  Value maximumValue = expandedMaximum->getOperand(0);
  auto maximum = maximumValue.getDefiningOp<linalg::GenericOp>();
  arith::AddFOp unused;
  if (!matchReduction(maximum, true, unused) ||
      maximum.getDpsInputs()[0] != input ||
      !matchRowExpand(expandedMaximum.getResult(), maximumValue, inputType))
    return std::nullopt;

  if (!hasOnlyUsers(maximumValue, {expandedMaximum.getOperation()}) ||
      !hasOnlyUsers(expandedMaximum.getResult(), {subtract.getOperation()}) ||
      !hasOnlyUsers(subtractValue, {exponent.getOperation()}) ||
      !hasOnlyUsers(exponentValue, {sum.getOperation(), divide.getOperation()}) ||
      !hasOnlyUsers(sumValue, {expandedSum.getOperation()}) ||
      !hasOnlyUsers(expandedSum.getResult(), {divide.getOperation()}))
    return std::nullopt;

  return SoftmaxMatch{maximum,
                      expandedMaximum,
                      subtract,
                      exponent,
                      sum,
                      expandedSum,
                      divide,
                      sumCombine,
                      subtractScalar,
                      divideScalar,
                      input,
                      inputType};
}

static Value emitHybridFusedSoftmax(IRRewriter &rewriter, SoftmaxMatch &match,
                                     unsigned vectorWidth,
                                     bool scalableVector) {
  rewriter.setInsertionPoint(match.divide);
  Location location = match.divide.getLoc();
  FloatType elementType = cast<FloatType>(match.inputType.getElementType());
  VectorType vectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, elementType,
      SmallVector<bool>{scalableVector});

  Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
  Value oneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
  Value rows = rewriter.create<arith::ConstantIndexOp>(
      location, match.inputType.getDimSize(0));
  Value columns = rewriter.create<arith::ConstantIndexOp>(
      location, match.inputType.getDimSize(1));
  Value minimumLanes = rewriter.create<arith::ConstantIndexOp>(
      location, static_cast<int64_t>(vectorWidth));
  Value step = minimumLanes;
  if (scalableVector) {
    Value vscale = rewriter.create<vector::VectorScaleOp>(location);
    step = rewriter.create<arith::MulIOp>(location, minimumLanes, vscale);
  }

  Value zero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
  Value negativeInfinity = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(
                    -std::numeric_limits<double>::infinity()));
  Value negativeInfinityVector = rewriter.create<vector::SplatOp>(
      location, vectorType, negativeInfinity);
  Value output = rewriter.create<tensor::EmptyOp>(
      location, match.inputType.getShape(), elementType);

  auto outer = rewriter.create<scf::ForOp>(
      location, zeroIndex, rows, oneIndex, ValueRange{output},
      [&](OpBuilder &rowBuilder, Location rowLocation, Value row,
          ValueRange rowArgs) {
        auto maximumLoop = rowBuilder.create<scf::ForOp>(
            rowLocation, zeroIndex, columns, step,
            ValueRange{negativeInfinityVector},
            [&](OpBuilder &columnBuilder, Location columnLocation, Value column,
                ValueRange columnArgs) {
              Value packet = columnBuilder.create<vector::TransferReadOp>(
                  columnLocation, vectorType, match.input,
                  ValueRange{row, column}, negativeInfinity);
              Value next = columnBuilder.create<arith::MaxNumFOp>(
                  columnLocation, columnArgs[0], packet);
              columnBuilder.create<scf::YieldOp>(columnLocation, next);
            });
        Value maximum = rowBuilder.create<vector::ReductionOp>(
            rowLocation, vector::CombiningKind::MAXNUMF,
            maximumLoop.getResult(0), negativeInfinity);
        auto exponentLoop = rowBuilder.create<scf::ForOp>(
            rowLocation, zeroIndex, columns, oneIndex,
            ValueRange{rowArgs[0], zero},
            [&](OpBuilder &columnBuilder, Location columnLocation, Value column,
                ValueRange columnArgs) {
              Value scalar = columnBuilder.create<tensor::ExtractOp>(
                  columnLocation, match.input, ValueRange{row, column});
              Value shifted = columnBuilder.create<arith::SubFOp>(
                  columnLocation, scalar, maximum,
                  match.subtractScalar.getFastmath());
              Value exponent =
                  columnBuilder.create<math::ExpOp>(columnLocation, shifted);
              Value accumulated = columnBuilder.create<arith::AddFOp>(
                  columnLocation, columnArgs[1], exponent,
                  match.sumCombine.getFastmath());
              Value written = columnBuilder.create<tensor::InsertOp>(
                  columnLocation, exponent, columnArgs[0],
                  ValueRange{row, column});
              columnBuilder.create<scf::YieldOp>(
                  columnLocation, ValueRange{written, accumulated});
            });
        Value exponentOutput = exponentLoop.getResult(0);
        Value denominator = exponentLoop.getResult(1);
        Value denominatorPacket = rowBuilder.create<vector::SplatOp>(
            rowLocation, vectorType, denominator);

        auto normalizeLoop = rowBuilder.create<scf::ForOp>(
            rowLocation, zeroIndex, columns, step,
            ValueRange{exponentOutput},
            [&](OpBuilder &columnBuilder, Location columnLocation, Value column,
                ValueRange columnArgs) {
              Value exponent = columnBuilder.create<vector::TransferReadOp>(
                  columnLocation, vectorType, columnArgs[0],
                  ValueRange{row, column}, zero);
              Value normalized = columnBuilder.create<arith::DivFOp>(
                  columnLocation, exponent, denominatorPacket,
                  match.divideScalar.getFastmath());
              auto write = columnBuilder.create<vector::TransferWriteOp>(
                  columnLocation, normalized, columnArgs[0],
                  ValueRange{row, column});
              Value written = write->getResult(0);
              columnBuilder.create<scf::YieldOp>(columnLocation, written);
            });
        rowBuilder.create<scf::YieldOp>(rowLocation,
                                        normalizeLoop.getResult(0));
      });
  return outer.getResult(0);
}

struct FuseLinalgSoftmaxPass
    : ::intent::impl::FuseLinalgSoftmaxBase<FuseLinalgSoftmaxPass> {
  using FuseLinalgSoftmaxBase::FuseLinalgSoftmaxBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (vectorWidth == 0 || (vectorWidth & (vectorWidth - 1)) != 0) {
      module.emitError("fused Softmax vector width must be a power of two");
      return signalPassFailure();
    }

    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp op) {
      if (matchBinaryPointwise<arith::DivFOp>(op))
        candidates.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    unsigned fusedCount = 0;
    for (linalg::GenericOp candidate : candidates) {
      if (!candidate->getBlock())
        continue;
      std::optional<SoftmaxMatch> match = matchSoftmax(candidate);
      if (!match)
        continue;
      Value replacement = emitHybridFusedSoftmax(
          rewriter, *match, vectorWidth, scalableVector);
      rewriter.replaceOp(candidate, replacement);
      ++fusedCount;
    }

    Builder builder(&getContext());
    module->setAttr("intent.rvv.fused_softmax_count",
                    builder.getI64IntegerAttr(fusedCount));
    module->setAttr("intent.rvv.fused_softmax_input_traversal_count",
                    builder.getI64IntegerAttr(2 * fusedCount));
    module->setAttr("intent.rvv.fused_softmax_full_row_traversal_count",
                    builder.getI64IntegerAttr(3 * fusedCount));
    module->setAttr("intent.rvv.fused_softmax_removed_materialization_count",
                    builder.getI64IntegerAttr(4 * fusedCount));
    module->setAttr("intent.rvv.fused_softmax_vector_reduction_count",
                    builder.getI64IntegerAttr(fusedCount));
    module->setAttr("intent.rvv.fused_softmax_scalar_reduction_count",
                    builder.getI64IntegerAttr(fusedCount));
    module->setAttr("intent.rvv.fused_softmax_vector_width",
                    builder.getI64IntegerAttr(fusedCount ? vectorWidth : 0));
    module->setAttr("intent.rvv.fused_softmax_scalable_count",
                    builder.getI64IntegerAttr(scalableVector ? fusedCount : 0));
    if (requireFusion && fusedCount == 0) {
      module.emitError("required fused Softmax matched no canonical stable graph");
      signalPassFailure();
    }
  }
};

} // namespace
