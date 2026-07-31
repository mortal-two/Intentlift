//===- VectorizeLinalgMixedPrecisionDotReduction.cpp ----------------------===//
//
// Build a real vector-dialect implementation state for a last-axis dot whose
// public inputs are f16 and whose accumulation/output are f32.  The matcher is
// structural and source-name blind.  It is intentionally separate from the
// generic f32 row-reduction pass because widening loads are part of the
// implementation topology, not a schedule spelling.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include <functional>
#include <optional>

namespace intent {
#define GEN_PASS_DEF_VECTORIZELINALGMIXEDPRECISIONDOTREDUCTION
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

static bool isParallelProjection(AffineMap map) {
  if (map.getNumDims() < 2 || map.getNumSymbols() != 0 ||
      map.getNumResults() + 1 != map.getNumDims())
    return false;
  for (unsigned dimension = 0; dimension < map.getNumResults(); ++dimension)
    if (map.getResult(dimension) !=
        getAffineDimExpr(dimension, map.getContext()))
      return false;
  return true;
}

static bool isZeroInitialized(linalg::GenericOp reduction) {
  auto fill = reduction.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  return fill && fill.getInputs().size() == 1 &&
         matchPattern(fill.getInputs().front(), m_AnyZeroFloat());
}

struct DotMatch {
  linalg::GenericOp reduction;
  Value lhs;
  Value rhs;
  RankedTensorType inputType;
  RankedTensorType outputType;
  arith::FastMathFlags fastMath;
};

static std::optional<DotMatch> matchMixedPrecisionDot(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumLoops() < 2 ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1 || !isZeroInitialized(op))
    return std::nullopt;

  ArrayRef<utils::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.back() != utils::IteratorType::reduction ||
      llvm::any_of(llvm::drop_end(iterators), [](utils::IteratorType iterator) {
        return iterator != utils::IteratorType::parallel;
      }))
    return std::nullopt;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !maps[0].isIdentity() || !maps[1].isIdentity() ||
      maps[0] != maps[1] || !isParallelProjection(maps[2]))
    return std::nullopt;

  auto lhsType = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  auto outputType =
      dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!lhsType || lhsType != rhsType || !lhsType.hasStaticShape() ||
      lhsType.getRank() != static_cast<int64_t>(op.getNumLoops()) ||
      !lhsType.getElementType().isF16() || !outputType ||
      !outputType.hasStaticShape() ||
      outputType.getRank() + 1 != lhsType.getRank() ||
      !outputType.getElementType().isF32())
    return std::nullopt;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3 ||
      !llvm::hasNItems(body.without_terminator(), 4))
    return std::nullopt;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield && yield.getValues().size() == 1
                 ? yield.getValues().front().getDefiningOp<arith::AddFOp>()
                 : arith::AddFOp{};
  if (!add || !arith::bitEnumContainsAll(add.getFastmath(),
                                         arith::FastMathFlags::reassoc))
    return std::nullopt;
  Value accumulator = body.getArgument(2);
  bool lhsAccumulator = add.getLhs() == accumulator;
  bool rhsAccumulator = add.getRhs() == accumulator;
  if (lhsAccumulator == rhsAccumulator)
    return std::nullopt;
  Value productValue = lhsAccumulator ? add.getRhs() : add.getLhs();
  auto multiply = productValue.getDefiningOp<arith::MulFOp>();
  if (!multiply)
    return std::nullopt;
  auto lhsExtend = multiply.getLhs().getDefiningOp<arith::ExtFOp>();
  auto rhsExtend = multiply.getRhs().getDefiningOp<arith::ExtFOp>();
  if (!lhsExtend || !rhsExtend ||
      lhsExtend.getResult().getType() != body.getArgument(2).getType() ||
      rhsExtend.getResult().getType() != body.getArgument(2).getType())
    return std::nullopt;
  bool direct = lhsExtend.getIn() == body.getArgument(0) &&
                rhsExtend.getIn() == body.getArgument(1);
  bool swapped = lhsExtend.getIn() == body.getArgument(1) &&
                 rhsExtend.getIn() == body.getArgument(0);
  if (!direct && !swapped)
    return std::nullopt;

  return DotMatch{op,
                  op.getDpsInputs()[0],
                  op.getDpsInputs()[1],
                  lhsType,
                  outputType,
                  add.getFastmath()};
}

static Value dimValue(OpBuilder &builder, Location location, Value tensor,
                      int64_t dimension) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(dimension))
    return builder.create<arith::ConstantIndexOp>(
        location, type.getDimSize(dimension));
  return builder.create<tensor::DimOp>(location, tensor, dimension);
}

static Value emitVectorDot(IRRewriter &rewriter, const DotMatch &match,
                           unsigned vectorWidth, bool scalableVector) {
  linalg::GenericOp op = match.reduction;
  rewriter.setInsertionPoint(op);
  Location location = op.getLoc();
  unsigned reductionDimension = match.inputType.getRank() - 1;
  SmallVector<bool> scalableDims{scalableVector};
  VectorType halfVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, rewriter.getF16Type(), scalableDims);
  VectorType floatVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, rewriter.getF32Type(), scalableDims);

  Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
  Value oneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
  Value columns = dimValue(rewriter, location, match.lhs, reductionDimension);
  Value minimumLanes = rewriter.create<arith::ConstantIndexOp>(
      location, static_cast<int64_t>(vectorWidth));
  Value step = minimumLanes;
  if (scalableVector) {
    Value vscale = rewriter.create<vector::VectorScaleOp>(location);
    step = rewriter.create<arith::MulIOp>(location, minimumLanes, vscale);
  }
  Value halfZero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF16FloatAttr(0.0));
  Value floatZero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
  Value zeroVector =
      rewriter.create<vector::SplatOp>(location, floatVector, floatZero);

  std::function<Value(OpBuilder &, unsigned, Value, ArrayRef<Value>)>
      emitParallelLoops;
  emitParallelLoops = [&](OpBuilder &builder, unsigned dimension, Value output,
                          ArrayRef<Value> parallelIndices) -> Value {
    if (dimension == reductionDimension) {
      auto inner = builder.create<scf::ForOp>(
          location, zeroIndex, columns, step, ValueRange{zeroVector},
          [&](OpBuilder &innerBuilder, Location innerLocation, Value column,
              ValueRange innerArgs) {
            SmallVector<Value> indices(parallelIndices);
            indices.push_back(column);
            Value lhsPacket = innerBuilder.create<vector::TransferReadOp>(
                innerLocation, halfVector, match.lhs, indices, halfZero);
            Value rhsPacket = innerBuilder.create<vector::TransferReadOp>(
                innerLocation, halfVector, match.rhs, indices, halfZero);
            Value lhsFloat = innerBuilder.create<arith::ExtFOp>(
                innerLocation, floatVector, lhsPacket);
            Value rhsFloat = innerBuilder.create<arith::ExtFOp>(
                innerLocation, floatVector, rhsPacket);
            Value product = innerBuilder.create<arith::MulFOp>(
                innerLocation, lhsFloat, rhsFloat);
            Value accumulated = innerBuilder.create<arith::AddFOp>(
                innerLocation, innerArgs[0], product, match.fastMath);
            innerBuilder.create<scf::YieldOp>(innerLocation, accumulated);
          });
      Value initial = builder.create<tensor::ExtractOp>(
          location, output, parallelIndices);
      Value reduced = builder.create<vector::ReductionOp>(
          location, vector::CombiningKind::ADD, inner.getResult(0), initial,
          match.fastMath);
      return builder.create<tensor::InsertOp>(location, reduced, output,
                                               parallelIndices);
    }

    Value upper = dimValue(builder, location, match.lhs, dimension);
    auto loop = builder.create<scf::ForOp>(
        location, zeroIndex, upper, oneIndex, ValueRange{output},
        [&](OpBuilder &nestedBuilder, Location nestedLocation, Value index,
            ValueRange nestedArgs) {
          SmallVector<Value> indices(parallelIndices);
          indices.push_back(index);
          Value updated = emitParallelLoops(nestedBuilder, dimension + 1,
                                            nestedArgs[0], indices);
          nestedBuilder.create<scf::YieldOp>(nestedLocation, updated);
        });
    return loop.getResult(0);
  };

  return emitParallelLoops(rewriter, 0, op.getDpsInits()[0],
                           ArrayRef<Value>{});
}

struct VectorizeLinalgMixedPrecisionDotReductionPass
    : ::intent::impl::VectorizeLinalgMixedPrecisionDotReductionBase<
          VectorizeLinalgMixedPrecisionDotReductionPass> {
  using VectorizeLinalgMixedPrecisionDotReductionBase::
      VectorizeLinalgMixedPrecisionDotReductionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (vectorWidth == 0 || (vectorWidth & (vectorWidth - 1)) != 0) {
      module.emitError("mixed dot vector width must be a power of two");
      return signalPassFailure();
    }

    SmallVector<linalg::GenericOp> worklist;
    module.walk([&](linalg::GenericOp op) {
      if (matchMixedPrecisionDot(op))
        worklist.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    unsigned count = 0;
    for (linalg::GenericOp op : worklist) {
      if (!op->getBlock())
        continue;
      std::optional<DotMatch> match = matchMixedPrecisionDot(op);
      if (!match)
        continue;
      Value replacement =
          emitVectorDot(rewriter, *match, vectorWidth, scalableVector);
      rewriter.replaceOp(op, replacement);
      ++count;
    }

    Builder builder(&getContext());
    module->setAttr("intent.rvv.mixed_dot_reduction_count",
                    builder.getI64IntegerAttr(count));
    module->setAttr("intent.rvv.mixed_dot_reduction_width",
                    builder.getI64IntegerAttr(count ? vectorWidth : 0));
    module->setAttr("intent.rvv.mixed_dot_reduction_scalable_count",
                    builder.getI64IntegerAttr(scalableVector ? count : 0));
    if (requireVectorization && count == 0) {
      module.emitError(
          "required mixed-precision dot vectorization matched no reduction");
      signalPassFailure();
    }
  }
};

} // namespace
