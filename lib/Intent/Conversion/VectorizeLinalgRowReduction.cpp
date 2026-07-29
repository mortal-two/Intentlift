//===- VectorizeLinalgRowReduction.cpp - row vector accumulators ----------===//
//
// This pass recognizes the canonical Linalg representation of a reassociable
// f32 last-axis sum.  It replaces the scalar reduction with a row loop whose
// vector accumulator remains live across the complete reduction axis and is
// horizontally finalized exactly once per row.  A scalable state derives its
// strip size from vector.vscale and therefore remains independent of a fixed
// implementation VLEN.  The matcher is structural: it neither names a kernel
// nor emits RISC-V code.
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include <functional>

namespace intent {
#define GEN_PASS_DEF_VECTORIZELINALGROWREDUCTION
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

static bool isScalarVectorElementType(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

/// Check that every operation before the final reduction add can be mapped
/// lane-wise.  Constants and captured scalar values are splatted; every other
/// operation must carry MLIR's Elementwise trait.  The accumulator may only be
/// consumed by the final add, so no hidden sequential recurrence is admitted.
static bool hasVectorizablePayload(linalg::GenericOp op, arith::AddFOp combine) {
  Block &body = op.getRegion().front();
  Value accumulator = body.getArguments().back();
  for (Operation &nested : body.without_terminator()) {
    if (nested.getNumRegions() != 0 || nested.getNumSuccessors() != 0)
      return false;
    if (&nested != combine.getOperation() &&
        llvm::is_contained(nested.getOperands(), accumulator))
      return false;
    if (!isa<arith::ConstantOp>(nested) &&
        !nested.hasTrait<OpTrait::Elementwise>())
      return false;
    if (llvm::any_of(nested.getResultTypes(),
                     [](Type type) { return !isScalarVectorElementType(type); }))
      return false;
  }
  return true;
}

/// Return the scalar add that defines the reduction body.  Requiring the
/// explicit reassoc flag keeps this observation-changing vector tree out of an
/// ordered reduction state even if a caller accidentally schedules the pass.
static arith::AddFOp getSupportedAddCombine(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumLoops() < 2 ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return {};

  ArrayRef<utils::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.size() != op.getNumLoops() ||
      iterators.back() != utils::IteratorType::reduction ||
      llvm::any_of(llvm::drop_end(iterators), [](utils::IteratorType iterator) {
        return iterator != utils::IteratorType::parallel;
      }))
    return {};

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !maps.front().isIdentity() ||
      maps.front().getNumResults() != op.getNumLoops() ||
      !isParallelProjection(maps.back()))
    return {};

  auto inputType =
      dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto outputType =
      dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!inputType || inputType.getRank() != op.getNumLoops() ||
      !inputType.getElementType().isF32() || !outputType ||
      outputType.getRank() + 1 != inputType.getRank() ||
      !outputType.getElementType().isF32())
    return {};

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2)
    return {};
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return {};
  auto add = yield.getValues().front().getDefiningOp<arith::AddFOp>();
  if (!add)
    return {};
  Value accumulator = body.getArgument(1);
  bool lhsIsAccumulator = add.getLhs() == accumulator;
  bool rhsIsAccumulator = add.getRhs() == accumulator;
  if (lhsIsAccumulator == rhsIsAccumulator ||
      !arith::bitEnumContainsAll(add.getFastmath(),
                                 arith::FastMathFlags::reassoc))
    return {};
  if (!hasVectorizablePayload(op, add))
    return {};
  return add;
}

static Value dimValue(OpBuilder &builder, Location location, Value tensor,
                      int64_t dimension) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(dimension))
    return builder.create<arith::ConstantIndexOp>(
        location, type.getDimSize(dimension));
  return builder.create<tensor::DimOp>(location, tensor, dimension);
}

static VectorType vectorTypeLike(VectorType lanes, Type elementType) {
  return VectorType::get(lanes.getShape(), elementType,
                         lanes.getScalableDims());
}

/// Clone the Linalg scalar payload lane-wise. MLIR arith/math operations carry
/// the Elementwise trait, so preserving the operation and its properties while
/// changing scalar result types to the corresponding vector type reuses the
/// dialect verifier and downstream lowering instead of maintaining a private
/// opcode table.
static Value emitVectorPayload(OpBuilder &builder, linalg::GenericOp op,
                               Value inputPacket, Value accumulator,
                               VectorType lanes) {
  Block &body = op.getRegion().front();
  IRMapping mapping;
  mapping.map(body.getArgument(0), inputPacket);
  mapping.map(body.getArgument(1), accumulator);

  auto mapCapturedScalar = [&](Value operand) -> Value {
    if (mapping.contains(operand))
      return mapping.lookup(operand);
    if (!isScalarVectorElementType(operand.getType()))
      return {};
    Value splat = builder.create<vector::SplatOp>(
        op.getLoc(), vectorTypeLike(lanes, operand.getType()), operand);
    mapping.map(operand, splat);
    return splat;
  };

  for (Operation &nested : body.without_terminator()) {
    if (auto constant = dyn_cast<arith::ConstantOp>(nested)) {
      Operation *scalar = builder.clone(nested);
      for (auto [source, cloned] :
           llvm::zip(nested.getResults(), scalar->getResults())) {
        Value splat = builder.create<vector::SplatOp>(
            op.getLoc(), vectorTypeLike(lanes, source.getType()), cloned);
        mapping.map(source, splat);
      }
      continue;
    }
    for (Value operand : nested.getOperands())
      if (!mapCapturedScalar(operand))
        return {};
    Operation *cloned = builder.clone(nested, mapping);
    for (auto [source, result] :
         llvm::zip(nested.getResults(), cloned->getResults()))
      result.setType(vectorTypeLike(lanes, source.getType()));
  }

  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  return mapping.lookupOrNull(yield.getValues().front());
}

struct VectorizeLinalgRowReductionPass
    : ::intent::impl::VectorizeLinalgRowReductionBase<
          VectorizeLinalgRowReductionPass> {
  using VectorizeLinalgRowReductionBase::VectorizeLinalgRowReductionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (vectorWidth == 0 || (vectorWidth & (vectorWidth - 1)) != 0) {
      module.emitError("row-reduction vector width must be a power of two");
      return signalPassFailure();
    }

    SmallVector<linalg::GenericOp> worklist;
    module.walk([&](linalg::GenericOp op) {
      if (getSupportedAddCombine(op))
        worklist.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    unsigned vectorizedCount = 0;
    unsigned generatedLoopCount = 0;
    for (linalg::GenericOp op : worklist) {
      arith::AddFOp combine = getSupportedAddCombine(op);
      if (!combine || !op->getBlock())
        continue;

      Location location = op.getLoc();
      rewriter.setInsertionPoint(op);
      Value input = op.getDpsInputs()[0];
      Value initialOutput = op.getDpsInits()[0];
      auto inputType = cast<RankedTensorType>(input.getType());
      auto elementType = cast<FloatType>(inputType.getElementType());
      unsigned reductionDimension = inputType.getRank() - 1;
      SmallVector<bool> scalableDims{scalableVector};
      auto vectorType = VectorType::get(
          {static_cast<int64_t>(vectorWidth)}, elementType, scalableDims);

      Value zeroIndex =
          rewriter.create<arith::ConstantIndexOp>(location, 0);
      Value oneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
      Value columns =
          dimValue(rewriter, location, input, reductionDimension);
      Value minimumLanes = rewriter.create<arith::ConstantIndexOp>(
          location, static_cast<int64_t>(vectorWidth));
      Value step = minimumLanes;
      if (scalableVector) {
        Value vscale = rewriter.create<vector::VectorScaleOp>(location);
        step = rewriter.create<arith::MulIOp>(location, minimumLanes, vscale);
      }

      arith::FastMathFlags flags = combine.getFastmath();
      std::function<Value(OpBuilder &, unsigned, Value, ArrayRef<Value>)>
          emitParallelLoops;
      emitParallelLoops = [&](OpBuilder &builder, unsigned dimension,
                              Value output,
                              ArrayRef<Value> parallelIndices) -> Value {
        if (dimension == reductionDimension) {
          Value zero = builder.create<arith::ConstantOp>(
              location, builder.getF32FloatAttr(0.0));
          Value zeroVector = builder.create<vector::SplatOp>(
              location, vectorType, zero);
          auto inner = builder.create<scf::ForOp>(
              location, zeroIndex, columns, step, ValueRange{zeroVector},
              [&](OpBuilder &innerBuilder, Location innerLocation, Value column,
                  ValueRange innerArgs) {
                SmallVector<Value> inputIndices(parallelIndices);
                inputIndices.push_back(column);
                // Out-of-bounds lanes are padded with the additive identity;
                // lowering keeps the tail masked for dynamic/scalable rows.
                Value packet = innerBuilder.create<vector::TransferReadOp>(
                    innerLocation, vectorType, input, inputIndices, zero);
                Value accumulated = emitVectorPayload(
                    innerBuilder, op, packet, innerArgs[0], vectorType);
                if (!accumulated) {
                  op.emitError("failed to vectorize admitted reduction payload");
                  accumulated = innerArgs[0];
                }
                innerBuilder.create<scf::YieldOp>(innerLocation, accumulated);
              });
          Value initial = builder.create<tensor::ExtractOp>(
              location, output, parallelIndices);
          Value reduced = builder.create<vector::ReductionOp>(
              location, vector::CombiningKind::ADD, inner.getResult(0), initial,
              flags);
          return builder.create<tensor::InsertOp>(
              location, reduced, output, parallelIndices);
        }

        Value upper = dimValue(builder, location, input, dimension);
        auto loop = builder.create<scf::ForOp>(
            location, zeroIndex, upper, oneIndex, ValueRange{output},
            [&](OpBuilder &nestedBuilder, Location nestedLocation, Value index,
                ValueRange nestedArgs) {
              SmallVector<Value> indices(parallelIndices);
              indices.push_back(index);
              Value updated = emitParallelLoops(
                  nestedBuilder, dimension + 1, nestedArgs[0], indices);
              nestedBuilder.create<scf::YieldOp>(nestedLocation, updated);
            });
        return loop.getResult(0);
      };

      Value replacement =
          emitParallelLoops(rewriter, 0, initialOutput, ArrayRef<Value>{});
      rewriter.replaceOp(op, replacement);
      ++vectorizedCount;
      generatedLoopCount += inputType.getRank();
    }

    Builder builder(&getContext());
    module->setAttr("intent.rvv.explicit_vector_reduction_count",
                    builder.getI64IntegerAttr(vectorizedCount));
    module->setAttr("intent.rvv.explicit_vector_reduction_width",
                    builder.getI64IntegerAttr(vectorizedCount ? vectorWidth : 0));
    module->setAttr("intent.rvv.explicit_vector_reduction_loop_count",
                    builder.getI64IntegerAttr(generatedLoopCount));
    module->setAttr("intent.rvv.explicit_vector_reduction_scalable_count",
                    builder.getI64IntegerAttr(scalableVector
                                                  ? vectorizedCount
                                                  : 0));
    if (requireVectorization && vectorizedCount == 0) {
      module.emitError(
          "required explicit row-reduction vectorization matched no operation");
      signalPassFailure();
    }
  }
};

} // namespace
