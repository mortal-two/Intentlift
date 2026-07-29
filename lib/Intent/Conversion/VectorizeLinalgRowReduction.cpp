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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <functional>
#include <limits>
#include <optional>

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
static bool hasVectorizablePayload(linalg::GenericOp op, Operation *combine) {
  Block &body = op.getRegion().front();
  Value accumulator = body.getArguments().back();
  for (Operation &nested : body.without_terminator()) {
    if (nested.getNumRegions() != 0 || nested.getNumSuccessors() != 0)
      return false;
    if (&nested != combine &&
        llvm::is_contained(nested.getOperands(), accumulator))
      return false;
    if (!isa<arith::ConstantOp>(nested) &&
        !nested.hasTrait<OpTrait::Elementwise>())
      return false;
    if (llvm::any_of(nested.getResultTypes(),
                     [](Type type) { return !isScalarVectorElementType(type); }))
      return false;
    for (Value operand : nested.getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand);
          argument && argument.getOwner() == &body)
        continue;
      if (Operation *definition = operand.getDefiningOp();
          definition && definition->getBlock() == &body)
        continue;
      if (!isScalarVectorElementType(operand.getType()))
        return false;
    }
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
  if (!hasVectorizablePayload(op, add.getOperation()))
    return {};
  return add;
}

/// Return the scalar maxnumf that defines an associative row maximum.  Unlike
/// floating add, maxnumf does not need a reassociation flag: the Intent max
/// reduction lowers to this explicit associative combiner.  The rest of the
/// shape/map contract is identical to the add path so the same strip-mined
/// vector accumulator and tail masking apply.
static arith::MaxNumFOp getSupportedMaxCombine(linalg::GenericOp op) {
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
  auto maximum = yield.getValues().front().getDefiningOp<arith::MaxNumFOp>();
  if (!maximum)
    return {};
  Value accumulator = body.getArgument(1);
  bool lhsIsAccumulator = maximum.getLhs() == accumulator;
  bool rhsIsAccumulator = maximum.getRhs() == accumulator;
  if (lhsIsAccumulator == rhsIsAccumulator ||
      !hasVectorizablePayload(op, maximum.getOperation()))
    return {};
  return maximum;
}

/// A joint-statistics state may look through pure elementwise tensor producers
/// only when every shaped operand is an identity-mapped view of the same
/// logical iteration space.  The destination argument must be dead: otherwise
/// the region contains a recurrence rather than a producer expression.
static bool isSupportedElementwiseProducer(linalg::GenericOp op,
                                             RankedTensorType expectedType) {
  if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() == 0 ||
      op.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      op.getNumLoops() != static_cast<unsigned>(expectedType.getRank()))
    return false;

  if (llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType iterator) {
        return iterator != utils::IteratorType::parallel;
      }))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != static_cast<size_t>(op.getNumDpsInputs() + 1) ||
      llvm::any_of(maps, [&](AffineMap map) {
        return !map.isIdentity() ||
               map.getNumResults() != static_cast<unsigned>(expectedType.getRank());
      }))
    return false;

  for (OpOperand *input : op.getDpsInputOperands())
    if (input->get().getType() != expectedType)
      return false;
  if (op.getDpsInits()[0].getType() != expectedType ||
      op.getResult(0).getType() != expectedType)
    return false;

  Block &body = op.getRegion().front();
  unsigned inputCount = op.getNumDpsInputs();
  if (body.getNumArguments() != inputCount + 1 ||
      !body.getArgument(inputCount).use_empty())
    return false;

  for (Operation &nested : body.without_terminator()) {
    if (nested.getNumRegions() != 0 || nested.getNumSuccessors() != 0 ||
        (!isa<arith::ConstantOp>(nested) &&
         !nested.hasTrait<OpTrait::Elementwise>()) ||
        llvm::any_of(nested.getResultTypes(), [](Type type) {
          return !isScalarVectorElementType(type);
        }))
      return false;
    for (Value operand : nested.getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        if (argument.getOwner() == &body &&
            argument.getArgNumber() < inputCount)
          continue;
      }
      if (Operation *definition = operand.getDefiningOp();
          definition && definition->getBlock() == &body)
        continue;
      if (!isScalarVectorElementType(operand.getType()))
        return false;
    }
  }

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yield && yield.getValues().size() == 1 &&
         yield.getValues().front() != body.getArgument(inputCount) &&
         yield.getValues().front().getType().isF32();
}

static bool isZeroInitialized(linalg::GenericOp reduction) {
  Value init = reduction.getDpsInits()[0];
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    return fill.getInputs().size() == 1 &&
           matchPattern(fill.getInputs().front(), m_AnyZeroFloat());
  auto generic = init.getDefiningOp<linalg::GenericOp>();
  auto initType = dyn_cast<RankedTensorType>(init.getType());
  if (!generic || !generic.hasPureTensorSemantics() ||
      generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1 || !initType ||
      generic.getNumLoops() != static_cast<unsigned>(initType.getRank()) ||
      llvm::any_of(generic.getIteratorTypesArray(),
                   [](utils::IteratorType iterator) {
                     return iterator != utils::IteratorType::parallel;
                   }))
    return false;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 1 || !maps.front().isIdentity() ||
      maps.front().getNumResults() !=
          static_cast<unsigned>(initType.getRank()))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
  return yield && yield.getValues().size() == 1 &&
         matchPattern(yield.getValues().front(), m_AnyZeroFloat());
}

struct ReductionPlan {
  linalg::GenericOp reduction;
  Value root;
  Value statisticalInput;
  SmallVector<linalg::GenericOp> producers;
  arith::FastMathFlags fastMath;
  bool synthesizeSquarePayload = false;
  linalg::GenericOp rawMomentMeanReduction;
};

static Value collectProducerRoot(
    Value value, RankedTensorType expectedType,
    SmallVectorImpl<linalg::GenericOp> &orderedProducers,
    llvm::SmallPtrSetImpl<Operation *> &visited,
    llvm::SmallPtrSetImpl<Operation *> &visiting, bool &valid) {
  if (!valid || value.getType() != expectedType) {
    valid = false;
    return {};
  }
  auto producer = value.getDefiningOp<linalg::GenericOp>();
  if (!producer)
    return value;
  // An unsupported Linalg producer is a materialized frontier, not something
  // this pass may clone.  It can still be the common root read by several
  // reductions; only supported identity-mapped elementwise DAGs are crossed.
  if (!isSupportedElementwiseProducer(producer, expectedType))
    return value;
  if (!visiting.insert(producer.getOperation()).second) {
    valid = false;
    return {};
  }

  Value commonRoot;
  for (OpOperand *input : producer.getDpsInputOperands()) {
    Value root = collectProducerRoot(input->get(), expectedType,
                                     orderedProducers, visited, visiting, valid);
    if (!valid || !root || (commonRoot && commonRoot != root)) {
      valid = false;
      visiting.erase(producer.getOperation());
      return {};
    }
    commonRoot = root;
  }
  visiting.erase(producer.getOperation());
  if (visited.insert(producer.getOperation()).second)
    orderedProducers.push_back(producer);
  return commonRoot;
}

static std::optional<ReductionPlan> buildReductionPlan(linalg::GenericOp op) {
  if (!getSupportedAddCombine(op) || !isZeroInitialized(op))
    return std::nullopt;
  Value input = op.getDpsInputs()[0];
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!inputType || !outputType || !outputType.hasStaticShape() ||
      outputType.getEncoding())
    return std::nullopt;

  ReductionPlan plan{op,
                     {},
                     input,
                     {},
                     getSupportedAddCombine(op).getFastmath(),
                     false,
                     {}};
  llvm::SmallPtrSet<Operation *, 8> visited;
  llvm::SmallPtrSet<Operation *, 8> visiting;
  bool valid = true;
  plan.root = collectProducerRoot(input, inputType, plan.producers, visited,
                                  visiting, valid);
  if (!valid || !plan.root || plan.root.getType() != inputType)
    return std::nullopt;
  return plan;
}

struct SquareReductionMatch {
  Value centeredInput;
  arith::FastMathFlags fastMath;
};

static bool hasScalarEquality(Value scalarTensor, int64_t expected,
                              Operation *context) {
  auto argument = dyn_cast<BlockArgument>(scalarTensor);
  auto function = context->getParentOfType<func::FuncOp>();
  if (!argument || !function || argument.getOwner() != &function.front())
    return false;
  auto equalities =
      function->getAttrOfType<DenseI64ArrayAttr>("intent.scalar_equalities");
  if (!equalities || equalities.size() % 2 != 0)
    return false;
  ArrayRef<int64_t> values = equalities.asArrayRef();
  for (unsigned index = 0; index < values.size(); index += 2)
    if (values[index] == static_cast<int64_t>(argument.getArgNumber()) &&
        values[index + 1] == expected)
      return true;
  return false;
}

static std::optional<SquareReductionMatch>
matchSquareSumReduction(linalg::GenericOp reduction) {
  if (arith::AddFOp combine = getSupportedAddCombine(reduction)) {
    Block &body = reduction.getRegion().front();
    Value accumulator = body.getArgument(1);
    Value payload = combine.getLhs() == accumulator ? combine.getRhs()
                                                    : combine.getLhs();
    auto square = payload.getDefiningOp<arith::MulFOp>();
    if (square && square.getLhs() == body.getArgument(0) &&
        square.getRhs() == body.getArgument(0))
      return SquareReductionMatch{reduction.getDpsInputs()[0],
                                  combine.getFastmath()};
    return std::nullopt;
  }

  if (!reduction.hasPureTensorSemantics() || reduction.getNumLoops() < 2 ||
      reduction.getNumDpsInputs() != 2 || reduction.getNumDpsInits() != 1 ||
      reduction->getNumResults() != 1)
    return std::nullopt;
  ArrayRef<utils::IteratorType> iterators = reduction.getIteratorTypesArray();
  if (iterators.back() != utils::IteratorType::reduction ||
      llvm::any_of(llvm::drop_end(iterators), [](utils::IteratorType iterator) {
        return iterator != utils::IteratorType::parallel;
      }))
    return std::nullopt;

  SmallVector<AffineMap> maps = reduction.getIndexingMapsArray();
  if (maps.size() != 3 || !isParallelProjection(maps.back()))
    return std::nullopt;
  auto outputType =
      dyn_cast<RankedTensorType>(reduction.getDpsInits()[0].getType());
  if (!outputType || !outputType.getElementType().isF32())
    return std::nullopt;

  Block &body = reduction.getRegion().front();
  if (body.getNumArguments() != 3)
    return std::nullopt;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto combine = yield && yield.getValues().size() == 1
                     ? yield.getValues().front().getDefiningOp<arith::AddFOp>()
                     : arith::AddFOp{};
  Value accumulator = body.getArgument(2);
  if (!combine ||
      !arith::bitEnumContainsAll(combine.getFastmath(),
                                 arith::FastMathFlags::reassoc))
    return std::nullopt;
  bool lhsIsAccumulator = combine.getLhs() == accumulator;
  bool rhsIsAccumulator = combine.getRhs() == accumulator;
  if (lhsIsAccumulator == rhsIsAccumulator)
    return std::nullopt;
  Value payload = lhsIsAccumulator ? combine.getRhs() : combine.getLhs();
  auto square = payload.getDefiningOp<arith::MulFOp>();
  if (!square || square.getLhs() != square.getRhs())
    return std::nullopt;
  auto select = square.getLhs().getDefiningOp<arith::SelectOp>();
  if (!select || !matchPattern(select.getFalseValue(), m_AnyZeroFloat()))
    return std::nullopt;
  auto centeredArgument = dyn_cast<BlockArgument>(select.getTrueValue());
  if (!centeredArgument || centeredArgument.getOwner() != &body ||
      centeredArgument.getArgNumber() >= 2)
    return std::nullopt;
  unsigned centeredIndex = centeredArgument.getArgNumber();
  if (!maps[centeredIndex].isIdentity())
    return std::nullopt;
  Value centeredInput = reduction.getDpsInputs()[centeredIndex];
  auto centeredType = dyn_cast<RankedTensorType>(centeredInput.getType());
  if (!centeredType || !centeredType.getElementType().isF32() ||
      centeredType.getRank() != reduction.getNumLoops() ||
      !centeredType.hasStaticShape())
    return std::nullopt;

  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;
  auto indexCast = compare.getLhs().getDefiningOp<arith::IndexCastOp>();
  auto index = indexCast
                   ? indexCast.getIn().getDefiningOp<linalg::IndexOp>()
                   : linalg::IndexOp{};
  auto boundArgument = dyn_cast<BlockArgument>(compare.getRhs());
  if (!index ||
      index.getDim() != static_cast<uint64_t>(centeredType.getRank() - 1) ||
      !boundArgument || boundArgument.getOwner() != &body ||
      boundArgument.getArgNumber() >= 2 ||
      boundArgument == centeredArgument)
    return std::nullopt;
  unsigned boundIndex = boundArgument.getArgNumber();
  if (maps[boundIndex].getNumResults() != 0 ||
      !hasScalarEquality(reduction.getDpsInputs()[boundIndex],
                         centeredType.getShape().back(), reduction))
    return std::nullopt;
  return SquareReductionMatch{centeredInput, combine.getFastmath()};
}

static bool isFeatureCountTensor(Value tensor, int64_t featureCount,
                                 Operation *context) {
  if (hasScalarEquality(tensor, featureCount, context))
    return true;
  auto producer = tensor.getDefiningOp<linalg::GenericOp>();
  if (!producer || !producer.hasPureTensorSemantics() ||
      producer.getNumLoops() != 0 || producer.getNumDpsInputs() != 1 ||
      producer.getNumDpsInits() != 1 || producer->getNumResults() != 1)
    return false;
  Block &body = producer.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 2 || !yield ||
      yield.getValues().size() != 1)
    return false;
  Value result = yield.getValues().front();
  if (auto cast = result.getDefiningOp<arith::SIToFPOp>())
    if (cast.getIn() == body.getArgument(0))
      return hasScalarEquality(producer.getDpsInputs()[0], featureCount,
                               context);
  return false;
}

static linalg::GenericOp matchFeatureAverage(Value meanValue,
                                             int64_t featureCount) {
  auto mean = meanValue.getDefiningOp<linalg::GenericOp>();
  auto meanType = dyn_cast<RankedTensorType>(meanValue.getType());
  if (!mean || !meanType || !mean.hasPureTensorSemantics() ||
      mean.getNumDpsInputs() == 0 || mean.getNumDpsInits() != 1 ||
      mean->getNumResults() != 1 ||
      mean.getNumLoops() != static_cast<unsigned>(meanType.getRank()) ||
      llvm::any_of(mean.getIteratorTypesArray(),
                   [](utils::IteratorType iterator) {
                     return iterator != utils::IteratorType::parallel;
                   }))
    return {};
  Block &body = mean.getRegion().front();
  unsigned inputCount = mean.getNumDpsInputs();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto divide = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<arith::DivFOp>()
                    : arith::DivFOp{};
  if (body.getNumArguments() != inputCount + 1 || !divide)
    return {};
  auto numerator = dyn_cast<BlockArgument>(divide.getLhs());
  if (!numerator || numerator.getOwner() != &body ||
      numerator.getArgNumber() >= inputCount)
    return {};
  unsigned numeratorIndex = numerator.getArgNumber();
  auto numeratorMap = mean.getIndexingMapsArray()[numeratorIndex];
  if (!numeratorMap.isIdentity())
    return {};
  Value numeratorTensor = mean.getDpsInputs()[numeratorIndex];
  auto reduction = numeratorTensor.getDefiningOp<linalg::GenericOp>();
  if (!reduction || !getSupportedAddCombine(reduction) ||
      !isZeroInitialized(reduction))
    return {};

  Value denominator = divide.getRhs();
  if (auto argument = dyn_cast<BlockArgument>(denominator)) {
    if (argument.getOwner() != &body ||
        argument.getArgNumber() >= inputCount)
      return {};
    unsigned denominatorIndex = argument.getArgNumber();
    AffineMap denominatorMap = mean.getIndexingMapsArray()[denominatorIndex];
    if (denominatorMap.getNumResults() != 0 ||
        !isFeatureCountTensor(mean.getDpsInputs()[denominatorIndex],
                              featureCount, mean))
      return {};
  } else {
    Attribute attribute;
    if (!matchPattern(denominator, m_Constant(&attribute)))
      return {};
    auto floatValue = dyn_cast<FloatAttr>(attribute);
    if (!floatValue ||
        floatValue.getValue().convertToDouble() !=
            static_cast<double>(featureCount))
      return {};
  }
  return reduction;
}

/// Recognize a centered-variance graph and replace only its statistical input:
///
///   sum(x) -> mean; sum((x - mean)^2)
///
/// becomes one joint traversal accumulating sum(x) and sum(x^2).  Finalization
/// reconstructs the original centered sum as sum(x^2) - N * mean^2.  The
/// transformation is observation-changing for floating point and is admitted
/// only in the same reassociation/QD state as the joint vector reduction.
static std::optional<ReductionPlan>
buildRawMomentVariancePlan(linalg::GenericOp varianceReduction) {
  std::optional<SquareReductionMatch> squareReduction =
      matchSquareSumReduction(varianceReduction);
  if (!squareReduction || !isZeroInitialized(varianceReduction))
    return std::nullopt;

  Value centeredValue = squareReduction->centeredInput;
  auto centeredType = dyn_cast<RankedTensorType>(centeredValue.getType());
  auto centered = centeredValue.getDefiningOp<linalg::GenericOp>();
  if (!centeredType || !centered || !centered.hasPureTensorSemantics() ||
      centered.getNumDpsInputs() != 2 || centered.getNumDpsInits() != 1 ||
      centered->getNumResults() != 1 ||
      centered.getNumLoops() != static_cast<unsigned>(centeredType.getRank()) ||
      llvm::any_of(centered.getIteratorTypesArray(),
                   [](utils::IteratorType iterator) {
                     return iterator != utils::IteratorType::parallel;
                   }))
    return std::nullopt;

  SmallVector<AffineMap> centeredMaps = centered.getIndexingMapsArray();
  if (centeredMaps.size() != 3 || !centeredMaps[0].isIdentity() ||
      !isParallelProjection(centeredMaps[1]) ||
      !centeredMaps[2].isIdentity())
    return std::nullopt;
  Block &centeredBody = centered.getRegion().front();
  if (centeredBody.getNumArguments() != 3 ||
      !centeredBody.getArgument(2).use_empty())
    return std::nullopt;
  auto centeredYield =
      dyn_cast<linalg::YieldOp>(centeredBody.getTerminator());
  auto subtract = centeredYield && centeredYield.getValues().size() == 1
                      ? centeredYield.getValues().front()
                            .getDefiningOp<arith::SubFOp>()
                      : arith::SubFOp{};
  if (!subtract || subtract.getLhs() != centeredBody.getArgument(0) ||
      subtract.getRhs() != centeredBody.getArgument(1))
    return std::nullopt;

  Value statisticalInput = centered.getDpsInputs()[0];
  Value meanValue = centered.getDpsInputs()[1];
  auto statisticalType = dyn_cast<RankedTensorType>(statisticalInput.getType());
  auto meanType = dyn_cast<RankedTensorType>(meanValue.getType());
  if (!statisticalType || statisticalType != centeredType || !meanType ||
      meanType.getRank() + 1 != statisticalType.getRank() ||
      !statisticalType.hasStaticShape())
    return std::nullopt;

  int64_t featureCount = statisticalType.getShape().back();
  linalg::GenericOp meanReduction =
      matchFeatureAverage(meanValue, featureCount);
  if (!meanReduction || meanReduction.getDpsInits()[0].getType() != meanType)
    return std::nullopt;

  std::optional<ReductionPlan> meanPlan = buildReductionPlan(meanReduction);
  if (!meanPlan || meanReduction.getDpsInputs()[0] != statisticalInput ||
      meanPlan->root.getType() != statisticalType)
    return std::nullopt;

  auto varianceOutputType =
      dyn_cast<RankedTensorType>(varianceReduction.getDpsInits()[0].getType());
  if (!varianceOutputType || varianceOutputType != meanType ||
      !varianceOutputType.hasStaticShape() || varianceOutputType.getEncoding())
    return std::nullopt;

  ReductionPlan plan{varianceReduction,
                     meanPlan->root,
                     statisticalInput,
                     meanPlan->producers,
                     squareReduction->fastMath,
                     true,
                     meanReduction};
  return plan;
}

static AffineMap statisticalInputMap(ReductionPlan plan) {
  SmallVector<AffineMap> maps = plan.reduction.getIndexingMapsArray();
  for (auto [index, input] : llvm::enumerate(plan.reduction.getDpsInputs()))
    if (input == plan.statisticalInput)
      return maps[index];
  // A raw-moment plan deliberately bypasses the materialized centered tensor:
  // its statistical input is the pre-centering root, while the reduction still
  // names the centered tensor as its identity-mapped operand.  The matcher has
  // already proved that exact centered-value structure; recover its logical map
  // by type and require it to be unique.
  AffineMap rawMomentMap;
  if (plan.rawMomentMeanReduction)
    for (auto [index, input] : llvm::enumerate(plan.reduction.getDpsInputs()))
      if (input.getType() == plan.statisticalInput.getType() &&
          maps[index].isIdentity()) {
        if (rawMomentMap)
          return {};
        rawMomentMap = maps[index];
      }
  if (rawMomentMap)
    return rawMomentMap;
  return {};
}

static bool sameJointDomain(const ReductionPlan &lhs,
                            const ReductionPlan &rhs) {
  linalg::GenericOp lhsReduction = lhs.reduction;
  linalg::GenericOp rhsReduction = rhs.reduction;
  AffineMap lhsStatisticalMap = statisticalInputMap(lhs);
  AffineMap rhsStatisticalMap = statisticalInputMap(rhs);
  AffineMap lhsOutputMap = lhsReduction.getIndexingMapsArray().back();
  AffineMap rhsOutputMap = rhsReduction.getIndexingMapsArray().back();
  if (lhs.root != rhs.root ||
      lhsReduction->getBlock() != rhsReduction->getBlock() ||
      lhs.statisticalInput.getType() != rhs.statisticalInput.getType() ||
      lhsReduction.getDpsInits()[0].getType() !=
          rhsReduction.getDpsInits()[0].getType() ||
      !lhsStatisticalMap || lhsStatisticalMap != rhsStatisticalMap ||
      lhsOutputMap != rhsOutputMap ||
      lhsReduction.getIteratorTypesArray() !=
          rhsReduction.getIteratorTypesArray())
    return false;
  return true;
}

static bool capturedValuesDominate(linalg::GenericOp op, Operation *anchor,
                                   DominanceInfo &dominance) {
  Block &body = op.getRegion().front();
  for (Operation &nested : body.getOperations())
    for (Value operand : nested.getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand);
          argument && argument.getOwner() == &body)
        continue;
      if (Operation *definition = operand.getDefiningOp();
          definition && definition->getBlock() == &body)
        continue;
      if (!dominance.dominates(operand, anchor))
        return false;
    }
  return true;
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

static Value emitElementwiseProducerPayload(OpBuilder &builder,
                                             linalg::GenericOp op,
                                             ArrayRef<Value> inputPackets,
                                             VectorType lanes) {
  Block &body = op.getRegion().front();
  IRMapping mapping;
  for (auto [argument, packet] :
       llvm::zip(body.getArguments().take_front(inputPackets.size()),
                 inputPackets))
    mapping.map(argument, packet);

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
    if (isa<arith::ConstantOp>(nested)) {
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

static Value emitProducerPacket(OpBuilder &builder, Value tensor,
                                DenseMap<Value, Value> &packets,
                                VectorType lanes) {
  if (auto found = packets.find(tensor); found != packets.end())
    return found->second;
  auto producer = tensor.getDefiningOp<linalg::GenericOp>();
  if (!producer)
    return {};
  SmallVector<Value> inputs;
  inputs.reserve(producer.getNumDpsInputs());
  for (OpOperand *input : producer.getDpsInputOperands()) {
    Value packet = emitProducerPacket(builder, input->get(), packets, lanes);
    if (!packet)
      return {};
    inputs.push_back(packet);
  }
  Value result =
      emitElementwiseProducerPayload(builder, producer, inputs, lanes);
  if (result)
    packets.try_emplace(tensor, result);
  return result;
}

static SmallVector<Value>
emitJointReductionGroup(IRRewriter &rewriter, ArrayRef<ReductionPlan> plans,
                        unsigned vectorWidth, bool scalableVector,
                        bool &payloadEmissionFailed) {
  assert(plans.size() >= 2 && "joint reduction requires multiple statistics");
  linalg::GenericOp firstReduction = plans.front().reduction;
  Operation *anchor = firstReduction.getOperation();
  for (const ReductionPlan &plan : llvm::drop_begin(plans)) {
    linalg::GenericOp reduction = plan.reduction;
    if (reduction->isBeforeInBlock(anchor))
      anchor = reduction.getOperation();
  }

  DominanceInfo dominance(anchor->getParentOp());
  for (const ReductionPlan &plan : plans) {
    if (!dominance.dominates(plan.root, anchor) ||
        !capturedValuesDominate(plan.reduction, anchor, dominance))
      return {};
    for (linalg::GenericOp producer : plan.producers)
      if (!capturedValuesDominate(producer, anchor, dominance))
        return {};
  }

  rewriter.setInsertionPoint(anchor);
  Location location = anchor->getLoc();
  Value root = plans.front().root;
  auto rootType = cast<RankedTensorType>(root.getType());
  unsigned reductionDimension = rootType.getRank() - 1;
  auto elementType = cast<FloatType>(rootType.getElementType());
  auto vectorType = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, elementType,
      SmallVector<bool>{scalableVector});

  Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
  Value oneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
  Value columns = dimValue(rewriter, location, root, reductionDimension);
  Value minimumLanes = rewriter.create<arith::ConstantIndexOp>(
      location, static_cast<int64_t>(vectorWidth));
  Value step = minimumLanes;
  if (scalableVector) {
    Value vscale = rewriter.create<vector::VectorScaleOp>(location);
    step = rewriter.create<arith::MulIOp>(location, minimumLanes, vscale);
  }
  Value zero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
  Value zeroVector =
      rewriter.create<vector::SplatOp>(location, vectorType, zero);

  SmallVector<Value> initialOutputs;
  initialOutputs.reserve(plans.size());
  for (const ReductionPlan &plan : plans) {
    linalg::GenericOp reduction = plan.reduction;
    auto outputType =
        cast<RankedTensorType>(reduction.getDpsInits()[0].getType());
    initialOutputs.push_back(rewriter.create<tensor::EmptyOp>(
        location, outputType.getShape(), outputType.getElementType()));
  }

  std::function<SmallVector<Value>(OpBuilder &, unsigned, ValueRange,
                                   ArrayRef<Value>)>
      emitParallelLoops;
  emitParallelLoops = [&](OpBuilder &builder, unsigned dimension,
                          ValueRange outputs,
                          ArrayRef<Value> parallelIndices) -> SmallVector<Value> {
    if (dimension == reductionDimension) {
      SmallVector<Value> zeroVectors(plans.size(), zeroVector);
      auto inner = builder.create<scf::ForOp>(
          location, zeroIndex, columns, step, zeroVectors,
          [&](OpBuilder &innerBuilder, Location innerLocation, Value column,
              ValueRange innerArgs) {
            SmallVector<Value> rootIndices(parallelIndices);
            rootIndices.push_back(column);
            Value rootPacket = innerBuilder.create<vector::TransferReadOp>(
                innerLocation, vectorType, root, rootIndices, zero);
            DenseMap<Value, Value> packets;
            packets.try_emplace(root, rootPacket);
            SmallVector<Value> accumulated;
            accumulated.reserve(plans.size());
            for (auto [index, plan] : llvm::enumerate(plans)) {
              linalg::GenericOp reduction = plan.reduction;
              Value inputPacket = emitProducerPacket(
                  innerBuilder, plan.statisticalInput, packets, vectorType);
              if (!inputPacket) {
                payloadEmissionFailed = true;
                accumulated.push_back(innerArgs[index]);
                continue;
              }
              Value next;
              if (plan.synthesizeSquarePayload) {
                Value square = innerBuilder.create<arith::MulFOp>(
                    innerLocation, inputPacket, inputPacket);
                next = innerBuilder.create<arith::AddFOp>(
                    innerLocation, innerArgs[index], square, plan.fastMath);
              } else {
                next = emitVectorPayload(innerBuilder, reduction, inputPacket,
                                         innerArgs[index], vectorType);
              }
              if (!next) {
                payloadEmissionFailed = true;
                next = innerArgs[index];
              }
              accumulated.push_back(next);
            }
            innerBuilder.create<scf::YieldOp>(innerLocation, accumulated);
          });

      SmallVector<Value> reducedScalars;
      reducedScalars.reserve(plans.size());
      for (auto [index, plan] : llvm::enumerate(plans)) {
        reducedScalars.push_back(builder.create<vector::ReductionOp>(
            location, vector::CombiningKind::ADD, inner.getResult(index), zero,
            plan.fastMath));
      }

      for (auto [index, plan] : llvm::enumerate(plans)) {
        if (!plan.rawMomentMeanReduction)
          continue;
        auto mean = llvm::find_if(plans, [&](const ReductionPlan &candidate) {
          return candidate.reduction == plan.rawMomentMeanReduction;
        });
        if (mean == plans.end()) {
          payloadEmissionFailed = true;
          continue;
        }
        unsigned meanIndex = std::distance(plans.begin(), mean);
        auto statisticalType =
            cast<RankedTensorType>(plan.statisticalInput.getType());
        int64_t featureCount = statisticalType.getShape().back();
        Value featureCountValue = builder.create<arith::ConstantOp>(
            location, builder.getF32FloatAttr(featureCount));
        Value meanValue = builder.create<arith::DivFOp>(
            location, reducedScalars[meanIndex], featureCountValue);
        Value meanSquare =
            builder.create<arith::MulFOp>(location, meanValue, meanValue);
        Value centeredCorrection = builder.create<arith::MulFOp>(
            location, featureCountValue, meanSquare);
        reducedScalars[index] = builder.create<arith::SubFOp>(
            location, reducedScalars[index], centeredCorrection);
      }

      SmallVector<Value> updatedOutputs;
      updatedOutputs.reserve(plans.size());
      for (auto [index, reduced] : llvm::enumerate(reducedScalars)) {
        updatedOutputs.push_back(builder.create<tensor::InsertOp>(
            location, reduced, outputs[index], parallelIndices));
      }
      return updatedOutputs;
    }

    Value upper = dimValue(builder, location, root, dimension);
    auto loop = builder.create<scf::ForOp>(
        location, zeroIndex, upper, oneIndex, outputs,
        [&](OpBuilder &nestedBuilder, Location nestedLocation, Value index,
            ValueRange nestedArgs) {
          SmallVector<Value> indices(parallelIndices);
          indices.push_back(index);
          SmallVector<Value> updated = emitParallelLoops(
              nestedBuilder, dimension + 1, nestedArgs, indices);
          nestedBuilder.create<scf::YieldOp>(nestedLocation, updated);
        });
    return SmallVector<Value>(loop.getResults().begin(), loop.getResults().end());
  };

  return emitParallelLoops(rewriter, 0, initialOutputs, ArrayRef<Value>{});
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
    if (requireJointReductions && !jointReductions) {
      module.emitError(
          "require-joint-reductions needs joint-reductions to be enabled");
      return signalPassFailure();
    }

    SmallVector<linalg::GenericOp> worklist;
    SmallVector<linalg::GenericOp> allGenerics;
    module.walk([&](linalg::GenericOp op) {
      allGenerics.push_back(op);
      if (getSupportedAddCombine(op) || getSupportedMaxCombine(op))
        worklist.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    unsigned vectorizedCount = 0;
    unsigned vectorizedMaxCount = 0;
    unsigned generatedLoopCount = 0;
    unsigned jointGroupCount = 0;
    unsigned jointReductionCount = 0;
    unsigned jointRawMomentVarianceCount = 0;
    bool payloadEmissionFailed = false;
    llvm::SmallPtrSet<Operation *, 16> jointedOperations;

    if (jointReductions) {
      SmallVector<ReductionPlan> plans;
      for (linalg::GenericOp op : allGenerics) {
        std::optional<ReductionPlan> plan = buildRawMomentVariancePlan(op);
        if (!plan && getSupportedAddCombine(op))
          plan = buildReductionPlan(op);
        if (plan)
          plans.push_back(std::move(*plan));
      }

      SmallVector<bool> consumed(plans.size(), false);
      for (unsigned leader = 0; leader < plans.size(); ++leader) {
        if (consumed[leader])
          continue;
        SmallVector<ReductionPlan> group{plans[leader]};
        SmallVector<unsigned> memberIndices{leader};
        for (unsigned candidate = leader + 1; candidate < plans.size();
             ++candidate) {
          if (!consumed[candidate] &&
              sameJointDomain(plans[leader], plans[candidate])) {
            group.push_back(plans[candidate]);
            memberIndices.push_back(candidate);
          }
        }
        if (group.size() < 2)
          continue;

        SmallVector<Value> replacements = emitJointReductionGroup(
            rewriter, group, vectorWidth, scalableVector,
            payloadEmissionFailed);
        if (payloadEmissionFailed) {
          module.emitError(
              "failed to emit an admitted joint row-reduction payload");
          return signalPassFailure();
        }
        if (replacements.size() != group.size())
          continue;
        for (auto [plan, replacement, memberIndex] :
             llvm::zip(group, replacements, memberIndices)) {
          jointedOperations.insert(plan.reduction.getOperation());
          consumed[memberIndex] = true;
          rewriter.replaceOp(plan.reduction, replacement);
        }
        ++jointGroupCount;
        jointReductionCount += group.size();
        jointRawMomentVarianceCount += llvm::count_if(
            group, [](const ReductionPlan &plan) {
              return static_cast<bool>(plan.rawMomentMeanReduction);
            });
        vectorizedCount += group.size();
        generatedLoopCount +=
            cast<RankedTensorType>(group.front().root.getType()).getRank();
      }
    }

    for (linalg::GenericOp op : worklist) {
      if (jointedOperations.contains(op.getOperation()))
        continue;
      arith::AddFOp combine = getSupportedAddCombine(op);
      arith::MaxNumFOp maximum = getSupportedMaxCombine(op);
      if ((!combine && !maximum) || !op->getBlock())
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

      arith::FastMathFlags flags = combine ? combine.getFastmath()
                                           : arith::FastMathFlags::none;
      vector::CombiningKind combiningKind =
          combine ? vector::CombiningKind::ADD
                  : vector::CombiningKind::MAXNUMF;
      std::function<Value(OpBuilder &, unsigned, Value, ArrayRef<Value>)>
          emitParallelLoops;
      emitParallelLoops = [&](OpBuilder &builder, unsigned dimension,
                              Value output,
                              ArrayRef<Value> parallelIndices) -> Value {
        if (dimension == reductionDimension) {
          Value identity = builder.create<arith::ConstantOp>(
              location,
              builder.getF32FloatAttr(
                  combine ? 0.0
                          : -std::numeric_limits<double>::infinity()));
          Value identityVector = builder.create<vector::SplatOp>(
              location, vectorType, identity);
          auto inner = builder.create<scf::ForOp>(
              location, zeroIndex, columns, step, ValueRange{identityVector},
              [&](OpBuilder &innerBuilder, Location innerLocation, Value column,
                  ValueRange innerArgs) {
                SmallVector<Value> inputIndices(parallelIndices);
                inputIndices.push_back(column);
                // Out-of-bounds lanes are padded with the combiner identity;
                // lowering keeps the tail masked for dynamic/scalable rows.
                Value packet = innerBuilder.create<vector::TransferReadOp>(
                    innerLocation, vectorType, input, inputIndices, identity);
                Value accumulated = emitVectorPayload(
                    innerBuilder, op, packet, innerArgs[0], vectorType);
                if (!accumulated) {
                  payloadEmissionFailed = true;
                  accumulated = innerArgs[0];
                }
                innerBuilder.create<scf::YieldOp>(innerLocation, accumulated);
              });
          Value initial = builder.create<tensor::ExtractOp>(
              location, output, parallelIndices);
          Value reduced = builder.create<vector::ReductionOp>(
              location, combiningKind, inner.getResult(0), initial,
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
      if (maximum)
        ++vectorizedMaxCount;
      generatedLoopCount += inputType.getRank();
    }

    if (payloadEmissionFailed) {
      module.emitError("failed to emit an admitted row-reduction payload");
      return signalPassFailure();
    }

    Builder builder(&getContext());
    module->setAttr("intent.rvv.explicit_vector_reduction_count",
                    builder.getI64IntegerAttr(vectorizedCount));
    module->setAttr("intent.rvv.explicit_vector_reduction_width",
                    builder.getI64IntegerAttr(vectorizedCount ? vectorWidth : 0));
    module->setAttr("intent.rvv.explicit_vector_max_reduction_count",
                    builder.getI64IntegerAttr(vectorizedMaxCount));
    module->setAttr("intent.rvv.explicit_vector_reduction_loop_count",
                    builder.getI64IntegerAttr(generatedLoopCount));
    module->setAttr("intent.rvv.explicit_vector_reduction_scalable_count",
                    builder.getI64IntegerAttr(scalableVector
                                                  ? vectorizedCount
                                                  : 0));
    module->setAttr("intent.rvv.joint_vector_reduction_group_count",
                    builder.getI64IntegerAttr(jointGroupCount));
    module->setAttr("intent.rvv.joint_vector_reduction_count",
                    builder.getI64IntegerAttr(jointReductionCount));
    module->setAttr("intent.rvv.joint_vector_reduction_root_load_count",
                    builder.getI64IntegerAttr(jointGroupCount));
    module->setAttr("intent.rvv.joint_raw_moment_variance_count",
                    builder.getI64IntegerAttr(jointRawMomentVarianceCount));
    if (requireJointReductions && jointGroupCount == 0) {
      module.emitError(
          "required joint row-reduction vectorization matched no group");
      return signalPassFailure();
    }
    if (requireVectorization && vectorizedCount == 0) {
      module.emitError(
          "required explicit row-reduction vectorization matched no operation");
      signalPassFailure();
    }
  }
};

} // namespace
