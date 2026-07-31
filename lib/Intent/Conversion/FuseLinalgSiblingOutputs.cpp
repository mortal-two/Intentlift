//===- FuseLinalgSiblingOutputs.cpp - returned multi-output fusion --------===//
//
// Merges independent returned elementwise siblings into one multi-output
// linalg.generic.  This is a structural matcher over Linalg, never a kernel-name
// recipe.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"

namespace intent {
#define GEN_PASS_DEF_FUSELINALGSIBLINGOUTPUTS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct SiblingMatch {
  func::ReturnOp result;
  linalg::GenericOp first;
  linalg::GenericOp second;
};

static bool isIdentityOutput(AffineMap map, unsigned rank) {
  return map.getNumDims() == rank && map.getNumSymbols() == 0 &&
         map.getNumResults() == rank && map.isIdentity();
}

static bool hasParallelIterators(linalg::GenericOp operation) {
  return llvm::all_of(operation.getIteratorTypesArray(),
                      [](utils::IteratorType iterator) {
                        return iterator == utils::IteratorType::parallel;
                      });
}

static bool isReturnedOnly(Value value, func::ReturnOp result) {
  return value.hasOneUse() && *value.getUsers().begin() == result.getOperation();
}

static bool isWriteOnlySingleOutput(linalg::GenericOp operation) {
  if (!operation.hasPureTensorSemantics() || operation->getNumResults() != 1 ||
      operation.getNumDpsInits() != 1 || operation.getNumLoops() == 0 ||
      !hasParallelIterators(operation))
    return false;
  Block &body = operation.getRegion().front();
  unsigned inputCount = operation.getNumDpsInputs();
  if (body.getNumArguments() != inputCount + 1 ||
      !body.getArgument(inputCount).use_empty())
    return false;
  SmallVector<AffineMap> maps = operation.getIndexingMapsArray();
  return maps.size() == inputCount + 1 &&
         isIdentityOutput(maps.back(), operation.getNumLoops());
}

static std::optional<SiblingMatch> matchSiblings(func::FuncOp function) {
  if (!function.getBody().hasOneBlock())
    return std::nullopt;
  auto result = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  if (!result || result.getNumOperands() != 2)
    return std::nullopt;
  auto first = result.getOperand(0).getDefiningOp<linalg::GenericOp>();
  auto second = result.getOperand(1).getDefiningOp<linalg::GenericOp>();
  if (!first || !second || first == second ||
      !isWriteOnlySingleOutput(first) || !isWriteOnlySingleOutput(second) ||
      first->getBlock() != second->getBlock() ||
      first.getNumLoops() != second.getNumLoops() ||
      first.getIteratorTypesArray() != second.getIteratorTypesArray() ||
      !isReturnedOnly(first.getResult(0), result) ||
      !isReturnedOnly(second.getResult(0), result))
    return std::nullopt;
  auto firstType = dyn_cast<RankedTensorType>(first.getResult(0).getType());
  auto secondType = dyn_cast<RankedTensorType>(second.getResult(0).getType());
  if (!firstType || !secondType || !firstType.hasStaticShape() ||
      !secondType.hasStaticShape() ||
      firstType.getShape() != secondType.getShape())
    return std::nullopt;
  return SiblingMatch{result, first, second};
}

struct UnifiedInput {
  Value value;
  AffineMap map;
};

static SmallVector<unsigned>
appendInputs(linalg::GenericOp operation,
             SmallVectorImpl<UnifiedInput> &unified) {
  SmallVector<unsigned> positions;
  SmallVector<AffineMap> maps = operation.getIndexingMapsArray();
  for (auto [index, value] : llvm::enumerate(operation.getDpsInputs())) {
    AffineMap map = maps[index];
    auto existing = llvm::find_if(unified, [&](const UnifiedInput &input) {
      return input.value == value && input.map == map;
    });
    if (existing == unified.end()) {
      positions.push_back(unified.size());
      unified.push_back(UnifiedInput{value, map});
    } else {
      positions.push_back(std::distance(unified.begin(), existing));
    }
  }
  return positions;
}

static Value cloneScalarBody(OpBuilder &builder, linalg::GenericOp source,
                             ArrayRef<unsigned> inputPositions,
                             ValueRange mergedArguments,
                             unsigned mergedInputCount,
                             unsigned outputPosition) {
  Block &sourceBody = source.getRegion().front();
  IRMapping mapping;
  for (auto [sourcePosition, mergedPosition] :
       llvm::enumerate(inputPositions))
    mapping.map(sourceBody.getArgument(sourcePosition),
                mergedArguments[mergedPosition]);
  mapping.map(sourceBody.getArgument(inputPositions.size()),
              mergedArguments[mergedInputCount + outputPosition]);
  for (Operation &operation : sourceBody.without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<linalg::YieldOp>(sourceBody.getTerminator());
  return mapping.lookupOrDefault(yield.getValues().front());
}

static void fuseSiblings(SiblingMatch match) {
  linalg::GenericOp insertion = match.first->isBeforeInBlock(match.second)
                                    ? match.first
                                    : match.second;
  OpBuilder builder(insertion);
  Location location = insertion.getLoc();
  SmallVector<UnifiedInput> unifiedInputs;
  SmallVector<unsigned> firstPositions =
      appendInputs(match.first, unifiedInputs);
  SmallVector<unsigned> secondPositions =
      appendInputs(match.second, unifiedInputs);
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  for (const UnifiedInput &input : unifiedInputs) {
    inputs.push_back(input.value);
    maps.push_back(input.map);
  }

  auto firstType = cast<RankedTensorType>(match.first.getResult(0).getType());
  auto secondType = cast<RankedTensorType>(match.second.getResult(0).getType());
  Value firstInit = builder.create<tensor::EmptyOp>(
      location, firstType.getShape(), firstType.getElementType());
  Value secondInit = builder.create<tensor::EmptyOp>(
      location, secondType.getShape(), secondType.getElementType());
  firstInit.getDefiningOp()->setAttr("intent.sibling_output_slot",
                                    builder.getI64IntegerAttr(0));
  secondInit.getDefiningOp()->setAttr("intent.sibling_output_slot",
                                     builder.getI64IntegerAttr(1));
  SmallVector<AffineMap> firstMaps = match.first.getIndexingMapsArray();
  SmallVector<AffineMap> secondMaps = match.second.getIndexingMapsArray();
  maps.push_back(firstMaps.back());
  maps.push_back(secondMaps.back());

  auto fused = builder.create<linalg::GenericOp>(
      location, TypeRange{firstType, secondType}, ValueRange(inputs),
      ValueRange{firstInit, secondInit}, maps,
      match.first.getIteratorTypesArray(),
      [&](OpBuilder &bodyBuilder, Location, ValueRange arguments) {
        Value firstResult = cloneScalarBody(
            bodyBuilder, match.first, firstPositions, arguments,
            unifiedInputs.size(), /*outputPosition=*/0);
        Value secondResult = cloneScalarBody(
            bodyBuilder, match.second, secondPositions, arguments,
            unifiedInputs.size(), /*outputPosition=*/1);
        bodyBuilder.create<linalg::YieldOp>(
            location, ValueRange{firstResult, secondResult});
      });

  match.first.getResult(0).replaceAllUsesWith(fused.getResult(0));
  match.second.getResult(0).replaceAllUsesWith(fused.getResult(1));
  match.first.erase();
  match.second.erase();
}

struct FuseLinalgSiblingOutputsPass
    : ::intent::impl::FuseLinalgSiblingOutputsBase<
          FuseLinalgSiblingOutputsPass> {
  using FuseLinalgSiblingOutputsBase::FuseLinalgSiblingOutputsBase;

  void runOnOperation() override {
    SmallVector<SiblingMatch> matches;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      if (std::optional<SiblingMatch> match = matchSiblings(function))
        matches.push_back(*match);
    if (matches.empty()) {
      if (requireFusion) {
        getOperation().emitError(
            "required returned elementwise sibling pair was not present");
        signalPassFailure();
      }
      return;
    }
    for (const SiblingMatch &match : matches)
      fuseSiblings(match);
    Builder builder(getOperation().getContext());
    getOperation()->setAttr("intent.rvv.sibling_output_fusion_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.sibling_output_eliminated_traversals",
                            builder.getI64IntegerAttr(matches.size()));
  }
};

} // namespace
