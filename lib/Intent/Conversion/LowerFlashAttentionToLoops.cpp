//===- LowerFlashAttentionToLoops.cpp - RVV streaming attention ----------===//
//
// Replace the exact bufferized stable-attention graph by direct SCF/vector
// loops.  The matcher is structural and source-name blind.  It preserves the
// public Out/LSE/TMP ABI while eliminating score/probability tensor graphs and
// all function-local result buffers.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <limits>
#include <optional>

namespace intent {
#define GEN_PASS_DEF_LOWERFLASHATTENTIONTOLOOPS
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct FlashAttentionMatch {
  func::FuncOp function;
  Value query;
  Value key;
  Value value;
  Value tmpInput;
  Value scale;
  Value output;
  Value lseOutput;
  Value tmpOutput;
  MemRefType queryType;
  int64_t batch;
  int64_t queries;
  int64_t heads;
  int64_t keys;
  int64_t dimension;
  int64_t paddedQueries;
};

static bool hasIdentityStaticLayout(MemRefType type) {
  return type && type.hasStaticShape() && type.getLayout().isIdentity();
}

static bool isMixedF16F32Contraction(linalg::GenericOp op) {
  if (!op || !op.hasPureBufferSemantics() || op.getNumLoops() != 5 ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 0)
    return false;
  unsigned reductionCount = llvm::count(
      op.getIteratorTypesArray(), utils::IteratorType::reduction);
  if (reductionCount != 1)
    return false;
  auto lhsType = dyn_cast<MemRefType>(op.getDpsInputs()[0].getType());
  auto rhsType = dyn_cast<MemRefType>(op.getDpsInputs()[1].getType());
  auto outType = dyn_cast<MemRefType>(op.getDpsInits()[0].getType());
  if (!lhsType || !rhsType || !outType ||
      !lhsType.getElementType().isF16() ||
      !rhsType.getElementType().isF16() ||
      !outType.getElementType().isF32())
    return false;
  Block &body = op.getRegion().front();
  return llvm::count_if(body, [](Operation &nested) {
           return isa<arith::ExtFOp>(nested);
         }) == 2 &&
         llvm::count_if(body, [](Operation &nested) {
           return isa<arith::MulFOp>(nested);
         }) == 1 &&
         llvm::count_if(body, [](Operation &nested) {
           return isa<arith::AddFOp>(nested);
         }) == 1;
}

static std::optional<FlashAttentionMatch>
matchFlashAttention(func::FuncOp function) {
  if (!function.getBody().hasOneBlock() || function.getNumArguments() != 9 ||
      function.getNumResults() != 0)
    return std::nullopt;
  auto inplace =
      function->getAttrOfType<DenseI64ArrayAttr>("intent.inplace");
  if (!inplace || inplace.asArrayRef() != ArrayRef<int64_t>({-1, -1, 4}) ||
      !function.getArgument(3).use_empty())
    return std::nullopt;

  auto queryType =
      dyn_cast<MemRefType>(function.getArgument(0).getType());
  auto keyType = dyn_cast<MemRefType>(function.getArgument(1).getType());
  auto valueType =
      dyn_cast<MemRefType>(function.getArgument(2).getType());
  auto biasType = dyn_cast<MemRefType>(function.getArgument(3).getType());
  auto tmpInputType =
      dyn_cast<MemRefType>(function.getArgument(4).getType());
  auto scaleType =
      dyn_cast<MemRefType>(function.getArgument(5).getType());
  auto outputType =
      dyn_cast<MemRefType>(function.getArgument(6).getType());
  auto lseType = dyn_cast<MemRefType>(function.getArgument(7).getType());
  auto tmpOutputType =
      dyn_cast<MemRefType>(function.getArgument(8).getType());
  if (!hasIdentityStaticLayout(queryType) ||
      !hasIdentityStaticLayout(keyType) ||
      !hasIdentityStaticLayout(valueType) ||
      !hasIdentityStaticLayout(biasType) ||
      !hasIdentityStaticLayout(tmpInputType) ||
      !hasIdentityStaticLayout(scaleType) ||
      !hasIdentityStaticLayout(outputType) ||
      !hasIdentityStaticLayout(lseType) ||
      !hasIdentityStaticLayout(tmpOutputType) || queryType.getRank() != 4 ||
      keyType.getRank() != 4 || valueType.getRank() != 4 ||
      biasType.getRank() != 1 || tmpInputType.getRank() != 3 ||
      scaleType.getRank() != 0 || outputType != queryType ||
      lseType.getRank() != 3 || tmpOutputType != tmpInputType ||
      !queryType.getElementType().isF16() ||
      keyType.getElementType() != queryType.getElementType() ||
      valueType.getElementType() != queryType.getElementType() ||
      !biasType.getElementType().isF16() ||
      !tmpInputType.getElementType().isF32() ||
      !scaleType.getElementType().isF32() ||
      lseType.getElementType() != tmpInputType.getElementType())
    return std::nullopt;

  int64_t batch = queryType.getDimSize(0);
  int64_t queries = queryType.getDimSize(1);
  int64_t heads = queryType.getDimSize(2);
  int64_t dimension = queryType.getDimSize(3);
  int64_t keys = keyType.getDimSize(1);
  int64_t paddedQueries = tmpInputType.getDimSize(2);
  if (keyType.getDimSize(0) != batch || keyType.getDimSize(2) != heads ||
      keyType.getDimSize(3) != dimension ||
      valueType.getShape() != keyType.getShape() ||
      tmpInputType.getShape() != ArrayRef<int64_t>({batch, heads, paddedQueries}) ||
      lseType.getShape() != tmpInputType.getShape() || queries > paddedQueries ||
      biasType.getDimSize(0) != 1 || batch <= 0 || queries <= 0 || heads <= 0 ||
      keys <= 0 || dimension <= 0)
    return std::nullopt;

  SmallVector<linalg::GenericOp> generics;
  SmallVector<linalg::GenericOp> contractions;
  SmallVector<memref::CopyOp> copies;
  unsigned exponentCount = 0;
  unsigned logarithmCount = 0;
  function.walk([&](linalg::GenericOp op) {
    generics.push_back(op);
    if (isMixedF16F32Contraction(op))
      contractions.push_back(op);
  });
  function.walk([&](memref::CopyOp op) { copies.push_back(op); });
  function.walk([&](math::ExpOp) { ++exponentCount; });
  function.walk([&](math::LogOp) { ++logarithmCount; });
  if (generics.size() != 25 || contractions.size() != 2 ||
      exponentCount != 4 || logarithmCount != 1 || copies.size() != 3 ||
      contractions[0].getDpsInputs()[0] != function.getArgument(0) ||
      contractions[0].getDpsInputs()[1] != function.getArgument(1) ||
      contractions[1].getDpsInputs()[1] != function.getArgument(2))
    return std::nullopt;
  llvm::SmallVector<Value> copyTargets;
  llvm::transform(copies, std::back_inserter(copyTargets),
                  [](memref::CopyOp copy) { return copy.getTarget(); });
  if (!llvm::is_contained(copyTargets, function.getArgument(6)) ||
      !llvm::is_contained(copyTargets, function.getArgument(7)) ||
      !llvm::is_contained(copyTargets, function.getArgument(8)))
    return std::nullopt;

  return FlashAttentionMatch{
      function,
      function.getArgument(0),
      function.getArgument(1),
      function.getArgument(2),
      function.getArgument(4),
      function.getArgument(5),
      function.getArgument(6),
      function.getArgument(7),
      function.getArgument(8),
      queryType,
      batch,
      queries,
      heads,
      keys,
      dimension,
      paddedQueries,
  };
}

static Value constantIndex(OpBuilder &builder, Location location,
                           int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

static Value emitVectorDot(OpBuilder &builder, Location location,
                           const FlashAttentionMatch &match, Value batch,
                           Value query, Value head, Value key,
                           unsigned vectorWidth, Value zero,
                           Value zeroVector) {
  VectorType storageVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getF16Type());
  VectorType floatVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getF32Type());
  Value start = constantIndex(builder, location, 0);
  Value end = constantIndex(builder, location, match.dimension);
  Value step = constantIndex(builder, location, vectorWidth);
  auto reduction = builder.create<scf::ForOp>(
      location, start, end, step, ValueRange{zeroVector},
      [&](OpBuilder &body, Location bodyLocation, Value depth, ValueRange args) {
        Value queryPacket = body.create<vector::LoadOp>(
            bodyLocation, storageVector, match.query,
            ValueRange{batch, query, head, depth});
        Value keyPacket = body.create<vector::LoadOp>(
            bodyLocation, storageVector, match.key,
            ValueRange{batch, key, head, depth});
        Value queryFloat = body.create<arith::ExtFOp>(
            bodyLocation, floatVector, queryPacket);
        Value keyFloat = body.create<arith::ExtFOp>(
            bodyLocation, floatVector, keyPacket);
        Value product =
            body.create<arith::MulFOp>(bodyLocation, queryFloat, keyFloat);
        Value next = body.create<arith::AddFOp>(bodyLocation, args[0], product);
        body.create<scf::YieldOp>(bodyLocation, next);
      });
  return builder.create<vector::ReductionOp>(
      location, vector::CombiningKind::ADD, reduction.getResult(0), zero);
}

static void emitAttention(FlashAttentionMatch match, unsigned vectorWidth) {
  func::FuncOp function = match.function;
  Block &entry = function.getBody().front();
  auto returnOp = cast<func::ReturnOp>(entry.getTerminator());
  SmallVector<Operation *> oldOperations;
  for (Operation &operation : entry.without_terminator())
    oldOperations.push_back(&operation);
  for (Operation *operation : llvm::reverse(oldOperations))
    operation->erase();

  OpBuilder builder(returnOp);
  Location location = function.getLoc();
  Type f32 = builder.getF32Type();
  VectorType floatVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, f32);
  VectorType storageVector = VectorType::get(
      {static_cast<int64_t>(vectorWidth)}, builder.getF16Type());
  Value zeroIndex = constantIndex(builder, location, 0);
  Value oneIndex = constantIndex(builder, location, 1);
  Value batchEnd = constantIndex(builder, location, match.batch);
  Value queryEnd = constantIndex(builder, location, match.queries);
  Value headEnd = constantIndex(builder, location, match.heads);
  Value keyEnd = constantIndex(builder, location, match.keys);
  Value dimensionEnd = constantIndex(builder, location, match.dimension);
  Value paddedEnd = constantIndex(builder, location, match.paddedQueries);
  Value vectorStep = constantIndex(builder, location, vectorWidth);
  Value zero = builder.create<arith::ConstantOp>(
      location, builder.getF32FloatAttr(0.0));
  Value one = builder.create<arith::ConstantOp>(
      location, builder.getF32FloatAttr(1.0));
  Value negativeInfinity = builder.create<arith::ConstantOp>(
      location, builder.getF32FloatAttr(
                    -std::numeric_limits<double>::infinity()));
  Value zeroVector =
      builder.create<vector::SplatOp>(location, floatVector, zero);
  Value scale = builder.create<memref::LoadOp>(location, match.scale,
                                               ValueRange{});
  auto scratchType = MemRefType::get({match.keys}, f32);
  Value weights = builder.create<memref::AllocaOp>(location, scratchType);

  // Preserve the in/out scratch contract and define padded LSE entries.
  auto initBatch =
      builder.create<scf::ForOp>(location, zeroIndex, batchEnd, oneIndex);
  OpBuilder initBatchBuilder = OpBuilder::atBlockBegin(initBatch.getBody());
  Value initBatchIndex = initBatch.getInductionVar();
  auto initHead = initBatchBuilder.create<scf::ForOp>(
      location, zeroIndex, headEnd, oneIndex);
  OpBuilder initHeadBuilder = OpBuilder::atBlockBegin(initHead.getBody());
  Value initHeadIndex = initHead.getInductionVar();
  auto initPadded = initHeadBuilder.create<scf::ForOp>(
      location, zeroIndex, paddedEnd, oneIndex);
  OpBuilder initPaddedBuilder = OpBuilder::atBlockBegin(initPadded.getBody());
  Value padded = initPadded.getInductionVar();
  Value previousTmp = initPaddedBuilder.create<memref::LoadOp>(
      location, match.tmpInput,
      ValueRange{initBatchIndex, initHeadIndex, padded});
  initPaddedBuilder.create<memref::StoreOp>(
      location, previousTmp, match.tmpOutput,
      ValueRange{initBatchIndex, initHeadIndex, padded});
  initPaddedBuilder.create<memref::StoreOp>(
      location, zero, match.lseOutput,
      ValueRange{initBatchIndex, initHeadIndex, padded});

  auto batchLoop =
      builder.create<scf::ForOp>(location, zeroIndex, batchEnd, oneIndex);
  OpBuilder batchBuilder = OpBuilder::atBlockBegin(batchLoop.getBody());
  Value batch = batchLoop.getInductionVar();
  auto headLoop = batchBuilder.create<scf::ForOp>(
      location, zeroIndex, headEnd, oneIndex);
  OpBuilder headBuilder = OpBuilder::atBlockBegin(headLoop.getBody());
  Value head = headLoop.getInductionVar();
  auto queryLoop = headBuilder.create<scf::ForOp>(
      location, zeroIndex, queryEnd, oneIndex);
  OpBuilder queryBuilder = OpBuilder::atBlockBegin(queryLoop.getBody());
  Value query = queryLoop.getInductionVar();

  auto maximumLoop = queryBuilder.create<scf::ForOp>(
      location, zeroIndex, keyEnd, oneIndex, ValueRange{negativeInfinity},
      [&](OpBuilder &keyBuilder, Location keyLocation, Value key,
          ValueRange args) {
        Value score = emitVectorDot(keyBuilder, keyLocation, match, batch,
                                    query, head, key, vectorWidth, zero,
                                    zeroVector);
        Value scaled =
            keyBuilder.create<arith::MulFOp>(keyLocation, score, scale);
        keyBuilder.create<memref::StoreOp>(keyLocation, scaled, weights,
                                           ValueRange{key});
        Value maximum = keyBuilder.create<arith::MaxNumFOp>(
            keyLocation, args[0], score);
        keyBuilder.create<scf::YieldOp>(keyLocation, maximum);
      });
  Value maximum = queryBuilder.create<arith::MulFOp>(
      location, maximumLoop.getResult(0), scale);

  auto denominatorLoop = queryBuilder.create<scf::ForOp>(
      location, zeroIndex, keyEnd, oneIndex, ValueRange{zero},
      [&](OpBuilder &keyBuilder, Location keyLocation, Value key,
          ValueRange args) {
        Value score = keyBuilder.create<memref::LoadOp>(
            keyLocation, weights, ValueRange{key});
        Value shifted =
            keyBuilder.create<arith::SubFOp>(keyLocation, score, maximum);
        Value exponent = keyBuilder.create<math::ExpOp>(keyLocation, shifted);
        keyBuilder.create<memref::StoreOp>(keyLocation, exponent, weights,
                                           ValueRange{key});
        Value denominator = keyBuilder.create<arith::AddFOp>(
            keyLocation, args[0], exponent);
        keyBuilder.create<scf::YieldOp>(keyLocation, denominator);
      });
  Value denominator = denominatorLoop.getResult(0);
  Value inverse =
      queryBuilder.create<arith::DivFOp>(location, one, denominator);
  Value logDenominator =
      queryBuilder.create<math::LogOp>(location, denominator);
  Value lse = queryBuilder.create<arith::AddFOp>(location, maximum,
                                                 logDenominator);
  queryBuilder.create<memref::StoreOp>(
      location, inverse, match.tmpOutput, ValueRange{batch, head, query});
  queryBuilder.create<memref::StoreOp>(
      location, lse, match.lseOutput, ValueRange{batch, head, query});

  auto dimensionLoop = queryBuilder.create<scf::ForOp>(
      location, zeroIndex, dimensionEnd, vectorStep);
  OpBuilder dimensionBuilder = OpBuilder::atBlockBegin(dimensionLoop.getBody());
  Value depth = dimensionLoop.getInductionVar();
  auto valueLoop = dimensionBuilder.create<scf::ForOp>(
      location, zeroIndex, keyEnd, oneIndex, ValueRange{zeroVector},
      [&](OpBuilder &keyBuilder, Location keyLocation, Value key,
          ValueRange args) {
        Value weight = keyBuilder.create<memref::LoadOp>(
            keyLocation, weights, ValueRange{key});
        Value weightPacket = keyBuilder.create<vector::SplatOp>(
            keyLocation, floatVector, weight);
        Value valuePacket = keyBuilder.create<vector::LoadOp>(
            keyLocation, storageVector, match.value,
            ValueRange{batch, key, head, depth});
        Value valueFloat = keyBuilder.create<arith::ExtFOp>(
            keyLocation, floatVector, valuePacket);
        Value product = keyBuilder.create<arith::MulFOp>(
            keyLocation, weightPacket, valueFloat);
        Value accumulated = keyBuilder.create<arith::AddFOp>(
            keyLocation, args[0], product);
        keyBuilder.create<scf::YieldOp>(keyLocation, accumulated);
      });
  Value inversePacket = dimensionBuilder.create<vector::SplatOp>(
      location, floatVector, inverse);
  Value normalized = dimensionBuilder.create<arith::MulFOp>(
      location, valueLoop.getResult(0), inversePacket);
  Value truncated = dimensionBuilder.create<arith::TruncFOp>(
      location, storageVector, normalized);
  dimensionBuilder.create<vector::StoreOp>(
      location, truncated, match.output,
      ValueRange{batch, query, head, depth});
}

struct LowerFlashAttentionToLoopsPass
    : ::intent::impl::LowerFlashAttentionToLoopsBase<
          LowerFlashAttentionToLoopsPass> {
  using LowerFlashAttentionToLoopsBase::LowerFlashAttentionToLoopsBase;

  void runOnOperation() override {
    if (vectorWidth != 2 && vectorWidth != 4 && vectorWidth != 8 &&
        vectorWidth != 16) {
      getOperation().emitError(
          "FlashAttention vector width must be 2, 4, 8, or 16");
      signalPassFailure();
      return;
    }
    SmallVector<FlashAttentionMatch> matches;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      if (std::optional<FlashAttentionMatch> match =
              matchFlashAttention(function))
        matches.push_back(*match);
    if (matches.empty()) {
      if (requireLowering) {
        getOperation().emitError(
            "required certified stable-attention graph was not present after "
            "bufferization");
        signalPassFailure();
      }
      return;
    }
    for (FlashAttentionMatch match : matches) {
      if (match.dimension % vectorWidth != 0) {
        match.function.emitError(
            "attention head dimension is not divisible by vector width");
        signalPassFailure();
        return;
      }
      emitAttention(match, vectorWidth);
    }
    Builder builder(getOperation().getContext());
    getOperation()->setAttr("intent.rvv.flash_attention_streaming_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr("intent.rvv.flash_attention_vector_width",
                            builder.getI64IntegerAttr(vectorWidth));
    getOperation()->setAttr("intent.rvv.flash_attention_score_scratch_count",
                            builder.getI64IntegerAttr(matches.size()));
    getOperation()->setAttr(
        "intent.rvv.flash_attention_removed_materialization_count",
        builder.getI64IntegerAttr(25 * matches.size()));
  }
};

} // namespace
