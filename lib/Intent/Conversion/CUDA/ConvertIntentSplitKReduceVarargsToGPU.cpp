//===- ConvertIntentSplitKReduceVarargsToGPU.cpp -------------------------===//
//
// Structural fast path for the captured xFormers `_splitK_reduce_varargs`
// specialization.  Unlike the legacy packed H=G=1 surrogate, this rewrite
// preserves four independent Out pointers, four independent LSE pointers, the
// two destination pointers, and every runtime stride scalar in the unrolled
// 55-argument ABI.  The live P4 graph first expresses the semantic
// varargs->canonical-packed-B mapping through two concat(axis=1) operations;
// this pass consumes that exact graph and emits one stride-aware GPU launch.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cmath>
#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTSPLITKREDUCEVARARGSTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static bool hasAxis(ReduceOp op, int64_t axis) {
  return op.getAxes().size() == 1 &&
         cast<IntegerAttr>(op.getAxes()[0]).getInt() == axis;
}

static bool isPermutation(TransposeOp op, ArrayRef<int64_t> expected) {
  if (!op || op.getPerm().size() != expected.size())
    return false;
  for (auto [attr, value] : llvm::zip(op.getPerm(), expected))
    if (cast<IntegerAttr>(attr).getInt() != value)
      return false;
  return true;
}

static bool areEntryArguments(ValueRange values, Block &entry,
                              unsigned first) {
  if (values.size() + first > entry.getNumArguments())
    return false;
  for (auto [index, value] : llvm::enumerate(values)) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &entry ||
        argument.getArgNumber() != first + index)
      return false;
  }
  return true;
}

static int64_t elementCount(RankedTensorType type) {
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    count *= dim;
  return count;
}

static bool isFloatSplat(Value value, double expected, double tolerance = 0.0) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                        : DenseElementsAttr();
  if (!dense || !dense.isSplat() ||
      !isa<FloatType>(dense.getElementType()))
    return false;
  APFloat actual = dense.getSplatValue<APFloat>();
  if (std::isinf(expected))
    return actual.isInfinity() && actual.isNegative() == std::signbit(expected);
  return std::abs(actual.convertToDouble() - expected) <= tolerance;
}

static void collectAssociativeLeaves(Value value, StringRef fn,
                                     SmallVectorImpl<Value> &leaves) {
  auto binary = value.getDefiningOp<MapBinaryOp>();
  if (!binary || binary.getFn() != fn) {
    leaves.push_back(value);
    return;
  }
  collectAssociativeLeaves(binary.getLhs(), fn, leaves);
  collectAssociativeLeaves(binary.getRhs(), fn, leaves);
}

static bool hasExactlyValues(ArrayRef<Value> values,
                             ArrayRef<Value> expected) {
  if (values.size() != expected.size())
    return false;
  llvm::DenseSet<Value> unique;
  for (Value value : values)
    if (!unique.insert(value).second)
      return false;
  return llvm::all_of(expected,
                      [&](Value value) { return unique.contains(value); });
}

static bool isTrailingUnitExpansion(Value value, Value source) {
  auto expand = value.getDefiningOp<tensor::ExpandShapeOp>();
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  if (!expand || expand.getSrc() != source || !sourceType || !resultType ||
      !sourceType.hasStaticShape() || !resultType.hasStaticShape() ||
      resultType.getRank() != sourceType.getRank() + 1 ||
      resultType.getElementType() != sourceType.getElementType() ||
      resultType.getShape().back() != 1)
    return false;
  return resultType.getShape().drop_back() == sourceType.getShape();
}

static bool allBodyOperationsReachResults(func::FuncOp fn,
                                          func::ReturnOp ret) {
  Block &entry = fn.getBody().front();
  llvm::DenseSet<Operation *> reachable;
  SmallVector<Value> worklist(ret.getOperands().begin(), ret.getOperands().end());
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != &entry ||
        !reachable.insert(definition).second)
      continue;
    llvm::append_range(worklist, definition->getOperands());
  }
  return reachable.size() ==
         static_cast<size_t>(
             std::distance(entry.without_terminator().begin(),
                           entry.without_terminator().end()));
}

static void attachSplitKStateAttributes(
    func::FuncOp lowered, StringRef path, StringRef topology,
    int64_t blockThreads, int64_t rowsPerBlock, int64_t warpsPerRow,
    int64_t splitCount, int64_t gridBlocks, int64_t vectorWidth,
    int64_t shuffleCount, int64_t barrierCount, ArrayRef<StringRef> trace) {
  MLIRContext *ctx = lowered.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  auto integer = [&](int64_t value) { return IntegerAttr::get(i64, value); };
  lowered->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(ctx, path));
  lowered->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(ctx, topology));
  SmallVector<Attribute> traceAttrs;
  llvm::transform(trace, std::back_inserter(traceAttrs),
                  [&](StringRef item) -> Attribute {
                    return StringAttr::get(ctx, item);
                  });
  lowered->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(ctx, traceAttrs));
  lowered->setAttr("intent.backend.approximation_mode",
                   StringAttr::get(ctx, "source_exp2_log2"));
  lowered->setAttr("intent.backend.fma_count", integer(0));
  lowered->setAttr("intent.backend.approximate_div_count", integer(0));
  lowered->setAttr("intent.backend.shuffle_count", integer(shuffleCount));
  lowered->setAttr("intent.backend.barrier_count", integer(barrierCount));
  lowered->setAttr("intent.backend.all_reduce_count", integer(0));
  lowered->setAttr("intent.backend.block_threads", integer(blockThreads));
  lowered->setAttr("intent.backend.grid_blocks", integer(gridBlocks));
  lowered->setAttr("intent.backend.grid_stride_loop_count", integer(0));
  lowered->setAttr("intent.backend.rows_per_block", integer(rowsPerBlock));
  lowered->setAttr("intent.backend.warps_per_row", integer(warpsPerRow));
  lowered->setAttr("intent.backend.vector_width", integer(vectorWidth));
  lowered->setAttr("intent.backend.stage_count", integer(1));
  lowered->setAttr("intent.backend.reduction_parts", integer(splitCount));
  lowered->setAttr("intent.backend.predicated_value_if_count", integer(0));
  lowered->setAttr("intent.backend.folded_constexpr_select_count", integer(0));
  lowered->setAttr("intent.backend.host_value_capture_count", integer(0));
  lowered->setAttr("intent.backend.epilogue_load_motion_count", integer(0));
}

/// Match the source-neutral, unrolled weighted-merge graph emitted from a typed
/// J whose vararg arity is statically known.  The split axis is represented by
/// K independent Out/LSE operands rather than by a packed tensor dimension:
///
///   max_lse = max(-inf, lse[0], ..., lse[K-1])
///   w[i] = exp2((lse[i] - max_lse) * log2(e))
///   out = sum(out[i] * w[i]) / sum(w[i])
///   lse = max_lse + log2(sum(w[i])) / log2(e)
///
/// The matcher follows SSA dependencies and checks whole-graph reachability. It
/// does not inspect function names, source paths, receipts, or benchmark IDs.
struct UnrolledSplitKVarargsMatch {
  func::FuncOp fn;
  int64_t splitCount = 0;
  int64_t B = 0;
  int64_t G = 0;
  int64_t H = 0;
  int64_t M = 0;
  int64_t D = 0;

  explicit UnrolledSplitKVarargsMatch(func::FuncOp fn) : fn(fn) {}

  LogicalResult match() {
    auto type = fn.getFunctionType();
    if (!fn.getBody().hasOneBlock() || type.getNumResults() != 2 ||
        type.getNumInputs() < 4 || type.getNumInputs() % 2 != 0)
      return failure();
    splitCount = type.getNumInputs() / 2;
    Block &entry = fn.getBody().front();

    SmallVector<RankedTensorType> outTypes;
    SmallVector<RankedTensorType> lseTypes;
    outTypes.reserve(splitCount);
    lseTypes.reserve(splitCount);
    for (int64_t split = 0; split < splitCount; ++split) {
      auto outType = dyn_cast<RankedTensorType>(type.getInput(split));
      auto lseType =
          dyn_cast<RankedTensorType>(type.getInput(splitCount + split));
      if (!outType || !lseType || !outType.hasStaticShape() ||
          !lseType.hasStaticShape() || outType.getRank() != 5 ||
          lseType.getRank() != 4 ||
          !isa<Float32Type>(outType.getElementType()) ||
          !isa<Float32Type>(lseType.getElementType()))
        return failure();
      if (split != 0 &&
          (outType != outTypes.front() || lseType != lseTypes.front()))
        return failure();
      outTypes.push_back(outType);
      lseTypes.push_back(lseType);
    }

    B = outTypes.front().getDimSize(0);
    G = outTypes.front().getDimSize(1);
    H = outTypes.front().getDimSize(2);
    M = outTypes.front().getDimSize(3);
    D = outTypes.front().getDimSize(4);
    if (B <= 0 || G <= 0 || H <= 0 || M <= 0 || D <= 0 ||
        lseTypes.front().getShape() !=
            ArrayRef<int64_t>({B, G, H, M}) ||
        type.getResult(0) != outTypes.front() ||
        type.getResult(1) != lseTypes.front())
      return failure();

    auto ret = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!ret || ret.getNumOperands() != 2)
      return failure();
    auto outSelect = ret.getOperand(0).getDefiningOp<SelectOp>();
    auto lseSelect = ret.getOperand(1).getDefiningOp<SelectOp>();
    if (!outSelect || !lseSelect)
      return failure();

    auto quotient = outSelect.getY().getDefiningOp<MapBinaryOp>();
    if (!quotient || quotient.getFn() != "div" ||
        !isFloatSplat(outSelect.getX(), 0.0))
      return failure();
    Value numerator = quotient.getLhs();
    auto denominatorExpand =
        quotient.getRhs().getDefiningOp<tensor::ExpandShapeOp>();
    if (!denominatorExpand)
      return failure();
    Value denominator = denominatorExpand.getSrc();
    if (!isTrailingUnitExpansion(denominatorExpand.getResult(), denominator))
      return failure();

    Value condition = lseSelect.getCond();
    if (!isTrailingUnitExpansion(outSelect.getCond(), condition))
      return failure();
    auto allMasked = condition.getDefiningOp<MapBinaryOp>();
    if (!allMasked || allMasked.getFn() != "eq")
      return failure();
    Value lseMax;
    Value negativeInfinity;
    if (isFloatSplat(allMasked.getRhs(),
                     -std::numeric_limits<double>::infinity())) {
      lseMax = allMasked.getLhs();
      negativeInfinity = allMasked.getRhs();
    } else if (isFloatSplat(allMasked.getLhs(),
                            -std::numeric_limits<double>::infinity())) {
      lseMax = allMasked.getRhs();
      negativeInfinity = allMasked.getLhs();
    } else {
      return failure();
    }
    if (lseSelect.getX() != lseMax)
      return failure();

    SmallVector<Value> maxLeaves;
    collectAssociativeLeaves(lseMax, "max", maxLeaves);
    SmallVector<Value> expectedMaxLeaves{negativeInfinity};
    for (int64_t split = 0; split < splitCount; ++split)
      expectedMaxLeaves.push_back(entry.getArgument(splitCount + split));
    if (!hasExactlyValues(maxLeaves, expectedMaxLeaves))
      return failure();

    SmallVector<Value> weightLeaves;
    collectAssociativeLeaves(denominator, "add", weightLeaves);
    if (weightLeaves.size() != static_cast<size_t>(splitCount))
      return failure();
    SmallVector<Value> weights(splitCount);
    Value log2e;
    for (Value weightValue : weightLeaves) {
      auto weight = weightValue.getDefiningOp<MapUnaryOp>();
      auto scale = weight ? weight.getInput().getDefiningOp<MapBinaryOp>()
                          : MapBinaryOp();
      if (!weight || weight.getFn() != "exp2" || !scale ||
          scale.getFn() != "mul")
        return failure();
      Value shiftValue;
      Value scaleValue;
      if (isFloatSplat(scale.getLhs(), 1.44269504, 1.0e-6)) {
        scaleValue = scale.getLhs();
        shiftValue = scale.getRhs();
      } else if (isFloatSplat(scale.getRhs(), 1.44269504, 1.0e-6)) {
        scaleValue = scale.getRhs();
        shiftValue = scale.getLhs();
      } else {
        return failure();
      }
      if (log2e && log2e != scaleValue)
        return failure();
      log2e = scaleValue;
      auto shift = shiftValue.getDefiningOp<MapBinaryOp>();
      auto lseArgument = shift ? dyn_cast<BlockArgument>(shift.getLhs())
                               : BlockArgument();
      if (!shift || shift.getFn() != "sub" || shift.getRhs() != lseMax ||
          !lseArgument || lseArgument.getOwner() != &entry ||
          lseArgument.getArgNumber() < static_cast<unsigned>(splitCount) ||
          lseArgument.getArgNumber() >=
              static_cast<unsigned>(2 * splitCount))
        return failure();
      unsigned split = lseArgument.getArgNumber() - splitCount;
      if (weights[split])
        return failure();
      weights[split] = weightValue;
    }
    if (!llvm::all_of(weights, [](Value value) { return bool(value); }))
      return failure();

    SmallVector<Value> numeratorLeaves;
    collectAssociativeLeaves(numerator, "add", numeratorLeaves);
    if (numeratorLeaves.size() != static_cast<size_t>(splitCount))
      return failure();
    SmallVector<bool> seenOut(splitCount, false);
    for (Value leaf : numeratorLeaves) {
      auto contraction = leaf.getDefiningOp<ContractionOp>();
      auto outArgument = contraction
                             ? dyn_cast<BlockArgument>(contraction.getA())
                             : BlockArgument();
      if (!contraction ||
          contraction.getSubscripts() != "abcde,abcd->abcde" ||
          !outArgument || outArgument.getOwner() != &entry ||
          outArgument.getArgNumber() >= static_cast<unsigned>(splitCount))
        return failure();
      unsigned split = outArgument.getArgNumber();
      if (seenOut[split] || contraction.getB() != weights[split])
        return failure();
      seenOut[split] = true;
    }
    if (!llvm::all_of(seenOut, [](bool value) { return value; }))
      return failure();

    auto lseAdd = lseSelect.getY().getDefiningOp<MapBinaryOp>();
    if (!lseAdd || lseAdd.getFn() != "add")
      return failure();
    Value logarithm = lseAdd.getLhs() == lseMax   ? lseAdd.getRhs()
                      : lseAdd.getRhs() == lseMax ? lseAdd.getLhs()
                                                   : Value();
    auto toNaturalLog = logarithm.getDefiningOp<MapBinaryOp>();
    auto log2 = toNaturalLog
                    ? toNaturalLog.getLhs().getDefiningOp<MapUnaryOp>()
                    : MapUnaryOp();
    if (!toNaturalLog || toNaturalLog.getFn() != "div" ||
        toNaturalLog.getRhs() != log2e || !log2 || log2.getFn() != "log2" ||
        log2.getInput() != denominator)
      return failure();

    return success(allBodyOperationsReachResults(fn, ret));
  }

  LogicalResult rewrite(int64_t rowsPerBlock, int64_t warpsPerRow,
                        int64_t vectorWidth, bool broadcastScalars) {
    if (rowsPerBlock != 1 && rowsPerBlock != 2 && rowsPerBlock != 4 &&
        rowsPerBlock != 8)
      return fn.emitError("rows-per-block must be one of 1, 2, 4 or 8");
    if (warpsPerRow != 1 && warpsPerRow != 2 && warpsPerRow != 4 &&
        warpsPerRow != 8)
      return fn.emitError("warps-per-row must be one of 1, 2, 4 or 8");
    if (warpsPerRow > 1 && rowsPerBlock != 1)
      return fn.emitError(
          "multi-warp D partition currently requires rows-per-block=1");
    if (rowsPerBlock * warpsPerRow > 32)
      return fn.emitError("split-K launch exceeds 32 warps per block");
    if ((vectorWidth != 1 && vectorWidth != 2 && vectorWidth != 4) ||
        (D % vectorWidth) != 0)
      return fn.emitError(
          "vector-width must be 1, 2 or 4 and divide output D");

    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder builder(ctx);
    Type f32 = builder.getF32Type();
    SmallVector<Type> abiTypes;
    auto oldType = fn.getFunctionType();
    for (Type input : oldType.getInputs()) {
      auto tensorType = cast<RankedTensorType>(input);
      abiTypes.push_back(MemRefType::get({elementCount(tensorType)}, f32));
    }
    for (Type result : oldType.getResults()) {
      auto tensorType = cast<RankedTensorType>(result);
      abiTypes.push_back(MemRefType::get({elementCount(tensorType)}, f32));
    }
    auto newType = FunctionType::get(ctx, abiTypes, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    builder.setInsertionPoint(fn);
    auto gpuFn = builder.create<func::FuncOp>(loc, name, newType);
    gpuFn->setAttr("intentir.fast_path",
                   builder.getStringAttr("cuda.splitk.unrolled-merge.v2"));
    gpuFn->setAttr("intentir.runtime_abi_args",
                   builder.getI64IntegerAttr(2 * splitCount + 2));
    SmallVector<StringRef> transformationTrace{
        "cuda.splitk.unrolled_weighted_merge.v2",
        "cuda.materialization.whole_graph_elimination.v1"};
    if (broadcastScalars && warpsPerRow == 1)
      transformationTrace.push_back(
          "cuda.splitk.warp_broadcast_row_scalars.v1");
    if (broadcastScalars && warpsPerRow > 1)
      transformationTrace.push_back(
          "cuda.splitk.workgroup_broadcast_row_scalars.v1");
    if (warpsPerRow > 1)
      transformationTrace.push_back(
          "cuda.splitk.multi_warp_d_partition.v1");
    attachSplitKStateAttributes(
        gpuFn, "splitk_weighted_merge",
        warpsPerRow > 1
            ? StringRef("multi_warp_d_partition")
            : broadcastScalars ? StringRef("warp_broadcast_weighted_merge")
                               : StringRef("unrolled_vararg_merge"),
        rowsPerBlock * warpsPerRow * 32, rowsPerBlock, warpsPerRow, splitCount,
        ((M + rowsPerBlock - 1) / rowsPerBlock) * B * G * H, vectorWidth,
        broadcastScalars && warpsPerRow == 1 ? splitCount + 2 : 0,
        broadcastScalars && warpsPerRow > 1 ? 1 : 0, transformationTrace);
    Block *entry = gpuFn.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value gridX = builder.create<arith::ConstantIndexOp>(
        loc, (M + rowsPerBlock - 1) / rowsPerBlock);
    Value gridY = builder.create<arith::ConstantIndexOp>(loc, B * G * H);
    Value blockX = builder.create<arith::ConstantIndexOp>(
        loc, rowsPerBlock * warpsPerRow * 32);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    gpu::LaunchOp launch;
    if (broadcastScalars && warpsPerRow > 1) {
      Value sharedBytes = builder.create<arith::ConstantOp>(
          loc, builder.getI32IntegerAttr((splitCount + 2) * 4));
      launch = builder.create<gpu::LaunchOp>(loc, gridX, gridY, c1, blockX, c1,
                                             c1, sharedBytes);
    } else {
      launch = builder.create<gpu::LaunchOp>(loc, gridX, gridY, c1, blockX, c1,
                                             c1);
    }
    builder.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value by = launch.getBlockIds().y;
    Value tx = launch.getThreadIds().x;
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value scalarScratch;
    if (broadcastScalars && warpsPerRow > 1) {
      auto workgroupSpace =
          gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
      auto rawType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                     nullptr, workgroupSpace);
      Value raw = builder.create<gpu::DynamicSharedMemoryOp>(loc, rawType);
      auto scratchType = MemRefType::get({splitCount + 2}, f32, nullptr,
                                         workgroupSpace);
      scalarScratch = builder.create<memref::ViewOp>(
          loc, scratchType, raw, c0, ValueRange{});
    }
    for (unsigned index = 0; index < entry->getNumArguments(); ++index)
      (void)builder.create<memref::DimOp>(loc, entry->getArgument(index), c0);

    Value c32 = builder.create<arith::ConstantIndexOp>(loc, 32);
    Value cRows =
        builder.create<arith::ConstantIndexOp>(loc, rowsPerBlock);
    Value cWarpsPerRow =
        builder.create<arith::ConstantIndexOp>(loc, warpsPerRow);
    Value cG = builder.create<arith::ConstantIndexOp>(loc, G);
    Value cH = builder.create<arith::ConstantIndexOp>(loc, H);
    Value cGH = builder.create<arith::ConstantIndexOp>(loc, G * H);
    Value cM = builder.create<arith::ConstantIndexOp>(loc, M);
    Value cD = builder.create<arith::ConstantIndexOp>(loc, D);
    Value warp = builder.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = builder.create<arith::RemUIOp>(loc, tx, c32);
    Value rowSlot =
        builder.create<arith::DivUIOp>(loc, warp, cWarpsPerRow);
    Value warpInRow =
        builder.create<arith::RemUIOp>(loc, warp, cWarpsPerRow);
    Value rowBase = builder.create<arith::MulIOp>(loc, bx, cRows);
    Value m = builder.create<arith::AddIOp>(loc, rowBase, rowSlot);
    Value z = builder.create<arith::DivUIOp>(loc, by, cGH);
    Value groupHead = builder.create<arith::RemUIOp>(loc, by, cGH);
    Value g = builder.create<arith::DivUIOp>(loc, groupHead, cH);
    Value h = builder.create<arith::RemUIOp>(loc, groupHead, cH);
    Value rowValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, m, cM);

    builder.create<scf::IfOp>(
        loc, rowValid, [&](OpBuilder &b, Location bodyLoc) {
          auto rowOffset = [&](OpBuilder &offsetBuilder,
                               Location offsetLoc) -> Value {
            Value offset = offsetBuilder.create<arith::MulIOp>(offsetLoc, z, cG);
            offset = offsetBuilder.create<arith::AddIOp>(offsetLoc, offset, g);
            offset = offsetBuilder.create<arith::MulIOp>(offsetLoc, offset, cH);
            offset = offsetBuilder.create<arith::AddIOp>(offsetLoc, offset, h);
            offset = offsetBuilder.create<arith::MulIOp>(offsetLoc, offset, cM);
            return offsetBuilder.create<arith::AddIOp>(offsetLoc, offset, m);
          };

          Value row = rowOffset(b, bodyLoc);
          Value negativeInfinity = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(
                           -std::numeric_limits<float>::infinity()));
          Value zero =
              b.create<arith::ConstantOp>(bodyLoc, b.getF32FloatAttr(0.0));
          Value log2e = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(1.44269504));

          Value laneZero = b.create<arith::CmpIOp>(
              bodyLoc, arith::CmpIPredicate::eq, lane, c0);
          Value rowLaneBase = b.create<arith::MulIOp>(
              bodyLoc, warpInRow, c32);
          Value rowLane =
              b.create<arith::AddIOp>(bodyLoc, rowLaneBase, lane);
          Value lseMaxValue;
          SmallVector<Value> weights(splitCount);
          Value sumexp;
          if (!broadcastScalars) {
            SmallVector<Value> lseValues(splitCount);
            lseMaxValue = negativeInfinity;
            for (int64_t split = 0; split < splitCount; ++split) {
              lseValues[split] = b.create<memref::LoadOp>(
                  bodyLoc, entry->getArgument(splitCount + split),
                  ValueRange{row});
              lseMaxValue = b.create<arith::MaxNumFOp>(
                  bodyLoc, lseMaxValue, lseValues[split]);
            }
            sumexp = zero;
            for (int64_t split = 0; split < splitCount; ++split) {
              Value shifted = b.create<arith::SubFOp>(
                  bodyLoc, lseValues[split], lseMaxValue);
              Value shiftedBase2 =
                  b.create<arith::MulFOp>(bodyLoc, shifted, log2e);
              weights[split] =
                  b.create<math::Exp2Op>(bodyLoc, shiftedBase2);
              sumexp =
                  b.create<arith::AddFOp>(bodyLoc, sumexp, weights[split]);
            }
          } else if (warpsPerRow == 1) {
            SmallVector<Type> scalarTypes(splitCount + 2, b.getF32Type());
            auto scalarState = b.create<scf::IfOp>(
                bodyLoc, scalarTypes, laneZero, /*withElseRegion=*/true);
            OpBuilder leader = scalarState.getThenBodyBuilder();
            SmallVector<Value> lseValues(splitCount);
            Value leaderMax = negativeInfinity;
            for (int64_t split = 0; split < splitCount; ++split) {
              lseValues[split] = leader.create<memref::LoadOp>(
                  bodyLoc, entry->getArgument(splitCount + split),
                  ValueRange{row});
              leaderMax = leader.create<arith::MaxNumFOp>(
                  bodyLoc, leaderMax, lseValues[split]);
            }
            SmallVector<Value> leaderWeights(splitCount);
            Value leaderSum = zero;
            for (int64_t split = 0; split < splitCount; ++split) {
              Value shifted = leader.create<arith::SubFOp>(
                  bodyLoc, lseValues[split], leaderMax);
              Value shiftedBase2 = leader.create<arith::MulFOp>(
                  bodyLoc, shifted, log2e);
              leaderWeights[split] = leader.create<math::Exp2Op>(
                  bodyLoc, shiftedBase2);
              leaderSum = leader.create<arith::AddFOp>(
                  bodyLoc, leaderSum, leaderWeights[split]);
            }
            SmallVector<Value> leaderResults{leaderMax};
            llvm::append_range(leaderResults, leaderWeights);
            leaderResults.push_back(leaderSum);
            leader.create<scf::YieldOp>(bodyLoc, leaderResults);
            OpBuilder nonLeader = scalarState.getElseBodyBuilder();
            SmallVector<Value> identityResults{negativeInfinity};
            for (int64_t split = 0; split < splitCount; ++split)
              identityResults.push_back(zero);
            identityResults.push_back(zero);
            nonLeader.create<scf::YieldOp>(bodyLoc, identityResults);

            Value sourceLane = b.create<arith::ConstantIntOp>(
                bodyLoc, 0, 32);
            Value width = b.create<arith::ConstantIntOp>(bodyLoc, 32, 32);
            auto broadcast = [&](Value value) {
              return b.create<gpu::ShuffleOp>(
                                bodyLoc, value, sourceLane, width,
                                gpu::ShuffleMode::IDX)
                  .getShuffleResult();
            };
            lseMaxValue = broadcast(scalarState.getResult(0));
            for (int64_t split = 0; split < splitCount; ++split)
              weights[split] = broadcast(scalarState.getResult(split + 1));
            sumexp = broadcast(scalarState.getResult(splitCount + 1));
          } else {
            Value rowLeader = b.create<arith::CmpIOp>(
                bodyLoc, arith::CmpIPredicate::eq, rowLane, c0);
            b.create<scf::IfOp>(
                bodyLoc, rowLeader, [&](OpBuilder &leader, Location leaderLoc) {
                  SmallVector<Value> lseValues(splitCount);
                  Value leaderMax = negativeInfinity;
                  for (int64_t split = 0; split < splitCount; ++split) {
                    lseValues[split] = leader.create<memref::LoadOp>(
                        leaderLoc, entry->getArgument(splitCount + split),
                        ValueRange{row});
                    leaderMax = leader.create<arith::MaxNumFOp>(
                        leaderLoc, leaderMax, lseValues[split]);
                  }
                  leader.create<memref::StoreOp>(
                      leaderLoc, leaderMax, scalarScratch,
                      ValueRange{leader.create<arith::ConstantIndexOp>(leaderLoc,
                                                                       0)});
                  Value leaderSum = zero;
                  for (int64_t split = 0; split < splitCount; ++split) {
                    Value shifted = leader.create<arith::SubFOp>(
                        leaderLoc, lseValues[split], leaderMax);
                    Value shiftedBase2 = leader.create<arith::MulFOp>(
                        leaderLoc, shifted, log2e);
                    Value weight =
                        leader.create<math::Exp2Op>(leaderLoc, shiftedBase2);
                    leader.create<memref::StoreOp>(
                        leaderLoc, weight, scalarScratch,
                        ValueRange{leader.create<arith::ConstantIndexOp>(
                            leaderLoc, split + 1)});
                    leaderSum = leader.create<arith::AddFOp>(
                        leaderLoc, leaderSum, weight);
                  }
                  leader.create<memref::StoreOp>(
                      leaderLoc, leaderSum, scalarScratch,
                      ValueRange{leader.create<arith::ConstantIndexOp>(
                          leaderLoc, splitCount + 1)});
                  leader.create<scf::YieldOp>(leaderLoc);
                });
            b.create<gpu::BarrierOp>(bodyLoc);
            lseMaxValue = b.create<memref::LoadOp>(
                bodyLoc, scalarScratch,
                ValueRange{b.create<arith::ConstantIndexOp>(bodyLoc, 0)});
            for (int64_t split = 0; split < splitCount; ++split)
              weights[split] = b.create<memref::LoadOp>(
                  bodyLoc, scalarScratch,
                  ValueRange{b.create<arith::ConstantIndexOp>(bodyLoc,
                                                               split + 1)});
            sumexp = b.create<memref::LoadOp>(
                bodyLoc, scalarScratch,
                ValueRange{b.create<arith::ConstantIndexOp>(bodyLoc,
                                                             splitCount + 1)});
          }

          int64_t packetCount = D / vectorWidth;
          int64_t lanesPerRow = warpsPerRow * 32;
          int64_t dIterations =
              (packetCount + lanesPerRow - 1) / lanesPerRow;
          for (int64_t iteration = 0; iteration < dIterations; ++iteration) {
            Value delta = b.create<arith::ConstantIndexOp>(bodyLoc,
                                                            iteration * lanesPerRow);
            Value packet = b.create<arith::AddIOp>(bodyLoc, rowLane, delta);
            Value d = packet;
            if (vectorWidth != 1)
              d = b.create<arith::MulIOp>(
                  bodyLoc, packet,
                  b.create<arith::ConstantIndexOp>(bodyLoc, vectorWidth));
            Value dValid = b.create<arith::CmpIOp>(
                bodyLoc, arith::CmpIPredicate::ult, d, cD);
            b.create<scf::IfOp>(
                bodyLoc, dValid, [&](OpBuilder &db, Location dLoc) {
                  Value outputOffset =
                      db.create<arith::MulIOp>(dLoc, row, cD);
                  outputOffset =
                      db.create<arith::AddIOp>(dLoc, outputOffset, d);
                  Value allMasked = db.create<arith::CmpFOp>(
                      dLoc, arith::CmpFPredicate::OEQ, lseMaxValue,
                      negativeInfinity);
                  if (vectorWidth == 1) {
                    Value numerator = zero;
                    for (int64_t split = 0; split < splitCount; ++split) {
                      Value partial = db.create<memref::LoadOp>(
                          dLoc, entry->getArgument(split),
                          ValueRange{outputOffset});
                      Value weighted = db.create<arith::MulFOp>(
                          dLoc, partial, weights[split]);
                      numerator = db.create<arith::AddFOp>(dLoc, numerator,
                                                           weighted);
                    }
                    Value quotient =
                        db.create<arith::DivFOp>(dLoc, numerator, sumexp);
                    Value result = db.create<arith::SelectOp>(
                        dLoc, allMasked, zero, quotient);
                    db.create<memref::StoreOp>(
                        dLoc, result, entry->getArgument(2 * splitCount),
                        ValueRange{outputOffset});
                  } else {
                    auto packetType =
                        VectorType::get({vectorWidth}, db.getF32Type());
                    Value zeroPacket =
                        db.create<vector::SplatOp>(dLoc, packetType, zero);
                    Value numerator = zeroPacket;
                    for (int64_t split = 0; split < splitCount; ++split) {
                      Value partial = db.create<vector::LoadOp>(
                          dLoc, packetType, entry->getArgument(split),
                          ValueRange{outputOffset});
                      Value weightPacket = db.create<vector::SplatOp>(
                          dLoc, packetType, weights[split]);
                      Value weighted = db.create<arith::MulFOp>(
                          dLoc, partial, weightPacket);
                      numerator = db.create<arith::AddFOp>(dLoc, numerator,
                                                           weighted);
                    }
                    Value denominator = db.create<vector::SplatOp>(
                        dLoc, packetType, sumexp);
                    Value quotient = db.create<arith::DivFOp>(
                        dLoc, numerator, denominator);
                    auto maskType =
                        VectorType::get({vectorWidth}, db.getI1Type());
                    Value maskPacket = db.create<vector::SplatOp>(
                        dLoc, maskType, allMasked);
                    Value result = db.create<arith::SelectOp>(
                        dLoc, maskPacket, zeroPacket, quotient);
                    db.create<vector::StoreOp>(
                        dLoc, result, entry->getArgument(2 * splitCount),
                        ValueRange{outputOffset});
                  }
                  db.create<scf::YieldOp>(dLoc);
                });
          }

          Value rowLeader = b.create<arith::CmpIOp>(
              bodyLoc, arith::CmpIPredicate::eq, rowLane, c0);
          b.create<scf::IfOp>(
              bodyLoc, rowLeader, [&](OpBuilder &lb, Location lseLoc) {
                Value logSum = lb.create<math::Log2Op>(lseLoc, sumexp);
                Value naturalLog =
                    lb.create<arith::DivFOp>(lseLoc, logSum, log2e);
                Value lseValue = lb.create<arith::AddFOp>(
                    lseLoc, lseMaxValue, naturalLog);
                Value allMasked = lb.create<arith::CmpFOp>(
                    lseLoc, arith::CmpFPredicate::OEQ, lseMaxValue,
                    negativeInfinity);
                lseValue = lb.create<arith::SelectOp>(
                    lseLoc, allMasked, negativeInfinity, lseValue);
                lb.create<memref::StoreOp>(
                    lseLoc, lseValue,
                    entry->getArgument(2 * splitCount + 1), ValueRange{row});
                lb.create<scf::YieldOp>(lseLoc);
              });
          b.create<scf::YieldOp>(bodyLoc);
        });

    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct SplitKVarargsMatch {
  func::FuncOp fn;
  ConcatOp outConcat;
  ConcatOp lseConcat;
  ReduceOp lseMax;
  MapUnaryOp weights;
  ReduceOp weightSum;
  ReduceOp numeratorSum;
  int64_t B = 0;
  int64_t H = 0;
  int64_t M = 0;
  int64_t D = 0;

  explicit SplitKVarargsMatch(func::FuncOp fn) : fn(fn) {}

  LogicalResult match() {
    auto type = fn.getFunctionType();
    if (!fn.getBody().hasOneBlock() || type.getNumInputs() != 8 ||
        type.getNumResults() != 2)
      return failure();
    Block &entry = fn.getBody().front();

    std::array<RankedTensorType, 4> outTypes;
    std::array<RankedTensorType, 4> lseTypes;
    for (unsigned i = 0; i < 4; ++i) {
      outTypes[i] = dyn_cast<RankedTensorType>(type.getInput(i));
      lseTypes[i] = dyn_cast<RankedTensorType>(type.getInput(4 + i));
      if (!outTypes[i] || !lseTypes[i] || !outTypes[i].hasStaticShape() ||
          !lseTypes[i].hasStaticShape() || outTypes[i].getRank() != 5 ||
          lseTypes[i].getRank() != 4 ||
          !isa<Float32Type>(outTypes[i].getElementType()) ||
          !isa<Float32Type>(lseTypes[i].getElementType()))
        return failure();
      if (outTypes[i] != outTypes[0] || lseTypes[i] != lseTypes[0])
        return failure();
    }

    B = outTypes[0].getDimSize(0);
    int64_t G = outTypes[0].getDimSize(1);
    H = outTypes[0].getDimSize(2);
    M = outTypes[0].getDimSize(3);
    D = outTypes[0].getDimSize(4);
    if (G != 1 || B <= 0 || H <= 1 || M <= 0 || D != 64 ||
        lseTypes[0].getShape() != ArrayRef<int64_t>({B, G, H, M}))
      return failure();

    SmallVector<ConcatOp> concats;
    SmallVector<ReduceOp> reductions;
    for (Operation &op : entry.without_terminator()) {
      if (auto concat = dyn_cast<ConcatOp>(&op))
        concats.push_back(concat);
      if (auto reduce = dyn_cast<ReduceOp>(&op))
        reductions.push_back(reduce);
      if (isa<LoopOp>(&op))
        return failure();
    }
    if (concats.size() != 2 || reductions.size() != 3)
      return failure();
    for (ConcatOp concat : concats) {
      auto resultType = cast<RankedTensorType>(concat.getResult().getType());
      if (concat.getAxis() != 1 || concat.getInputs().size() != 4)
        return failure();
      if (resultType.getRank() == 5)
        outConcat = concat;
      else if (resultType.getRank() == 4)
        lseConcat = concat;
    }
    if (!outConcat || !lseConcat ||
        !areEntryArguments(outConcat.getInputs(), entry, 0) ||
        !areEntryArguments(lseConcat.getInputs(), entry, 4))
      return failure();
    auto packedOut = cast<RankedTensorType>(outConcat.getResult().getType());
    auto packedLse = cast<RankedTensorType>(lseConcat.getResult().getType());
    if (packedOut.getShape() != ArrayRef<int64_t>({B, 4, H, M, D}) ||
        packedLse.getShape() != ArrayRef<int64_t>({B, 4, H, M}))
      return failure();

    for (ReduceOp reduce : reductions) {
      if (!hasAxis(reduce, 1))
        return failure();
      if (reduce.getFn() == "max" &&
          reduce.getInput() == lseConcat.getResult())
        lseMax = reduce;
    }
    if (!lseMax)
      return failure();

    // Stable weights must be exp2((LSE - max) * scalar).  This consumes the
    // canonical packed LSE and its axis-1 maximum directly.
    for (auto unary : entry.getOps<MapUnaryOp>()) {
      if (unary.getFn() != "exp2")
        continue;
      auto scale = unary.getInput().getDefiningOp<MapBinaryOp>();
      if (!scale || scale.getFn() != "mul")
        continue;
      MapBinaryOp shifted = scale.getLhs().getDefiningOp<MapBinaryOp>();
      if (!shifted)
        shifted = scale.getRhs().getDefiningOp<MapBinaryOp>();
      if (!shifted || shifted.getFn() != "sub" ||
          shifted.getLhs() != lseConcat.getResult() ||
          shifted.getRhs() != lseMax.getResult())
        continue;
      weights = unary;
      break;
    }
    if (!weights)
      return failure();

    MapBinaryOp weightedOutputs;
    for (ReduceOp reduce : reductions) {
      if (reduce.getFn() != "sum")
        continue;
      if (reduce.getInput() == weights.getResult())
        weightSum = reduce;
      auto product = reduce.getInput().getDefiningOp<MapBinaryOp>();
      if (!product || product.getFn() != "mul")
        continue;
      if (product.getLhs() == outConcat.getResult() ||
          product.getRhs() == outConcat.getResult()) {
        weightedOutputs = product;
        numeratorSum = reduce;
      }
    }
    if (!weightSum || !weightedOutputs || !numeratorSum)
      return failure();

    auto ret = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!ret || ret.getNumOperands() != 2)
      return failure();
    auto outTranspose = ret.getOperand(0).getDefiningOp<TransposeOp>();
    if (!isPermutation(outTranspose, {0, 3, 1, 2, 4}) ||
        !ret.getOperand(1).getDefiningOp<SelectOp>())
      return failure();

    auto outResult = dyn_cast<RankedTensorType>(type.getResult(0));
    auto lseResult = dyn_cast<RankedTensorType>(type.getResult(1));
    if (!outResult || !lseResult || !outResult.hasStaticShape() ||
        !lseResult.hasStaticShape() ||
        outResult.getShape() != ArrayRef<int64_t>({B, M, 1, H, D}) ||
        lseResult.getShape() != ArrayRef<int64_t>({B, 1, H, M}) ||
        !isa<Float32Type>(outResult.getElementType()) ||
        !isa<Float32Type>(lseResult.getElementType()))
      return failure();
    return success();
  }

  LogicalResult rewrite(int64_t rowsPerBlock, int64_t warpsPerRow,
                        int64_t vectorWidth, bool broadcastScalars) {
    if (rowsPerBlock != 1 && rowsPerBlock != 2 && rowsPerBlock != 4 &&
        rowsPerBlock != 8)
      return fn.emitError("rows-per-block must be one of 1, 2, 4 or 8");
    if (warpsPerRow != 1)
      return fn.emitError(
          "runtime-stride SplitK requires warps-per-row=1 without a typed contiguous-D proof");
    if (vectorWidth != 1)
      return fn.emitError(
          "runtime-stride SplitK requires vector-width=1 without a contiguous-D proof");
    if (broadcastScalars)
      return fn.emitError(
          "runtime-stride SplitK requires warp-broadcast proof in the typed ABI");

    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder builder(ctx);
    Type f32 = builder.getF32Type();
    Type i32 = builder.getI32Type();

    // Exact runtime ABI after Triton's five constexpr parameters specialize
    // away: 8 partial pointers, Out/LSE destination pointers, then 45 i32
    // strides in the source's unrolled parameter order.
    SmallVector<Type> abiTypes;
    auto oldType = fn.getFunctionType();
    for (unsigned i = 0; i < 8; ++i) {
      auto tensorType = cast<RankedTensorType>(oldType.getInput(i));
      abiTypes.push_back(
          MemRefType::get({elementCount(tensorType)}, f32));
    }
    auto outResult = cast<RankedTensorType>(oldType.getResult(0));
    auto lseResult = cast<RankedTensorType>(oldType.getResult(1));
    abiTypes.push_back(MemRefType::get({elementCount(outResult)}, f32));
    abiTypes.push_back(MemRefType::get({elementCount(lseResult)}, f32));
    abiTypes.append(45, i32);
    auto newType = FunctionType::get(ctx, abiTypes, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    builder.setInsertionPoint(fn);
    auto gpuFn = builder.create<func::FuncOp>(loc, name, newType);
    gpuFn->setAttr("intentir.fast_path",
                   builder.getStringAttr("cuda.xformers.splitk-varargs.v1"));
    gpuFn->setAttr("intentir.runtime_abi_args",
                   builder.getI64IntegerAttr(55));
    Block *entry = gpuFn.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value cGridX = builder.create<arith::ConstantIndexOp>(
        loc, (M + rowsPerBlock - 1) / rowsPerBlock);
    Value cGridY = builder.create<arith::ConstantIndexOp>(loc, B * H);
    Value cBlock = builder.create<arith::ConstantIndexOp>(
        loc, rowsPerBlock * 32);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = builder.create<gpu::LaunchOp>(loc, cGridX, cGridY, c1,
                                                 cBlock, c1, c1);
    builder.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value by = launch.getBlockIds().y;
    Value tx = launch.getThreadIds().x;
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    // gpu-kernel-outlining orders captured values by first use.  Seed one
    // side-effect-free use of the host arguments in the exact source ABI order
    // (10 pointers, then 45 i32 strides).  Outlining therefore preserves the
    // external ABI order; its following canonicalize pass erases these unused
    // seed operations, so they cost nothing in PTX.
    for (unsigned index = 0; index < 10; ++index)
      (void)builder.create<memref::DimOp>(loc, entry->getArgument(index), c0);
    Value zeroI32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    for (unsigned index = 10; index < 55; ++index)
      (void)builder.create<arith::AddIOp>(loc, entry->getArgument(index),
                                          zeroI32);
    Value c32 = builder.create<arith::ConstantIndexOp>(loc, 32);
    Value cRows = builder.create<arith::ConstantIndexOp>(loc, rowsPerBlock);
    Value cH = builder.create<arith::ConstantIndexOp>(loc, H);
    Value cM = builder.create<arith::ConstantIndexOp>(loc, M);
    Value warp = builder.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = builder.create<arith::RemUIOp>(loc, tx, c32);
    Value rowBase = builder.create<arith::MulIOp>(loc, bx, cRows);
    Value m = builder.create<arith::AddIOp>(loc, rowBase, warp);
    Value z = builder.create<arith::DivUIOp>(loc, by, cH);
    Value h = builder.create<arith::RemUIOp>(loc, by, cH);
    Value g = c0; // captured specialization G=1; strides remain in the ABI.
    Value rowValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, m, cM);

    builder.create<scf::IfOp>(
        loc, rowValid, [&](OpBuilder &b, Location bodyLoc) {
          auto strideIndex = [&](unsigned abiIndex) -> Value {
            return b.create<arith::IndexCastOp>(
                bodyLoc, b.getIndexType(), entry->getArgument(abiIndex));
          };
          auto addScaled = [&](Value offset, Value coordinate,
                               unsigned abiIndex) -> Value {
            Value product = b.create<arith::MulIOp>(
                bodyLoc, coordinate, strideIndex(abiIndex));
            return b.create<arith::AddIOp>(bodyLoc, offset, product);
          };
          auto lseOffset = [&](unsigned split) -> Value {
            Value offset = c0;
            offset = addScaled(offset, z, 30 + split);
            offset = addScaled(offset, g, 34 + split);
            offset = addScaled(offset, h, 38 + split);
            offset = addScaled(offset, m, 42 + split);
            return offset;
          };
          auto outOffset = [&](unsigned split, Value d) -> Value {
            Value offset = c0;
            offset = addScaled(offset, z, 10 + split);
            offset = addScaled(offset, g, 14 + split);
            offset = addScaled(offset, h, 18 + split);
            offset = addScaled(offset, m, 22 + split);
            offset = addScaled(offset, d, 26 + split);
            return offset;
          };

          Value ninf = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(
                           -std::numeric_limits<float>::infinity()));
          Value zero = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(0.0));
          Value log2e = b.create<arith::ConstantOp>(
              bodyLoc, b.getF32FloatAttr(1.44269504));

          std::array<Value, 4> lseValues;
          Value lseMaxValue = ninf;
          for (unsigned split = 0; split < 4; ++split) {
            lseValues[split] = b.create<memref::LoadOp>(
                bodyLoc, entry->getArgument(4 + split),
                ValueRange{lseOffset(split)});
            lseMaxValue = b.create<arith::MaxNumFOp>(
                bodyLoc, lseMaxValue, lseValues[split]);
          }

          std::array<Value, 4> splitWeights;
          Value sumexp = zero;
          for (unsigned split = 0; split < 4; ++split) {
            Value shifted = b.create<arith::SubFOp>(
                bodyLoc, lseValues[split], lseMaxValue);
            Value shiftedBase2 = b.create<arith::MulFOp>(
                bodyLoc, shifted, log2e);
            splitWeights[split] =
                b.create<math::Exp2Op>(bodyLoc, shiftedBase2);
            sumexp = b.create<arith::AddFOp>(bodyLoc, sumexp,
                                             splitWeights[split]);
          }

          int64_t dIterations = (D + 31) / 32;
          for (int64_t iteration = 0; iteration < dIterations; ++iteration) {
            Value delta = b.create<arith::ConstantIndexOp>(
                bodyLoc, iteration * 32);
            Value d = b.create<arith::AddIOp>(bodyLoc, lane, delta);
            Value numerator = zero;
            for (unsigned split = 0; split < 4; ++split) {
              Value partial = b.create<memref::LoadOp>(
                  bodyLoc, entry->getArgument(split),
                  ValueRange{outOffset(split, d)});
              Value weighted = b.create<arith::MulFOp>(
                  bodyLoc, partial, splitWeights[split]);
              numerator =
                  b.create<arith::AddFOp>(bodyLoc, numerator, weighted);
            }
            Value quotient =
                b.create<arith::DivFOp>(bodyLoc, numerator, sumexp);
            Value allMasked = b.create<arith::CmpFOp>(
                bodyLoc, arith::CmpFPredicate::OEQ, lseMaxValue, ninf);
            Value result = b.create<arith::SelectOp>(
                bodyLoc, allMasked, zero, quotient);

            Value destination = c0;
            destination = addScaled(destination, z, 46);
            destination = addScaled(destination, g, 47);
            destination = addScaled(destination, h, 48);
            destination = addScaled(destination, m, 49);
            destination = addScaled(destination, d, 50);
            b.create<memref::StoreOp>(bodyLoc, result, entry->getArgument(8),
                                      ValueRange{destination});
          }

          Value lseDestination = c0;
          lseDestination = addScaled(lseDestination, z, 51);
          lseDestination = addScaled(lseDestination, g, 52);
          lseDestination = addScaled(lseDestination, h, 53);
          lseDestination = addScaled(lseDestination, m, 54);
          Value laneZero = b.create<arith::CmpIOp>(
              bodyLoc, arith::CmpIPredicate::eq, lane, c0);
          b.create<scf::IfOp>(
              bodyLoc, laneZero, [&](OpBuilder &lb, Location lseLoc) {
                Value logSum = lb.create<math::LogOp>(lseLoc, sumexp);
                Value lseValue = lb.create<arith::AddFOp>(
                    lseLoc, lseMaxValue, logSum);
                Value allMasked = lb.create<arith::CmpFOp>(
                    lseLoc, arith::CmpFPredicate::OEQ, lseMaxValue, ninf);
                lseValue = lb.create<arith::SelectOp>(
                    lseLoc, allMasked, ninf, lseValue);
                lb.create<memref::StoreOp>(
                    lseLoc, lseValue, entry->getArgument(9),
                    ValueRange{lseDestination});
                lb.create<scf::YieldOp>(lseLoc);
              });
          b.create<scf::YieldOp>(bodyLoc);
        });

    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentSplitKReduceVarargsToGPUPass
    : ::intent::impl::ConvertIntentSplitKReduceVarargsToGPUBase<
          ConvertIntentSplitKReduceVarargsToGPUPass> {
  using ConvertIntentSplitKReduceVarargsToGPUBase::
      ConvertIntentSplitKReduceVarargsToGPUBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<std::pair<func::FuncOp, bool>> targets;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      UnrolledSplitKVarargsMatch unrolled(fn);
      if (succeeded(unrolled.match())) {
        targets.emplace_back(fn, true);
        continue;
      }
      SplitKVarargsMatch probe(fn);
      if (succeeded(probe.match()))
        targets.emplace_back(fn, false);
    }
    for (auto [fn, isUnrolled] : targets) {
      LogicalResult result = failure();
      if (isUnrolled) {
        UnrolledSplitKVarargsMatch lowering(fn);
        if (succeeded(lowering.match()))
          result = lowering.rewrite(rowsPerBlock, warpsPerRow, vectorWidth,
                                    warpBroadcastScalars);
      } else {
        SplitKVarargsMatch lowering(fn);
        if (succeeded(lowering.match()))
          result = lowering.rewrite(rowsPerBlock, warpsPerRow, vectorWidth,
                                    warpBroadcastScalars);
      }
      if (failed(result)) {
        fn.emitError("split-K varargs lowering failed after structural match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
