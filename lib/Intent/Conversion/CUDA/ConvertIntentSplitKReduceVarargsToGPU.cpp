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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
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

  LogicalResult rewrite(int64_t rowsPerBlock) {
    if (rowsPerBlock != 1 && rowsPerBlock != 2 && rowsPerBlock != 4 &&
        rowsPerBlock != 8)
      return fn.emitError("rows-per-block must be one of 1, 2, 4 or 8");

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
    SmallVector<func::FuncOp> targets;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      SplitKVarargsMatch probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      SplitKVarargsMatch lowering(fn);
      if (failed(lowering.match()) ||
          failed(lowering.rewrite(rowsPerBlock))) {
        fn.emitError("split-K varargs lowering failed after structural match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
