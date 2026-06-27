//===- ConvertIntentSplitKReduceBackwardToGPU.cpp - splitK reduce bwd -----===//
//
// Lowers the xformers split-K attention REDUCE BACKWARD
// (`_splitK_reduce_varargs_backward`) -- the GRADIENT of the online
// log-sum-exp merge that combines the split-K partial outputs -- to one
// block-cooperative `gpu.launch` (NO matmul). This is the M9b counterpart of
// `convert-intent-splitk-reduce-to-gpu` (the forward merge).
//
// The forward merge takes per-split partials (Out_splitK[S,Mq,D],
// LSE_splitK[S,Mq,1]) and produces the merged (Out[Mq,D], LSE[Mq,1]). The
// backward redistributes the merged-output gradients (DOut[Mq,D], DLSE[Mq,1])
// back to each split chunk's gradient (dOut_splitK[S,Mq,D],
// dLSE_splitK[S,Mq,1]). Per (z,g,h,m) output row, with two passes over the
// split axis:
//
//   pass 1:  lse_max = max_s LSE_splitK[s]                 (over the split axis)
//   denom   = exp(LSE - lse_max)                           (== sumexp)
//   pass 2 (per split s):
//     w_s            = exp(LSE_splitK[s] - lse_max) / denom
//     dOut_splitK[s] = w_s * DOut                          (elementwise over D)
//     corr           = sum_d ((Out_splitK[s] - Out) * w_s * DOut)  (reduce D)
//     dLSE_splitK[s] = w_s * DLSE + corr
//
// (`w_s == exp(LSE_splitK[s] - lse_max)/exp(LSE - lse_max)` matches the real
// kernel's `dattn_dattn_i`; the `corr` is its `tl.sum(dattn_dlse_i * dattn)`.)
//
// The varargs list of per-chunk tensors is normalized to the packed
// [S,Mq,D]/[S,Mq,1] form upstream (the SAME contract as the forward reduce), so
// `_splitK_reduce_backward` and `_splitK_reduce_varargs_backward` lift to the
// SAME IntentIR (a RowList of S split chunks).
//
// IntentIR DAG (straight-line, NO loop, NO contraction): a `reduce{max,axis=0}`
// (lse_max), a `reduce{sum,axis=2}` (corr over the head dim), and broadcast
// map_binary/map_unary for the rest. grid = (Mq,1,1) (one block per output
// row), block = (D,1,1) (one thread per head-dim element). lse_max/denom depend
// only on the per-row LSE scalars (every thread recomputes them); the `corr`
// head-dim reduction is a `gpu.all_reduce` ADD across the block's D threads
// (inside the split-axis scf.for -- the G2 nested-all_reduce seam, proven in
// M6/M8). dLSE_splitK[s] is written once by thread 0.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

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
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTSPLITKREDUCEBACKWARDTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct SplitKReduceBwdLowering {
  func::FuncOp fn;

  // populated by match()
  int64_t S = 0, Mq = 0, D = 0;
  Type f32;

  explicit SplitKReduceBwdLowering(func::FuncOp fn) : fn(fn) {}

  static bool axesAre(ReduceOp r, ArrayRef<int64_t> want) {
    auto axes = r.getAxes();
    if (axes.size() != want.size())
      return false;
    for (size_t i = 0; i < want.size(); ++i)
      if (cast<IntegerAttr>(axes[i]).getInt() != want[i])
        return false;
    return true;
  }

  LogicalResult match() {
    auto fty = fn.getFunctionType();
    // Six inputs in the canonical order (mirrors the real kernel's parameter
    // order Out_splitK, LSE_splitK, ..., Out, LSE, DOut, DLSE), two results.
    if (fty.getNumInputs() != 6 || fty.getNumResults() != 2 ||
        !fn.getBody().hasOneBlock())
      return failure();

    auto shapeOf = [](Type t) -> RankedTensorType {
      return dyn_cast<RankedTensorType>(t);
    };
    auto isF32Static = [](RankedTensorType t, int64_t rank) {
      return t && t.hasStaticShape() && t.getRank() == rank &&
             isa<Float32Type>(t.getElementType());
    };

    auto osk = shapeOf(fty.getInput(0)); // Out_splitK [S,Mq,D]
    auto lsk = shapeOf(fty.getInput(1)); // LSE_splitK [S,Mq,1]
    auto out = shapeOf(fty.getInput(2)); // Out        [Mq,D]
    auto lse = shapeOf(fty.getInput(3)); // LSE        [Mq,1]
    auto dout = shapeOf(fty.getInput(4)); // DOut      [Mq,D]
    auto dlse = shapeOf(fty.getInput(5)); // DLSE      [Mq,1]
    if (!isF32Static(osk, 3) || !isF32Static(lsk, 3) || !isF32Static(out, 2) ||
        !isF32Static(lse, 2) || !isF32Static(dout, 2) || !isF32Static(dlse, 2))
      return failure();

    S = osk.getDimSize(0);
    Mq = osk.getDimSize(1);
    D = osk.getDimSize(2);
    f32 = osk.getElementType();
    if (lsk.getDimSize(0) != S || lsk.getDimSize(1) != Mq ||
        lsk.getDimSize(2) != 1)
      return failure();
    if (out.getDimSize(0) != Mq || out.getDimSize(1) != D)
      return failure();
    if (lse.getDimSize(0) != Mq || lse.getDimSize(1) != 1)
      return failure();
    if (dout.getDimSize(0) != Mq || dout.getDimSize(1) != D)
      return failure();
    if (dlse.getDimSize(0) != Mq || dlse.getDimSize(1) != 1)
      return failure();

    // Results: dOut_splitK [S,Mq,D], dLSE_splitK [S,Mq,1].
    auto r0 = shapeOf(fty.getResult(0));
    auto r1 = shapeOf(fty.getResult(1));
    if (!isF32Static(r0, 3) || !isF32Static(r1, 3))
      return failure();
    if (r0.getDimSize(0) != S || r0.getDimSize(1) != Mq || r0.getDimSize(2) != D)
      return failure();
    if (r1.getDimSize(0) != S || r1.getDimSize(1) != Mq || r1.getDimSize(2) != 1)
      return failure();

    // Body: straight-line map/reduce DAG (NO loop, NO contraction); exactly two
    // reduces -- max over the split axis (lse_max) and sum over the head dim
    // (the corr term). Structural signature, NOT a kernel-name match.
    Block &body = fn.getBody().front();
    ReduceOp maxRed, sumRed;
    for (Operation &op : body.without_terminator()) {
      if (auto r = dyn_cast<ReduceOp>(&op)) {
        if (r.getFn() == "max" && axesAre(r, {0})) {
          if (maxRed)
            return failure();
          maxRed = r;
        } else if (r.getFn() == "sum" && axesAre(r, {2})) {
          if (sumRed)
            return failure();
          sumRed = r;
        } else {
          return failure();
        }
      } else if (!isa<arith::ConstantOp, MapBinaryOp, MapUnaryOp, SelectOp>(
                     &op)) {
        return failure();
      }
    }
    if (!maxRed || !sumRed)
      return failure();
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: 6 input memrefs + appended destination-passing outputs
    // dOut_splitK[S,Mq,D], dLSE_splitK[S,Mq,1].
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      auto t = cast<RankedTensorType>(at);
      inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    inTys.push_back(MemRefType::get({S, Mq, D}, f32)); // dOut_splitK
    inTys.push_back(MemRefType::get({S, Mq, 1}, f32)); // dLSE_splitK
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value oskMem = entry->getArgument(0);
    Value lskMem = entry->getArgument(1);
    Value outMem = entry->getArgument(2);
    Value lseMem = entry->getArgument(3);
    Value doutMem = entry->getArgument(4);
    Value dlseMem = entry->getArgument(5);
    Value dOutSkMem = entry->getArgument(6);
    Value dLseSkMem = entry->getArgument(7);

    Value cMq = b.create<arith::ConstantIndexOp>(loc, Mq);
    Value cD = b.create<arith::ConstantIndexOp>(loc, D);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    // grid = (Mq,1,1), block = (D,1,1).
    auto launch = b.create<gpu::LaunchOp>(loc, cMq, c1, c1, cD, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value m = launch.getBlockIds().x;  // output row
    Value d = launch.getThreadIds().x; // head-dim element
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cS = b.create<arith::ConstantIndexOp>(loc, S);
    Value ninf = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

    // pass 1: lse_max = max_s LSE_splitK[s,m] (every thread recomputes it).
    auto maxLoop = b.create<scf::ForOp>(loc, c0, cS, c1, ValueRange{ninf});
    {
      OpBuilder lb(maxLoop.getBody(), maxLoop.getBody()->begin());
      Value s = maxLoop.getInductionVar();
      Value acc = maxLoop.getRegionIterArg(0);
      Value ls = lb.create<memref::LoadOp>(loc, lskMem, ValueRange{s, m, c0});
      Value mn = lb.create<arith::MaxNumFOp>(loc, acc, ls);
      lb.create<scf::YieldOp>(loc, ValueRange{mn});
    }
    Value lseMax = maxLoop.getResult(0);

    // per-thread merged values + denom = exp(LSE - lse_max) (== sumexp).
    Value out = b.create<memref::LoadOp>(loc, outMem, ValueRange{m, d});
    Value dout = b.create<memref::LoadOp>(loc, doutMem, ValueRange{m, d});
    Value lse = b.create<memref::LoadOp>(loc, lseMem, ValueRange{m, c0});
    Value dlse = b.create<memref::LoadOp>(loc, dlseMem, ValueRange{m, c0});
    Value lseDiff = b.create<arith::SubFOp>(loc, lse, lseMax);
    Value denom = b.create<math::ExpOp>(loc, lseDiff);

    Value isLead =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, d, c0);

    // pass 2: per split s, redistribute the gradients.
    auto sLoop = b.create<scf::ForOp>(loc, c0, cS, c1);
    {
      OpBuilder lb(sLoop.getBody(), sLoop.getBody()->begin());
      Value s = sLoop.getInductionVar();
      Value ls = lb.create<memref::LoadOp>(loc, lskMem, ValueRange{s, m, c0});
      Value wNum = lb.create<arith::SubFOp>(loc, ls, lseMax);
      Value wE = lb.create<math::ExpOp>(loc, wNum);
      Value w = lb.create<arith::DivFOp>(loc, wE, denom);
      Value outSk = lb.create<memref::LoadOp>(loc, oskMem, ValueRange{s, m, d});

      // dOut_splitK[s,m,d] = w * DOut.
      Value doutSk = lb.create<arith::MulFOp>(loc, w, dout);
      lb.create<memref::StoreOp>(loc, doutSk, dOutSkMem, ValueRange{s, m, d});

      // corr = sum_d ((Out_splitK - Out) * w * DOut) = sum_d ((outSk-out)*doutSk).
      Value diff = lb.create<arith::SubFOp>(loc, outSk, out);
      Value contrib = lb.create<arith::MulFOp>(loc, diff, doutSk);
      auto allRed = lb.create<gpu::AllReduceOp>(
          loc, contrib,
          gpu::AllReduceOperationAttr::get(ctx, gpu::AllReduceOperation::ADD),
          /*uniform=*/UnitAttr::get(ctx));
      Value corr = allRed.getResult();

      // dLSE_splitK[s,m,0] = w * DLSE + corr (single writer = thread 0).
      Value wdl = lb.create<arith::MulFOp>(loc, w, dlse);
      Value dlseSk = lb.create<arith::AddFOp>(loc, wdl, corr);
      lb.create<scf::IfOp>(loc, isLead, [&](OpBuilder &nb, Location nl) {
        nb.create<memref::StoreOp>(nl, dlseSk, dLseSkMem,
                                   ValueRange{s, m, c0});
        nb.create<scf::YieldOp>(nl);
      });
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentSplitKReduceBackwardToGPUPass
    : ::intent::impl::ConvertIntentSplitKReduceBackwardToGPUBase<
          ConvertIntentSplitKReduceBackwardToGPUPass> {
  using ConvertIntentSplitKReduceBackwardToGPUBase::
      ConvertIntentSplitKReduceBackwardToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      SplitKReduceBwdLowering probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      SplitKReduceBwdLowering low(fn);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("split-K reduce backward lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
