//===- ConvertIntentSplitKReduceToGPU.cpp - split-K reduce -> gpu ---------===//
//
// Lowers the xformers split-K attention REDUCE (`_splitK_reduce` /
// `_splitK_reduce_varargs`) -- the online log-sum-exp merge of the split-K
// partial outputs, with NO matmul -- to one block-cooperative `gpu.launch`.
//
// The merge is expressed as ONE `intent.loop` scanning the split axis of a
// packed partial output `O[S,Mq,D]` and partial log-sum-exp `LSE[S,Mq,1]`
// (the varargs list of per-chunk tensors is normalized to this packed form
// upstream, so `_splitK_reduce` and `_splitK_reduce_varargs` lift to the SAME
// IntentIR). It carries the online state (`lse_max[Mq,1]`, `sumexp[Mq,1]`,
// `O_acc[Mq,D]`); each trip s does:
//
//   m_new  = max(lse_max, LSE_s)
//   corr   = exp(lse_max - m_new)        (0 on the first trip: -inf - finite)
//   w      = exp(LSE_s   - m_new)
//   sumexp = sumexp * corr + w
//   O_acc  = O_acc  * corr + w * O_s
//   lse_max = m_new
//
// epilogue: O = O_acc / sumexp (0 where sumexp == 0). This reconstructs the full
// attention output: w_s = exp(LSE_s - max) = l_s * exp(m_s - max), and since the
// partial O_s is already softmax(chunk)@V_chunk = (sum exp(s-m_s) v)/l_s, the
// weighted sum is sum_j exp(s_j - max) v_j / sum_j exp(s_j - max).
//
// Because lse_max/sumexp depend only on the per-row LSE scalars, every thread
// recomputes them redundantly -- NO shared memory, NO barriers. grid = (Mq,1,1)
// (one block per output row), block = (D,1,1) (one thread per head-dim element
// owning a single O_acc[d] register), scanning the split axis with a plain
// scf.for carrying (lse_max, sumexp, O_acc[d]) as f32 iter_args.
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
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTSPLITKREDUCETOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct SplitKReduceLowering {
  func::FuncOp fn;

  // populated by match()
  int64_t S = 0, Mq = 0, D = 0;
  unsigned oArgIdx = 0, lseArgIdx = 0;
  LoopOp loop;
  Type f32;

  explicit SplitKReduceLowering(func::FuncOp fn) : fn(fn) {}

  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 1 || !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    // Body: const init splats + exactly one intent.loop + a straight-line divide
    // epilogue (O = O_acc / sumexp) + return.
    LoopOp theLoop;
    for (Operation &op : body.without_terminator()) {
      if (auto lp = dyn_cast<LoopOp>(&op)) {
        if (theLoop)
          return failure();
        theLoop = lp;
      } else if (!isa<arith::ConstantOp, MapBinaryOp, MapUnaryOp, SelectOp>(
                     &op)) {
        return failure();
      }
    }
    if (!theLoop)
      return failure();
    loop = theLoop;

    // Two scan inputs (packed partial O[S,Mq,D], LSE[S,Mq,1]) scanned over the
    // split axis 0; three carries (lse_max[Mq,1], sumexp[Mq,1], O_acc[Mq,D]).
    if (loop.getScanInputs().size() != 2 || loop.getInits().size() != 3)
      return failure();
    auto oArg = dyn_cast<BlockArgument>(loop.getScanInputs()[0]);
    auto lseArg = dyn_cast<BlockArgument>(loop.getScanInputs()[1]);
    if (!oArg || !lseArg || oArg.getOwner() != &body ||
        lseArg.getOwner() != &body)
      return failure();
    oArgIdx = oArg.getArgNumber();
    lseArgIdx = lseArg.getArgNumber();

    auto oTy = dyn_cast<RankedTensorType>(oArg.getType());
    auto lseTy = dyn_cast<RankedTensorType>(lseArg.getType());
    if (!oTy || !lseTy || !oTy.hasStaticShape() || !lseTy.hasStaticShape())
      return failure();
    if (oTy.getRank() != 3 || lseTy.getRank() != 3)
      return failure();
    if (!isa<Float32Type>(oTy.getElementType()) ||
        !isa<Float32Type>(lseTy.getElementType()))
      return failure();
    S = oTy.getDimSize(0);
    Mq = oTy.getDimSize(1);
    D = oTy.getDimSize(2);
    if (lseTy.getDimSize(0) != S || lseTy.getDimSize(1) != Mq ||
        lseTy.getDimSize(2) != 1)
      return failure();
    f32 = oTy.getElementType();

    // The merge is a pure elementwise/broadcast online update: NO matmul, NO
    // row reduction in the loop body (shape/DAG signature, not a name match).
    Block &lbody = loop.getBody().front();
    for (Operation &op : lbody.without_terminator())
      if (isa<ContractionOp, ReduceOp>(&op))
        return failure();

    // Single result O[Mq,D] f32.
    auto resTy = dyn_cast<RankedTensorType>(fty.getResult(0));
    if (!resTy || !resTy.hasStaticShape() || resTy.getRank() != 2 ||
        resTy.getDimSize(0) != Mq || resTy.getDimSize(1) != D ||
        !isa<Float32Type>(resTy.getElementType()))
      return failure();
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: packed inputs O[S,Mq,D], LSE[S,Mq,1] (memrefs) +
    // destination-passing output O[Mq,D].
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      if (auto t = dyn_cast<RankedTensorType>(at))
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      else
        inTys.push_back(at);
    }
    inTys.push_back(MemRefType::get({Mq, D}, f32));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value oSk = entry->getArgument(oArgIdx);
    Value lseSk = entry->getArgument(lseArgIdx);
    Value oOut = entry->getArgument(fn.getFunctionType().getNumInputs());

    Value cMq = b.create<arith::ConstantIndexOp>(loc, Mq);
    Value cD = b.create<arith::ConstantIndexOp>(loc, D);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    // grid = (Mq, 1, 1), block = (D, 1, 1).
    auto launch = b.create<gpu::LaunchOp>(loc, cMq, c1, c1, cD, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value m = launch.getBlockIds().x;   // output row
    Value d = launch.getThreadIds().x;  // head-dim element
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cS = b.create<arith::ConstantIndexOp>(loc, S);
    Value ninf = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));

    // Online merge over the split axis: scf.for carrying (lse_max, sumexp,
    // O_acc[d]) as f32 iter_args.
    auto forOp = b.create<scf::ForOp>(loc, c0, cS, c1,
                                      ValueRange{ninf, zeroF, zeroF});
    {
      OpBuilder lb(forOp.getBody(), forOp.getBody()->begin());
      Value s = forOp.getInductionVar();
      Value lmax = forOp.getRegionIterArg(0);
      Value se = forOp.getRegionIterArg(1);
      Value oacc = forOp.getRegionIterArg(2);

      Value ls = lb.create<memref::LoadOp>(loc, lseSk, ValueRange{s, m, c0});
      Value os = lb.create<memref::LoadOp>(loc, oSk, ValueRange{s, m, d});

      Value mNew = lb.create<arith::MaxNumFOp>(loc, lmax, ls);
      Value dCorr = lb.create<arith::SubFOp>(loc, lmax, mNew);
      Value corr = lb.create<math::ExpOp>(loc, dCorr);
      Value dW = lb.create<arith::SubFOp>(loc, ls, mNew);
      Value w = lb.create<math::ExpOp>(loc, dW);

      Value seC = lb.create<arith::MulFOp>(loc, se, corr);
      Value seNew = lb.create<arith::AddFOp>(loc, seC, w);
      Value oaC = lb.create<arith::MulFOp>(loc, oacc, corr);
      Value wO = lb.create<arith::MulFOp>(loc, w, os);
      Value oaNew = lb.create<arith::AddFOp>(loc, oaC, wO);

      lb.create<scf::YieldOp>(loc, ValueRange{mNew, seNew, oaNew});
    }

    Value seFinal = forOp.getResult(1);
    Value oaFinal = forOp.getResult(2);
    Value div = b.create<arith::DivFOp>(loc, oaFinal, seFinal);
    // Guard sumexp == 0 (a fully-masked row) -> 0, matching the reference.
    Value isZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                           seFinal, zeroF);
    Value out = b.create<arith::SelectOp>(loc, isZero, zeroF, div);
    b.create<memref::StoreOp>(loc, out, oOut, ValueRange{m, d});

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentSplitKReduceToGPUPass
    : ::intent::impl::ConvertIntentSplitKReduceToGPUBase<
          ConvertIntentSplitKReduceToGPUPass> {
  using ConvertIntentSplitKReduceToGPUBase::
      ConvertIntentSplitKReduceToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      SplitKReduceLowering probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      SplitKReduceLowering low(fn);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("split-K reduce lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
