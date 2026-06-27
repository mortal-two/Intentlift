//===- ConvertIntentAttentionBackwardToGPU.cpp - flash bwd -> gpu ---------===//
//
// Lowers a func whose body is the FlashAttention BACKWARD as one `intent.loop`
// scanning the QUERY axis (scan inputs Q, dO, LSE, Delta; carries the per-key-row
// gradients dk[Sk,D], dv[Sk,D], init zeros) to ONE tiled tensor-core
// `gpu.launch`. Modeled structurally on ConvertIntentLoopContractionToGPU (M8
// flash fwd): the matmuls run on WMMA (the M7 tiled-K primitive), the
// recompute-P / dS elementwise transforms + the dk/dv accumulators live RESIDENT
// IN DYNAMIC SHARED MEMORY (the M8 cooperative-shared design).
//
// THE 5-CONTRACTION FUSION (structural match, NOT a kernel-name match):
//   recompute  P  = exp(scale*S - LSE_row)          (S below; LSE from fwd)
//   S   = Q@K^T          "qd,kd->qk"   recompute the scores            [BM,BN]
//   dV += P^T@dO         "qk,qd->kd"   reduce over query -> dv carry   [BN,D]
//   dP  = dO@V^T         "qd,kd->qk"                                   [BM,BN]
//   dS  = P*(dP - Delta_row)*scale     (elementwise)                   [BM,BN]
//   dK += dS^T@Q         "qk,qd->kd"   reduce over query -> dk carry   [BN,D]
//   dQ  = dS@K           "qk,kd->qd"   reduce over key, scatter DQ     [BM,D]
//
// SERIAL DQ (SEQUENCE_PARALLEL=false): one block per (b,h) owns the whole head.
// The schedule is an OUTER scf.for over key tiles (BN=64; dk/dv reset per key
// tile in shared, written to DK/DV after the inner loop) wrapping an INNER
// scf.for over query tiles (BM=64; the intent.loop's scan). dQ is read-modify-
// written into pre-zeroed global DQ deterministically across the outer key loop
// (the cross-program atomic_add mode + the serial==atomic gate are M9b).
//
// dV=P^T@dO and dK=dS^T@Q reduce over the QUERY axis, so their [q,*] operands are
// loaded TRANSPOSED on the WMMA A operand (AOp-transpose probed to lower through
// wmma.load.a...col.f16 on sm_120a). The S/dP elementwise + dS recompute are
// plain cooperative shared arith over the BM rows (NOT fragment-lane indexing).
//
// Per query tile (Sd = [BM,BN] f32 scratch, Pds = [BM,BN] f16 scratch, reused):
//   1. S = Q@Kj^T via WMMA -> Sd.
//   2. coop: P = (f16)exp(scale*Sd - LSE_row) -> Pds.
//   3. dV product = P^T@dO via WMMA -> Sd ; coop dv_sh += Sd.
//   4. dP = dO@Vj^T via WMMA -> Sd.
//   5. coop: dS = (f16)(P*(Sd - Delta_row)*scale) -> Pds  (P's last use).
//   6. dK product = dS^T@Q via WMMA -> Sd ; coop dk_sh += Sd.
//   7. dQ product = dS@Kj via WMMA -> Sd ; coop DQ[start_m] += Sd  (RMW global).
//
// Q/K/V/dO f16 (WMMA operands), LSE/Delta/scale + accumulators f32. Static
// shapes, BM=BN=64, D%16==0, Sq%64==0, Sk%64==0 (non-causal, no bias). Bare-ptr
// ABI; the lowered module goes to lower_memref_module_to_kernels with
// scf_to_cf=True. The pass appends three destination outputs dQ, dK, dV.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTATTENTIONBACKWARDTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Tile geometry. BM/BN = the block's query/key tile (64x64), 4 warps each owning
// a 16-row C strip. kWMMA is the WMMA k (16). D is the head dim.
static constexpr int64_t BM = 64, BN = 64, kWMMA = 16, kBlock = 128;

static int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

/// One flash-bwd func's tensor-core lowering. Reads the original (tensor) func
/// body (two zero splats + one intent.loop + return) and builds a new
/// memref+gpu.launch func with the same name, erasing the old.
struct FlashBwdLowering {
  func::FuncOp fn;

  // populated by match()
  int64_t Sq = 0, Sk = 0, D = 0;
  unsigned qArgIdx = 0, kArgIdx = 0, vArgIdx = 0, doArgIdx = 0, lseArgIdx = 0,
           deltaArgIdx = 0, scaleArgIdx = 0;
  LoopOp loop;
  Type f16, f32;

  explicit FlashBwdLowering(func::FuncOp fn) : fn(fn) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    // Two results (dk, dv = the loop carries); the pass appends dQ/dK/dV
    // destinations regardless.
    if (fty.getNumResults() != 2 || !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    // Body: zero splats (init carries) + exactly one intent.loop + return.
    LoopOp theLoop;
    for (Operation &op : body.without_terminator()) {
      if (auto lp = dyn_cast<LoopOp>(&op)) {
        if (theLoop)
          return failure();
        theLoop = lp;
      } else if (!isa<arith::ConstantOp>(&op)) {
        return failure();
      }
    }
    if (!theLoop)
      return failure();
    loop = theLoop;

    // The loop scans (Q, dO, LSE, Delta) along the query axis and carries
    // (dk[Sk,D], dv[Sk,D]).
    if (loop.getScanInputs().size() != 4 || loop.getInits().size() != 2)
      return failure();
    auto qArg = dyn_cast<BlockArgument>(loop.getScanInputs()[0]);
    auto doArg = dyn_cast<BlockArgument>(loop.getScanInputs()[1]);
    auto lseArg = dyn_cast<BlockArgument>(loop.getScanInputs()[2]);
    auto deltaArg = dyn_cast<BlockArgument>(loop.getScanInputs()[3]);
    if (!qArg || !doArg || !lseArg || !deltaArg)
      return failure();
    if (qArg.getOwner() != &body || doArg.getOwner() != &body ||
        lseArg.getOwner() != &body || deltaArg.getOwner() != &body)
      return failure();
    qArgIdx = qArg.getArgNumber();
    doArgIdx = doArg.getArgNumber();
    lseArgIdx = lseArg.getArgNumber();
    deltaArgIdx = deltaArg.getArgNumber();

    // Loop body: exactly five intent.contraction with the bwd subscript multiset
    // {2x "qd,kd->qk", 2x "qk,qd->kd", 1x "qk,kd->qd"}. Identify each by its
    // operands (shape/DAG driven, NOT a kernel-name match).
    Block &lbody = loop.getBody().front();
    Value Qt = lbody.getArgument(0); // query tile (scan slice 0)
    Value dOt = lbody.getArgument(1);
    ContractionOp sC, dpC, dvC, dkC, dqC;
    for (Operation &op : lbody.without_terminator()) {
      if (auto c = dyn_cast<ContractionOp>(&op)) {
        StringRef sub = c.getSubscripts();
        if (sub == "qd,kd->qk") {
          if (c.getA() == Qt)
            sC = c; // S = Q@K^T
          else if (c.getA() == dOt)
            dpC = c; // dP = dO@V^T
          else
            return failure();
        } else if (sub == "qk,qd->kd") {
          if (c.getB() == dOt)
            dvC = c; // dV = P^T@dO
          else if (c.getB() == Qt)
            dkC = c; // dK = dS^T@Q
          else
            return failure();
        } else if (sub == "qk,kd->qd") {
          dqC = c; // dQ = dS@K
        } else {
          return failure();
        }
      } else if (!isa<MapBinaryOp, MapUnaryOp, SelectOp, arith::ConstantOp,
                      tensor::SplatOp>(&op)) {
        return failure();
      }
    }
    if (!sC || !dpC || !dvC || !dkC || !dqC)
      return failure();

    // K = the B operand of S (Q@K^T); V = the B operand of dP (dO@V^T); both must
    // be func block args. dQ's B must also be K.
    auto kArg = dyn_cast<BlockArgument>(sC.getB());
    auto vArg = dyn_cast<BlockArgument>(dpC.getB());
    if (!kArg || !vArg || kArg.getOwner() != &body || vArg.getOwner() != &body)
      return failure();
    if (dqC.getB() != sC.getB())
      return failure();
    kArgIdx = kArg.getArgNumber();
    vArgIdx = vArg.getArgNumber();

    // Shapes + element types. Q[Sq,D] f16 (scan input, full query axis), K/V
    // [Sk,D] f16, dO[Sq,D] f16, LSE/Delta [Sq,1] f32.
    auto qTy = dyn_cast<RankedTensorType>(qArg.getType());
    auto kTy = dyn_cast<RankedTensorType>(kArg.getType());
    auto vTy = dyn_cast<RankedTensorType>(vArg.getType());
    auto doTy = dyn_cast<RankedTensorType>(doArg.getType());
    if (!qTy || !kTy || !vTy || !doTy)
      return failure();
    if (!qTy.hasStaticShape() || !kTy.hasStaticShape() ||
        !vTy.hasStaticShape() || !doTy.hasStaticShape())
      return failure();
    if (qTy.getRank() != 2 || kTy.getRank() != 2 || vTy.getRank() != 2 ||
        doTy.getRank() != 2)
      return failure();
    if (!isa<Float16Type>(qTy.getElementType()) ||
        !isa<Float16Type>(kTy.getElementType()) ||
        !isa<Float16Type>(vTy.getElementType()) ||
        !isa<Float16Type>(doTy.getElementType()))
      return failure();
    f16 = qTy.getElementType();

    Sq = qTy.getDimSize(0);
    D = qTy.getDimSize(1);
    Sk = kTy.getDimSize(0);
    if (kTy.getDimSize(1) != D || vTy.getDimSize(0) != Sk ||
        vTy.getDimSize(1) != D || doTy.getDimSize(0) != Sq ||
        doTy.getDimSize(1) != D)
      return failure();
    if (Sq % BM != 0 || Sk % BN != 0 || D % kWMMA != 0)
      return failure();

    // Carries dk/dv are f32 [Sk,D]; the func results equal the carries.
    for (Value init : loop.getInits()) {
      auto t = dyn_cast<RankedTensorType>(init.getType());
      if (!t || !t.hasStaticShape() || !isa<Float32Type>(t.getElementType()) ||
          t.getRank() != 2 || t.getDimSize(0) != Sk || t.getDimSize(1) != D)
        return failure();
      f32 = t.getElementType();
    }

    // The scale: the func's lone scalar f32 arg.
    bool haveScale = false;
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      if (isa<Float32Type>(fty.getInput(i))) {
        scaleArgIdx = i;
        haveScale = true;
      } else if (!isa<RankedTensorType>(fty.getInput(i))) {
        return failure();
      }
    }
    if (!haveScale)
      return failure();
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: Q,K,V,dO,LSE,Delta memrefs + scale f32 + appended
    // destination-passing dQ[Sq,D], dK[Sk,D], dV[Sk,D] f32.
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      if (auto t = dyn_cast<RankedTensorType>(at))
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      else
        inTys.push_back(at); // scale f32 by value
    }
    inTys.push_back(MemRefType::get({Sq, D}, f32)); // dQ
    inTys.push_back(MemRefType::get({Sk, D}, f32)); // dK
    inTys.push_back(MemRefType::get({Sk, D}, f32)); // dV
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value qMem = entry->getArgument(qArgIdx);
    Value kMem = entry->getArgument(kArgIdx);
    Value vMem = entry->getArgument(vArgIdx);
    Value doMem = entry->getArgument(doArgIdx);
    Value lseMem = entry->getArgument(lseArgIdx);
    Value deltaMem = entry->getArgument(deltaArgIdx);
    Value scale = entry->getArgument(scaleArgIdx);
    unsigned numIn = fn.getFunctionType().getNumInputs();
    Value dqMem = entry->getArgument(numIn);
    Value dkMem = entry->getArgument(numIn + 1);
    Value dvMem = entry->getArgument(numIn + 2);

    int64_t numKT = Sk / BN; // outer key tiles
    int64_t numQT = Sq / BM; // inner query tiles

    // Dynamic-shared layout (one raw i8 buffer + 16-byte-aligned views). dk/dv
    // accumulators + K/V key tile resident across the inner query loop; Q/dO/
    // LSE/Delta loaded per query tile; Sd[BM,BN] f32 + Pds[BM,BN] f16 scratch
    // reused across the 7 steps.
    int64_t oK = 0;
    int64_t oV = align16(oK + BN * D * 2);
    int64_t oDK = align16(oV + BN * D * 2);
    int64_t oDV = align16(oDK + BN * D * 4);
    int64_t oQ = align16(oDV + BN * D * 4);
    int64_t oDO = align16(oQ + BM * D * 2);
    int64_t oSd = align16(oDO + BM * D * 2);
    int64_t oPds = align16(oSd + BM * BN * 4);
    int64_t oLSE = align16(oPds + BM * BN * 2);
    int64_t oDelta = align16(oLSE + BM * 4);
    int64_t shTotal = align16(oDelta + BM * 4);

    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value shBytes =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shTotal));

    // grid = (1,1,1): ONE block per (b,h) owns the whole head (serial DQ).
    auto launch =
        b.create<gpu::LaunchOp>(loc, c1, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value cWMMA = b.create<arith::ConstantIndexOp>(loc, kWMMA);
    Value warpRow16 = b.create<arith::MulIOp>(loc, warp, cWMMA);

    // Shared views.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto viewF16 = [&](int64_t off, ArrayRef<int64_t> shp) -> Value {
      auto ty = MemRefType::get(shp, f16, nullptr, wgSpace);
      Value o = b.create<arith::ConstantIndexOp>(loc, off);
      return b.create<memref::ViewOp>(loc, ty, raw, o, ValueRange{});
    };
    auto viewF32 = [&](int64_t off, ArrayRef<int64_t> shp) -> Value {
      auto ty = MemRefType::get(shp, f32, nullptr, wgSpace);
      Value o = b.create<arith::ConstantIndexOp>(loc, off);
      return b.create<memref::ViewOp>(loc, ty, raw, o, ValueRange{});
    };
    Value kSh = viewF16(oK, {BN, D});
    Value vSh = viewF16(oV, {BN, D});
    Value dkSh = viewF32(oDK, {BN, D});
    Value dvSh = viewF32(oDV, {BN, D});
    Value qSh = viewF16(oQ, {BM, D});
    Value doSh = viewF16(oDO, {BM, D});
    Value sdSh = viewF32(oSd, {BM, BN});
    Value pdsSh = viewF16(oPds, {BM, BN});
    Value lseSh = viewF32(oLSE, {BM});
    Value deltaSh = viewF32(oDelta, {BM});

    // MMA fragment types + reusable attrs.
    auto copTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f32, "COp");
    auto aopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "AOp");
    auto bopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "BOp");
    UnitAttr transpose = b.getUnitAttr();
    IntegerAttr ldD = b.getIndexAttr(D);
    IntegerAttr ldBN = b.getIndexAttr(BN);

    // Cooperative element loop over an [R,C] tile (R*C multiple of kBlock).
    auto forEachElem = [&](OpBuilder &nb, int64_t R, int64_t C,
                           function_ref<void(OpBuilder &, Value, Value)> bodyFn) {
      Value cC = nb.create<arith::ConstantIndexOp>(loc, C);
      int64_t iters = (R * C) / kBlock;
      for (int64_t it = 0; it < iters; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value i = nb.create<arith::AddIOp>(loc, tx, base);
        Value r = nb.create<arith::DivUIOp>(loc, i, cC);
        Value c = nb.create<arith::RemUIOp>(loc, i, cC);
        bodyFn(nb, r, c);
      }
    };

    // Generic WMMA contraction: out[I,J] = sum_l A * B over Kext, FRESH (init 0),
    // stored to outSh. Each warp owns rows [warp*16,+16) of the I axis (I=64).
    //   aTrans=false: A is [I,Kext], frag tile at [warpRow16, ks*16], lead=aLD.
    //   aTrans=true : A is stored [Kext,I], frag at [ks*16, warpRow16], lead=aLD.
    //   bTrans=false: B is [Kext,J], frag tile at [ks*16, j*16], lead=bLD.
    //   bTrans=true : B is stored [J,Kext], frag at [j*16, ks*16], lead=bLD.
    auto emitMMA = [&](OpBuilder &nb, Value aSh, IntegerAttr aLD, bool aTrans,
                       Value bSh, IntegerAttr bLD, bool bTrans, Value outSh,
                       IntegerAttr outLD, int64_t Jext, int64_t Kext) {
      int64_t nCol = Jext / kWMMA, kSub = Kext / kWMMA;
      SmallVector<Value> acc;
      for (int64_t j = 0; j < nCol; ++j)
        acc.push_back(
            nb.create<gpu::SubgroupMmaConstantMatrixOp>(loc, copTy, zeroF));
      for (int64_t ks = 0; ks < kSub; ++ks) {
        Value ksc = nb.create<arith::ConstantIndexOp>(loc, ks * kWMMA);
        SmallVector<Value> aIdx = aTrans ? SmallVector<Value>{ksc, warpRow16}
                                         : SmallVector<Value>{warpRow16, ksc};
        Value aFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc, aopTy, aSh, aIdx, aLD, aTrans ? transpose : UnitAttr());
        for (int64_t j = 0; j < nCol; ++j) {
          Value cj = nb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
          SmallVector<Value> bIdx = bTrans ? SmallVector<Value>{cj, ksc}
                                           : SmallVector<Value>{ksc, cj};
          Value bFrag = nb.create<gpu::SubgroupMmaLoadMatrixOp>(
              loc, bopTy, bSh, bIdx, bLD, bTrans ? transpose : UnitAttr());
          acc[j] = nb.create<gpu::SubgroupMmaComputeOp>(
              loc, copTy, aFrag, bFrag, acc[j], /*a_transpose=*/nullptr,
              /*b_transpose=*/nullptr);
        }
      }
      for (int64_t j = 0; j < nCol; ++j) {
        Value cj = nb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
        nb.create<gpu::SubgroupMmaStoreMatrixOp>(loc, acc[j], outSh,
                                                 ValueRange{warpRow16, cj},
                                                 outLD, /*transpose=*/nullptr);
      }
    };

    // ---- OUTER loop over key tiles -------------------------------------- //
    Value cNumKT = b.create<arith::ConstantIndexOp>(loc, numKT);
    Value cBNv = b.create<arith::ConstantIndexOp>(loc, BN);
    Value cBMv = b.create<arith::ConstantIndexOp>(loc, BM);
    auto knLoop = b.create<scf::ForOp>(loc, c0, cNumKT, c1);
    {
      OpBuilder ob(knLoop.getBody(), knLoop.getBody()->begin());
      Value kn = knLoop.getInductionVar();
      Value keyBase = ob.create<arith::MulIOp>(loc, kn, cBNv);

      // Load Kj, Vj -> shared; reset dk/dv = 0.
      forEachElem(ob, BN, D, [&](OpBuilder &nb, Value r, Value c) {
        Value gr = nb.create<arith::AddIOp>(loc, keyBase, r);
        Value kv = nb.create<memref::LoadOp>(loc, kMem, ValueRange{gr, c});
        nb.create<memref::StoreOp>(loc, kv, kSh, ValueRange{r, c});
        Value vv = nb.create<memref::LoadOp>(loc, vMem, ValueRange{gr, c});
        nb.create<memref::StoreOp>(loc, vv, vSh, ValueRange{r, c});
        nb.create<memref::StoreOp>(loc, zeroF, dkSh, ValueRange{r, c});
        nb.create<memref::StoreOp>(loc, zeroF, dvSh, ValueRange{r, c});
      });
      ob.create<gpu::BarrierOp>(loc);

      // ---- INNER loop over query tiles (the intent.loop scan) ----------- //
      Value cNumQT = ob.create<arith::ConstantIndexOp>(loc, numQT);
      auto qmLoop = ob.create<scf::ForOp>(loc, c0, cNumQT, c1);
      {
        OpBuilder lb(qmLoop.getBody(), qmLoop.getBody()->begin());
        Value qm = qmLoop.getInductionVar();
        Value rowBase = lb.create<arith::MulIOp>(loc, qm, cBMv);

        // Load Qt, dOt -> shared; LSE_row, Delta_row -> shared.
        forEachElem(lb, BM, D, [&](OpBuilder &nb, Value r, Value c) {
          Value gr = nb.create<arith::AddIOp>(loc, rowBase, r);
          Value q = nb.create<memref::LoadOp>(loc, qMem, ValueRange{gr, c});
          nb.create<memref::StoreOp>(loc, q, qSh, ValueRange{r, c});
          Value d = nb.create<memref::LoadOp>(loc, doMem, ValueRange{gr, c});
          nb.create<memref::StoreOp>(loc, d, doSh, ValueRange{r, c});
        });
        {
          Value isRow = lb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, tx, cBMv);
          lb.create<scf::IfOp>(loc, isRow, [&](OpBuilder &nb, Location nl) {
            Value gr = nb.create<arith::AddIOp>(nl, rowBase, tx);
            Value c0i = nb.create<arith::ConstantIndexOp>(nl, 0);
            Value l =
                nb.create<memref::LoadOp>(nl, lseMem, ValueRange{gr, c0i});
            nb.create<memref::StoreOp>(nl, l, lseSh, ValueRange{tx});
            Value dl =
                nb.create<memref::LoadOp>(nl, deltaMem, ValueRange{gr, c0i});
            nb.create<memref::StoreOp>(nl, dl, deltaSh, ValueRange{tx});
            nb.create<scf::YieldOp>(nl);
          });
        }
        lb.create<gpu::BarrierOp>(loc);

        // (1) S = Qt@Kj^T -> Sd. A=Q[q,d] normal, B=K[k,d] transposed.
        emitMMA(lb, qSh, ldD, false, kSh, ldD, true, sdSh, ldBN, BN, D);
        lb.create<gpu::BarrierOp>(loc);

        // (2) coop: P = (f16)exp(scale*Sd - LSE_row) -> Pds.
        forEachElem(lb, BM, BN, [&](OpBuilder &nb, Value r, Value c) {
          Value s = nb.create<memref::LoadOp>(loc, sdSh, ValueRange{r, c});
          Value ss = nb.create<arith::MulFOp>(loc, s, scale);
          Value lse = nb.create<memref::LoadOp>(loc, lseSh, ValueRange{r});
          Value sm = nb.create<arith::SubFOp>(loc, ss, lse);
          Value p = nb.create<math::ExpOp>(loc, sm);
          Value ph = nb.create<arith::TruncFOp>(loc, f16, p);
          nb.create<memref::StoreOp>(loc, ph, pdsSh, ValueRange{r, c});
        });
        lb.create<gpu::BarrierOp>(loc);

        // (3) dV product = P^T@dO -> Sd ; coop dv_sh += Sd.
        //     out[k,d]=sum_q P[q,k]*dO[q,d]. A=P transposed (stored [q,k]),
        //     B=dO[q,d] normal, Kext=BM (query).
        emitMMA(lb, pdsSh, ldBN, true, doSh, ldD, false, sdSh, ldD, D, BM);
        lb.create<gpu::BarrierOp>(loc);
        forEachElem(lb, BN, D, [&](OpBuilder &nb, Value r, Value c) {
          Value acc = nb.create<memref::LoadOp>(loc, dvSh, ValueRange{r, c});
          Value add = nb.create<memref::LoadOp>(loc, sdSh, ValueRange{r, c});
          Value nv = nb.create<arith::AddFOp>(loc, acc, add);
          nb.create<memref::StoreOp>(loc, nv, dvSh, ValueRange{r, c});
        });
        lb.create<gpu::BarrierOp>(loc);

        // (4) dP = dO@Vj^T -> Sd. A=dO[q,d] normal, B=V[k,d] transposed.
        emitMMA(lb, doSh, ldD, false, vSh, ldD, true, sdSh, ldBN, BN, D);
        lb.create<gpu::BarrierOp>(loc);

        // (5) coop: dS = (f16)(P*(Sd - Delta_row)*scale) -> Pds (P's last use).
        forEachElem(lb, BM, BN, [&](OpBuilder &nb, Value r, Value c) {
          Value p = nb.create<memref::LoadOp>(loc, pdsSh, ValueRange{r, c});
          Value pf = nb.create<arith::ExtFOp>(loc, f32, p);
          Value dp = nb.create<memref::LoadOp>(loc, sdSh, ValueRange{r, c});
          Value dl = nb.create<memref::LoadOp>(loc, deltaSh, ValueRange{r});
          Value diff = nb.create<arith::SubFOp>(loc, dp, dl);
          Value pds = nb.create<arith::MulFOp>(loc, pf, diff);
          Value ds = nb.create<arith::MulFOp>(loc, pds, scale);
          Value dsh = nb.create<arith::TruncFOp>(loc, f16, ds);
          nb.create<memref::StoreOp>(loc, dsh, pdsSh, ValueRange{r, c});
        });
        lb.create<gpu::BarrierOp>(loc);

        // (6) dK product = dS^T@Q -> Sd ; coop dk_sh += Sd.
        //     out[k,d]=sum_q dS[q,k]*Q[q,d]. A=dS transposed, B=Q normal, Kext=BM.
        emitMMA(lb, pdsSh, ldBN, true, qSh, ldD, false, sdSh, ldD, D, BM);
        lb.create<gpu::BarrierOp>(loc);
        forEachElem(lb, BN, D, [&](OpBuilder &nb, Value r, Value c) {
          Value acc = nb.create<memref::LoadOp>(loc, dkSh, ValueRange{r, c});
          Value add = nb.create<memref::LoadOp>(loc, sdSh, ValueRange{r, c});
          Value nk = nb.create<arith::AddFOp>(loc, acc, add);
          nb.create<memref::StoreOp>(loc, nk, dkSh, ValueRange{r, c});
        });
        lb.create<gpu::BarrierOp>(loc);

        // (7) dQ product = dS@Kj -> Sd ; coop DQ[start_m] += Sd (RMW global).
        //     out[q,d]=sum_k dS[q,k]*K[k,d]. A=dS[q,k] normal, B=K[k,d] normal,
        //     Kext=BN (key).
        emitMMA(lb, pdsSh, ldBN, false, kSh, ldD, false, sdSh, ldD, D, BN);
        lb.create<gpu::BarrierOp>(loc);
        forEachElem(lb, BM, D, [&](OpBuilder &nb, Value r, Value c) {
          Value gr = nb.create<arith::AddIOp>(loc, rowBase, r);
          Value acc = nb.create<memref::LoadOp>(loc, dqMem, ValueRange{gr, c});
          Value add = nb.create<memref::LoadOp>(loc, sdSh, ValueRange{r, c});
          Value nq = nb.create<arith::AddFOp>(loc, acc, add);
          nb.create<memref::StoreOp>(loc, nq, dqMem, ValueRange{gr, c});
        });
        lb.create<gpu::BarrierOp>(loc);
      }

      // Write dk/dv shared accumulators -> DK[kn], DV[kn] global.
      forEachElem(ob, BN, D, [&](OpBuilder &nb, Value r, Value c) {
        Value gr = nb.create<arith::AddIOp>(loc, keyBase, r);
        Value dk = nb.create<memref::LoadOp>(loc, dkSh, ValueRange{r, c});
        nb.create<memref::StoreOp>(loc, dk, dkMem, ValueRange{gr, c});
        Value dv = nb.create<memref::LoadOp>(loc, dvSh, ValueRange{r, c});
        nb.create<memref::StoreOp>(loc, dv, dvMem, ValueRange{gr, c});
      });
      ob.create<gpu::BarrierOp>(loc); // dk/dv written before next key tile reset
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentAttentionBackwardToGPUPass
    : ::intent::impl::ConvertIntentAttentionBackwardToGPUBase<
          ConvertIntentAttentionBackwardToGPUPass> {
  using ConvertIntentAttentionBackwardToGPUBase::
      ConvertIntentAttentionBackwardToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      FlashBwdLowering probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      FlashBwdLowering low(fn);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("flash-bwd lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
