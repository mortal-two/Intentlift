//===- ConvertIntentLoopContractionToGPU.cpp - fused flash fwd -> gpu ------===//
//
// Lowers a func whose body is the FlashAttention FORWARD as one `intent.loop`
// scanning the key axis (carries the online-softmax state m[Sq,1], l[Sq,1],
// O[Sq,D]; body = QK^T contraction -> row max/exp/sum -> PV contraction ->
// online rescale) to ONE tiled tensor-core `gpu.launch`.
//
// THE FUSION (sidesteps the opaque-fragment rowmax/rescale that blocked the
// per-thread-scalar flash_emit). The online-softmax state lives RESIDENT IN
// DYNAMIC SHARED MEMORY (S, P, O, m, l, alpha), and only the two matmuls run on
// tensor cores (WMMA, reusing the M7 contraction primitive). Per block
// (BM=BN=64 query/key tile, 4 warps each owning 16 C rows):
//
//   Q[BM,D] staged into shared ONCE (cooperative f16 copy).
//   O[BM,D]=0, m[BM]=-inf, l[BM]=0 (cooperative init).
//   scf.for kt = 0..Sk step BN  (NO iter_args -- the carries live in shared,
//                                made visible across trips by gpu.barrier):
//     (1) cooperative copy K[kt:+BN,:], V[kt:+BN,:] global -> shared (f16).
//     (2) S = Q@Kj^T via WMMA (AOp=Q, BOp=Kj loaded TRANSPOSED, "qd,kd->qk"),
//         store the COp fragments to the shared [BM,BN] f32 S tile (M7b extract).
//     (3) COOPERATIVE online softmax over the shared S tile, one thread per row
//         (tx<BM): m_new = max(m, max_c scale*S[r,c]); alpha = exp(m - m_new);
//         P[r,c] = (f16)exp(scale*S[r,c] - m_new); l = l*alpha + sum_c P;
//         m := m_new; stash alpha[r].   (plain shared arith, NOT fragment lanes.)
//     (4) PV = P@Vj via WMMA (AOp=P, BOp=Vj NOT transposed, "qk,kd->qd"), store
//         to a shared [BM,D] f32 PV tile.
//     (5) cooperative O[r,c] = O[r,c]*alpha[r] + PV[r,c].
//   Epilogue: cooperative O_global[r,c] = O_sh[r,c] / l[r].
//
// The single-tile case (Sk=BN) degenerates correctly: m_old=-inf so alpha=0,
// O=0*0+PV=PV, l=0*0+rowsum -> O=softmax(scale*QK^T)@V.
//
// Q/K/V are f16 (WMMA operands), the softmax/accumulator state is f32. Static
// shapes, BM=BN=64, D%16==0, Sq%64==0, Sk%64==0 (non-causal, no bias). Bare-ptr
// ABI; the lowered module goes to lower_memref_module_to_kernels with
// scf_to_cf=True (the gpu->nvvm pipeline needs scf->cf to legalize gpu.launch).
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
#define GEN_PASS_DEF_CONVERTINTENTLOOPCONTRACTIONTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Tile geometry. BM/BN = the block's query/key tile (64x64), 4 warps each owning
// a 16-row C strip. The contraction k-step is the WMMA k (16). D is the head dim.
static constexpr int64_t BM = 64, BN = 64, kWMMA = 16, kBlock = 128;

static int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

/// One fused-flash func's tensor-core lowering. Reads the original (tensor) func
/// body (init splats + one intent.loop + a divide epilogue) and builds a new
/// memref+gpu.launch func with the same name, erasing the old.
struct FusedAttnLoopLowering {
  func::FuncOp fn;

  // populated by match()
  int64_t Sq = 0, Sk = 0, D = 0;
  // Split-K (batched) support: K/V/O carry a leading split/batch dim Z mapped to
  // grid.y; Q stays rank-2 (shared across splits). LSE = m + log(l) is an
  // optional second result (the split-K partial log-sum-exp).
  bool batched = false;
  int64_t Z = 1;
  bool hasLSE = false;
  unsigned oResultIdx = 0, lseResultIdx = 0;
  unsigned qArgIdx = 0, kArgIdx = 0, vArgIdx = 0, scaleArgIdx = 0;
  LoopOp loop;
  Type f16, f32;

  explicit FusedAttnLoopLowering(func::FuncOp fn) : fn(fn) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if ((fty.getNumResults() != 1 && fty.getNumResults() != 2) ||
        !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    // Body: constant splats (init carries) + exactly one intent.loop + a
    // straight-line divide epilogue (O = Oacc / l) + return. Anything else
    // (a second loop, a reduce outside the loop) is out of scope.
    LoopOp theLoop;
    for (Operation &op : body.without_terminator()) {
      if (auto lp = dyn_cast<LoopOp>(&op)) {
        if (theLoop)
          return failure();
        theLoop = lp;
      } else if (!isa<arith::ConstantOp, MapBinaryOp, MapUnaryOp>(&op)) {
        return failure();
      }
    }
    if (!theLoop)
      return failure();
    loop = theLoop;

    // The loop scans (K, V) and carries (m[Sq,1], l[Sq,1], O[Sq,D]).
    if (loop.getScanInputs().size() != 2 || loop.getInits().size() != 3)
      return failure();
    auto kArg = dyn_cast<BlockArgument>(loop.getScanInputs()[0]);
    auto vArg = dyn_cast<BlockArgument>(loop.getScanInputs()[1]);
    if (!kArg || !vArg || kArg.getOwner() != &body || vArg.getOwner() != &body)
      return failure();
    kArgIdx = kArg.getArgNumber();
    vArgIdx = vArg.getArgNumber();

    // The loop body: exactly two intent.contraction (QK^T then PV) and two
    // intent.reduce (max then sum) over the last axis. Recognize by the contract
    // subscripts + reduce attrs (shape/DAG driven, NOT a kernel-name match).
    Block &lbody = loop.getBody().front();
    ContractionOp qk, pv;
    int nReduce = 0;
    bool sawMax = false, sawSum = false;
    for (Operation &op : lbody.without_terminator()) {
      if (auto c = dyn_cast<ContractionOp>(&op)) {
        // Plain (M8a) or batched-RHS (split-K) subscripts. In the batched form
        // the key/value tiles carry a leading split dim z (Q stays rank-2 and
        // is shared): S = Q@Kj^T over each z, PV = P@Vj over each z.
        StringRef sub = c.getSubscripts();
        if (sub == "qd,kd->qk" || sub == "qd,zkd->zqk")
          qk = c;
        else if (sub == "qk,kd->qd" || sub == "zqk,zkd->zqd")
          pv = c;
        else
          return failure();
      } else if (auto r = dyn_cast<ReduceOp>(&op)) {
        ++nReduce;
        if (r.getFn() == "max")
          sawMax = true;
        else if (r.getFn() == "sum")
          sawSum = true;
      } else if (!isa<MapBinaryOp, MapUnaryOp, SelectOp, arith::ConstantOp,
                      tensor::SplatOp>(&op)) {
        return failure();
      }
    }
    if (!qk || !pv || nReduce != 2 || !sawMax || !sawSum)
      return failure();

    // QK^T A operand = Q, a func block arg [Sq,D] f16; its B operand = the K tile
    // (loop body scan arg 0). PV B operand = the V tile (loop body scan arg 1).
    auto qArg = dyn_cast<BlockArgument>(qk.getA());
    if (!qArg || qArg.getOwner() != &body)
      return failure();
    qArgIdx = qArg.getArgNumber();
    if (qk.getB() != lbody.getArgument(0) || pv.getB() != lbody.getArgument(1))
      return failure();

    // Shapes + element types. Q[Sq,D] f16 (always rank-2, shared across splits);
    // K/V/O either rank-2 (M8a: [Sk,D]/[Sq,D]) or rank-3 (split-K: a leading
    // split dim Z -> K/V[Z,Sk,D], O[Z,Sq,D]).
    auto qTy = dyn_cast<RankedTensorType>(qk.getA().getType());
    auto kTy = dyn_cast<RankedTensorType>(kArg.getType());
    auto vTy = dyn_cast<RankedTensorType>(vArg.getType());
    if (!qTy || !kTy || !vTy)
      return failure();
    if (!qTy.hasStaticShape() || !kTy.hasStaticShape() || !vTy.hasStaticShape())
      return failure();
    if (qTy.getRank() != 2)
      return failure();
    batched = (kTy.getRank() == 3);
    int64_t kvRank = batched ? 3 : 2;
    if (kTy.getRank() != kvRank || vTy.getRank() != kvRank)
      return failure();
    if (!isa<Float16Type>(qTy.getElementType()) ||
        !isa<Float16Type>(kTy.getElementType()) ||
        !isa<Float16Type>(vTy.getElementType()))
      return failure();
    f16 = qTy.getElementType();

    Sq = qTy.getDimSize(0);
    D = qTy.getDimSize(1);
    if (batched) {
      Z = kTy.getDimSize(0);
      Sk = kTy.getDimSize(1);
      if (kTy.getDimSize(2) != D || vTy.getDimSize(0) != Z ||
          vTy.getDimSize(1) != Sk || vTy.getDimSize(2) != D)
        return failure();
    } else {
      Z = 1;
      Sk = kTy.getDimSize(0);
      if (kTy.getDimSize(1) != D || vTy.getDimSize(0) != Sk ||
          vTy.getDimSize(1) != D)
        return failure();
    }
    if (Sq % BM != 0 || Sk % BN != 0 || D % kWMMA != 0)
      return failure();

    // Classify the func results: O is the f32 [.,Sq,D] result; an optional
    // second result LSE is the f32 [.,Sq,1] split-K log-sum-exp.
    int64_t oRank = batched ? 3 : 2;
    hasLSE = false;
    bool foundO = false;
    for (unsigned i = 0; i < fty.getNumResults(); ++i) {
      auto rt = dyn_cast<RankedTensorType>(fty.getResult(i));
      if (!rt || !rt.hasStaticShape() || !isa<Float32Type>(rt.getElementType()))
        return failure();
      int64_t last = rt.getDimSize(rt.getRank() - 1);
      if (rt.getRank() == oRank && last == D) {
        if (foundO)
          return failure();
        // O[.,Sq,D]
        if (batched) {
          if (rt.getDimSize(0) != Z || rt.getDimSize(1) != Sq)
            return failure();
        } else if (rt.getDimSize(0) != Sq) {
          return failure();
        }
        oResultIdx = i;
        foundO = true;
        f32 = rt.getElementType();
      } else if (rt.getRank() == oRank && last == 1) {
        // LSE[.,Sq,1]
        if (batched) {
          if (rt.getDimSize(0) != Z || rt.getDimSize(1) != Sq)
            return failure();
        } else if (rt.getDimSize(0) != Sq) {
          return failure();
        }
        lseResultIdx = i;
        hasLSE = true;
      } else {
        return failure();
      }
    }
    if (!foundO)
      return failure();
    if (fty.getNumResults() == 2 && !hasLSE)
      return failure();

    // The scale: the func's lone scalar f32 arg. (The other three ranked f16
    // args are Q/K/V.) Optional -- absent means scale = 1.0.
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
      return failure(); // require an explicit scale arg for the clean ABI
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: Q,K,V memrefs + scale f32 + destination-passing O[Sq,D] f32.
    SmallVector<Type> inTys;
    for (Type at : fn.getFunctionType().getInputs()) {
      if (auto t = dyn_cast<RankedTensorType>(at))
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      else
        inTys.push_back(at); // scale f32 by value
    }
    // Destination-passing outputs appended in a fixed order: O then (optional)
    // LSE. Shapes carry the leading split dim Z when batched.
    SmallVector<int64_t> oShape = batched ? SmallVector<int64_t>{Z, Sq, D}
                                          : SmallVector<int64_t>{Sq, D};
    inTys.push_back(MemRefType::get(oShape, f32));
    if (hasLSE) {
      SmallVector<int64_t> lseShape = batched ? SmallVector<int64_t>{Z, Sq, 1}
                                              : SmallVector<int64_t>{Sq, 1};
      inTys.push_back(MemRefType::get(lseShape, f32));
    }
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
    Value scale = entry->getArgument(scaleArgIdx);
    unsigned numIn = fn.getFunctionType().getNumInputs();
    Value oMem = entry->getArgument(numIn);
    Value lseMem = hasLSE ? entry->getArgument(numIn + 1) : Value();

    int64_t gridM = Sq / BM;
    int64_t numKT = Sk / BN;
    int64_t nColQK = BN / kWMMA; // # column MMA tiles for S (per warp)
    int64_t nColPV = D / kWMMA;  // # column MMA tiles for PV (per warp)
    int64_t kSubQK = D / kWMMA;  // # k-subtiles for QK^T (contraction = D)
    int64_t kSubPV = BN / kWMMA; // # k-subtiles for PV (contraction = BN)

    // Dynamic-shared layout (one raw i8 buffer + 16-byte-aligned views).
    int64_t oQ = 0;
    int64_t oK = align16(oQ + BM * D * 2);
    int64_t oV = align16(oK + BN * D * 2);
    int64_t oS = align16(oV + BN * D * 2);
    int64_t oP = align16(oS + BM * BN * 4);
    int64_t oPV = align16(oP + BM * BN * 2);
    int64_t oO = align16(oPV + BM * D * 4);
    int64_t oM = align16(oO + BM * D * 4);
    int64_t oL = align16(oM + BM * 4);
    int64_t oA = align16(oL + BM * 4);
    int64_t shTotal = align16(oA + BM * 4);

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cGridY = b.create<arith::ConstantIndexOp>(loc, batched ? Z : 1);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value shBytes =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shTotal));

    // grid = (Sq/BM, Z, 1): blockIdx.x selects the query tile, blockIdx.y the
    // split-K chunk (== 1 when not batched).
    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, cGridY, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value by = launch.getBlockIds().y; // split-K chunk index (batched)
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cBM = b.create<arith::ConstantIndexOp>(loc, BM);
    Value ninf = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));

    Value rowBlk = b.create<arith::MulIOp>(loc, bx, cBM); // this block's query base
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value cWMMA = b.create<arith::ConstantIndexOp>(loc, kWMMA);
    Value warpRow16 = b.create<arith::MulIOp>(
        loc, warp, cWMMA); // this warp's row offset within the BM tile

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
    Value qSh = viewF16(oQ, {BM, D});
    Value kSh = viewF16(oK, {BN, D});
    Value vSh = viewF16(oV, {BN, D});
    Value sSh = viewF32(oS, {BM, BN});
    Value pSh = viewF16(oP, {BM, BN});
    Value pvSh = viewF32(oPV, {BM, D});
    Value oSh = viewF32(oO, {BM, D});
    Value mSh = viewF32(oM, {BM});
    Value lSh = viewF32(oL, {BM});
    Value aSh = viewF32(oA, {BM});

    // ---- cooperative element copy / init helpers ------------------------- //
    // Run `body(i, r, c)` for each linear element i of an [R,C] tile, strided by
    // the block size (kBlock). R*C must be a multiple of kBlock.
    auto forEachElem = [&](int64_t R, int64_t C,
                           function_ref<void(OpBuilder &, Value, Value)>
                               bodyFn) {
      Value cC = b.create<arith::ConstantIndexOp>(loc, C);
      int64_t iters = (R * C) / kBlock;
      for (int64_t it = 0; it < iters; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value i = b.create<arith::AddIOp>(loc, tx, base);
        Value r = b.create<arith::DivUIOp>(loc, i, cC);
        Value c = b.create<arith::RemUIOp>(loc, i, cC);
        bodyFn(b, r, c);
      }
    };

    // Stage Q[rowBlk:+BM, :] -> qSh (f16), once.
    forEachElem(BM, D, [&](OpBuilder &nb, Value r, Value c) {
      Value gr = nb.create<arith::AddIOp>(loc, rowBlk, r);
      Value v = nb.create<memref::LoadOp>(loc, qMem, ValueRange{gr, c});
      nb.create<memref::StoreOp>(loc, v, qSh, ValueRange{r, c});
    });
    // Init O=0, m=-inf, l=0.
    forEachElem(BM, D, [&](OpBuilder &nb, Value r, Value c) {
      nb.create<memref::StoreOp>(loc, zeroF, oSh, ValueRange{r, c});
    });
    {
      Value isRow = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx,
                                            cBM);
      b.create<scf::IfOp>(loc, isRow, [&](OpBuilder &nb, Location nl) {
        nb.create<memref::StoreOp>(nl, ninf, mSh, ValueRange{tx});
        nb.create<memref::StoreOp>(nl, zeroF, lSh, ValueRange{tx});
        nb.create<scf::YieldOp>(nl);
      });
    }
    b.create<gpu::BarrierOp>(loc);

    // MMA fragment types.
    auto copTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f32, "COp");
    auto aopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "AOp");
    auto bopTy = gpu::MMAMatrixType::get({kWMMA, kWMMA}, f16, "BOp");
    IntegerAttr ldD = b.getIndexAttr(D);
    IntegerAttr ldBN = b.getIndexAttr(BN);
    UnitAttr transpose = b.getUnitAttr();

    // ---- the K-tile loop (no iter_args; carries live in shared) ---------- //
    Value cNumKT = b.create<arith::ConstantIndexOp>(loc, numKT);
    auto forOp = b.create<scf::ForOp>(loc, c0, cNumKT, c1);
    {
      OpBuilder lb(forOp.getBody(), forOp.getBody()->begin());
      Value kt = forOp.getInductionVar();
      Value cBNl = lb.create<arith::ConstantIndexOp>(loc, BN);
      Value keyBase = lb.create<arith::MulIOp>(loc, kt, cBNl);

      // (1) cooperative copy K[keyBase:+BN,:], V[keyBase:+BN,:] -> shared.
      Value cDl = lb.create<arith::ConstantIndexOp>(loc, D);
      {
        int64_t iters = (BN * D) / kBlock;
        for (int64_t it = 0; it < iters; ++it) {
          Value base = lb.create<arith::ConstantIndexOp>(loc, it * kBlock);
          Value i = lb.create<arith::AddIOp>(loc, tx, base);
          Value r = lb.create<arith::DivUIOp>(loc, i, cDl);
          Value c = lb.create<arith::RemUIOp>(loc, i, cDl);
          Value gr = lb.create<arith::AddIOp>(loc, keyBase, r);
          // batched: K/V are [Z,Sk,D]; index the split chunk with blockIdx.y.
          SmallVector<Value> kvIdx = batched ? SmallVector<Value>{by, gr, c}
                                             : SmallVector<Value>{gr, c};
          Value kv = lb.create<memref::LoadOp>(loc, kMem, kvIdx);
          lb.create<memref::StoreOp>(loc, kv, kSh, ValueRange{r, c});
          Value vv = lb.create<memref::LoadOp>(loc, vMem, kvIdx);
          lb.create<memref::StoreOp>(loc, vv, vSh, ValueRange{r, c});
        }
      }
      lb.create<gpu::BarrierOp>(loc);

      // (2) S = Q@Kj^T via WMMA. Per warp: nColQK COp accumulators over kSubQK.
      SmallVector<Value> sAcc;
      for (int64_t j = 0; j < nColQK; ++j)
        sAcc.push_back(
            lb.create<gpu::SubgroupMmaConstantMatrixOp>(loc, copTy, zeroF));
      for (int64_t ks = 0; ks < kSubQK; ++ks) {
        Value ksc = lb.create<arith::ConstantIndexOp>(loc, ks * kWMMA);
        Value aFrag = lb.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc, aopTy, qSh, ValueRange{warpRow16, ksc}, ldD,
            /*transpose=*/nullptr);
        for (int64_t j = 0; j < nColQK; ++j) {
          Value cj = lb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
          Value bFrag = lb.create<gpu::SubgroupMmaLoadMatrixOp>(
              loc, bopTy, kSh, ValueRange{cj, ksc}, ldD, transpose);
          sAcc[j] = lb.create<gpu::SubgroupMmaComputeOp>(
              loc, copTy, aFrag, bFrag, sAcc[j], /*a_transpose=*/nullptr,
              /*b_transpose=*/nullptr);
        }
      }
      for (int64_t j = 0; j < nColQK; ++j) {
        Value cj = lb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
        lb.create<gpu::SubgroupMmaStoreMatrixOp>(loc, sAcc[j], sSh,
                                                 ValueRange{warpRow16, cj}, ldBN,
                                                 /*transpose=*/nullptr);
      }
      lb.create<gpu::BarrierOp>(loc);

      // (3) cooperative online softmax over the shared S tile, one thread/row.
      Value isRow = lb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx,
                                             cBM);
      lb.create<scf::IfOp>(loc, isRow, [&](OpBuilder &nb, Location nl) {
        Value row = tx;
        Value mOld = nb.create<memref::LoadOp>(nl, mSh, ValueRange{row});
        Value lOld = nb.create<memref::LoadOp>(nl, lSh, ValueRange{row});
        // rowmax of scale*S[row, :]
        Value rmax = mOld;
        for (int64_t c = 0; c < BN; ++c) {
          Value cc = nb.create<arith::ConstantIndexOp>(nl, c);
          Value s = nb.create<memref::LoadOp>(nl, sSh, ValueRange{row, cc});
          Value ss = nb.create<arith::MulFOp>(nl, s, scale);
          rmax = nb.create<arith::MaxNumFOp>(nl, rmax, ss);
        }
        Value mNew = rmax;
        Value mdiff = nb.create<arith::SubFOp>(nl, mOld, mNew);
        Value alpha = nb.create<math::ExpOp>(nl, mdiff);
        // P[row, :] = exp(scale*S - mNew); rowsum
        Value rsum = zeroF;
        for (int64_t c = 0; c < BN; ++c) {
          Value cc = nb.create<arith::ConstantIndexOp>(nl, c);
          Value s = nb.create<memref::LoadOp>(nl, sSh, ValueRange{row, cc});
          Value ss = nb.create<arith::MulFOp>(nl, s, scale);
          Value sd = nb.create<arith::SubFOp>(nl, ss, mNew);
          Value p = nb.create<math::ExpOp>(nl, sd);
          Value ph = nb.create<arith::TruncFOp>(nl, f16, p);
          nb.create<memref::StoreOp>(nl, ph, pSh, ValueRange{row, cc});
          rsum = nb.create<arith::AddFOp>(nl, rsum, p);
        }
        Value lScaled = nb.create<arith::MulFOp>(nl, lOld, alpha);
        Value lNew = nb.create<arith::AddFOp>(nl, lScaled, rsum);
        nb.create<memref::StoreOp>(nl, mNew, mSh, ValueRange{row});
        nb.create<memref::StoreOp>(nl, lNew, lSh, ValueRange{row});
        nb.create<memref::StoreOp>(nl, alpha, aSh, ValueRange{row});
        nb.create<scf::YieldOp>(nl);
      });
      lb.create<gpu::BarrierOp>(loc);

      // (4) PV = P@Vj via WMMA. Per warp: nColPV COp accumulators over kSubPV.
      SmallVector<Value> pvAcc;
      for (int64_t j = 0; j < nColPV; ++j)
        pvAcc.push_back(
            lb.create<gpu::SubgroupMmaConstantMatrixOp>(loc, copTy, zeroF));
      for (int64_t ks = 0; ks < kSubPV; ++ks) {
        Value ksc = lb.create<arith::ConstantIndexOp>(loc, ks * kWMMA);
        Value aFrag = lb.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc, aopTy, pSh, ValueRange{warpRow16, ksc}, ldBN,
            /*transpose=*/nullptr);
        for (int64_t j = 0; j < nColPV; ++j) {
          Value cj = lb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
          Value bFrag = lb.create<gpu::SubgroupMmaLoadMatrixOp>(
              loc, bopTy, vSh, ValueRange{ksc, cj}, ldD, /*transpose=*/nullptr);
          pvAcc[j] = lb.create<gpu::SubgroupMmaComputeOp>(
              loc, copTy, aFrag, bFrag, pvAcc[j], /*a_transpose=*/nullptr,
              /*b_transpose=*/nullptr);
        }
      }
      IntegerAttr ldDpv = lb.getIndexAttr(D);
      for (int64_t j = 0; j < nColPV; ++j) {
        Value cj = lb.create<arith::ConstantIndexOp>(loc, j * kWMMA);
        lb.create<gpu::SubgroupMmaStoreMatrixOp>(loc, pvAcc[j], pvSh,
                                                 ValueRange{warpRow16, cj},
                                                 ldDpv, /*transpose=*/nullptr);
      }
      lb.create<gpu::BarrierOp>(loc);

      // (5) cooperative O[r,c] = O[r,c]*alpha[r] + PV[r,c].
      {
        int64_t iters = (BM * D) / kBlock;
        for (int64_t it = 0; it < iters; ++it) {
          Value base = lb.create<arith::ConstantIndexOp>(loc, it * kBlock);
          Value i = lb.create<arith::AddIOp>(loc, tx, base);
          Value r = lb.create<arith::DivUIOp>(loc, i, cDl);
          Value c = lb.create<arith::RemUIOp>(loc, i, cDl);
          Value oo = lb.create<memref::LoadOp>(loc, oSh, ValueRange{r, c});
          Value al = lb.create<memref::LoadOp>(loc, aSh, ValueRange{r});
          Value pv = lb.create<memref::LoadOp>(loc, pvSh, ValueRange{r, c});
          Value sc = lb.create<arith::MulFOp>(loc, oo, al);
          Value nn = lb.create<arith::AddFOp>(loc, sc, pv);
          lb.create<memref::StoreOp>(loc, nn, oSh, ValueRange{r, c});
        }
      }
      lb.create<gpu::BarrierOp>(loc);
    }

    // ---- epilogue: O_global[r,c] = O_sh[r,c] / l[r] ---------------------- //
    forEachElem(BM, D, [&](OpBuilder &nb, Value r, Value c) {
      Value oo = nb.create<memref::LoadOp>(loc, oSh, ValueRange{r, c});
      Value l = nb.create<memref::LoadOp>(loc, lSh, ValueRange{r});
      Value out = nb.create<arith::DivFOp>(loc, oo, l);
      Value gr = nb.create<arith::AddIOp>(loc, rowBlk, r);
      SmallVector<Value> oIdx = batched ? SmallVector<Value>{by, gr, c}
                                        : SmallVector<Value>{gr, c};
      nb.create<memref::StoreOp>(loc, out, oMem, oIdx);
    });

    // ---- split-K LSE epilogue: LSE[r] = m[r] + log(l[r]) ---------------- //
    if (hasLSE) {
      Value isRow = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx,
                                            cBM);
      b.create<scf::IfOp>(loc, isRow, [&](OpBuilder &nb, Location nl) {
        Value m = nb.create<memref::LoadOp>(nl, mSh, ValueRange{tx});
        Value l = nb.create<memref::LoadOp>(nl, lSh, ValueRange{tx});
        Value logl = nb.create<math::LogOp>(nl, l);
        Value lse = nb.create<arith::AddFOp>(nl, m, logl);
        Value gr = nb.create<arith::AddIOp>(nl, rowBlk, tx);
        Value c0i = nb.create<arith::ConstantIndexOp>(nl, 0);
        SmallVector<Value> lseIdx = batched
                                        ? SmallVector<Value>{by, gr, c0i}
                                        : SmallVector<Value>{gr, c0i};
        nb.create<memref::StoreOp>(nl, lse, lseMem, lseIdx);
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

struct ConvertIntentLoopContractionToGPUPass
    : ::intent::impl::ConvertIntentLoopContractionToGPUBase<
          ConvertIntentLoopContractionToGPUPass> {
  using ConvertIntentLoopContractionToGPUBase::
      ConvertIntentLoopContractionToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      FusedAttnLoopLowering probe(fn);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      FusedAttnLoopLowering low(fn);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("fused-attn loop lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
