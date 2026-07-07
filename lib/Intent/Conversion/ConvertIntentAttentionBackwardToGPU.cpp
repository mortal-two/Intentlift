//===- ConvertIntentAttentionBackwardToGPU.cpp - flash bwd -> gpu ---------===//
//
// Lowers a func whose body is the FlashAttention BACKWARD as one `intent.loop`
// scanning the QUERY axis (scan inputs Q, dO, LSE, Delta; carries the per-key-row
// gradients dk[Sk,D], dv[Sk,D], init zeros) to ONE tiled tensor-core
// `gpu.launch`. N4: the 5 matmuls run on REGISTER-RESIDENT nvgpu.mma.sync
// m16n8k16 (the forward pass's proven recipe), NOT WMMA wmma.load.
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
// THE N4 REGISTER-RESIDENT PORT (breaking the flash-bwd wall):
//  * The dV/dK accumulators live in REGISTERS -- carried across the inner query
//    loop as scf.for iter_args (freeing the ~32KB dvSh/dkSh shared tiles).
//  * S, dP, dS and the online-softmax recompute run IN REGISTERS (like the fwd);
//    only P (then dS, overwriting it) is published to a single [BM,BN] f16 shared
//    scratch (Pds) -- REQUIRED there because dV/dK need P^T / dS^T, a cross-lane
//    [q,*]->[*,q] transpose the mma.sync A/B operands cannot form from registers.
//  * dV=P^T@dO and dK=dS^T@Q reduce over the QUERY axis; BOTH their operands are
//    loaded TRANSPOSED DIRECTLY from the untransposed Pds/dO/Q tiles via
//    nvgpu.ldmatrix transpose=true (the ldmatrix.trans PTX) -- NO separate
//    transpose materialization, so the smem budget stays ~65KB << the 99KB cap.
//  * dQ=dS@K needs only B=K^T transposed (ldmatrix.trans of kSh).
//  * K/V (per key tile) and Q/dO (per query tile) stage via cp.async.
//
// Warp layout (8 warps, m16n8k16): the q-outputs (S/dP/dQ, M=BM=128) use 8 row-
// strips of 16 rows; the k-outputs (dV/dK, M=BN=64) use 4 row-strips x 2
// column-groups. Per query tile:
//   1. S  = Q@Kj^T via mma.sync -> registers.
//   2. P  = exp(scale*S - LSE_row) in registers ; publish (f16) to Pds.
//   3. dV += P^T@dO via mma.sync (ldmatrix.trans P, dO) -> dV registers.
//   4. dP = dO@Vj^T via mma.sync -> registers.
//   5. dS = P*(dP - Delta_row)*scale in registers ; publish (f16) to Pds.
//   6. dK += dS^T@Q via mma.sync (ldmatrix.trans dS, Q) -> dK registers.
//   7. dQ = dS@Kj via mma.sync (ldmatrix.trans K) ; RMW into global DQ.
//
// SERIAL DQ (SEQUENCE_PARALLEL=false): one block per (b,h); OUTER scf.for over
// key tiles wrapping the INNER scf.for over query tiles; dQ read-modify-written
// into pre-zeroed global DQ. ATOMIC (SEQUENCE_PARALLEL=true): grid.x=cdiv(Sk,BN),
// dQ via memref.atomic_rmw addf (the serial==atomic gate stays exact).
//
// Q/K/V/dO f16, LSE/Delta/scale + accumulators f32. Static shapes, BM=128, BN=64,
// D%16==0, Sq%128==0, Sk%64==0 (non-causal, no bias). Bare-ptr ABI; the lowered
// module goes to lower_memref_module_to_kernels with scf_to_cf=True. The pass
// appends three destination outputs dQ, dK, dV.
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
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

// Tile geometry. BM/BN = the block's query/key tile (64x64). N4: the 5 bwd
// matmuls run on register-resident nvgpu.mma.sync m16n8k16 (the fwd's proven
// recipe) instead of WMMA wmma.load; 4 warps, each owning a 16-row M strip.
// kWMMA aliases MMA_K for the match() divisibility check.
// BM=128 (Triton's bwd BLOCK_M): the block scans 128 query rows per inner tile,
// halving the query-tile count -> half the barrier/loop overhead. BN=64 key tile.
// The q-outputs (S, dP, dQ) are [128,64] -> 8 warps as 8 row-strips (16 rows,
// full 64 cols). The k-outputs (dV, dK) are [64,64] -> 8 warps as 4 row-strips x
// 2 column-groups (16 k-rows, 32 d-cols). This asymmetry is the BM!=BN cost.
static constexpr int64_t BM = 128, BN = 64;
static constexpr int64_t MMA_M = 16, MMA_N = 8, MMA_K = 16;
static constexpr int64_t kWMMA = MMA_K;

static int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

/// One flash-bwd func's tensor-core lowering. Reads the original (tensor) func
/// body (two zero splats + one intent.loop + return) and builds a new
/// memref+gpu.launch func with the same name, erasing the old.
struct FlashBwdLowering {
  func::FuncOp fn;
  std::string dqMode = "serial"; // "serial" | "atomic" (set by the pass option)

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

  // ---- emission (N4: register-resident nvgpu.mma.sync) ------------------- //
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

    int64_t numKT = Sk / BN;           // outer key tiles
    int64_t numQT = Sq / BM;           // inner query tiles
    // 8 warps. q-outputs (M=BM=128): 8 row-strips, each 16 rows x full N. The
    // k-outputs dV/dK (M=BN=64): 4 row-strips x 2 column-groups (kNCG=2), each
    // 16 k-rows x half the d-columns -- carried register-resident.
    int64_t numWarps = BM / MMA_M;     // 8
    int64_t kBlock = numWarps * 32;    // 256 threads
    int64_t kNCG = 2;                  // dV/dK column groups
    int64_t nColS = BN / MMA_N;        // S/dP n8 column tiles (over k) = 8
    int64_t nColD = D / MMA_N;         // dV/dK/dQ n8 column tiles (over d)
    int64_t hColK = nColD / kNCG;      // per-warp dV/dK column tiles
    int64_t kSubD = D / MMA_K;         // contraction over d (S, dP)
    int64_t kSubQ = BM / MMA_K;        // contraction over query (dV, dK) = 8
    int64_t kSubK = BN / MMA_K;        // contraction over key (dQ)

    bool atomicDQ = (dqMode == "atomic");

    // Dynamic-shared layout. N4: the dV/dK accumulators are now REGISTER-RESIDENT
    // (scf.for iter_args across the inner query loop). N4b: S/dP/dS are ALSO
    // register-resident now (the online-softmax recompute + dS run in registers,
    // like the forward), so the 32KB f32 Sd scratch is gone; only Pds[BM,BN] f16
    // remains -- P (then dS) MUST be materialized there for the dV/dK transpose
    // ldmatrix (a cross-lane [q,*]->[*,q] transpose). LSE/Delta are read straight
    // from global per lane (2 rows each). K/V key tile resident across the query
    // loop; Q/dO per query tile. Total ~65KB, well within the 99KB cap.
    int64_t oK = 0;
    int64_t oV = align16(oK + BN * D * 2);
    int64_t oQ = align16(oV + BN * D * 2);
    int64_t oDO = align16(oQ + BM * D * 2);
    int64_t oPds = align16(oDO + BM * D * 2);
    int64_t shTotal = align16(oPds + BM * BN * 2);

    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value shBytes =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shTotal));

    // grid.x: serial = 1 (ONE block per (b,h) owns the whole head, the outer key
    // loop is an scf.for); atomic = numKT (one PROGRAM per key block, parallel
    // over key tiles, dQ accumulated cross-program via atomic_rmw addf).
    Value gridX = atomicDQ ? b.create<arith::ConstantIndexOp>(loc, numKT) : c1;
    auto launch =
        b.create<gpu::LaunchOp>(loc, gridX, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cBMv = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBNv = b.create<arith::ConstantIndexOp>(loc, BN);

    // Lane/warp decomposition (the fwd/N6 m16n8k16 recipe). TWO warp layouts:
    //  q-layout (S/dP/dQ, M=128): warpRowQ = warp*16 (8 strips), full N columns.
    //  k-layout (dV/dK, M=64):    warpRowK = (warp/2)*16 (4 strips), colGroupK =
    //                             warp%2 owns d-columns [colGroupK*hColK, +hColK).
    Value cNCG = b.create<arith::ConstantIndexOp>(loc, kNCG);
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = b.create<arith::RemUIOp>(loc, tx, c32);
    Value warpRowQ = b.create<arith::MulIOp>(loc, warp, c16); // q-strip base
    Value kStrip = b.create<arith::DivUIOp>(loc, warp, cNCG);
    Value colGroupK = b.create<arith::RemUIOp>(loc, warp, cNCG);
    Value warpRowK = b.create<arith::MulIOp>(loc, kStrip, c16); // k-strip base
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);
    Value laneDiv16 = b.create<arith::DivUIOp>(loc, lane, c16);
    Value aColLane = b.create<arith::MulIOp>(loc, laneDiv16, c8); // (lane/16)*8
    Value aRowQ = b.create<arith::AddIOp>(loc, warpRowQ, laneMod16);
    Value laneMod8 = b.create<arith::RemUIOp>(loc, lane, c8);
    Value laneDiv8 = b.create<arith::DivUIOp>(loc, lane, c8);
    Value bColLane = b.create<arith::MulIOp>(loc, laneDiv8, c8); // (lane/8)*8
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value stCol = b.create<arith::MulIOp>(loc, laneMod4, c2); // (lane%4)*2
    // Transpose-A lane offsets (ldmatrix.trans, numTiles=4): a lane reads 8
    // contiguous M elements of source row kk = kkBase + (lane/16)*8 + lane%8, at
    // column (the M base) warpRow16 + ((lane/8)%2)*8. The .trans op delivers the
    // 8x8 transposed, so the fragment is [m,kk] as mma.sync's A operand expects.
    Value aTKk = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, laneDiv16, c8), laneMod8);
    Value laneDiv8Mod2 = b.create<arith::RemUIOp>(loc, laneDiv8, c2);
    // Transpose-A M base uses the k-layout (dV/dK output rows are k).
    Value aTCol = b.create<arith::AddIOp>(
        loc, warpRowK, b.create<arith::MulIOp>(loc, laneDiv8Mod2, c8));

    // Shared views.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto viewF16 = [&](int64_t off, ArrayRef<int64_t> shp) -> Value {
      auto ty = MemRefType::get(shp, f16, nullptr, wgSpace);
      Value o = b.create<arith::ConstantIndexOp>(loc, off);
      return b.create<memref::ViewOp>(loc, ty, raw, o, ValueRange{});
    };
    Value kSh = viewF16(oK, {BN, D});
    Value vSh = viewF16(oV, {BN, D});
    Value qSh = viewF16(oQ, {BM, D});
    Value doSh = viewF16(oDO, {BM, D});
    Value pdsSh = viewF16(oPds, {BM, BN});

    // mma.sync fragment types + the m16n8k16 shape attr.
    auto aFragTy = VectorType::get({4, 2}, f16);
    auto bFragTy = VectorType::get({2, 2}, f16);
    auto accTy = VectorType::get({2, 2}, f32);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));

    // Vectorized global->shared staging via cp.async (16-byte/8-f16 cg copies,
    // L1 bypassed) -- far fewer/wider transactions than the scalar cooperative
    // loads. Issues R*(C/8) copies of two f16 tensors (src0->dst0, src1->dst1)
    // sharing the same [R,C] tile shape and global row base, groups + waits +
    // barriers them. Used for the K/V (per key tile) and Q/dO (per query tile)
    // stages -- the tiles that were previously a serial scalar-load bottleneck.
    auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr dstElems = b.getIndexAttr(8);
    UnitAttr bypassL1 = b.getUnitAttr();
    auto cpAsyncPair = [&](OpBuilder &nb, int64_t R, int64_t C, Value rowBaseG,
                           Value src0, Value dst0, Value src1, Value dst1) {
      Value cCPR = nb.create<arith::ConstantIndexOp>(loc, C / 8);
      int64_t vPerThread = (R * (C / 8)) / kBlock;
      SmallVector<Value> toks;
      for (int64_t it = 0; it < vPerThread; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value vt = nb.create<arith::AddIOp>(loc, tx, base);
        Value r = nb.create<arith::DivUIOp>(loc, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(loc, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(loc, colg, c8);
        Value gr = nb.create<arith::AddIOp>(loc, rowBaseG, r);
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            loc, tokTy, dst0, ValueRange{r, cc}, src0, ValueRange{gr, cc},
            dstElems, /*srcElements=*/Value(), bypassL1));
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            loc, tokTy, dst1, ValueRange{r, cc}, src1, ValueRange{gr, cc},
            dstElems, /*srcElements=*/Value(), bypassL1));
      }
      Value grp = nb.create<nvgpu::DeviceAsyncCreateGroupOp>(loc, tokTy, toks);
      nb.create<nvgpu::DeviceAsyncWaitOp>(loc, grp, b.getI32IntegerAttr(0));
    };

    // ---- ldmatrix fragment loaders (the four operand layouts) ------------ //
    // A (16x16, [m,kk], kk contiguous), non-transpose. Used by the q-outputs
    // (S/dP/dQ), so the M rows are the q-layout strip (aRowQ).
    auto ldA = [&](OpBuilder &nb, Value src, Value kkBase) -> Value {
      Value col = nb.create<arith::AddIOp>(loc, kkBase, aColLane);
      return nb.create<nvgpu::LdMatrixOp>(loc, aFragTy, src,
                                          ValueRange{aRowQ, col},
                                          /*transpose=*/false, /*numTiles=*/4);
    };
    // A from a source stored [kk,m] (m contiguous): ldmatrix.trans -> [m,kk].
    auto ldAT = [&](OpBuilder &nb, Value src, Value kkBase) -> Value {
      Value row = nb.create<arith::AddIOp>(loc, kkBase, aTKk);
      return nb.create<nvgpu::LdMatrixOp>(loc, aFragTy, src,
                                          ValueRange{row, aTCol},
                                          /*transpose=*/true, /*numTiles=*/4);
    };
    // B (8x16, stored [n,kk], kk contiguous), non-transpose.
    auto ldB = [&](OpBuilder &nb, Value src, Value nBase, Value kkBase) -> Value {
      Value row = nb.create<arith::AddIOp>(loc, nBase, laneMod8);
      Value col = nb.create<arith::AddIOp>(loc, kkBase, bColLane);
      return nb.create<nvgpu::LdMatrixOp>(loc, bFragTy, src,
                                          ValueRange{row, col},
                                          /*transpose=*/false, /*numTiles=*/2);
    };
    // B from a source stored [kk,n] (n contiguous): ldmatrix.trans -> [n,kk].
    // A lane reads 8 contiguous n at source row kk = kkBase + lane%16, col nBase.
    auto ldBT = [&](OpBuilder &nb, Value src, Value nBase, Value kkBase) -> Value {
      Value row = nb.create<arith::AddIOp>(loc, kkBase, laneMod16);
      return nb.create<nvgpu::LdMatrixOp>(loc, bFragTy, src,
                                          ValueRange{row, nBase},
                                          /*transpose=*/true, /*numTiles=*/2);
    };

    // Generic warp mma: acc[j] += sum_ks A(ks) * B(j,ks), m16n8k16, over the
    // nFrag n8 column tiles and kSub k16 subtiles. loadA/loadB return the
    // ldmatrix fragments for a given (ks) / (j,ks). `seed` seeds the accumulators
    // (zero for a fresh product, or the carried dV/dK for a reduction).
    auto mmaCompute =
        [&](OpBuilder &nb, ArrayRef<Value> seed, int64_t nFrag, int64_t kSub,
            function_ref<Value(OpBuilder &, int64_t)> loadA,
            function_ref<Value(OpBuilder &, int64_t, int64_t)> loadB)
        -> SmallVector<Value> {
      SmallVector<Value> acc(seed.begin(), seed.end());
      for (int64_t ks = 0; ks < kSub; ++ks) {
        Value aFrag = loadA(nb, ks);
        for (int64_t j = 0; j < nFrag; ++j) {
          Value bFrag = loadB(nb, j, ks);
          acc[j] =
              nb.create<nvgpu::MmaSyncOp>(loc, aFrag, bFrag, acc[j], mmaShape);
        }
      }
      return acc;
    };

    // Store a warp's [16, hCol*8] accumulator (its column group, based at
    // colBaseElem). mode 0 = plain store; 1 = serial read-modify-write add; 2 =
    // atomic_rmw addf. Textbook m16n8 layout:
    //   d[i][jj] -> row = rowBaseG + lane/4 + i*8,
    //               col = colBaseElem + j*8 + (lane%4)*2 + jj.
    auto storeAcc = [&](OpBuilder &nb, ArrayRef<Value> acc, int64_t hCol,
                        Value rowBaseG, Value colBaseElem, Value mem, int mode) {
      for (int64_t j = 0; j < hCol; ++j) {
        Value nBase = nb.create<arith::AddIOp>(
            loc, colBaseElem, nb.create<arith::ConstantIndexOp>(loc, j * MMA_N));
        for (int i = 0; i < 2; ++i) {
          Value ic = nb.create<arith::ConstantIndexOp>(loc, i * 8);
          Value row = nb.create<arith::AddIOp>(
              loc, rowBaseG, nb.create<arith::AddIOp>(loc, laneDiv4, ic));
          for (int jj = 0; jj < 2; ++jj) {
            Value jc = nb.create<arith::ConstantIndexOp>(loc, jj);
            Value col = nb.create<arith::AddIOp>(
                loc, nBase, nb.create<arith::AddIOp>(loc, stCol, jc));
            Value v = nb.create<vector::ExtractOp>(
                loc, acc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
            if (mode == 2) {
              nb.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v,
                                             mem, ValueRange{row, col});
            } else if (mode == 1) {
              Value cur =
                  nb.create<memref::LoadOp>(loc, mem, ValueRange{row, col});
              Value nq = nb.create<arith::AddFOp>(loc, cur, v);
              nb.create<memref::StoreOp>(loc, nq, mem, ValueRange{row, col});
            } else {
              nb.create<memref::StoreOp>(loc, v, mem, ValueRange{row, col});
            }
          }
        }
      }
    };

    // Store a q-output f32 accumulator (S/dP layout: 16 q-rows x full N) into a
    // shared f16 scratch (row = warpRowQ+lane/4+i*8, col = j*8+(lane%4)*2+jj),
    // truncating to f16. Used to publish P / dS to pdsSh for the transpose loads.
    auto storeAccF16 = [&](OpBuilder &nb, ArrayRef<Value> acc, Value dst) {
      for (int64_t j = 0; j < (int64_t)acc.size(); ++j) {
        Value nBase = nb.create<arith::ConstantIndexOp>(loc, j * MMA_N);
        for (int i = 0; i < 2; ++i) {
          Value ic = nb.create<arith::ConstantIndexOp>(loc, i * 8);
          Value row = nb.create<arith::AddIOp>(
              loc, warpRowQ, nb.create<arith::AddIOp>(loc, laneDiv4, ic));
          for (int jj = 0; jj < 2; ++jj) {
            Value jc = nb.create<arith::ConstantIndexOp>(loc, jj);
            Value col = nb.create<arith::AddIOp>(
                loc, nBase, nb.create<arith::AddIOp>(loc, stCol, jc));
            Value v = nb.create<vector::ExtractOp>(
                loc, acc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
            Value vh = nb.create<arith::TruncFOp>(loc, f16, v);
            nb.create<memref::StoreOp>(loc, vh, dst, ValueRange{row, col});
          }
        }
      }
    };

    // Register online-softmax recompute: P = exp(scale*S - LSE_row), per q-output
    // vector (LSE is per q-row; each lane owns rows i=0,1 -> lse0/lse1). f32.
    auto computeP = [&](OpBuilder &nb, ArrayRef<Value> sAcc, Value lse0,
                        Value lse1) -> SmallVector<Value> {
      SmallVector<Value> p;
      for (int64_t j = 0; j < (int64_t)sAcc.size(); ++j) {
        Value v = sAcc[j];
        for (int i = 0; i < 2; ++i) {
          Value lse = i == 0 ? lse0 : lse1;
          for (int jj = 0; jj < 2; ++jj) {
            Value s = nb.create<vector::ExtractOp>(
                loc, sAcc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
            Value ss = nb.create<arith::MulFOp>(loc, s, scale);
            Value sm = nb.create<arith::SubFOp>(loc, ss, lse);
            Value e = nb.create<math::ExpOp>(loc, sm);
            v = nb.create<vector::InsertOp>(
                loc, e, v, ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
          }
        }
        p.push_back(v);
      }
      return p;
    };

    // Register dS = P*(dP - Delta_row)*scale, per q-output vector (Delta per
    // q-row -> d0/d1). f32.
    auto computeDS = [&](OpBuilder &nb, ArrayRef<Value> pAcc,
                         ArrayRef<Value> dpAcc, Value d0,
                         Value d1) -> SmallVector<Value> {
      SmallVector<Value> ds;
      for (int64_t j = 0; j < (int64_t)pAcc.size(); ++j) {
        Value v = dpAcc[j];
        for (int i = 0; i < 2; ++i) {
          Value dl = i == 0 ? d0 : d1;
          for (int jj = 0; jj < 2; ++jj) {
            Value dp = nb.create<vector::ExtractOp>(
                loc, dpAcc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
            Value pp = nb.create<vector::ExtractOp>(
                loc, pAcc[j], ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
            Value diff = nb.create<arith::SubFOp>(loc, dp, dl);
            Value m = nb.create<arith::MulFOp>(loc, pp, diff);
            Value r = nb.create<arith::MulFOp>(loc, m, scale);
            v = nb.create<vector::InsertOp>(
                loc, r, v, ArrayRef<int64_t>{(int64_t)i, (int64_t)jj});
          }
        }
        ds.push_back(v);
      }
      return ds;
    };

    // q-outputs (S/dP/dQ) use the full N columns (colBase 0). dV/dK use the
    // k-layout column group base (colGroupK * hColK * 8).
    Value colBaseK = b.create<arith::MulIOp>(
        loc, colGroupK, b.create<arith::ConstantIndexOp>(loc, hColK * MMA_N));

    // ---- per-key-tile body --------------------------------------------- //
    auto emitKeyTile = [&](OpBuilder &ob, Value keyBase) {
      // Load Kj, Vj -> shared (cp.async).
      cpAsyncPair(ob, BN, D, keyBase, kMem, kSh, vMem, vSh);
      ob.create<gpu::BarrierOp>(loc);

      // ---- INNER loop over query tiles; dV/dK accumulators carried in
      // registers as scf.for iter_args: [dvAcc(hColK), dkAcc(hColK)]. -------- //
      Value cNumQT = ob.create<arith::ConstantIndexOp>(loc, numQT);
      SmallVector<Value> iterInit(2 * hColK, zeroAcc);
      auto qmLoop = ob.create<scf::ForOp>(
          loc, c0, cNumQT, c1, iterInit,
          [&](OpBuilder &lb, Location nl, Value qm, ValueRange args) {
            Value rowBase = lb.create<arith::MulIOp>(loc, qm, cBMv);

            // Load Qt, dOt -> shared (cp.async). LSE/Delta are read from global
            // per lane (its 2 q-rows) in the register recompute below.
            cpAsyncPair(lb, BM, D, rowBase, qMem, qSh, doMem, doSh);
            lb.create<gpu::BarrierOp>(loc); // B0: Q/dO visible for step-1 ldmatrix

            SmallVector<Value> zeroS(nColS, zeroAcc);  // q-outputs S/dP (full N)
            SmallVector<Value> zeroDq(nColD, zeroAcc); // dQ (full d columns)
            auto colFull = [&](OpBuilder &nb, int64_t j) {
              return nb.create<arith::ConstantIndexOp>(loc, j * MMA_N);
            };
            auto colKgrp = [&](OpBuilder &nb, int64_t j) {
              return nb.create<arith::AddIOp>(
                  loc, colBaseK,
                  nb.create<arith::ConstantIndexOp>(loc, j * MMA_N));
            };
            auto ksc = [&](OpBuilder &nb, int64_t ks) {
              return nb.create<arith::ConstantIndexOp>(loc, ks * MMA_K);
            };

            // Per-lane LSE/Delta for its two owned q-rows (i=0,1):
            //   qGlobal_i = rowBase + warpRowQ + lane/4 + i*8.
            Value c0i = lb.create<arith::ConstantIndexOp>(loc, 0);
            Value qRowL = lb.create<arith::AddIOp>(
                loc, lb.create<arith::AddIOp>(loc, rowBase, warpRowQ), laneDiv4);
            Value lse0 = lb.create<memref::LoadOp>(loc, lseMem,
                                                   ValueRange{qRowL, c0i});
            Value del0 = lb.create<memref::LoadOp>(loc, deltaMem,
                                                   ValueRange{qRowL, c0i});
            Value qRowL1 = lb.create<arith::AddIOp>(
                loc, qRowL, lb.create<arith::ConstantIndexOp>(loc, 8));
            Value lse1 = lb.create<memref::LoadOp>(loc, lseMem,
                                                   ValueRange{qRowL1, c0i});
            Value del1 = lb.create<memref::LoadOp>(loc, deltaMem,
                                                   ValueRange{qRowL1, c0i});

            // (1) S = Qt@Kj^T -> registers. A=Q[q,d], B=K[k,d] (d contiguous).
            SmallVector<Value> sAcc = mmaCompute(
                lb, zeroS, nColS, kSubD,
                [&](OpBuilder &nb, int64_t ks) {
                  return ldA(nb, qSh, ksc(nb, ks));
                },
                [&](OpBuilder &nb, int64_t j, int64_t ks) {
                  return ldB(nb, kSh, colFull(nb, j), ksc(nb, ks));
                });

            // (2) P = exp(scale*S - LSE) in registers; publish to pdsSh (f16)
            //     for the dV transpose load.
            SmallVector<Value> pAcc = computeP(lb, sAcc, lse0, lse1);
            storeAccF16(lb, pAcc, pdsSh);
            lb.create<gpu::BarrierOp>(loc); // B1: P visible in pdsSh

            // (3) dV += P^T@dO. out[k,d]=sum_q P[q,k]*dO[q,d]. A=P^T (ldmatrix.
            //     trans of Pds[q,k]), B=dO^T (ldmatrix.trans of doSh[q,d]).
            SmallVector<Value> dvIn(args.begin(), args.begin() + hColK);
            SmallVector<Value> dvAcc = mmaCompute(
                lb, dvIn, hColK, kSubQ,
                [&](OpBuilder &nb, int64_t ks) {
                  return ldAT(nb, pdsSh, ksc(nb, ks));
                },
                [&](OpBuilder &nb, int64_t j, int64_t ks) {
                  return ldBT(nb, doSh, colKgrp(nb, j), ksc(nb, ks));
                });

            // (4) dP = dO@Vj^T -> registers (never leaves registers). A=dO[q,d],
            //     B=V[k,d] (d contiguous).
            SmallVector<Value> dpAcc = mmaCompute(
                lb, zeroS, nColS, kSubD,
                [&](OpBuilder &nb, int64_t ks) {
                  return ldA(nb, doSh, ksc(nb, ks));
                },
                [&](OpBuilder &nb, int64_t j, int64_t ks) {
                  return ldB(nb, vSh, colFull(nb, j), ksc(nb, ks));
                });

            // (5) dS = P*(dP - Delta)*scale in registers; publish to pdsSh (f16),
            //     overwriting P. B2 guards dV's ldmatrix reads of P before this.
            SmallVector<Value> dsAcc = computeDS(lb, pAcc, dpAcc, del0, del1);
            lb.create<gpu::BarrierOp>(loc); // B2: dV done reading Pds; safe to reuse
            storeAccF16(lb, dsAcc, pdsSh);
            lb.create<gpu::BarrierOp>(loc); // B3: dS visible in pdsSh

            // (6) dK += dS^T@Q. out[k,d]=sum_q dS[q,k]*Q[q,d]. A=dS^T (ldmatrix.
            //     trans of Pds[q,k]), B=Q^T (ldmatrix.trans of qSh[q,d]).
            SmallVector<Value> dkIn(args.begin() + hColK, args.end());
            SmallVector<Value> dkAcc = mmaCompute(
                lb, dkIn, hColK, kSubQ,
                [&](OpBuilder &nb, int64_t ks) {
                  return ldAT(nb, pdsSh, ksc(nb, ks));
                },
                [&](OpBuilder &nb, int64_t j, int64_t ks) {
                  return ldBT(nb, qSh, colKgrp(nb, j), ksc(nb, ks));
                });

            // (7) dQ = dS@Kj. out[q,d]=sum_k dS[q,k]*K[k,d]. A=dS[q,k] (k
            //     contiguous), B=K^T (ldmatrix.trans of kSh[k,d]). RMW into DQ.
            SmallVector<Value> dqAcc = mmaCompute(
                lb, zeroDq, nColD, kSubK,
                [&](OpBuilder &nb, int64_t ks) {
                  return ldA(nb, pdsSh, ksc(nb, ks));
                },
                [&](OpBuilder &nb, int64_t j, int64_t ks) {
                  return ldBT(nb, kSh, colFull(nb, j), ksc(nb, ks));
                });
            Value dqRowBase = lb.create<arith::AddIOp>(loc, rowBase, warpRowQ);
            storeAcc(lb, dqAcc, nColD, dqRowBase, c0, dqMem, atomicDQ ? 2 : 1);

            // B4: before the next iteration's cp.async overwrites qSh/doSh (step6
            // read qSh, steps 3/4 read doSh) and before pdsSh is reused.
            lb.create<gpu::BarrierOp>(loc);

            SmallVector<Value> res(dvAcc.begin(), dvAcc.end());
            res.append(dkAcc.begin(), dkAcc.end());
            lb.create<scf::YieldOp>(loc, res);
          });

      // Write dV/dK register accumulators -> DV/DK[keyBase] global. Each warp
      // owns a disjoint 16-row k-strip of the key block (no atomics needed).
      SmallVector<Value> dvFinal(qmLoop.getResults().begin(),
                                 qmLoop.getResults().begin() + hColK);
      SmallVector<Value> dkFinal(qmLoop.getResults().begin() + hColK,
                                 qmLoop.getResults().end());
      Value keyRowBase = ob.create<arith::AddIOp>(loc, keyBase, warpRowK);
      storeAcc(ob, dvFinal, hColK, keyRowBase, colBaseK, dvMem, 0);
      storeAcc(ob, dkFinal, hColK, keyRowBase, colBaseK, dkMem, 0);
      ob.create<gpu::BarrierOp>(loc); // shared tiles free before next key tile
    };

    // ---- dispatch: serial outer scf.for vs atomic one-program-per-key-block //
    if (atomicDQ) {
      Value kn = launch.getBlockIds().x; // key tile = this program's block id
      Value keyBase = b.create<arith::MulIOp>(loc, kn, cBNv);
      emitKeyTile(b, keyBase);
    } else {
      Value cNumKT = b.create<arith::ConstantIndexOp>(loc, numKT);
      auto knLoop = b.create<scf::ForOp>(loc, c0, cNumKT, c1);
      OpBuilder ob(knLoop.getBody(), knLoop.getBody()->begin());
      Value kn = knLoop.getInductionVar();
      Value keyBase = ob.create<arith::MulIOp>(loc, kn, cBNv);
      emitKeyTile(ob, keyBase);
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
    if (this->dqMode != "serial" && this->dqMode != "atomic") {
      mod.emitError("flash-bwd dq-mode must be 'serial' or 'atomic', got '")
          << this->dqMode << "'";
      return signalPassFailure();
    }
    for (func::FuncOp fn : targets) {
      FlashBwdLowering low(fn);
      low.dqMode = this->dqMode;
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("flash-bwd lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
