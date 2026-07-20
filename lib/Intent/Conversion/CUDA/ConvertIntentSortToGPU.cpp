//===- ConvertIntentSortToGPU.cpp - sort/topk/argsort -> gpu.launch -------===//
//
// The C6 HARD-tier order-statistics pass. Lowers a func whose body is a single
// `intent.sort` over a rank-2 `[M,N]` static-f32 row (axis 1) to ONE
// block-cooperative `gpu.launch`, ONE BLOCK PER ROW. It carries TWO single-block
// lowerings and picks between them per shape by a calibrated cost model
// (see match()):
//
//   * BITONIC (rewriteBitonic): the O(P log^2 P) compare-exchange network over
//     P = next_pow2(N) sentinel-padded elements. Tiny per-op constant, so it
//     WINS whenever N is at/near a power of two — measured on sm_120 at
//     N=2048/4096 ~1.2x torch and N=8192 (pow2) 0.52x (beats torch CUB ~2x).
//     Its weakness is the pow2 padding: a row just past a power of two nearly
//     doubles P and its work.
//   * MATCH RADIX (rewriteRadixMatch): the CUB-style WARP-MATCH REGISTER-RANKING
//     LSD radix. O(8N) and immune to padding like the legacy radix, but the
//     per-4-bit-digit ranking is done ENTIRELY in registers — no shared
//     histogram at all. Each thread holds C = ceil(N/T) keys; per digit the warp
//     peer-mask of same-digit lanes is built from kBits `nvvm.vote.ballot.sync`
//     and ranked with `math.ctpop`, and a tiny numWarps*kDigits cross-warp table
//     (one exclusive scan) replaces the kDigits*T histogram sweep + count RMW.
//     That removes the bottleneck that capped the legacy radix at ~6-9% HBM ->
//     bandwidth-bound. It now WINS the power-of-two GATE too (measured on
//     sm_120a: [4096,2048] 0.61x, [4096,4096] 0.99x vs torch CUB — was bitonic
//     1.17x / 1.25x). Selected whenever it fits (T a whole number of warps,
//     C<=32); the legacy `rewriteRadix` is the fallback when it does not.
//   * RADIX (rewriteRadix): the legacy per-row SEGMENTED LSD RADIX — the CUB
//     `DeviceSegmentedRadixSort` shape with a per-thread shared histogram column
//     + warp-shuffle block scan + scatter. Fallback for shapes the match path
//     cannot fit (kept for large rows: N=9000 0.68x, N=10240 0.95x vs torch).
//   IEEE-f32 -> order-preserving UNSIGNED key (all paths sort ascending): flip
//   `key = bits ^ (sign ? 0xFFFFFFFF : 0x80000000)` (negatives reverse, sit
//   below positives), inverted on output.
//
// RADIX mechanism (per row, in dynamic shared memory):
//   (0) LOAD (coalesced, strided i in [tx,N) step T): flip the row into a `[N]`
//       i32 key buffer keyA; for argsort a parallel `[N]` i32 index buffer.
//   (1) 8 LSD PASSES over 4-bit digits, ping-ponging keyA<->keyB. Each pass over
//       BLOCKED chunks of T threads:
//         count:   each thread tallies its chunk's digits into its own shared
//                  column hist[d*T+tx] (RMW; no dynamically-indexed local mem).
//         scan:    exclusive prefix sum of hist[0..kDigits*T) in flat digit-
//                  major order (a per-thread segment scan + warp-shuffle block
//                  scan of the T segment totals) -> hist[d*T+tx] becomes the
//                  STABLE per-slot scatter offset directly.
//         scatter: each thread re-walks its chunk IN ORDER, emitting each key
//                  (and index) to hist[slot]++ -> STABLE, which makes LSD
//                  correct across passes.
//   (2) STORE (coalesced, strided w in [tx,W)): unflip keyA -> f32 values / i32
//       indices for the first W columns; descending reverse-reads srcW = N-1-w
//       (mirror of the ascending sort); topk is W<N.
//
// One block per row keeps all M rows resident across the ~170 SMs; both sorts
// run entirely in shared memory (global traffic is the minimal read-once/
// write-once). Outputs are one memref per USED result: `values` `[M,W]` f32 then
// (if returned) `indices` `[M,W]` i32.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdlib>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTSORTTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static int64_t nextPow2(int64_t n) {
  int64_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

// Radix: 4-bit digits -> 16 buckets, ceil(32/4) = 8 LSD passes over f32 keys.
static constexpr int64_t kBits = 4;
static constexpr int64_t kDigits = 1 << kBits; // 16
static constexpr int64_t kPasses = 32 / kBits; // 8

struct SortLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match()
  SortOp sort;
  int64_t M = 0, N = 0, W = 0, T = 0, C = 0, P = 0;
  bool descending = false;
  bool needValues = false, needIndices = false;
  bool useRadix = false;
  bool matchRadix = false; // CUB-style warp-match register-ranking radix
  Type f32, i32;

  SortLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    sort = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (auto s = dyn_cast<SortOp>(&op)) {
        if (sort)
          return failure(); // only one
        sort = s;
      } else {
        return failure();
      }
    }
    if (!sort)
      return failure();
    // The func returns only results of this sort (values and/or indices).
    for (Value r : ret.getOperands())
      if (r != sort.getValues() && r != sort.getIndices())
        return failure();
    // input must be the sole block arg.
    if (body.getNumArguments() != 1 ||
        body.getArgument(0) != sort.getInput())
      return failure();

    auto inTy = dyn_cast<RankedTensorType>(sort.getInput().getType());
    auto vTy = dyn_cast<RankedTensorType>(sort.getValues().getType());
    auto iTy = dyn_cast<RankedTensorType>(sort.getIndices().getType());
    if (!inTy || !vTy || !iTy || !inTy.hasStaticShape() ||
        !vTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 2 || !inTy.getElementType().isF32())
      return failure();
    if (sort.getAxis() != 1)
      return failure();
    if (!iTy.getElementType().isInteger(32))
      return failure();

    M = inTy.getDimSize(0);
    N = inTy.getDimSize(1);
    W = vTy.getDimSize(1);
    if (W <= 0 || W > N || vTy.getDimSize(0) != M)
      return failure();
    descending = sort.getDescending();
    needValues = !sort.getValues().use_empty();
    needIndices = !sort.getIndices().use_empty();
    if (!needValues && !needIndices)
      return failure();

    f32 = inTy.getElementType();
    i32 = IntegerType::get(fn.getContext(), 32);

    // --- ALGORITHM SELECTION ------------------------------------------------
    // Two single-block-per-row lowerings:
    //   * BITONIC: O(P log^2 P) network over P = next_pow2(N) padded elements.
    //     Very low per-op constant, so it WINS whenever N is at/near a power of
    //     two (measured: N=2048/4096 ~1.2x, N=8192 0.52x — beats torch 2x). Its
    //     weakness is the pow2 PADDING: a row just past a power of two nearly
    //     doubles P and its work.
    //   * RADIX: O(8N) LSD segmented radix (CUB-style), work-linear and immune
    //     to padding, but a ~20x higher per-element constant (shared histogram +
    //     scatter). WINS only for large rows with heavy bitonic padding waste
    //     (measured: N=10240 -> radix 0.95x beats torch, bitonic 1.71x loses).
    // Pick by a calibrated cost model: radix iff its estimated element-work is
    // below bitonic's AND its ping-pong buffers + histogram fit the sm_120
    // ~100 KiB dynamic-shared budget.

    // --- MATCH-RADIX geometry (the CUB warp-match register-ranking path) -----
    // T is a multiple of the 32-lane warp so every warp is full (the ballot
    // active-mask is -1); C = ceil(N/T) keys/thread are held IN REGISTERS across
    // the pass (the slot loop is fully unrolled, so C is a compile-time bound and
    // no dynamically-indexed local memory appears). 256 threads is the sweet
    // spot (8 warps, high occupancy); grow to 512/1024 only if C would exceed the
    // register budget. There is NO kDigits*T shared histogram -- ranking is done
    // with nvvm.vote.ballot.sync + math.ctpop in registers, and the only shared
    // scratch is a tiny numWarps*kDigits warp-digit table -- so the match path is
    // bandwidth-bound (its per-element constant is far below the old radix's).
    int64_t Tm;
    if (const char *e = getenv("RADIX_T")) {
      Tm = atoi(e);
    } else {
      // Measured sweet spot: ~16 keys/thread with a SMALL block (<=256 threads)
      // so many blocks stay resident per SM (high occupancy) while each thread
      // still carries enough register-resident work to hide the ballot latency.
      Tm = ((N / 16) + 31) / 32 * 32; // round up N/16 to a whole warp
      if (Tm > 256)
        Tm = 256;
    }
    Tm = (Tm / 32) * 32; // whole number of warps
    if (Tm < 32)
      Tm = 32;
    if (Tm > 1024)
      Tm = 1024;
    while ((N + Tm - 1) / Tm > 32 && Tm < 1024)
      Tm <<= 1; // keep C = items/thread within the register budget
    int64_t Cm = (N + Tm - 1) / Tm;
    int64_t numWarpsM = Tm / 32;
    int64_t matchSmem = (int64_t)(needIndices ? 4 : 2) * N * 4 +
                        numWarpsM * kDigits * 4; // keys/idx ping-pong + warp table
    bool matchValid = Cm <= 32 && matchSmem <= 101000;

    // --- legacy shared-histogram radix geometry (fallback when match is unfit) -
    int64_t Trdx = 64;
    if (const char *e = getenv("RADIX_T"))
      Trdx = atoi(e);
    while (Trdx > N)
      Trdx >>= 1;
    while (Trdx & (Trdx - 1))
      Trdx &= (Trdx - 1);
    if (Trdx < 1)
      Trdx = 1;
    int64_t numWarps = (Trdx + 31) / 32;
    int64_t radixSmem = (int64_t)(needIndices ? 4 : 2) * N * 4 +
                        kDigits * Trdx * 4 + Trdx * 4 + numWarps * 4;
    bool radixFits = matchValid || radixSmem <= 101000;

    // Bitonic thread count: power of two dividing P = next_pow2(N).
    P = nextPow2(N);
    int64_t Tbit = blockThreads ? (int64_t)blockThreads : autoThreads(P);
    bool bitonicValid =
        Tbit > 0 && (Tbit & (Tbit - 1)) == 0 && Tbit <= P && P % Tbit == 0;

    // Cost model (element-ops): bitonic = P * L(L+1)/2 stages; radix = 8N with a
    // per-op constant. The match radix's constant (kMatchOpCost) is far below the
    // legacy shared-histogram radix's (kRadixOpCost): register ballot-ranking
    // instead of a full kDigits*T histogram sweep + count RMW. This lowers the
    // radix<->bitonic crossover so the match path also wins the power-of-two GATE.
    int64_t L = 0;
    for (int64_t p = P; p > 1; p >>= 1)
      ++L;
    int64_t bitonicOps = P * (L * (L + 1) / 2);
    int64_t kRadixOpCost = matchValid ? 5 : 12;
    if (const char *e = getenv("RADIX_OP_COST"))
      kRadixOpCost = atoi(e);
    int64_t radixOps = kRadixOpCost * kPasses * N;
    useRadix = radixFits && bitonicOps > radixOps;
    if (const char *e = getenv("RADIX_MIN_N")) // test override: force by N
      useRadix = radixFits && N >= atoi(e);
    if (!useRadix && !bitonicValid) {
      if (!radixFits)
        return failure();
      useRadix = true; // bitonic geometry unusable but radix fits -> use radix.
    }

    if (useRadix) {
      matchRadix = matchValid;
      if (matchRadix) {
        T = Tm;
        C = Cm;
      } else {
        T = Trdx;
        C = (N + T - 1) / T;
      }
    } else {
      T = Tbit;
    }
    return success();
  }

  Value cIdx(OpBuilder &b, Location loc, int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v);
  }
  Value cI32(OpBuilder &b, Location loc, int32_t v) {
    return b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(v));
  }

  // f32 bit pattern -> order-preserving unsigned key.
  Value flip(OpBuilder &b, Location loc, Value valF32) {
    Value bits = b.create<arith::BitcastOp>(loc, i32, valF32);
    Value sign = b.create<arith::ShRSIOp>(loc, bits, cI32(b, loc, 31));
    Value mask = b.create<arith::OrIOp>(loc, sign,
                                        cI32(b, loc, (int32_t)0x80000000));
    return b.create<arith::XOrIOp>(loc, bits, mask);
  }
  // order-preserving unsigned key -> f32 bit pattern.
  Value unflip(OpBuilder &b, Location loc, Value key) {
    Value top = b.create<arith::ShRUIOp>(loc, key, cI32(b, loc, 31));
    Value tm = b.create<arith::SubIOp>(loc, top, cI32(b, loc, 1));
    Value mask =
        b.create<arith::OrIOp>(loc, tm, cI32(b, loc, (int32_t)0x80000000));
    Value bits = b.create<arith::XOrIOp>(loc, key, mask);
    return b.create<arith::BitcastOp>(loc, f32, bits);
  }
  // digit = (key >> shift) & (kDigits-1), returned as index.
  Value digitIdx(OpBuilder &b, Location loc, Value key, Value shift) {
    Value sh = b.create<arith::ShRUIOp>(loc, key, shift);
    Value d = b.create<arith::AndIOp>(loc, sh, cI32(b, loc, kDigits - 1));
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), d);
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    if (!useRadix)
      return rewriteBitonic();
    return matchRadix ? rewriteRadixMatch() : rewriteRadix();
  }

  // Build the new func signature (input + one memref per USED result) and the
  // one-block-per-row gpu.launch skeleton; returns the launch + IO handles.
  struct Skeleton {
    gpu::LaunchOp launch;
    Value inMem, valOut, idxOut, bx, tx, raw;
  };

  Skeleton buildSkeleton(OpBuilder &b, Location loc, int64_t shBytes) {
    MLIRContext *ctx = fn.getContext();
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    SmallVector<Type> inTys;
    inTys.push_back(MemRefType::get({M, N}, f32));
    if (needValues)
      inTys.push_back(MemRefType::get({M, W}, f32));
    if (needIndices)
      inTys.push_back(MemRefType::get({M, W}, i32));
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Skeleton s;
    s.inMem = entry->getArgument(0);
    unsigned argi = 1;
    s.valOut = needValues ? entry->getArgument(argi++) : Value();
    s.idxOut = needIndices ? entry->getArgument(argi++) : Value();
    Value cM = cIdx(b, loc, M), c1 = cIdx(b, loc, 1), cT = cIdx(b, loc, T);
    Value shBytesV =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shBytes));
    s.launch = b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesV);
    b.setInsertionPointToStart(&s.launch.getBody().front());
    s.bx = s.launch.getBlockIds().x;
    s.tx = s.launch.getThreadIds().x;
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    s.raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    return s;
  }

  // ---- BITONIC network (moderate-N winner) ------------------------------- //
  LogicalResult rewriteBitonic() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    int64_t shBytes = (int64_t)P * 4 + (needIndices ? (int64_t)P * 4 : 0);
    Skeleton s = buildSkeleton(b, loc, shBytes);
    Value inMem = s.inMem, valOut = s.valOut, idxOut = s.idxOut;
    Value bx = s.bx, tx = s.tx, raw = s.raw;

    Value cT = cIdx(b, loc, T), cP = cIdx(b, loc, P), cW = cIdx(b, loc, W);
    Value cN = cIdx(b, loc, N), c0i = cIdx(b, loc, 0);

    auto valTy = MemRefType::get({P}, f32, nullptr, wgSpace);
    auto idxTy = MemRefType::get({P}, i32, nullptr, wgSpace);
    Value valBuf =
        b.create<memref::ViewOp>(loc, valTy, raw, cIdx(b, loc, 0), ValueRange{});
    Value idxBuf;
    if (needIndices)
      idxBuf = b.create<memref::ViewOp>(loc, idxTy, raw, cIdx(b, loc, P * 4),
                                        ValueRange{});

    APFloat inf = APFloat::getInf(APFloat::IEEEsingle(), /*neg=*/descending);
    Value sentinel = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(inf.convertToFloat()));

    { // (0) LOAD row, sentinel-pad N..P-1; idx[i]=i.
      auto loop = b.create<scf::ForOp>(loc, tx, cP, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value i = loop.getInductionVar();
      Value isReal =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, cN);
      auto sel = b.create<scf::IfOp>(loc, TypeRange{f32}, isReal, true);
      {
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(sel.thenBlock());
        Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{bx, i});
        b.create<scf::YieldOp>(loc, v);
        b.setInsertionPointToStart(sel.elseBlock());
        b.create<scf::YieldOp>(loc, sentinel);
      }
      b.create<memref::StoreOp>(loc, sel.getResult(0), valBuf, ValueRange{i});
      if (needIndices) {
        Value ii = b.create<arith::IndexCastOp>(loc, i32, i);
        b.create<memref::StoreOp>(loc, ii, idxBuf, ValueRange{i});
      }
    }

    // (1) BITONIC NETWORK.
    for (int64_t k = 2; k <= P; k <<= 1) {
      Value cK = cIdx(b, loc, k);
      for (int64_t j = k >> 1; j > 0; j >>= 1) {
        Value cJ = cIdx(b, loc, j);
        b.create<gpu::BarrierOp>(loc);
        auto loop = b.create<scf::ForOp>(loc, tx, cP, cT);
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(loop.getBody());
        Value i = loop.getInductionVar();
        Value ixj = b.create<arith::XOrIOp>(loc, i, cJ);
        Value lower =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, ixj, i);
        auto owns = b.create<scf::IfOp>(loc, lower, /*withElse=*/false);
        {
          OpBuilder::InsertionGuard g2(b);
          b.setInsertionPointToStart(owns.thenBlock());
          Value ik = b.create<arith::AndIOp>(loc, i, cK);
          Value ascend = b.create<arith::CmpIOp>(
              loc,
              descending ? arith::CmpIPredicate::ne : arith::CmpIPredicate::eq,
              ik, c0i);
          Value vi = b.create<memref::LoadOp>(loc, valBuf, ValueRange{i});
          Value vj = b.create<memref::LoadOp>(loc, valBuf, ValueRange{ixj});
          Value gt =
              b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, vi, vj);
          Value lt =
              b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, vi, vj);
          Value doSwap = b.create<arith::SelectOp>(loc, ascend, gt, lt);
          auto sw = b.create<scf::IfOp>(loc, doSwap, /*withElse=*/false);
          {
            OpBuilder::InsertionGuard g3(b);
            b.setInsertionPointToStart(sw.thenBlock());
            b.create<memref::StoreOp>(loc, vj, valBuf, ValueRange{i});
            b.create<memref::StoreOp>(loc, vi, valBuf, ValueRange{ixj});
            if (needIndices) {
              Value ii = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{i});
              Value ij = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{ixj});
              b.create<memref::StoreOp>(loc, ij, idxBuf, ValueRange{i});
              b.create<memref::StoreOp>(loc, ii, idxBuf, ValueRange{ixj});
            }
          }
        }
      }
    }

    // (2) final barrier, write the first W sorted entries.
    b.create<gpu::BarrierOp>(loc);
    {
      auto loop = b.create<scf::ForOp>(loc, tx, cW, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value t = loop.getInductionVar();
      if (needValues) {
        Value v = b.create<memref::LoadOp>(loc, valBuf, ValueRange{t});
        b.create<memref::StoreOp>(loc, v, valOut, ValueRange{bx, t});
      }
      if (needIndices) {
        Value id = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{t});
        b.create<memref::StoreOp>(loc, id, idxOut, ValueRange{bx, t});
      }
    }
    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(s.launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    (void)c0i;
    return success();
  }

  // ---- RADIX (large-N, CUB-parity) --------------------------------------- //
  LogicalResult rewriteRadix() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: input memref + one output memref per USED result.
    SmallVector<Type> inTys;
    inTys.push_back(MemRefType::get({M, N}, f32));
    if (needValues)
      inTys.push_back(MemRefType::get({M, W}, f32));
    if (needIndices)
      inTys.push_back(MemRefType::get({M, W}, i32));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    unsigned argi = 1;
    Value valOut = needValues ? entry->getArgument(argi++) : Value();
    Value idxOut = needIndices ? entry->getArgument(argi++) : Value();

    Value cT = cIdx(b, loc, T);
    Value c1 = cIdx(b, loc, 1);
    Value cN = cIdx(b, loc, N);
    Value cW = cIdx(b, loc, W);
    Value c0 = cIdx(b, loc, 0);
    Value cC = cIdx(b, loc, C);
    Value cDig = cIdx(b, loc, kDigits);
    Value cM = cIdx(b, loc, M);

    // Shared layout (bytes): keyA | keyB | [idxA | idxB] | hist[kDigits*T] |
    // seg[T] | warpTot[numWarps].  hist is scanned in flat digit-major order so
    // its exclusive prefix IS the stable scatter offset; seg/warpTot are the
    // two-level (per-thread segment + warp) block-scan scratch.
    int64_t numWarps = (T + 31) / 32;
    int64_t nBytes = N * 4;
    int64_t offKeyA = 0, offKeyB = nBytes;
    int64_t offIdxA = 2 * nBytes, offIdxB = 3 * nBytes;
    int64_t offHist = (needIndices ? 4 : 2) * nBytes;
    int64_t offSeg = offHist + kDigits * T * 4;
    int64_t offWarp = offSeg + T * 4;
    int64_t shBytes = offWarp + numWarps * 4;
    Value shBytesV =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shBytes));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesV);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto keyTy = MemRefType::get({N}, i32, nullptr, wgSpace);
    auto histTy = MemRefType::get({kDigits * T}, i32, nullptr, wgSpace);
    auto segTy = MemRefType::get({T}, i32, nullptr, wgSpace);
    auto warpTy = MemRefType::get({numWarps}, i32, nullptr, wgSpace);
    auto view = [&](Type ty, int64_t off) -> Value {
      return b.create<memref::ViewOp>(loc, ty, raw, cIdx(b, loc, off),
                                      ValueRange{});
    };
    Value keyA = view(keyTy, offKeyA);
    Value keyB = view(keyTy, offKeyB);
    Value idxA, idxB;
    if (needIndices) {
      idxA = view(keyTy, offIdxA);
      idxB = view(keyTy, offIdxB);
    }
    Value hist = view(histTy, offHist);
    Value seg = view(segTy, offSeg);
    Value warpTot = numWarps > 0 ? view(warpTy, offWarp) : Value();

    // Helper: this thread's histogram slot for digit d -> hist[d*T + tx].
    auto histSlot = [&](Value d) -> Value {
      return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, d, cT),
                                     tx);
    };

    // (0) LOAD: strided (coalesced) flip of the row into keyA; idxA[i]=i.
    {
      auto loop = b.create<scf::ForOp>(loc, tx, cN, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value i = loop.getInductionVar();
      Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{bx, i});
      b.create<memref::StoreOp>(loc, flip(b, loc, v), keyA, ValueRange{i});
      if (needIndices) {
        Value ii = b.create<arith::IndexCastOp>(loc, i32, i);
        b.create<memref::StoreOp>(loc, ii, idxA, ValueRange{i});
      }
    }

    // chunk bounds for thread tx: [start, end), end = min(start+C, N).
    Value start = b.create<arith::MulIOp>(loc, tx, cC);
    Value e0 = b.create<arith::AddIOp>(loc, start, cC);
    Value end = b.create<arith::MinUIOp>(loc, e0, cN);

    // (1) 8 LSD RADIX PASSES, ping-ponging.
    for (int64_t p = 0; p < kPasses; ++p) {
      Value srcK = (p & 1) ? keyB : keyA;
      Value dstK = (p & 1) ? keyA : keyB;
      Value srcI = (p & 1) ? idxB : idxA;
      Value dstI = (p & 1) ? idxA : idxB;
      Value shift = cI32(b, loc, (int32_t)(p * kBits));

      b.create<gpu::BarrierOp>(loc); // src ready + hist reuse safe

      // COUNT: this thread's per-digit counts live in its shared column
      // hist[d*T+tx] (avoids dynamically-indexed local memory). Zero it, then
      // read-modify-write over the blocked chunk.
      {
        auto z = b.create<scf::ForOp>(loc, c0, cDig, c1);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(z.getBody());
          b.create<memref::StoreOp>(loc, cI32(b, loc, 0), hist,
                                    ValueRange{histSlot(z.getInductionVar())});
        }
        auto loop = b.create<scf::ForOp>(loc, start, end, c1);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(loop.getBody());
          Value k = b.create<memref::LoadOp>(loc, srcK,
                                              ValueRange{loop.getInductionVar()});
          Value slot = histSlot(digitIdx(b, loc, k, shift));
          Value cur = b.create<memref::LoadOp>(loc, hist, ValueRange{slot});
          Value inc = b.create<arith::AddIOp>(loc, cur, cI32(b, loc, 1));
          b.create<memref::StoreOp>(loc, inc, hist, ValueRange{slot});
        }
      }

      b.create<gpu::BarrierOp>(loc);

      // SCAN: exclusive prefix sum of hist[0..kDigits*T) in flat digit-major
      // order -> hist becomes the stable per-slot scatter offset directly. Done
      // work-efficiently in two levels: (A) each thread scans its own kDigits-
      // long contiguous segment [tx*kDigits, +kDigits); (B) a warp-shuffle block
      // scan of the T segment totals; (C) add the segment base back.
      Value segStart = b.create<arith::MulIOp>(loc, tx, cDig);
      { // (A) per-thread serial exclusive scan of a kDigits-long segment.
        auto sc = b.create<scf::ForOp>(loc, c0, cDig, c1,
                                       ValueRange{cI32(b, loc, 0)});
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(sc.getBody());
        Value run = sc.getRegionIterArgs()[0];
        Value idx = b.create<arith::AddIOp>(loc, segStart, sc.getInductionVar());
        Value v = b.create<memref::LoadOp>(loc, hist, ValueRange{idx});
        b.create<memref::StoreOp>(loc, run, hist, ValueRange{idx});
        b.create<scf::YieldOp>(loc, ValueRange{b.create<arith::AddIOp>(loc, run, v)});
        b.setInsertionPointAfter(sc);
        b.create<memref::StoreOp>(loc, sc.getResult(0), seg, ValueRange{tx});
      }
      // No barrier here: (B) reads only seg[tx] written by this same thread.

      // (B) block scan of seg[0..T): inclusive warp-shuffle scan, per-warp totals
      // to warpTot, thread 0 exclusive-scans warpTot, then combine to exclusive.
      Value txI = b.create<arith::IndexCastOp>(loc, i32, tx);
      Value lane = b.create<arith::AndIOp>(loc, txI, cI32(b, loc, 31));
      Value orig = b.create<memref::LoadOp>(loc, seg, ValueRange{tx});
      Value incl = orig;
      for (int off = 1; off < 32 && off < T; off <<= 1) {
        Value y = b.create<gpu::ShuffleOp>(loc, incl, off, 32,
                                           gpu::ShuffleMode::UP)
                      .getShuffleResult();
        Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lane,
                                           cI32(b, loc, off));
        Value sum = b.create<arith::AddIOp>(loc, incl, y);
        incl = b.create<arith::SelectOp>(loc, ge, sum, incl);
      }
      Value warpBase = cI32(b, loc, 0);
      if (numWarps > 1) {
        Value warpId = b.create<arith::DivUIOp>(loc, tx, cIdx(b, loc, 32));
        Value isLast =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane,
                                    cI32(b, loc, 31));
        auto ifL = b.create<scf::IfOp>(loc, isLast, /*withElse=*/false);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(ifL.thenBlock());
          b.create<memref::StoreOp>(loc, incl, warpTot, ValueRange{warpId});
        }
        b.create<gpu::BarrierOp>(loc);
        { // thread 0 exclusive-scans the numWarps totals in place.
          Value is0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx,
                                              c0);
          auto if0 = b.create<scf::IfOp>(loc, is0, /*withElse=*/false);
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(if0.thenBlock());
          auto sc = b.create<scf::ForOp>(loc, c0, cIdx(b, loc, numWarps), c1,
                                         ValueRange{cI32(b, loc, 0)});
          OpBuilder::InsertionGuard g2(b);
          b.setInsertionPointToStart(sc.getBody());
          Value run = sc.getRegionIterArgs()[0];
          Value w = sc.getInductionVar();
          Value v = b.create<memref::LoadOp>(loc, warpTot, ValueRange{w});
          b.create<memref::StoreOp>(loc, run, warpTot, ValueRange{w});
          b.create<scf::YieldOp>(loc,
                                 ValueRange{b.create<arith::AddIOp>(loc, run, v)});
        }
        b.create<gpu::BarrierOp>(loc);
        warpBase = b.create<memref::LoadOp>(loc, warpTot, ValueRange{warpId});
      }
      // exclusive segment base = (inclusive - own) + warpBase.
      {
        Value excl = b.create<arith::SubIOp>(loc, incl, orig);
        excl = b.create<arith::AddIOp>(loc, excl, warpBase);
        b.create<memref::StoreOp>(loc, excl, seg, ValueRange{tx});
      }
      // No barrier here: (C) reads seg[tx] and hist[segment], both written by
      // this same thread; the barrier before SCATTER publishes them cross-thread.

      // (C) add each thread's segment base to its kDigits-long segment -> hist
      // now holds the flat exclusive prefix = stable scatter offsets.
      {
        Value sb = b.create<memref::LoadOp>(loc, seg, ValueRange{tx});
        auto sc = b.create<scf::ForOp>(loc, c0, cDig, c1);
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(sc.getBody());
        Value idx = b.create<arith::AddIOp>(loc, segStart, sc.getInductionVar());
        Value v = b.create<memref::LoadOp>(loc, hist, ValueRange{idx});
        b.create<memref::StoreOp>(loc, b.create<arith::AddIOp>(loc, v, sb), hist,
                                  ValueRange{idx});
      }
      b.create<gpu::BarrierOp>(loc);

      // SCATTER: hist[d*T+tx] is now this thread's running destination offset
      // for digit d; re-walk the chunk IN ORDER emitting each key/index to it.
      {
        auto loop = b.create<scf::ForOp>(loc, start, end, c1);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(loop.getBody());
          Value i = loop.getInductionVar();
          Value k = b.create<memref::LoadOp>(loc, srcK, ValueRange{i});
          Value slot = histSlot(digitIdx(b, loc, k, shift));
          Value pos = b.create<memref::LoadOp>(loc, hist, ValueRange{slot});
          Value posIdx =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), pos);
          b.create<memref::StoreOp>(loc, k, dstK, ValueRange{posIdx});
          if (needIndices) {
            Value id = b.create<memref::LoadOp>(loc, srcI, ValueRange{i});
            b.create<memref::StoreOp>(loc, id, dstI, ValueRange{posIdx});
          }
          Value nx = b.create<arith::AddIOp>(loc, pos, cI32(b, loc, 1));
          b.create<memref::StoreOp>(loc, nx, hist, ValueRange{slot});
        }
      }
    }
    // kPasses is even -> sorted result is back in keyA / idxA.

    b.create<gpu::BarrierOp>(loc);

    // (2) STORE: strided (coalesced) write of the first W entries. descending
    // reverse-reads srcW = N-1-w (internal sort is ascending).
    {
      auto loop = b.create<scf::ForOp>(loc, tx, cW, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value w = loop.getInductionVar();
      Value srcW = w;
      if (descending) {
        Value cNm1 = cIdx(b, loc, N - 1);
        srcW = b.create<arith::SubIOp>(loc, cNm1, w);
      }
      if (needValues) {
        Value k = b.create<memref::LoadOp>(loc, keyA, ValueRange{srcW});
        b.create<memref::StoreOp>(loc, unflip(b, loc, k), valOut,
                                  ValueRange{bx, w});
      }
      if (needIndices) {
        Value id = b.create<memref::LoadOp>(loc, idxA, ValueRange{srcW});
        b.create<memref::StoreOp>(loc, id, idxOut, ValueRange{bx, w});
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  // ---- MATCH RADIX (CUB-style warp-match register-ranking) --------------- //
  // The single-block LSD radix that BEATS CUB at the power-of-two GATE by
  // removing the shared histogram entirely. Per 4-bit digit pass, keys are held
  // in registers (C = ceil(N/T) per thread, the slot loop UNROLLED so no local
  // spill) and ranked with the warp-match trick:
  //   * peer_mask = the active lanes of my warp holding my digit, built with
  //     kBits `nvvm.vote.ballot.sync` (one per digit bit) ANDed together;
  //   * my intra-warp rank = popc(peer_mask & lanes-below-me), leader adds
  //     popc(peer_mask) to a tiny numWarps*kDigits shared warp table.
  // A single cross-warp exclusive scan of that table (digit-major, warp-minor)
  // yields each (warp,digit) block base; each key's destination is base + its
  // running warp-digit offset. Order (digit, warp, slot, lane) == ascending
  // buffer position, so the pass is STABLE -> LSD is correct. No count RMW, no
  // kDigits*T histogram sweep -> bandwidth-bound.
  LogicalResult rewriteRadixMatch() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    SmallVector<Type> inTys;
    inTys.push_back(MemRefType::get({M, N}, f32));
    if (needValues)
      inTys.push_back(MemRefType::get({M, W}, f32));
    if (needIndices)
      inTys.push_back(MemRefType::get({M, W}, i32));
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    unsigned argi = 1;
    Value valOut = needValues ? entry->getArgument(argi++) : Value();
    Value idxOut = needIndices ? entry->getArgument(argi++) : Value();

    int64_t numWarps = T / 32;
    int64_t nTab = numWarps * kDigits;
    int64_t nBytes = N * 4;
    int64_t offKeyA = 0, offKeyB = nBytes;
    int64_t offIdxA = 2 * nBytes, offIdxB = 3 * nBytes;
    int64_t offTab = (needIndices ? 4 : 2) * nBytes;
    int64_t shBytes = offTab + nTab * 4;

    Value cT = cIdx(b, loc, T), c1 = cIdx(b, loc, 1);
    Value cM = cIdx(b, loc, M);
    Value shBytesV =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shBytes));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesV);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    // Constants used INSIDE the launch region must be defined inside it, else the
    // kernel outliner captures them as launch args (the harness only maps ptr /
    // index/i64 args, not i32).
    Value cN = cIdx(b, loc, N), cW = cIdx(b, loc, W), c0 = cIdx(b, loc, 0);
    Value c0i = cI32(b, loc, 0), cNeg1 = cI32(b, loc, -1);

    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto keyTy = MemRefType::get({N}, i32, nullptr, wgSpace);
    auto tabTy = MemRefType::get({nTab}, i32, nullptr, wgSpace);
    auto view = [&](Type ty, int64_t off) -> Value {
      return b.create<memref::ViewOp>(loc, ty, raw, cIdx(b, loc, off),
                                      ValueRange{});
    };
    Value keyA = view(keyTy, offKeyA), keyB = view(keyTy, offKeyB);
    Value idxA, idxB;
    if (needIndices) {
      idxA = view(keyTy, offIdxA);
      idxB = view(keyTy, offIdxB);
    }
    Value tab = view(tabTy, offTab);

    // warp-relative geometry.
    Value c32 = cIdx(b, loc, 32);
    Value warpId = b.create<arith::DivUIOp>(loc, tx, c32);   // index
    Value laneIdx = b.create<arith::RemUIOp>(loc, tx, c32);  // index
    Value lane = b.create<arith::IndexCastOp>(loc, i32, laneIdx);
    Value warpBaseElems =
        b.create<arith::MulIOp>(loc, warpId, cIdx(b, loc, 32 * C)); // w*32*C
    Value warpTabRow =
        b.create<arith::MulIOp>(loc, warpId, cIdx(b, loc, kDigits)); // w*16
    // lanes strictly below me: (1<<lane) - 1.
    Value ltMask = b.create<arith::SubIOp>(
        loc, b.create<arith::ShLIOp>(loc, cI32(b, loc, 1), lane),
        cI32(b, loc, 1));

    auto ballot = [&](Value predI1) -> Value {
      return b.create<NVVM::VoteBallotOp>(loc, i32, cNeg1, predI1);
    };
    auto popc = [&](Value v) -> Value {
      return b.create<math::CtPopOp>(loc, v);
    };
    auto warpSync = [&]() { b.create<NVVM::SyncWarpOp>(loc, cNeg1); };

    // (0) LOAD: strided coalesced flip of the row into keyA; idxA[i]=i.
    {
      auto loop = b.create<scf::ForOp>(loc, tx, cN, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value i = loop.getInductionVar();
      Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{bx, i});
      b.create<memref::StoreOp>(loc, flip(b, loc, v), keyA, ValueRange{i});
      if (needIndices) {
        Value ii = b.create<arith::IndexCastOp>(loc, i32, i);
        b.create<memref::StoreOp>(loc, ii, idxA, ValueRange{i});
      }
    }

    Value cTab = cIdx(b, loc, nTab);
    Value cKD = cIdx(b, loc, kDigits);

    // (1) 8 LSD RADIX PASSES.
    for (int64_t p = 0; p < kPasses; ++p) {
      Value srcK = (p & 1) ? keyB : keyA;
      Value dstK = (p & 1) ? keyA : keyB;
      Value srcI = (p & 1) ? idxB : idxA;
      Value dstI = (p & 1) ? idxA : idxB;
      Value shift = cI32(b, loc, (int32_t)(p * kBits));

      b.create<gpu::BarrierOp>(loc); // src ready + tab reuse safe

      // zero the warp table.
      {
        Value inRange =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx, cTab);
        auto ifz = b.create<scf::IfOp>(loc, inRange, /*withElse=*/false);
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifz.thenBlock());
        b.create<memref::StoreOp>(loc, c0i, tab, ValueRange{tx});
      }
      b.create<gpu::BarrierOp>(loc);

      // SWEEP 1: rank each register-resident key, accumulate the warp table.
      // Only the key, its running offset and index need cross the barrier; the
      // digit and active flag are recomputed in sweep 2 (cheaper than holding
      // them live -> lower register pressure -> higher occupancy).
      SmallVector<Value> keyS(C), idxS(C), offS(C);
      for (int64_t s = 0; s < C; ++s) {
        Value sBase = b.create<arith::AddIOp>(loc, warpBaseElems,
                                              cIdx(b, loc, s * 32));
        Value pos = b.create<arith::AddIOp>(loc, sBase, laneIdx);
        Value active =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, pos, cN);
        Value posSafe = b.create<arith::SelectOp>(loc, active, pos, c0);
        Value key = b.create<memref::LoadOp>(loc, srcK, ValueRange{posSafe});
        Value idx;
        if (needIndices)
          idx = b.create<memref::LoadOp>(loc, srcI, ValueRange{posSafe});
        Value digI32 = b.create<arith::AndIOp>(
            loc, b.create<arith::ShRUIOp>(loc, key, shift),
            cI32(b, loc, kDigits - 1));
        Value digIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
                                                    digI32);
        // warp-match: peer_mask = active lanes of my warp with my digit.
        Value activeBallot = ballot(active);
        Value peer = activeBallot;
        for (int bit = 0; bit < kBits; ++bit) {
          Value shd =
              b.create<arith::ShRUIOp>(loc, digI32, cI32(b, loc, bit));
          Value b1 = b.create<arith::AndIOp>(loc, shd, cI32(b, loc, 1));
          Value bitSet =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, b1, c0i);
          Value pred = b.create<arith::AndIOp>(loc, active, bitSet);
          Value bal = ballot(pred);
          Value notBal = b.create<arith::XOrIOp>(loc, bal, cNeg1);
          Value elseMask = b.create<arith::AndIOp>(loc, activeBallot, notBal);
          Value sel = b.create<arith::SelectOp>(loc, bitSet, bal, elseMask);
          peer = b.create<arith::AndIOp>(loc, peer, sel);
        }
        Value peerLt = b.create<arith::AndIOp>(loc, peer, ltMask);
        Value rankBefore = popc(peerLt);
        Value peerCount = popc(peer);
        Value isLeaderRaw = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, peerLt, c0i);
        Value isLeader = b.create<arith::AndIOp>(loc, isLeaderRaw, active);
        // running warp-digit offset.
        Value tabSlot = b.create<arith::AddIOp>(loc, warpTabRow, digIdx);
        Value runBefore = b.create<memref::LoadOp>(loc, tab, ValueRange{tabSlot});
        Value warpOff = b.create<arith::AddIOp>(loc, runBefore, rankBefore);
        warpSync(); // all lanes read runBefore before any leader writes
        {
          auto ifl = b.create<scf::IfOp>(loc, isLeader, /*withElse=*/false);
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(ifl.thenBlock());
          Value upd = b.create<arith::AddIOp>(loc, runBefore, peerCount);
          b.create<memref::StoreOp>(loc, upd, tab, ValueRange{tabSlot});
        }
        warpSync(); // publish the leader writes before the next slot
        keyS[s] = key;
        idxS[s] = idx;
        offS[s] = warpOff;
      }

      b.create<gpu::BarrierOp>(loc);

      // SCAN: thread 0 exclusive-scans the warp table in place, digit-major /
      // warp-minor, so tab[w*16+d] becomes the block base for (warp w, digit d).
      {
        Value is0 =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, c0);
        auto if0 = b.create<scf::IfOp>(loc, is0, /*withElse=*/false);
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(if0.thenBlock());
        auto od = b.create<scf::ForOp>(loc, c0, cKD, c1, ValueRange{c0i});
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPointToStart(od.getBody());
        Value d = od.getInductionVar();
        Value runD = od.getRegionIterArgs()[0];
        auto iw = b.create<scf::ForOp>(loc, c0, cIdx(b, loc, numWarps), c1,
                                       ValueRange{runD});
        {
          OpBuilder::InsertionGuard g3(b);
          b.setInsertionPointToStart(iw.getBody());
          Value w = iw.getInductionVar();
          Value run = iw.getRegionIterArgs()[0];
          Value idx = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, w, cKD), d);
          Value v = b.create<memref::LoadOp>(loc, tab, ValueRange{idx});
          b.create<memref::StoreOp>(loc, run, tab, ValueRange{idx});
          b.create<scf::YieldOp>(loc,
                                 ValueRange{b.create<arith::AddIOp>(loc, run, v)});
        }
        b.create<scf::YieldOp>(loc, ValueRange{iw.getResult(0)});
      }
      b.create<gpu::BarrierOp>(loc);

      // SWEEP 2: scatter each key to base[warp,digit] + its running offset. The
      // digit and active flag are recomputed here (not held across the barrier).
      for (int64_t s = 0; s < C; ++s) {
        Value sBase = b.create<arith::AddIOp>(loc, warpBaseElems,
                                              cIdx(b, loc, s * 32));
        Value pos = b.create<arith::AddIOp>(loc, sBase, laneIdx);
        Value active =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, pos, cN);
        auto ifa = b.create<scf::IfOp>(loc, active, /*withElse=*/false);
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifa.thenBlock());
        Value digI32 = b.create<arith::AndIOp>(
            loc, b.create<arith::ShRUIOp>(loc, keyS[s], shift),
            cI32(b, loc, kDigits - 1));
        Value digIdx =
            b.create<arith::IndexCastOp>(loc, b.getIndexType(), digI32);
        Value baseSlot = b.create<arith::AddIOp>(loc, warpTabRow, digIdx);
        Value base = b.create<memref::LoadOp>(loc, tab, ValueRange{baseSlot});
        Value fp = b.create<arith::AddIOp>(loc, base, offS[s]);
        Value fpIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), fp);
        b.create<memref::StoreOp>(loc, keyS[s], dstK, ValueRange{fpIdx});
        if (needIndices)
          b.create<memref::StoreOp>(loc, idxS[s], dstI, ValueRange{fpIdx});
      }
    }
    // kPasses is even -> sorted result is back in keyA / idxA.

    b.create<gpu::BarrierOp>(loc);

    // (2) STORE: coalesced write of the first W entries; descending mirror-reads.
    {
      auto loop = b.create<scf::ForOp>(loc, tx, cW, cT);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      Value w = loop.getInductionVar();
      Value srcW = w;
      if (descending) {
        Value cNm1 = cIdx(b, loc, N - 1);
        srcW = b.create<arith::SubIOp>(loc, cNm1, w);
      }
      if (needValues) {
        Value k = b.create<memref::LoadOp>(loc, keyA, ValueRange{srcW});
        b.create<memref::StoreOp>(loc, unflip(b, loc, k), valOut,
                                  ValueRange{bx, w});
      }
      if (needIndices) {
        Value id = b.create<memref::LoadOp>(loc, idxA, ValueRange{srcW});
        b.create<memref::StoreOp>(loc, id, idxOut, ValueRange{bx, w});
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentSortToGPUPass
    : ::intent::impl::ConvertIntentSortToGPUBase<ConvertIntentSortToGPUPass> {
  using ConvertIntentSortToGPUBase::ConvertIntentSortToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      SortLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      SortLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("sort lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
