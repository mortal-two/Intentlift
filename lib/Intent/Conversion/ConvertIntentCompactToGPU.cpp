//===- ConvertIntentCompactToGPU.cpp - dynamic-shape compaction -> gpu ----===//
//
// The C8 dynamic-shape-output pass — the ONE genuinely-new mechanism vs every
// prior cluster: a data-dependent output EXTENT. Lowers a func whose body is
// exactly ONE `intent.nonzero` / `intent.masked_select` / `intent.masked_scatter`
// to a MULTI-BLOCK parallel stream compaction (three `gpu.launch`es sharing a
// device scratch), replacing the old single-block (grid (1,1,1)) serial
// prefix-sum that serialized the whole compaction on one SM (~1400x slower than
// torch on large masks — the phase-1 base count re-scanned [0, t*seg) per thread,
// O(Ntotal * T) work on a single block).
//
// The compaction primitive is an EXCLUSIVE PREFIX-SUM of the flattened 0/1 mask,
// now done in the standard 3-phase parallel form. The `Ntotal = prod(shape)`
// flattened elements are split into B contiguous CHUNKS (chunk = Ntotal/B), one
// per grid block (block = 1 thread):
//
//   Phase 1 (count)   grid(B) block(1): block b serially counts the true mask
//                     elements in its chunk [b*chunk, (b+1)*chunk) -> scratch[b].
//   Phase 2 (scan)    grid(1) block(1): a single thread exclusive-scans the B
//                     block counts in place (scratch[b] <- sum of all prior
//                     blocks = that block's global base slot) and writes the
//                     grand total `nnz` to the `count` buffer.
//   Phase 3 (scatter) grid(B) block(1): block b re-scans its chunk ascending,
//                     emitting each true element at scratch[b] + local_prefix
//                     (local_prefix = the block's own running counter).
//
// Contiguous chunks in block order + ascending within each block => the global
// write order is EXACT row-major (matches torch.nonzero / torch.masked_select),
// identical to the old single-block order but parallelized across all SMs. The
// serial scan (phase 2) is over only B (<= 8192) elements, negligible.
//
//   nonzero        : dest (cond, out[Ntotal,ndim] i64, count[1] i64). Each true
//                    flat index i is unraveled to its ndim-vector coord and
//                    stored as out[slot, 0..ndim).
//   masked_select  : dest (data, mask, sel[Ntotal], count[1] i64). sel[slot] =
//                    data[coord] at each true position.
//   masked_scatter : dest (data, mask, src, out) — out DATA-SHAPED (static).
//                    out[coord] = mask[coord] ? src[slot] : data[coord]. No
//                    `count` buffer; still uses phases 1/2/3 (the scan gives each
//                    block its base offset into `src`); blocks write disjoint
//                    coords so the else-copy is race-free.
//
// The B-element scratch is a `memref.alloc` (lowers to a host malloc the launcher
// discovers as a loose intermediate buffer). Correctness-only (bit-faithful,
// exact row-major order). Static shapes. Run BEFORE `convert-intent-to-linalg`.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCOMPACTTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's dynamic-shape compaction lowering. Reads the original (tensor)
/// func body, builds a new memref+gpu.launch func with the same name, erases the
/// old.
struct CompactLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match(): exactly one of these is set.
  NonzeroOp nonzero;
  MaskedSelectOp mselect;
  MaskedScatterOp mscatter;

  // The mask/cond tensor whose true elements drive the compaction.
  RankedTensorType maskTy;
  SmallVector<int64_t> shape; // static shape of the mask/cond
  int64_t Ntotal = 0, B = 0, chunk = 0;
  int64_t Ts = 0, G = 0; // scan block: Ts threads, each owns G = B/Ts counts

  CompactLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  /// Largest power-of-two divisor of N that is <= cap (>= 1). Guarantees an even
  /// contiguous split so no thread carries a ragged tail.
  static int64_t chooseBlocks(int64_t N, int64_t cap = 8192) {
    int64_t best = 1;
    for (int64_t b = 1; b <= cap && b <= N; b <<= 1)
      if (N % b == 0)
        best = b;
    return best;
  }

  LogicalResult setGeometry(RankedTensorType m) {
    maskTy = m;
    if (!m.hasStaticShape() || m.getRank() < 1)
      return failure();
    shape.assign(m.getShape().begin(), m.getShape().end());
    Ntotal = 1;
    for (int64_t d : shape)
      Ntotal *= d;
    if (Ntotal <= 0)
      return failure();
    B = chooseBlocks(Ntotal);
    if (B <= 0 || Ntotal % B != 0)
      return failure();
    chunk = Ntotal / B;
    // Scan block: Ts threads (<= 1024, a power of two dividing B), each owning
    // G = B/Ts of the block counts for the two-level parallel scan.
    Ts = B < 1024 ? B : 1024;
    G = B / Ts;
    return success();
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    if (!only || ret.getOperand(0) != only->getResult(0))
      return failure();

    if (auto nz = dyn_cast<NonzeroOp>(only)) {
      nonzero = nz;
      return setGeometry(cast<RankedTensorType>(nz.getCond().getType()));
    }
    if (auto ms = dyn_cast<MaskedSelectOp>(only)) {
      mselect = ms;
      return setGeometry(cast<RankedTensorType>(ms.getMask().getType()));
    }
    if (auto ms = dyn_cast<MaskedScatterOp>(only)) {
      mscatter = ms;
      return setGeometry(cast<RankedTensorType>(ms.getMask().getType()));
    }
    return failure();
  }

  // ---- emission helpers -------------------------------------------------- //
  Value cIdx(OpBuilder &b, Location loc, int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v);
  }

  /// Row-major unravel of a flat index into per-dim coords (index type).
  SmallVector<Value> unravel(OpBuilder &b, Location loc, Value flat) {
    int D = (int)shape.size();
    SmallVector<Value> coords(D);
    Value rem = flat;
    for (int k = D - 1; k >= 0; --k) {
      Value dk = cIdx(b, loc, shape[k]);
      coords[k] = b.create<arith::RemUIOp>(loc, rem, dk);
      if (k > 0)
        rem = b.create<arith::DivUIOp>(loc, rem, dk);
    }
    return coords;
  }

  /// Truth test of a loaded element: `!= 0` (NaN counts as nonzero, matching
  /// torch/numpy). f32/int/i1 accepted.
  Value isTrue(OpBuilder &b, Location loc, Value v) {
    Type t = v.getType();
    if (isa<FloatType>(t)) {
      Value z = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 0.0));
      // UNE: unordered not-equal -> NaN is "true" (nonzero), like numpy.
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v, z);
    }
    if (t.isInteger(1))
      return v;
    Value z = b.create<arith::ConstantIntOp>(loc, 0,
                                             cast<IntegerType>(t).getWidth());
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v, z);
  }

  /// Count the true mask elements over the half-open flat range [lo, hi).
  Value countTrue(OpBuilder &b, Location loc, Value maskMem, Value lo, Value hi) {
    Value c0 = cIdx(b, loc, 0);
    Value c1 = cIdx(b, loc, 1);
    auto loop = b.create<scf::ForOp>(loc, lo, hi, c1, ValueRange{c0});
    OpBuilder lb = OpBuilder::atBlockBegin(loop.getBody());
    Value j = loop.getInductionVar();
    Value acc = loop.getRegionIterArg(0);
    SmallVector<Value> coord = unravel(lb, loc, j);
    Value v = lb.create<memref::LoadOp>(loc, maskMem, coord);
    Value tru = isTrue(lb, loc, v);
    Value inc = lb.create<arith::SelectOp>(loc, tru, c1, c0);
    Value next = lb.create<arith::AddIOp>(loc, acc, inc);
    lb.create<scf::YieldOp>(loc, next);
    return loop.getResult(0);
  }

  // ---- rewrite ----------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Type i64 = b.getI64Type();

    // Build the destination-passing signature per op.
    SmallVector<Type> argTys;
    MemRefType maskMemTy = memrefOf(maskTy);
    int D = (int)shape.size();

    MemRefType outMemTy, countMemTy = MemRefType::get({1}, i64);
    MemRefType dataMemTy, srcMemTy;
    if (nonzero) {
      auto idxTy = cast<RankedTensorType>(nonzero.getIndices().getType());
      Type ielt = idxTy.getElementType();
      outMemTy = MemRefType::get({Ntotal, (int64_t)D}, ielt);
      argTys = {maskMemTy, outMemTy, countMemTy};
    } else if (mselect) {
      dataMemTy = memrefOf(cast<RankedTensorType>(mselect.getData().getType()));
      Type delt = dataMemTy.getElementType();
      outMemTy = MemRefType::get({Ntotal}, delt);
      argTys = {dataMemTy, maskMemTy, outMemTy, countMemTy};
    } else {
      dataMemTy = memrefOf(cast<RankedTensorType>(mscatter.getData().getType()));
      srcMemTy = memrefOf(cast<RankedTensorType>(mscatter.getSrc().getType()));
      outMemTy = memrefOf(cast<RankedTensorType>(mscatter.getResult().getType()));
      argTys = {dataMemTy, maskMemTy, srcMemTy, outMemTy};
    }

    auto newFty = FunctionType::get(ctx, argTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    // Named argument bindings.
    Value maskMem, dataMem, srcMem, outMem, countMem;
    if (nonzero) {
      maskMem = entry->getArgument(0);
      outMem = entry->getArgument(1);
      countMem = entry->getArgument(2);
    } else if (mselect) {
      dataMem = entry->getArgument(0);
      maskMem = entry->getArgument(1);
      outMem = entry->getArgument(2);
      countMem = entry->getArgument(3);
    } else {
      dataMem = entry->getArgument(0);
      maskMem = entry->getArgument(1);
      srcMem = entry->getArgument(2);
      outMem = entry->getArgument(3);
    }

    // Device scratch: one i64 count/base slot per block.
    auto scratchTy = MemRefType::get({B}, i64);
    Value scratch = b.create<memref::AllocOp>(loc, scratchTy);

    Value c1 = cIdx(b, loc, 1);
    Value cB = cIdx(b, loc, B);
    Value cChunk = cIdx(b, loc, chunk);
    Value c0 = cIdx(b, loc, 0);

    // ---- Phase 1 (count): grid(B) block(1). scratch[b] = trues in chunk b. --
    {
      auto launch = b.create<gpu::LaunchOp>(loc, cB, c1, c1, c1, c1, c1);
      OpBuilder lb = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value bid = launch.getBlockIds().x;
      Value lo = lb.create<arith::MulIOp>(loc, bid, cChunk);
      Value hi = lb.create<arith::AddIOp>(loc, lo, cChunk);
      Value cnt = countTrue(lb, loc, maskMem, lo, hi);
      Value cnt64 = lb.create<arith::IndexCastOp>(loc, i64, cnt);
      lb.create<memref::StoreOp>(loc, cnt64, scratch, ValueRange{bid});
      lb.create<gpu::TerminatorOp>(loc);
    }

    // ---- Phase 2 (scan): grid(1) block(1). Exclusive scan of scratch[0..B),
    //      in place; grand total -> count buffer. --------------------------- //
    // A two-level parallel scan: Ts threads each serially reduce their G counts
    // into a shared[Ts] partial, an unrolled Hillis-Steele inclusive scan over
    // the Ts partials, then each thread serially distributes the exclusive base
    // across its G entries (scratch[i] <- global base slot). The single-thread
    // serial scan this replaces was the dominant fixed cost (~120ns/count).
    {
      auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
      auto cTs = cIdx(b, loc, Ts);
      Value shBytesV = b.create<arith::ConstantOp>(
          loc, b.getI32IntegerAttr((int32_t)(Ts * 8)));
      auto launch =
          b.create<gpu::LaunchOp>(loc, c1, c1, c1, cTs, c1, c1, shBytesV);
      OpBuilder lb = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value tx = launch.getThreadIds().x;
      Value z64 = lb.create<arith::ConstantIntOp>(loc, 0, 64);
      Value cG = cIdx(lb, loc, G);

      auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                   nullptr, wgSpace);
      Value raw = lb.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
      auto shTy = MemRefType::get({Ts}, i64, nullptr, wgSpace);
      Value sh = lb.create<memref::ViewOp>(loc, shTy, raw, cIdx(lb, loc, 0),
                                           ValueRange{});

      // (1) each thread reduces its G contiguous counts -> shared[tx].
      Value tG = lb.create<arith::MulIOp>(loc, tx, cG);       // tx*G
      Value tGe = lb.create<arith::AddIOp>(loc, tG, cG);      // tx*G+G
      auto red = lb.create<scf::ForOp>(loc, tG, tGe, c1, ValueRange{z64});
      {
        OpBuilder rb = OpBuilder::atBlockBegin(red.getBody());
        Value v = rb.create<memref::LoadOp>(loc, scratch,
                                            ValueRange{red.getInductionVar()});
        Value nx = rb.create<arith::AddIOp>(loc, red.getRegionIterArg(0), v);
        rb.create<scf::YieldOp>(loc, nx);
      }
      lb.create<memref::StoreOp>(loc, red.getResult(0), sh, ValueRange{tx});
      lb.create<gpu::BarrierOp>(loc);

      // (2) unrolled Hillis-Steele inclusive scan over shared[0..Ts).
      for (int64_t d = 1; d < Ts; d <<= 1) {
        Value cd = cIdx(lb, loc, d);
        Value ge = lb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, tx,
                                            cd);
        // read neighbor (guarded) into a register, then barrier before writing.
        auto rd = lb.create<scf::IfOp>(loc, TypeRange{i64}, ge, true);
        {
          OpBuilder tb = rd.getThenBodyBuilder();
          Value idx = tb.create<arith::SubIOp>(loc, tx, cd);
          Value nv = tb.create<memref::LoadOp>(loc, sh, ValueRange{idx});
          tb.create<scf::YieldOp>(loc, nv);
          OpBuilder eb = rd.getElseBodyBuilder();
          eb.create<scf::YieldOp>(loc, z64);
        }
        lb.create<gpu::BarrierOp>(loc);
        auto wr = lb.create<scf::IfOp>(loc, ge, false);
        {
          OpBuilder tb = wr.getThenBodyBuilder();
          Value cur = tb.create<memref::LoadOp>(loc, sh, ValueRange{tx});
          Value sum = tb.create<arith::AddIOp>(loc, cur, rd.getResult(0));
          tb.create<memref::StoreOp>(loc, sum, sh, ValueRange{tx});
        }
        lb.create<gpu::BarrierOp>(loc);
      }

      // (3) exclusive base of thread tx = inclusive[tx-1] (0 for tx==0).
      Value isFirst = lb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx,
                                               c0);
      auto baseIf = lb.create<scf::IfOp>(loc, TypeRange{i64}, isFirst, true);
      {
        OpBuilder tb = baseIf.getThenBodyBuilder();
        tb.create<scf::YieldOp>(loc, z64);
        OpBuilder eb = baseIf.getElseBodyBuilder();
        Value tm1 = eb.create<arith::SubIOp>(loc, tx, c1);
        Value pv = eb.create<memref::LoadOp>(loc, sh, ValueRange{tm1});
        eb.create<scf::YieldOp>(loc, pv);
      }

      // distribute: scratch[i] <- running exclusive base over the G entries.
      auto dist = lb.create<scf::ForOp>(loc, tG, tGe, c1,
                                        ValueRange{baseIf.getResult(0)});
      {
        OpBuilder db = OpBuilder::atBlockBegin(dist.getBody());
        Value i = dist.getInductionVar();
        Value run = dist.getRegionIterArg(0);
        Value v = db.create<memref::LoadOp>(loc, scratch, ValueRange{i});
        db.create<memref::StoreOp>(loc, run, scratch, ValueRange{i});
        Value nx = db.create<arith::AddIOp>(loc, run, v);
        db.create<scf::YieldOp>(loc, nx);
      }

      // (4) grand total nnz = inclusive[Ts-1]; thread 0 writes the count buffer.
      if (countMem) {
        auto totIf = lb.create<scf::IfOp>(loc, isFirst, false);
        OpBuilder tb = totIf.getThenBodyBuilder();
        Value last = cIdx(tb, loc, Ts - 1);
        Value total = tb.create<memref::LoadOp>(loc, sh, ValueRange{last});
        tb.create<memref::StoreOp>(loc, total, countMem, ValueRange{c0});
      }
      lb.create<gpu::TerminatorOp>(loc);
    }

    // ---- Phase 3 (scatter): grid(B) block(1). Block b emits its chunk's true
    //      elements at scratch[b] + local running prefix. ------------------- //
    {
      auto launch = b.create<gpu::LaunchOp>(loc, cB, c1, c1, c1, c1, c1);
      OpBuilder lb = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value bid = launch.getBlockIds().x;
      Value lo = lb.create<arith::MulIOp>(loc, bid, cChunk);
      Value hi = lb.create<arith::AddIOp>(loc, lo, cChunk);
      Value base64 = lb.create<memref::LoadOp>(loc, scratch, ValueRange{bid});
      Value base = lb.create<arith::IndexCastOp>(loc, lb.getIndexType(), base64);

      auto emit = lb.create<scf::ForOp>(loc, lo, hi, c1, ValueRange{base});
      {
        OpBuilder eb = OpBuilder::atBlockBegin(emit.getBody());
        Value j = emit.getInductionVar();
        Value slot = emit.getRegionIterArg(0);
        SmallVector<Value> coord = unravel(eb, loc, j);
        Value mv = eb.create<memref::LoadOp>(loc, maskMem, coord);
        Value tru = isTrue(eb, loc, mv);

        auto ifOp = eb.create<scf::IfOp>(loc, TypeRange{eb.getIndexType()}, tru,
                                         /*withElse=*/true);
        OpBuilder tb = ifOp.getThenBodyBuilder();
        if (nonzero) {
          for (int k = 0; k < D; ++k) {
            Value ci = tb.create<arith::IndexCastOp>(loc, i64, coord[k]);
            Value ck = cIdx(tb, loc, k);
            tb.create<memref::StoreOp>(loc, ci, outMem, ValueRange{slot, ck});
          }
        } else if (mselect) {
          Value dv = tb.create<memref::LoadOp>(loc, dataMem, coord);
          tb.create<memref::StoreOp>(loc, dv, outMem, ValueRange{slot});
        } else { // masked_scatter: masked -> out[coord] = src[slot]
          Value sv = tb.create<memref::LoadOp>(loc, srcMem, ValueRange{slot});
          tb.create<memref::StoreOp>(loc, sv, outMem, coord);
        }
        Value slotp1 = tb.create<arith::AddIOp>(loc, slot, c1);
        tb.create<scf::YieldOp>(loc, slotp1);

        OpBuilder ebElse = ifOp.getElseBodyBuilder();
        if (mscatter) { // unmasked -> out[coord] = data[coord]
          Value dv = ebElse.create<memref::LoadOp>(loc, dataMem, coord);
          ebElse.create<memref::StoreOp>(loc, dv, outMem, coord);
        }
        ebElse.create<scf::YieldOp>(loc, slot);

        eb.create<scf::YieldOp>(loc, ifOp.getResult(0));
      }
      lb.create<gpu::TerminatorOp>(loc);
    }

    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentCompactToGPUPass
    : ::intent::impl::ConvertIntentCompactToGPUBase<
          ConvertIntentCompactToGPUPass> {
  using ConvertIntentCompactToGPUBase::ConvertIntentCompactToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      CompactLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      CompactLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("compact lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
