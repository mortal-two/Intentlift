//===- ConvertIntentCompactToGPU.cpp - dynamic-shape compaction -> gpu ----===//
//
// The C8 dynamic-shape-output pass — the ONE genuinely-new mechanism vs every
// prior cluster: a data-dependent output EXTENT. Lowers a func whose body is
// exactly ONE `intent.nonzero` / `intent.masked_select` / `intent.masked_scatter`
// to a single block-cooperative prefix-sum `gpu.launch` (grid (1,1,1), block
// (T,1,1)).
//
// The compaction primitive is an EXCLUSIVE PREFIX-SUM of the flattened 0/1 mask.
// The `Ntotal = prod(shape)` flattened elements are split into T contiguous
// SEGMENTS (seg = Ntotal/T); thread `t` first counts the true elements in ALL
// prior segments `[0, t*seg)` — its exclusive-prefix base slot — then re-scans
// its own segment ascending, emitting each true element at the running slot.
// Contiguous ascending segments => the global write order is EXACT row-major
// (matches torch.nonzero / torch.masked_select). No shared memory / barriers
// (each thread is independent); the last thread writes the total `nnz` to a
// `count` buffer so the host can slice a MAX-sized output to `[:nnz]`.
//
//   nonzero        : dest (cond, out[Ntotal,ndim] i64, count[1] i64). Each true
//                    flat index i is unraveled to its ndim-vector coord and
//                    stored as out[slot, 0..ndim).
//   masked_select  : dest (data, mask, sel[Ntotal], count[1] i64). sel[slot] =
//                    data[coord] at each true position.
//   masked_scatter : dest (data, mask, src, out) — out DATA-SHAPED (static).
//                    out[coord] = mask[coord] ? src[slot] : data[coord].
//
// Correctness-only (bit-faithful, exact row-major order). Static shapes,
// Ntotal % T == 0. Run BEFORE `convert-intent-to-linalg`.
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
  int64_t Ntotal = 0, T = 0, seg = 0;

  CompactLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
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
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Ntotal);
    if (T <= 0 || Ntotal % T != 0)
      return failure(); // non-divisible tail is a documented follow-up
    seg = Ntotal / T;
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

    Value cT = cIdx(b, loc, T);
    Value c1 = cIdx(b, loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, c1, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value tx = launch.getThreadIds().x;

    Value cSeg = cIdx(b, loc, seg);
    Value lo = b.create<arith::MulIOp>(loc, tx, cSeg);         // t*seg
    Value hi = b.create<arith::AddIOp>(loc, lo, cSeg);         // (t+1)*seg
    Value c0 = cIdx(b, loc, 0);

    // Phase 1: exclusive-prefix base = trues in [0, t*seg).
    Value base = countTrue(b, loc, maskMem, c0, lo);

    // Phase 2: re-scan own segment ascending, emit at the running slot.
    auto emit = b.create<scf::ForOp>(loc, lo, hi, c1, ValueRange{base});
    {
      OpBuilder lb = OpBuilder::atBlockBegin(emit.getBody());
      Value j = emit.getInductionVar();
      Value slot = emit.getRegionIterArg(0);
      SmallVector<Value> coord = unravel(lb, loc, j);
      Value mv = lb.create<memref::LoadOp>(loc, maskMem, coord);
      Value tru = isTrue(lb, loc, mv);

      auto ifOp = lb.create<scf::IfOp>(loc, TypeRange{lb.getIndexType()}, tru,
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

      OpBuilder eb = ifOp.getElseBodyBuilder();
      if (mscatter) { // unmasked -> out[coord] = data[coord]
        Value dv = eb.create<memref::LoadOp>(loc, dataMem, coord);
        eb.create<memref::StoreOp>(loc, dv, outMem, coord);
      }
      eb.create<scf::YieldOp>(loc, slot);

      lb.create<scf::YieldOp>(loc, ifOp.getResult(0));
    }

    // The last thread's final slot == total nnz -> write the count buffer.
    if (countMem) {
      Value cTm1 = cIdx(b, loc, T - 1);
      Value last = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx,
                                           cTm1);
      auto ifLast = b.create<scf::IfOp>(loc, last, /*withElse=*/false);
      OpBuilder cb = ifLast.getThenBodyBuilder();
      Value total = cb.create<arith::IndexCastOp>(loc, i64, emit.getResult(0));
      Value z = cb.create<arith::ConstantIndexOp>(loc, 0);
      cb.create<memref::StoreOp>(loc, total, countMem, ValueRange{z});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
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
