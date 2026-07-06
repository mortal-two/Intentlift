//===- ConvertIntentPoolToGPU.cpp - windowed pooling -> gpu.launch --------===//
//
// The C7 windowed-reduce pass. Lowers a func whose body is exactly ONE
// `intent.pool` (avg / max / max_pool2d_with_indices) over a rank-4 `[N,C,H,W]`
// f32 input to a single block-cooperative `gpu.launch`: grid = (N*C, OH, 1) (one
// block per channel-plane row of the output), block = (T,1,1) threads over the OW
// output columns (K = OW/T columns each, unrolled).
//
// The mechanism (vs. the full-axis `intent.reduce`): a BOUNDED per-output-cell
// reduction over the static `kH x kW` window. Each thread walks
// `(ih,iw) = (oh*sH - padH + kh, ow*sW - padW + kw)`, guards each source coord
// with an in-bounds `scf.if` (out-of-bounds = skip), and accumulates:
//   avg — sum the in-bounds cells (padded = 0) then divide by kH*kW (torch's
//         default count_include_pad=True).
//   max — running max; when the `indices` output is present, carry the argmax
//         FLAT index `ih*W + iw` as an i64 PAIR alongside the value (strict `>`
//         update = first-index tie-break, matching torch/flag_gems).
//
// Destination-passing: new signature = (input memref) ++ [out] (++ [indices] for
// max-with-indices). Correctness-only (bit-faithful values, exact indices).
// Static shapes, OW % T == 0. The other passes' match() reject any body with an
// intent.pool, so this pass is disjoint and composes freely.
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

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTPOOLTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct PoolLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  PoolOp pool;
  RankedTensorType inTy, outTy;
  int64_t N = 0, C = 0, H = 0, W = 0, OH = 0, OW = 0;
  int64_t kH = 0, kW = 0, sH = 0, sW = 0, padH = 0, padW = 0;
  int64_t T = 0, K = 0;
  bool isMax = false, hasIdx = false;

  PoolLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    pool = dyn_cast_or_null<PoolOp>(only);
    if (!pool)
      return failure();
    inTy = cast<RankedTensorType>(pool.getInput().getType());
    outTy = cast<RankedTensorType>(pool.getOutput().getType());
    if (inTy.getRank() != 4 || outTy.getRank() != 4 || !inTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !inTy.getElementType().isF32())
      return failure();
    N = inTy.getDimSize(0);
    C = inTy.getDimSize(1);
    H = inTy.getDimSize(2);
    W = inTy.getDimSize(3);
    OH = outTy.getDimSize(2);
    OW = outTy.getDimSize(3);
    kH = pool.getKH();
    kW = pool.getKW();
    sH = pool.getSH();
    sW = pool.getSW();
    padH = pool.getPadH();
    padW = pool.getPadW();
    isMax = pool.getFn() == "max";
    hasIdx = pool.getIndices() != nullptr;
    if (hasIdx && !isMax)
      return failure();
    T = blockThreads ? (int64_t)blockThreads : autoThreads(OW);
    if (T <= 0 || OW % T != 0)
      return failure();
    K = OW / T;
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Type f32 = b.getF32Type();
    Type i64 = b.getI64Type();

    SmallVector<Type> inTys;
    inTys.push_back(memrefOf(inTy));
    inTys.push_back(memrefOf(outTy));
    Type idxElt;
    if (hasIdx) {
      idxElt = cast<RankedTensorType>(pool.getIndices().getType()).getElementType();
      inTys.push_back(MemRefType::get(outTy.getShape(), idxElt));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    Value outMem = entry->getArgument(1);
    Value idxMem = hasIdx ? entry->getArgument(2) : nullptr;

    Value cNC = b.create<arith::ConstantIndexOp>(loc, N * C);
    Value cOH = b.create<arith::ConstantIndexOp>(loc, OH);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cNC, cOH, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;  // n*C + c
    Value by = launch.getBlockIds().y;  // oh
    Value tx = launch.getThreadIds().x;

    Value cC = b.create<arith::ConstantIndexOp>(loc, C);
    Value nIdx = b.create<arith::DivUIOp>(loc, bx, cC);
    Value cIdx = b.create<arith::RemUIOp>(loc, bx, cC);
    Value ohI64 = b.create<arith::IndexCastOp>(loc, i64, by);

    auto ci64 = [&](int64_t v) {
      return b.create<arith::ConstantIntOp>(loc, v, 64).getResult();
    };
    Value cHi64 = ci64(H), cWi64 = ci64(W);
    Value zero64 = ci64(0);

    for (int64_t k = 0; k < K; ++k) {
      Value ow = tx;
      if (k != 0)
        ow = b.create<arith::AddIOp>(loc, tx,
                                     b.create<arith::ConstantIndexOp>(loc, k * T));
      Value owI64 = b.create<arith::IndexCastOp>(loc, i64, ow);

      // Accumulator init.
      Value acc = isMax ? b.create<arith::ConstantOp>(
                              loc, b.getFloatAttr(
                                       f32, -std::numeric_limits<float>::infinity()))
                              .getResult()
                        : b.create<arith::ConstantOp>(loc, b.getFloatAttr(f32, 0.0))
                              .getResult();
      Value idxAcc = hasIdx ? zero64 : nullptr;

      for (int64_t kh = 0; kh < kH; ++kh) {
        for (int64_t kw = 0; kw < kW; ++kw) {
          // ih = oh*sH - padH + kh ; iw = ow*sW - padW + kw (i64, may be < 0).
          Value ihI64 = b.create<arith::AddIOp>(
              loc,
              b.create<arith::SubIOp>(
                  loc, b.create<arith::MulIOp>(loc, ohI64, ci64(sH)), ci64(padH)),
              ci64(kh));
          Value iwI64 = b.create<arith::AddIOp>(
              loc,
              b.create<arith::SubIOp>(
                  loc, b.create<arith::MulIOp>(loc, owI64, ci64(sW)), ci64(padW)),
              ci64(kw));
          Value inH = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ihI64, zero64),
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ihI64, cHi64));
          Value inW = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iwI64, zero64),
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iwI64, cWi64));
          Value cond = b.create<arith::AndIOp>(loc, inH, inW);

          SmallVector<Type> resTys{f32};
          if (hasIdx)
            resTys.push_back(i64);
          auto ifOp = b.create<scf::IfOp>(loc, resTys, cond, /*withElse=*/true);
          {
            OpBuilder tb = ifOp.getThenBodyBuilder();
            Value ihIdx = tb.create<arith::IndexCastOp>(loc, tb.getIndexType(), ihI64);
            Value iwIdx = tb.create<arith::IndexCastOp>(loc, tb.getIndexType(), iwI64);
            Value val = tb.create<memref::LoadOp>(
                loc, inMem, ValueRange{nIdx, cIdx, ihIdx, iwIdx});
            if (isMax) {
              Value isNew =
                  tb.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, val, acc);
              Value newAcc = tb.create<arith::SelectOp>(loc, isNew, val, acc);
              SmallVector<Value> yields{newAcc};
              if (hasIdx) {
                Value flat = tb.create<arith::AddIOp>(
                    loc, tb.create<arith::MulIOp>(loc, ihI64, cWi64), iwI64);
                Value newIdx =
                    tb.create<arith::SelectOp>(loc, isNew, flat, idxAcc);
                yields.push_back(newIdx);
              }
              tb.create<scf::YieldOp>(loc, yields);
            } else {
              Value sum = tb.create<arith::AddFOp>(loc, acc, val);
              tb.create<scf::YieldOp>(loc, ValueRange{sum});
            }
          }
          {
            OpBuilder eb = ifOp.getElseBodyBuilder();
            SmallVector<Value> yields{acc};
            if (hasIdx)
              yields.push_back(idxAcc);
            eb.create<scf::YieldOp>(loc, yields);
          }
          acc = ifOp.getResult(0);
          if (hasIdx)
            idxAcc = ifOp.getResult(1);
        }
      }

      Value outVal = acc;
      if (!isMax) {
        Value div = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(f32, (double)(kH * kW)));
        outVal = b.create<arith::DivFOp>(loc, acc, div);
      }
      b.create<memref::StoreOp>(loc, outVal, outMem,
                                ValueRange{nIdx, cIdx, by, ow});
      if (hasIdx) {
        Value stored = idxAcc;
        if (!idxElt.isInteger(64))
          stored = b.create<arith::TruncIOp>(loc, idxElt, idxAcc);
        b.create<memref::StoreOp>(loc, stored, idxMem,
                                  ValueRange{nIdx, cIdx, by, ow});
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentPoolToGPUPass
    : ::intent::impl::ConvertIntentPoolToGPUBase<ConvertIntentPoolToGPUPass> {
  using ConvertIntentPoolToGPUBase::ConvertIntentPoolToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      PoolLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      PoolLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("pool lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
