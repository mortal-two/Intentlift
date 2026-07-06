//===- ConvertIntentArgReduceToGPU.cpp - argmax/argmin -> gpu.launch ------===//
//
// The C6 order-statistics arg-reduce pass. Lowers a func whose body is a single
// `intent.arg_reduce` (argmax/argmin) over a rank-2 `[M,N]` static-f32 row
// (axis 1) to ONE block-cooperative `gpu.launch` returning the INTEGER INDEX of
// the row extremum ([M] or keepdims [M,1] i32): one block per row, the feature
// axis split over T threads (K = N/T CONTIGUOUS columns per thread).
//
// The new mechanism vs. the value-only `intent.reduce` (one `gpu.all_reduce`) is
// that the block reduction carries a `(best_val, best_idx)` PAIR:
//   (1) each thread scans its K columns ASCENDING with a STRICT compare, keeping
//       the extremum's (value, column) — ties keep the LOWEST column (torch's
//       first-index tie-break);
//   (2) a power-of-two shared-memory tree over two dynamic-shared buffers
//       (val[T] f32, idx[T] i32) reduces the T pairs, again keeping the lower
//       index on value ties;
//   (3) thread 0 stores idx[0] (the global first-argmax/argmin column).
//
// Static shapes, N % T == 0, T a power of two. The other passes' match() reject
// any body with an intent.arg_reduce, so this pass is disjoint and composes.
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
#define GEN_PASS_DEF_CONVERTINTENTARGREDUCETOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct ArgReduceLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match()
  ArgReduceOp arg;
  int64_t M = 0, N = 0, T = 0, K = 0;
  bool isMax = true;  // argmax vs argmin
  bool keepdims = false;
  Type f32, i32;

  ArgReduceLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    // Exactly one real op, the arg_reduce, whose result is returned.
    arg = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (auto a = dyn_cast<ArgReduceOp>(&op)) {
        if (arg)
          return failure(); // only one
        arg = a;
      } else {
        return failure();
      }
    }
    if (!arg || ret.getOperand(0) != arg.getResult())
      return failure();

    auto inTy = dyn_cast<RankedTensorType>(arg.getInput().getType());
    auto rTy = dyn_cast<RankedTensorType>(arg.getResult().getType());
    if (!inTy || !rTy || !inTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 2 || !inTy.getElementType().isF32())
      return failure();
    if (arg.getAxis() != 1)
      return failure();
    if (!rTy.getElementType().isInteger(32))
      return failure();
    // input must be the sole block arg (a bare [M,N] row tensor).
    if (body.getNumArguments() != 1 ||
        body.getArgument(0) != arg.getInput())
      return failure();

    M = inTy.getDimSize(0);
    N = inTy.getDimSize(1);
    keepdims = rTy.getRank() == 2; // [M,1] vs [M]
    isMax = arg.getFn() == "argmax";
    f32 = inTy.getElementType();
    i32 = IntegerType::get(fn.getContext(), 32);

    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || N % T != 0)
      return failure();
    K = N / T;
    if ((T & (T - 1)) != 0)
      return failure(); // power-of-two block for the shared-memory tree
    return success();
  }

  // combine two (val,idx) candidates keeping the extremum; ties -> lower idx.
  std::pair<Value, Value> combine(OpBuilder &b, Location loc, Value aV, Value aI,
                                  Value bV, Value bI) {
    arith::CmpFPredicate strict =
        isMax ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
    Value better = b.create<arith::CmpFOp>(loc, strict, aV, bV);
    Value eq = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, aV, bV);
    Value idxLower =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, aI, bI);
    Value takeA = b.create<arith::OrIOp>(
        loc, better, b.create<arith::AndIOp>(loc, eq, idxLower));
    Value val = b.create<arith::SelectOp>(loc, takeA, aV, bV);
    Value idx = b.create<arith::SelectOp>(loc, takeA, aI, bI);
    return {val, idx};
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    SmallVector<Type> inTys;
    inTys.push_back(MemRefType::get({M, N}, f32));
    inTys.push_back(keepdims ? MemRefType::get({M, 1}, i32)
                             : MemRefType::get({M}, i32));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    Value outMem = entry->getArgument(1);

    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value c0i = b.create<arith::ConstantIndexOp>(loc, 0);
    // Dynamic shared: [T] f32 values + [T] i32 indices.
    int64_t shBytes = (int64_t)T * 4 + (int64_t)T * 4;
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
    auto valTy = MemRefType::get({T}, f32, nullptr, wgSpace);
    auto idxTy = MemRefType::get({T}, i32, nullptr, wgSpace);
    Value zoff = b.create<arith::ConstantIndexOp>(loc, 0);
    Value idxByteOff = b.create<arith::ConstantIndexOp>(loc, (int64_t)T * 4);
    Value valBuf =
        b.create<memref::ViewOp>(loc, valTy, raw, zoff, ValueRange{});
    Value idxBuf =
        b.create<memref::ViewOp>(loc, idxTy, raw, idxByteOff, ValueRange{});

    // (1) per-thread local best over K CONTIGUOUS columns (col = tx*K + k).
    Value cK = b.create<arith::ConstantIndexOp>(loc, K);
    Value base = b.create<arith::MulIOp>(loc, tx, cK);
    Value bestV, bestI;
    for (int64_t k = 0; k < K; ++k) {
      Value col = base;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k);
        col = b.create<arith::AddIOp>(loc, base, off);
      }
      Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{bx, col});
      Value ci = b.create<arith::IndexCastOp>(loc, i32, col);
      if (k == 0) {
        bestV = v;
        bestI = ci;
      } else {
        auto [nv, ni] = combine(b, loc, bestV, bestI, v, ci);
        bestV = nv;
        bestI = ni;
      }
    }
    b.create<memref::StoreOp>(loc, bestV, valBuf, ValueRange{tx});
    b.create<memref::StoreOp>(loc, bestI, idxBuf, ValueRange{tx});
    b.create<gpu::BarrierOp>(loc);

    // (2) power-of-two shared-memory tree over the (val,idx) pairs.
    for (int64_t stride = T / 2; stride >= 1; stride >>= 1) {
      Value cStride = b.create<arith::ConstantIndexOp>(loc, stride);
      Value active =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx, cStride);
      auto ifOp = b.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        Value other = b.create<arith::AddIOp>(loc, tx, cStride);
        Value aV = b.create<memref::LoadOp>(loc, valBuf, ValueRange{tx});
        Value aI = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{tx});
        Value bV = b.create<memref::LoadOp>(loc, valBuf, ValueRange{other});
        Value bI = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{other});
        auto [nv, ni] = combine(b, loc, aV, aI, bV, bI);
        b.create<memref::StoreOp>(loc, nv, valBuf, ValueRange{tx});
        b.create<memref::StoreOp>(loc, ni, idxBuf, ValueRange{tx});
      }
      b.create<gpu::BarrierOp>(loc);
    }

    // (3) thread 0 stores the winning index.
    Value isLead =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, c0i);
    auto ifLead = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifLead.thenBlock());
      Value idx = b.create<memref::LoadOp>(loc, idxBuf, ValueRange{c0i});
      if (keepdims)
        b.create<memref::StoreOp>(loc, idx, outMem, ValueRange{bx, c0i});
      else
        b.create<memref::StoreOp>(loc, idx, outMem, ValueRange{bx});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentArgReduceToGPUPass
    : ::intent::impl::ConvertIntentArgReduceToGPUBase<
          ConvertIntentArgReduceToGPUPass> {
  using ConvertIntentArgReduceToGPUBase::ConvertIntentArgReduceToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ArgReduceLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ArgReduceLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("arg-reduce lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
