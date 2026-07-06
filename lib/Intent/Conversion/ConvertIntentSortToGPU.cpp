//===- ConvertIntentSortToGPU.cpp - sort/topk/argsort -> gpu.launch -------===//
//
// The C6 HARD-tier order-statistics pass — the last order-stats gap. Lowers a
// func whose body is a single `intent.sort` over a rank-2 `[M,N]` static-f32 row
// (axis 1) to ONE block-cooperative BLOCK BITONIC SORT `gpu.launch`: one block
// per row, T threads.
//
// The mechanism (vs. the value-only reductions/scans, which are a single
// tree/scan): an O(N log^2 N) barrier-separated compare-exchange NETWORK in
// dynamic-shared memory.
//   (0) each block loads its row into a `[P]`-element shared value buffer, P the
//       next power of two >= N, padding N..P-1 with a `+inf` (ascending) /
//       `-inf` (descending) SENTINEL so the pad sorts to the END. For argsort a
//       parallel `[P]` i32 index buffer is initialized `idx[i] = i`.
//   (1) the bitonic network: for k = 2,4,...,P (outer) and j = k/2,...,1 (inner)
//       — log2(P)*(log2(P)+1)/2 stages — each stage is a `gpu.barrier` then a
//       grid-stride sweep in which the thread owning the LOWER index of each
//       pair (i, i^j) compare-swaps under the direction bit (i & k)==0 (inverted
//       for descending). The i32 index buffer is swapped in lockstep.
//   (2) a final `gpu.barrier`, then thread t writes `shared[t] -> values[row,t]`
//       (and `idx[t] -> indices[row,t]`) for the first W columns (W < N = topk).
//
// The func's outputs are, in order, one memref per USED result: `values`
// `[M,W]` f32 then (if the argsort indices are returned) `indices` `[M,W]` i32.
// Static shapes, axis last, T a power of two dividing P.
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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTSORTTOGPU
#include "Intent/Conversion/Passes.h.inc"
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

struct SortLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match()
  SortOp sort;
  int64_t M = 0, N = 0, W = 0, P = 0, T = 0;
  bool descending = false;
  bool needValues = false, needIndices = false;
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

    P = nextPow2(N);
    T = blockThreads ? (int64_t)blockThreads : autoThreads(P);
    if (T <= 0 || (T & (T - 1)) != 0)
      return failure(); // T a power of two
    if (T > P || P % T != 0)
      return failure();
    return success();
  }

  Value cIdx(OpBuilder &b, Location loc, int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v);
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // New signature: input memref + one output memref per USED result, in result
    // order (values before indices).
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

    Value cM = cIdx(b, loc, M);
    Value cT = cIdx(b, loc, T);
    Value c1 = cIdx(b, loc, 1);
    Value cN = cIdx(b, loc, N);
    Value cP = cIdx(b, loc, P);
    Value cW = cIdx(b, loc, W);
    Value c0i = cIdx(b, loc, 0);

    // Dynamic shared: [P] f32 values (+ [P] i32 indices when argsort is needed).
    int64_t shBytes = (int64_t)P * 4 + (needIndices ? (int64_t)P * 4 : 0);
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
    auto valTy = MemRefType::get({P}, f32, nullptr, wgSpace);
    auto idxTy = MemRefType::get({P}, i32, nullptr, wgSpace);
    Value valBuf =
        b.create<memref::ViewOp>(loc, valTy, raw, cIdx(b, loc, 0), ValueRange{});
    Value idxBuf;
    if (needIndices)
      idxBuf = b.create<memref::ViewOp>(loc, idxTy, raw, cIdx(b, loc, P * 4),
                                        ValueRange{});

    // Sentinel: +inf (ascending) / -inf (descending) so the pad sorts to the END.
    APFloat inf = APFloat::getInf(APFloat::IEEEsingle(), /*neg=*/descending);
    Value sentinel = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(
                                                          inf.convertToFloat()));

    // (0) LOAD: each thread grid-strides i in [tx, P) step T; real cols loaded
    //     from the row, pads set to the sentinel; idx[i] = i.
    {
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

    // (1) BITONIC NETWORK. for k = 2,4,...,P (outer); j = k/2,...,1 (inner).
    //     log2(P)*(log2(P)+1)/2 barrier-separated compare-exchange stages.
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
        // Only the lower member of each pair performs the swap.
        Value lower =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, ixj, i);
        auto owns = b.create<scf::IfOp>(loc, lower, /*withElse=*/false);
        {
          OpBuilder::InsertionGuard g2(b);
          b.setInsertionPointToStart(owns.thenBlock());
          // direction bit: ascending block iff (i & k)==0; XOR descending.
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
          // ascend: swap when vi > vj; descending block: swap when vi < vj.
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

    // (2) final barrier, then write the first W sorted entries out.
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
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    (void)c0i;
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
