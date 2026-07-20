//===- ConvertIntentGatherToGPU.cpp - gather/scatter -> gpu.launch --------===//
//
// The C3 data-dependent-addressing pass. Lowers a func whose body is exactly ONE
// `intent.gather` (torch.gather / index_select / embedding row-gather) or ONE
// `intent.scatter` (torch.scatter / scatter_add_) over rank-2 data to a single
// block-cooperative `gpu.launch`: one block per output/scatter row, the column
// axis split over the block's T threads (K = cols/T columns each, unrolled).
//
// The NEW mechanism vs. the elementwise/reduce passes: the per-element access
// coordinate on `data` is a RUNTIME value read from the integer `index` memref —
// `idx = index_cast(memref.load index[...])` — used directly as a
// `memref.load`/`memref.store`/`memref.atomic_rmw` coordinate on `data`. This is
// a data-dependent gather/scatter address, not an affine function of the loop.
//
//   gather : destination-passing. new signature = (data, index) ++ [Ro,Co] out.
//            out[bx,col] = data[<coord with the index axis read from index>].
//   scatter: IN-PLACE on the (caller-cloned) data memref. new signature =
//            (data, index, src). data[<coord with axis=index[bx,col]>] (op)=
//            src[bx,col]; op = store (reduce none) or atomic_rmw addf (add).
//
// Correctness-only (the index stream is exact -> bit-faithful). Static shapes,
// cols % T == 0 (tail-mask is a follow-up). The two match() predicates for the
// existing passes reject any body with a gather/scatter op (they only accept
// map/reduce/select/iota), so this pass is disjoint and composes freely.
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTGATHERTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's gather/scatter lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct GatherScatterLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match()
  GatherOp gather;    // set for a gather func
  ScatterOp scatter;  // set for a scatter func
  int64_t axis = 0;
  // Iteration geometry: one block per row (Ro), T threads over Co columns, K
  // columns per thread. For gather these are the OUTPUT dims; for scatter the
  // SRC/index dims (that is what drives the writes).
  int64_t Ro = 0, Co = 0, T = 0, K = 0;

  GatherScatterLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult setGeometry(int64_t rows, int64_t cols) {
    Ro = rows;
    Co = cols;
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Co);
    if (T <= 0 || Co % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented follow-up
    K = Co / T;
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
    // Exactly one non-terminator op, and it produces the returned value.
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    if (!only || ret.getOperand(0) != only->getResult(0))
      return failure();

    if (auto g = dyn_cast<GatherOp>(only)) {
      gather = g;
      auto rTy = cast<RankedTensorType>(g.getResult().getType());
      if (rTy.getRank() != 2 || !rTy.hasStaticShape())
        return failure();
      if (cast<RankedTensorType>(g.getData().getType()).getRank() != 2)
        return failure();
      axis = g.getAxis();
      return setGeometry(rTy.getDimSize(0), rTy.getDimSize(1));
    }
    if (auto s = dyn_cast<ScatterOp>(only)) {
      scatter = s;
      auto dTy = cast<RankedTensorType>(s.getData().getType());
      auto srcTy = cast<RankedTensorType>(s.getSrc().getType());
      if (dTy.getRank() != 2 || srcTy.getRank() != 2 || !srcTy.hasStaticShape())
        return failure();
      axis = s.getAxis();
      return setGeometry(srcTy.getDimSize(0), srcTy.getDimSize(1));
    }
    return failure();
  }

  // ---- emission ---------------------------------------------------------- //
  // Load index[coord] and cast it to an `index`-typed value for use as a
  // data-memref coordinate. `coord` is 1-D (take: index is rank-1) or 2-D.
  Value loadIndex(OpBuilder &b, Location loc, Value idxMem, ValueRange coord) {
    Value raw = b.create<memref::LoadOp>(loc, idxMem, coord);
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), raw);
  }

  LogicalResult rewriteGather() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(gather.getData().getType());
    auto idxTy = cast<RankedTensorType>(gather.getIndex().getType());
    auto outTy = cast<RankedTensorType>(gather.getResult().getType());
    bool alongAxis = gather.getMode() == "along_axis";

    // New signature: data, index (as memrefs) ++ one [Ro,Co] output memref.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(outTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value outMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      // Data-dependent load coordinate: read the index, replace the `axis`
      // coordinate of the data access with it.
      Value idx;
      if (alongAxis) // index is [Ro,Co]: read at the output coordinate
        idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      else           // take: index is rank-1, selecting along `axis`
        idx = loadIndex(b, loc, idxMem, ValueRange{axis == 0 ? bx : col});
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      Value v = b.create<memref::LoadOp>(loc, dataMem, ValueRange{r0, r1});
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewriteScatter() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(scatter.getData().getType());
    auto idxTy = cast<RankedTensorType>(scatter.getIndex().getType());
    auto srcTy = cast<RankedTensorType>(scatter.getSrc().getType());
    bool add = scatter.getReduce() == "add";

    // New signature: data (in/out, caller-cloned), index, src — all memrefs.
    // Scatter is IN-PLACE on `data`: no appended output arg.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(srcTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value srcMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      Value v = b.create<memref::LoadOp>(loc, srcMem, ValueRange{bx, col});
      Value idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      // Data-dependent store coordinate on `data`: replace the `axis` coord.
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      if (add)
        b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v,
                                      dataMem, ValueRange{r0, r1});
      else
        b.create<memref::StoreOp>(loc, v, dataMem, ValueRange{r0, r1});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (gather)
      return rewriteGather();
    return rewriteScatter();
  }
};

struct ConvertIntentGatherToGPUPass
    : ::intent::impl::ConvertIntentGatherToGPUBase<
          ConvertIntentGatherToGPUPass> {
  using ConvertIntentGatherToGPUBase::ConvertIntentGatherToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      GatherScatterLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      GatherScatterLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("gather/scatter lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
