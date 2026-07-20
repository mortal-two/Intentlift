//===- ConvertIntentLayoutToGPU.cpp - structural layout -> gpu.launch -----===//
//
// The C4 structural-layout pass. Lowers a func whose body is exactly ONE
// `intent.concat` / `intent.pad` / `intent.tile` / `intent.flip` / `intent.triu`
// over rank-2 operands to a single block-cooperative `gpu.launch`: one block per
// OUTPUT row, the output column axis split over the block's T threads (K =
// cols/T columns each, unrolled).
//
// The mechanism (vs. the data-dependent gather) is pure DATA MOVEMENT: each
// output element's SOURCE coordinate is a STATIC structural function of the
// output coord (an affine index / a piecewise-by-offset select), then a
// `memref.load` (or the pad constant) — no `index` stream is read at runtime.
//
//   concat: piecewise by the cumulative axis offsets — a per-input
//           `scf.if(off <= coord < off+extent)` loads that input at the
//           offset-shifted coord and stores (variadic + offset placement).
//   pad   : `scf.if(in-bounds)` yields `input[coord-lo]` else the pad `value`.
//   tile  : `input[coord mod input_extent]` via `arith.remui` (no guard).
//   flip  : `input[N-1-coord]` on the flipped axes (affine remap).
//   triu  : `arith.select(col >= row + k, input[row,col], 0)`.
//
// Destination-passing: new signature = (inputs as memrefs) ++ [Ro,Co] out memref.
// Correctness-only (pure movement -> bit-faithful). Static shapes, cols % T == 0
// (tail-mask is a follow-up). The other passes' match() reject any body with a
// layout op, so this pass is disjoint and composes freely.
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
#define GEN_PASS_DEF_CONVERTINTENTLAYOUTTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's structural-layout lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct LayoutLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match(): exactly one of these is set.
  ConcatOp concat;
  PadOp pad;
  TileOp tile;
  FlipOp flip;
  TriuOp triu;

  RankedTensorType outTy;
  // Iteration geometry over the OUTPUT: one block per row (Ro), T threads over
  // Co columns, K columns per thread.
  int64_t Ro = 0, Co = 0, T = 0, K = 0;

  LayoutLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult setGeometry(RankedTensorType o) {
    outTy = o;
    if (o.getRank() != 2 || !o.hasStaticShape())
      return failure();
    Ro = o.getDimSize(0);
    Co = o.getDimSize(1);
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
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    if (!only || ret.getOperand(0) != only->getResult(0))
      return failure();

    auto rank2 = [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && t.getRank() == 2;
    };

    if (auto c = dyn_cast<ConcatOp>(only)) {
      concat = c;
      for (Value in : c.getInputs())
        if (!rank2(in))
          return failure();
      return setGeometry(cast<RankedTensorType>(c.getResult().getType()));
    }
    if (auto p = dyn_cast<PadOp>(only)) {
      pad = p;
      if (!rank2(p.getInput()))
        return failure();
      return setGeometry(cast<RankedTensorType>(p.getResult().getType()));
    }
    if (auto t = dyn_cast<TileOp>(only)) {
      tile = t;
      if (!rank2(t.getInput()))
        return failure();
      return setGeometry(cast<RankedTensorType>(t.getResult().getType()));
    }
    if (auto f = dyn_cast<FlipOp>(only)) {
      flip = f;
      if (!rank2(f.getInput()))
        return failure();
      return setGeometry(cast<RankedTensorType>(f.getResult().getType()));
    }
    if (auto tr = dyn_cast<TriuOp>(only)) {
      triu = tr;
      if (!rank2(tr.getInput()))
        return failure();
      return setGeometry(cast<RankedTensorType>(tr.getResult().getType()));
    }
    return failure();
  }

  // ---- emission ---------------------------------------------------------- //
  static SmallVector<int64_t> i64Array(ArrayAttr a) {
    SmallVector<int64_t> v;
    for (Attribute e : a)
      v.push_back(cast<IntegerAttr>(e).getInt());
    return v;
  }

  // Body op operands (the tensor inputs), in order.
  SmallVector<Value> operands() {
    if (concat)
      return SmallVector<Value>(concat.getInputs().begin(),
                                concat.getInputs().end());
    Value in = pad ? pad.getInput()
                   : tile ? tile.getInput()
                          : flip ? flip.getInput() : triu.getInput();
    return {in};
  }

  // Store `v` at out[row, col].
  void store(OpBuilder &b, Location loc, Value v, Value outMem, Value row,
             Value col) {
    b.create<memref::StoreOp>(loc, v, outMem, ValueRange{row, col});
  }

  void emitConcat(OpBuilder &b, Location loc, ValueRange inMems, Value outMem,
                  Value row, Value col) {
    int64_t axis = concat.getAxis();
    int64_t off = 0;
    for (auto [i, in] : llvm::enumerate(concat.getInputs())) {
      auto iTy = cast<RankedTensorType>(in.getType());
      int64_t ext = iTy.getDimSize(axis);
      Value pos = axis == 0 ? row : col;
      Value cOff = b.create<arith::ConstantIndexOp>(loc, off);
      Value cEnd = b.create<arith::ConstantIndexOp>(loc, off + ext);
      Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, pos, cOff);
      Value lt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pos, cEnd);
      Value cond = b.create<arith::AndIOp>(loc, ge, lt);
      auto ifOp = b.create<scf::IfOp>(loc, cond, /*withElse=*/false);
      OpBuilder tb = ifOp.getThenBodyBuilder();
      Value shifted = tb.create<arith::SubIOp>(loc, pos, cOff);
      Value r = axis == 0 ? shifted : row;
      Value c = axis == 1 ? shifted : col;
      Value v = tb.create<memref::LoadOp>(loc, inMems[i], ValueRange{r, c});
      tb.create<memref::StoreOp>(loc, v, outMem, ValueRange{row, col});
      off += ext;
    }
  }

  void emitPad(OpBuilder &b, Location loc, Value inMem, Value outMem, Value row,
               Value col) {
    auto iTy = cast<RankedTensorType>(pad.getInput().getType());
    auto lo = i64Array(pad.getLo());
    Type elt = iTy.getElementType();
    // in-bounds predicate: lo[d] <= coord[d] < lo[d]+in[d] for d in {0,1}.
    Value cond;
    Value srcCoord[2];
    for (int d = 0; d < 2; ++d) {
      Value pos = d == 0 ? row : col;
      Value cLo = b.create<arith::ConstantIndexOp>(loc, lo[d]);
      Value cHi =
          b.create<arith::ConstantIndexOp>(loc, lo[d] + iTy.getDimSize(d));
      Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, pos, cLo);
      Value lt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pos, cHi);
      Value in = b.create<arith::AndIOp>(loc, ge, lt);
      cond = d == 0 ? in : b.create<arith::AndIOp>(loc, cond, in);
      srcCoord[d] = b.create<arith::SubIOp>(loc, pos, cLo);
    }
    Value padVal = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(elt, pad.getValue().convertToDouble()));
    auto ifOp = b.create<scf::IfOp>(loc, TypeRange{elt}, cond,
                                    /*withElse=*/true);
    OpBuilder tb = ifOp.getThenBodyBuilder();
    Value v = tb.create<memref::LoadOp>(loc, inMem,
                                        ValueRange{srcCoord[0], srcCoord[1]});
    tb.create<scf::YieldOp>(loc, v);
    OpBuilder eb = ifOp.getElseBodyBuilder();
    eb.create<scf::YieldOp>(loc, padVal);
    store(b, loc, ifOp.getResult(0), outMem, row, col);
  }

  void emitTile(OpBuilder &b, Location loc, Value inMem, Value outMem, Value row,
                Value col) {
    auto iTy = cast<RankedTensorType>(tile.getInput().getType());
    Value ci0 = b.create<arith::ConstantIndexOp>(loc, iTy.getDimSize(0));
    Value ci1 = b.create<arith::ConstantIndexOp>(loc, iTy.getDimSize(1));
    Value r = b.create<arith::RemUIOp>(loc, row, ci0);
    Value c = b.create<arith::RemUIOp>(loc, col, ci1);
    Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{r, c});
    store(b, loc, v, outMem, row, col);
  }

  void emitFlip(OpBuilder &b, Location loc, Value inMem, Value outMem, Value row,
                Value col) {
    auto iTy = cast<RankedTensorType>(flip.getInput().getType());
    auto axes = i64Array(flip.getAxes());
    bool f0 = llvm::is_contained(axes, 0), f1 = llvm::is_contained(axes, 1);
    auto rev = [&](Value pos, int64_t n) -> Value {
      Value nm1 = b.create<arith::ConstantIndexOp>(loc, n - 1);
      return b.create<arith::SubIOp>(loc, nm1, pos);
    };
    Value r = f0 ? rev(row, iTy.getDimSize(0)) : row;
    Value c = f1 ? rev(col, iTy.getDimSize(1)) : col;
    Value v = b.create<memref::LoadOp>(loc, inMem, ValueRange{r, c});
    store(b, loc, v, outMem, row, col);
  }

  void emitTriu(OpBuilder &b, Location loc, Value inMem, Value outMem, Value row,
                Value col) {
    Type elt = cast<RankedTensorType>(triu.getInput().getType()).getElementType();
    // cond = col >= row + k, computed in i64 so a negative k is signed-correct.
    Value rowI = b.create<arith::IndexCastOp>(loc, b.getI64Type(), row);
    Value colI = b.create<arith::IndexCastOp>(loc, b.getI64Type(), col);
    Value kI = b.create<arith::ConstantIntOp>(loc, triu.getK(), 64);
    Value diag = b.create<arith::AddIOp>(loc, rowI, kI);
    Value cond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, colI, diag);
    Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elt, 0.0));
    // GUARD the load: only read the input inside the kept (upper) triangle. The
    // old form read the full N^2 then `select`ed the lower triangle to 0, paying
    // 2x the read bandwidth (the measured ~2x loss vs a triangle-aware library);
    // reading only where kept halves the input traffic. Warps diverge only on
    // the diagonal band; whole below/above blocks stay uniform (all-skip / all-
    // load), so coalescing is preserved off the diagonal.
    auto ifOp = b.create<scf::IfOp>(loc, TypeRange{elt}, cond, /*withElse=*/true);
    OpBuilder tb = ifOp.getThenBodyBuilder();
    Value v = tb.create<memref::LoadOp>(loc, inMem, ValueRange{row, col});
    tb.create<scf::YieldOp>(loc, v);
    OpBuilder eb = ifOp.getElseBodyBuilder();
    eb.create<scf::YieldOp>(loc, zero);
    store(b, loc, ifOp.getResult(0), outMem, row, col);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    SmallVector<Value> ins = operands();
    SmallVector<Type> inTys;
    for (Value in : ins)
      inTys.push_back(memrefOf(cast<RankedTensorType>(in.getType())));
    inTys.push_back(memrefOf(outTy)); // appended destination
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    SmallVector<Value> inMems;
    for (unsigned i = 0; i < ins.size(); ++i)
      inMems.push_back(entry->getArgument(i));
    Value outMem = entry->getArgument(ins.size());

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
        Value offc = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, offc);
      }
      if (concat)
        emitConcat(b, loc, inMems, outMem, bx, col);
      else if (pad)
        emitPad(b, loc, inMems[0], outMem, bx, col);
      else if (tile)
        emitTile(b, loc, inMems[0], outMem, bx, col);
      else if (flip)
        emitFlip(b, loc, inMems[0], outMem, bx, col);
      else
        emitTriu(b, loc, inMems[0], outMem, bx, col);
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentLayoutToGPUPass
    : ::intent::impl::ConvertIntentLayoutToGPUBase<
          ConvertIntentLayoutToGPUPass> {
  using ConvertIntentLayoutToGPUBase::ConvertIntentLayoutToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      LayoutLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      LayoutLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("layout lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
