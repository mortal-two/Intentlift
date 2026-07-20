//===- ConvertIntentRngToGPU.cpp - seeded philox RNG -> gpu.launch --------===//
//
// The C5 stateful-RNG pass. Lowers a func whose body is a straight-line SCALAR
// elementwise graph over rank-2 `[Ro,Co]` operands containing at least one
// `intent.rand` to a single block-cooperative `gpu.launch`: one block per OUTPUT
// row, the output column axis split over the block's T threads (K = cols/T
// columns each, unrolled).
//
// The new mechanism vs. the pure-elementwise pass is a STATELESS counter-based
// RNG: each output element at `(row,col)` computes
// `philox4x32-10(seed, row*Co + col)` entirely in `arith` (10 rounds of integer
// mul-hi/xor over the public philox constants; the seed splits into the two key
// words, the element index is the counter), maps the first output word to a
// 24-bit-mantissa uniform[0,1) f32, then runs the composed scalar transform
// (map_unary / map_binary incl. compare / select / arith.constant splat, reusing
// the C0/M3 scalar emitters) and stores. No global RNG state, no visited-order
// dependence — same seed -> bit-identical, different seed -> decorrelated.
//
//   rand / rand_like : one rand -> uniform, stored.
//   dropout          : select(rand >= p, x * 1/(1-p), 0).
//   randn / normal   : Box-Muller of two rands (sqrt(-2 ln u1) * cos(2pi u2)).
//   uniform(a,b)     : a + (b-a)*rand.
//
// Destination-passing: new signature = (tensor inputs as memrefs) ++ [Ro,Co] out
// memref. Verification is matched-seed vs the interpreter's identical philox
// (determinism) and STATISTICAL vs torch/flag_gems (the philox counter layout is
// NOT Triton's tl.rand layout, so library parity is distributional). Static
// shapes, cols % T == 0 (tail-mask is a follow-up). The other passes' match()
// reject any body with an intent.rand, so this pass is disjoint and composes.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTRNGTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Public philox4x32 constants (Salmon et al. 2011).
static constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
static constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
static constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
static constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

/// One RNG func's lowering. Reads the original (tensor) func body, builds a new
/// memref+gpu.launch func with the same name, erases the old.
struct RngLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  RankedTensorType outTy;
  Value retVal;                    // the tensor value returned
  SmallVector<Value> argVals;      // rank-2 tensor func args (in order)
  int64_t Ro = 0, Co = 0, T = 0, K = 0;

  RngLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  // ---- recognition ------------------------------------------------------- //
  static bool isKnownOp(Operation *op) {
    if (isa<RandOp, MapUnaryOp, MapBinaryOp, SelectOp>(op))
      return true;
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto dea = dyn_cast<DenseElementsAttr>(c.getValue());
      return dea && dea.isSplat();
    }
    return false;
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    retVal = ret.getOperand(0);
    outTy = dyn_cast<RankedTensorType>(retVal.getType());
    if (!outTy || outTy.getRank() != 2 || !outTy.hasStaticShape() ||
        !outTy.getElementType().isF32())
      return failure();
    Ro = outTy.getDimSize(0);
    Co = outTy.getDimSize(1);
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Co);
    if (T <= 0 || Co % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented follow-up
    K = Co / T;

    bool hasRand = false;
    for (Operation &op : body.without_terminator()) {
      if (!isKnownOp(&op))
        return failure();
      if (isa<RandOp>(op))
        hasRand = true;
    }
    if (!hasRand)
      return failure(); // only fire on genuine RNG funcs

    // Tensor func args must all be rank-2 [Ro,Co] f32 (e.g. dropout's x).
    for (BlockArgument arg : body.getArguments()) {
      auto t = dyn_cast<RankedTensorType>(arg.getType());
      if (!t || t.getShape() != outTy.getShape() ||
          !t.getElementType().isF32())
        return failure();
      argVals.push_back(arg);
    }
    return success();
  }

  // ---- philox ------------------------------------------------------------ //
  // Two independent uniforms in [0,1) from philox4x32-10(seed, idx).
  std::pair<Value, Value> emitPhilox(OpBuilder &b, Location loc, Value idxI32,
                                     uint64_t seed) {
    Type i32 = b.getI32Type(), i64 = b.getI64Type();
    auto ci = [&](uint32_t v) {
      return b.create<arith::ConstantIntOp>(loc, (int64_t)(int32_t)v, 32)
          .getResult();
    };
    Value c0 = idxI32, c1 = ci(0), c2 = ci(0), c3 = ci(0);
    Value k0 = ci((uint32_t)(seed & 0xFFFFFFFFu));
    Value k1 = ci((uint32_t)(seed >> 32));
    Value M0 = ci(kPhiloxM0), M1 = ci(kPhiloxM1);
    Value W0 = ci(kPhiloxW0), W1 = ci(kPhiloxW1);
    Value sh32 = b.create<arith::ConstantIntOp>(loc, 32, 64);
    auto mulhilo = [&](Value x, Value y) -> std::pair<Value, Value> {
      Value xe = b.create<arith::ExtUIOp>(loc, i64, x);
      Value ye = b.create<arith::ExtUIOp>(loc, i64, y);
      Value prod = b.create<arith::MulIOp>(loc, xe, ye);
      Value lo = b.create<arith::TruncIOp>(loc, i32, prod);
      Value hi =
          b.create<arith::TruncIOp>(loc, i32,
                                    b.create<arith::ShRUIOp>(loc, prod, sh32));
      return {hi, lo};
    };
    for (int r = 0; r < 10; ++r) {
      auto [hi0, lo0] = mulhilo(M0, c0);
      auto [hi1, lo1] = mulhilo(M1, c2);
      Value nc0 = b.create<arith::XOrIOp>(
          loc, b.create<arith::XOrIOp>(loc, hi1, c1), k0);
      Value nc2 = b.create<arith::XOrIOp>(
          loc, b.create<arith::XOrIOp>(loc, hi0, c3), k1);
      c0 = nc0;
      c1 = lo1;
      c2 = nc2;
      c3 = lo0;
      if (r < 9) {
        k0 = b.create<arith::AddIOp>(loc, k0, W0);
        k1 = b.create<arith::AddIOp>(loc, k1, W1);
      }
    }
    // top 24 bits -> [0,1) f32 (matches uitofp(c>>8) * 2^-24 in the interpreter).
    Value sh8 = ci(8);
    auto unit = [&](Value c) -> Value {
      Value top = b.create<arith::ShRUIOp>(loc, c, sh8);
      Value f = b.create<arith::UIToFPOp>(loc, b.getF32Type(), top);
      Value scale = b.create<arith::ConstantOp>(
          loc, b.getF32FloatAttr(1.0f / 16777216.0f));
      return b.create<arith::MulFOp>(loc, f, scale);
    };
    return {unit(c0), unit(c1)};
  }

  // ---- per-element evaluation -------------------------------------------- //
  // Compute out[row,col] by walking the scalar DAG once. `idxI32` = row*Co+col.
  Value emitElement(OpBuilder &b, Location loc, ValueRange inMems, Value row,
                    Value col, Value idxI32) {
    DenseMap<Value, Value> sc;
    for (auto [i, arg] : llvm::enumerate(argVals))
      sc[arg] = b.create<memref::LoadOp>(loc, inMems[i], ValueRange{row, col});

    Block &body = fn.getBody().front();
    for (Operation &op : body.without_terminator()) {
      Value res = op.getResult(0);
      if (auto c = dyn_cast<arith::ConstantOp>(&op)) {
        auto dea = cast<DenseElementsAttr>(c.getValue());
        sc[res] = b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(
                     dea.getSplatValue<APFloat>().convertToFloat()));
      } else if (auto rnd = dyn_cast<RandOp>(&op)) {
        sc[res] = emitPhilox(b, loc, idxI32, (uint64_t)rnd.getSeed()).first;
      } else if (auto mu = dyn_cast<MapUnaryOp>(&op)) {
        sc[res] = emitScalarUnary(b, loc, mu.getFn(), sc[mu.getInput()]);
      } else if (auto mb = dyn_cast<MapBinaryOp>(&op)) {
        Value x = sc[mb.getLhs()], y = sc[mb.getRhs()];
        StringRef f = mb.getFn();
        Value v = emitScalarBinary(b, loc, f, x, y);
        if (!v)
          v = emitScalarCmp(b, loc, f, x, y);
        if (!v && f == "ne") // numpy `ne`: NaN-true (unordered)
          v = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, x, y);
        sc[res] = v;
      } else if (auto sel = dyn_cast<SelectOp>(&op)) {
        sc[res] = b.create<arith::SelectOp>(loc, sc[sel.getCond()],
                                            sc[sel.getX()], sc[sel.getY()]);
      }
    }
    return sc[retVal];
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    SmallVector<Type> inTys;
    for (Value in : argVals)
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
    for (unsigned i = 0; i < argVals.size(); ++i)
      inMems.push_back(entry->getArgument(i));
    Value outMem = entry->getArgument(argVals.size());

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value cCo = b.create<arith::ConstantIndexOp>(loc, Co);
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
      // linear element index row*Co + col, as i32 (the philox counter).
      Value lin = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, bx, cCo), col);
      Value idxI32 =
          b.create<arith::IndexCastOp>(loc, b.getI32Type(), lin);
      Value v = emitElement(b, loc, inMems, bx, col, idxI32);
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentRngToGPUPass
    : ::intent::impl::ConvertIntentRngToGPUBase<ConvertIntentRngToGPUPass> {
  using ConvertIntentRngToGPUBase::ConvertIntentRngToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      RngLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      RngLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("rng lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
