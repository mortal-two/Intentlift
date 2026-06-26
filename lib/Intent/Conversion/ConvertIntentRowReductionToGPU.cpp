//===- ConvertIntentRowReductionToGPU.cpp - row reduce -> gpu.launch ------===//
//
// Lowers a "row normalization" intent func (a straight-line map/reduce graph
// whose reduces are all over the last axis, keepdims) to ONE read-once,
// block-cooperative `gpu.launch`: one block per row, the feature dim split over
// the block's threads, each thread caching its columns once, one standalone
// `gpu.all_reduce` per reduce. This is the verified path (research/
// 29-kernel-characterization.md + the rms-norm-pk probe); it is general over
// the DAG — there is NO per-kernel "recognize rms_norm" branch. rms_norm
// (one mean reduce) and softmax (max then sum) flow through identically.
//
// Schedule per block (row `bx`), per thread `tx` owning columns col = tx + k*T:
//   alloca cache_i : memref<Kxf32>   (one per row-shaped global input)
//   load uniforms (rank-0 scalar args) once
//   for each reduce R in program order:
//     %p = scf.for k=0..K iter_args(acc=ident):
//            [pass 0 only] load every row input X[bx,col] -> cache_i[k]
//            acc = combine(acc, <R.input evaluated at col>)
//     %s = gpu.all_reduce <op> %p          (mean: %s = %s / N)
//     hoist every now-uniform map op once (per-row scalars: +eps, sqrt, recip…)
//   epilogue: for k=0..K: store <output evaluated at col> -> Y[bx,col]
//
// Read-once (cache, not a 2nd global load of X) is the whole game — a 2-global-
// pass version pays 1.5x traffic. Static shapes only (bare-ptr ABI). Emits
// math.exp/sqrt/rsqrt; the `-lower-gpu-math-to-nvvm` peephole turns those into
// NVVM approx intrinsics (libdevice externs are a ptxas dead end).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTROWREDUCTIONTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// Scalar arith/math op for one `map_unary` fn. nullptr = unsupported here.
static Value emitScalarUnary(OpBuilder &b, Location loc, StringRef fn, Value x) {
  if (fn == "exp")
    return b.create<math::ExpOp>(loc, x);
  if (fn == "log")
    return b.create<math::LogOp>(loc, x);
  if (fn == "sqrt")
    return b.create<math::SqrtOp>(loc, x);
  if (fn == "rsqrt")
    return b.create<math::RsqrtOp>(loc, x);
  if (fn == "abs")
    return b.create<math::AbsFOp>(loc, x);
  if (fn == "neg")
    return b.create<arith::NegFOp>(loc, x);
  if (fn == "tanh")
    return b.create<math::TanhOp>(loc, x);
  if (fn == "recip") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::DivFOp>(loc, one, x);
  }
  if (fn == "sigmoid") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
    return b.create<arith::DivFOp>(loc, one, b.create<arith::AddFOp>(loc, one, e));
  }
  return nullptr;
}

/// Scalar arith op for one `map_binary` fn. nullptr = unsupported here.
static Value emitScalarBinary(OpBuilder &b, Location loc, StringRef fn, Value a,
                              Value c) {
  if (fn == "add")
    return b.create<arith::AddFOp>(loc, a, c);
  if (fn == "sub")
    return b.create<arith::SubFOp>(loc, a, c);
  if (fn == "mul")
    return b.create<arith::MulFOp>(loc, a, c);
  if (fn == "div")
    return b.create<arith::DivFOp>(loc, a, c);
  if (fn == "max")
    return b.create<arith::MaxNumFOp>(loc, a, c);
  if (fn == "min")
    return b.create<arith::MinNumFOp>(loc, a, c);
  if (fn == "pow")
    return b.create<math::PowFOp>(loc, a, c);
  return nullptr;
}

/// Block reduction op + per-thread identity for a `reduce` fn.
struct ReduceInfo {
  gpu::AllReduceOperation op;
  double ident;
  bool isMean;
};
static std::optional<ReduceInfo> reduceInfoFor(StringRef fn) {
  if (fn == "sum")
    return ReduceInfo{gpu::AllReduceOperation::ADD, 0.0, false};
  if (fn == "mean")
    return ReduceInfo{gpu::AllReduceOperation::ADD, 0.0, true};
  if (fn == "prod")
    return ReduceInfo{gpu::AllReduceOperation::MUL, 1.0, false};
  if (fn == "max")
    return ReduceInfo{gpu::AllReduceOperation::MAXNUMF,
                      -std::numeric_limits<double>::infinity(), false};
  if (fn == "min")
    return ReduceInfo{gpu::AllReduceOperation::MINNUMF,
                      std::numeric_limits<double>::infinity(), false};
  return std::nullopt;
}

static Value combineScalar(OpBuilder &b, Location loc, gpu::AllReduceOperation op,
                           Value a, Value c) {
  switch (op) {
  case gpu::AllReduceOperation::ADD:
    return b.create<arith::AddFOp>(loc, a, c);
  case gpu::AllReduceOperation::MUL:
    return b.create<arith::MulFOp>(loc, a, c);
  case gpu::AllReduceOperation::MAXNUMF:
    return b.create<arith::MaxNumFOp>(loc, a, c);
  case gpu::AllReduceOperation::MINNUMF:
    return b.create<arith::MinNumFOp>(loc, a, c);
  default:
    return nullptr;
  }
}

/// Per-arg semantic role, inferred purely from static shape vs (M,N).
enum class Role { Row, Col, Scalar };

/// One func's row-reduction lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct RowReductionLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // populated by match()
  int64_t M = 0, N = 0, T = 0, K = 0;
  Type f32;
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<ReduceOp> reduces;     // in program order
  Value outputVal;                   // the returned [M,N] value

  RowReductionLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 1)
      return failure();
    auto outTy = dyn_cast<RankedTensorType>(fty.getResult(0));
    if (!outTy || outTy.getRank() != 2 || !outTy.hasStaticShape())
      return failure();
    f32 = outTy.getElementType();
    if (!isa<Float32Type>(f32))
      return failure();
    M = outTy.getDimSize(0);
    N = outTy.getDimSize(1);

    // Single basic block, terminated by a return of the output value.
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 1)
      return failure();
    outputVal = ret.getOperand(0);

    // Args: classify each by shape; only Row/Col/Scalar f32 tensors allowed.
    for (Type at : fty.getInputs()) {
      auto t = dyn_cast<RankedTensorType>(at);
      if (!t || !t.hasStaticShape() || t.getElementType() != f32)
        return failure();
      ArrayRef<int64_t> s = t.getShape();
      if (s.size() == 2 && s[0] == M && s[1] == N)
        argRoles.push_back(Role::Row);
      else if (s.size() == 1 && s[0] == N)
        argRoles.push_back(Role::Col);
      else if (s.size() == 0)
        argRoles.push_back(Role::Scalar);
      else
        return failure(); // [M,1] etc. as an *input* not supported in M1
    }

    // Body ops: only map_unary/map_binary/reduce; reduces last-axis keepdims.
    for (Operation &op : body.without_terminator()) {
      if (auto r = dyn_cast<ReduceOp>(&op)) {
        auto inTy = cast<RankedTensorType>(r.getInput().getType());
        auto rTy = cast<RankedTensorType>(r.getResult().getType());
        if (inTy.getRank() != 2 || rTy.getRank() != 2)
          return failure();
        if (r.getAxes().size() != 1)
          return failure();
        int64_t ax = cast<IntegerAttr>(r.getAxes()[0]).getInt();
        if (ax != 1 || rTy.getDimSize(0) != M || rTy.getDimSize(1) != 1)
          return failure();
        if (!reduceInfoFor(r.getFn()))
          return failure();
        reduces.push_back(r);
      } else if (!isa<MapUnaryOp, MapBinaryOp>(&op)) {
        return failure();
      }
    }
    if (reduces.empty())
      return failure(); // pure elementwise -> not our pass

    // Geometry: T threads over the feature dim, K columns per thread.
    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || N % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented M1 follow-up
    K = N / T;
    return success();
  }

  static int64_t autoThreads(int64_t N) {
    int64_t t = 1;
    while (t * 2 <= N && t * 2 <= 256)
      t *= 2;
    return t;
  }

  // ---- emission ---------------------------------------------------------- //
  // The feature tile is held in REGISTERS, not a memref cache: K is static and
  // small, so we fully unroll over k and keep each Row input's K loaded values
  // as SSA (exactly liger's register tile). A dynamically-indexed memref.alloca
  // would land in local memory (no mem2reg) and cap bandwidth ~40x below peak.
  DenseMap<Value, Value> uniform;                // value -> hoisted per-row scalar
  DenseMap<Value, Value> argMemref;              // Row/Col/Scalar arg -> memref arg
  DenseMap<Value, SmallVector<Value>> rowVals;   // Row arg -> K loaded SSA values
  SmallVector<Value> cols;                       // K column indices (tx + k*T)

  /// Evaluate an original value for the thread's k-th column (fully unrolled).
  /// Row inputs resolve to the once-loaded register value; uniforms to their
  /// hoisted per-row scalar; Col inputs (W) are loaded at col[k].
  Value emitAt(OpBuilder &b, Location loc, Value v, unsigned k) {
    if (Value u = uniform.lookup(v))
      return u;
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Row)
        return rowVals[v][k];
      if (role == Role::Col)
        return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{cols[k]});
      return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{}); // scalar
    }
    Operation *def = v.getDefiningOp();
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return emitScalarUnary(b, loc, mu.getFn(),
                             emitAt(b, loc, mu.getInput(), k));
    if (auto mb = dyn_cast<MapBinaryOp>(def))
      return emitScalarBinary(b, loc, mb.getFn(), emitAt(b, loc, mb.getLhs(), k),
                              emitAt(b, loc, mb.getRhs(), k));
    return nullptr; // a reduce result not yet hoisted: impossible in topo order
  }

  /// Is an original value uniform (per-row / rank-0)? Row/Col args are not.
  bool isUniform(Value v) {
    if (uniform.count(v))
      return true;
    if (auto barg = dyn_cast<BlockArgument>(v))
      return argRoles[barg.getArgNumber()] == Role::Scalar;
    return false;
  }

  /// Hoist every map op whose operands are all uniform into `uniform` (once),
  /// at the current (top-of-launch) insertion point. Fixpoint. Uniform subtrees
  /// never index a column, so the `k` passed to emitAt is unused (0).
  void hoistUniforms(OpBuilder &b, Location loc) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : fn.getBody().front().without_terminator()) {
        Value res = op.getNumResults() ? op.getResult(0) : Value();
        if (!res || uniform.count(res))
          continue;
        if (auto mu = dyn_cast<MapUnaryOp>(&op)) {
          if (isUniform(mu.getInput())) {
            uniform[res] = emitScalarUnary(b, loc, mu.getFn(),
                                           emitAt(b, loc, mu.getInput(), 0));
            changed = true;
          }
        } else if (auto mb = dyn_cast<MapBinaryOp>(&op)) {
          if (isUniform(mb.getLhs()) && isUniform(mb.getRhs())) {
            uniform[res] = emitScalarBinary(b, loc, mb.getFn(),
                                            emitAt(b, loc, mb.getLhs(), 0),
                                            emitAt(b, loc, mb.getRhs(), 0));
            changed = true;
          }
        }
      }
    }
  }

  Type memrefOf(Role role) {
    if (role == Role::Row)
      return MemRefType::get({M, N}, f32);
    if (role == Role::Col)
      return MemRefType::get({N}, f32);
    return MemRefType::get({}, f32);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: original args (as memrefs) ++ one output memref per result.
    SmallVector<Type> inTys;
    for (Role r : argRoles)
      inTys.push_back(memrefOf(r));
    inTys.push_back(MemRefType::get({M, N}, f32)); // Y (destination-passing)
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src"); // free up the symbol for the new func

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);
    Value yMem = entry->getArgument(argRoles.size());

    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    auto launch = b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Column indices col[k] = tx + k*T (k*T is a compile-time constant), and the
    // read-once register tile: each Row input's K values loaded ONCE into SSA.
    cols.clear();
    for (int64_t k = 0; k < K; ++k) {
      Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
      cols.push_back(b.create<arith::AddIOp>(loc, tx, off));
    }
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        SmallVector<Value> vals;
        for (int64_t k = 0; k < K; ++k)
          vals.push_back(b.create<memref::LoadOp>(loc, argMemref[arg],
                                                  ValueRange{bx, cols[k]}));
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::Scalar) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
      }
    }
    hoistUniforms(b, loc);

    Value nF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr((double)N));

    // One cooperative pass per reduce (program order = topo order): the per-
    // thread partial is an unrolled reduction over the K register-tile values,
    // then a standalone gpu.all_reduce across the block.
    for (size_t idx = 0; idx < reduces.size(); ++idx) {
      ReduceOp red = reduces[idx];
      ReduceInfo info = *reduceInfoFor(red.getFn());
      Value acc =
          b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
      for (int64_t k = 0; k < K; ++k)
        acc = combineScalar(b, loc, info.op, acc,
                            emitAt(b, loc, red.getInput(), k));
      auto allRed = b.create<gpu::AllReduceOp>(
          loc, acc, gpu::AllReduceOperationAttr::get(ctx, info.op),
          /*uniform=*/UnitAttr::get(ctx));
      Value scalar = allRed.getResult();
      if (info.isMean)
        scalar = b.create<arith::DivFOp>(loc, scalar, nF);
      uniform[red.getResult()] = scalar;
      hoistUniforms(b, loc); // +eps, sqrt, recip, ... become available now
    }

    // Epilogue: write the output row from the register tile (read-once).
    for (int64_t k = 0; k < K; ++k) {
      Value out = emitAt(b, loc, outputVal, k);
      b.create<memref::StoreOp>(loc, out, yMem, ValueRange{bx, cols[k]});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentRowReductionToGPUPass
    : ::intent::impl::ConvertIntentRowReductionToGPUBase<
          ConvertIntentRowReductionToGPUPass> {
  using ConvertIntentRowReductionToGPUBase::
      ConvertIntentRowReductionToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      RowReductionLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      RowReductionLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("row-reduction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
