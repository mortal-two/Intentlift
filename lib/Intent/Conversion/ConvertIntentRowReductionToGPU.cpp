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

#include "MapEmit.h"
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

/// Per-result role: a full [M,N] row output (written from the register tile),
/// or a per-row [M,1] scalar output (a stat like Mean/RSTD, written once/block).
enum class ResultRole { RowOut, PerRowOut };

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
  SmallVector<Value> outputVals;     // the returned values (>=1, topo order)
  SmallVector<ResultRole> outRoles;  // RowOut [M,N] / PerRowOut [M,1] per result

  RowReductionLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    unsigned nRes = fty.getNumResults();
    if (nRes < 1)
      return failure();
    // Results are static f32, each per-row in M: a RowOut is rank-2 [M,N] (the
    // full feature row); a per-row scalar output (Mean/RSTD, or flash's delta)
    // is rank-2 [M,1] OR rank-1 [M]. Anchor M from every result's leading dim;
    // anchor N from a RowOut if any. An all-PerRowOut func (e.g. flash
    // _bwd_preprocess_do_o_dot: delta = sum(o*do, axis=1)) has no RowOut, so N
    // is derived from a Row-shaped INPUT below.
    int64_t anchorM = -1, anchorN = -1;
    for (Type rt : fty.getResults()) {
      auto t = dyn_cast<RankedTensorType>(rt);
      if (!t || !t.hasStaticShape())
        return failure();
      if (!isa<Float32Type>(t.getElementType()))
        return failure();
      f32 = t.getElementType();
      int64_t rank = t.getRank();
      if (rank != 1 && rank != 2)
        return failure();
      int64_t m = t.getDimSize(0);
      if (anchorM < 0)
        anchorM = m;
      else if (m != anchorM)
        return failure();
      if (rank == 2 && t.getDimSize(1) != 1) { // a RowOut
        if (anchorN < 0)
          anchorN = t.getDimSize(1);
        else if (t.getDimSize(1) != anchorN)
          return failure();
      }
    }
    M = anchorM;
    // No RowOut: derive N from the first Row-shaped input ([M, X] static f32).
    if (anchorN < 0) {
      for (Type at : fty.getInputs()) {
        auto t = dyn_cast<RankedTensorType>(at);
        if (t && t.hasStaticShape() && isa<Float32Type>(t.getElementType()) &&
            t.getRank() == 2 && t.getDimSize(0) == M) {
          anchorN = t.getDimSize(1);
          break;
        }
      }
    }
    if (anchorN < 0)
      return failure(); // nothing to tile the feature dim over
    N = anchorN;
    for (Type rt : fty.getResults()) {
      auto t = cast<RankedTensorType>(rt);
      if (t.getRank() == 2 && t.getDimSize(1) == N)
        outRoles.push_back(ResultRole::RowOut);
      else // rank-2 [M,1] or rank-1 [M]
        outRoles.push_back(ResultRole::PerRowOut);
    }

    // Single basic block, terminated by a return of the output values.
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != nRes)
      return failure();
    outputVals.assign(ret.getOperands().begin(), ret.getOperands().end());

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
        if (inTy.getRank() != 2)
          return failure();
        if (r.getAxes().size() != 1)
          return failure();
        int64_t ax = cast<IntegerAttr>(r.getAxes()[0]).getInt();
        // Per-row last-axis reduce: result is [M,1] (keepdims) or [M] (collapsed,
        // flash's delta layout). Both are treated as a per-row uniform scalar.
        bool okRes =
            (rTy.getRank() == 2 && rTy.getDimSize(0) == M &&
             rTy.getDimSize(1) == 1) ||
            (rTy.getRank() == 1 && rTy.getDimSize(0) == M);
        if (ax != 1 || !okRes)
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
    for (ResultRole rr : outRoles) // destination-passing: one memref per result
      inTys.push_back(rr == ResultRole::RowOut ? MemRefType::get({M, N}, f32)
                                               : MemRefType::get({M}, f32));
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
    SmallVector<Value> outMems;
    for (unsigned j = 0; j < outRoles.size(); ++j)
      outMems.push_back(entry->getArgument(argRoles.size() + j));

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

    // Epilogue: each RowOut written from the register tile (read-once); each
    // PerRowOut (Mean/RSTD) is a per-row uniform written ONCE by thread 0
    // (single-writer guard — M*N stores for Y vs M scalar stores for a stat).
    bool anyPerRow = false;
    for (ResultRole r : outRoles)
      anyPerRow |= (r == ResultRole::PerRowOut);
    Value isLead;
    if (anyPerRow) {
      Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
      isLead = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, c0);
    }
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      if (outRoles[j] == ResultRole::RowOut) {
        for (int64_t k = 0; k < K; ++k) {
          Value out = emitAt(b, loc, outputVals[j], k);
          b.create<memref::StoreOp>(loc, out, outMems[j],
                                    ValueRange{bx, cols[k]});
        }
      } else {
        Value scalar = emitAt(b, loc, outputVals[j], 0); // per-row uniform
        auto ifOp = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        b.create<memref::StoreOp>(loc, scalar, outMems[j], ValueRange{bx});
      }
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
