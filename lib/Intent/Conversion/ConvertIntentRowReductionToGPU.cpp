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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
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

/// Constant-evaluate an affine expr over integer dim values (no symbols). Used
/// to recover channels-per-group from a ColSeg's affine index map without
/// parsing coefficients. Floordiv/mod assume non-negative operands (col,group
/// are non-negative), matching the runtime arith form below.
static int64_t evalAffineExprConst(AffineExpr e, ArrayRef<int64_t> dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return c.getValue();
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  int64_t l = evalAffineExprConst(bin.getLHS(), dims);
  int64_t r = evalAffineExprConst(bin.getRHS(), dims);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return l + r;
  case AffineExprKind::Mul:
    return l * r;
  case AffineExprKind::FloorDiv:
    return l / r;
  case AffineExprKind::Mod:
    return l % r;
  case AffineExprKind::CeilDiv:
    return (l + r - 1) / r;
  default:
    return 0;
  }
}

/// Emit the index arith for an affine expr over `dims` (index-typed SSA values),
/// producing an index Value. floordiv->divui / mod->remui are valid because the
/// dim operands (group, col) are always non-negative; a future signed map would
/// need divsi/remsi. This is the structured affine gather of a ColSeg arg.
static Value emitAffineExpr(OpBuilder &b, Location loc, AffineExpr e,
                            ValueRange dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return b.create<arith::ConstantIndexOp>(loc, c.getValue());
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Value l = emitAffineExpr(b, loc, bin.getLHS(), dims);
  Value r = emitAffineExpr(b, loc, bin.getRHS(), dims);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return b.create<arith::AddIOp>(loc, l, r);
  case AffineExprKind::Mul:
    return b.create<arith::MulIOp>(loc, l, r);
  case AffineExprKind::FloorDiv:
    return b.create<arith::DivUIOp>(loc, l, r);
  case AffineExprKind::Mod:
    return b.create<arith::RemUIOp>(loc, l, r);
  case AffineExprKind::CeilDiv:
    return b.create<arith::CeilDivUIOp>(loc, l, r);
  default:
    return nullptr;
  }
}

/// Per-result role: a full [M,N] row output (written from the register tile);
/// a per-row [M,1]/[M] scalar output (a stat like Mean/RSTD, written once/block);
/// or a rank-0 [] cross-row scalar (a loss total) accumulated across blocks via a
/// thread-0 `memref.atomic_rmw addf` — the "AtomicOut" fold that fuses the host
/// `.sum()` the library leaves separate.
enum class ResultRole { RowOut, PerRowOut, AtomicOut };

/// One func's row-reduction lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct RowReductionLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  unsigned tileColsMax;

  // populated by match()
  int64_t M = 0, N = 0, T = 0, K = 0;
  // Tiling: the per-thread feature strip of K columns is split into `numTiles`
  // tiles of `Kpt` columns each (K == Kpt * numTiles). numTiles==1 (K <=
  // tileColsMax) is the M1 register-tile path, emitted byte-identically.
  int64_t Kpt = 0, numTiles = 1;
  Type f32;
  Type rowIntTy;                     // element type of a RowInt [M] arg (Y), if any
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<ReduceOp> reduces;     // per-row (axis-1) reduces, program order
  SmallVector<ReduceOp> crossReduces; // cross-row (axis-0 sum) reduces -> AtomicOut
  SmallVector<ScanOp> scans;         // per-row (axis-1) inclusive cumsum scans
  SmallVector<Value> outputVals;     // the returned values (>=1, topo order)
  SmallVector<ResultRole> outRoles;  // RowOut / PerRowOut / AtomicOut per result
  // ColSeg (group_norm W/B) state, indexed by func-arg position. colSegLen[i] is
  // the C extent of the [C] memref; colSegMap[i] is the affine (group,col)->chan
  // index map. numGroups (G) is recovered from C/CPG so group = bx % G.
  SmallVector<int64_t> colSegLen;
  SmallVector<AffineMap> colSegMap;
  int64_t numGroups = 0;
  Value groupVal;                    // bx % G (per-block uniform), set in rewrite

  RowReductionLowering(func::FuncOp fn, unsigned bt, unsigned tc)
      : fn(fn), blockThreads(bt), tileColsMax(tc) {}

  /// Is `v` a compile-time-uniform (rank-0 scalar) value? A Scalar arg, or a map
  /// subtree of only Scalar args. A reduce result is per-row uniform but NOT a
  /// scalar constant, so it returns false (it is the cross-row reduce's input,
  /// never its scaling factor).
  bool isStaticUniform(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return argRoles[ba.getArgNumber()] == Role::Scalar;
    Operation *d = v.getDefiningOp();
    if (!d)
      return false;
    if (auto mu = dyn_cast<MapUnaryOp>(d))
      return isStaticUniform(mu.getInput());
    if (auto mb = dyn_cast<MapBinaryOp>(d))
      return isStaticUniform(mb.getLhs()) && isStaticUniform(mb.getRhs());
    return false;
  }

  /// Walk from an AtomicOut value back to its cross-row reduce, allowing only
  /// uniform-scalar mul/div scaling on the path (linear ops that distribute over
  /// the cross-row sum). Returns the axis-0 `sum` ReduceOp, or null if the chain
  /// is anything else (non-linear add/sub, or a non-uniform second operand).
  ReduceOp atomicSource(Value v) {
    while (true) {
      if (auto r = v.getDefiningOp<ReduceOp>())
        return r; // the axis-0 sum cross-row reduce (already shape-checked)
      auto mb = v.getDefiningOp<MapBinaryOp>();
      if (!mb)
        return nullptr;
      StringRef fn = mb.getFn();
      if (fn == "div") {
        if (!isStaticUniform(mb.getRhs())) // x / uniform
          return nullptr;
        v = mb.getLhs();
      } else if (fn == "mul") {
        if (isStaticUniform(mb.getRhs()))
          v = mb.getLhs();
        else if (isStaticUniform(mb.getLhs()))
          v = mb.getRhs();
        else
          return nullptr;
      } else {
        return nullptr;
      }
    }
  }

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
      if (rank > 2)
        return failure();
      if (rank == 0)
        continue; // a rank-0 AtomicOut (cross-row scalar): does not anchor M/N
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
    // No RowOut / no per-row result (an all-AtomicOut func, e.g. kl_div
    // batchmean returns only the scalar): derive M (if still unknown) and N
    // from the first Row-shaped input ([M, X] static f32).
    if (anchorN < 0) {
      for (Type at : fty.getInputs()) {
        auto t = dyn_cast<RankedTensorType>(at);
        if (t && t.hasStaticShape() && isa<Float32Type>(t.getElementType()) &&
            t.getRank() == 2) {
          if (M < 0)
            M = t.getDimSize(0);
          if (t.getDimSize(0) == M) {
            anchorN = t.getDimSize(1);
            break;
          }
        }
      }
    }
    if (M < 0 || anchorN < 0)
      return failure(); // nothing to tile the feature dim over
    N = anchorN;
    for (Type rt : fty.getResults()) {
      auto t = cast<RankedTensorType>(rt);
      if (t.getRank() == 0)
        outRoles.push_back(ResultRole::AtomicOut);
      else if (t.getRank() == 2 && t.getDimSize(1) == N)
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

    // Args: classify each by shape. f32 tensors are Row[M,N]/Col[N]/Scalar[];
    // a rank-1 [M] INTEGER tensor is a RowInt (cross_entropy's target class Y),
    // loaded once per block as Y[bx].
    colSegLen.assign(fty.getNumInputs(), 0);
    colSegMap.assign(fty.getNumInputs(), AffineMap());
    for (unsigned ai = 0; ai < fty.getNumInputs(); ++ai) {
      auto t = dyn_cast<RankedTensorType>(fty.getInput(ai));
      if (!t || !t.hasStaticShape())
        return failure();
      ArrayRef<int64_t> s = t.getShape();
      // ColSeg (group_norm W/B): a 1-D f32 arg carrying an affine column-index
      // map. The arg-attr drives the gather (general, no per-kernel branch); the
      // [C] extent need not equal N. CPG = map(1,col)-map(0,col) (the group
      // coefficient); G = C / CPG so group = bx % G.
      if (auto mapAttr = fn.getArgAttrOfType<AffineMapAttr>(
              ai, "intent.col_index_map")) {
        AffineMap m = mapAttr.getAffineMap();
        if (s.size() != 1 || m.getNumDims() != 2 || m.getNumResults() != 1 ||
            t.getElementType() != f32)
          return failure();
        int64_t cpg = evalAffineExprConst(m.getResult(0), {1, 0}) -
                      evalAffineExprConst(m.getResult(0), {0, 0});
        if (cpg <= 0 || s[0] % cpg != 0)
          return failure();
        int64_t g = s[0] / cpg;
        if (numGroups == 0)
          numGroups = g;
        else if (numGroups != g)
          return failure();
        colSegLen[ai] = s[0];
        colSegMap[ai] = m;
        argRoles.push_back(Role::ColSeg);
        continue;
      }
      if (s.size() == 1 && s[0] == M && t.getElementType().isInteger()) {
        rowIntTy = t.getElementType();
        argRoles.push_back(Role::RowInt);
        continue;
      }
      if (t.getElementType() != f32)
        return failure();
      if (s.size() == 2 && s[0] == M && s[1] == N)
        argRoles.push_back(Role::Row);
      else if (s.size() == 1 && s[0] == N)
        argRoles.push_back(Role::Col);
      else if (s.size() == 0)
        argRoles.push_back(Role::Scalar);
      else
        return failure(); // [M,1] etc. as an *input* not supported in M1
    }

    // Body ops: only map_unary/map_binary/reduce. A reduce is either a per-row
    // last-axis reduce (axis 1, result [M,1]/[M] -> a per-row uniform scalar) or
    // a cross-row fold (axis 0 over a per-row [M] vector -> a rank-0 scalar that
    // becomes an AtomicOut: one atomicAdd/block, host pre-zeroes).
    for (Operation &op : body.without_terminator()) {
      if (auto r = dyn_cast<ReduceOp>(&op)) {
        auto inTy = cast<RankedTensorType>(r.getInput().getType());
        auto rTy = cast<RankedTensorType>(r.getResult().getType());
        if (r.getAxes().size() != 1)
          return failure();
        int64_t ax = cast<IntegerAttr>(r.getAxes()[0]).getInt();
        if (ax == 0) {
          // Cross-row fold: [M] per-row vector -> rank-0 scalar, sum only (the
          // atomicAdd identity is +). The input is a per-row uniform scalar.
          if (inTy.getRank() != 1 || inTy.getDimSize(0) != M ||
              rTy.getRank() != 0 || r.getFn() != "sum")
            return failure();
          crossReduces.push_back(r);
          continue;
        }
        if (inTy.getRank() != 2)
          return failure();
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
      } else if (auto sc = dyn_cast<ScanOp>(&op)) {
        // A per-row inclusive cumsum scan (sparsemax's cumsum of the sorted row).
        // The scan keeps a row-varying per-column value; it forces the contiguous
        // column-to-thread tiling + dynamic-shared scan path (rewriteScan). Only
        // axis-1 (the feature axis), [M,N]->[M,N], fn=sum is supported.
        auto inTy = cast<RankedTensorType>(sc.getInput().getType());
        if (sc.getAxis() != 1 || inTy.getRank() != 2 ||
            inTy.getDimSize(0) != M || inTy.getDimSize(1) != N ||
            sc.getFn() != "sum")
          return failure();
        scans.push_back(sc);
      } else if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, IotaOp>(&op)) {
        // SelectOp + comparison map_binary (-> i1) + IotaOp are the M6 residual:
        // cross_entropy's target gather/one-hot (`select(iota == Y, X, 0)`). The
        // ops are general (owned by M3/M6); this just teaches emitAt to evaluate
        // them inside the row-reduction register tile.
        return failure();
      }
    }
    if (reduces.empty() && scans.empty())
      return failure(); // pure elementwise -> not our pass

    // Validate every AtomicOut: its value must reach a cross-row reduce through
    // only uniform mul/div scaling (linear ops distribute over the cross-row
    // sum, so the scale folds into each block's atomicAdd contribution), and
    // that cross-row reduce result must be single-use (the uniform-injection in
    // rewrite() would otherwise lie to any other consumer).
    for (unsigned j = 0; j < outRoles.size(); ++j) {
      if (outRoles[j] != ResultRole::AtomicOut)
        continue;
      ReduceOp cr = atomicSource(outputVals[j]);
      if (!cr || !cr.getResult().hasOneUse())
        return failure();
    }

    // Geometry: T threads over the feature dim, K columns per thread.
    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || N % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented M1 follow-up
    K = N / T;
    // Register budget: if the per-thread strip K exceeds tileColsMax, split it
    // into numTiles tiles of Kpt <= tileColsMax columns each (Kpt must divide K,
    // smallest such numTiles). numTiles==1 is the M1 register-tile path.
    Kpt = K;
    numTiles = 1;
    if (tileColsMax > 0 && K > (int64_t)tileColsMax) {
      for (int64_t nt = 2; nt <= K; ++nt) {
        if (K % nt == 0 && K / nt <= (int64_t)tileColsMax) {
          numTiles = nt;
          Kpt = K / nt;
          break;
        }
      }
    }
    // Multi-tile (tiled + smem-cached) path scope: exactly one per-row reduce,
    // no cross-row/atomic fold, outputs are RowOut/PerRowOut only. The general
    // tiled multi-reduce case is out of M5 scope (its kernels are small-N /
    // single-tile). The single-tile path is unrestricted (M1/M2/M5a).
    if (numTiles > 1) {
      if (reduces.size() != 1 || !crossReduces.empty())
        return failure();
      // ColSeg's per-(group,col) gather is only wired in the single-tile epilogue
      // (group_norm's hidden-per-group is small -> single tile). The tiled +
      // smem-cached path does not emit the affine channel index.
      if (numGroups != 0)
        return failure();
      for (ResultRole rr : outRoles)
        if (rr == ResultRole::AtomicOut)
          return failure();
      // The tiled block reduction is a power-of-two shared-memory tree (so it
      // never emits a gpu.all_reduce static buffer that would overlap the
      // dynamic-shared cache). Every real geometry (num_warps*32) is pow2.
      if ((T & (T - 1)) != 0)
        return failure();
    }
    // Scan path (sparsemax fwd): the contiguous register tile + dynamic-shared
    // block scan. Requires a single register tile (K <= tile-cols so each
    // thread's contiguous segment is in registers), a power-of-two T (the
    // cross-thread exclusive scan + the manual shared reduction tree), and no
    // cross-row/atomic fold (the row-local cumsum is the whole game).
    if (!scans.empty()) {
      if (numTiles != 1 || !crossReduces.empty() || numGroups != 0)
        return failure();
      if ((T & (T - 1)) != 0)
        return failure();
    }
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
  DenseMap<Value, SmallVector<Value>> scanVals;  // scan result -> K cumsum SSA values
  SmallVector<Value> cols;                       // K column indices (tx + k*T)
  DenseMap<Value, Value> frontierBuf;            // cached row value -> smem memref

  /// Evaluate an original value for the thread's k-th column (fully unrolled).
  /// Row inputs resolve to the once-loaded register value; uniforms to their
  /// hoisted per-row scalar; Col inputs (W) are loaded at col[k].
  Value emitAt(OpBuilder &b, Location loc, Value v, unsigned k) {
    if (Value u = uniform.lookup(v))
      return u;
    // A scan (cumsum) result resolves to the thread's precomputed inclusive
    // prefix value for its k-th (contiguous) column.
    {
      auto it = scanVals.find(v);
      if (it != scanVals.end())
        return it->second[k];
    }
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Row)
        return rowVals[v][k];
      if (role == Role::Col)
        return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{cols[k]});
      if (role == Role::ColSeg) {
        // channel = col_index_map(group, col[k]); load W/B[channel].
        AffineMap m = colSegMap[barg.getArgNumber()];
        Value ch = emitAffineExpr(b, loc, m.getResult(0),
                                  ValueRange{groupVal, cols[k]});
        return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{ch});
      }
      return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{}); // scalar
    }
    Operation *def = v.getDefiningOp();
    if (auto io = dyn_cast<IotaOp>(def)) {
      // iota at column col[k] is the column index itself, cast to the op's
      // integer width (no materialized vector). Feeds `iota == Y` (a cmpi).
      Type elt = cast<RankedTensorType>(io.getResult().getType()).getElementType();
      return b.create<arith::IndexCastOp>(loc, elt, cols[k]);
    }
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return emitScalarUnary(b, loc, mu.getFn(),
                             emitAt(b, loc, mu.getInput(), k));
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      Value lhs = emitAt(b, loc, mb.getLhs(), k);
      Value rhs = emitAt(b, loc, mb.getRhs(), k);
      if (isCmpFn(mb.getFn())) // i1 condition for a select
        return isa<IntegerType>(lhs.getType())
                   ? emitScalarCmpI(b, loc, mb.getFn(), lhs, rhs)
                   : emitScalarCmp(b, loc, mb.getFn(), lhs, rhs);
      // An integer operand in float arithmetic (sparsemax's `iota + 1` for the
      // k_arange divisor) is coerced to f32 (signed int -> float). The only
      // integer-typed emitAt source is iota; its only non-comparison use is this.
      if (isa<IntegerType>(lhs.getType()))
        lhs = b.create<arith::SIToFPOp>(loc, f32, lhs);
      if (isa<IntegerType>(rhs.getType()))
        rhs = b.create<arith::SIToFPOp>(loc, f32, rhs);
      return emitScalarBinary(b, loc, mb.getFn(), lhs, rhs);
    }
    if (auto sel = dyn_cast<SelectOp>(def))
      return b.create<arith::SelectOp>(loc, emitAt(b, loc, sel.getCond(), k),
                                       emitAt(b, loc, sel.getX(), k),
                                       emitAt(b, loc, sel.getY(), k));
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

  Type memrefOf(unsigned argIdx) {
    Role role = argRoles[argIdx];
    if (role == Role::Row)
      return MemRefType::get({M, N}, f32);
    if (role == Role::Col)
      return MemRefType::get({N}, f32);
    if (role == Role::ColSeg)
      return MemRefType::get({colSegLen[argIdx]}, f32);
    if (role == Role::RowInt)
      return MemRefType::get({M}, rowIntTy);
    return MemRefType::get({}, f32);
  }

  // ---- tiled + smem-cached path (numTiles > 1) --------------------------- //
  // A value is "row-varying" if it transitively depends on a Row[M,N] arg. A
  // Col[N] or Scalar[] arg is NOT row-varying (Cols are re-loaded cheaply in the
  // epilogue; Scalars are uniform). A reduce result is per-row uniform.
  bool isRowVarying(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return argRoles[ba.getArgNumber()] == Role::Row;
    Operation *d = v.getDefiningOp();
    if (!d)
      return false;
    if (auto mu = dyn_cast<MapUnaryOp>(d))
      return isRowVarying(mu.getInput());
    if (auto mb = dyn_cast<MapBinaryOp>(d))
      return isRowVarying(mb.getLhs()) || isRowVarying(mb.getRhs());
    return false; // a reduce result is per-row uniform, not row-varying
  }

  // The "row frontier" of an output is the maximal row-varying subexpression(s)
  // that must be cached so the normalize epilogue recomputes only uniform/Col
  // ops on top (never re-reading a Row arg). Walk down from `v`: stop (cache `v`)
  // at a Row arg, or at a map_binary whose BOTH operands are row-varying (an
  // intermediate like A = X + residual that is reused — cache the combined value,
  // one buffer, not its two operands); descend through unary ops and through the
  // single row-varying operand of a uniform/Col combine. This yields {X} for
  // rms_norm and {A} for rms_norm_add — exactly one buffer each.
  void markFrontier(Value v, llvm::SetVector<Value> &fset) {
    if (!isRowVarying(v))
      return;
    if (isa<BlockArgument>(v)) {
      fset.insert(v);
      return;
    }
    Operation *d = v.getDefiningOp();
    if (auto mu = dyn_cast<MapUnaryOp>(d)) {
      markFrontier(mu.getInput(), fset);
      return;
    }
    if (auto mb = dyn_cast<MapBinaryOp>(d)) {
      bool l = isRowVarying(mb.getLhs()), r = isRowVarying(mb.getRhs());
      if (l && r) {
        fset.insert(v);
        return;
      }
      markFrontier(l ? mb.getLhs() : mb.getRhs(), fset);
      return;
    }
    fset.insert(v); // defensive: an unexpected row-varying producer
  }

  // Evaluate an original value at a dynamic column `col` (loading Row/Col args
  // fresh, memoized per-column in `cur`). When `fromSmem`, a frontier value is
  // read from its workgroup cache instead of recomputed from Row args (the
  // read-once epilogue: the row was cached during the reduction pass).
  Value emitAtCol(OpBuilder &b, Location loc, Value v, Value col, Value bx,
                  DenseMap<Value, Value> &cur, bool fromSmem) {
    if (Value c = cur.lookup(v))
      return c;
    if (Value u = uniform.lookup(v))
      return u;
    if (fromSmem) {
      if (Value buf = frontierBuf.lookup(v)) {
        Value ld = b.create<memref::LoadOp>(loc, buf, ValueRange{col});
        cur[v] = ld;
        return ld;
      }
    }
    Value res;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[ba.getArgNumber()];
      if (role == Role::Row)
        res = b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{bx, col});
      else if (role == Role::Col)
        res = b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{col});
      else
        res = uniform.lookup(v); // scalar already loaded once at launch top
    } else {
      Operation *d = v.getDefiningOp();
      if (auto mu = dyn_cast<MapUnaryOp>(d))
        res = emitScalarUnary(b, loc, mu.getFn(),
                              emitAtCol(b, loc, mu.getInput(), col, bx, cur,
                                        fromSmem));
      else if (auto mb = dyn_cast<MapBinaryOp>(d))
        res = emitScalarBinary(
            b, loc, mb.getFn(),
            emitAtCol(b, loc, mb.getLhs(), col, bx, cur, fromSmem),
            emitAtCol(b, loc, mb.getRhs(), col, bx, cur, fromSmem));
      else
        res = nullptr;
    }
    cur[v] = res;
    return res;
  }

  LogicalResult rewriteTiled() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // Frontier set: the row-varying values to cache in workgroup shared memory.
    llvm::SetVector<Value> frontierSet;
    for (Value o : outputVals)
      markFrontier(o, frontierSet);
    SmallVector<Value> frontiers(frontierSet.begin(), frontierSet.end());

    // New signature: original args (as memrefs) ++ one output memref per result.
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inTys.push_back(memrefOf(i));
    for (ResultRole rr : outRoles) {
      if (rr == ResultRole::RowOut)
        inTys.push_back(MemRefType::get({M, N}, f32));
      else // PerRowOut (AtomicOut rejected for the tiled path in match())
        inTys.push_back(MemRefType::get({M}, f32));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

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
    // Dynamic shared memory: one [N] f32 row cache per frontier, plus a [T] f32
    // scratch for the block reduction tree. We do NOT use gpu.all_reduce here:
    // its lowering emits a *separate static* `.shared` buffer that ptxas packs
    // INSIDE our (demoted) dynamic-shared cache region, clobbering it. Keeping
    // the only `.shared` symbol the single dynamic buffer (the G1-proven config)
    // sidesteps that overlap entirely.
    int64_t cacheBytes = (int64_t)N * (int64_t)frontiers.size() * 4;
    int64_t shBytes = cacheBytes + (int64_t)T * 4;
    Value shBytesV =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shBytes));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesV);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Carve the dynamic shared region into one memref<Nxf32, workgroup> per
    // frontier at byte offset f*N*4, then a memref<Txf32, workgroup> reduction
    // scratch right after the caches.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto bufTy = MemRefType::get({N}, f32, nullptr, wgSpace);
    for (size_t f = 0; f < frontiers.size(); ++f) {
      Value off = b.create<arith::ConstantIndexOp>(loc, (int64_t)f * N * 4);
      Value buf = b.create<memref::ViewOp>(loc, bufTy, raw, off, ValueRange{});
      frontierBuf[frontiers[f]] = buf;
    }
    auto scratchTy = MemRefType::get({T}, f32, nullptr, wgSpace);
    Value scratchOff = b.create<arith::ConstantIndexOp>(loc, cacheBytes);
    Value scratch =
        b.create<memref::ViewOp>(loc, scratchTy, raw, scratchOff, ValueRange{});

    // Scalar args -> uniform (loaded once). NO upfront register tile (the strip
    // is too large for registers — that is why we tile + cache).
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Scalar)
        uniform[arg] = b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
    }
    hoistUniforms(b, loc);

    Value nF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr((double)N));

    // Per-thread base columns of tile 0: baseCol[k] = tx + k*T. The k_tile offset
    // (k_tile * Kpt * T) is added INSIDE the loop (it depends on the trip index).
    SmallVector<Value> baseCol;
    for (int64_t k = 0; k < Kpt; ++k) {
      Value o = b.create<arith::ConstantIndexOp>(loc, k * T);
      baseCol.push_back(b.create<arith::AddIOp>(loc, tx, o));
    }
    Value c0i = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cNT = b.create<arith::ConstantIndexOp>(loc, numTiles);
    Value cKptT = b.create<arith::ConstantIndexOp>(loc, Kpt * T);

    // The single per-row reduce, tiled: one scf.for over feature tiles building
    // a per-thread partial (iter_arg), the frontier row cached to smem on the
    // way, then ONE gpu.all_reduce AFTER the loop.
    ReduceOp red = reduces[0];
    ReduceInfo info = *reduceInfoFor(red.getFn());
    Value ident =
        b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
    auto forOp = b.create<scf::ForOp>(
        loc, c0i, cNT, c1, ValueRange{ident},
        [&](OpBuilder &nb, Location nl, Value kt, ValueRange iter) {
          Value ktOff = nb.create<arith::MulIOp>(nl, kt, cKptT);
          Value acc = iter[0];
          for (int64_t k = 0; k < Kpt; ++k) {
            Value col = nb.create<arith::AddIOp>(nl, baseCol[k], ktOff);
            DenseMap<Value, Value> cur;
            // cache each frontier (computed from the fresh Row loads) to smem
            for (Value F : frontiers) {
              Value fv = emitAtCol(nb, nl, F, col, bx, cur, /*fromSmem=*/false);
              nb.create<memref::StoreOp>(nl, fv, frontierBuf[F],
                                         ValueRange{col});
            }
            // reduce input reuses the same fresh loads (read-once of X per col)
            Value rin =
                emitAtCol(nb, nl, red.getInput(), col, bx, cur, false);
            acc = combineScalar(nb, nl, info.op, acc, rin);
          }
          nb.create<scf::YieldOp>(nl, acc);
        });
    // Block reduction: a power-of-two shared-memory tree over `scratch[tx]`
    // (fully unrolled; T static). scratch[0] holds the block result, broadcast
    // by every thread re-loading it.
    b.create<memref::StoreOp>(loc, forOp.getResult(0), scratch, ValueRange{tx});
    b.create<gpu::BarrierOp>(loc);
    for (int64_t stride = T / 2; stride >= 1; stride >>= 1) {
      Value cStride = b.create<arith::ConstantIndexOp>(loc, stride);
      Value active =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx, cStride);
      auto ifOp = b.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        Value a = b.create<memref::LoadOp>(loc, scratch, ValueRange{tx});
        Value other = b.create<arith::AddIOp>(loc, tx, cStride);
        Value bb = b.create<memref::LoadOp>(loc, scratch, ValueRange{other});
        Value c = combineScalar(b, loc, info.op, a, bb);
        b.create<memref::StoreOp>(loc, c, scratch, ValueRange{tx});
      }
      b.create<gpu::BarrierOp>(loc);
    }
    Value c0r = b.create<arith::ConstantIndexOp>(loc, 0);
    Value scalar = b.create<memref::LoadOp>(loc, scratch, ValueRange{c0r});
    if (info.isMean)
      scalar = b.create<arith::DivFOp>(loc, scalar, nF);
    uniform[red.getResult()] = scalar;
    hoistUniforms(b, loc); // rstd = rsqrt(mean+eps), etc. become available

    // Barrier between the cache pass and the normalize epilogue. Each thread
    // reads back only its OWN cached columns (col = tx + ...), so this is a
    // safety fence (the all_reduce already synced the block); it also documents
    // the cache->use boundary and keeps in-place X writes after all X reads.
    b.create<gpu::BarrierOp>(loc);

    // PerRowOut (e.g. a stat) is a per-row uniform written once by thread 0.
    bool anyLead = false;
    for (ResultRole r : outRoles)
      anyLead |= (r == ResultRole::PerRowOut);
    if (anyLead) {
      Value isLead =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tx, c0i);
      auto ifOp = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(ifOp.thenBlock());
      DenseMap<Value, Value> cur;
      for (unsigned j = 0; j < outputVals.size(); ++j) {
        if (outRoles[j] != ResultRole::PerRowOut)
          continue;
        Value s = emitAtCol(b, loc, outputVals[j], c0i, bx, cur, /*fromSmem=*/true);
        b.create<memref::StoreOp>(loc, s, outMems[j], ValueRange{bx});
      }
    }

    // RowOut epilogue: one scf.for over the same feature tiles, reading each
    // frontier back from its smem cache (NOT a 2nd HBM read of X) and writing
    // every RowOut from it. For rms_norm_add the in-place A output is a frontier
    // read from smem and stored back into the X memref (read-after-write across
    // the barrier makes the alias safe).
    bool anyRow = false;
    for (ResultRole r : outRoles)
      anyRow |= (r == ResultRole::RowOut);
    if (anyRow) {
      b.create<scf::ForOp>(
          loc, c0i, cNT, c1, ValueRange{},
          [&](OpBuilder &nb, Location nl, Value kt, ValueRange) {
            Value ktOff = nb.create<arith::MulIOp>(nl, kt, cKptT);
            for (int64_t k = 0; k < Kpt; ++k) {
              Value col = nb.create<arith::AddIOp>(nl, baseCol[k], ktOff);
              DenseMap<Value, Value> cur;
              for (unsigned j = 0; j < outputVals.size(); ++j) {
                if (outRoles[j] != ResultRole::RowOut)
                  continue;
                Value o = emitAtCol(nb, nl, outputVals[j], col, bx, cur,
                                    /*fromSmem=*/true);
                nb.create<memref::StoreOp>(nl, o, outMems[j],
                                           ValueRange{bx, col});
              }
            }
            nb.create<scf::YieldOp>(nl);
          });
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }

  // ---- scan path (sparsemax fwd: block-cooperative inclusive cumsum) ------ //
  // One block per row, T threads, K = N/T CONTIGUOUS columns per thread
  // (col = tx*K + k, NOT the interleaved tx + k*T) so a thread's values form a
  // contiguous segment of the row. The full-row inclusive cumsum is:
  //   (1) each thread sequential-prefixes its K contiguous values, keeping the
  //       segment total;
  //   (2) thread 0 sequential-exclusive-scans the T segment totals through a
  //       dynamic-shared [T] buffer (+ gpu.barrier);
  //   (3) each thread adds its exclusive prefix to its local prefixes.
  // The reduces (k-count, s-sum) use the SAME [T] dynamic-shared scratch as a
  // manual tree (NOT gpu.all_reduce, whose static .shared would overlap the
  // dynamic buffer). Everything else (iota, select, map, the RowOut epilogue)
  // is the shared emitAt machinery.
  LogicalResult rewriteScan() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inTys.push_back(memrefOf(i));
    for (ResultRole rr : outRoles) {
      if (rr == ResultRole::RowOut)
        inTys.push_back(MemRefType::get({M, N}, f32));
      else // PerRowOut
        inTys.push_back(MemRefType::get({M}, f32));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

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
    // Dynamic shared: a single [T] f32 scratch, reused (sequentially, separated
    // by gpu.barrier) for the cross-thread exclusive scan and each reduce tree.
    int64_t shBytes = (int64_t)T * 4;
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
    auto scratchTy = MemRefType::get({T}, f32, nullptr, wgSpace);
    Value zoff = b.create<arith::ConstantIndexOp>(loc, 0);
    Value scratch =
        b.create<memref::ViewOp>(loc, scratchTy, raw, zoff, ValueRange{});

    // Scalars -> uniform (loaded once); then hoist constant-uniform subtrees.
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Scalar)
        uniform[arg] = b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
    }
    hoistUniforms(b, loc);

    // Contiguous column indices col[k] = tx*K + k, and the read-once register
    // tile (each Row input's K contiguous values loaded once into SSA).
    Value cK = b.create<arith::ConstantIndexOp>(loc, K);
    Value base = b.create<arith::MulIOp>(loc, tx, cK); // tx * K (contiguous)
    cols.clear();
    for (int64_t k = 0; k < K; ++k) {
      Value off = b.create<arith::ConstantIndexOp>(loc, k);
      cols.push_back(b.create<arith::AddIOp>(loc, base, off));
    }
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        SmallVector<Value> vals;
        for (int64_t k = 0; k < K; ++k)
          vals.push_back(b.create<memref::LoadOp>(loc, argMemref[arg],
                                                  ValueRange{bx, cols[k]}));
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::RowInt) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
      }
    }
    hoistUniforms(b, loc);

    Value c0i = b.create<arith::ConstantIndexOp>(loc, 0);

    // For each scan: build the full-row inclusive cumsum register tile.
    for (ScanOp sc : scans) {
      // (1) per-thread sequential inclusive prefix over its K contiguous values.
      SmallVector<Value> local(K);
      Value running;
      for (int64_t k = 0; k < K; ++k) {
        Value v = emitAt(b, loc, sc.getInput(), k);
        running = (k == 0) ? v : b.create<arith::AddFOp>(loc, running, v);
        local[k] = running;
      }
      Value segTotal = local[K - 1];
      // (2) cross-thread INCLUSIVE scan of the T segment totals via a parallel
      // Hillis-Steele scan over `scratch` (log2(T) steps, fully unrolled; T is
      // static). Each step reads scratch[tx-stride] into a register, barriers,
      // then writes scratch[tx] += that (so no read/write hazard on the buffer);
      // exclusive prefix = inclusive - segTotal.
      b.create<memref::StoreOp>(loc, segTotal, scratch, ValueRange{tx});
      b.create<gpu::BarrierOp>(loc);
      for (int64_t stride = 1; stride < T; stride <<= 1) {
        Value cStride = b.create<arith::ConstantIndexOp>(loc, stride);
        Value active =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, tx, cStride);
        // v = (tx >= stride) ? scratch[tx-stride] : 0  (read phase)
        Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
        auto rdIf = b.create<scf::IfOp>(loc, TypeRange{f32}, active,
                                        /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(rdIf.thenBlock());
          Value src = b.create<arith::SubIOp>(loc, tx, cStride);
          Value v = b.create<memref::LoadOp>(loc, scratch, ValueRange{src});
          b.create<scf::YieldOp>(loc, v);
          b.setInsertionPointToStart(rdIf.elseBlock());
          b.create<scf::YieldOp>(loc, zeroF);
        }
        b.create<gpu::BarrierOp>(loc);
        // scratch[tx] += v  (write phase, only active lanes)
        auto wrIf = b.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(wrIf.thenBlock());
          Value cur = b.create<memref::LoadOp>(loc, scratch, ValueRange{tx});
          Value nv = b.create<arith::AddFOp>(loc, cur, rdIf.getResult(0));
          b.create<memref::StoreOp>(loc, nv, scratch, ValueRange{tx});
        }
        b.create<gpu::BarrierOp>(loc);
      }
      Value incl = b.create<memref::LoadOp>(loc, scratch, ValueRange{tx});
      Value excl = b.create<arith::SubFOp>(loc, incl, segTotal);
      // (3) cssv[k] = excl + local[k] (full-row inclusive prefix).
      SmallVector<Value> vals(K);
      for (int64_t k = 0; k < K; ++k)
        vals[k] = b.create<arith::AddFOp>(loc, excl, local[k]);
      scanVals[sc.getResult()] = vals;
      // A later reduce may consume scan results; fence the scratch reuse.
      b.create<gpu::BarrierOp>(loc);
    }

    Value nF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr((double)N));

    // Per-row reduces via the manual shared tree (reusing `scratch`).
    for (ReduceOp red : reduces) {
      ReduceInfo info = *reduceInfoFor(red.getFn());
      Value acc =
          b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
      for (int64_t k = 0; k < K; ++k)
        acc = combineScalar(b, loc, info.op, acc,
                            emitAt(b, loc, red.getInput(), k));
      b.create<memref::StoreOp>(loc, acc, scratch, ValueRange{tx});
      b.create<gpu::BarrierOp>(loc);
      for (int64_t stride = T / 2; stride >= 1; stride >>= 1) {
        Value cStride = b.create<arith::ConstantIndexOp>(loc, stride);
        Value active =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx, cStride);
        auto ifOp = b.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(ifOp.thenBlock());
          Value a = b.create<memref::LoadOp>(loc, scratch, ValueRange{tx});
          Value other = b.create<arith::AddIOp>(loc, tx, cStride);
          Value bb = b.create<memref::LoadOp>(loc, scratch, ValueRange{other});
          Value c = combineScalar(b, loc, info.op, a, bb);
          b.create<memref::StoreOp>(loc, c, scratch, ValueRange{tx});
        }
        b.create<gpu::BarrierOp>(loc);
      }
      Value scalar = b.create<memref::LoadOp>(loc, scratch, ValueRange{c0i});
      if (info.isMean)
        scalar = b.create<arith::DivFOp>(loc, scalar, nF);
      uniform[red.getResult()] = scalar;
      hoistUniforms(b, loc); // tau = (s-1)/max(k,1) etc. become available now
      b.create<gpu::BarrierOp>(loc); // fence scratch before the next tree
    }

    // RowOut epilogue: write each output from the register tile (read-once).
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      if (outRoles[j] != ResultRole::RowOut)
        continue;
      for (int64_t k = 0; k < K; ++k) {
        Value out = emitAt(b, loc, outputVals[j], k);
        b.create<memref::StoreOp>(loc, out, outMems[j],
                                  ValueRange{bx, cols[k]});
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (!scans.empty())
      return rewriteScan();
    if (numTiles > 1)
      return rewriteTiled();
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: original args (as memrefs) ++ one output memref per result.
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inTys.push_back(memrefOf(i));
    for (ResultRole rr : outRoles) { // destination-passing: one memref per result
      if (rr == ResultRole::RowOut)
        inTys.push_back(MemRefType::get({M, N}, f32));
      else if (rr == ResultRole::PerRowOut)
        inTys.push_back(MemRefType::get({M}, f32));
      else // AtomicOut: a rank-0 scalar accumulated across blocks (host pre-zeroes)
        inTys.push_back(MemRefType::get({}, f32));
    }
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

    // ColSeg (group_norm): the block's group index = bx % G (rows flattened as
    // batch*G + group). Uniform per block, computed once; the affine channel map
    // is applied per column in emitAt.
    if (numGroups != 0) {
      Value cG = b.create<arith::ConstantIndexOp>(loc, numGroups);
      groupVal = b.create<arith::RemUIOp>(loc, bx, cG);
    }

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
      } else if (argRoles[i] == Role::RowInt) {
        // Y[bx] is uniform across the block's threads: load once per block.
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
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

    // Cross-row fold: each cross-row (axis-0 sum) reduce passes THIS block's
    // per-row scalar through as if it were the whole-tensor sum; the per-row
    // uniform scale (e.g. /(BT*V), /BT) then hoists onto it. Summing every
    // block's contribution via the atomicAdd below reconstructs the true total.
    for (ReduceOp cr : crossReduces)
      uniform[cr.getResult()] = uniform.lookup(cr.getInput());
    hoistUniforms(b, loc);

    // Epilogue: each RowOut written from the register tile (read-once); each
    // PerRowOut (Mean/RSTD) is a per-row uniform written ONCE by thread 0
    // (single-writer guard — M*N stores for Y vs M scalar stores for a stat).
    bool anyLead = false; // a single-writer/block store (PerRowOut or AtomicOut)
    for (ResultRole r : outRoles)
      anyLead |= (r != ResultRole::RowOut);
    Value isLead;
    if (anyLead) {
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
      } else if (outRoles[j] == ResultRole::PerRowOut) {
        Value scalar = emitAt(b, loc, outputVals[j], 0); // per-row uniform
        auto ifOp = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        b.create<memref::StoreOp>(loc, scalar, outMems[j], ValueRange{bx});
      } else { // AtomicOut: thread 0 atomic-adds this block's scaled contribution
        Value contrib = uniform.lookup(outputVals[j]); // folded per-block scalar
        if (!contrib)
          return failure(); // scale chain failed to hoist to a uniform (unreached)
        auto ifOp = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, contrib,
                                      outMems[j], ValueRange{});
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
      RowReductionLowering probe(fn, blockThreads, tileCols);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      RowReductionLowering low(fn, blockThreads, tileCols);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("row-reduction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
