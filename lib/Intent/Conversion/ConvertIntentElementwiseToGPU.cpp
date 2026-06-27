//===- ConvertIntentElementwiseToGPU.cpp - elementwise -> gpu.launch ------===//
//
// Lowers a PURE-elementwise intent func (a straight-line map_unary/map_binary
// graph with NO reduce) to ONE read-once, block-cooperative `gpu.launch`: one
// block per row, the feature dim split over the block's threads, each thread
// caching its columns once in SSA registers (NOT a memref.alloca), per-row
// uniform scalars hoisted once, and a multi-output epilogue writing each result
// from the register tile. There is NO `gpu.all_reduce` — nothing is reduced.
//
// This is the sibling of `convert-intent-row-reduction-to-gpu`: that pass owns
// the funcs with >=1 reduce; this pass owns the pure-map funcs. The two match()
// predicates are disjoint (row-reduction returns failure when it finds no
// reduce; this pass returns failure the moment it finds one), so they compose in
// either order without overlap. It is general over the map DAG — there is NO
// per-kernel "recognize dyt/swiglu/geglu" branch; the three activation kernels
// (dyt = tanh(alpha*x)*gamma, swiglu = (a*sigmoid(a))*b, geglu = the tanh-approx
// GELU gate * b) flow through identically.
//
// Schedule per block (row `bx`), per thread `tx` owning columns col = tx + k*T:
//   load each Scalar arg once (uniform); hoist every fully-uniform map subtree;
//   load each Row input's K values ONCE into SSA (the register tile);
//   epilogue: for k=0..K, for each result: store <result evaluated at col> -> Y.
//
// Read-once register tile is the whole game: a dynamically-indexed alloca lands
// in local memory and caps bandwidth far below peak. Static shapes only
// (bare-ptr ABI). Emits math.exp/tanh/...; the `-lower-gpu-math-to-nvvm`
// peephole turns those into NVVM approx intrinsics downstream.
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTELEMENTWISETOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's pure-elementwise lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct ElementwiseLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  // The 2-D logical layout of one register-tile access. A Row/result arg is
  // accessed at element (row*rowStride + col*colStride) of its bare pointer; the
  // strides come from an `intent.strides = [row, col]` arg-attr (default the
  // contiguous [N, 1]). A NON-default stride means the arg is a strided VIEW of a
  // larger buffer (rope's q_lo/q_hi half-views have rowStride = hd = 2N; llama4's
  // even/odd views have colStride = 2): the index-remap that makes rope a pure
  // elementwise op at matching logical column instead of a cross-column rotation.
  struct Strides {
    int64_t row, col;
    bool isDefault(int64_t N) const { return row == N && col == 1; }
  };

  // populated by match()
  int64_t M = 0, N = 0, T = 0, K = 0;
  Type f32;
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<int64_t> argRowDiv;    // per arg: 1, or M/d0 for a broadcast Row
  SmallVector<Strides> argStrides;   // per arg: [row,col] (Row only; else unused)
  SmallVector<Strides> resStrides;   // per result: [row,col]
  SmallVector<int64_t> inplaceArg;   // per result: aliased input arg idx, or -1
  SmallVector<Value> outputVals;     // the returned values (>=1, all [M,N])

  ElementwiseLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  /// memref type for a Row/result view: `memref<d0 x N x f32>` (identity layout
  /// when contiguous, so dyt/swiglu/geglu stay byte-identical) or a strided
  /// `memref<d0 x N x f32, strided<[row, col], offset: 0>>`. Offset is 0: the
  /// bare pointer the harness passes is the strided torch view's data_ptr, which
  /// already folds in any base offset (q[:, hd/2:].data_ptr()).
  MemRefType rowMemref(MLIRContext *ctx, int64_t d0, Strides s) {
    if (s.isDefault(N))
      return MemRefType::get({d0, N}, f32);
    return MemRefType::get({d0, N}, f32,
                           StridedLayoutAttr::get(ctx, 0, {s.row, s.col}));
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    auto fty = fn.getFunctionType();
    unsigned nRes = fty.getNumResults();
    if (nRes < 1)
      return failure();
    // Every result is a full rank-2 static f32 [M,N] row output. Anchor (M,N)
    // from the first; all must agree (pure elementwise = no shape change).
    int64_t anchorM = -1, anchorN = -1;
    for (Type rt : fty.getResults()) {
      auto t = dyn_cast<RankedTensorType>(rt);
      if (!t || t.getRank() != 2 || !t.hasStaticShape())
        return failure();
      if (!isa<Float32Type>(t.getElementType()))
        return failure();
      f32 = t.getElementType();
      if (anchorN < 0) {
        anchorM = t.getDimSize(0);
        anchorN = t.getDimSize(1);
      } else if (t.getDimSize(0) != anchorM || t.getDimSize(1) != anchorN)
        return failure();
    }
    M = anchorM;
    N = anchorN;

    // Single basic block, terminated by a return of the output values.
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != nRes)
      return failure();
    outputVals.assign(ret.getOperands().begin(), ret.getOperands().end());

    // Per-result strides (default contiguous). `intent.result_strides` is a flat
    // [row0,col0, row1,col1, ...] of length 2*nRes — rope's outputs are strided
    // half-views of a fresh [M,hd] buffer (rowStride hd = 2N).
    resStrides.assign(nRes, Strides{N, 1});
    if (auto a = fn->getAttrOfType<DenseI64ArrayAttr>("intent.result_strides")) {
      if ((unsigned)a.size() != 2 * nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j)
        resStrides[j] = {a[2 * j], a[2 * j + 1]};
    }

    // In-place aliasing: `intent.inplace = [a0, a1, ...]` (length nRes) maps each
    // result to the input arg whose buffer it overwrites (-1 = a fresh
    // destination memref). rope/llama4/qwen2vl rotate q/k IN PLACE: y_lo aliases
    // q_lo, y_hi aliases q_hi. The read-once register tile loads every input
    // before any store, so writing back through an input view is safe.
    inplaceArg.assign(nRes, -1);
    if (auto a = fn->getAttrOfType<DenseI64ArrayAttr>("intent.inplace")) {
      if ((unsigned)a.size() != nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j)
        inplaceArg[j] = a[j];
    }

    // Args: classify each by shape; only Row/Col/Scalar f32 tensors allowed.
    // A 2-D arg whose first dim d0 < M but DIVIDES M is a "broadcast Row": one
    // row is shared by M/d0 consecutive blocks (rope's cos/sin: one per token,
    // shared by the token's n_head blocks). An `intent.strides` arg-attr makes
    // the Row a strided view of a larger buffer (half / even-odd partner views).
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      auto t = dyn_cast<RankedTensorType>(fty.getInput(i));
      if (!t || !t.hasStaticShape() || t.getElementType() != f32)
        return failure();
      ArrayRef<int64_t> s = t.getShape();
      Strides st{N, 1};
      if (auto a = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "intent.strides")) {
        if (a.size() != 2)
          return failure();
        st = {a[0], a[1]};
      }
      int64_t div = 1;
      Role role;
      if (s.size() == 2 && s[1] == N) {
        if (s[0] == M)
          div = 1;
        else if (s[0] > 0 && M % s[0] == 0)
          div = M / s[0];
        else
          return failure();
        role = Role::Row;
      } else if (s.size() == 1 && s[0] == N) {
        role = Role::Col;
      } else if (s.size() == 0) {
        role = Role::Scalar;
      } else {
        return failure();
      }
      argRoles.push_back(role);
      argRowDiv.push_back(div);
      argStrides.push_back(st);
    }

    // Body ops: only map_unary / map_binary. ANY reduce => not our pass (it
    // belongs to convert-intent-row-reduction-to-gpu). This keeps the two
    // passes' match() predicates strictly disjoint.
    // map_unary / map_binary (incl. comparison fns -> i1) / select. The i1
    // comparison tensor is an intermediate only — match constrains arg/result
    // element types, not intermediates. ANY reduce => not our pass.
    for (Operation &op : body.without_terminator()) {
      if (isa<ReduceOp>(&op))
        return failure();
      if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, IotaOp>(&op))
        return failure();
    }

    // Geometry: T threads over the feature dim, K columns per thread.
    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || N % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented follow-up
    K = N / T;
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  // The feature tile is held in REGISTERS, not a memref cache: K is static and
  // small, so we fully unroll over k and keep each Row input's K loaded values
  // as SSA (exactly the library's register tile).
  DenseMap<Value, Value> uniform;              // value -> hoisted per-row scalar
  DenseMap<Value, Value> argMemref;            // arg -> memref arg
  DenseMap<Value, SmallVector<Value>> rowVals; // Row arg -> K loaded SSA values
  SmallVector<Value> cols;                     // K column indices (tx + k*T)

  /// Evaluate an original value for the thread's k-th column (fully unrolled).
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
    if (auto io = dyn_cast<IotaOp>(def)) {
      // iota at column col[k] is the column index itself, cast to the op's
      // integer width (no materialized vector).
      Type elt =
          cast<RankedTensorType>(io.getResult().getType()).getElementType();
      return b.create<arith::IndexCastOp>(loc, elt, cols[k]);
    }
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return emitScalarUnary(b, loc, mu.getFn(), emitAt(b, loc, mu.getInput(), k));
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      Value lhs = emitAt(b, loc, mb.getLhs(), k);
      Value rhs = emitAt(b, loc, mb.getRhs(), k);
      if (isCmpFn(mb.getFn()))
        return isa<IntegerType>(lhs.getType())
                   ? emitScalarCmpI(b, loc, mb.getFn(), lhs, rhs)
                   : emitScalarCmp(b, loc, mb.getFn(), lhs, rhs);
      return emitScalarBinary(b, loc, mb.getFn(), lhs, rhs);
    }
    if (auto sel = dyn_cast<SelectOp>(def))
      return b.create<arith::SelectOp>(loc, emitAt(b, loc, sel.getCond(), k),
                                       emitAt(b, loc, sel.getX(), k),
                                       emitAt(b, loc, sel.getY(), k));
    return nullptr;
  }

  /// Is an original value uniform (per-row / rank-0)? Row/Col args are not.
  bool isUniform(Value v) {
    if (uniform.count(v))
      return true;
    if (auto barg = dyn_cast<BlockArgument>(v))
      return argRoles[barg.getArgNumber()] == Role::Scalar;
    return false;
  }

  /// Hoist every map op whose operands are all uniform into `uniform` (once).
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
          // Comparison fns are never hoisted here: their condition depends on a
          // Row input (p > q), and emitScalarBinary returns null for a cmp fn.
          // A fully-uniform cmp/select (rare) is simply recomputed per column in
          // emitAt — still correct, just not hoisted.
          if (isCmpFn(mb.getFn()))
            continue;
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

  Type memrefOfArg(MLIRContext *ctx, unsigned i) {
    Role role = argRoles[i];
    if (role == Role::Row)
      return rowMemref(ctx, M / argRowDiv[i], argStrides[i]);
    if (role == Role::Col)
      return MemRefType::get({N}, f32);
    return MemRefType::get({}, f32);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: original args (as memrefs) ++ one [M,N] output memref per
    // result (destination-passing).
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inTys.push_back(memrefOfArg(ctx, i));
    for (unsigned j = 0; j < outputVals.size(); ++j)
      if (inplaceArg[j] < 0) // fresh destination; in-place reuses an input arg
        inTys.push_back(rowMemref(ctx, M, resStrides[j]));
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
    SmallVector<Value> outMems(outputVals.size());
    unsigned freshIdx = argRoles.size();
    for (unsigned j = 0; j < outputVals.size(); ++j)
      outMems[j] = inplaceArg[j] >= 0
                       ? entry->getArgument(inplaceArg[j])
                       : entry->getArgument(freshIdx++);

    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    auto launch = b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Column indices col[k] = tx + k*T, and the read-once register tile: each
    // Row input's K values loaded ONCE into SSA.
    cols.clear();
    for (int64_t k = 0; k < K; ++k) {
      Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
      cols.push_back(b.create<arith::AddIOp>(loc, tx, off));
    }
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        // Broadcast Row (div>1): one source row shared by `div` consecutive
        // blocks -> source row index = bx / div (rope cos: token = bx/n_head).
        Value ridx = bx;
        if (argRowDiv[i] != 1) {
          Value d = b.create<arith::ConstantIndexOp>(loc, argRowDiv[i]);
          ridx = b.create<arith::DivUIOp>(loc, bx, d);
        }
        SmallVector<Value> vals;
        for (int64_t k = 0; k < K; ++k)
          vals.push_back(b.create<memref::LoadOp>(loc, argMemref[arg],
                                                  ValueRange{ridx, cols[k]}));
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::Scalar) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
      }
    }
    hoistUniforms(b, loc);

    // Epilogue: each result written from the register tile (read-once). No
    // reduction, no all_reduce, no thread-0 guard — every thread writes its own
    // columns for every output.
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      for (int64_t k = 0; k < K; ++k) {
        Value out = emitAt(b, loc, outputVals[j], k);
        b.create<memref::StoreOp>(loc, out, outMems[j], ValueRange{bx, cols[k]});
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentElementwiseToGPUPass
    : ::intent::impl::ConvertIntentElementwiseToGPUBase<
          ConvertIntentElementwiseToGPUPass> {
  using ConvertIntentElementwiseToGPUBase::ConvertIntentElementwiseToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ElementwiseLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ElementwiseLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("elementwise lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
