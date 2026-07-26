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
// (bare-ptr ABI). f16/bf16 storage is promoted to f32 within each semantic map
// op and rounded back at that op's typed result boundary. Emits
// math.exp/tanh/...; the `-lower-gpu-math-to-nvvm` peephole turns those into
// NVVM approx intrinsics downstream.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTELEMENTWISETOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// Paper extraction represents literal broadcasts as tensor splat constants.
/// They are uniform scalar values for this elementwise lowering, independent of
/// whether the source tensor type is rank-0 or a larger splat tensor.
static bool isSupportedSplatConstant(arith::ConstantOp cst) {
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return false;
  auto shaped = dyn_cast<ShapedType>(dense.getType());
  return shaped && isa<FloatType, IntegerType>(shaped.getElementType());
}

static Value emitSplatScalarConstant(OpBuilder &b, Location loc,
                                     arith::ConstantOp cst) {
  auto dense = cast<DenseElementsAttr>(cst.getValue());
  Type elemTy = cast<ShapedType>(dense.getType()).getElementType();
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(floatTy, dense.getSplatValue<APFloat>()));
  auto intTy = cast<IntegerType>(elemTy);
  return b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(intTy, dense.getSplatValue<APInt>()));
}

static void attachElementwiseStateReceipt(func::FuncOp lowered, int64_t threads,
                                          StringRef approximationMode,
                                          bool oneDimensionalResults,
                                          unsigned hostValueCaptureCount) {
  MLIRContext *context = lowered.getContext();
  SmallVector<Attribute> trace{
      StringAttr::get(context, "cuda.elementwise.row.vectorized.v1")};
  if (oneDimensionalResults)
    trace.push_back(
        StringAttr::get(context, "cuda.elementwise.rank1.output.v1"));
  if (approximationMode == "native_tanh")
    trace.push_back(
        StringAttr::get(context, "cuda.numeric.tanh_native_ptx.f32.v1"));
  else if (approximationMode == "stable_tanh")
    trace.push_back(
        StringAttr::get(context, "cuda.numeric.tanh_stable_ex2.f32.v1"));
  if (hostValueCaptureCount != 0)
    trace.push_back(
        StringAttr::get(context, "cuda.b.uniform_host_value_capture.v1"));
  auto i64 = IntegerType::get(context, 64);
  lowered->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(context, "elementwise"));
  lowered->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(context, "none"));
  lowered->setAttr("intent.backend.approximation_mode",
                   StringAttr::get(context, approximationMode));
  lowered->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(context, trace));
  lowered->setAttr("intent.backend.fma_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.shuffle_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.barrier_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.all_reduce_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.block_threads",
                   IntegerAttr::get(i64, threads));
  lowered->setAttr("intent.backend.predicated_value_if_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.folded_constexpr_select_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.host_value_capture_count",
                   IntegerAttr::get(i64, hostValueCaptureCount));
  lowered->setAttr("intent.backend.epilogue_load_motion_count",
                   IntegerAttr::get(i64, 0));
}

/// One func's pure-elementwise lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct ElementwiseLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  bool requestFlattenContiguous;
  unsigned requestFlatVectorWidth;
  bool requestPromoteUniformScalars;
  std::string requestTanhApproximation;

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
  int64_t M = 0, N = 0, T = 0, K = 0, gridBlocks = 0;
  bool oneDimensionalResults = false;
  bool flattenContiguous = false;
  bool tailMasked = false;
  // Contiguous vectorization (N3/N4): prefer float4 when K is divisible by 4,
  // otherwise use float2 when K is divisible by 2. Every Row arg/result must
  // be contiguous and the body must have no iota/select scalar gather. This
  // lets a four-warp N=256 schedule retain Triton's coalesced v2 memory shape
  // while the normalized Intent DAG still exposes target-level fast sigmoid.
  int64_t vectorWidth = 1;
  int64_t numVectorTiles = 0;
  Type storageType;
  Type computeType;
  bool containsTanh = false;
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<Type> argElementTypes; // per func arg storage/broadcast type
  SmallVector<int64_t> argRowDiv;    // per arg: 1, or M/d0 for a broadcast Row
  SmallVector<Strides> argStrides;   // per arg: [row,col] (Row only; else unused)
  SmallVector<Strides> resStrides;   // per result: [row,col]
  SmallVector<int64_t> inplaceArg;   // per result: aliased input arg idx, or -1
  SmallVector<Value> outputVals;     // the returned values (>=1, all [M,N])

  ElementwiseLowering(func::FuncOp fn, unsigned bt, bool flatten,
                      unsigned flatWidth, bool promoteUniformScalars,
                      StringRef tanhApproximation)
      : fn(fn), blockThreads(bt), requestFlattenContiguous(flatten),
        requestFlatVectorWidth(flatWidth),
        requestPromoteUniformScalars(promoteUniformScalars),
        requestTanhApproximation(tanhApproximation.str()) {}

  /// Memref type for a Row/result view in the function's storage type.
  /// `memref<d0 x N x storage>` uses identity layout when contiguous, otherwise
  /// a strided
  /// `memref<d0 x N x storage, strided<[row, col], offset: 0>>`. Offset is 0: the
  /// bare pointer the harness passes is the strided torch view's data_ptr, which
  /// already folds in any base offset (q[:, hd/2:].data_ptr()).
  MemRefType rowMemref(MLIRContext *ctx, int64_t d0, Strides s) {
    if (s.isDefault(N))
      return MemRefType::get({d0, N}, storageType);
    return MemRefType::get({d0, N}, storageType,
                           StridedLayoutAttr::get(ctx, 0, {s.row, s.col}));
  }

  /// Vector-element memref for a contiguous Row/result view. Only used when
  /// vectorWidth > 1; the caller has already proved default contiguous strides.
  MemRefType rowVecMemref(MLIRContext *ctx, int64_t d0) {
    return MemRefType::get({d0, N / vectorWidth},
                           VectorType::get({vectorWidth}, storageType));
  }

  MemRefType flatVecMemref(MLIRContext *ctx) {
    return MemRefType::get({M * N / vectorWidth},
                           VectorType::get({vectorWidth}, storageType));
  }

  /// Destination-passing ABI for a result.  Public rank-1 tensors remain
  /// rank-1 memrefs; they are not silently reshaped to [1,N], because that
  /// would make the compiler ABI disagree with the accepted J shape.
  MemRefType resultMemref(MLIRContext *ctx, Strides s) {
    if (oneDimensionalResults)
      return MemRefType::get({N}, storageType);
    return rowMemref(ctx, M, s);
  }

  MemRefType resultVecMemref(MLIRContext *ctx) {
    if (oneDimensionalResults)
      return MemRefType::get({N / vectorWidth},
                             VectorType::get({vectorWidth}, storageType));
    return rowVecMemref(ctx, M);
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (requestTanhApproximation != "stable-ex2" &&
        requestTanhApproximation != "native-ptx")
      return failure();
    auto fty = fn.getFunctionType();
    unsigned nRes = fty.getNumResults();
    if (nRes < 1)
      return failure();
    // Every result is either a static floating [N] vector or a full rank-2
    // floating [M,N] row output.  Rank-1 is represented as one logical row for
    // launch geometry only; its public destination-passing ABI remains [N].
    // f16 and bf16 are storage types; scalar computation is promoted to f32.
    // Anchor (rank,M,N,storage) from the first; all results must agree.
    int64_t anchorRank = -1, anchorM = -1, anchorN = -1;
    for (Type rt : fty.getResults()) {
      auto t = dyn_cast<RankedTensorType>(rt);
      if (!t || (t.getRank() != 1 && t.getRank() != 2) ||
          !t.hasStaticShape())
        return failure();
      Type elementType = t.getElementType();
      if (!isa<Float16Type, BFloat16Type, Float32Type>(elementType))
        return failure();
      if (anchorN < 0) {
        anchorRank = t.getRank();
        anchorM = anchorRank == 1 ? 1 : t.getDimSize(0);
        anchorN = t.getDimSize(anchorRank - 1);
        storageType = elementType;
      } else if (t.getRank() != anchorRank ||
                 (anchorRank == 2 && t.getDimSize(0) != anchorM) ||
                 t.getDimSize(anchorRank - 1) != anchorN)
        return failure();
      else if (elementType != storageType)
        return failure();
    }
    M = anchorM;
    N = anchorN;
    oneDimensionalResults = anchorRank == 1;
    computeType = Float32Type::get(fn.getContext());

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
      // The registered relation currently defines row/column strides only for
      // rank-2 views.  Fail closed instead of inventing rank-1 semantics.
      if (oneDimensionalResults)
        return failure();
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
      // Rank-2 Row aliases are loaded into the complete register tile before
      // any output store.  Rank-1 inputs are Col operands and are loaded on
      // demand, so accepting an alias could make one result overwrite a value
      // subsequently consumed by another result.  Keep this case conservative
      // until a rank-1 read-once tile is represented explicitly.
      if (oneDimensionalResults)
        return failure();
      if ((unsigned)a.size() != nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j)
        inplaceArg[j] = a[j];
    }

    // Args: Row inputs use the result storage type. Col/Scalar broadcasts may
    // additionally be f32 (the common mixed-precision scale/bias contract).
    // A singleton rank-1 tensor is also a scalar broadcast: the public ABI
    // remains memref<1xT>, while the kernel reads element zero once.  Closed J
    // artifacts use this spelling for scalar-like learned parameters.
    // A 2-D arg whose first dim d0 < M but DIVIDES M is a "broadcast Row": one
    // row is shared by M/d0 consecutive blocks (rope's cos/sin: one per token,
    // shared by the token's n_head blocks). An `intent.strides` arg-attr makes
    // the Row a strided view of a larger buffer (half / even-odd partner views).
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      auto t = dyn_cast<RankedTensorType>(fty.getInput(i));
      if (!t || !t.hasStaticShape())
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
      bool floating =
          isa<Float16Type, BFloat16Type, Float32Type>(t.getElementType());
      if (body.getArgument(i).use_empty() &&
          isa<FloatType, IntegerType>(t.getElementType())) {
        // Constexpr control specialization can make a branch-only ABI input
        // dead (for example an optional label vector). Preserve the public
        // memref ABI, but do not force an unused tensor into a Row/Col/Scalar
        // execution role or synthesize a device load.
        role = Role::Unused;
      } else if (s.size() == 2 && s[1] == N && floating) {
        if (s[0] == M)
          div = 1;
        else if (s[0] > 0 && M % s[0] == 0)
          div = M / s[0];
        else
          return failure();
        role = Role::Row;
      } else if (s.size() == 1 && s[0] == N && floating) {
        role = Role::Col;
      } else if ((s.empty() || (s.size() == 1 && s[0] == 1)) &&
                 isa<FloatType, IntegerType>(t.getElementType())) {
        role = Role::Scalar;
      } else {
        return failure();
      }
      if (role == Role::Row && t.getElementType() != storageType)
        return failure();
      argRoles.push_back(role);
      argElementTypes.push_back(t.getElementType());
      argRowDiv.push_back(div);
      argStrides.push_back(st);
    }

    // Body ops: tensor splat constants plus map_unary / map_binary / select /
    // iota. ANY reduce => not our pass (it belongs to
    // convert-intent-row-reduction-to-gpu). Paper extraction emits literal
    // broadcasts as rank-0 tensor constants, so rejecting arith.constant here
    // would force a semantically faithful primitive activation graph onto the
    // generic multi-kernel fallback.
    for (Operation &op : body.without_terminator()) {
      if (isa<ReduceOp>(&op))
        return failure();
      if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
        if (!isSupportedSplatConstant(cst))
          return failure();
        continue;
      }
      if (auto unary = dyn_cast<MapUnaryOp>(&op))
        containsTanh |= unary.getFn() == "tanh";
      if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, IotaOp, CastOp>(&op))
        return failure();
    }
    if (requestTanhApproximation == "native-ptx" && !containsTanh)
      return failure();

    // A flattened schedule is admitted only for a fully contiguous Row-only
    // graph. It deliberately rejects Col/scalar parameters, broadcast rows,
    // strided views and indexed/select bodies rather than inventing a mixed
    // linear/row mapping. Each thread owns exactly one vec4 packet.
    if (requestFlattenContiguous) {
      T = blockThreads ? (int64_t)blockThreads : 256;
      int64_t flatWidth = static_cast<int64_t>(requestFlatVectorWidth);
      bool eligible = T >= 32 && T % 32 == 0 &&
                      (flatWidth == 2 || flatWidth == 4);
      for (unsigned i = 0; i < argRoles.size(); ++i)
        eligible &= argRoles[i] == Role::Row && argRowDiv[i] == 1 &&
                    argStrides[i].isDefault(N);
      for (Strides strides : resStrides)
        eligible &= strides.isDefault(N);
      for (Operation &op : body.without_terminator())
        eligible &= !isa<IotaOp, SelectOp>(&op);
      int64_t totalElements = M * N;
      eligible &= totalElements % (T * flatWidth) == 0;
      if (!eligible)
        return failure();
      flattenContiguous = true;
      vectorWidth = flatWidth;
      K = flatWidth;
      numVectorTiles = 1;
      gridBlocks = totalElements / (T * vectorWidth);
      return success();
    }

    // Default geometry: T threads over the feature dim, K columns per thread.
    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || T > 1024)
      return failure();
    tailMasked = N % T != 0;
    K = (N + T - 1) / T;
    gridBlocks = M;

    // Vector gate: K a multiple of 4 (preferred) or 2, every Row arg + every
    // result view fully
    // CONTIGUOUS (default [N,1] strides — a strided half/even-odd view cannot be
    // a contiguous vec4 load, so rope/llama4/qwen2vl stay on the scalar path),
    // and no iota/select in the body (those index a per-column scalar, not a
    // vectorizable lane). Col/Scalar args stay scalar-loaded at col[k]/once —
    // the vector column layout keeps cols[k] pointing at the right scalar column.
    int64_t candidateVectorWidth =
        !tailMasked && T >= 32
            ? (K % 4 == 0 ? 4 : (K % 2 == 0 ? 2 : 1))
            : 1;
    if (candidateVectorWidth > 1) {
      bool ok = true;
      for (unsigned i = 0; i < argRoles.size(); ++i)
        if (argRoles[i] == Role::Row && !argStrides[i].isDefault(N))
          ok = false;
      for (unsigned j = 0; j < resStrides.size(); ++j)
        if (!resStrides[j].isDefault(N))
          ok = false;
      for (Operation &op : body.without_terminator())
        if (isa<IotaOp, SelectOp>(&op))
          ok = false;
      if (ok) {
        vectorWidth = candidateVectorWidth;
        numVectorTiles = K / vectorWidth;
      }
    }
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

  Value toCompute(OpBuilder &b, Location loc, Value value) const {
    if (value.getType() == computeType || !isa<FloatType>(value.getType()))
      return value;
    if (isa<Float16Type, BFloat16Type>(value.getType()))
      return b.create<arith::ExtFOp>(loc, computeType, value);
    return value;
  }

  Value toStorage(OpBuilder &b, Location loc, Value value) const {
    if (value.getType() == storageType)
      return value;
    if (value.getType() == computeType &&
        isa<Float16Type, BFloat16Type>(storageType))
      return b.create<arith::TruncFOp>(loc, storageType, value);
    return value;
  }

  Type tensorElementType(Value value) const {
    return cast<RankedTensorType>(value.getType()).getElementType();
  }

  Value castFloatTo(OpBuilder &b, Location loc, Value value,
                    Type targetType) const {
    if (value.getType() == targetType || !isa<FloatType>(value.getType()) ||
        !isa<FloatType>(targetType))
      return value;
    if (targetType == computeType &&
        isa<Float16Type, BFloat16Type>(value.getType()))
      return b.create<arith::ExtFOp>(loc, targetType, value);
    if (value.getType() == computeType &&
        isa<Float16Type, BFloat16Type>(targetType))
      return b.create<arith::TruncFOp>(loc, targetType, value);
    return value;
  }

  Value emitMappedUnary(OpBuilder &b, Location loc, MapUnaryOp op,
                        Value input) const {
    Type resultType = tensorElementType(op.getResult());
    if (isa<Float16Type, BFloat16Type>(input.getType()) ||
        isa<Float16Type, BFloat16Type>(resultType))
      input = toCompute(b, loc, input);
    Value result;
    if (op.getFn() == "tanh" &&
        requestTanhApproximation == "native-ptx") {
      // Keep the semantic math op intact until the outlined NVIDIA device
      // module.  emitScalarUnary's default tanh is already an ex2/reciprocal
      // expansion, so tagging its final arith.subf cannot possibly recover a
      // native tanh instruction later in the target pipeline.
      auto tanh = b.create<math::TanhOp>(loc, input);
      tanh->setAttr(
          "intent.backend.tanh_approximation",
          StringAttr::get(b.getContext(), "native-ptx"));
      result = tanh;
    } else {
      result = emitScalarUnary(b, loc, op.getFn(), input);
    }
    return result ? castFloatTo(b, loc, result, resultType) : Value();
  }

  Value emitMappedBinary(OpBuilder &b, Location loc, MapBinaryOp op,
                         Value lhs, Value rhs) const {
    if (isCmpFn(op.getFn())) {
      if (isa<IntegerType>(lhs.getType()))
        return emitScalarCmpI(b, loc, op.getFn(), lhs, rhs);
      if (lhs.getType() != computeType || rhs.getType() != computeType) {
        lhs = toCompute(b, loc, lhs);
        rhs = toCompute(b, loc, rhs);
      }
      return emitScalarCmp(b, loc, op.getFn(), lhs, rhs);
    }

    Type resultType = tensorElementType(op.getResult());
    if (isa<FloatType>(lhs.getType()) &&
        (lhs.getType() != computeType || rhs.getType() != computeType ||
         resultType != computeType)) {
      lhs = toCompute(b, loc, lhs);
      rhs = toCompute(b, loc, rhs);
    }
    Value result = emitScalarBinary(b, loc, op.getFn(), lhs, rhs);
    return result ? castFloatTo(b, loc, result, resultType) : Value();
  }

  /// Evaluate an original value for the thread's k-th column (fully unrolled).
  Value emitAt(OpBuilder &b, Location loc, Value v, unsigned k) {
    if (Value u = uniform.lookup(v))
      return u;
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Row)
        return rowVals[v][k];
      if (role == Role::Col)
        return b.create<memref::LoadOp>(loc, argMemref[v],
                                       ValueRange{cols[k]});
      if (role == Role::Unused)
        return nullptr;
      return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{});
    }
    Operation *def = v.getDefiningOp();
    if (auto cst = dyn_cast<arith::ConstantOp>(def))
      return emitSplatScalarConstant(b, loc, cst);
    if (auto io = dyn_cast<IotaOp>(def)) {
      // iota at column col[k] is the column index itself, cast to the op's
      // integer width (no materialized vector).
      Type elt =
          cast<RankedTensorType>(io.getResult().getType()).getElementType();
      return b.create<arith::IndexCastOp>(loc, elt, cols[k]);
    }
    if (auto castOp = dyn_cast<CastOp>(def)) {
      Type target = tensorElementType(castOp.getResult());
      return castFloatTo(
          b, loc, emitAt(b, loc, castOp.getInput(), k), target);
    }
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return emitMappedUnary(b, loc, mu, emitAt(b, loc, mu.getInput(), k));
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      Value lhs = emitAt(b, loc, mb.getLhs(), k);
      Value rhs = emitAt(b, loc, mb.getRhs(), k);
      return emitMappedBinary(b, loc, mb, lhs, rhs);
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
    if (auto cst = v.getDefiningOp<arith::ConstantOp>())
      return isSupportedSplatConstant(cst);
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
        if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
          uniform[res] = emitSplatScalarConstant(b, loc, cst);
          changed = true;
        } else if (auto castOp = dyn_cast<CastOp>(&op)) {
          if (isUniform(castOp.getInput())) {
            uniform[res] = castFloatTo(
                b, loc, emitAt(b, loc, castOp.getInput(), 0),
                tensorElementType(castOp.getResult()));
            changed = true;
          }
        } else if (auto mu = dyn_cast<MapUnaryOp>(&op)) {
          if (isUniform(mu.getInput())) {
            uniform[res] = emitMappedUnary(
                b, loc, mu, emitAt(b, loc, mu.getInput(), 0));
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
            uniform[res] = emitMappedBinary(
                b, loc, mb, emitAt(b, loc, mb.getLhs(), 0),
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
      return MemRefType::get({N}, argElementTypes[i]);
    auto original = cast<RankedTensorType>(fn.getFunctionType().getInput(i));
    if (role == Role::Unused)
      return MemRefType::get(original.getShape(), argElementTypes[i]);
    if (original.getRank() == 1)
      return MemRefType::get({1}, argElementTypes[i]);
    return MemRefType::get({}, argElementTypes[i]);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // New signature: original args (as memrefs) ++ one shape-preserving output
    // memref per result (destination-passing).
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (flattenContiguous && argRoles[i] == Role::Row)
        inTys.push_back(flatVecMemref(ctx));
      else if (vectorWidth > 1 && argRoles[i] == Role::Row)
        inTys.push_back(rowVecMemref(ctx, M / argRowDiv[i]));
      else
        inTys.push_back(memrefOfArg(ctx, i));
    }
    for (unsigned j = 0; j < outputVals.size(); ++j)
      if (inplaceArg[j] < 0) // fresh destination; in-place reuses an input arg
        inTys.push_back(
            flattenContiguous
                ? flatVecMemref(ctx)
                : (vectorWidth > 1 ? resultVecMemref(ctx)
                                   : resultMemref(ctx, resStrides[j])));
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

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridBlocks);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    // A read-only, non-alias B capability may admit host-by-value capture for
    // uniform invocation inputs.  This is a real search arm: the false arm
    // below retains the historical device-side load in every launched lane.
    unsigned hostValueCaptureCount = 0;
    if (requestPromoteUniformScalars) {
      for (unsigned i = 0; i < argRoles.size(); ++i) {
        if (argRoles[i] != Role::Scalar)
          continue;
        Value arg = srcBlock.getArgument(i);
        auto original = cast<RankedTensorType>(
            fn.getFunctionType().getInput(i));
        if (original.getRank() == 1) {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          uniform[arg] = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{zero});
        } else {
          uniform[arg] =
              b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
        }
        ++hostValueCaptureCount;
      }
    }

    auto launch = b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Column indices + the read-once register tile: each Row input's K values
    // loaded ONCE into SSA. Two layouts:
    //  - scalar path: col[k] = tx + k*T (interleaved, coalesced scalar loads).
    //  - vector path: the K columns split into K/vectorWidth tiles; the register
    //    value at k = vt*W + j has scalar col vt*(T*W) + tx*W + j, i.e. vec index
    //    (tx + vt*T) in memref<d0x(N/W)xvector<Wxstorage>>. Each tile is one
    //    coalesced vector load across the warp; its lanes are extracted
    //    into the SAME register tile the scalar path feeds emitAt. Col/Scalar
    //    args still index cols[k]/once, so they need no retype.
    cols.clear();
    cols.resize(K);
    Type vectorTy = VectorType::get({vectorWidth}, storageType);
    SmallVector<Value> vecIdx; // vector tile index tx+vt*T (reused by store)
    if (vectorWidth > 1) {
      vecIdx.resize(numVectorTiles);
      Value cWidth = b.create<arith::ConstantIndexOp>(loc, vectorWidth);
      Value flatBlockBase;
      if (flattenContiguous) {
        Value flatBlockThreads =
            b.create<arith::ConstantIndexOp>(loc, T);
        flatBlockBase =
            b.create<arith::MulIOp>(loc, bx, flatBlockThreads);
      }
      for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
        Value off = b.create<arith::ConstantIndexOp>(loc, vt * T);
        Value threadTile = b.create<arith::AddIOp>(loc, tx, off);
        vecIdx[vt] = flattenContiguous
                         ? b.create<arith::AddIOp>(loc, flatBlockBase,
                                                  threadTile)
                         : threadTile;
        Value base = b.create<arith::MulIOp>(loc, vecIdx[vt], cWidth);
        for (int64_t j = 0; j < vectorWidth; ++j) {
          Value cj = b.create<arith::ConstantIndexOp>(loc, j);
          cols[vt * vectorWidth + j] =
              b.create<arith::AddIOp>(loc, base, cj);
        }
      }
    } else {
      for (int64_t k = 0; k < K; ++k) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        cols[k] = b.create<arith::AddIOp>(loc, tx, off);
      }
    }
    SmallVector<Value> columnActive(K);
    if (tailMasked) {
      Value cN = b.create<arith::ConstantIndexOp>(loc, N);
      Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
      for (int64_t k = 0; k < K; ++k) {
        columnActive[k] = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, cols[k], cN);
        // Use an in-bounds address for the inactive tail lane. Its value is
        // observationally dead because the result store is guarded below.
        // This lowers to one predicated select + ordinary coalesced load,
        // avoiding a divergent scf.if around every input load.
        cols[k] = b.create<arith::SelectOp>(loc, columnActive[k], cols[k], c0);
      }
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
        SmallVector<Value> vals(K);
        if (vectorWidth > 1) {
          // One coalesced vector load per tile; the flattened schedule uses a
          // rank-1 packet memref, while the row schedule keeps [row, packet].
          for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
            Value v = flattenContiguous
                          ? b.create<memref::LoadOp>(
                                loc, argMemref[arg], ValueRange{vecIdx[vt]})
                          : b.create<memref::LoadOp>(
                                loc, argMemref[arg],
                                ValueRange{ridx, vecIdx[vt]});
            for (int64_t j = 0; j < vectorWidth; ++j)
              vals[vt * vectorWidth + j] =
                  b.create<vector::ExtractOp>(loc, v,
                                              ArrayRef<int64_t>{j});
          }
        } else {
          for (int64_t k = 0; k < K; ++k) {
            vals[k] = b.create<memref::LoadOp>(
                loc, argMemref[arg], ValueRange{ridx, cols[k]});
          }
        }
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::Scalar) {
        if (uniform.count(arg))
          continue;
        auto original = cast<RankedTensorType>(
            fn.getFunctionType().getInput(i));
        if (original.getRank() == 1) {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          uniform[arg] = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{zero});
        } else {
          uniform[arg] =
              b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
        }
      }
    }
    hoistUniforms(b, loc);

    // Epilogue: each result written from the register tile (read-once). No
    // reduction, no all_reduce, no thread-0 guard — every thread writes its own
    // columns for every output.
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      if (vectorWidth > 1) {
        // Group the contiguous lanes of each tile into one vector store.
        for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
          SmallVector<Value> lanes(vectorWidth);
          for (int64_t lj = 0; lj < vectorWidth; ++lj)
            lanes[lj] = toStorage(
                b, loc,
                emitAt(b, loc, outputVals[j], vt * vectorWidth + lj));
          Value vec = b.create<vector::FromElementsOp>(loc, vectorTy, lanes);
          if (flattenContiguous)
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{vecIdx[vt]});
          else if (oneDimensionalResults)
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{vecIdx[vt]});
          else
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{bx, vecIdx[vt]});
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          if (!tailMasked) {
            Value out =
                toStorage(b, loc, emitAt(b, loc, outputVals[j], k));
            if (oneDimensionalResults)
              b.create<memref::StoreOp>(loc, out, outMems[j],
                                        ValueRange{cols[k]});
            else
              b.create<memref::StoreOp>(loc, out, outMems[j],
                                        ValueRange{bx, cols[k]});
            continue;
          }
          auto guarded = b.create<scf::IfOp>(
              loc, columnActive[k], /*withElseRegion=*/false);
          OpBuilder thenBuilder = guarded.getThenBodyBuilder();
          Value out = toStorage(
              thenBuilder, loc,
              emitAt(thenBuilder, loc, outputVals[j], k));
          if (oneDimensionalResults)
            thenBuilder.create<memref::StoreOp>(
                loc, out, outMems[j], ValueRange{cols[k]});
          else
            thenBuilder.create<memref::StoreOp>(
                loc, out, outMems[j], ValueRange{bx, cols[k]});
        }
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    attachElementwiseStateReceipt(
        gpuFn, T,
        containsTanh
            ? (requestTanhApproximation == "native-ptx" ? "native_tanh"
                                                         : "stable_tanh")
            : "none",
        oneDimensionalResults, hostValueCaptureCount);
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
      ElementwiseLowering probe(fn, blockThreads, flattenContiguous,
                                flatVectorWidth, promoteUniformScalars,
                                tanhApproximation);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ElementwiseLowering low(fn, blockThreads, flattenContiguous,
                              flatVectorWidth, promoteUniformScalars,
                              tanhApproximation);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("elementwise lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
