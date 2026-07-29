//===- ConvertIntentRowReductionToGPU.cpp - row reduce -> gpu.launch ------===//
//
// Lowers a "row normalization" intent func (a straight-line map/reduce graph
// whose reduces are all over the last axis, keepdims) to ONE read-once,
// block-cooperative `gpu.launch`: one block per row, the feature dim split over
// the block's threads, each thread caching its columns once. Independent row
// reductions share a grouped warp/shared hierarchy and one barrier; dependent
// reductions begin a new group. This is the verified path (research/
// 29-kernel-characterization.md + the rms-norm-pk probe); it is general over
// the DAG — there is NO per-kernel "recognize rms_norm" branch. rms_norm
// (one mean reduce) and softmax (max then sum) flow through identically.
//
// Schedule per block (row `bx`), per thread `tx` owning columns col = tx + k*T:
//   alloca cache_i : memref<Kxf32>   (one per row-shaped global input)
//   load uniforms (rank-0 scalar args) once
//   for each independent reduce group in program order:
//     accumulate every per-thread partial from the register tile
//     warp-shuffle each partial; publish all warp totals; ONE shared barrier
//     each warp redundantly folds the compact totals (mean: scalar /= N)
//     hoist every now-uniform map op once (per-row scalars: +eps, sqrt, recip…)
//   epilogue: for k=0..K: store <output evaluated at col> -> Y[bx,col]
//
// Read-once (cache, not a 2nd global load of X) is the whole game — a 2-global-
// pass version pays 1.5x traffic. Static shapes only (bare-ptr ABI). Emits
// math.exp/sqrt/rsqrt; the `-lower-gpu-math-to-nvvm` peephole turns those into
// NVVM approx intrinsics (libdevice externs are a ptxas dead end).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Schedule.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTROWREDUCTIONTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
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

/// True when an explicit two-input einsum has no contracted label.  These
/// live-extraction forms are broadcast/outer multiplication, not a reduction:
/// RMSNorm uses ``mn,m->mn`` and cross entropy uses ``i,j->ij`` to materialize
/// a singleton index column / row mask.
static bool isNoReductionContraction(ContractionOp op) {
  StringRef sub = op.getSubscripts();
  size_t arrow = sub.find("->");
  size_t comma = sub.find(',');
  if (arrow == StringRef::npos || comma == StringRef::npos || comma > arrow)
    return false;
  StringRef lhs = sub.substr(0, comma);
  StringRef rhs = sub.substr(comma + 1, arrow - comma - 1);
  StringRef out = sub.substr(arrow + 2);
  for (char label : lhs)
    if (!out.contains(label))
      return false;
  for (char label : rhs)
    if (!out.contains(label))
      return false;
  return true;
}

/// The scalar row emitter evaluates both contraction operands at the current
/// logical coordinate, so it can only consume no-reduction einsums whose output
/// labels preserve first-appearance order.  This covers broadcast/outer forms
/// such as `mn,m->mn`, `i,j->ij`, and `ng,k->ngk`, while rejecting silent
/// permutations such as `ij,->ji`.
static bool isCoordinatePreservingContraction(ContractionOp op) {
  if (!isNoReductionContraction(op))
    return false;
  StringRef sub = op.getSubscripts();
  size_t arrow = sub.find("->");
  size_t comma = sub.find(',');
  StringRef lhs = sub.substr(0, comma);
  StringRef rhs = sub.substr(comma + 1, arrow - comma - 1);
  StringRef out = sub.substr(arrow + 2);
  auto labelsUnique = [](StringRef labels) {
    bool seen[256] = {};
    for (unsigned char label : labels) {
      if (seen[label])
        return false;
      seen[label] = true;
    }
    return true;
  };
  if (!labelsUnique(lhs) || !labelsUnique(rhs) || !labelsUnique(out))
    return false;
  std::string firstAppearance;
  auto appendLabels = [&](StringRef labels) {
    for (char label : labels)
      if (firstAppearance.find(label) == std::string::npos)
        firstAppearance.push_back(label);
  };
  appendLabels(lhs);
  appendLabels(rhs);
  return firstAppearance == out;
}

/// The canonical emitter makes a non-trailing broadcast explicit with
/// tensor.expand_shape.  The row kernel already evaluates the source at its
/// logical row/column coordinate, so a pure unit-axis insertion is an index
/// view and needs no device instruction.  Reject dimension splitting/merging:
/// only the ordered non-unit extents may survive unchanged.
static bool isUnitAxisExpansion(tensor::ExpandShapeOp op) {
  auto source = dyn_cast<RankedTensorType>(op.getSrc().getType());
  auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!source || !result || !source.hasStaticShape() ||
      !result.hasStaticShape() ||
      source.getElementType() != result.getElementType())
    return false;
  SmallVector<int64_t> sourceNonUnit;
  SmallVector<int64_t> resultNonUnit;
  llvm::copy_if(source.getShape(), std::back_inserter(sourceNonUnit),
                [](int64_t extent) { return extent != 1; });
  llvm::copy_if(result.getShape(), std::back_inserter(resultNonUnit),
                [](int64_t extent) { return extent != 1; });
  return sourceNonUnit == resultNonUnit;
}

/// Prove that a rank-1 argument denotes the trailing feature axis from its
/// compiler-visible use graph. Intent map/select broadcasting is right-aligned:
/// a rank-1 value used by a [M,N] result is therefore Col[N], even when M == N
/// makes its extent shape-ambiguous. Rank-1 elementwise/cast chains are
/// transparent. Any indexing/contraction/side use or incompatible shape fails
/// closed.
static bool isTrailingColumnBroadcast(Value root, int64_t M, int64_t N) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> seen;
  bool reachesRowDomain = false;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    if (value.use_empty())
      return false;

    for (Operation *user : value.getUsers()) {
      Value result;
      if (auto cast = dyn_cast<CastOp>(user))
        result = cast.getResult();
      else if (auto unary = dyn_cast<MapUnaryOp>(user))
        result = unary.getResult();
      else if (auto binary = dyn_cast<MapBinaryOp>(user))
        result = binary.getResult();
      else if (auto select = dyn_cast<SelectOp>(user))
        result = select.getResult();
      else
        return false;

      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape())
        return false;
      if (type.getRank() == 1 && type.getDimSize(0) == N) {
        worklist.push_back(result);
        continue;
      }
      if (type.getRank() == 2 && type.getDimSize(0) == M &&
          type.getDimSize(1) == N) {
        reachesRowDomain = true;
        continue;
      }
      return false;
    }
  }
  return reachesRowDomain;
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

/// Predicate a pure tensor value conditional into ordinary dataflow.
///
/// Canonical J materializes `value_if` as an `scf.if` whose condition is a
/// rank-0 tensor extraction.  The structured control itself is part of M0, but
/// the row lowering consumes a straight-line tensor graph.  When both arms
/// contain only closed, side-effect-free tensor arithmetic, clone both arms and
/// select their yielded tensors with the original rank-0 condition.  This is a
/// general compiler transformation over typed MLIR; it neither consults a
/// kernel name nor specializes the runtime predicate in Python.
static unsigned predicatePureTensorValueIfs(func::FuncOp fn) {
  SmallVector<scf::IfOp> candidates;
  fn.walk([&](scf::IfOp ifOp) { candidates.push_back(ifOp); });

  unsigned transformed = 0;
  for (scf::IfOp ifOp : candidates) {
    if (ifOp->getParentOfType<func::FuncOp>() != fn ||
        ifOp.getNumResults() != 1 || ifOp.getElseRegion().empty() ||
        !ifOp.getThenRegion().hasOneBlock() ||
        !ifOp.getElseRegion().hasOneBlock())
      continue;
    auto extract = ifOp.getCondition().getDefiningOp<tensor::ExtractOp>();
    auto conditionType =
        extract ? dyn_cast<RankedTensorType>(extract.getTensor().getType())
                : RankedTensorType();
    if (!extract || !extract.getIndices().empty() || !conditionType ||
        conditionType.getRank() != 0 ||
        !conditionType.getElementType().isInteger(1))
      continue;

    auto armIsPure = [](Block &block) {
      for (Operation &operation : block.without_terminator()) {
        if (!isa<arith::ConstantOp, MapUnaryOp, MapBinaryOp, SelectOp, CastOp,
                 TransposeOp, tensor::ExpandShapeOp>(&operation))
          return false;
      }
      auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
      return yield && yield.getNumOperands() == 1;
    };
    Block &positiveBlock = ifOp.getThenRegion().front();
    Block &negativeBlock = ifOp.getElseRegion().front();
    if (!armIsPure(positiveBlock) || !armIsPure(negativeBlock))
      continue;

    OpBuilder builder(ifOp);
    auto cloneArm = [&](Block &block) {
      IRMapping mapping;
      for (Operation &operation : block.without_terminator())
        builder.clone(operation, mapping);
      auto yield = cast<scf::YieldOp>(block.getTerminator());
      return mapping.lookupOrDefault(yield.getOperand(0));
    };
    Value positive = cloneArm(positiveBlock);
    Value negative = cloneArm(negativeBlock);
    Value selected = builder.create<SelectOp>(
        ifOp.getLoc(), ifOp.getResult(0).getType(), extract.getTensor(),
        positive, negative);
    ifOp.getResult(0).replaceAllUsesWith(selected);
    ifOp.erase();
    if (extract->use_empty())
      extract.erase();
    ++transformed;
  }
  return transformed;
}

/// Fold rank-0 constant tensor selects and erase the newly dead pure chain.
/// These constants are emitted from J-authored constexpr bindings, so this is
/// ordinary typed constant propagation rather than a backend permission or a
/// source/kernel-name rule.
static unsigned foldRankZeroConstantSelects(func::FuncOp fn) {
  SmallVector<SelectOp> candidates;
  fn.walk([&](SelectOp select) { candidates.push_back(select); });
  unsigned folded = 0;
  for (SelectOp select : candidates) {
    auto constant = select.getCond().getDefiningOp<arith::ConstantOp>();
    auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                          : DenseElementsAttr();
    auto conditionType =
        dense ? dyn_cast<RankedTensorType>(dense.getType()) : RankedTensorType();
    if (!dense || !dense.isSplat() || !conditionType ||
        conditionType.getRank() != 0 ||
        !conditionType.getElementType().isInteger(1))
      continue;
    bool condition = dense.getSplatValue<APInt>().isOne();
    select.getResult().replaceAllUsesWith(condition ? select.getX()
                                                    : select.getY());
    select.erase();
    ++folded;
  }

  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    for (Operation &operation :
         llvm::reverse(fn.getBody().front().without_terminator())) {
      if (operation.getNumResults() == 0 ||
          llvm::any_of(operation.getResults(),
                       [](Value result) { return !result.use_empty(); }))
        continue;
      if (isa<arith::ConstantOp, MapUnaryOp, MapBinaryOp, SelectOp, CastOp,
              TransposeOp, tensor::ExpandShapeOp>(&operation))
        dead.push_back(&operation);
    }
    for (Operation *operation : dead) {
      operation->erase();
      changed = true;
    }
  }
  return folded;
}

/// Attach a machine-readable receipt to the concrete MLIR state produced by
/// this C++ lowering.  The outer search controller reads only these facts and
/// the resulting IR digest; it never infers a mechanism from a kernel name or
/// from the requested pass option.  In particular, an FMA request that matched
/// no concrete mul+add is recorded with fma_count=0.
static void attachConcreteStateReceipt(func::FuncOp lowered, int64_t threads,
                                       unsigned predicatedValueIfCount,
                                       unsigned foldedConstexprSelectCount,
                                       int64_t rowsPerBlock = 1,
                                       unsigned approximateDivCount = 0,
                                       unsigned hostValueCaptureCount = 0,
                                       unsigned epilogueLoadMotionCount = 0,
                                       bool materializedSharedRowOutput = false,
                                       bool elidedSharedRowMaterialization = false,
                                       StringRef topologyOverride = {},
                                       int64_t stageCount = 1,
                                       int64_t reductionParts = 1) {
  MLIRContext *ctx = lowered.getContext();
  unsigned fmaCount = 0;
  unsigned shuffleCount = 0;
  unsigned barrierCount = 0;
  unsigned allReduceCount = 0;
  lowered.walk([&](math::FmaOp) { ++fmaCount; });
  lowered.walk([&](gpu::ShuffleOp) { ++shuffleCount; });
  lowered.walk([&](gpu::BarrierOp) { ++barrierCount; });
  lowered.walk([&](gpu::AllReduceOp) { ++allReduceCount; });

  StringRef topology = "none";
  if (rowsPerBlock > 1 && shuffleCount != 0)
    topology = "subgroup_rows";
  else if (shuffleCount != 0)
    topology = threads <= 32 ? "subgroup_shuffle"
                             : "workgroup_hierarchical";
  else if (allReduceCount != 0)
    topology = "gpu_all_reduce";
  if (!topologyOverride.empty())
    topology = topologyOverride;

  SmallVector<Attribute> trace{
      StringAttr::get(ctx, "cuda.row_reduction.read_once.v1"),
      StringAttr::get(ctx, ("cuda.reduction." + topology + ".v1").str())};
  if (predicatedValueIfCount != 0)
    trace.push_back(
        StringAttr::get(ctx, "cuda.control.value_if_to_select.v1"));
  if (foldedConstexprSelectCount != 0)
    trace.push_back(
        StringAttr::get(ctx, "cuda.control.constexpr_select_fold.v1"));
  if (fmaCount != 0)
    trace.push_back(StringAttr::get(ctx, "cuda.numeric.fma.f32.v1"));
  if (approximateDivCount != 0)
    trace.push_back(
        StringAttr::get(ctx, "cuda.numeric.div_reciprocal_approx.f32.v1"));
  if (hostValueCaptureCount != 0)
    trace.push_back(
        StringAttr::get(ctx, "cuda.b.rank0_host_value_capture.v1"));
  if (epilogueLoadMotionCount != 0)
    trace.push_back(
        StringAttr::get(ctx, "cuda.b.epilogue_readonly_load_motion.v1"));
  if (materializedSharedRowOutput)
    trace.push_back(StringAttr::get(
        ctx, "cuda.materialize.intermediate_row_output.v1"));
  if (elidedSharedRowMaterialization)
    trace.push_back(StringAttr::get(
        ctx, "cuda.elide.intermediate_row_materialization.v1"));

  lowered->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(ctx, "row_reduction"));
  lowered->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(ctx, topology));
  lowered->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(ctx, trace));
  lowered->setAttr("intent.backend.fma_count",
                   IntegerAttr::get(IntegerType::get(ctx, 64), fmaCount));
  lowered->setAttr(
      "intent.backend.approximate_div_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), approximateDivCount));
  lowered->setAttr("intent.backend.shuffle_count",
                   IntegerAttr::get(IntegerType::get(ctx, 64), shuffleCount));
  lowered->setAttr("intent.backend.barrier_count",
                   IntegerAttr::get(IntegerType::get(ctx, 64), barrierCount));
  lowered->setAttr("intent.backend.all_reduce_count",
                   IntegerAttr::get(IntegerType::get(ctx, 64), allReduceCount));
  lowered->setAttr("intent.backend.block_threads",
                   IntegerAttr::get(IntegerType::get(ctx, 64),
                                    threads * rowsPerBlock));
  lowered->setAttr("intent.backend.rows_per_block",
                   IntegerAttr::get(IntegerType::get(ctx, 64), rowsPerBlock));
  lowered->setAttr("intent.backend.stage_count",
                   IntegerAttr::get(IntegerType::get(ctx, 64), stageCount));
  lowered->setAttr("intent.backend.reduction_parts",
                   IntegerAttr::get(IntegerType::get(ctx, 64), reductionParts));
  lowered->setAttr(
      "intent.backend.predicated_value_if_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), predicatedValueIfCount));
  lowered->setAttr(
      "intent.backend.folded_constexpr_select_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), foldedConstexprSelectCount));
  lowered->setAttr(
      "intent.backend.host_value_capture_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), hostValueCaptureCount));
  lowered->setAttr(
      "intent.backend.epilogue_load_motion_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), epilogueLoadMotionCount));
}

/// Match the common normalization scalar fast path `recip(sqrt(x))`.  The
/// single-use condition makes the replacement local: no other consumer loses
/// the explicitly materialized sqrt value.
static MapUnaryOp reciprocalSqrtOperand(MapUnaryOp reciprocal) {
  if (!reciprocal || reciprocal.getFn() != "recip")
    return nullptr;
  auto squareRoot = reciprocal.getInput().getDefiningOp<MapUnaryOp>();
  if (!squareRoot || squareRoot.getFn() != "sqrt" ||
      !squareRoot.getResult().hasOneUse())
    return nullptr;
  return squareRoot;
}

static bool isFusedReciprocalSqrtInput(MapUnaryOp squareRoot) {
  if (!squareRoot || squareRoot.getFn() != "sqrt" ||
      !squareRoot.getResult().hasOneUse())
    return false;
  auto reciprocal =
      dyn_cast<MapUnaryOp>(*squareRoot.getResult().getUsers().begin());
  return reciprocalSqrtOperand(reciprocal) == squareRoot;
}

/// Materialize one element of a tensor splat constant.  Paper extraction emits
/// scalar broadcast literals as rank-0 tensor arith.constant ops (rather than
/// adding synthetic scalar function arguments), so the structural fast path
/// must consume those constants directly.
static Value emitSplatScalarConstant(OpBuilder &b, Location loc,
                                     arith::ConstantOp cst) {
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return nullptr;
  auto shaped = dyn_cast<ShapedType>(dense.getType());
  if (!shaped)
    return nullptr;
  Type elemTy = shaped.getElementType();
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(floatTy, dense.getSplatValue<APFloat>()));
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(intTy, dense.getSplatValue<APInt>()));
  return nullptr;
}

/// A gather is legal only when bounds are mechanically established.
/// Scalar/tensor integer constants are checked directly; dynamic index graphs
/// require the typed Layer-B fact emitted by the validated JSON frontend.
static bool hasProvenGatherBounds(GatherOp gather) {
  auto dataTy = cast<RankedTensorType>(gather.getData().getType());
  if (!dataTy.hasStaticShape())
    return false;
  int64_t axis = gather.getAxis();
  if (axis < 0)
    axis += dataTy.getRank();
  if (axis < 0 || axis >= dataTy.getRank())
    return false;
  int64_t extent = dataTy.getDimSize(axis);
  if (auto cst = gather.getIndex().getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (dense && isa<IntegerType>(dense.getElementType())) {
      for (APInt value : dense.getValues<APInt>()) {
        if (value.isNegative() || value.getSExtValue() >= extent)
          return false;
      }
      return true;
    }
  }
  auto proven =
      gather->getAttrOfType<BoolAttr>("intent.index_bounds_proven");
  return proven && proven.getValue();
}

/// Form a memory-safe index before a dynamic gather address is constructed.
/// A validated Layer-B proof keeps the original index unchanged. Otherwise a
/// dynamic row gather is clamped to the static data extent; statically invalid
/// constants are rejected by the matcher and never reach this helper.
static Value emitSafeGatherIndex(OpBuilder &b, Location loc, GatherOp gather,
                                 Value index) {
  if (!index)
    return nullptr;
  if (!isa<IndexType>(index.getType()))
    index = b.create<arith::IndexCastOp>(loc, b.getIndexType(), index);
  if (hasProvenGatherBounds(gather))
    return index;
  auto dataTy = cast<RankedTensorType>(gather.getData().getType());
  int64_t axis = gather.getAxis();
  if (axis < 0)
    axis += dataTy.getRank();
  if (axis < 0 || axis >= dataTy.getRank())
    return nullptr;
  int64_t extent = dataTy.getDimSize(axis);
  if (extent <= 0)
    return nullptr;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value last = b.create<arith::ConstantIndexOp>(loc, extent - 1);
  Value below = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, index, zero);
  Value nonnegative = b.create<arith::SelectOp>(loc, below, zero, index);
  Value above = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, nonnegative, last);
  return b.create<arith::SelectOp>(loc, above, last, nonnegative);
}

/// Scalar form of intent.cast for the row fast path.  In particular, boolean
/// support masks cast to integers as 0/1 (zero extension), not 0/-1.
static Value emitRowScalarCast(OpBuilder &b, Location loc, Type dst, Value v) {
  Type src = v.getType();
  if (src == dst)
    return v;
  auto sf = dyn_cast<FloatType>(src);
  auto df = dyn_cast<FloatType>(dst);
  auto si = dyn_cast<IntegerType>(src);
  auto di = dyn_cast<IntegerType>(dst);
  if (dst.isInteger(1)) {
    if (si) {
      Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(si, 0));
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v, zero);
    }
    if (sf) {
      Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(sf, 0.0));
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v, zero);
    }
  }
  if (sf && df)
    return sf.getWidth() < df.getWidth()
               ? b.create<arith::ExtFOp>(loc, dst, v).getResult()
               : b.create<arith::TruncFOp>(loc, dst, v).getResult();
  if (sf && di)
    return b.create<arith::FPToSIOp>(loc, dst, v);
  if (si && df)
    return si.getWidth() == 1 ? b.create<arith::UIToFPOp>(loc, dst, v).getResult()
                              : b.create<arith::SIToFPOp>(loc, dst, v).getResult();
  if (si && di) {
    if (si.getWidth() < di.getWidth())
      return si.getWidth() == 1
                 ? b.create<arith::ExtUIOp>(loc, dst, v).getResult()
                 : b.create<arith::ExtSIOp>(loc, dst, v).getResult();
    return b.create<arith::TruncIOp>(loc, dst, v);
  }
  return nullptr;
}

/// Type-directed map_binary emission.  The shared MapEmit helper is the f32
/// path; live extraction also produces small integer subgraphs (iota + 1 and
/// max(support_count, 1)), which stay integer until their explicit intent.cast.
static Value emitRowScalarBinary(OpBuilder &b, Location loc, StringRef fn,
                                 Value a, Value c, Type f32) {
  if (!a || !c)
    return nullptr;
  if (a.getType() != c.getType()) {
    if (isa<IntegerType>(a.getType()) && isa<FloatType>(c.getType()))
      a = emitRowScalarCast(b, loc, c.getType(), a);
    else if (isa<FloatType>(a.getType()) && isa<IntegerType>(c.getType()))
      c = emitRowScalarCast(b, loc, a.getType(), c);
    else
      return nullptr;
  }
  if (isa<IntegerType>(a.getType())) {
    if (fn == "add")
      return b.create<arith::AddIOp>(loc, a, c);
    if (fn == "sub")
      return b.create<arith::SubIOp>(loc, a, c);
    if (fn == "mul")
      return b.create<arith::MulIOp>(loc, a, c);
    if (fn == "div")
      return b.create<arith::DivSIOp>(loc, a, c);
    if (fn == "max")
      return b.create<arith::MaxSIOp>(loc, a, c);
    if (fn == "min")
      return b.create<arith::MinSIOp>(loc, a, c);
    if (fn == "and" || fn == "logical_and")
      return b.create<arith::AndIOp>(loc, a, c);
    if (fn == "or" || fn == "logical_or")
      return b.create<arith::OrIOp>(loc, a, c);
    if (fn == "xor" || fn == "logical_xor")
      return b.create<arith::XOrIOp>(loc, a, c);
    return nullptr;
  }
  (void)f32;
  return emitScalarBinary(b, loc, fn, a, c);
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
enum class ResultRole { RowOut, PerRowOut, PerRowVectorOut, AtomicOut };

static bool isHalfStorageType(Type type) {
  return isa<Float16Type, BFloat16Type>(type);
}

/// Recognize only the canonical, numerically-stable mixed-precision softmax
/// graph.  This is intentionally narrower than the general row-reduction
/// matcher: accepting one exact graph lets the lowering use f16/bf16 for
/// storage while keeping every arithmetic and collective value in f32 without
/// accidentally changing the semantics of other half-typed reductions.
static Type exactHalfSoftmaxStorageType(func::FuncOp fn) {
  auto fty = fn.getFunctionType();
  if (fty.getNumInputs() != 1 || fty.getNumResults() != 1 ||
      fn->hasAttr("intent.inplace") || !fn.getBody().hasOneBlock())
    return {};

  auto inputTy = dyn_cast<RankedTensorType>(fty.getInput(0));
  auto outputTy = dyn_cast<RankedTensorType>(fty.getResult(0));
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape() || inputTy.getRank() != 2 ||
      outputTy.getRank() != 2 || inputTy.getShape() != outputTy.getShape() ||
      inputTy.getDimSize(0) <= 0 || inputTy.getDimSize(1) <= 0 ||
      inputTy.getElementType() != outputTy.getElementType() ||
      !isHalfStorageType(inputTy.getElementType()))
    return {};

  Block &body = fn.getBody().front();
  auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!ret || ret.getNumOperands() != 1 ||
      std::distance(body.begin(), body.end()) != 6)
    return {};

  auto div = ret.getOperand(0).getDefiningOp<MapBinaryOp>();
  if (!div || div.getFn() != "div")
    return {};
  auto exponent = div.getLhs().getDefiningOp<MapUnaryOp>();
  auto sum = div.getRhs().getDefiningOp<ReduceOp>();
  if (!exponent || exponent.getFn() != "exp" || !sum ||
      sum.getFn() != "sum" || sum.getInput() != exponent.getResult() ||
      sum.getAssoc() != "float_nondeterministic")
    return {};
  auto subtract = exponent.getInput().getDefiningOp<MapBinaryOp>();
  if (!subtract || subtract.getFn() != "sub" ||
      subtract.getLhs() != fn.getArgument(0))
    return {};
  auto maximum = subtract.getRhs().getDefiningOp<ReduceOp>();
  if (!maximum || maximum.getFn() != "max" ||
      maximum.getInput() != fn.getArgument(0) ||
      maximum.getAssoc() != "associative")
    return {};

  auto isAxisOne = [](ReduceOp reduce) {
    if (reduce.getAxes().size() != 1)
      return false;
    auto axis = dyn_cast<IntegerAttr>(reduce.getAxes()[0]);
    return axis && axis.getInt() == 1;
  };
  if (!isAxisOne(maximum) || !isAxisOne(sum))
    return {};

  Type storageType = inputTy.getElementType();
  int64_t M = inputTy.getDimSize(0);
  int64_t N = inputTy.getDimSize(1);
  auto hasTensorType = [&](Value value, ArrayRef<int64_t> shape) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.hasStaticShape() && type.getShape() == shape &&
           type.getElementType() == storageType;
  };
  if (!hasTensorType(maximum.getResult(), {M, 1}) ||
      !hasTensorType(subtract.getResult(), {M, N}) ||
      !hasTensorType(exponent.getResult(), {M, N}) ||
      !hasTensorType(sum.getResult(), {M, 1}) ||
      !hasTensorType(div.getResult(), {M, N}))
    return {};

  auto hasExactUsers = [](Value value, ArrayRef<Operation *> expected) {
    SmallVector<Operation *> actual(value.getUsers().begin(),
                                    value.getUsers().end());
    if (actual.size() != expected.size())
      return false;
    return llvm::all_of(expected, [&](Operation *operation) {
      return llvm::is_contained(actual, operation);
    });
  };
  if (!hasExactUsers(fn.getArgument(0),
                     {maximum.getOperation(), subtract.getOperation()}) ||
      !hasExactUsers(maximum.getResult(), {subtract.getOperation()}) ||
      !hasExactUsers(subtract.getResult(), {exponent.getOperation()}) ||
      !hasExactUsers(exponent.getResult(),
                     {sum.getOperation(), div.getOperation()}) ||
      !hasExactUsers(sum.getResult(), {div.getOperation()}) ||
      !hasExactUsers(div.getResult(), {ret.getOperation()}))
    return {};

  return storageType;
}

/// A mixed-precision row kernel may expose half storage at its ABI only when
/// the Intent graph makes the f32 compute boundary explicit.  Requiring every
/// use of the argument to be a shape-preserving half->f32 intent.cast prevents
/// an unrelated half map/reduce graph from being silently promoted.
static bool explicitlyPromotesHalfArgument(func::FuncOp fn, unsigned index) {
  auto inputTy = dyn_cast<RankedTensorType>(fn.getFunctionType().getInput(index));
  if (!inputTy || !inputTy.hasStaticShape() ||
      !isHalfStorageType(inputTy.getElementType()) ||
      !fn.getBody().hasOneBlock())
    return false;
  Value argument = fn.getArgument(index);
  bool sawCast = false;
  for (Operation *user : argument.getUsers()) {
    auto castOp = dyn_cast<CastOp>(user);
    auto resultTy = castOp
                        ? dyn_cast<RankedTensorType>(castOp.getResult().getType())
                        : RankedTensorType();
    if (!castOp || castOp.getInput() != argument || !resultTy ||
        !resultTy.hasStaticShape() ||
        resultTy.getShape() != inputTy.getShape() ||
        !isa<Float32Type>(resultTy.getElementType()))
      return false;
    sawCast = true;
  }
  return sawCast;
}

/// Symmetric ABI proof for a half result: the returned value must be a single
/// shape-preserving f32->half intent.cast.  Statistics may remain f32 in other
/// results (LayerNorm mean/rstd), but no implicit output demotion is inferred.
static bool explicitlyDemotesHalfResult(func::FuncOp fn, unsigned index) {
  auto outputTy =
      dyn_cast<RankedTensorType>(fn.getFunctionType().getResult(index));
  if (!outputTy || !outputTy.hasStaticShape() || outputTy.getRank() != 2 ||
      !isHalfStorageType(outputTy.getElementType()) ||
      !fn.getBody().hasOneBlock() || fn->hasAttr("intent.inplace"))
    return false;
  auto ret = dyn_cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  if (!ret || index >= ret.getNumOperands())
    return false;
  auto castOp = ret.getOperand(index).getDefiningOp<CastOp>();
  auto inputTy = castOp
                     ? dyn_cast<RankedTensorType>(castOp.getInput().getType())
                     : RankedTensorType();
  return castOp && castOp.getResult().hasOneUse() && inputTy &&
         inputTy.hasStaticShape() && inputTy.getShape() == outputTy.getShape() &&
         isa<Float32Type>(inputTy.getElementType());
}

/// One func's row-reduction lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct RowReductionLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  unsigned tileColsMax;
  unsigned requestedVectorWidth;
  bool enableFmaFusion;
  bool enableApproximateDiv;
  bool deferEpilogueColumnLoads;
  bool materializeSharedRowOutput;
  bool promoteRank0ScalarInputs;
  bool enableSortedPrefix;
  unsigned requestedRowsPerBlock;
  unsigned requestedReductionParts;
  unsigned predicatedValueIfCount;
  unsigned foldedConstexprSelectCount;
  unsigned emittedApproximateDivCount = 0;

  // populated by match()
  int64_t M = 0, N = 0, T = 0, K = 0;
  int64_t groupedBatch = 0, groupedGroups = 0;
  // Flash backward preprocessing preserves the source ABI as
  // [1,M,1,N] -> [1,1,M].  With statically singleton batch/head axes these
  // tensors are physically identical to one contiguous [M,N] row domain and
  // one [M] statistic output.  The matcher proves that exact shape/transpose
  // relationship before enabling this view; no materialized reshape or
  // transpose is inserted.
  bool flatSingletonRank4 = false;
  // A source schedule may launch more feature threads than the static row
  // extent (Flash uses 4 warps for D=64).  Such lanes participate in the block
  // collective with the reduction identity while loading a safe in-bounds
  // address.  This mode is restricted to statistic-only outputs.
  bool hasInactiveFeatureThreads = false;
  // Ragged final register tile for any supported storage type. Every invalid
  // per-thread column contributes the reduction identity and its output store
  // is suppressed; all threads still execute every collective.
  bool hasFeatureTail = false;
  // Tiling: the per-thread feature strip of K columns is split into `numTiles`
  // tiles of `Kpt` columns each (K == Kpt * numTiles). numTiles==1 (K <=
  // tileColsMax) is the M1 register-tile path, emitted byte-identically.
  int64_t Kpt = 0, numTiles = 1;
  // float4 vectorization (N2): when the register-tile path has K divisible by 4
  // and only Row/Col/Scalar args (no ColSeg gather, no iota/select), the K
  // columns of each thread are grouped into K/4 CONTIGUOUS-vec4 tiles so the Row
  // and ordinary contiguous Col loads plus RowOut stores lower to
  // `ld.global.v4.f32`/`st.global.v4.f32` (4x fewer memory instructions — the
  // memory-read bulk of these HBM-bound kernels). ColSeg/indexed gathers stay on
  // the scalar path because the affine channel map need not be contiguous. The
  // per-thread column layout becomes col = vt*(T*4) + tx*4 + j (vec index tx+vt*T
  // in the retyped memref<Mx(N/4)xvector<4xf32>>), staying fully coalesced.
  bool useVec4 = false;
  // Mixed-precision register kernels use four contiguous 16-bit storage lanes.
  // Arithmetic remains scalar f32 after extraction/promotion; only global
  // Row/Col loads and RowOut stores use vector<4xhalf> packets.
  bool useHalfVec4 = false;
  Type halfVectorStorageType;
  int64_t numVecTiles = 0;
  Type f32;                         // compute type
  Type storageType;                 // f32 generally; f16/bf16 for exact softmax
  bool halfSoftmax = false;
  bool explicitHalfStorage = false;
  Type rowIntTy;                     // element type of a RowInt [M] arg (Y), if any
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<ReduceOp> reduces;     // per-row (axis-1) reduces, program order
  SmallVector<ReduceOp> crossReduces; // cross-row (axis-0 sum) reduces -> AtomicOut
  SmallVector<ScanOp> scans;         // per-row (axis-1) inclusive cumsum scans
  SmallVector<Value> outputVals;     // the returned values (>=1, topo order)
  SmallVector<int64_t> inplaceArg;   // result -> aliased input, or -1
  SmallVector<ResultRole> outRoles;  // RowOut / PerRowOut / AtomicOut per result
  SmallVector<int64_t> outVectorWidths; // 0 except PerRowVectorOut [M,P]
  // ColSeg (group_norm W/B) state, indexed by func-arg position. colSegLen[i] is
  // the C extent of the [C] memref; colSegMap[i] is the affine (group,col)->chan
  // index map. numGroups (G) is recovered from C/CPG so group = bx % G.
  SmallVector<int64_t> colSegLen;
  SmallVector<AffineMap> colSegMap;
  int64_t numGroups = 0;
  // Receipt-bound Layer-B equalities materialized by the JSON emitter as
  // (argument-index, signed-value) pairs. The specialized kernel may constant
  // fold these rank-0 integer inputs; the automatic selection artifact records
  // the same contract and keeps the generic path as the mismatch fallback.
  DenseMap<unsigned, int64_t> scalarEqualities;
  Value groupVal;                    // bx % G (per-block uniform), set in rewrite
  Value rowIndex;                    // bx, used by per-row inputs/gathers
  enum class IotaRole { Column, Group, OutputVector };
  DenseMap<Value, IotaRole> iotaRoles;

  // M6: the first-class Schedule this lowering emits. Populated by match();
  // `kind` carries the four-way fork decision (0=register-tile, 1=tiled-smem,
  // 2=scan, 3=multi-stage partial/finalize/epilogue) so rewrite() dispatches on
  // the Schedule, not on ad-hoc predicates.
  // Each rewrite path fills `smemBytes` and attaches the structured launch plan.
  enum RowKind { KRegister = 0, KTiled = 1, KScan = 2, KMultiStage = 3 };
  Schedule sched;

  RowReductionLowering(func::FuncOp fn, unsigned bt, unsigned tc,
                       unsigned vw, bool fuseFma, bool approximateDiv,
                       bool deferColLoads, bool materializeRowOutput,
                       bool promoteScalars, bool enablePrefix,
                       unsigned predicatedValueIfs,
                       unsigned foldedConstexprSelects,
                       unsigned rowsPerBlock, unsigned reductionParts)
      : fn(fn), blockThreads(bt), tileColsMax(tc),
        requestedVectorWidth(vw),
        enableFmaFusion(fuseFma),
        enableApproximateDiv(approximateDiv),
        deferEpilogueColumnLoads(deferColLoads),
        materializeSharedRowOutput(materializeRowOutput),
        promoteRank0ScalarInputs(promoteScalars),
        enableSortedPrefix(enablePrefix),
        requestedRowsPerBlock(rowsPerBlock),
        requestedReductionParts(reductionParts),
        predicatedValueIfCount(predicatedValueIfs),
        foldedConstexprSelectCount(foldedConstexprSelects) {}

  /// Finalize a static f32 mean.  The exact state retains divf.  The optional
  /// precision-search state materializes the rounded f32 reciprocal as an
  /// ordinary arith.constant and multiplies, removing NVPTX's precise division
  /// correction/slowpath while leaving the candidate visible in post-pass MLIR.
  Value finalizeMean(OpBuilder &b, Location loc, Value sum, Value divisor) {
    if (!enableApproximateDiv)
      return b.create<arith::DivFOp>(loc, sum, divisor);
    ++emittedApproximateDivCount;
    Value reciprocal = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(1.0 / static_cast<double>(N)));
    return b.create<arith::MulFOp>(loc, sum, reciprocal);
  }

  /// Prove that an authored scalar tensor is the static trailing extent N.
  /// This consumes only the concrete tensor constant or the receipt-bound
  /// scalar equality already admitted by match(); runtime sample values never
  /// participate. Casts and a select whose two arms prove the same extent are
  /// transparent.
  bool isStaticFeatureExtent(Value value) const {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto found = scalarEqualities.find(argument.getArgNumber());
      return found != scalarEqualities.end() && found->second == N;
    }
    if (auto castOp = value.getDefiningOp<CastOp>())
      return isStaticFeatureExtent(castOp.getInput());
    if (auto select = value.getDefiningOp<SelectOp>())
      return isStaticFeatureExtent(select.getX()) &&
             isStaticFeatureExtent(select.getY());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat())
        return false;
      if (isa<IntegerType>(dense.getElementType()))
        return dense.getSplatValue<APInt>().getSExtValue() == N;
      if (isa<FloatType>(dense.getElementType()))
        return dense.getSplatValue<APFloat>().convertToDouble() ==
               static_cast<double>(N);
    }
    return false;
  }

  Value emitMappedBinary(OpBuilder &b, Location loc, MapBinaryOp op, Value lhs,
                         Value rhs, Type f32) {
    if (enableApproximateDiv && op.getFn() == "div" && lhs && rhs &&
        lhs.getType() == f32 && rhs.getType() == f32) {
      if (isStaticFeatureExtent(op.getRhs())) {
        ++emittedApproximateDivCount;
        Value reciprocal = b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(1.0 / static_cast<double>(N)));
        return b.create<arith::MulFOp>(loc, lhs, reciprocal);
      }
      // SSA identity is a sufficient proof that unrolled lanes share one
      // divisor. Materialize one explicitly searched reciprocal for each
      // distinct emitted rhs; lane-varying divisors have distinct SSA values
      // and therefore retain one reciprocal each.
      Value reciprocal = approximateReciprocals.lookup(rhs);
      if (!reciprocal) {
        Value one = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0));
        auto division = b.create<arith::DivFOp>(loc, one, rhs);
        division->setAttr("intent.cuda.approximate_div", b.getUnitAttr());
        reciprocal = division;
        approximateReciprocals[rhs] = reciprocal;
        ++emittedApproximateDivCount;
      }
      return b.create<arith::MulFOp>(loc, lhs, reciprocal);
    }
    return emitRowScalarBinary(b, loc, op.getFn(), lhs, rhs, f32);
  }

  /// A live LayerNorm extraction may express row reductions by transposing
  /// [M,N] to [N,M], reducing axis 0, and transposing the normalized value
  /// back.  In the one-block-per-original-row lowering this is an index view,
  /// not a materialized data movement.  Accept only the exact two-dimensional
  /// swap over the already anchored M/N domain.
  bool isRowDomainTranspose(TransposeOp op) {
    // A square swap has the same input/result type, so shape alone cannot
    // establish whether it is the supported reduction-domain view or a real
    // output transpose.  Fail closed until the paired view is proven.
    if (M == N)
      return false;
    if (!op || op.getPerm().size() != 2 ||
        cast<IntegerAttr>(op.getPerm()[0]).getInt() != 1 ||
        cast<IntegerAttr>(op.getPerm()[1]).getInt() != 0)
      return false;
    auto input = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!input || !result || !input.hasStaticShape() ||
        !result.hasStaticShape() || input.getElementType() != f32 ||
        result.getElementType() != f32)
      return false;
    bool forward = input.getShape() == ArrayRef<int64_t>({M, N}) &&
                   result.getShape() == ArrayRef<int64_t>({N, M});
    bool backward = input.getShape() == ArrayRef<int64_t>({N, M}) &&
                    result.getShape() == ArrayRef<int64_t>({M, N});
    return forward || backward;
  }

  /// Prove the terminal singleton-domain view used by Flash backward
  /// preprocessing: [1,M,1] --perm[0,2,1]--> [1,1,M].  Both singleton axes
  /// make this a physical no-op, but accepting it is safe only when it is an
  /// externally returned value of the already-proven rank-4 row domain.
  bool isFlatSingletonOutputTranspose(TransposeOp op) {
    if (!flatSingletonRank4 || !op || op.getPerm().size() != 3 ||
        cast<IntegerAttr>(op.getPerm()[0]).getInt() != 0 ||
        cast<IntegerAttr>(op.getPerm()[1]).getInt() != 2 ||
        cast<IntegerAttr>(op.getPerm()[2]).getInt() != 1 ||
        !llvm::is_contained(outputVals, op.getResult()))
      return false;
    auto input = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto result = dyn_cast<RankedTensorType>(op.getResult().getType());
    return input && result && input.hasStaticShape() &&
           result.hasStaticShape() && input.getElementType() == f32 &&
           result.getElementType() == f32 &&
           input.getShape() == ArrayRef<int64_t>({1, M, 1}) &&
           result.getShape() == ArrayRef<int64_t>({1, 1, M});
  }

  /// Prove a rank-1 coordinate's logical axis from its terminal einsum label.
  /// Pointwise scalar transforms may sit between the iota and that
  /// contraction.  A right-aligned map/select broadcast into [M,N] is an
  /// equally strong proof of the trailing column axis.  This disambiguates
  /// equal M/N and B/G/N extents without relying on a kernel name.
  bool isProvablyCoordinate(Value value, IotaRole role) {
    if (role == IotaRole::Column && isTrailingColumnBroadcast(value, M, N))
      return true;
    if (value.use_empty())
      return false;
    for (Operation *user : value.getUsers()) {
      Value next;
      if (auto unary = dyn_cast<MapUnaryOp>(user)) {
        if (unary.getInput() != value)
          return false;
        next = unary.getResult();
      } else if (auto castOp = dyn_cast<CastOp>(user)) {
        if (castOp.getInput() != value)
          return false;
        next = castOp.getResult();
      } else if (auto binary = dyn_cast<MapBinaryOp>(user)) {
        if (binary.getLhs() != value && binary.getRhs() != value)
          return false;
        next = binary.getResult();
      } else if (auto contraction = dyn_cast<ContractionOp>(user)) {
        StringRef sub = contraction.getSubscripts();
        size_t arrow = sub.find("->");
        size_t comma = sub.find(',');
        if (arrow == StringRef::npos || comma == StringRef::npos ||
            comma > arrow)
          return false;
        StringRef labels = contraction.getA() == value
                               ? sub.substr(0, comma)
                               : contraction.getB() == value
                                     ? sub.substr(comma + 1,
                                                  arrow - comma - 1)
                                     : StringRef();
        StringRef out = sub.substr(arrow + 2);
        auto resultTy = dyn_cast<RankedTensorType>(
            contraction.getResult().getType());
        if (labels.size() != 1 || out.empty() || !resultTy ||
            static_cast<int64_t>(out.size()) != resultTy.getRank())
          return false;
        size_t position = out.find(labels.front());
        if (position == StringRef::npos ||
            out.find(labels.front(), position + 1) != StringRef::npos)
          return false;
        if (role == IotaRole::Column) {
          if (position + 1 != out.size() ||
              resultTy.getDimSize(position) != N)
            return false;
        } else {
          if (groupedGroups <= 0 || resultTy.getDimSize(position) != groupedGroups)
            return false;
          bool rank2Group =
              resultTy.getRank() == 2 && position == 0 &&
              resultTy.getShape() == ArrayRef<int64_t>({groupedGroups, N});
          bool rank3Group =
              resultTy.getRank() == 3 && position == 1 &&
              resultTy.getShape() ==
                  ArrayRef<int64_t>({groupedBatch, groupedGroups, N});
          if (!rank2Group && !rank3Group)
            return false;
        }
        continue;
      } else {
        return false;
      }
      auto nextTy = dyn_cast<RankedTensorType>(next.getType());
      if (!nextTy || nextTy.getRank() != 1 ||
          !isProvablyCoordinate(next, role))
        return false;
    }
    return true;
  }

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
    if (auto cst = dyn_cast<arith::ConstantOp>(d))
      return isa<DenseElementsAttr>(cst.getValue()) &&
             cast<DenseElementsAttr>(cst.getValue()).isSplat();
    if (auto castOp = dyn_cast<CastOp>(d))
      return isStaticUniform(castOp.getInput());
    if (auto mu = dyn_cast<MapUnaryOp>(d))
      return isStaticUniform(mu.getInput());
    if (auto mb = dyn_cast<MapBinaryOp>(d))
      return isStaticUniform(mb.getLhs()) && isStaticUniform(mb.getRhs());
    if (auto sel = dyn_cast<SelectOp>(d))
      return isStaticUniform(sel.getCond()) && isStaticUniform(sel.getX()) &&
             isStaticUniform(sel.getY());
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
    if (requestedRowsPerBlock != 1 && requestedRowsPerBlock != 2 &&
        requestedRowsPerBlock != 4 && requestedRowsPerBlock != 8)
      return failure();
    if (requestedReductionParts == 0 || requestedReductionParts > 128 ||
        (requestedReductionParts & (requestedReductionParts - 1)) != 0 ||
        (requestedReductionParts > 1 && requestedRowsPerBlock != 1))
      return failure();
    if (requestedVectorWidth != 0 && requestedVectorWidth != 1 &&
        requestedVectorWidth != 4)
      return failure();
    f32 = Float32Type::get(fn.getContext());
    storageType = f32;
    if (Type halfStorage = exactHalfSoftmaxStorageType(fn)) {
      halfSoftmax = true;
      storageType = halfStorage;
    }
    auto fty = fn.getFunctionType();
    unsigned nRes = fty.getNumResults();
    if (nRes < 1)
      return failure();
    scalarEqualities.clear();
    if (auto equalities =
            fn->getAttrOfType<DenseI64ArrayAttr>("intent.scalar_equalities")) {
      ArrayRef<int64_t> values = equalities.asArrayRef();
      if (values.size() % 2 != 0)
        return failure();
      for (size_t pair = 0; pair < values.size(); pair += 2) {
        int64_t rawIndex = values[pair];
        int64_t value = values[pair + 1];
        if (rawIndex < 0 || rawIndex >= fty.getNumInputs())
          return failure();
        unsigned index = static_cast<unsigned>(rawIndex);
        auto tensorTy = dyn_cast<RankedTensorType>(fty.getInput(index));
        auto integerTy =
            tensorTy ? dyn_cast<IntegerType>(tensorTy.getElementType()) : nullptr;
        if (!tensorTy || tensorTy.getRank() != 0 || !integerTy ||
            integerTy.getWidth() > 64)
          return failure();
        APInt represented(integerTy.getWidth(), static_cast<uint64_t>(value),
                          /*isSigned=*/true);
        if (represented.getSExtValue() != value)
          return failure();
        auto [it, inserted] = scalarEqualities.try_emplace(index, value);
        if (!inserted && it->second != value)
          return failure();
      }
    }
    // Results are normally static f32.  The one exception is the exact
    // f16/bf16 softmax graph proven above: its public storage type remains half
    // while this lowering's compute type is still f32.  A RowOut is rank-2
    // [M,N] (the full feature row); a per-row scalar output (Mean/RSTD, or
    // flash's delta) is rank-2 [M,1] OR rank-1 [M]. Anchor M from every result's
    // leading dim and N from a RowOut if any.
    int64_t anchorM = -1, anchorN = -1;
    // First recognize the exact singleton rank-4 ABI before the general
    // rank-3 grouped-output logic.  Otherwise [1,1,M] would be misclassified as
    // [batch=1, group=1, hidden=M].  Type shapes only establish the candidate;
    // the body walk below additionally proves reduce(axis=3) and the terminal
    // [0,2,1] output view.
    int64_t rank4M = -1, rank4N = -1;
    bool rank4InputCandidate = false;
    for (Type at : fty.getInputs()) {
      auto t = dyn_cast<RankedTensorType>(at);
      if (!t || !t.hasStaticShape() || t.getRank() != 4 ||
          !isa<FloatType>(t.getElementType()) || t.getDimSize(0) != 1 ||
          t.getDimSize(2) != 1 || t.getDimSize(1) <= 0 ||
          t.getDimSize(3) <= 0)
        continue;
      if (!rank4InputCandidate) {
        rank4M = t.getDimSize(1);
        rank4N = t.getDimSize(3);
        rank4InputCandidate = true;
      } else if (rank4M != t.getDimSize(1) || rank4N != t.getDimSize(3)) {
        rank4InputCandidate = false;
        break;
      }
    }
    bool rank4ResultCandidate = rank4InputCandidate;
    bool sawRank4Result = false;
    if (rank4ResultCandidate) {
      for (Type rt : fty.getResults()) {
        auto t = dyn_cast<RankedTensorType>(rt);
        if (!t || !t.hasStaticShape() ||
            !isa<Float32Type>(t.getElementType())) {
          rank4ResultCandidate = false;
          break;
        }
        if (t.getRank() == 0)
          continue;
        if (t.getRank() != 3 ||
            t.getShape() != ArrayRef<int64_t>({1, 1, rank4M})) {
          rank4ResultCandidate = false;
          break;
        }
        sawRank4Result = true;
      }
    }
    flatSingletonRank4 = rank4ResultCandidate && sawRank4Result;
    if (flatSingletonRank4) {
      anchorM = rank4M;
      anchorN = rank4N;
    }
    // A grouped row graph may preserve its source [batch,group,hidden]
    // boundary.  Internally the one-block-per-row kernel flattens the first two
    // contiguous dimensions to M=batch*group while retaining the original ABI
    // pointer and exact [batch,group] statistic outputs.
    for (unsigned resultIndex = 0; resultIndex < fty.getNumResults();
         ++resultIndex) {
      if (flatSingletonRank4)
        break;
      Type rt = fty.getResult(resultIndex);
      auto t = dyn_cast<RankedTensorType>(rt);
      if (!t || !t.hasStaticShape())
        return failure();
      bool explicitHalfResult =
          explicitlyDemotesHalfResult(fn, resultIndex);
      if (t.getElementType() != f32 &&
          !(halfSoftmax && t.getElementType() == storageType) &&
          !explicitHalfResult)
        return failure();
      explicitHalfStorage |= explicitHalfResult;
      if (t.getRank() != 3)
        continue;
      if (groupedBatch == 0) {
        groupedBatch = t.getDimSize(0);
        groupedGroups = t.getDimSize(1);
        anchorN = t.getDimSize(2);
      } else if (t.getShape() !=
                 ArrayRef<int64_t>({groupedBatch, groupedGroups, anchorN})) {
        return failure();
      }
    }
    if (groupedBatch > 0) {
      anchorM = groupedBatch * groupedGroups;
      numGroups = groupedGroups;
    }
    for (Type rt : fty.getResults()) {
      auto t = cast<RankedTensorType>(rt);
      int64_t rank = t.getRank();
      if (rank > 3)
        return failure();
      if (rank == 0)
        continue; // a rank-0 AtomicOut (cross-row scalar): does not anchor M/N
      if (flatSingletonRank4) {
        if (rank != 3 ||
            t.getShape() != ArrayRef<int64_t>({1, 1, anchorM}))
          return failure();
        continue;
      }
      int64_t m = t.getDimSize(0);
      if (groupedBatch > 0) {
        if (rank == 3) {
          if (t.getShape() !=
              ArrayRef<int64_t>({groupedBatch, groupedGroups, anchorN}))
            return failure();
          continue;
        }
        if (rank == 2 && t.getDimSize(0) == groupedBatch &&
            t.getDimSize(1) == groupedGroups)
          continue;
        if (rank == 1 && t.getDimSize(0) == anchorM)
          continue;
        return failure();
      }
      if (anchorM < 0)
        anchorM = m;
      else if (m != anchorM)
        return failure();
      if (rank == 2 && t.getDimSize(1) != 1) { // a RowOut
        anchorN = std::max(anchorN, t.getDimSize(1));
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
    outVectorWidths.clear();
    for (Type rt : fty.getResults()) {
      auto t = cast<RankedTensorType>(rt);
      int64_t vectorWidth = 0;
      if (t.getRank() == 0) {
        outRoles.push_back(ResultRole::AtomicOut);
      } else if (flatSingletonRank4) {
        outRoles.push_back(ResultRole::PerRowOut);
      } else if (groupedBatch > 0 && t.getRank() == 3) {
        outRoles.push_back(ResultRole::RowOut);
      } else if (groupedBatch > 0) {
        outRoles.push_back(ResultRole::PerRowOut);
      } else if (t.getRank() == 2 && t.getDimSize(1) == N) {
        outRoles.push_back(ResultRole::RowOut);
      } else if (t.getRank() == 2 && t.getDimSize(1) > 1) {
        vectorWidth = t.getDimSize(1);
        if (vectorWidth > 16)
          return failure();
        outRoles.push_back(ResultRole::PerRowVectorOut);
      } else { // rank-2 [M,1] or rank-1 [M]
        outRoles.push_back(ResultRole::PerRowOut);
      }
      outVectorWidths.push_back(vectorWidth);
    }

    // Single basic block, terminated by a return of the output values.
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != nRes)
      return failure();
    outputVals.assign(ret.getOperands().begin(), ret.getOperands().end());

    // Explicit Layer-B destination aliases.  The typed attribute is emitted
    // only for validated single-writer conditional writebacks.  An aliased
    // result reuses the corresponding input memref instead of adding a fresh
    // destination argument, matching the source raw-kernel ABI.
    inplaceArg.assign(nRes, -1);
    if (auto inplace =
            fn->getAttrOfType<DenseI64ArrayAttr>("intent.inplace")) {
      if ((unsigned)inplace.size() != nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j) {
        int64_t arg = inplace[j];
        if (arg < -1 || arg >= (int64_t)fty.getNumInputs())
          return failure();
        if (arg >= 0 && fty.getInput(arg) != fty.getResult(j))
          return failure();
        inplaceArg[j] = arg;
      }
    }
    for (const auto &equality : scalarEqualities) {
      if (llvm::is_contained(inplaceArg, static_cast<int64_t>(equality.first)))
        return failure();
    }

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
      // Preserve required-but-unread ABI carriers exactly.  Their lack of uses
      // is compiler-visible after J constexpr control has been specialized;
      // they remain function arguments but generate no target loads.
      if (fn.getArgument(ai).use_empty()) {
        argRoles.push_back(Role::Unused);
        continue;
      }
      bool explicitHalfArgument = explicitlyPromotesHalfArgument(fn, ai);
      explicitHalfStorage |= explicitHalfArgument;
      // ColSeg (group_norm W/B): a 1-D f32 arg carrying an affine column-index
      // map. The arg-attr drives the gather (general, no per-kernel branch); the
      // [C] extent need not equal N. CPG = map(1,col)-map(0,col) (the group
      // coefficient); G = C / CPG so group = bx % G.
      if (auto mapAttr = fn.getArgAttrOfType<AffineMapAttr>(
              ai, "intent.col_index_map")) {
        AffineMap m = mapAttr.getAffineMap();
        if (s.size() != 1 || m.getNumDims() != 2 || m.getNumSymbols() != 0 ||
            m.getNumResults() != 1 || t.getElementType() != f32)
          return failure();
        int64_t cpg = evalAffineExprConst(m.getResult(0), {1, 0}) -
                      evalAffineExprConst(m.getResult(0), {0, 0});
        if (cpg <= 0 || s[0] % cpg != 0)
          return failure();
        int64_t g = s[0] / cpg;
        if (g <= 0 || N <= 0 || g > 1000000 / N)
          return failure();
        for (int64_t group = 0; group < g; ++group)
          for (int64_t col = 0; col < N; ++col) {
            int64_t channel =
                evalAffineExprConst(m.getResult(0), {group, col});
            if (channel < 0 || channel >= s[0])
              return failure();
          }
        if (numGroups == 0)
          numGroups = g;
        else if (numGroups != g)
          return failure();
        colSegLen[ai] = s[0];
        colSegMap[ai] = m;
        argRoles.push_back(Role::ColSeg);
        continue;
      }
      if (groupedBatch > 0 && t.getElementType() == f32 && s.size() == 3 &&
          s == ArrayRef<int64_t>({groupedBatch, groupedGroups, N})) {
        argRoles.push_back(Role::Row);
        continue;
      }
      if (flatSingletonRank4 && isa<FloatType>(t.getElementType()) &&
          s == ArrayRef<int64_t>({1, M, 1, N})) {
        argRoles.push_back(Role::Row);
        continue;
      }
      // A rank-1 f32 operand addressed as the data side of a validated
      // take(axis=0) gather is a parameter vector even when its extent happens
      // to equal M or N.  Resolve this from use semantics before the otherwise
      // ambiguous shape-only Col/PerRow classification.
      bool isTakeGatherParameter = false;
      if (t.getElementType() == f32 && s.size() == 1) {
        for (Operation *user : fn.getArgument(ai).getUsers()) {
          auto gather = dyn_cast<GatherOp>(user);
          if (gather && gather.getData() == fn.getArgument(ai) &&
              gather.getMode() == "take" && gather.getAxis() == 0) {
            isTakeGatherParameter = true;
            break;
          }
        }
      }
      if (isTakeGatherParameter) {
        argRoles.push_back(Role::ParamVector);
        continue;
      }
      if (s.size() == 1 && s[0] == M && t.getElementType().isInteger()) {
        rowIntTy = t.getElementType();
        argRoles.push_back(Role::RowInt);
        continue;
      }
      // Shape-only classification cannot distinguish a per-row vector from a
      // per-column vector on square domains. Direct right-aligned map/select
      // use is a compiler-visible proof of Col semantics; contractions and
      // other ambiguous constructions still fail closed.
      if ((t.getElementType() == f32 || explicitHalfArgument) &&
          s.size() == 1 && s[0] == M && M == N) {
        if (isTrailingColumnBroadcast(fn.getArgument(ai), M, N)) {
          argRoles.push_back(Role::Col);
          continue;
        }
        return failure();
      }
      bool isRowStorage = t.getElementType() == f32 ||
                          (halfSoftmax &&
                           t.getElementType() == storageType) ||
                          explicitHalfArgument;
      if (isRowStorage && s.size() == 2 && s[0] == M && s[1] == N)
        argRoles.push_back(Role::Row);
      else if ((t.getElementType() == f32 || explicitHalfArgument) &&
               s.size() == 1 && s[0] == N)
        argRoles.push_back(Role::Col);
      else if (t.getElementType() == f32 && s.size() == 1 && s[0] == M)
        argRoles.push_back(Role::PerRow);
      else if (t.getElementType() == f32 && s.size() == 1)
        argRoles.push_back(Role::ParamVector);
      else if (s.empty() &&
               isa<FloatType, IntegerType>(t.getElementType()))
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
        bool transposedRowReduce =
            ax == 0 && inTy.getRank() == 2 &&
            inTy.getDimSize(0) == N && inTy.getDimSize(1) == M &&
            rTy.getRank() == 1 && rTy.getDimSize(0) == M;
        bool groupedRowReduce =
            groupedBatch > 0 && ax == 2 && inTy.getRank() == 3 &&
            inTy.getShape() ==
                ArrayRef<int64_t>({groupedBatch, groupedGroups, N}) &&
            rTy.getRank() == 2 && rTy.getDimSize(0) == groupedBatch &&
            rTy.getDimSize(1) == groupedGroups;
        bool flatSingletonRowReduce =
            flatSingletonRank4 && ax == 3 && inTy.getRank() == 4 &&
            inTy.getShape() == ArrayRef<int64_t>({1, M, 1, N}) &&
            rTy.getRank() == 3 &&
            rTy.getShape() == ArrayRef<int64_t>({1, M, 1});
        if (ax == 0 && !transposedRowReduce) {
          // Cross-row fold: [M] per-row vector -> rank-0 scalar, sum only (the
          // atomicAdd identity is +). The input is a per-row uniform scalar.
          if (inTy.getRank() != 1 || inTy.getDimSize(0) != M ||
              rTy.getRank() != 0 || r.getFn() != "sum")
            return failure();
          crossReduces.push_back(r);
          continue;
        }
        if (inTy.getRank() != 2 && !groupedRowReduce &&
            !flatSingletonRowReduce)
          return failure();
        // Per-row last-axis reduce: result is [M,1] (keepdims) or [M] (collapsed,
        // flash's delta layout). Both are treated as a per-row uniform scalar.
        bool okRes = groupedRowReduce || flatSingletonRowReduce ||
            (rTy.getRank() == 2 && rTy.getDimSize(0) == M &&
             rTy.getDimSize(1) == 1) ||
            (rTy.getRank() == 1 && rTy.getDimSize(0) == M);
        if ((!transposedRowReduce && !groupedRowReduce &&
             !flatSingletonRowReduce && ax != 1) ||
            !okRes)
          return failure();
        if (!reduceInfoFor(r.getFn()))
          return failure();
        // A reduction over a statically singleton feature axis is an identity
        // (live cross entropy uses this to collapse [M,1] gather/LSE values to
        // [M]).  It is hoisted as a per-row uniform, not executed as an N-wide
        // block reduction.
        if (inTy.getRank() == 2 && inTy.getDimSize(1) == 1) {
          if (rTy.getRank() == 1)
            continue;
          // The current uniform table represents collapsed [M] values.  A
          // keepdims [M,1] identity would otherwise enter the N-wide reducer
          // and multiply the same scalar by N.
          return failure();
        }
        Type inElem = inTy.getElementType();
        Type outElem = rTy.getElementType();
        if (inElem != outElem)
          return failure();
        // The manual scan-path tree supports the sparsemax support-count
        // reduction natively as i32 sum.  The exact half-softmax graph is the
        // only other non-f32 source type; its emitted accumulator is f32.
        if (inElem != f32 && !(halfSoftmax && inElem == storageType) &&
            (!inElem.isInteger(32) || r.getFn() != "sum"))
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
            inTy.getElementType() != f32 ||
            !reduceInfoFor(sc.getFn()) || sc.getFn() == "mean" ||
            sc.getFn() == "prod")
          return failure();
        scans.push_back(sc);
      } else if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
        auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
        if (!dense || !dense.isSplat() ||
            !isa<FloatType, IntegerType>(
                cast<ShapedType>(dense.getType()).getElementType()))
          return failure();
      } else if (auto contraction = dyn_cast<ContractionOp>(&op)) {
        auto resultTy = cast<RankedTensorType>(contraction.getResult().getType());
        bool standardShape =
            resultTy.getRank() == 2 && resultTy.getDimSize(0) == M &&
            (resultTy.getDimSize(1) == 1 || resultTy.getDimSize(1) == N);
        bool groupedBroadcast =
            groupedBatch > 0 && resultTy.getRank() == 3 &&
            resultTy.getDimSize(0) == groupedBatch &&
            resultTy.getDimSize(1) == groupedGroups &&
            (resultTy.getDimSize(2) == 1 || resultTy.getDimSize(2) == N);
        bool groupedIndexGrid =
            groupedBatch > 0 && resultTy.getRank() == 2 &&
            resultTy.getDimSize(0) == groupedGroups &&
            resultTy.getDimSize(1) == N;
        bool perRowVectorShape =
            resultTy.getRank() == 2 && resultTy.getDimSize(0) == M &&
            resultTy.getDimSize(1) > 1 && resultTy.getDimSize(1) <= 16;
        if (!isCoordinatePreservingContraction(contraction) ||
            !resultTy.hasStaticShape() ||
            (!standardShape && !groupedBroadcast && !groupedIndexGrid &&
             !perRowVectorShape))
          return failure();
      } else if (auto gather = dyn_cast<GatherOp>(&op)) {
        auto dataTy = cast<RankedTensorType>(gather.getData().getType());
        auto indexTy = cast<RankedTensorType>(gather.getIndex().getType());
        auto resultTy = cast<RankedTensorType>(gather.getResult().getType());
        auto dataArg = dyn_cast<BlockArgument>(gather.getData());
        bool rowGather =
            gather.getMode() == "along_axis" && gather.getAxis() == 1 &&
            dataTy.getRank() == 2 && dataTy.getDimSize(0) == M &&
            dataTy.getDimSize(1) == N && indexTy.getRank() == 2 &&
            indexTy.getDimSize(0) == M && indexTy.getDimSize(1) == 1 &&
            resultTy.getShape() == indexTy.getShape() &&
            (hasProvenGatherBounds(gather) ||
             !gather.getIndex().getDefiningOp<arith::ConstantOp>());
        bool parameterGather =
            dataArg && argRoles[dataArg.getArgNumber()] == Role::ParamVector &&
            gather.getMode() == "take" && gather.getAxis() == 0 &&
            dataTy.getRank() == 1 && hasProvenGatherBounds(gather) &&
            resultTy.getShape() == indexTy.getShape();
        if (!rowGather && !parameterGather)
          return failure();
      } else if (auto concat = dyn_cast<ConcatOp>(&op)) {
        auto resultTy = cast<RankedTensorType>(concat.getResult().getType());
        if (concat.getAxis() != 1 || resultTy.getRank() != 2 ||
            resultTy.getDimSize(0) != M || resultTy.getDimSize(1) < 2 ||
            resultTy.getDimSize(1) > 16 ||
            concat.getInputs().size() !=
                static_cast<size_t>(resultTy.getDimSize(1)))
          return failure();
        for (Value input : concat.getInputs()) {
          auto inputTy = cast<RankedTensorType>(input.getType());
          if (inputTy.getRank() != 2 || inputTy.getDimSize(0) != M ||
              inputTy.getDimSize(1) != 1)
            return failure();
        }
      } else if (auto transpose = dyn_cast<TransposeOp>(&op)) {
        if (!isRowDomainTranspose(transpose) &&
            !isFlatSingletonOutputTranspose(transpose))
          return failure();
      } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(&op)) {
        if (!isUnitAxisExpansion(expand))
          return failure();
      } else if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, IotaOp, CastOp>(&op)) {
        // SelectOp + comparison map_binary (-> i1) + IotaOp are the M6 residual:
        // cross_entropy's target gather/one-hot (`select(iota == Y, X, 0)`). The
        // ops are general (owned by M3/M6); this just teaches emitAt to evaluate
        // them inside the row-reduction register tile.
        return failure();
      }
    }
    if (reduces.empty() && scans.empty())
      return failure(); // pure elementwise -> not our pass
    // Assign every iota an explicit logical role before emission.  Extent is
    // sufficient only when it matches one supported axis.  Collisions among
    // grouped B/G/N (or square M/N) require def-use/einsum proof; batch-axis
    // iotas are not implemented by this row path and therefore fail closed.
    iotaRoles.clear();
    for (Operation &op : body.without_terminator())
      if (auto iota = dyn_cast<IotaOp>(&op)) {
        auto type = cast<RankedTensorType>(iota.getResult().getType());
        if (type.getRank() != 1)
          return failure();
        int64_t extent = type.getDimSize(0);
        bool columnExtent = extent == N;
        bool groupExtent = groupedGroups > 0 && extent == groupedGroups;
        bool batchExtent = groupedBatch > 0 && extent == groupedBatch;
        bool outputVectorExtent = llvm::is_contained(outVectorWidths, extent);
        if (groupedBatch > 0) {
          int matches = static_cast<int>(columnExtent) +
                        static_cast<int>(groupExtent) +
                        static_cast<int>(batchExtent);
          if (matches == 0)
            return failure();
          if (matches == 1) {
            if (batchExtent)
              return failure();
            iotaRoles[iota.getResult()] =
                columnExtent ? IotaRole::Column : IotaRole::Group;
            continue;
          }
          bool columnProof =
              columnExtent &&
              isProvablyCoordinate(iota.getResult(), IotaRole::Column);
          bool groupProof =
              groupExtent &&
              isProvablyCoordinate(iota.getResult(), IotaRole::Group);
          if (columnProof == groupProof)
            return failure();
          iotaRoles[iota.getResult()] =
              columnProof ? IotaRole::Column : IotaRole::Group;
          continue;
        }
        if (!columnExtent && outputVectorExtent) {
          iotaRoles[iota.getResult()] = IotaRole::OutputVector;
          continue;
        }
        if (!columnExtent ||
            (M == N &&
             !isProvablyCoordinate(iota.getResult(), IotaRole::Column)))
          return failure();
        iotaRoles[iota.getResult()] = IotaRole::Column;
      }
    // Integer reductions currently use the dynamically-shared manual tree in
    // rewriteScan.  Do not claim a non-scan graph until the gpu.all_reduce and
    // tiled paths have an equivalent integer implementation.
    if (scans.empty())
      for (ReduceOp r : reduces)
        if (Type elementType =
                cast<RankedTensorType>(r.getInput().getType()).getElementType();
            elementType != f32 &&
            !(halfSoftmax && elementType == storageType))
          return failure();

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
    if (T <= 0)
      return failure();
    if (N % T == 0) {
      K = N / T;
    } else {
      hasFeatureTail = true;
      K = (N + T - 1) / T;
    }
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
    // The mixed-precision implementation is deliberately the read-once
    // register path.  The tiled/shared-cache and scan paths still model their
    // frontier buffers in the source element type and therefore must not claim
    // half-softmax support.
    if ((halfSoftmax || explicitHalfStorage) &&
        (numTiles > 1 || !scans.empty() || requestedReductionParts > 1))
      return failure();
    // Multi-tile (tiled + smem-cached) path scope: one or more per-row reduces,
    // no cross-row/atomic fold, outputs are RowOut/PerRowOut only. Independent
    // reductions in a dependency level share one feature-tile pass; a dependent
    // reduction starts a new pass over the already cached row frontier.  Thus
    // the implementation retains read-once global traffic without guessing a
    // benchmark-specific normalization shape.
    if (numTiles > 1 && requestedReductionParts == 1) {
      if (reduces.empty() || !crossReduces.empty())
        return failure();
      // ColSeg's per-(group,col) gather is only wired in the single-tile epilogue
      // (group_norm's hidden-per-group is small -> single tile). The tiled +
      // smem-cached path does not emit the affine channel index.
      if (numGroups != 0)
        return failure();
      for (ResultRole rr : outRoles)
        if (rr != ResultRole::RowOut && rr != ResultRole::PerRowOut)
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
      if (numTiles != 1 || !crossReduces.empty() || numGroups != 0 ||
          requestedReductionParts != 1)
        return failure();
      if ((T & (T - 1)) != 0)
        return failure();
      // rewriteScan currently owns only full row outputs.  Reject compact
      // per-row/atomic/vector outputs here instead of silently giving them a
      // rank-1 ABI and omitting their stores.
      for (ResultRole rr : outRoles)
        if (rr != ResultRole::RowOut)
          return failure();
    }
    if ((numTiles > 1 && requestedReductionParts == 1) || !scans.empty())
      for (Role role : argRoles)
        if (role == Role::PerRow)
          return failure(); // per-row value inputs are wired in register mode

    // Multi-stage path: one source-neutral row reduction is split across P
    // independent blocks, finalized in a second launch, and—when needed—fed to
    // a flat full-row epilogue launch.  This first implementation deliberately
    // excludes dependent/multiple reductions, scans, cross-row atomics, grouped
    // indexing, half storage, and compact vector outputs.  Every excluded graph
    // remains on an existing lowering state rather than being approximated.
    if (requestedReductionParts > 1) {
      if (reduces.size() != 1 || !crossReduces.empty() || !scans.empty() ||
          groupedBatch != 0 || numGroups != 0 || storageType != f32 ||
          T < 32 || T > 1024 || (T & (T - 1)) != 0 ||
          N < T * static_cast<int64_t>(requestedReductionParts))
        return failure();
      for (ResultRole role : outRoles)
        if (role != ResultRole::RowOut && role != ResultRole::PerRowOut)
          return failure();
      for (unsigned index = 0; index < outRoles.size(); ++index)
        if (cast<RankedTensorType>(fty.getResult(index)).getElementType() != f32)
          return failure();
    }

    // M6: fold the launch geometry + fork decision into the first-class
    // Schedule. grid = one block per row (M), block = T threads over the feature
    // dim; perThreadTile = the register/tiled strip Kpt; splitFactor = numTiles.
    // `kind` is the three-way fork rewrite() then dispatches on. `smemBytes` is
    // filled per path (it depends on the path's shared-memory carving).
    sched.gridX = M;
    sched.blockX = T;
    sched.perThreadTile = Kpt;
    sched.splitFactor = numTiles;
    sched.gridStyle = GridStyle::OneBlockPerRow;
    if (requestedReductionParts > 1) {
      sched.kind = KMultiStage;
      sched.gridX = M * static_cast<int64_t>(requestedReductionParts);
      sched.perThreadTile =
          (N + T * static_cast<int64_t>(requestedReductionParts) - 1) /
          (T * static_cast<int64_t>(requestedReductionParts));
      sched.splitFactor = requestedReductionParts;
    } else if (!scans.empty())
      sched.kind = KScan;
    else if (numTiles > 1)
      sched.kind = KTiled;
    else
      sched.kind = KRegister;
    if ((sched.kind == KTiled || sched.kind == KScan) &&
        std::any_of(inplaceArg.begin(), inplaceArg.end(),
                    [](int64_t arg) { return arg >= 0; }))
      return failure();
    if (requestedRowsPerBlock > 1) {
      bool hasUnsupportedOutput = llvm::any_of(
          outRoles, [](ResultRole role) {
            return role == ResultRole::AtomicOut ||
                   role == ResultRole::PerRowVectorOut;
          });
      bool hasInplaceOutput = llvm::any_of(
          inplaceArg, [](int64_t argument) { return argument >= 0; });
      if (T != 32 || M % requestedRowsPerBlock != 0 ||
          sched.kind != KRegister || !scans.empty() ||
          !crossReduces.empty() || groupedBatch != 0 || numGroups != 0 ||
          flatSingletonRank4 || hasUnsupportedOutput || hasInplaceOutput ||
          requestedReductionParts != 1)
        return failure();
      sched.gridX = M / requestedRowsPerBlock;
      sched.blockX = T * requestedRowsPerBlock;
    }

    // Contiguous vector gate (register-tile path only): vector<4xf32> for
    // native f32 graphs, vector<4xf16/bf16> for mixed-precision storage.
    // Vectorization changes only the packetization of in-bounds Row/Col loads
    // and RowOut stores.  Iota/comparison/select DAGs still execute lane by lane
    // through emitAt using the exact scalar column represented by each packet
    // lane, so column-varying control is not a reason to suppress packet I/O.
    // The no-tail/no-inactive-thread guards above are the memory-safety proof.
    // Every Row/Col ABI value and RowOut must share the selected storage type.
    if (requestedVectorWidth != 1 && sched.kind == KRegister &&
        !hasInactiveFeatureThreads && !hasFeatureTail && numGroups == 0) {
      bool allVectorABIValuesF32 = true;
      for (unsigned index = 0; index < argRoles.size(); ++index)
        if ((argRoles[index] == Role::Row || argRoles[index] == Role::Col) &&
            cast<RankedTensorType>(fty.getInput(index)).getElementType() != f32)
          allVectorABIValuesF32 = false;
      for (unsigned index = 0; index < outRoles.size(); ++index)
        if (outRoles[index] == ResultRole::RowOut &&
            cast<RankedTensorType>(fty.getResult(index)).getElementType() != f32)
          allVectorABIValuesF32 = false;
      if (allVectorABIValuesF32 && (K % 4 == 0)) {
        useVec4 = true;
        numVecTiles = K / 4;
        sched.vectorWidth = 4; // M6: record the vectorized load/store plan
      } else if ((K % 4 == 0) && (halfSoftmax || explicitHalfStorage) &&
                 llvm::none_of(inplaceArg,
                               [](int64_t arg) { return arg >= 0; })) {
        Type candidate;
        bool legalHalfABI = true;
        bool sawRow = false;
        for (unsigned index = 0; index < argRoles.size(); ++index) {
          if (argRoles[index] != Role::Row && argRoles[index] != Role::Col)
            continue;
          Type elementType =
              cast<RankedTensorType>(fty.getInput(index)).getElementType();
          if (!isHalfStorageType(elementType) ||
              (candidate && candidate != elementType)) {
            legalHalfABI = false;
            break;
          }
          candidate = elementType;
          sawRow |= argRoles[index] == Role::Row;
        }
        for (unsigned index = 0; legalHalfABI && index < outRoles.size();
             ++index) {
          if (outRoles[index] != ResultRole::RowOut)
            continue;
          Type elementType =
              cast<RankedTensorType>(fty.getResult(index)).getElementType();
          if (!candidate || elementType != candidate)
            legalHalfABI = false;
        }
        if (legalHalfABI && sawRow && candidate) {
          useHalfVec4 = true;
          halfVectorStorageType = candidate;
          numVecTiles = K / 4;
          sched.vectorWidth = 4;
        }
      }
    } else if ((requestedVectorWidth == 0 || requestedVectorWidth == 4) &&
               sched.kind == KScan &&
               (K % 4 == 0) && numGroups == 0) {
      // The scan owns contiguous columns per thread.  Retype each Row pointer
      // to vec4 so every four columns use one 16-byte global transaction; the
      // register scan still sees the extracted scalar lanes in exact order.
      useVec4 = true;
      numVecTiles = K / 4;
      sched.vectorWidth = 4;
    }
    if (requestedVectorWidth == 4 && !useVec4 && !useHalfVec4)
      return failure();
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
  DenseMap<Value, SmallVector<Value>> colVals;   // vec4 Col arg -> K loaded SSA values
  DenseMap<Value, SmallVector<Value>> scanVals;  // scan result -> K cumsum SSA values
  DenseMap<Value, SmallVector<Value>> pointVals; // common per-column DAG values
  DenseMap<Value, Value> approximateReciprocals; // emitted rhs -> searched rcp
  SmallVector<Value> cols;                       // K column indices (tx + k*T)
  DenseMap<Value, Value> frontierBuf;            // cached row value -> smem memref

  /// A rank-1, length-one tensor is a typed singleton broadcast, not a general
  /// parameter vector.  Keep the ParamVector role so take(axis=0) legality and
  /// ABI shape remain unchanged, but allow direct map/contraction consumers to
  /// treat its sole element as a per-launch uniform scalar.
  bool isSingletonParamVector(unsigned argument) {
    if (argument >= argRoles.size() ||
        argRoles[argument] != Role::ParamVector)
      return false;
    auto type = dyn_cast<RankedTensorType>(
        fn.getFunctionType().getInput(argument));
    return type && type.hasStaticShape() && type.getRank() == 1 &&
           type.getDimSize(0) == 1;
  }

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
    {
      auto it = pointVals.find(v);
      if (it != pointVals.end() && k < it->second.size() && it->second[k])
        return it->second[k];
    }
    auto remember = [&](Value result) -> Value {
      if (!result)
        return nullptr;
      SmallVector<Value> &values = pointVals[v];
      if (values.empty())
        values.resize(K);
      values[k] = result;
      return result;
    };
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Row)
        return rowVals[v][k];
      if (role == Role::Col) {
        auto it = colVals.find(v);
        if (it != colVals.end())
          return it->second[k];
        return remember(
            b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{cols[k]}));
      }
      if (role == Role::ColSeg) {
        // channel = col_index_map(group, col[k]); load W/B[channel].
        AffineMap m = colSegMap[barg.getArgNumber()];
        Value ch = emitAffineExpr(b, loc, m.getResult(0),
                                  ValueRange{groupVal, cols[k]});
        return remember(
            b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{ch}));
      }
      if (role == Role::ParamVector) {
        if (!isSingletonParamVector(barg.getArgNumber()))
          return nullptr; // addressed only by a validated intent.gather
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        return remember(
            b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{zero}));
      }
      return remember(
          b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{})); // scalar
    }
    Operation *def = v.getDefiningOp();
    if (auto cst = dyn_cast<arith::ConstantOp>(def))
      return remember(emitSplatScalarConstant(b, loc, cst));
    if (auto io = dyn_cast<IotaOp>(def)) {
      // iota at column col[k] is the column index itself, cast to the op's
      // integer width (no materialized vector). Feeds `iota == Y` (a cmpi).
      auto ioTy = cast<RankedTensorType>(io.getResult().getType());
      Type elt = ioTy.getElementType();
      if (ioTy.getRank() != 1)
        return nullptr;
      auto role = iotaRoles.find(io.getResult());
      if (role == iotaRoles.end())
        return nullptr;
      if (role->second == IotaRole::Column)
        return remember(b.create<arith::IndexCastOp>(loc, elt, cols[k]));
      if (role->second == IotaRole::Group && groupVal)
        return remember(b.create<arith::IndexCastOp>(loc, elt, groupVal));
      return nullptr;
    }
    if (auto castOp = dyn_cast<CastOp>(def)) {
      Type dst = cast<RankedTensorType>(castOp.getResult().getType())
                     .getElementType();
      return remember(emitRowScalarCast(
          b, loc, dst, emitAt(b, loc, castOp.getInput(), k)));
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      if (!isUnitAxisExpansion(expand))
        return nullptr;
      return remember(emitAt(b, loc, expand.getSrc(), k));
    }
    if (auto transpose = dyn_cast<TransposeOp>(def))
      return remember(emitAt(b, loc, transpose.getInput(), k));
    if (auto mu = dyn_cast<MapUnaryOp>(def)) {
      if (MapUnaryOp squareRoot = reciprocalSqrtOperand(mu))
        return remember(emitScalarUnary(
            b, loc, "rsqrt", emitAt(b, loc, squareRoot.getInput(), k)));
      return remember(emitScalarUnary(b, loc, mu.getFn(),
                                      emitAt(b, loc, mu.getInput(), k)));
    }
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      if (enableFmaFusion && mb.getFn() == "add") {
        auto emitFma = [&](Value maybeMul, Value addend) -> Value {
          auto mul = maybeMul.getDefiningOp<MapBinaryOp>();
          if (!mul || mul.getFn() != "mul" || !mul.getResult().hasOneUse())
            return nullptr;
          Value lhs = emitAt(b, loc, mul.getLhs(), k);
          Value rhs = emitAt(b, loc, mul.getRhs(), k);
          Value bias = emitAt(b, loc, addend, k);
          if (!lhs || !rhs || !bias || lhs.getType() != f32 ||
              rhs.getType() != f32 || bias.getType() != f32)
            return nullptr;
          return b.create<math::FmaOp>(loc, lhs, rhs, bias);
        };
        if (Value fused = emitFma(mb.getLhs(), mb.getRhs()))
          return remember(fused);
        if (Value fused = emitFma(mb.getRhs(), mb.getLhs()))
          return remember(fused);
      }
      Value lhs = emitAt(b, loc, mb.getLhs(), k);
      Value rhs = emitAt(b, loc, mb.getRhs(), k);
      if (isCmpFn(mb.getFn())) // i1 condition for a select
        return remember(isa<IntegerType>(lhs.getType())
                            ? emitScalarCmpI(b, loc, mb.getFn(), lhs, rhs)
                            : emitScalarCmp(b, loc, mb.getFn(), lhs, rhs));
      return remember(emitMappedBinary(b, loc, mb, lhs, rhs, f32));
    }
    if (auto contraction = dyn_cast<ContractionOp>(def)) {
      if (!isCoordinatePreservingContraction(contraction))
        return nullptr;
      Value lhs = emitAt(b, loc, contraction.getA(), k);
      Value rhs = emitAt(b, loc, contraction.getB(), k);
      if (!lhs || !rhs)
        return nullptr;
      Type dst = cast<RankedTensorType>(contraction.getResult().getType())
                     .getElementType();
      if (lhs.getType() != dst)
        lhs = emitRowScalarCast(b, loc, dst, lhs);
      if (rhs.getType() != dst)
        rhs = emitRowScalarCast(b, loc, dst, rhs);
      if (!lhs || !rhs)
        return nullptr;
      return remember(isa<FloatType>(dst)
                          ? Value(b.create<arith::MulFOp>(loc, lhs, rhs))
                          : Value(b.create<arith::MulIOp>(loc, lhs, rhs)));
    }
    if (auto gather = dyn_cast<GatherOp>(def)) {
      auto dataArg = dyn_cast<BlockArgument>(gather.getData());
      if (!dataArg ||
          argRoles[dataArg.getArgNumber()] != Role::ParamVector ||
          gather.getMode() != "take" || gather.getAxis() != 0)
        return nullptr;
      Value index = emitAt(b, loc, gather.getIndex(), k);
      index = emitSafeGatherIndex(b, loc, gather, index);
      if (!index)
        return nullptr;
      return remember(b.create<memref::LoadOp>(
          loc, argMemref[gather.getData()], ValueRange{index}));
    }
    if (auto sel = dyn_cast<SelectOp>(def)) {
      Value cond = emitAt(b, loc, sel.getCond(), k);
      Value x = emitAt(b, loc, sel.getX(), k);
      Value y = emitAt(b, loc, sel.getY(), k);
      if (!cond || !x || !y || !cond.getType().isInteger(1) ||
          x.getType() != y.getType())
        return nullptr;
      return remember(b.create<arith::SelectOp>(loc, cond, x, y));
    }
    return nullptr; // a reduce result not yet hoisted: impossible in topo order
  }

  /// Evaluate a compact per-row vector output at one logical vector column.
  /// This is the output-side sibling of emitAt's feature-column evaluator.
  /// Accepted DAGs are ordinary scalar maps/selects plus coordinate-preserving
  /// no-reduction contractions over per-row uniforms and a width-P iota.  It
  /// generalizes the historical singleton-column concat without inventing a
  /// PolyNorm-specific recipe.
  Value emitPerRowVectorAt(OpBuilder &b, Location loc, Value value,
                           int64_t column, int64_t width) {
    if (Value scalar = uniform.lookup(value))
      return scalar;

    auto evaluateOperand = [&](Value operand) -> Value {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || !type.hasStaticShape())
        return nullptr;
      if (type.getRank() == 0)
        return emitAt(b, loc, operand, 0);
      if (type.getRank() == 1 && type.getDimSize(0) == 1)
        return emitAt(b, loc, operand, 0);
      if (type.getRank() == 1 && type.getDimSize(0) == width)
        return emitPerRowVectorAt(b, loc, operand, column, width);
      if (type.getRank() == 1 && type.getDimSize(0) == M)
        return emitAt(b, loc, operand, 0);
      if (type.getRank() == 2 && type.getDimSize(0) == M &&
          type.getDimSize(1) == width)
        return emitPerRowVectorAt(b, loc, operand, column, width);
      if (type.getRank() == 2 && type.getDimSize(0) == M &&
          type.getDimSize(1) == 1)
        return emitAt(b, loc, operand, 0);
      return nullptr;
    };

    Operation *definition = value.getDefiningOp();
    if (auto constant = dyn_cast_or_null<arith::ConstantOp>(definition))
      return emitSplatScalarConstant(b, loc, constant);
    if (auto iota = dyn_cast_or_null<IotaOp>(definition)) {
      auto type = cast<RankedTensorType>(iota.getResult().getType());
      auto role = iotaRoles.find(iota.getResult());
      if (type.getRank() != 1 || type.getDimSize(0) != width ||
          role == iotaRoles.end() ||
          role->second != IotaRole::OutputVector)
        return nullptr;
      Value index = b.create<arith::ConstantIndexOp>(loc, column);
      return b.create<arith::IndexCastOp>(loc, type.getElementType(), index);
    }
    if (auto castOp = dyn_cast_or_null<CastOp>(definition)) {
      Value input = evaluateOperand(castOp.getInput());
      Type target = cast<RankedTensorType>(castOp.getResult().getType())
                        .getElementType();
      return input ? emitRowScalarCast(b, loc, target, input) : Value();
    }
    if (auto unary = dyn_cast_or_null<MapUnaryOp>(definition)) {
      Value input = evaluateOperand(unary.getInput());
      return input ? emitScalarUnary(b, loc, unary.getFn(), input) : Value();
    }
    if (auto binary = dyn_cast_or_null<MapBinaryOp>(definition)) {
      Value lhs = evaluateOperand(binary.getLhs());
      Value rhs = evaluateOperand(binary.getRhs());
      if (!lhs || !rhs)
        return nullptr;
      if (isCmpFn(binary.getFn()))
        return isa<IntegerType>(lhs.getType())
                   ? emitScalarCmpI(b, loc, binary.getFn(), lhs, rhs)
                   : emitScalarCmp(b, loc, binary.getFn(), lhs, rhs);
      return emitMappedBinary(b, loc, binary, lhs, rhs, f32);
    }
    if (auto select = dyn_cast_or_null<SelectOp>(definition)) {
      Value condition = evaluateOperand(select.getCond());
      Value positive = evaluateOperand(select.getX());
      Value negative = evaluateOperand(select.getY());
      if (!condition || !positive || !negative ||
          !condition.getType().isInteger(1) ||
          positive.getType() != negative.getType())
        return nullptr;
      return b.create<arith::SelectOp>(loc, condition, positive, negative);
    }
    if (auto contraction = dyn_cast_or_null<ContractionOp>(definition)) {
      if (!isCoordinatePreservingContraction(contraction))
        return nullptr;
      Value lhs = evaluateOperand(contraction.getA());
      Value rhs = evaluateOperand(contraction.getB());
      if (!lhs || !rhs)
        return nullptr;
      Type target = cast<RankedTensorType>(contraction.getResult().getType())
                        .getElementType();
      if (lhs.getType() != target)
        lhs = emitRowScalarCast(b, loc, target, lhs);
      if (rhs.getType() != target)
        rhs = emitRowScalarCast(b, loc, target, rhs);
      if (!lhs || !rhs)
        return nullptr;
      return isa<FloatType>(target)
                 ? Value(b.create<arith::MulFOp>(loc, lhs, rhs))
                 : Value(b.create<arith::MulIOp>(loc, lhs, rhs));
    }
    return nullptr;
  }

  /// Accumulate one register-tile term.  An explicit Layer-C FMA candidate may
  /// fuse a single-use elementwise product into an additive row reduction,
  /// matching the source compiler's f32 contraction while retaining the same
  /// Intent DAG and float-nondeterministic reduction policy.
  Value accumulateRegisterReduction(OpBuilder &b, Location loc, ReduceOp red,
                                    const ReduceInfo &info, Value acc,
                                    unsigned k, Value featureLaneActive,
                                    ArrayRef<Value> featureTileActive) {
    if (enableFmaFusion && info.op == gpu::AllReduceOperation::ADD &&
        red.getAssoc() == "float_nondeterministic") {
      auto mul = red.getInput().getDefiningOp<MapBinaryOp>();
      if (mul && mul.getFn() == "mul" && mul.getResult().hasOneUse()) {
        Value lhs = emitAt(b, loc, mul.getLhs(), k);
        Value rhs = emitAt(b, loc, mul.getRhs(), k);
        if (lhs && rhs && lhs.getType() == f32 && rhs.getType() == f32) {
          Value fused = b.create<math::FmaOp>(loc, lhs, rhs, acc);
          Value active =
              hasFeatureTail ? featureTileActive[k] : featureLaneActive;
          // A tail lane loaded a mechanically safe address, but its arithmetic
          // is semantically inert. Preserve the incoming accumulator instead
          // of evaluating `acc + identity` after the fused operation.
          if (active)
            fused = b.create<arith::SelectOp>(loc, active, fused, acc);
          return fused;
        }
      }
    }
    Value term = emitAt(b, loc, red.getInput(), k);
    Value active =
        hasFeatureTail ? featureTileActive[k] : featureLaneActive;
    if (active) {
      Value identity = b.create<arith::ConstantOp>(
          loc, b.getF32FloatAttr(info.ident));
      term = b.create<arith::SelectOp>(loc, active, term, identity);
    }
    return combineScalar(b, loc, info.op, acc, term);
  }

  /// Is an original value uniform (per-row / rank-0)? Row/Col args are not.
  bool isUniform(Value v) {
    if (uniform.count(v))
      return true;
    if (auto barg = dyn_cast<BlockArgument>(v))
      return argRoles[barg.getArgNumber()] == Role::Scalar ||
             isSingletonParamVector(barg.getArgNumber());
    return false;
  }

  /// True when `v`'s tensor DAG consumes `target`.  Scan-path reductions that
  /// are mutually independent can publish all of their warp partials before a
  /// single block barrier; a dependent reduction starts a new barrier group.
  bool dependsOn(Value v, Value target) {
    if (v == target)
      return true;
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    for (Value operand : def->getOperands())
      if (dependsOn(operand, target))
        return true;
    return false;
  }

  /// True when at least two row reductions can amortize one publication
  /// barrier. Single reductions and fully dependent chains keep gpu.all_reduce;
  /// the redundant cross-warp finalization of the grouped hierarchy is only a
  /// win when one barrier serves multiple independent accumulators.
  bool hasIndependentReductionGroup() {
    size_t groupStart = 0;
    while (groupStart < reduces.size()) {
      size_t groupEnd = groupStart + 1;
      while (groupEnd < reduces.size()) {
        bool dependent = false;
        for (size_t prior = groupStart; prior < groupEnd; ++prior)
          dependent |= dependsOn(reduces[groupEnd].getInput(),
                                 reduces[prior].getResult());
        if (dependent)
          break;
        ++groupEnd;
      }
      if (groupEnd - groupStart > 1)
        return true;
      groupStart = groupEnd;
    }
    return false;
  }

  /// A B-certified, exact sparsemax graph can use the sorted-prefix algorithm
  /// in `rewriteSortedSparsemax`.  Return (X arg index, SortedX arg index) only
  /// when the proof is entirely structural:
  ///
  ///   scan(SortedX) -> strict support predicate
  ///   k = sum(cast_i32(support))
  ///   s = sum(select(support, SortedX, 0))
  ///   O = max(X - tau(k, s), 0)
  ///
  /// The descending-order fact itself is never inferred from names; it must be
  /// the validated Layer-B BoolAttr materialized by P4.
  std::optional<std::pair<unsigned, unsigned>>
  sortedSparsemaxArgIndices() {
    auto sortedAttr = fn->getAttrOfType<BoolAttr>("intent_sorted_descending");
    if (!enableSortedPrefix || !sortedAttr || !sortedAttr.getValue() ||
        scans.size() != 1 ||
        reduces.size() != 2 || outputVals.size() != 1 ||
        outRoles[0] != ResultRole::RowOut || fn.getNumArguments() != 2 ||
        T < 32 || T > 1024 || (T % 32) != 0 || (N % (T * 4)) != 0)
      return std::nullopt;

    auto isZeroSplat = [](Value value) {
      auto cst = value.getDefiningOp<arith::ConstantOp>();
      auto dense = cst ? dyn_cast<DenseElementsAttr>(cst.getValue())
                       : DenseElementsAttr();
      return dense && dense.isSplat() &&
             isa<FloatType>(dense.getElementType()) &&
             dense.getSplatValue<APFloat>().isZero();
    };

    auto sortedArg = dyn_cast<BlockArgument>(scans[0].getInput());
    if (!sortedArg || argRoles[sortedArg.getArgNumber()] != Role::Row)
      return std::nullopt;

    ReduceOp countReduce;
    ReduceOp sumReduce;
    CastOp countCast;
    SelectOp sumSelect;
    for (ReduceOp reduce : reduces) {
      if (reduce.getFn() != "sum")
        continue;
      if (auto castOp = reduce.getInput().getDefiningOp<CastOp>()) {
        auto sourceType =
            dyn_cast<RankedTensorType>(castOp.getInput().getType());
        auto resultType =
            dyn_cast<RankedTensorType>(castOp.getResult().getType());
        if (sourceType && resultType &&
            sourceType.getElementType().isInteger(1) &&
            resultType.getElementType().isInteger(32)) {
          countReduce = reduce;
          countCast = castOp;
        }
      }
      if (auto selectOp = reduce.getInput().getDefiningOp<SelectOp>())
        if (selectOp.getX() == sortedArg && isZeroSplat(selectOp.getY())) {
          sumReduce = reduce;
          sumSelect = selectOp;
        }
    }
    if (!countReduce || !sumReduce ||
        countCast.getInput() != sumSelect.getCond())
      return std::nullopt;

    Value support = countCast.getInput();
    auto compare = support.getDefiningOp<MapBinaryOp>();
    if (!compare || compare.getFn() != "gt" ||
        compare.getLhs() != sortedArg ||
        !dependsOn(compare.getRhs(), scans[0].getResult()))
      return std::nullopt;

    auto outputMax = outputVals[0].getDefiningOp<MapBinaryOp>();
    if (!outputMax || outputMax.getFn() != "max")
      return std::nullopt;
    Value shifted;
    if (isZeroSplat(outputMax.getRhs()))
      shifted = outputMax.getLhs();
    else if (isZeroSplat(outputMax.getLhs()))
      shifted = outputMax.getRhs();
    else
      return std::nullopt;
    auto subtract = shifted.getDefiningOp<MapBinaryOp>();
    if (!subtract || subtract.getFn() != "sub")
      return std::nullopt;
    auto xArg = dyn_cast<BlockArgument>(subtract.getLhs());
    if (!xArg || xArg == sortedArg ||
        argRoles[xArg.getArgNumber()] != Role::Row ||
        !dependsOn(subtract.getRhs(), countReduce.getResult()) ||
        !dependsOn(subtract.getRhs(), sumReduce.getResult()))
      return std::nullopt;

    return std::make_pair(xArg.getArgNumber(), sortedArg.getArgNumber());
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
        if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
          if (Value scalar = emitSplatScalarConstant(b, loc, cst)) {
            uniform[res] = scalar;
            changed = true;
          }
        } else if (auto castOp = dyn_cast<CastOp>(&op)) {
          if (isUniform(castOp.getInput())) {
            Type dst = cast<RankedTensorType>(castOp.getResult().getType())
                           .getElementType();
            if (Value scalar = emitRowScalarCast(
                    b, loc, dst, emitAt(b, loc, castOp.getInput(), 0))) {
              uniform[res] = scalar;
              changed = true;
            }
          }
        } else if (auto reduce = dyn_cast<ReduceOp>(&op)) {
          auto inputTy = cast<RankedTensorType>(reduce.getInput().getType());
          auto resultTy = cast<RankedTensorType>(reduce.getResult().getType());
          if (inputTy.getRank() == 2 && inputTy.getDimSize(1) == 1 &&
              resultTy.getRank() == 1 && isUniform(reduce.getInput())) {
            uniform[res] = emitAt(b, loc, reduce.getInput(), 0);
            changed = true;
          }
        } else if (auto contraction = dyn_cast<ContractionOp>(&op)) {
          if (isNoReductionContraction(contraction) &&
              isUniform(contraction.getA()) &&
              isUniform(contraction.getB())) {
            if (Value scalar = emitAt(b, loc, res, 0)) {
              uniform[res] = scalar;
              changed = true;
            }
          }
        } else if (auto gather = dyn_cast<GatherOp>(&op)) {
          auto dataArg = dyn_cast<BlockArgument>(gather.getData());
          if (dataArg &&
              argRoles[dataArg.getArgNumber()] == Role::ParamVector &&
              isUniform(gather.getIndex())) {
            Value index = emitAt(b, loc, gather.getIndex(), 0);
            index = emitSafeGatherIndex(b, loc, gather, index);
            if (index) {
              uniform[res] = b.create<memref::LoadOp>(
                  loc, argMemref[gather.getData()], ValueRange{index});
              changed = true;
            }
          } else if (dataArg && argRoles[dataArg.getArgNumber()] == Role::Row &&
              isUniform(gather.getIndex()) && rowIndex) {
            Value index = emitAt(b, loc, gather.getIndex(), 0);
            index = emitSafeGatherIndex(b, loc, gather, index);
            if (index) {
              uniform[res] = b.create<memref::LoadOp>(
                  loc, argMemref[gather.getData()],
                  ValueRange{rowIndex, index});
              changed = true;
            }
          }
        } else if (auto mu = dyn_cast<MapUnaryOp>(&op)) {
          if (isFusedReciprocalSqrtInput(mu))
            continue;
          if (MapUnaryOp squareRoot = reciprocalSqrtOperand(mu);
              squareRoot && isUniform(squareRoot.getInput())) {
            uniform[res] = emitScalarUnary(
                b, loc, "rsqrt", emitAt(b, loc, squareRoot.getInput(), 0));
            changed = true;
          } else if (isUniform(mu.getInput())) {
            uniform[res] = emitScalarUnary(b, loc, mu.getFn(),
                                           emitAt(b, loc, mu.getInput(), 0));
            changed = true;
          }
        } else if (auto mb = dyn_cast<MapBinaryOp>(&op)) {
          if (isUniform(mb.getLhs()) && isUniform(mb.getRhs())) {
            Value lhs = emitAt(b, loc, mb.getLhs(), 0);
            Value rhs = emitAt(b, loc, mb.getRhs(), 0);
            Value scalar;
            if (lhs && rhs && isCmpFn(mb.getFn()))
              scalar = isa<IntegerType>(lhs.getType())
                           ? emitScalarCmpI(b, loc, mb.getFn(), lhs, rhs)
                           : emitScalarCmp(b, loc, mb.getFn(), lhs, rhs);
            else if (lhs && rhs)
              scalar = emitMappedBinary(b, loc, mb, lhs, rhs, f32);
            if (scalar) {
              uniform[res] = scalar;
              changed = true;
            }
          }
        } else if (auto sel = dyn_cast<SelectOp>(&op)) {
          if (isUniform(sel.getCond()) && isUniform(sel.getX()) &&
              isUniform(sel.getY())) {
            uniform[res] = b.create<arith::SelectOp>(
                loc, emitAt(b, loc, sel.getCond(), 0),
                emitAt(b, loc, sel.getX(), 0),
                emitAt(b, loc, sel.getY(), 0));
            changed = true;
          }
        }
      }
    }
  }

  Type memrefOf(unsigned argIdx) {
    Role role = argRoles[argIdx];
    auto tensorTy =
        cast<RankedTensorType>(fn.getFunctionType().getInput(argIdx));
    if (role == Role::Unused)
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    if (role == Role::Row)
      return MemRefType::get({M, N}, tensorTy.getElementType());
    if (role == Role::Col)
      return MemRefType::get({N}, tensorTy.getElementType());
    if (role == Role::ColSeg)
      return MemRefType::get({colSegLen[argIdx]}, f32);
    if (role == Role::ParamVector)
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    if (role == Role::RowInt)
      return MemRefType::get({M}, rowIntTy);
    if (role == Role::PerRow)
      return MemRefType::get({M}, tensorTy.getElementType());
    return MemRefType::get({}, tensorTy.getElementType());
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
    if (isa<arith::ConstantOp>(d))
      return false;
    if (auto castOp = dyn_cast<CastOp>(d))
      return isRowVarying(castOp.getInput());
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(d))
      return isRowVarying(expand.getSrc());
    if (auto transpose = dyn_cast<TransposeOp>(d))
      return isRowVarying(transpose.getInput());
    if (auto mu = dyn_cast<MapUnaryOp>(d))
      return isRowVarying(mu.getInput());
    if (auto mb = dyn_cast<MapBinaryOp>(d))
      return isRowVarying(mb.getLhs()) || isRowVarying(mb.getRhs());
    if (auto contraction = dyn_cast<ContractionOp>(d))
      return isCoordinatePreservingContraction(contraction) &&
             (isRowVarying(contraction.getA()) ||
              isRowVarying(contraction.getB()));
    if (auto gather = dyn_cast<GatherOp>(d))
      return isRowVarying(gather.getIndex());
    if (auto select = dyn_cast<SelectOp>(d))
      return isRowVarying(select.getCond()) ||
             isRowVarying(select.getX()) || isRowVarying(select.getY());
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
    if (auto castOp = dyn_cast<CastOp>(d)) {
      markFrontier(castOp.getInput(), fset);
      return;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(d)) {
      markFrontier(expand.getSrc(), fset);
      return;
    }
    if (auto transpose = dyn_cast<TransposeOp>(d)) {
      markFrontier(transpose.getInput(), fset);
      return;
    }
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
    if (auto contraction = dyn_cast<ContractionOp>(d)) {
      bool lhs = isRowVarying(contraction.getA());
      bool rhs = isRowVarying(contraction.getB());
      if (lhs && rhs) {
        fset.insert(v);
        return;
      }
      markFrontier(lhs ? contraction.getA() : contraction.getB(), fset);
      return;
    }
    if (isa<GatherOp>(d)) {
      fset.insert(v);
      return;
    }
    if (isa<SelectOp>(d)) {
      // A select can combine a row-varying predicate and two row-varying arms.
      // Cache its already-evaluated result as one frontier instead of expanding
      // three independent shared buffers.
      fset.insert(v);
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
      if (auto constant = dyn_cast<arith::ConstantOp>(d)) {
        res = emitSplatScalarConstant(b, loc, constant);
      } else if (auto castOp = dyn_cast<CastOp>(d)) {
        Type dst = cast<RankedTensorType>(castOp.getResult().getType())
                       .getElementType();
        Value input =
            emitAtCol(b, loc, castOp.getInput(), col, bx, cur, fromSmem);
        if (input)
          res = emitRowScalarCast(b, loc, dst, input);
      } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(d)) {
        res = emitAtCol(b, loc, expand.getSrc(), col, bx, cur, fromSmem);
      } else if (auto mu = dyn_cast<MapUnaryOp>(d)) {
        Value input;
        StringRef unaryFn = mu.getFn();
        if (MapUnaryOp squareRoot = reciprocalSqrtOperand(mu)) {
          unaryFn = "rsqrt";
          input = emitAtCol(b, loc, squareRoot.getInput(), col, bx, cur,
                            fromSmem);
        } else {
          input = emitAtCol(b, loc, mu.getInput(), col, bx, cur, fromSmem);
        }
        if (input)
          res = emitScalarUnary(b, loc, unaryFn, input);
      } else if (auto mb = dyn_cast<MapBinaryOp>(d)) {
        Value lhs =
            emitAtCol(b, loc, mb.getLhs(), col, bx, cur, fromSmem);
        Value rhs =
            emitAtCol(b, loc, mb.getRhs(), col, bx, cur, fromSmem);
        if (lhs && rhs && isCmpFn(mb.getFn()))
          res = isa<IntegerType>(lhs.getType())
                    ? emitScalarCmpI(b, loc, mb.getFn(), lhs, rhs)
                    : emitScalarCmp(b, loc, mb.getFn(), lhs, rhs);
        else if (lhs && rhs)
          res = emitScalarBinary(b, loc, mb.getFn(), lhs, rhs);
      } else if (auto contraction = dyn_cast<ContractionOp>(d)) {
        if (isCoordinatePreservingContraction(contraction)) {
          Value lhs = emitAtCol(b, loc, contraction.getA(), col, bx, cur,
                                fromSmem);
          Value rhs = emitAtCol(b, loc, contraction.getB(), col, bx, cur,
                                fromSmem);
          Type dst = cast<RankedTensorType>(contraction.getResult().getType())
                         .getElementType();
          if (lhs && rhs && lhs.getType() != dst)
            lhs = emitRowScalarCast(b, loc, dst, lhs);
          if (lhs && rhs && rhs.getType() != dst)
            rhs = emitRowScalarCast(b, loc, dst, rhs);
          if (lhs && rhs)
            res = isa<FloatType>(dst)
                      ? Value(b.create<arith::MulFOp>(loc, lhs, rhs))
                      : Value(b.create<arith::MulIOp>(loc, lhs, rhs));
        }
      } else if (auto gather = dyn_cast<GatherOp>(d)) {
        auto dataArg = dyn_cast<BlockArgument>(gather.getData());
        if (dataArg &&
            argRoles[dataArg.getArgNumber()] == Role::ParamVector &&
            gather.getMode() == "take" && gather.getAxis() == 0) {
          Value index = emitAtCol(b, loc, gather.getIndex(), col, bx, cur,
                                  fromSmem);
          if (index)
            index = emitSafeGatherIndex(b, loc, gather, index);
          if (index)
            res = b.create<memref::LoadOp>(
                loc, argMemref[gather.getData()], ValueRange{index});
        }
      } else if (auto select = dyn_cast<SelectOp>(d)) {
        Value cond =
            emitAtCol(b, loc, select.getCond(), col, bx, cur, fromSmem);
        Value x = emitAtCol(b, loc, select.getX(), col, bx, cur, fromSmem);
        Value y = emitAtCol(b, loc, select.getY(), col, bx, cur, fromSmem);
        if (cond && x && y)
          res = b.create<arith::SelectOp>(loc, cond, x, y);
      } else if (auto iota = dyn_cast<IotaOp>(d)) {
        auto role = iotaRoles.find(iota.getResult());
        auto type = cast<RankedTensorType>(iota.getResult().getType());
        if (role != iotaRoles.end() && role->second == IotaRole::Column &&
            type.getRank() == 1)
          res = b.create<arith::IndexCastOp>(loc, type.getElementType(), col);
      } else if (auto transpose = dyn_cast<TransposeOp>(d))
        res = emitAtCol(b, loc, transpose.getInput(), col, bx, cur, fromSmem);
      else
        res = nullptr;
    }
    cur[v] = res;
    return res;
  }

  void resetDynamicEmissionState() {
    uniform.clear();
    rowVals.clear();
    colVals.clear();
    scanVals.clear();
    pointVals.clear();
    approximateReciprocals.clear();
    cols.clear();
    frontierBuf.clear();
    groupVal = nullptr;
    rowIndex = nullptr;
  }

  LogicalResult bindUniformInputs(OpBuilder &b, Location loc, Value row,
                                  Block &srcBlock,
                                  const DenseMap<Value, Value> &launchScalars) {
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Scalar) {
        auto equality = scalarEqualities.find(i);
        if (equality != scalarEqualities.end()) {
          auto tensorTy =
              cast<RankedTensorType>(fn.getFunctionType().getInput(i));
          auto integerTy = cast<IntegerType>(tensorTy.getElementType());
          uniform[arg] = b.create<arith::ConstantOp>(
              loc, b.getIntegerAttr(integerTy, equality->second));
        } else if (promoteRank0ScalarInputs) {
          uniform[arg] = launchScalars.lookup(arg);
          if (!uniform[arg]) {
            fn.emitError("rank-0 scalar was not promoted to launch ABI");
            return failure();
          }
        } else {
          uniform[arg] = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{});
        }
      } else if (isSingletonParamVector(i)) {
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{zero});
      } else if (argRoles[i] == Role::RowInt ||
                 argRoles[i] == Role::PerRow) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{row});
      }
    }
    hoistUniforms(b, loc);
    return success();
  }

  /// Split one independent row reduction over multiple CTAs, finalize its
  /// partials in a second launch, then evaluate full-row outputs in a flat third
  /// launch.  The compiler-emitted host graph owns both scratch buffers and the
  /// launch order; the portable runner transports that graph unchanged.
  LogicalResult rewriteMultiStage() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Block &srcBlock = fn.getBody().front();
    ReduceOp reduction = reduces.front();
    ReduceInfo info = *reduceInfoFor(reduction.getFn());
    int64_t parts = static_cast<int64_t>(requestedReductionParts);

    SmallVector<Type> inputTypes;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inputTypes.push_back(memrefOf(i));
    for (unsigned index = 0; index < outRoles.size(); ++index) {
      if (inplaceArg[index] >= 0)
        continue;
      if (outRoles[index] == ResultRole::RowOut)
        inputTypes.push_back(MemRefType::get({M, N}, f32));
      else
        inputTypes.push_back(MemRefType::get({M}, f32));
    }

    auto newType = FunctionType::get(ctx, inputTypes, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newType);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    argMemref.clear();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);
    SmallVector<Value> outputMemrefs;
    unsigned freshOutput = argRoles.size();
    for (unsigned index = 0; index < outRoles.size(); ++index) {
      if (inplaceArg[index] >= 0)
        outputMemrefs.push_back(entry->getArgument(inplaceArg[index]));
      else
        outputMemrefs.push_back(entry->getArgument(freshOutput++));
    }

    DenseMap<Value, Value> launchScalars;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (!promoteRank0ScalarInputs || argRoles[i] != Role::Scalar ||
          scalarEqualities.contains(i))
        continue;
      Value sourceArg = srcBlock.getArgument(i);
      launchScalars[sourceArg] = b.create<memref::LoadOp>(
          loc, argMemref[sourceArg], ValueRange{});
    }

    auto partialType = MemRefType::get({M, parts}, f32);
    auto statisticType = MemRefType::get({M}, f32);
    Value partials = b.create<memref::AllocOp>(loc, partialType);
    Value statistics = b.create<memref::AllocOp>(loc, statisticType);

    sched.smemBytes = 0;
    attachLaunchPlan(gpuFn, sched);
    if (Attribute equalities = fn->getAttr("intent.scalar_equalities"))
      gpuFn->setAttr("intent.scalar_equalities", equalities);

    Value hostOne = b.create<arith::ConstantIndexOp>(loc, 1);
    Value hostThreads = b.create<arith::ConstantIndexOp>(loc, T);
    Value hostPartialGrid =
        b.create<arith::ConstantIndexOp>(loc, M * parts);

    // Stage 1: block (row, part) walks a disjoint strided subset of the row and
    // publishes exactly one monoid partial.
    {
      auto launch = b.create<gpu::LaunchOp>(loc, hostPartialGrid, hostOne,
                                            hostOne, hostThreads, hostOne,
                                            hostOne);
      OpBuilder stage = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value block = launch.getBlockIds().x;
      Value thread = launch.getThreadIds().x;
      Value cParts = stage.create<arith::ConstantIndexOp>(loc, parts);
      Value row = stage.create<arith::DivUIOp>(loc, block, cParts);
      Value part = stage.create<arith::RemUIOp>(loc, block, cParts);
      Value cThreads = stage.create<arith::ConstantIndexOp>(loc, T);
      Value partBase = stage.create<arith::MulIOp>(loc, part, cThreads);
      Value begin = stage.create<arith::AddIOp>(loc, partBase, thread);
      Value stride = stage.create<arith::ConstantIndexOp>(loc, parts * T);
      Value upper = stage.create<arith::ConstantIndexOp>(loc, N);
      Value identity = stage.create<arith::ConstantOp>(
          loc, stage.getF32FloatAttr(info.ident));

      resetDynamicEmissionState();
      rowIndex = row;
      if (failed(bindUniformInputs(stage, loc, row, srcBlock, launchScalars)))
        return failure();
      auto loop = stage.create<scf::ForOp>(loc, begin, upper, stride,
                                           ValueRange{identity});
      OpBuilder body = OpBuilder::atBlockBegin(loop.getBody());
      DenseMap<Value, Value> current;
      Value term = emitAtCol(body, loc, reduction.getInput(),
                             loop.getInductionVar(), row, current,
                             /*fromSmem=*/false);
      if (!term)
        return failure();
      Value next = combineScalar(body, loc, info.op,
                                 loop.getRegionIterArg(0), term);
      body.create<scf::YieldOp>(loc, next);
      auto blockReduce = stage.create<gpu::AllReduceOp>(
          loc, loop.getResult(0),
          gpu::AllReduceOperationAttr::get(ctx, info.op),
          /*uniform=*/UnitAttr::get(ctx));
      Value zero = stage.create<arith::ConstantIndexOp>(loc, 0);
      Value lead = stage.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, thread, zero);
      auto store = stage.create<scf::IfOp>(loc, lead, false);
      OpBuilder thenBuilder = store.getThenBodyBuilder();
      thenBuilder.create<memref::StoreOp>(
          loc, blockReduce.getResult(), partials, ValueRange{row, part});
      stage.create<gpu::TerminatorOp>(loc);
    }

    Value hostRows = b.create<arith::ConstantIndexOp>(loc, M);

    // Stage 2: one block per row folds P partials, stores the statistic for the
    // epilogue, and materializes every per-row result exactly once.
    {
      auto launch = b.create<gpu::LaunchOp>(loc, hostRows, hostOne, hostOne,
                                            hostThreads, hostOne, hostOne);
      OpBuilder stage = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value row = launch.getBlockIds().x;
      Value thread = launch.getThreadIds().x;
      Value cParts = stage.create<arith::ConstantIndexOp>(loc, parts);
      Value cThreads = stage.create<arith::ConstantIndexOp>(loc, T);
      Value identity = stage.create<arith::ConstantOp>(
          loc, stage.getF32FloatAttr(info.ident));
      auto loop = stage.create<scf::ForOp>(loc, thread, cParts, cThreads,
                                           ValueRange{identity});
      OpBuilder body = OpBuilder::atBlockBegin(loop.getBody());
      Value partial = body.create<memref::LoadOp>(
          loc, partials, ValueRange{row, loop.getInductionVar()});
      Value next = combineScalar(body, loc, info.op,
                                 loop.getRegionIterArg(0), partial);
      body.create<scf::YieldOp>(loc, next);
      auto blockReduce = stage.create<gpu::AllReduceOp>(
          loc, loop.getResult(0),
          gpu::AllReduceOperationAttr::get(ctx, info.op),
          /*uniform=*/UnitAttr::get(ctx));
      Value zero = stage.create<arith::ConstantIndexOp>(loc, 0);
      Value lead = stage.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, thread, zero);
      auto finalize = stage.create<scf::IfOp>(loc, lead, false);
      OpBuilder thenBuilder = finalize.getThenBodyBuilder();
      Value value = blockReduce.getResult();
      if (info.isMean) {
        Value divisor = thenBuilder.create<arith::ConstantOp>(
            loc, thenBuilder.getF32FloatAttr(static_cast<double>(N)));
        value = finalizeMean(thenBuilder, loc, value, divisor);
      }
      thenBuilder.create<memref::StoreOp>(loc, value, statistics,
                                          ValueRange{row});
      resetDynamicEmissionState();
      rowIndex = row;
      if (failed(bindUniformInputs(thenBuilder, loc, row, srcBlock,
                                   launchScalars)))
        return failure();
      uniform[reduction.getResult()] = value;
      hoistUniforms(thenBuilder, loc);
      DenseMap<Value, Value> current;
      for (unsigned index = 0; index < outRoles.size(); ++index) {
        if (outRoles[index] != ResultRole::PerRowOut)
          continue;
        Value output = emitAtCol(thenBuilder, loc, outputVals[index], zero, row,
                                 current, /*fromSmem=*/false);
        if (!output)
          return failure();
        thenBuilder.create<memref::StoreOp>(
            loc, output, outputMemrefs[index], ValueRange{row});
      }
      stage.create<gpu::TerminatorOp>(loc);
    }

    bool hasRowOutput = llvm::is_contained(outRoles, ResultRole::RowOut);
    int64_t stageCount = 2;
    if (hasRowOutput) {
      ++stageCount;
      int64_t totalElements = M * N;
      int64_t epilogueBlocks = (totalElements + T - 1) / T;
      Value hostEpilogueGrid =
          b.create<arith::ConstantIndexOp>(loc, epilogueBlocks);
      auto launch = b.create<gpu::LaunchOp>(loc, hostEpilogueGrid, hostOne,
                                            hostOne, hostThreads, hostOne,
                                            hostOne);
      OpBuilder stage = OpBuilder::atBlockBegin(&launch.getBody().front());
      Value block = launch.getBlockIds().x;
      Value thread = launch.getThreadIds().x;
      Value cThreads = stage.create<arith::ConstantIndexOp>(loc, T);
      Value linearBase = stage.create<arith::MulIOp>(loc, block, cThreads);
      Value linear = stage.create<arith::AddIOp>(loc, linearBase, thread);
      Value total = stage.create<arith::ConstantIndexOp>(loc, totalElements);
      Value active = stage.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, linear, total);
      auto guarded = stage.create<scf::IfOp>(loc, active, false);
      OpBuilder thenBuilder = guarded.getThenBodyBuilder();
      Value cN = thenBuilder.create<arith::ConstantIndexOp>(loc, N);
      Value row = thenBuilder.create<arith::DivUIOp>(loc, linear, cN);
      Value column = thenBuilder.create<arith::RemUIOp>(loc, linear, cN);
      resetDynamicEmissionState();
      rowIndex = row;
      if (failed(bindUniformInputs(thenBuilder, loc, row, srcBlock,
                                   launchScalars)))
        return failure();
      uniform[reduction.getResult()] = thenBuilder.create<memref::LoadOp>(
          loc, statistics, ValueRange{row});
      hoistUniforms(thenBuilder, loc);
      DenseMap<Value, Value> current;
      for (unsigned index = 0; index < outRoles.size(); ++index) {
        if (outRoles[index] != ResultRole::RowOut)
          continue;
        Value output = emitAtCol(thenBuilder, loc, outputVals[index], column,
                                 row, current, /*fromSmem=*/false);
        if (!output)
          return failure();
        thenBuilder.create<memref::StoreOp>(
            loc, output, outputMemrefs[index], ValueRange{row, column});
      }
      stage.create<gpu::TerminatorOp>(loc);
    }

    b.create<func::ReturnOp>(loc);
    attachConcreteStateReceipt(
        gpuFn, T, predicatedValueIfCount, foldedConstexprSelectCount,
        /*rowsPerBlock=*/1, emittedApproximateDivCount,
        static_cast<unsigned>(launchScalars.size()),
        /*epilogueLoadMotionCount=*/0,
        /*materializedSharedRowOutput=*/false,
        /*elidedSharedRowMaterialization=*/false,
        "multi_stage_partial_finalize",
        stageCount, parts);
    gpuFn->setAttr("intent.backend.grid_blocks",
                   b.getI64IntegerAttr(M * parts));
    fn.erase();
    return success();
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

    // New signature: original args as memrefs, followed only by fresh results.
    // An `intent.inplace` result reuses its aliased input memref directly.
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
    // Dynamic shared memory: one [N] f32 row cache per frontier.  Production
    // thread geometries additionally use one compact [numWarps] slot per
    // reduction; sub-warp test geometries retain the old [T] tree scratch.
    // We do NOT use gpu.all_reduce here: its lowering emits a separate static
    // `.shared` buffer that ptxas may pack inside our demoted dynamic-shared
    // cache region, clobbering it.
    bool useTiledWarpHierarchy = T >= 32 && (T % 32) == 0;
    int64_t tiledNumWarps = useTiledWarpHierarchy ? T / 32 : 0;
    int64_t cacheBytes = (int64_t)N * (int64_t)frontiers.size() * 4;
    int64_t reductionScratchBytes =
        useTiledWarpHierarchy
            ? (tiledNumWarps > 1
                   ? static_cast<int64_t>(reduces.size()) * tiledNumWarps * 4
                   : 0)
            : T * 4;
    int64_t shBytes = cacheBytes + reductionScratchBytes;
    sched.smemBytes = shBytes; // M6: record the tiled path's dynamic-shared plan
    attachLaunchPlan(gpuFn, sched);
    Value shBytesV =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shBytes));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesV);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    rowIndex = bx;

    // Carve the dynamic shared region into one memref<Nxf32, workgroup> per
    // frontier at byte offset f*N*4, then compact reduction scratch right after
    // the caches.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto bufTy = MemRefType::get({N}, f32, nullptr, wgSpace);
    for (size_t f = 0; f < frontiers.size(); ++f) {
      Value off = b.create<arith::ConstantIndexOp>(loc, (int64_t)f * N * 4);
      Value buf = b.create<memref::ViewOp>(loc, bufTy, raw, off, ValueRange{});
      frontierBuf[frontiers[f]] = buf;
    }
    Value treeScratch;
    SmallVector<Value> warpScratch;
    if (useTiledWarpHierarchy) {
      if (tiledNumWarps > 1) {
        auto scratchTy =
            MemRefType::get({tiledNumWarps}, f32, nullptr, wgSpace);
        int64_t scratchSlotBytes = tiledNumWarps * 4;
        for (size_t index = 0; index < reduces.size(); ++index) {
          Value offset = b.create<arith::ConstantIndexOp>(
              loc, cacheBytes +
                       static_cast<int64_t>(index) * scratchSlotBytes);
          warpScratch.push_back(b.create<memref::ViewOp>(
              loc, scratchTy, raw, offset, ValueRange{}));
        }
      }
    } else {
      auto scratchTy = MemRefType::get({T}, f32, nullptr, wgSpace);
      Value scratchOff = b.create<arith::ConstantIndexOp>(loc, cacheBytes);
      treeScratch = b.create<memref::ViewOp>(loc, scratchTy, raw, scratchOff,
                                            ValueRange{});
    }

    // Scalar args -> uniform (loaded once). NO upfront register tile (the strip
    // is too large for registers — that is why we tile + cache).
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Scalar)
        uniform[arg] = b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
      else if (argRoles[i] == Role::RowInt || argRoles[i] == Role::PerRow)
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
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

    // Reduce one per-thread partial through the reusable power-of-two shared
    // tree.  The final barrier at every level also makes the row frontier cache
    // visible before a dependent reduction or the epilogue reads it.
    auto reduceBlockPartial = [&](Value partial,
                                  const ReduceInfo &info) -> Value {
      b.create<memref::StoreOp>(loc, partial, treeScratch, ValueRange{tx});
      b.create<gpu::BarrierOp>(loc);
      for (int64_t stride = T / 2; stride >= 1; stride >>= 1) {
        Value cStride = b.create<arith::ConstantIndexOp>(loc, stride);
        Value active = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, tx, cStride);
        auto ifOp =
            b.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(ifOp.thenBlock());
          Value a =
              b.create<memref::LoadOp>(loc, treeScratch, ValueRange{tx});
          Value other = b.create<arith::AddIOp>(loc, tx, cStride);
          Value bb =
              b.create<memref::LoadOp>(loc, treeScratch, ValueRange{other});
          Value combined = combineScalar(b, loc, info.op, a, bb);
          b.create<memref::StoreOp>(loc, combined, treeScratch,
                                    ValueRange{tx});
        }
        b.create<gpu::BarrierOp>(loc);
      }
      Value c0r = b.create<arith::ConstantIndexOp>(loc, 0);
      Value scalar =
          b.create<memref::LoadOp>(loc, treeScratch, ValueRange{c0r});
      return info.isMean ? finalizeMean(b, loc, scalar, nF) : scalar;
    };

    // Partition reductions by data dependency, exactly as the register path
    // does.  Every independent group shares one tiled feature traversal and one
    // per-column expression memo table.  The first group computes and stores the
    // frontier from global inputs; later groups consume that workgroup cache, so
    // a dependent mean/variance chain still reads every global row only once.
    bool tiledEmissionFailed = false;
    bool populateFrontier = true;
    Value tiledLane;
    Value tiledWarpId;
    Value tiledLaneZero;
    if (useTiledWarpHierarchy) {
      Value txI32 =
          b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
      tiledLane = b.create<arith::AndIOp>(
          loc, txI32, b.create<arith::ConstantIntOp>(loc, 31, 32));
      if (tiledNumWarps > 1) {
        tiledWarpId = b.create<arith::DivUIOp>(
            loc, tx, b.create<arith::ConstantIndexOp>(loc, 32));
        tiledLaneZero = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, tiledLane,
            b.create<arith::ConstantIntOp>(loc, 0, 32));
      }
    }
    size_t groupStart = 0;
    while (groupStart < reduces.size()) {
      size_t groupEnd = groupStart + 1;
      while (groupEnd < reduces.size()) {
        bool dependent = false;
        for (size_t prior = groupStart; prior < groupEnd; ++prior)
          dependent |= dependsOn(reduces[groupEnd].getInput(),
                                 reduces[prior].getResult());
        if (dependent)
          break;
        ++groupEnd;
      }

      SmallVector<Value> identities;
      identities.reserve(groupEnd - groupStart);
      for (size_t index = groupStart; index < groupEnd; ++index) {
        ReduceInfo info = *reduceInfoFor(reduces[index].getFn());
        identities.push_back(b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(info.ident)));
      }
      auto forOp = b.create<scf::ForOp>(
          loc, c0i, cNT, c1, identities,
          [&](OpBuilder &nb, Location nl, Value kt, ValueRange iter) {
            Value ktOff = nb.create<arith::MulIOp>(nl, kt, cKptT);
            SmallVector<Value> accumulators(iter.begin(), iter.end());
            for (int64_t k = 0; k < Kpt; ++k) {
              Value col = nb.create<arith::AddIOp>(nl, baseCol[k], ktOff);
              DenseMap<Value, Value> cur;
              if (populateFrontier) {
                for (Value frontier : frontiers) {
                  Value value = emitAtCol(nb, nl, frontier, col, bx, cur,
                                          /*fromSmem=*/false);
                  if (!value) {
                    tiledEmissionFailed = true;
                    value = nb.create<arith::ConstantOp>(
                        nl, nb.getF32FloatAttr(0.0));
                  }
                  nb.create<memref::StoreOp>(nl, value, frontierBuf[frontier],
                                             ValueRange{col});
                }
              }
              for (size_t index = groupStart; index < groupEnd; ++index) {
                ReduceOp red = reduces[index];
                ReduceInfo info = *reduceInfoFor(red.getFn());
                Value input = emitAtCol(nb, nl, red.getInput(), col, bx, cur,
                                        /*fromSmem=*/!populateFrontier);
                if (!input) {
                  tiledEmissionFailed = true;
                  input = nb.create<arith::ConstantOp>(
                      nl, nb.getF32FloatAttr(info.ident));
                }
                size_t localIndex = index - groupStart;
                accumulators[localIndex] =
                    combineScalar(nb, nl, info.op,
                                  accumulators[localIndex], input);
              }
            }
            nb.create<scf::YieldOp>(nl, accumulators);
          });
      if (tiledEmissionFailed) {
        gpuFn.erase();
        fn.setName(name);
        return failure();
      }

      if (useTiledWarpHierarchy) {
        // Fold each per-thread partial in registers.  Independent reductions
        // publish disjoint warp totals before one group barrier; every warp then
        // redundantly folds the compact partial vector, yielding a block scalar
        // in every lane without a second barrier.
        SmallVector<Value> warpPartials;
        warpPartials.reserve(groupEnd - groupStart);
        for (size_t index = groupStart; index < groupEnd; ++index) {
          ReduceInfo info = *reduceInfoFor(reduces[index].getFn());
          Value warpPartial = forOp.getResult(index - groupStart);
          for (int32_t delta = 1; delta < 32; delta <<= 1) {
            Value other = b.create<gpu::ShuffleOp>(
                                loc, warpPartial, delta, 32,
                                gpu::ShuffleMode::XOR)
                              .getShuffleResult();
            warpPartial =
                combineScalar(b, loc, info.op, warpPartial, other);
          }
          if (tiledNumWarps > 1) {
            auto publish = b.create<scf::IfOp>(
                loc, tiledLaneZero, /*withElseRegion=*/false);
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(publish.thenBlock());
            b.create<memref::StoreOp>(loc, warpPartial, warpScratch[index],
                                      ValueRange{tiledWarpId});
          }
          warpPartials.push_back(warpPartial);
        }
        // This is also the frontier-cache visibility fence for a dependent
        // reduction group and the post-reduction epilogue.
        b.create<gpu::BarrierOp>(loc);
        for (size_t index = groupStart; index < groupEnd; ++index) {
          ReduceOp red = reduces[index];
          ReduceInfo info = *reduceInfoFor(red.getFn());
          Value scalar = warpPartials[index - groupStart];
          if (tiledNumWarps > 1) {
            Value activeLane = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, tiledLane,
                b.create<arith::ConstantIntOp>(loc, tiledNumWarps, 32));
            auto loadWarp = b.create<scf::IfOp>(
                loc, TypeRange{f32}, activeLane, /*withElseRegion=*/true);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(loadWarp.thenBlock());
              Value laneIndex = b.create<arith::IndexCastOp>(
                  loc, b.getIndexType(), tiledLane);
              Value partial = b.create<memref::LoadOp>(
                  loc, warpScratch[index], ValueRange{laneIndex});
              b.create<scf::YieldOp>(loc, partial);
              b.setInsertionPointToStart(loadWarp.elseBlock());
              Value identity = b.create<arith::ConstantOp>(
                  loc, b.getF32FloatAttr(info.ident));
              b.create<scf::YieldOp>(loc, identity);
            }
            scalar = loadWarp.getResult(0);
            for (int32_t delta = 1; delta < 32; delta <<= 1) {
              Value other = b.create<gpu::ShuffleOp>(
                                  loc, scalar, delta, 32,
                                  gpu::ShuffleMode::XOR)
                                .getShuffleResult();
              scalar = combineScalar(b, loc, info.op, scalar, other);
            }
          }
          if (info.isMean)
            scalar = finalizeMean(b, loc, scalar, nF);
          uniform[red.getResult()] = scalar;
          hoistUniforms(b, loc);
        }
      } else {
        for (size_t index = groupStart; index < groupEnd; ++index) {
          ReduceOp red = reduces[index];
          ReduceInfo info = *reduceInfoFor(red.getFn());
          Value scalar =
              reduceBlockPartial(forOp.getResult(index - groupStart), info);
          uniform[red.getResult()] = scalar;
          hoistUniforms(b, loc);
        }
      }
      populateFrontier = false;
      groupStart = groupEnd;
    }

    // The final reduction-group barrier above is the cache->epilogue fence.  It
    // guarantees that every input/frontier read finishes before an in-place row
    // output can overwrite its aliased input, so no additional full-block fence
    // is required here.

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
        if (!s) {
          tiledEmissionFailed = true;
          s = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
        }
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
                if (!o) {
                  tiledEmissionFailed = true;
                  o = nb.create<arith::ConstantOp>(
                      nl, nb.getF32FloatAttr(0.0));
                }
                nb.create<memref::StoreOp>(nl, o, outMems[j],
                                           ValueRange{bx, col});
              }
            }
            nb.create<scf::YieldOp>(nl);
          });
    }

    if (tiledEmissionFailed) {
      gpuFn.erase();
      fn.setName(name);
      return failure();
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    StringRef tiledTopology =
        useTiledWarpHierarchy
            ? (reduces.size() > 1 ? "tiled_frontier_warp_multi_reduce"
                                  : "tiled_frontier_warp")
            : (reduces.size() > 1 ? "tiled_shared_multi_reduce"
                                  : "tiled_shared");
    attachConcreteStateReceipt(gpuFn, T, predicatedValueIfCount,
                               foldedConstexprSelectCount, 1,
                               emittedApproximateDivCount, 0, 0,
                               /*materializedSharedRowOutput=*/false,
                               /*elidedSharedRowMaterialization=*/false,
                               tiledTopology);
    fn.erase();
    return success();
  }

  // ---- B-certified sorted sparsemax prefix path -------------------------- //
  //
  // A descending row has a prefix support set for
  //
  //   z[r] > (cumsum(z)[r] - 1) / (r + 1).
  //
  // Process one coalesced float4 per thread at a time (4*T sorted elements per
  // chunk) and stop reading SortedX after the first chunk whose last element is
  // outside the support.  The common captured distribution normally stops
  // after the first 1024 elements at T=256, while a worst-case all-supported
  // row simply continues through all chunks.  X is read only after tau is
  // known, in a separately coalesced output pass.  Thus the optimization is
  // exact for the B-certified domain and removes most SortedX HBM traffic on
  // the common path rather than merely retuning the same three full-row passes.
  LogicalResult rewriteSortedSparsemax(unsigned xArgIndex,
                                       unsigned sortedArgIndex) {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Type i32 = b.getI32Type();
    Type f32Ty = b.getF32Type();
    Type vec4Ty = VectorType::get({4}, f32Ty);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    MemRefType rowVecMemref = MemRefType::get({M, N / 4}, vec4Ty);

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto newFty = FunctionType::get(
        ctx, {rowVecMemref, rowVecMemref, rowVecMemref}, {});
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    gpuFn->setAttr("intent_sorted_descending", b.getBoolAttr(true));
    gpuFn->setAttr("intent.fast_path_id",
                   b.getStringAttr("cuda.row_scan.sparsemax.sorted_prefix.v1"));
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value xMem = entry->getArgument(xArgIndex);
    Value sortedMem = entry->getArgument(sortedArgIndex);
    Value outMem = entry->getArgument(2);
    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    int64_t numWarps = T / 32;
    int64_t scratchSlotBytes = numWarps * 4;
    int64_t shBytes = scratchSlotBytes * 3;
    sched.smemBytes = shBytes;
    sched.vectorWidth = 4;
    attachLaunchPlan(gpuFn, sched);
    Value shBytesValue =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(shBytes));
    auto launch =
        b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1, shBytesValue);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    auto rawType = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                   nullptr, wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawType);
    auto warpF32Type = MemRefType::get({numWarps}, f32Ty, nullptr, wgSpace);
    auto warpI32Type = MemRefType::get({numWarps}, i32, nullptr, wgSpace);
    Value c0Index = b.create<arith::ConstantIndexOp>(loc, 0);
    Value scanScratch = b.create<memref::ViewOp>(
        loc, warpF32Type, raw, c0Index, ValueRange{});
    Value rankOffset =
        b.create<arith::ConstantIndexOp>(loc, scratchSlotBytes);
    Value rankScratch = b.create<memref::ViewOp>(
        loc, warpI32Type, raw, rankOffset, ValueRange{});
    Value sumOffset =
        b.create<arith::ConstantIndexOp>(loc, scratchSlotBytes * 2);
    Value sumScratch = b.create<memref::ViewOp>(
        loc, warpF32Type, raw, sumOffset, ValueRange{});

    Value zeroF =
        b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    Value oneF =
        b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0));
    Value zeroI = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value falseValue = b.create<arith::ConstantIntOp>(loc, 0, 1);
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cChunks =
        b.create<arith::ConstantIndexOp>(loc, N / (T * 4));

    // Loop state: accumulated sum of fully-supported preceding chunks, the
    // best support rank/sum seen so far, and a uniform done flag.  Completed
    // rows take the empty branch on later static iterations; no loads/barriers
    // occur there.
    auto chunkLoop = b.create<scf::ForOp>(
        loc, c0, cChunks, c1,
        ValueRange{zeroF, zeroI, zeroF, falseValue});
    {
      OpBuilder lb(chunkLoop.getBody(), chunkLoop.getBody()->begin());
      Value chunk = chunkLoop.getInductionVar();
      Value carry = chunkLoop.getRegionIterArg(0);
      Value previousRank = chunkLoop.getRegionIterArg(1);
      Value previousSum = chunkLoop.getRegionIterArg(2);
      Value done = chunkLoop.getRegionIterArg(3);
      Value trueValue = lb.create<arith::ConstantIntOp>(loc, 1, 1);
      Value active = lb.create<arith::XOrIOp>(loc, done, trueValue);
      auto activeIf = lb.create<scf::IfOp>(
          loc, TypeRange{f32Ty, i32, f32Ty, b.getI1Type()}, active,
          /*withElseRegion=*/true);

      {
        OpBuilder tb(activeIf.thenBlock(), activeIf.thenBlock()->begin());
        Value chunkBase = tb.create<arith::MulIOp>(loc, chunk, cT);
        Value vecIndex = tb.create<arith::AddIOp>(loc, chunkBase, tx);
        auto sortedLoad = tb.create<memref::LoadOp>(
            loc, sortedMem, ValueRange{bx, vecIndex});
        sortedLoad.setNontemporal(true);
        Value sortedVector = sortedLoad;
        SmallVector<Value> z(4);
        for (int64_t laneIndex = 0; laneIndex < 4; ++laneIndex)
          z[laneIndex] = tb.create<vector::ExtractOp>(
              loc, sortedVector, ArrayRef<int64_t>{laneIndex});
        SmallVector<Value> localPrefix(4);
        localPrefix[0] = z[0];
        for (int64_t laneIndex = 1; laneIndex < 4; ++laneIndex)
          localPrefix[laneIndex] = tb.create<arith::AddFOp>(
              loc, localPrefix[laneIndex - 1], z[laneIndex]);
        Value segmentTotal = localPrefix[3];

        Value txI32 = tb.create<arith::IndexCastOp>(loc, i32, tx);
        Value lane = tb.create<arith::AndIOp>(
            loc, txI32, tb.create<arith::ConstantIntOp>(loc, 31, 32));
        Value c32Index = tb.create<arith::ConstantIndexOp>(loc, 32);
        Value warpId = tb.create<arith::DivUIOp>(loc, tx, c32Index);
        Value warpInclusive = segmentTotal;
        for (int32_t offset = 1; offset < 32; offset <<= 1) {
          Value prior = tb.create<gpu::ShuffleOp>(
                              loc, warpInclusive, offset, 32,
                              gpu::ShuffleMode::UP)
                            .getShuffleResult();
          Value participates = tb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, lane,
              tb.create<arith::ConstantIntOp>(loc, offset, 32));
          Value combined =
              tb.create<arith::AddFOp>(loc, warpInclusive, prior);
          warpInclusive = tb.create<arith::SelectOp>(
              loc, participates, combined, warpInclusive);
        }
        Value lastLane = tb.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, lane,
            tb.create<arith::ConstantIntOp>(loc, 31, 32));
        auto publishScan =
            tb.create<scf::IfOp>(loc, lastLane, /*withElseRegion=*/false);
        {
          OpBuilder pb(publishScan.thenBlock(),
                       publishScan.thenBlock()->begin());
          pb.create<memref::StoreOp>(loc, warpInclusive, scanScratch,
                                     ValueRange{warpId});
        }
        tb.create<gpu::BarrierOp>(loc);

        Value activeWarpLane = tb.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, lane,
            tb.create<arith::ConstantIntOp>(loc, numWarps, 32));
        auto loadWarp = tb.create<scf::IfOp>(
            loc, TypeRange{f32Ty}, activeWarpLane,
            /*withElseRegion=*/true);
        {
          OpBuilder wb(loadWarp.thenBlock(), loadWarp.thenBlock()->begin());
          Value laneIndex =
              wb.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
          Value value = wb.create<memref::LoadOp>(
              loc, scanScratch, ValueRange{laneIndex});
          wb.create<scf::YieldOp>(loc, ValueRange{value});
          OpBuilder eb(loadWarp.elseBlock(), loadWarp.elseBlock()->begin());
          eb.create<scf::YieldOp>(loc, ValueRange{zeroF});
        }
        Value ownWarpTotal = loadWarp.getResult(0);
        Value warpPrefix = ownWarpTotal;
        for (int32_t offset = 1; offset < 32; offset <<= 1) {
          Value prior = tb.create<gpu::ShuffleOp>(
                              loc, warpPrefix, offset, 32,
                              gpu::ShuffleMode::UP)
                            .getShuffleResult();
          Value participates = tb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, lane,
              tb.create<arith::ConstantIntOp>(loc, offset, 32));
          Value combined = tb.create<arith::AddFOp>(loc, warpPrefix, prior);
          warpPrefix = tb.create<arith::SelectOp>(
              loc, participates, combined, warpPrefix);
        }
        Value warpExclusive =
            tb.create<arith::SubFOp>(loc, warpPrefix, ownWarpTotal);
        Value warpIdI32 = tb.create<arith::IndexCastOp>(loc, i32, warpId);
        Value width32 = tb.create<arith::ConstantIntOp>(loc, 32, 32);
        Value warpBase = tb.create<gpu::ShuffleOp>(
                               loc, warpExclusive, warpIdI32, width32,
                               gpu::ShuffleMode::IDX)
                             .getShuffleResult();
        Value lastWarpLane =
            tb.create<arith::ConstantIntOp>(loc, numWarps - 1, 32);
        Value chunkTotal = tb.create<gpu::ShuffleOp>(
                                loc, warpPrefix, lastWarpLane, width32,
                                gpu::ShuffleMode::IDX)
                              .getShuffleResult();
        Value laneExclusive =
            tb.create<arith::SubFOp>(loc, warpInclusive, segmentTotal);
        Value prefixBase =
            tb.create<arith::AddFOp>(loc, carry, warpBase);
        prefixBase =
            tb.create<arith::AddFOp>(loc, prefixBase, laneExclusive);

        Value scalarBase = tb.create<arith::MulIOp>(
            loc, vecIndex, tb.create<arith::ConstantIndexOp>(loc, 4));
        Value threadBestRank = zeroI;
        Value threadBestSum = zeroF;
        for (int64_t laneIndex = 0; laneIndex < 4; ++laneIndex) {
          Value prefix = tb.create<arith::AddFOp>(
              loc, prefixBase, localPrefix[laneIndex]);
          Value scalarIndex = tb.create<arith::AddIOp>(
              loc, scalarBase,
              tb.create<arith::ConstantIndexOp>(loc, laneIndex));
          Value rank = tb.create<arith::AddIOp>(
              loc, tb.create<arith::IndexCastOp>(loc, i32, scalarIndex),
              tb.create<arith::ConstantIntOp>(loc, 1, 32));
          Value rankF = tb.create<arith::SIToFPOp>(loc, f32Ty, rank);
          Value thresholdNumerator =
              tb.create<arith::SubFOp>(loc, prefix, oneF);
          Value threshold =
              tb.create<arith::DivFOp>(loc, thresholdNumerator, rankF);
          Value supported = tb.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OGT, z[laneIndex], threshold);
          threadBestRank = tb.create<arith::SelectOp>(
              loc, supported, rank, threadBestRank);
          threadBestSum = tb.create<arith::SelectOp>(
              loc, supported, prefix, threadBestSum);
        }

        // Pair all-reduce by maximum supported rank.  The selected prefix sum
        // travels with its rank, first within each warp and then through two
        // compact shared arrays across warps.
        for (int32_t delta = 1; delta < 32; delta <<= 1) {
          Value otherRank = tb.create<gpu::ShuffleOp>(
                                  loc, threadBestRank, delta, 32,
                                  gpu::ShuffleMode::XOR)
                                .getShuffleResult();
          Value otherSum = tb.create<gpu::ShuffleOp>(
                                 loc, threadBestSum, delta, 32,
                                 gpu::ShuffleMode::XOR)
                               .getShuffleResult();
          Value takeOther = tb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, otherRank, threadBestRank);
          threadBestRank = tb.create<arith::SelectOp>(
              loc, takeOther, otherRank, threadBestRank);
          threadBestSum = tb.create<arith::SelectOp>(
              loc, takeOther, otherSum, threadBestSum);
        }
        Value laneZero = tb.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, lane,
            tb.create<arith::ConstantIntOp>(loc, 0, 32));
        auto publishPair =
            tb.create<scf::IfOp>(loc, laneZero, /*withElseRegion=*/false);
        {
          OpBuilder pb(publishPair.thenBlock(),
                       publishPair.thenBlock()->begin());
          pb.create<memref::StoreOp>(loc, threadBestRank, rankScratch,
                                     ValueRange{warpId});
          pb.create<memref::StoreOp>(loc, threadBestSum, sumScratch,
                                     ValueRange{warpId});
        }
        tb.create<gpu::BarrierOp>(loc);

        auto loadRank = tb.create<scf::IfOp>(
            loc, TypeRange{i32}, activeWarpLane,
            /*withElseRegion=*/true);
        {
          OpBuilder rb(loadRank.thenBlock(), loadRank.thenBlock()->begin());
          Value laneIndex =
              rb.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
          Value value = rb.create<memref::LoadOp>(
              loc, rankScratch, ValueRange{laneIndex});
          rb.create<scf::YieldOp>(loc, ValueRange{value});
          OpBuilder eb(loadRank.elseBlock(), loadRank.elseBlock()->begin());
          eb.create<scf::YieldOp>(loc, ValueRange{zeroI});
        }
        auto loadSum = tb.create<scf::IfOp>(
            loc, TypeRange{f32Ty}, activeWarpLane,
            /*withElseRegion=*/true);
        {
          OpBuilder sb(loadSum.thenBlock(), loadSum.thenBlock()->begin());
          Value laneIndex =
              sb.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
          Value value = sb.create<memref::LoadOp>(
              loc, sumScratch, ValueRange{laneIndex});
          sb.create<scf::YieldOp>(loc, ValueRange{value});
          OpBuilder eb(loadSum.elseBlock(), loadSum.elseBlock()->begin());
          eb.create<scf::YieldOp>(loc, ValueRange{zeroF});
        }
        Value chunkBestRank = loadRank.getResult(0);
        Value chunkBestSum = loadSum.getResult(0);
        for (int32_t delta = 1; delta < 32; delta <<= 1) {
          Value otherRank = tb.create<gpu::ShuffleOp>(
                                  loc, chunkBestRank, delta, 32,
                                  gpu::ShuffleMode::XOR)
                                .getShuffleResult();
          Value otherSum = tb.create<gpu::ShuffleOp>(
                                 loc, chunkBestSum, delta, 32,
                                 gpu::ShuffleMode::XOR)
                               .getShuffleResult();
          Value takeOther = tb.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, otherRank, chunkBestRank);
          chunkBestRank = tb.create<arith::SelectOp>(
              loc, takeOther, otherRank, chunkBestRank);
          chunkBestSum = tb.create<arith::SelectOp>(
              loc, takeOther, otherSum, chunkBestSum);
        }

        Value takeChunk = tb.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, chunkBestRank, previousRank);
        Value nextBestRank = tb.create<arith::SelectOp>(
            loc, takeChunk, chunkBestRank, previousRank);
        Value nextBestSum = tb.create<arith::SelectOp>(
            loc, takeChunk, chunkBestSum, previousSum);
        Value chunkPlusOne = tb.create<arith::AddIOp>(loc, chunk, c1);
        Value chunkEndIndex = tb.create<arith::MulIOp>(
            loc, chunkPlusOne,
            tb.create<arith::ConstantIndexOp>(loc, T * 4));
        Value chunkEnd =
            tb.create<arith::IndexCastOp>(loc, i32, chunkEndIndex);
        Value nextDone = tb.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, chunkBestRank, chunkEnd);
        Value nextCarry =
            tb.create<arith::AddFOp>(loc, carry, chunkTotal);
        tb.create<scf::YieldOp>(
            loc, ValueRange{nextCarry, nextBestRank, nextBestSum, nextDone});
      }
      {
        OpBuilder eb(activeIf.elseBlock(), activeIf.elseBlock()->begin());
        eb.create<scf::YieldOp>(
            loc, ValueRange{carry, previousRank, previousSum, done});
      }
      lb.create<scf::YieldOp>(loc, activeIf.getResults());
    }

    b.setInsertionPointAfter(chunkLoop);
    Value bestRank = chunkLoop.getResult(1);
    Value bestSum = chunkLoop.getResult(2);
    Value clampedRank = b.create<arith::MaxSIOp>(
        loc, bestRank, b.create<arith::ConstantIntOp>(loc, 1, 32));
    Value rankF = b.create<arith::SIToFPOp>(loc, f32Ty, clampedRank);
    Value tauNumerator = b.create<arith::SubFOp>(loc, bestSum, oneF);
    Value tau = b.create<arith::DivFOp>(loc, tauNumerator, rankF);

    // Coalesced X -> O pass.  Adjacent lanes own adjacent float4 vectors.
    int64_t outputVecTiles = N / (T * 4);
    for (int64_t tile = 0; tile < outputVecTiles; ++tile) {
      Value vecIndex = b.create<arith::AddIOp>(
          loc, tx, b.create<arith::ConstantIndexOp>(loc, tile * T));
      auto xLoad =
          b.create<memref::LoadOp>(loc, xMem, ValueRange{bx, vecIndex});
      xLoad.setNontemporal(true);
      Value xVector = xLoad;
      SmallVector<Value> outputs(4);
      for (int64_t laneIndex = 0; laneIndex < 4; ++laneIndex) {
        Value x = b.create<vector::ExtractOp>(
            loc, xVector, ArrayRef<int64_t>{laneIndex});
        Value shifted = b.create<arith::SubFOp>(loc, x, tau);
        outputs[laneIndex] =
            b.create<arith::MaxNumFOp>(loc, shifted, zeroF);
      }
      Value outputVector =
          b.create<vector::FromElementsOp>(loc, vec4Ty, outputs);
      auto store = b.create<memref::StoreOp>(
          loc, outputVector, outMem, ValueRange{bx, vecIndex});
      store.setNontemporal(true);
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachConcreteStateReceipt(gpuFn, T, predicatedValueIfCount,
                               foldedConstexprSelectCount, 1,
                               emittedApproximateDivCount);
    fn.erase();
    return success();
  }

  // ---- scan path (sparsemax fwd: block-cooperative inclusive cumsum) ------ //
  // One block per row, T threads, K = N/T CONTIGUOUS columns per thread
  // (col = tx*K + k, NOT the interleaved tx + k*T) so a thread's values form a
  // contiguous segment of the row. The full-row inclusive cumsum is:
  //   (1) each thread sequential-prefixes its K contiguous values, keeping the
  //       segment total;
  //   (2) warp shuffles scan segment totals; one compact per-warp prefix is
  //       exchanged through dynamic shared memory;
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
    Type vec4Ty = VectorType::get({4}, f32);
    MemRefType rowVecMemref = MemRefType::get({M, N / 4}, vec4Ty);

    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (argRoles[i] == Role::Row)
        inTys.push_back(
            useVec4 ? rowVecMemref : MemRefType::get({M, N}, f32));
      else
        inTys.push_back(memrefOf(i));
    }
    for (ResultRole rr : outRoles) {
      if (rr == ResultRole::RowOut)
        inTys.push_back(useVec4 ? rowVecMemref : MemRefType::get({M, N}, f32));
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
    // One disjoint compact [numWarps] slot per scan/reduce.  Separating phases
    // removes the old end-of-phase fence: a following reduction cannot clobber
    // a prefix another warp has not read yet because it writes another slot.
    int64_t blockWarps = (T + 31) / 32;
    // A sub-warp launch cannot rely on the target's full-warp shuffle active
    // mask semantics. Give those scans one scalar slot per participating
    // thread; T>=32 keeps the compact one-slot-per-warp layout.
    int64_t scratchElements = T < 32 ? T : blockWarps;
    int64_t scratchSlots = (int64_t)scans.size() + (int64_t)reduces.size();
    int64_t scratchSlotBytes = scratchElements * 4;
    int64_t shBytes = scratchSlots * scratchSlotBytes;
    sched.smemBytes = shBytes; // M6: record the scan path's dynamic-shared plan
    attachLaunchPlan(gpuFn, sched);
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
    auto scratchTy = MemRefType::get({scratchElements}, f32, nullptr, wgSpace);
    auto scratchI32Ty =
        MemRefType::get({scratchElements}, b.getI32Type(), nullptr, wgSpace);
    SmallVector<Value> scanScratch;
    SmallVector<Value> reduceScratch;
    SmallVector<Value> reduceScratchI32;
    for (int64_t slot = 0; slot < scratchSlots; ++slot) {
      Value off =
          b.create<arith::ConstantIndexOp>(loc, slot * scratchSlotBytes);
      Value fview =
          b.create<memref::ViewOp>(loc, scratchTy, raw, off, ValueRange{});
      if (slot < (int64_t)scans.size()) {
        scanScratch.push_back(fview);
      } else {
        reduceScratch.push_back(fview);
        reduceScratchI32.push_back(b.create<memref::ViewOp>(
            loc, scratchI32Ty, raw, off, ValueRange{}));
      }
    }

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
    SmallVector<Value> scanVecIdx;
    if (useVec4) {
      scanVecIdx.resize(numVecTiles);
      Value cNumVec =
          b.create<arith::ConstantIndexOp>(loc, numVecTiles);
      Value vecBase = b.create<arith::MulIOp>(loc, tx, cNumVec);
      for (int64_t vt = 0; vt < numVecTiles; ++vt)
        scanVecIdx[vt] = b.create<arith::AddIOp>(
            loc, vecBase, b.create<arith::ConstantIndexOp>(loc, vt));
    }
    cols.clear();
    for (int64_t k = 0; k < K; ++k) {
      Value off = b.create<arith::ConstantIndexOp>(loc, k);
      cols.push_back(b.create<arith::AddIOp>(loc, base, off));
    }
    // Keep row inputs live only for the phase that consumes them.  In
    // sparsemax, SortedX feeds the scan/reductions while X is used only by the
    // final `max(X - tau, 0)` epilogue.  Loading X here would keep its complete
    // register tile live across the scan and both summaries, increasing
    // register pressure and memory-latency exposure.  Classify such
    // epilogue-only rows structurally from data dependencies and load them only
    // after the summaries have become uniform.  This is a general lifetime
    // optimization; it does not inspect the function or source-kernel name.
    SmallVector<Value> deferredRowArgs;
    auto neededBeforeSummary = [&](Value arg) {
      for (ScanOp scan : scans)
        if (dependsOn(scan.getInput(), arg))
          return true;
      for (ReduceOp reduce : reduces)
        if (dependsOn(reduce.getInput(), arg))
          return true;
      return false;
    };
    auto loadRowTile = [&](Value arg) {
      SmallVector<Value> vals(K);
      if (useVec4) {
        for (int64_t vt = 0; vt < numVecTiles; ++vt) {
          auto load = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{bx, scanVecIdx[vt]});
          load.setNontemporal(true);
          Value vec = load;
          for (int64_t lane = 0; lane < 4; ++lane)
            vals[vt * 4 + lane] = b.create<vector::ExtractOp>(
                loc, vec, ArrayRef<int64_t>{lane});
        }
      } else {
        for (int64_t k = 0; k < K; ++k)
          vals[k] = b.create<memref::LoadOp>(loc, argMemref[arg],
                                             ValueRange{bx, cols[k]});
      }
      rowVals[arg] = std::move(vals);
    };
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        if (neededBeforeSummary(arg))
          loadRowTile(arg);
        else
          deferredRowArgs.push_back(arg);
      } else if (argRoles[i] == Role::RowInt) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
      }
    }
    hoistUniforms(b, loc);

    // For each scan: build the full-row inclusive scan register tile. `fn`
    // selects the monoid combine (sum=+, max=max, min=min).  The cross-thread
    // prefix is hierarchical: warp shuffle scan of segment totals, one compact
    // scan of the per-warp totals, then broadcast the warp base.  This replaces
    // the old 3*log2(T)-barrier shared-memory Hillis-Steele implementation with
    // one block barrier regardless of T, matching the GPU execution model
    // used by high-performance Triton scans.
    for (size_t scanIndex = 0; scanIndex < scans.size(); ++scanIndex) {
      ScanOp sc = scans[scanIndex];
      Value phaseScratch = scanScratch[scanIndex];
      ReduceInfo sinfo = *reduceInfoFor(sc.getFn());
      bool isSum = sinfo.op == gpu::AllReduceOperation::ADD;
      // (1) per-thread sequential inclusive prefix over its K contiguous values.
      SmallVector<Value> local(K);
      Value running;
      for (int64_t k = 0; k < K; ++k) {
        Value v = emitAt(b, loc, sc.getInput(), k);
        running = (k == 0) ? v : combineScalar(b, loc, sinfo.op, running, v);
        local[k] = running;
      }
      Value segTotal = local[K - 1];
      Value identF =
          b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(sinfo.ident));

      // CUDA/NVVM shuffle lowering assumes a hardware warp participation
      // contract that is fragile for launches with fewer than 32 threads.
      // Small scans are important for routing/count metadata (8/16 experts),
      // so use a deterministic shared-memory exclusive prefix there. The
      // quadratic work is bounded by 31*31 scalar combines and is outside the
      // large-row performance path.
      if (T < 32) {
        b.create<memref::StoreOp>(loc, segTotal, phaseScratch,
                                  ValueRange{tx});
        b.create<gpu::BarrierOp>(loc);
        Value exclusive = identF;
        for (int64_t prior = 0; prior < T; ++prior) {
          Value priorIndex = b.create<arith::ConstantIndexOp>(loc, prior);
          Value priorTotal = b.create<memref::LoadOp>(
              loc, phaseScratch, ValueRange{priorIndex});
          Value include = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ugt, tx, priorIndex);
          Value contribution =
              b.create<arith::SelectOp>(loc, include, priorTotal, identF);
          exclusive =
              combineScalar(b, loc, sinfo.op, exclusive, contribution);
        }
        SmallVector<Value> vals(K);
        for (int64_t k = 0; k < K; ++k)
          vals[k] =
              combineScalar(b, loc, sinfo.op, exclusive, local[k]);
        scanVals[sc.getResult()] = vals;
        continue;
      }

      // (2a) inclusive scan inside each hardware warp.
      int64_t warpWidth = std::min<int64_t>(T, 32);
      int64_t numWarps = (T + 31) / 32;
      Value txI32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
      Value laneMask =
          b.create<arith::ConstantIntOp>(loc, warpWidth - 1, 32);
      Value lane = b.create<arith::AndIOp>(loc, txI32, laneMask);
      Value warpIncl = segTotal;
      for (int32_t off = 1; off < warpWidth; off <<= 1) {
        Value prev = b.create<gpu::ShuffleOp>(loc, warpIncl, off, warpWidth,
                                              gpu::ShuffleMode::UP)
                         .getShuffleResult();
        Value active = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, lane,
            b.create<arith::ConstantIntOp>(loc, off, 32));
        Value combined = combineScalar(b, loc, sinfo.op, warpIncl, prev);
        warpIncl = b.create<arith::SelectOp>(loc, active, combined, warpIncl);
      }

      Value warpBase = identF;
      if (numWarps > 1) {
        Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
        Value warpId = b.create<arith::DivUIOp>(loc, tx, c32);
        Value lastLane = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, lane,
            b.create<arith::ConstantIntOp>(loc, warpWidth - 1, 32));
        auto publish =
            b.create<scf::IfOp>(loc, lastLane, /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(publish.thenBlock());
          b.create<memref::StoreOp>(loc, warpIncl, phaseScratch,
                                    ValueRange{warpId});
        }
        b.create<gpu::BarrierOp>(loc);

        // (2b) every warp scans the same compact per-warp-total vector.  The
        // desired exclusive base is then an IDX shuffle from lane=warpId.  This
        // replication is only five shuffles and avoids a second block barrier.
        Value activeLane = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, lane,
            b.create<arith::ConstantIntOp>(loc, numWarps, 32));
        auto loadWarp = b.create<scf::IfOp>(
            loc, TypeRange{f32}, activeLane, /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(loadWarp.thenBlock());
          Value laneIdx =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
          Value total = b.create<memref::LoadOp>(loc, phaseScratch,
                                                  ValueRange{laneIdx});
          b.create<scf::YieldOp>(loc, ValueRange{total});
          b.setInsertionPointToStart(loadWarp.elseBlock());
          b.create<scf::YieldOp>(loc, identF);
        }
        Value warpPrefix = loadWarp.getResult(0);
        for (int32_t off = 1; off < 32; off <<= 1) {
          Value prev = b.create<gpu::ShuffleOp>(
                            loc, warpPrefix, off, 32, gpu::ShuffleMode::UP)
                           .getShuffleResult();
          Value ge = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, lane,
              b.create<arith::ConstantIntOp>(loc, off, 32));
          Value combined = combineScalar(b, loc, sinfo.op, warpPrefix, prev);
          warpPrefix = b.create<arith::SelectOp>(loc, ge, combined, warpPrefix);
        }
        Value warpExclusive;
        if (isSum) {
          warpExclusive =
              b.create<arith::SubFOp>(loc, warpPrefix, loadWarp.getResult(0));
        } else {
          Value prev = b.create<gpu::ShuffleOp>(loc, warpPrefix, 1, 32,
                                                gpu::ShuffleMode::UP)
                           .getShuffleResult();
          Value hasPrev = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, lane,
              b.create<arith::ConstantIntOp>(loc, 0, 32));
          warpExclusive =
              b.create<arith::SelectOp>(loc, hasPrev, prev, identF);
        }
        Value warpIdI32 =
            b.create<arith::IndexCastOp>(loc, b.getI32Type(), warpId);
        Value width32 = b.create<arith::ConstantIntOp>(loc, 32, 32);
        warpBase = b.create<gpu::ShuffleOp>(loc, warpExclusive, warpIdI32,
                                             width32, gpu::ShuffleMode::IDX)
                       .getShuffleResult();
      }

      // Thread-exclusive segment prefix = warp base + the preceding lanes in
      // this warp.  Sum uses incl-own; max/min use a one-lane shuffle.
      Value laneExclusive;
      if (isSum) {
        laneExclusive = b.create<arith::SubFOp>(loc, warpIncl, segTotal);
      } else {
        Value prev = b.create<gpu::ShuffleOp>(loc, warpIncl, 1, warpWidth,
                                              gpu::ShuffleMode::UP)
                         .getShuffleResult();
        Value hasPrev = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, lane,
            b.create<arith::ConstantIntOp>(loc, 0, 32));
        laneExclusive = b.create<arith::SelectOp>(loc, hasPrev, prev, identF);
      }
      Value excl = combineScalar(b, loc, sinfo.op, warpBase, laneExclusive);
      // (3) out[k] = combine(excl, local[k]) (full-row inclusive prefix).
      SmallVector<Value> vals(K);
      for (int64_t k = 0; k < K; ++k)
        vals[k] = combineScalar(b, loc, sinfo.op, excl, local[k]);
      scanVals[sc.getResult()] = vals;
    }

    Value nF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr((double)N));

    // B-certified sparsemax summary: for descending SortedX, the support mask
    // `z > (cumsum(z)-1)/rank` is a prefix.  Therefore k is the last supported
    // rank and s is the cumsum at that rank.  Reduce the pair (rank, cssv) by
    // maximum rank instead of separately summing a boolean mask and selected z.
    // This optimization is legal only when P4 materializes the explicit Layer-B
    // precondition; the graph shape alone is insufficient for unsorted inputs.
    bool prefixSummaryReduced = false;
    auto sortedAttr = fn->getAttrOfType<BoolAttr>("intent_sorted_descending");
    if (sortedAttr && sortedAttr.getValue() && scans.size() == 1 &&
        reduces.size() == 2) {
      ScanOp prefixScan = scans[0];
      ReduceOp countRed;
      ReduceOp sumRed;
      CastOp countCast;
      SelectOp sumSelect;
      size_t countIndex = 0, sumIndex = 0;
      for (size_t i = 0; i < reduces.size(); ++i) {
        ReduceOp red = reduces[i];
        if (auto castOp = red.getInput().getDefiningOp<CastOp>()) {
          auto srcTy = cast<RankedTensorType>(castOp.getInput().getType());
          auto dstTy = cast<RankedTensorType>(castOp.getResult().getType());
          if (red.getFn() == "sum" && srcTy.getElementType().isInteger(1) &&
              dstTy.getElementType().isInteger(32)) {
            countRed = red;
            countCast = castOp;
            countIndex = i;
          }
        }
        if (auto selectOp = red.getInput().getDefiningOp<SelectOp>()) {
          if (red.getFn() == "sum" &&
              selectOp.getX() == prefixScan.getInput()) {
            auto zero = selectOp.getY().getDefiningOp<arith::ConstantOp>();
            auto dense = zero ? dyn_cast<DenseElementsAttr>(zero.getValue())
                              : DenseElementsAttr();
            if (dense && dense.isSplat() &&
                dense.getSplatValue<APFloat>().isZero()) {
              sumRed = red;
              sumSelect = selectOp;
              sumIndex = i;
            }
          }
        }
      }
      if (countRed && sumRed && countCast.getInput() == sumSelect.getCond()) {
        int64_t warpWidth = std::min<int64_t>(T, 32);
        int64_t numWarps = (T + 31) / 32;
        Value txI32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
        Value lane = b.create<arith::AndIOp>(
            loc, txI32,
            b.create<arith::ConstantIntOp>(loc, warpWidth - 1, 32));
        Value bestRank = b.create<arith::ConstantIntOp>(loc, 0, 32);
        Value bestSum =
            b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
        for (int64_t k = 0; k < K; ++k) {
          Value supported = emitAt(b, loc, countCast.getInput(), k);
          Value rank = b.create<arith::AddIOp>(
              loc, b.create<arith::IndexCastOp>(loc, b.getI32Type(), cols[k]),
              b.create<arith::ConstantIntOp>(loc, 1, 32));
          bestRank =
              b.create<arith::SelectOp>(loc, supported, rank, bestRank);
          bestSum = b.create<arith::SelectOp>(
              loc, supported, scanVals[prefixScan.getResult()][k], bestSum);
        }
        for (int32_t delta = 1; delta < warpWidth; delta <<= 1) {
          Value otherRank = b.create<gpu::ShuffleOp>(
                                 loc, bestRank, delta, warpWidth,
                                 gpu::ShuffleMode::XOR)
                                .getShuffleResult();
          Value otherSum = b.create<gpu::ShuffleOp>(
                                loc, bestSum, delta, warpWidth,
                                gpu::ShuffleMode::XOR)
                               .getShuffleResult();
          Value takeOther = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, otherRank, bestRank);
          bestRank =
              b.create<arith::SelectOp>(loc, takeOther, otherRank, bestRank);
          bestSum =
              b.create<arith::SelectOp>(loc, takeOther, otherSum, bestSum);
        }
        if (numWarps > 1) {
          Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
          Value warpId = b.create<arith::DivUIOp>(loc, tx, c32);
          Value lane0 = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, lane,
              b.create<arith::ConstantIntOp>(loc, 0, 32));
          auto publish =
              b.create<scf::IfOp>(loc, lane0, /*withElseRegion=*/false);
          {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToStart(publish.thenBlock());
            b.create<memref::StoreOp>(loc, bestRank,
                                      reduceScratchI32[countIndex],
                                      ValueRange{warpId});
            b.create<memref::StoreOp>(loc, bestSum, reduceScratch[sumIndex],
                                      ValueRange{warpId});
          }
          b.create<gpu::BarrierOp>(loc);
          Value activeLane = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, lane,
              b.create<arith::ConstantIntOp>(loc, numWarps, 32));
          auto loadRank = b.create<scf::IfOp>(
              loc, TypeRange{b.getI32Type()}, activeLane,
              /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToStart(loadRank.thenBlock());
            Value laneIdx =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
            Value value = b.create<memref::LoadOp>(
                loc, reduceScratchI32[countIndex], ValueRange{laneIdx});
            b.create<scf::YieldOp>(loc, ValueRange{value});
            b.setInsertionPointToStart(loadRank.elseBlock());
            b.create<scf::YieldOp>(
                loc, ValueRange{b.create<arith::ConstantIntOp>(loc, 0, 32)});
          }
          auto loadSum = b.create<scf::IfOp>(
              loc, TypeRange{f32}, activeLane, /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToStart(loadSum.thenBlock());
            Value laneIdx =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
            Value value = b.create<memref::LoadOp>(
                loc, reduceScratch[sumIndex], ValueRange{laneIdx});
            b.create<scf::YieldOp>(loc, ValueRange{value});
            b.setInsertionPointToStart(loadSum.elseBlock());
            b.create<scf::YieldOp>(
                loc, ValueRange{b.create<arith::ConstantOp>(
                         loc, b.getF32FloatAttr(0.0))});
          }
          bestRank = loadRank.getResult(0);
          bestSum = loadSum.getResult(0);
          for (int32_t delta = 1; delta < 32; delta <<= 1) {
            Value otherRank = b.create<gpu::ShuffleOp>(
                                   loc, bestRank, delta, 32,
                                   gpu::ShuffleMode::XOR)
                                  .getShuffleResult();
            Value otherSum = b.create<gpu::ShuffleOp>(
                                  loc, bestSum, delta, 32,
                                  gpu::ShuffleMode::XOR)
                                 .getShuffleResult();
            Value takeOther = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, otherRank, bestRank);
            bestRank =
                b.create<arith::SelectOp>(loc, takeOther, otherRank, bestRank);
            bestSum =
                b.create<arith::SelectOp>(loc, takeOther, otherSum, bestSum);
          }
        }
        uniform[countRed.getResult()] = bestRank;
        uniform[sumRed.getResult()] = bestSum;
        hoistUniforms(b, loc);
        prefixSummaryReduced = true;
      }
    }

    // Per-row reductions use a warp-shuffle hierarchy.  Mutually independent
    // reductions (sparsemax's support count and supported-value sum) publish
    // their disjoint warp partials before ONE shared barrier.  A reduction whose
    // input consumes an earlier result starts a new group (e.g. softmax's
    // sum(exp(x-max))), preserving program-order semantics.
    if (!prefixSummaryReduced) {
    struct PendingReduce {
      size_t index;
      ReduceOp red;
      ReduceInfo info;
      Type elemTy;
      Type treeElemTy;
      bool booleanCount;
      bool integerTree;
      Value treeScratch;
      Value warpAcc;
    };
    int64_t reduceWarpWidth = std::min<int64_t>(T, 32);
    int64_t reduceNumWarps = (T + 31) / 32;
    Value reduceTxI32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
    Value reduceLane = b.create<arith::AndIOp>(
        loc, reduceTxI32,
        b.create<arith::ConstantIntOp>(loc, reduceWarpWidth - 1, 32));
    Value reduceC32;
    Value reduceWarpId;
    Value reduceLane0;
    if (reduceNumWarps > 1) {
      reduceC32 = b.create<arith::ConstantIndexOp>(loc, 32);
      reduceWarpId = b.create<arith::DivUIOp>(loc, tx, reduceC32);
      reduceLane0 = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, reduceLane,
          b.create<arith::ConstantIntOp>(loc, 0, 32));
    }

    size_t groupStart = 0;
    while (groupStart < reduces.size()) {
      size_t groupEnd = groupStart + 1;
      while (groupEnd < reduces.size()) {
        bool dependent = false;
        for (size_t prior = groupStart; prior < groupEnd; ++prior)
          dependent |= dependsOn(reduces[groupEnd].getInput(),
                                 reduces[prior].getResult());
        if (dependent)
          break;
        ++groupEnd;
      }

      SmallVector<PendingReduce> pending;
      pending.reserve(groupEnd - groupStart);
      for (size_t reduceIndex = groupStart; reduceIndex < groupEnd;
           ++reduceIndex) {
        ReduceOp red = reduces[reduceIndex];
        ReduceInfo info = *reduceInfoFor(red.getFn());
        Type elemTy = cast<RankedTensorType>(red.getInput().getType())
                          .getElementType();
        bool integerReduce = isa<IntegerType>(elemTy);
        auto inputCast = red.getInput().getDefiningOp<CastOp>();
        bool booleanCount =
            integerReduce && inputCast &&
            cast<RankedTensorType>(inputCast.getInput().getType())
                .getElementType()
                .isInteger(1);
        // Sum an i1 mask in exact f32 (N is static and <= 2^24), then cast the
        // one scalar result back to the declared i32 graph type.
        bool integerTree = integerReduce && !booleanCount;
        Type treeElemTy = integerTree ? elemTy : f32;
        Value treeScratch = integerTree ? reduceScratchI32[reduceIndex]
                                        : reduceScratch[reduceIndex];
        Value acc = integerTree
                        ? b.create<arith::ConstantIntOp>(loc, 0, 32).getResult()
                        : b.create<arith::ConstantOp>(
                               loc, b.getF32FloatAttr(info.ident))
                              .getResult();
        for (int64_t k = 0; k < K; ++k) {
          Value item;
          if (booleanCount) {
            Value predicate = emitAt(b, loc, inputCast.getInput(), k);
            item = b.create<arith::UIToFPOp>(loc, f32, predicate);
          } else {
            item = emitAt(b, loc, red.getInput(), k);
          }
          acc = integerTree
                    ? b.create<arith::AddIOp>(loc, acc, item).getResult()
                    : combineScalar(b, loc, info.op, acc, item);
        }
        Value warpAcc = acc;
        for (int32_t delta = 1; delta < reduceWarpWidth; delta <<= 1) {
          Value other = b.create<gpu::ShuffleOp>(
                             loc, warpAcc, delta, reduceWarpWidth,
                             gpu::ShuffleMode::XOR)
                            .getShuffleResult();
          warpAcc = integerTree
                        ? b.create<arith::AddIOp>(loc, warpAcc, other).getResult()
                        : combineScalar(b, loc, info.op, warpAcc, other);
        }
        if (reduceNumWarps > 1) {
          auto publish = b.create<scf::IfOp>(loc, reduceLane0,
                                             /*withElseRegion=*/false);
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(publish.thenBlock());
          b.create<memref::StoreOp>(loc, warpAcc, treeScratch,
                                    ValueRange{reduceWarpId});
        }
        pending.push_back(PendingReduce{reduceIndex, red, info, elemTy,
                                        treeElemTy, booleanCount, integerTree,
                                        treeScratch, warpAcc});
      }

      if (reduceNumWarps > 1)
        b.create<gpu::BarrierOp>(loc);

      for (PendingReduce &item : pending) {
        Value scalar = item.warpAcc;
        if (reduceNumWarps > 1) {
          Value activeLane = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, reduceLane,
              b.create<arith::ConstantIntOp>(loc, reduceNumWarps, 32));
          auto loadWarp = b.create<scf::IfOp>(
              loc, TypeRange{item.treeElemTy}, activeLane,
              /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToStart(loadWarp.thenBlock());
            Value laneIdx = b.create<arith::IndexCastOp>(
                loc, b.getIndexType(), reduceLane);
            Value total = b.create<memref::LoadOp>(
                loc, item.treeScratch, ValueRange{laneIdx});
            b.create<scf::YieldOp>(loc, ValueRange{total});
            b.setInsertionPointToStart(loadWarp.elseBlock());
            Value identity =
                item.integerTree
                    ? b.create<arith::ConstantIntOp>(loc, 0, 32).getResult()
                    : b.create<arith::ConstantOp>(
                           loc, b.getF32FloatAttr(item.info.ident))
                          .getResult();
            b.create<scf::YieldOp>(loc, identity);
          }
          scalar = loadWarp.getResult(0);
          for (int32_t delta = 1; delta < 32; delta <<= 1) {
            Value other = b.create<gpu::ShuffleOp>(
                               loc, scalar, delta, 32,
                               gpu::ShuffleMode::XOR)
                              .getShuffleResult();
            scalar = item.integerTree
                         ? b.create<arith::AddIOp>(loc, scalar, other).getResult()
                         : combineScalar(b, loc, item.info.op, scalar, other);
          }
        }
        if (item.info.isMean)
          scalar = finalizeMean(b, loc, scalar, nF);
        if (item.booleanCount)
          scalar = b.create<arith::FPToSIOp>(loc, item.elemTy, scalar);
        uniform[item.red.getResult()] = scalar;
        hoistUniforms(b, loc);
      }
      groupStart = groupEnd;
    }
    }

    // Once every value above the row epilogue is uniform, the output no longer
    // needs the scan's contiguous-per-thread column order.  Remap epilogue-only
    // rows to the regular coalesced vec4 layout
    //
    //   vec = tx + vt*T
    //
    // so adjacent lanes access adjacent 16-byte vectors.  The scan itself keeps
    // `tx*numVecTiles + vt`, because each thread must own a contiguous SortedX
    // segment.  Treat already-hoisted summaries as terminals while proving the
    // remap: a direct dependency on a scan value or on a non-deferred Row input
    // disables it.  This makes the transformation safe for arbitrary graph
    // names and leaves mixed scan/output graphs on the original layout.
    bool remapEpilogue = useVec4 && !deferredRowArgs.empty();
    DenseSet<Value> deferredRows(deferredRowArgs.begin(),
                                 deferredRowArgs.end());
    std::function<bool(Value)> remappableValue = [&](Value value) {
      if (uniform.count(value))
        return true;
      if (scanVals.count(value))
        return false;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        Role role = argRoles[arg.getArgNumber()];
        if (role == Role::Row)
          return deferredRows.contains(value);
        return true;
      }
      Operation *def = value.getDefiningOp();
      if (!def)
        return false;
      for (Value operand : def->getOperands())
        if (!remappableValue(operand))
          return false;
      return true;
    };
    if (remapEpilogue)
      for (unsigned j = 0; j < outputVals.size(); ++j)
        if (outRoles[j] == ResultRole::RowOut &&
            !remappableValue(outputVals[j]))
          remapEpilogue = false;
    if (remapEpilogue) {
      Value cFour = b.create<arith::ConstantIndexOp>(loc, 4);
      for (int64_t vt = 0; vt < numVecTiles; ++vt) {
        scanVecIdx[vt] = b.create<arith::AddIOp>(
            loc, tx, b.create<arith::ConstantIndexOp>(loc, vt * T));
        Value scalarBase =
            b.create<arith::MulIOp>(loc, scanVecIdx[vt], cFour);
        for (int64_t laneIndex = 0; laneIndex < 4; ++laneIndex)
          cols[vt * 4 + laneIndex] = b.create<arith::AddIOp>(
              loc, scalarBase,
              b.create<arith::ConstantIndexOp>(loc, laneIndex));
      }
      pointVals.clear();
    }

    // Materialize epilogue-only rows now that scan/reduction temporaries have
    // reached the end of their live ranges.  This mirrors the source Triton
    // scheduling boundary while remaining graph-structural.
    for (Value arg : deferredRowArgs)
      loadRowTile(arg);

    // RowOut epilogue: write each output from the register tile (read-once).
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      if (outRoles[j] != ResultRole::RowOut)
        continue;
      if (useVec4) {
        for (int64_t vt = 0; vt < numVecTiles; ++vt) {
          SmallVector<Value> lanes(4);
          for (int64_t lane = 0; lane < 4; ++lane)
            lanes[lane] =
                emitAt(b, loc, outputVals[j], vt * 4 + lane);
          Value vec = b.create<vector::FromElementsOp>(loc, vec4Ty, lanes);
          auto store = b.create<memref::StoreOp>(
              loc, vec, outMems[j], ValueRange{bx, scanVecIdx[vt]});
          store.setNontemporal(true);
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          Value out = emitAt(b, loc, outputVals[j], k);
          auto store = b.create<memref::StoreOp>(loc, out, outMems[j],
                                                  ValueRange{bx, cols[k]});
          store.setNontemporal(true);
        }
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    attachConcreteStateReceipt(gpuFn, T, predicatedValueIfCount,
                               foldedConstexprSelectCount, 1,
                               emittedApproximateDivCount);
    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    // Dispatch on the Schedule's fork decision (set in match()).
    if (sched.kind == KMultiStage)
      return rewriteMultiStage();
    if (sched.kind == KScan) {
      if (auto args = sortedSparsemaxArgIndices())
        return rewriteSortedSparsemax(args->first, args->second);
      return rewriteScan();
    }
    if (sched.kind == KTiled)
      return rewriteTiled();
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    // Register vector path: four contiguous f32 or f16/bf16 storage lanes.
    // Row/Col/RowOut pointers are retyped only after match() proves a uniform
    // storage ABI. Host pointers are unchanged under the bare-pointer calling
    // convention; extraction feeds the same scalar register-tile evaluator.
    bool useRegisterVector = useVec4 || useHalfVec4;
    int64_t registerVectorWidth = 4;
    Type registerVectorElement =
        useHalfVec4 ? halfVectorStorageType : f32;
    Type registerVectorType =
        VectorType::get({registerVectorWidth}, registerVectorElement);
    MemRefType rowVecMemref = MemRefType::get(
        {M, N / registerVectorWidth}, registerVectorType);
    MemRefType colVecMemref =
        MemRefType::get({N / registerVectorWidth}, registerVectorType);

    // New signature: original args (as memrefs) ++ one output memref per result.
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (useRegisterVector && argRoles[i] == Role::Row)
        inTys.push_back(rowVecMemref);
      else if (useRegisterVector && argRoles[i] == Role::Col)
        inTys.push_back(colVecMemref);
      else
        inTys.push_back(memrefOf(i));
    }
    for (unsigned j = 0; j < outRoles.size(); ++j) {
      if (inplaceArg[j] >= 0)
        continue;
      ResultRole rr = outRoles[j];
      Type outputElementType =
          cast<RankedTensorType>(fn.getFunctionType().getResult(j))
              .getElementType();
      if (rr == ResultRole::RowOut)
        inTys.push_back(useRegisterVector
                            ? rowVecMemref
                            : MemRefType::get({M, N}, outputElementType));
      else if (rr == ResultRole::PerRowOut)
        inTys.push_back(MemRefType::get({M}, outputElementType));
      else if (rr == ResultRole::PerRowVectorOut)
        inTys.push_back(
            MemRefType::get({M, outVectorWidths[j]}, outputElementType));
      else // AtomicOut: a rank-0 scalar accumulated across blocks (host pre-zeroes)
        inTys.push_back(MemRefType::get({}, outputElementType));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src"); // free up the symbol for the new func

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    bool useMultiRowWarpReduction = requestedRowsPerBlock > 1;
    bool useWarpLocalRegisterReduction =
        sched.kind == KRegister && T == 32;
    // Register-tile reductions use one compact f32 slot per (reduction, warp).
    // Independent reductions publish all warp partials before a shared barrier,
    // avoiding one standalone gpu.all_reduce synchronization sequence per op.
    // A production-sized dependent chain uses the same compact hierarchy one
    // group at a time.  The separate per-reduction slots preserve cross-warp
    // ordering without changing the original reduction DAG or its centered-
    // variance arithmetic.
    // A structurally single reduction uses the same scratch for a compact
    // two-level hierarchy: five intra-warp shuffles, log2(numWarps) shuffles in
    // warp 0, then one shared broadcast. This avoids gpu.all_reduce's padded
    // shared allocation and redundant 32-lane second-level fold.
    int64_t reduceNumWarps = (T + 31) / 32;
    bool useCompactDependentRegisterReductions =
        !useMultiRowWarpReduction && reduces.size() > 1 && T >= 64 &&
        (T % 32) == 0 &&
        reduceNumWarps <= 32 &&
        (reduceNumWarps & (reduceNumWarps - 1)) == 0;
    bool useGroupedRegisterReductions =
        !useMultiRowWarpReduction && T > 0 && (T & (T - 1)) == 0 &&
        (hasIndependentReductionGroup() ||
         useCompactDependentRegisterReductions);
    bool useCompactSingleRegisterReduction =
        !useMultiRowWarpReduction && !useGroupedRegisterReductions &&
        reduces.size() == 1 && T >= 64 &&
        (T % 32) == 0 && reduceNumWarps <= 32 &&
        (reduceNumWarps & (reduceNumWarps - 1)) == 0;
    bool useLeaderOnlySingleReduction =
        useCompactSingleRegisterReduction &&
        llvm::all_of(outRoles, [](ResultRole role) {
          return role != ResultRole::RowOut;
        });
    int64_t reduceScratchSlotBytes = reduceNumWarps * 4;
    int64_t reduceSharedBytes =
        (useGroupedRegisterReductions || useCompactSingleRegisterReduction) &&
                reduceNumWarps > 1
            ? static_cast<int64_t>(reduces.size()) * reduceScratchSlotBytes
            : 0;
    sched.smemBytes = reduceSharedBytes;
    attachLaunchPlan(gpuFn, sched);
    if (Attribute equalities = fn->getAttr("intent.scalar_equalities"))
      gpuFn->setAttr("intent.scalar_equalities", equalities);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);
    SmallVector<Value> outMems;
    unsigned freshOutput = argRoles.size();
    for (unsigned j = 0; j < outRoles.size(); ++j) {
      if (inplaceArg[j] >= 0)
        outMems.push_back(entry->getArgument(inplaceArg[j]));
      else
        outMems.push_back(entry->getArgument(freshOutput++));
    }

    // Preserve the public rank-0 memref ABI while promoting invocation
    // scalars to by-value gpu.launch captures.  The host stub performs one
    // load; the outlined kernel receives an ordinary scalar argument instead
    // of every lane redundantly loading the same device address.  Authored
    // value/dimension equalities remain compile-time constants below.
    DenseMap<Value, Value> launchScalars;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (!promoteRank0ScalarInputs || argRoles[i] != Role::Scalar ||
          scalarEqualities.contains(i))
        continue;
      Value sourceArg = srcBlock.getArgument(i);
      launchScalars[sourceArg] = b.create<memref::LoadOp>(
          loc, argMemref[sourceArg], ValueRange{});
    }

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, sched.gridX);
    Value cBlock = b.create<arith::ConstantIndexOp>(loc, sched.blockX);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    gpu::LaunchOp launch;
    if (reduceSharedBytes > 0) {
      Value sharedBytes = b.create<arith::ConstantOp>(
          loc, b.getI32IntegerAttr(reduceSharedBytes));
      launch = b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1,
                                      sharedBytes);
    } else {
      launch = b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1);
    }
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    if (useMultiRowWarpReduction) {
      // Keep the logical-warp width inside the device region.  A constant
      // created before gpu.launch is an explicit launch capture after GPU
      // outlining; NVPTX then sees an arbitrary kernel argument here and emits
      // the full generic u64 div/rem sequence for `tx / 32` and `tx % 32`.
      // This path has already proved T == 32 in match(), so an in-region
      // constant folds to the intended shift/mask address calculation.
      Value cLogicalRowThreads =
          b.create<arith::ConstantIndexOp>(loc, T);
      Value rowInBlock =
          b.create<arith::DivUIOp>(loc, tx, cLogicalRowThreads);
      tx = b.create<arith::RemUIOp>(loc, tx, cLogicalRowThreads);
      Value cRows = b.create<arith::ConstantIndexOp>(
          loc, requestedRowsPerBlock);
      Value rowBase = b.create<arith::MulIOp>(loc, bx, cRows);
      bx = b.create<arith::AddIOp>(loc, rowBase, rowInBlock);
    }
    rowIndex = bx;
    Value featureLaneActive;
    SmallVector<Value> featureTileActive;

    SmallVector<Value> reduceScratch;
    if (reduceSharedBytes > 0) {
      auto wgSpace =
          gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
      auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                                   nullptr, wgSpace);
      Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
      auto scratchTy =
          MemRefType::get({reduceNumWarps}, f32, nullptr, wgSpace);
      for (size_t index = 0; index < reduces.size(); ++index) {
        Value offset = b.create<arith::ConstantIndexOp>(
            loc, static_cast<int64_t>(index) * reduceScratchSlotBytes);
        reduceScratch.push_back(b.create<memref::ViewOp>(
            loc, scratchTy, raw, offset, ValueRange{}));
      }
    }

    // ColSeg (group_norm): the block's group index = bx % G (rows flattened as
    // batch*G + group). Uniform per block, computed once; the affine channel map
    // is applied per column in emitAt.
    if (numGroups != 0) {
      Value cG = b.create<arith::ConstantIndexOp>(loc, numGroups);
      groupVal = b.create<arith::RemUIOp>(loc, bx, cG);
    }

    // Column indices + the read-once register tile. Two layouts:
    //  - scalar path: col[k] = tx + k*T (interleaved, coalesced scalar loads).
    //  - vector path: the K columns split into contiguous four-lane storage
    //    packets. The register value at k=vt*W+j has scalar column
    //    (tx+vt*T)*W+j; extracted lanes feed the same scalar emitAt graph.
    cols.clear();
    cols.resize(K);
    // vecIdx[vt] = tx + vt*T: packet index into the retyped Row/Col memref.
    SmallVector<Value> vecIdx;
    if (useRegisterVector) {
      vecIdx.resize(numVecTiles);
      Value cWidth =
          b.create<arith::ConstantIndexOp>(loc, registerVectorWidth);
      for (int64_t vt = 0; vt < numVecTiles; ++vt) {
        Value off = b.create<arith::ConstantIndexOp>(loc, vt * T);
        vecIdx[vt] = b.create<arith::AddIOp>(loc, tx, off);
        Value base = b.create<arith::MulIOp>(loc, vecIdx[vt], cWidth);
        for (int64_t j = 0; j < registerVectorWidth; ++j) {
          Value cj = b.create<arith::ConstantIndexOp>(loc, j);
          cols[vt * registerVectorWidth + j] =
              b.create<arith::AddIOp>(loc, base, cj);
        }
      }
    } else if (hasFeatureTail) {
      // Each register position has its own tail predicate.  Invalid positions
      // read a safe address so no speculative load can go out of bounds; their
      // reduction term is replaced by the monoid identity below and the final
      // store is guarded by the same predicate.
      Value cN = b.create<arith::ConstantIndexOp>(loc, N);
      Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
      featureTileActive.resize(K);
      for (int64_t k = 0; k < K; ++k) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        Value logicalCol = b.create<arith::AddIOp>(loc, tx, off);
        featureTileActive[k] = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, logicalCol, cN);
        cols[k] = b.create<arith::SelectOp>(
            loc, featureTileActive[k], logicalCol, c0);
      }
    } else if (hasInactiveFeatureThreads) {
      // Keep every launched lane in the collective while preventing the lanes
      // beyond N from issuing an out-of-bounds load.  They read the safe first
      // column, but every reduction term from those lanes is replaced by the
      // monoid identity below, so the loaded value is semantically inert.
      Value cN = b.create<arith::ConstantIndexOp>(loc, N);
      Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
      featureLaneActive = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, tx, cN);
      cols[0] = b.create<arith::SelectOp>(loc, featureLaneActive, tx, c0);
    } else {
      for (int64_t k = 0; k < K; ++k) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        cols[k] = b.create<arith::AddIOp>(loc, tx, off);
      }
    }
    SmallVector<Value> deferredColArgs;
    auto loadVectorCol = [&](Value arg) {
      // A Col[N] parameter follows the same logical column mapping as every
      // Row[M,N] operand. Load each contiguous packet once and keep the
      // extracted storage lanes in SSA so all uses of W[col] share that load.
      SmallVector<Value> vals(K);
      for (int64_t vt = 0; vt < numVecTiles; ++vt) {
        Value v = b.create<memref::LoadOp>(loc, argMemref[arg],
                                           ValueRange{vecIdx[vt]});
        for (int64_t j = 0; j < registerVectorWidth; ++j)
          vals[vt * registerVectorWidth + j] =
              b.create<vector::ExtractOp>(loc, v, ArrayRef<int64_t>{j});
      }
      colVals[arg] = std::move(vals);
    };
    auto neededBeforeEpilogue = [&](Value arg) {
      for (ReduceOp red : reduces)
        if (dependsOn(red.getInput(), arg))
          return true;
      for (ReduceOp red : crossReduces)
        if (dependsOn(red.getInput(), arg))
          return true;
      for (ScanOp scan : scans)
        if (dependsOn(scan.getInput(), arg))
          return true;
      return false;
    };
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        SmallVector<Value> vals(K);
        if (useRegisterVector) {
          // One coalesced vector load per packet; extract storage lanes into
          // the same scalar register tile the non-vector path feeds emitAt.
          for (int64_t vt = 0; vt < numVecTiles; ++vt) {
            Value v = b.create<memref::LoadOp>(loc, argMemref[arg],
                                               ValueRange{bx, vecIdx[vt]});
            for (int64_t j = 0; j < registerVectorWidth; ++j) {
              Value lane = b.create<vector::ExtractOp>(
                  loc, v, ArrayRef<int64_t>{j});
              if (halfSoftmax)
                lane = b.create<arith::ExtFOp>(loc, f32, lane);
              vals[vt * registerVectorWidth + j] = lane;
            }
          }
        } else {
          for (int64_t k = 0; k < K; ++k) {
            Value loaded = b.create<memref::LoadOp>(
                loc, argMemref[arg], ValueRange{bx, cols[k]});
            if (halfSoftmax)
              loaded = b.create<arith::ExtFOp>(loc, f32, loaded);
            vals[k] = loaded;
          }
        }
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::Col && useRegisterVector) {
        if (neededBeforeEpilogue(arg) || !deferEpilogueColumnLoads)
          loadVectorCol(arg);
        else
          deferredColArgs.push_back(arg);
      } else if (argRoles[i] == Role::Scalar) {
        auto equality = scalarEqualities.find(i);
        if (equality != scalarEqualities.end()) {
          auto tensorTy = cast<RankedTensorType>(
              fn.getFunctionType().getInput(i));
          auto integerTy = cast<IntegerType>(tensorTy.getElementType());
          uniform[arg] = b.create<arith::ConstantOp>(
              loc, b.getIntegerAttr(integerTy, equality->second));
        } else {
          if (promoteRank0ScalarInputs) {
            uniform[arg] = launchScalars.lookup(arg);
            if (!uniform[arg]) {
              fn.emitError("rank-0 scalar was not promoted to launch ABI");
              return failure();
            }
          } else {
            uniform[arg] = b.create<memref::LoadOp>(
                loc, argMemref[arg], ValueRange{});
          }
        }
      } else if (isSingletonParamVector(i)) {
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{zero});
      } else if (argRoles[i] == Role::RowInt) {
        // Y[bx] is uniform across the block's threads: load once per block.
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
      } else if (argRoles[i] == Role::PerRow) {
        uniform[arg] =
            b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{bx});
      }
    }
    hoistUniforms(b, loc);

    // A required full-row output can double as the graph's explicit spill
    // buffer when the same pointwise value feeds the single reduction and a
    // post-reduction output. This is the structural add+RMSNorm shape:
    // materialize the required write before the block collective, then reload
    // it after the collective instead of keeping the whole row tile live across
    // both barriers. For an in-place result every Row input has already been
    // loaded above, so overwriting the aliased input cannot affect a later
    // source read. Fail closed for selects, multiple candidates, non-vec4
    // paths, or graphs without a downstream consumer.
    std::optional<unsigned> sharedRowOutput;
    if (useVec4 && useCompactSingleRegisterReduction && reduces.size() == 1) {
      SmallVector<unsigned> candidates;
      for (unsigned j = 0; j < outputVals.size(); ++j) {
        if (outRoles[j] != ResultRole::RowOut ||
            !outputVals[j].getDefiningOp<MapBinaryOp>() ||
            !dependsOn(reduces.front().getInput(), outputVals[j]))
          continue;
        if (inplaceArg[j] >= 0 &&
            argRoles[static_cast<unsigned>(inplaceArg[j])] != Role::Row)
          continue;
        bool downstreamOutput = false;
        for (unsigned other = 0; other < outputVals.size(); ++other)
          if (other != j && dependsOn(outputVals[other], outputVals[j])) {
            downstreamOutput = true;
            break;
          }
        if (!downstreamOutput)
          continue;
        bool containsSelect = false;
        DenseSet<Value> visited;
        std::function<void(Value)> inspect = [&](Value value) {
          if (containsSelect || !visited.insert(value).second)
            return;
          Operation *def = value.getDefiningOp();
          if (!def)
            return;
          if (isa<SelectOp>(def)) {
            containsSelect = true;
            return;
          }
          for (Value operand : def->getOperands())
            inspect(operand);
        };
        inspect(outputVals[j]);
        if (!containsSelect)
          candidates.push_back(j);
      }
      if (candidates.size() == 1)
        sharedRowOutput = candidates.front();
    }
    if (!materializeSharedRowOutput && !sharedRowOutput) {
      fn.emitError("requested row materialization elision has no exact shared "
                   "row-output graph");
      return failure();
    }
    std::optional<unsigned> stagedRowOutput =
        materializeSharedRowOutput ? sharedRowOutput : std::nullopt;
    bool elidedSharedRowMaterialization =
        !materializeSharedRowOutput && sharedRowOutput.has_value();
    if (stagedRowOutput) {
      unsigned j = *stagedRowOutput;
      for (int64_t vt = 0; vt < numVecTiles; ++vt) {
        SmallVector<Value> lanes(4);
        for (int64_t lane = 0; lane < 4; ++lane)
          lanes[lane] = emitAt(b, loc, outputVals[j], vt * 4 + lane);
        if (std::any_of(lanes.begin(), lanes.end(),
                        [](Value lane) { return !lane; })) {
          fn.emitError() << "staged row output " << j
                         << " has an unsupported scalar expression";
          return failure();
        }
        Value vec =
            b.create<vector::FromElementsOp>(loc, registerVectorType, lanes);
        b.create<memref::StoreOp>(loc, vec, outMems[j],
                                  ValueRange{bx, vecIdx[vt]});
      }
    }

    Value nF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr((double)N));

    // Non-power-of-two diagnostic schedules retain gpu.all_reduce, whose
    // verifier/runtime defines those widths. Production power-of-two schedules
    // use the grouped warp/shared hierarchy below.
    if (useWarpLocalRegisterReduction) {
      // With 32 logical lanes every hardware warp owns exactly one independent
      // row, whether the launch contains one row or several rows per block.
      // Reductions are therefore complete warp-local shuffle trees: no
      // gpu.all_reduce, shared scratch, or block barrier is needed. Dependent
      // reductions are emitted in program order and hoisted after each tree.
      for (ReduceOp red : reduces) {
        ReduceInfo info = *reduceInfoFor(red.getFn());
        Value acc =
            b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
        for (int64_t k = 0; k < K; ++k)
          acc = accumulateRegisterReduction(b, loc, red, info, acc, k,
                                            featureLaneActive,
                                            featureTileActive);
        for (int32_t delta = 1; delta < 32; delta <<= 1) {
          Value other = b.create<gpu::ShuffleOp>(
                            loc, acc, delta, 32, gpu::ShuffleMode::XOR)
                            .getShuffleResult();
          acc = combineScalar(b, loc, info.op, acc, other);
        }
        if (info.isMean)
          acc = finalizeMean(b, loc, acc, nF);
        uniform[red.getResult()] = acc;
        hoistUniforms(b, loc);
      }
    } else if (!useGroupedRegisterReductions &&
        !useCompactSingleRegisterReduction) {
      for (ReduceOp red : reduces) {
        ReduceInfo info = *reduceInfoFor(red.getFn());
        Value acc =
            b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
        for (int64_t k = 0; k < K; ++k)
          acc = accumulateRegisterReduction(b, loc, red, info, acc, k,
                                            featureLaneActive,
                                            featureTileActive);
        auto allRed = b.create<gpu::AllReduceOp>(
            loc, acc, gpu::AllReduceOperationAttr::get(ctx, info.op),
            /*uniform=*/UnitAttr::get(ctx));
        Value scalar = allRed.getResult();
        if (info.isMean)
          scalar = finalizeMean(b, loc, scalar, nF);
        uniform[red.getResult()] = scalar;
        hoistUniforms(b, loc);
      }
    } else if (useCompactSingleRegisterReduction) {
      ReduceOp red = reduces.front();
      ReduceInfo info = *reduceInfoFor(red.getFn());
      Value acc =
          b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
      for (int64_t k = 0; k < K; ++k)
        acc = accumulateRegisterReduction(b, loc, red, info, acc, k,
                                          featureLaneActive,
                                          featureTileActive);

      // First level: every warp reduces its thread partial in five shuffles.
      Value warpAcc = acc;
      for (int32_t delta = 1; delta < 32; delta <<= 1) {
        Value other = b.create<gpu::ShuffleOp>(loc, warpAcc, delta, 32,
                                               gpu::ShuffleMode::XOR)
                          .getShuffleResult();
        warpAcc = combineScalar(b, loc, info.op, warpAcc, other);
      }
      Value txI32 =
          b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
      Value lane = b.create<arith::AndIOp>(
          loc, txI32, b.create<arith::ConstantIntOp>(loc, 31, 32));
      Value laneZero = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lane,
          b.create<arith::ConstantIntOp>(loc, 0, 32));
      Value warpId = b.create<arith::DivUIOp>(
          loc, tx, b.create<arith::ConstantIndexOp>(loc, 32));
      {
        auto publish = b.create<scf::IfOp>(loc, laneZero,
                                           /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(publish.thenBlock());
        b.create<memref::StoreOp>(loc, warpAcc, reduceScratch.front(),
                                  ValueRange{warpId});
      }
      b.create<gpu::BarrierOp>(loc);

      // Second level: only global lanes [0,numWarps) read the compact partials.
      // Shuffle width=numWarps performs exactly log2(numWarps) exchanges in the
      // first warp (two for the formal xFormers 4-warp schedule), not five.
      Value activeWarpLane = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, tx,
          b.create<arith::ConstantIndexOp>(loc, reduceNumWarps));
      auto loadWarp = b.create<scf::IfOp>(
          loc, TypeRange{f32}, activeWarpLane, /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loadWarp.thenBlock());
        Value partial = b.create<memref::LoadOp>(
            loc, reduceScratch.front(), ValueRange{tx});
        b.create<scf::YieldOp>(loc, partial);
        b.setInsertionPointToStart(loadWarp.elseBlock());
        Value identity = b.create<arith::ConstantOp>(
            loc, b.getF32FloatAttr(info.ident));
        b.create<scf::YieldOp>(loc, identity);
      }
      Value blockAcc = loadWarp.getResult(0);
      for (int32_t delta = 1; delta < reduceNumWarps; delta <<= 1) {
        Value other = b.create<gpu::ShuffleOp>(
                            loc, blockAcc, delta, reduceNumWarps,
                            gpu::ShuffleMode::XOR)
                            .getShuffleResult();
        blockAcc = combineScalar(b, loc, info.op, blockAcc, other);
      }
      // If every result is written by the block leader, lane 0 can consume the
      // final warp-0 value directly.  Publishing it back to shared memory and
      // synchronizing the whole block only broadcasts a value that no other
      // thread observes.  RowOut epilogues still take the two-barrier path.
      Value scalar = blockAcc;
      if (!useLeaderOnlySingleReduction) {
        Value txZero = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, tx,
            b.create<arith::ConstantIndexOp>(loc, 0));
        {
          auto publish = b.create<scf::IfOp>(loc, txZero,
                                             /*withElseRegion=*/false);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(publish.thenBlock());
          b.create<memref::StoreOp>(
              loc, blockAcc, reduceScratch.front(),
              ValueRange{b.create<arith::ConstantIndexOp>(loc, 0)});
        }
        b.create<gpu::BarrierOp>(loc);
        scalar = b.create<memref::LoadOp>(
            loc, reduceScratch.front(),
            ValueRange{b.create<arith::ConstantIndexOp>(loc, 0)});
      }
      if (info.isMean)
        scalar = finalizeMean(b, loc, scalar, nF);
      uniform[red.getResult()] = scalar;
      hoistUniforms(b, loc);
    } else {
    // Per-row reductions use the same grouped warp/shared hierarchy as the scan
    // path. Mutually independent reductions publish their disjoint warp totals
    // before ONE barrier. A reduction that consumes an earlier result begins a
    // new group, preserving the original topological semantics.
    struct PendingRegisterReduce {
      size_t index;
      ReduceOp red;
      ReduceInfo info;
      Value warpAcc;
    };
    int64_t reduceWarpWidth = std::min<int64_t>(T, 32);
    Value reduceTxI32 =
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), tx);
    Value reduceLane = b.create<arith::AndIOp>(
        loc, reduceTxI32,
        b.create<arith::ConstantIntOp>(loc, reduceWarpWidth - 1, 32));
    Value reduceWarpId;
    Value reduceLaneZero;
    if (reduceNumWarps > 1) {
      reduceWarpId = b.create<arith::DivUIOp>(
          loc, tx, b.create<arith::ConstantIndexOp>(loc, 32));
      reduceLaneZero = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, reduceLane,
          b.create<arith::ConstantIntOp>(loc, 0, 32));
    }

    size_t groupStart = 0;
    while (groupStart < reduces.size()) {
      size_t groupEnd = groupStart + 1;
      while (groupEnd < reduces.size()) {
        bool dependent = false;
        for (size_t prior = groupStart; prior < groupEnd; ++prior)
          dependent |= dependsOn(reduces[groupEnd].getInput(),
                                 reduces[prior].getResult());
        if (dependent)
          break;
        ++groupEnd;
      }

      SmallVector<PendingRegisterReduce> pending;
      pending.reserve(groupEnd - groupStart);
      for (size_t index = groupStart; index < groupEnd; ++index) {
        ReduceOp red = reduces[index];
        ReduceInfo info = *reduceInfoFor(red.getFn());
        Value acc =
            b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(info.ident));
        for (int64_t k = 0; k < K; ++k)
          acc = accumulateRegisterReduction(b, loc, red, info, acc, k,
                                            featureLaneActive,
                                            featureTileActive);
        Value warpAcc = acc;
        for (int32_t delta = 1; delta < reduceWarpWidth; delta <<= 1) {
          Value other = b.create<gpu::ShuffleOp>(
                            loc, warpAcc, delta, reduceWarpWidth,
                            gpu::ShuffleMode::XOR)
                            .getShuffleResult();
          warpAcc = combineScalar(b, loc, info.op, warpAcc, other);
        }
        if (reduceNumWarps > 1) {
          auto publish = b.create<scf::IfOp>(loc, reduceLaneZero,
                                             /*withElseRegion=*/false);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(publish.thenBlock());
          b.create<memref::StoreOp>(loc, warpAcc, reduceScratch[index],
                                    ValueRange{reduceWarpId});
        }
        pending.push_back(PendingRegisterReduce{index, red, info, warpAcc});
      }

      if (reduceNumWarps > 1)
        b.create<gpu::BarrierOp>(loc);

      for (PendingRegisterReduce &item : pending) {
        Value scalar = item.warpAcc;
        if (reduceNumWarps > 1) {
          // Every warp redundantly folds the compact partial vector.  Using a
          // width-32 identity-padded fold gives every lane the block scalar
          // without a second block barrier; measured Layer-C selection decides
          // whether this synchronization tradeoff beats other candidates.
          Value activeLane = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, reduceLane,
              b.create<arith::ConstantIntOp>(loc, reduceNumWarps, 32));
          auto loadWarp = b.create<scf::IfOp>(
              loc, TypeRange{f32}, activeLane, /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(loadWarp.thenBlock());
            Value laneIndex = b.create<arith::IndexCastOp>(
                loc, b.getIndexType(), reduceLane);
            Value total = b.create<memref::LoadOp>(
                loc, reduceScratch[item.index], ValueRange{laneIndex});
            b.create<scf::YieldOp>(loc, total);
            b.setInsertionPointToStart(loadWarp.elseBlock());
            Value identity = b.create<arith::ConstantOp>(
                loc, b.getF32FloatAttr(item.info.ident));
            b.create<scf::YieldOp>(loc, identity);
          }
          scalar = loadWarp.getResult(0);
          for (int32_t delta = 1; delta < 32; delta <<= 1) {
            Value other = b.create<gpu::ShuffleOp>(
                              loc, scalar, delta, 32,
                              gpu::ShuffleMode::XOR)
                              .getShuffleResult();
            scalar =
                combineScalar(b, loc, item.info.op, scalar, other);
          }
        }
        if (item.info.isMean)
          scalar = finalizeMean(b, loc, scalar, nF);
        uniform[item.red.getResult()] = scalar;
        hoistUniforms(b, loc);
      }
      groupStart = groupEnd;
    }
    }

    // Drop the pre-reduction SSA tile from the post-reduction live range.  The
    // required RowOut store above is the authoritative materialization, so a
    // coalesced reload supplies every downstream epilogue use and the final
    // RowOut loop must not write the same destination again.
    if (stagedRowOutput) {
      unsigned j = *stagedRowOutput;
      SmallVector<Value> reloaded(K);
      for (int64_t vt = 0; vt < numVecTiles; ++vt) {
        Value vec = b.create<memref::LoadOp>(loc, outMems[j],
                                             ValueRange{bx, vecIdx[vt]});
        for (int64_t lane = 0; lane < 4; ++lane)
          reloaded[vt * 4 + lane] = b.create<vector::ExtractOp>(
              loc, vec, ArrayRef<int64_t>{lane});
      }
      pointVals[outputVals[j]] = std::move(reloaded);
    }

    // Column parameters used only by the pointwise epilogue must not stay live
    // across one or more block collectives.  Load their vec4 packets here,
    // after the last reduction barrier.  Columns that participate in a reduce
    // or scan were proven above and loaded before the collective instead.
    for (Value arg : deferredColArgs)
      loadVectorCol(arg);

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
    auto conditionalWriteForAlias = [&](unsigned j) -> Value {
      if (inplaceArg[j] < 0)
        return nullptr;
      // A validated partial writeback can be expressed in a transposed row
      // domain so a rank-1 per-row predicate broadcasts along the last axis:
      //   transpose(select(cond, transpose(alias), value)).
      // Transposes admitted by match() are index views in this lowering. Strip
      // those views both above the select and on its branches before checking
      // whether one branch preserves the aliased destination.
      auto stripTransposeViews = [](Value value) {
        while (auto transpose = value.getDefiningOp<TransposeOp>())
          value = transpose.getInput();
        return value;
      };
      Value selected = stripTransposeViews(outputVals[j]);
      auto select = selected.getDefiningOp<SelectOp>();
      if (!select || !isUniform(select.getCond()))
        return nullptr;
      Value alias = srcBlock.getArgument(inplaceArg[j]);
      Value cond = emitAt(b, loc, select.getCond(), 0);
      if (!cond || !cond.getType().isInteger(1))
        return nullptr;
      Value x = stripTransposeViews(select.getX());
      Value y = stripTransposeViews(select.getY());
      if (y == alias)
        return cond; // false branch preserves the aliased destination
      if (x == alias) {
        Value cFalse =
            b.create<arith::ConstantOp>(loc, b.getBoolAttr(false));
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cond,
                                      cFalse);
      }
      return nullptr;
    };
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      if (stagedRowOutput && j == *stagedRowOutput)
        continue;
      if (outRoles[j] == ResultRole::RowOut) {
        // Materialize the scalar/vector epilogue before entering a conditional
        // write region. emitAt memoizes pointwise SSA values across outputs; if
        // the first output were computed inside scf.if, a later output sharing
        // that DAG would reuse values defined in the child region and violate
        // dominance. Only the destination stores belong in the conditional.
        SmallVector<Value> prepared;
        if (useRegisterVector) {
          prepared.reserve(numVecTiles);
          // Repack the contiguous scalar epilogue lanes into one storage
          // vector. Exact half softmax owns an implicit final demotion;
          // explicit mixed-precision graphs already end in intent.cast.
          for (int64_t vt = 0; vt < numVecTiles; ++vt) {
            SmallVector<Value> lanes(registerVectorWidth);
            for (int64_t lj = 0; lj < registerVectorWidth; ++lj) {
              Value lane = emitAt(
                  b, loc, outputVals[j], vt * registerVectorWidth + lj);
              if (halfSoftmax && lane)
                lane = b.create<arith::TruncFOp>(loc,
                                                  halfVectorStorageType, lane);
              lanes[lj] = lane;
            }
            if (std::any_of(lanes.begin(), lanes.end(),
                            [](Value lane) { return !lane; })) {
              fn.emitError() << "row output " << j
                             << " has an unsupported scalar epilogue";
              return failure();
            }
            prepared.push_back(
                b.create<vector::FromElementsOp>(loc, registerVectorType,
                                                 lanes));
          }
        } else {
          prepared.reserve(K);
          for (int64_t k = 0; k < K; ++k) {
            Value out = emitAt(b, loc, outputVals[j], k);
            if (!out) {
              fn.emitError() << "row output " << j
                             << " has an unsupported scalar epilogue";
              return failure();
            }
            if (halfSoftmax)
              out = b.create<arith::TruncFOp>(loc, storageType, out);
            prepared.push_back(out);
          }
        }
        auto emitPreparedStores = [&]() {
          for (int64_t index = 0; index < static_cast<int64_t>(prepared.size());
               ++index) {
            // Do not retain a ValueRange backed by a temporary initializer
            // list.  Optimized builds may consume the dangling storage while
            // constructing memref.store, which previously made the row
            // reduction pass crash under -O2 even though debug builds passed.
            SmallVector<Value, 2> indices{
                bx, useRegisterVector ? vecIdx[index] : cols[index]};
            if (hasFeatureTail) {
              auto ifOp = b.create<scf::IfOp>(
                  loc, featureTileActive[index], /*withElseRegion=*/false);
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(ifOp.thenBlock());
              b.create<memref::StoreOp>(loc, prepared[index], outMems[j],
                                        indices);
            } else {
              b.create<memref::StoreOp>(loc, prepared[index], outMems[j],
                                        indices);
            }
          }
        };
        if (Value write = conditionalWriteForAlias(j)) {
          auto ifOp = b.create<scf::IfOp>(loc, write, /*withElseRegion=*/false);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(ifOp.thenBlock());
          emitPreparedStores();
        } else {
          emitPreparedStores();
        }
      } else if (outRoles[j] == ResultRole::PerRowOut) {
        Value scalar = emitAt(b, loc, outputVals[j], 0); // per-row uniform
        if (!scalar) {
          fn.emitError() << "per-row output " << j
                         << " has an unsupported scalar epilogue";
          return failure();
        }
        Value write = isLead;
        if (Value conditional = conditionalWriteForAlias(j))
          write = b.create<arith::AndIOp>(loc, isLead, conditional);
        auto ifOp = b.create<scf::IfOp>(loc, write, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        b.create<memref::StoreOp>(loc, scalar, outMems[j], ValueRange{bx});
      } else if (outRoles[j] == ResultRole::PerRowVectorOut) {
        auto concat = outputVals[j].getDefiningOp<ConcatOp>();
        auto ifOp = b.create<scf::IfOp>(loc, isLead, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(ifOp.thenBlock());
        if (concat) {
          if (concat.getInputs().size() !=
              static_cast<size_t>(outVectorWidths[j])) {
            fn.emitError() << "per-row vector output " << j
                           << " has a mismatched singleton-column concat";
            return failure();
          }
          unsigned column = 0;
          for (Value input : concat.getInputs()) {
            Value scalar = emitAt(b, loc, input, 0);
            if (!scalar) {
              fn.emitError() << "per-row vector output " << j
                             << " has an unsupported concat element";
              return failure();
            }
            Value c = b.create<arith::ConstantIndexOp>(loc, column++);
            b.create<memref::StoreOp>(loc, scalar, outMems[j],
                                      ValueRange{bx, c});
          }
        } else {
          for (int64_t column = 0; column < outVectorWidths[j]; ++column) {
            Value scalar = emitPerRowVectorAt(
                b, loc, outputVals[j], column, outVectorWidths[j]);
            if (!scalar) {
              fn.emitError() << "per-row vector output " << j
                             << " has an unsupported compact DAG";
              return failure();
            }
            Value c = b.create<arith::ConstantIndexOp>(loc, column);
            b.create<memref::StoreOp>(loc, scalar, outMems[j],
                                      ValueRange{bx, c});
          }
        }
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

    attachConcreteStateReceipt(gpuFn, T, predicatedValueIfCount,
                               foldedConstexprSelectCount,
                               requestedRowsPerBlock,
                               emittedApproximateDivCount,
                               launchScalars.size(),
                               deferredColArgs.size(),
                               stagedRowOutput.has_value(),
                               elidedSharedRowMaterialization);
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
    struct Target {
      func::FuncOp fn;
      unsigned predicatedValueIfs;
      unsigned foldedConstexprSelects;
    };
    SmallVector<Target> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      unsigned predicatedValueIfs =
          predicatePureValueIf ? predicatePureTensorValueIfs(fn) : 0;
      unsigned foldedConstexprSelects = foldRankZeroConstantSelects(fn);
      RowReductionLowering probe(fn, blockThreads, tileCols, vectorWidth,
                                 fuseFma, approximateDiv,
                                 deferEpilogueColLoads,
                                 materializeSharedRowOutput,
                                 promoteRank0Scalars,
                                 enableSortedPrefix, predicatedValueIfs,
                                 foldedConstexprSelects, rowsPerBlock,
                                 reductionParts);
      if (succeeded(probe.match()))
        targets.push_back(
            {fn, predicatedValueIfs, foldedConstexprSelects});
    }
    for (Target &target : targets) {
      RowReductionLowering low(target.fn, blockThreads, tileCols, vectorWidth,
                               fuseFma, approximateDiv,
                               deferEpilogueColLoads,
                               materializeSharedRowOutput,
                               promoteRank0Scalars,
                               enableSortedPrefix,
                               target.predicatedValueIfs,
                               target.foldedConstexprSelects, rowsPerBlock,
                               reductionParts);
      if (failed(low.match()) || failed(low.rewrite())) {
        target.fn.emitError("row-reduction lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
