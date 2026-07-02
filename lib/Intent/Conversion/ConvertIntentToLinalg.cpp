//===- ConvertIntentToLinalg.cpp - intent -> linalg/tensor/arith ----------===//
//
// First leg of the back-end spine. Lowers Layer-A atomic ops to standard MLIR
// so the stock linalg->gpu->nvvm pipeline can carry them to a GPU binary.
// Scope today: elementwise intent.map_binary (equal shapes, no broadcast),
// which is the minimum needed to reach a first intent-sourced cubin (M5).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTTOLINALG
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// Emit the scalar arith op for one `map_binary` fn over two scalar operands.
/// Returns nullptr for fns that have no float scalar form here yet.
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
    return b.create<arith::MaximumFOp>(loc, a, c);
  if (fn == "min")
    return b.create<arith::MinimumFOp>(loc, a, c);
  if (fn == "pow")
    return b.create<math::PowFOp>(loc, a, c);
  return nullptr;
}

/// Emit the scalar op for one `map_unary` fn over a scalar operand.
/// Returns nullptr for fns that have no float scalar form here yet.
static Value emitScalarUnary(OpBuilder &b, Location loc, StringRef fn, Value x) {
  if (fn == "exp")
    return b.create<math::ExpOp>(loc, x);
  if (fn == "log")
    return b.create<math::LogOp>(loc, x);
  if (fn == "abs")
    return b.create<math::AbsFOp>(loc, x);
  if (fn == "sqrt")
    return b.create<math::SqrtOp>(loc, x);
  if (fn == "rsqrt")
    return b.create<math::RsqrtOp>(loc, x);
  if (fn == "neg")
    return b.create<arith::NegFOp>(loc, x);
  if (fn == "recip") {
    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::DivFOp>(loc, one, x);
  }
  if (fn == "tanh")
    return b.create<math::TanhOp>(loc, x);
  // CPU/linalg path: raw math.* transcendentals (libdevice concerns are the GPU
  // pass's problem; the NVVM approx peephole lives in MapEmit.h's emitter).
  if (fn == "cos")
    return b.create<math::CosOp>(loc, x);
  if (fn == "sin")
    return b.create<math::SinOp>(loc, x);
  if (fn == "tan")
    return b.create<arith::DivFOp>(loc, b.create<math::SinOp>(loc, x),
                                   b.create<math::CosOp>(loc, x));
  if (fn == "exp2")
    return b.create<math::Exp2Op>(loc, x);
  if (fn == "erf")
    return b.create<math::ErfOp>(loc, x);
  if (fn == "sigmoid") {
    // 1 / (1 + exp(-x))
    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value negx = b.create<arith::NegFOp>(loc, x);
    Value e = b.create<math::ExpOp>(loc, negx);
    Value den = b.create<arith::AddFOp>(loc, one, e);
    return b.create<arith::DivFOp>(loc, one, den);
  }
  return nullptr;
}

struct MapUnaryLowering : OpRewritePattern<MapUnaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapUnaryOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    if (!isa<FloatType>(resTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "non-float unary not implemented");
    StringRef fn = op.getFn();
    if (!emitScalarUnary(rewriter, op.getLoc(), fn, op.getInput()))
      return rewriter.notifyMatchFailure(op, "fn has no scalar lowering");

    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    unsigned rank = resTy.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getInput()}, ValueRange{init},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, emitScalarUnary(b, l, fn, args[0]));
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Build the linalg input indexing map for a (possibly broadcasting) operand
/// against a result of rank `R`. Operand dims align right (numpy rules); a dim
/// whose operand extent is 1 but result extent is >1 broadcasts (constant-0
/// index). Returns the operand's affine map over the R-dim iteration domain.
static AffineMap broadcastMap(MLIRContext *ctx, ArrayRef<int64_t> opShape,
                              ArrayRef<int64_t> resShape) {
  unsigned R = resShape.size(), r = opShape.size();
  SmallVector<AffineExpr> exprs;
  for (unsigned i = 0; i < r; ++i) {
    unsigned d = R - r + i; // right-aligned result dim
    if (opShape[i] == 1 && resShape[d] != 1)
      exprs.push_back(getAffineConstantExpr(0, ctx));
    else
      exprs.push_back(getAffineDimExpr(d, ctx));
  }
  return AffineMap::get(R, 0, exprs, ctx);
}

struct MapBinaryLowering : OpRewritePattern<MapBinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapBinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    auto lhsTy = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(op.getRhs().getType());

    if (!isa<FloatType>(resTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "non-float elementwise not implemented");
    if (!emitScalarBinary(rewriter, op.getLoc(), op.getFn(),
                          op.getLhs(), op.getRhs()))
      return rewriter.notifyMatchFailure(op, "fn has no scalar lowering");

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    unsigned rank = resTy.getRank();
    // Result is the broadcast shape; each input maps in (identity when equal,
    // constant-0 on broadcast dims). Output is identity.
    SmallVector<AffineMap> maps = {
        broadcastMap(ctx, lhsTy.getShape(), resTy.getShape()),
        broadcastMap(ctx, rhsTy.getShape(), resTy.getShape()),
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    StringRef fn = op.getFn();
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, emitScalarBinary(b, l, fn, args[0], args[1]));
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Scalar combiner + identity element for a `reduce` fn.
static Value emitReduceCombine(OpBuilder &b, Location loc, StringRef fn, Value a,
                               Value c) {
  if (fn == "sum" || fn == "mean")
    return b.create<arith::AddFOp>(loc, a, c);
  if (fn == "max")
    return b.create<arith::MaximumFOp>(loc, a, c);
  if (fn == "min")
    return b.create<arith::MinimumFOp>(loc, a, c);
  if (fn == "prod")
    return b.create<arith::MulFOp>(loc, a, c);
  return nullptr;
}

static std::optional<double> reduceIdentity(StringRef fn) {
  if (fn == "sum" || fn == "mean")
    return 0.0;
  if (fn == "prod")
    return 1.0;
  if (fn == "max")
    return -std::numeric_limits<double>::infinity();
  if (fn == "min")
    return std::numeric_limits<double>::infinity();
  return std::nullopt;
}

/// Lower `intent.reduce` to a linalg.generic reduction. Reduced axes become
/// reduction iterators; the output drops them. If the result keeps the reduced
/// axes as size-1 (keepdims, result rank == input rank), a tensor.expand_shape
/// re-inserts them so downstream broadcasts (e.g. S - rowmax) line up.
struct ReduceLowering : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy = cast<RankedTensorType>(op.getInput().getType());
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = inTy.getElementType();
    if (!isa<FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "non-float reduce not implemented");
    StringRef fn = op.getFn();
    auto ident = reduceIdentity(fn);
    if (!ident)
      return rewriter.notifyMatchFailure(op, "reduce fn has no scalar lowering");

    unsigned R = inTy.getRank();
    SmallVector<bool> isReduced(R, false);
    for (Attribute a : op.getAxes()) {
      int64_t ax = cast<IntegerAttr>(a).getInt();
      if (ax < 0)
        ax += R;
      if (ax < 0 || ax >= (int64_t)R)
        return rewriter.notifyMatchFailure(op, "reduce axis out of range");
      isReduced[ax] = true;
    }

    // reducedTy = input with reduced axes removed.
    SmallVector<int64_t> reducedShape;
    for (unsigned d = 0; d < R; ++d)
      if (!isReduced[d])
        reducedShape.push_back(inTy.getDimSize(d));
    auto reducedTy = RankedTensorType::get(reducedShape, elemTy);
    unsigned r = reducedShape.size();
    if (r == 0)
      return rewriter.notifyMatchFailure(op, "full reduction not implemented");

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    // Iteration domain = all input dims. Input map = identity; output map keeps
    // only non-reduced dims (in order).
    SmallVector<utils::IteratorType> iters(R, utils::IteratorType::parallel);
    SmallVector<AffineExpr> outExprs;
    for (unsigned d = 0; d < R; ++d) {
      if (isReduced[d])
        iters[d] = utils::IteratorType::reduction;
      else
        outExprs.push_back(getAffineDimExpr(d, ctx));
    }
    SmallVector<AffineMap> maps = {rewriter.getMultiDimIdentityMap(R),
                                   AffineMap::get(R, 0, outExprs, ctx)};

    Value identVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemTy, *ident));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, reducedShape, elemTy);
    Value init = rewriter.create<linalg::FillOp>(loc, ValueRange{identVal},
                                                 ValueRange{empty}).getResult(0);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{reducedTy}, ValueRange{op.getInput()}, ValueRange{init},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, emitReduceCombine(b, l, fn, args[1], args[0]));
        });
    Value reduced = generic.getResult(0);

    // mean = sum / count, where count = product of the reduced extents (static).
    if (fn == "mean") {
      int64_t count = 1;
      for (unsigned d = 0; d < R; ++d) {
        if (!isReduced[d])
          continue;
        if (inTy.isDynamicDim(d))
          return rewriter.notifyMatchFailure(op, "mean over dynamic extent");
        count *= inTy.getDimSize(d);
      }
      Value inv = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemTy, 1.0 / (double)count));
      Value empty2 = rewriter.create<tensor::EmptyOp>(loc, reducedShape, elemTy);
      SmallVector<AffineMap> smaps(2, rewriter.getMultiDimIdentityMap(r));
      SmallVector<utils::IteratorType> siters(r, utils::IteratorType::parallel);
      reduced = rewriter
                    .create<linalg::GenericOp>(
                        loc, TypeRange{reducedTy}, ValueRange{reduced},
                        ValueRange{empty2}, smaps, siters,
                        [&](OpBuilder &b, Location l, ValueRange args) {
                          Value scaled =
                              b.create<arith::MulFOp>(l, args[0], inv);
                          b.create<linalg::YieldOp>(l, scaled);
                        })
                    .getResult(0);
    }

    // keepdims: result rank == input rank, reduced axes are size-1. Expand back.
    if (resTy.getRank() == (int64_t)R) {
      SmallVector<ReassociationIndices> reassoc(r);
      int src = -1;
      for (unsigned d = 0; d < R; ++d) {
        if (!isReduced[d])
          ++src;
        reassoc[src < 0 ? 0 : src].push_back(d);
      }
      Value expanded = rewriter.create<tensor::ExpandShapeOp>(
          loc, resTy, reduced, reassoc);
      rewriter.replaceOp(op, expanded);
      return success();
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Lower an einsum-style contraction to a linalg.generic over the union of
/// index labels: output labels are parallel iterators, contracted labels
/// (present in inputs but not the output) are reduction iterators. Body is a
/// fused multiply-accumulate; the output is zero-initialized.
struct ContractionLowering : OpRewritePattern<ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractionOp op,
                                PatternRewriter &rewriter) const override {
    StringRef sub = op.getSubscripts();
    size_t arrow = sub.find("->");
    size_t comma = sub.find(',');
    if (arrow == StringRef::npos || comma == StringRef::npos)
      return rewriter.notifyMatchFailure(op, "malformed subscripts");
    StringRef aL = sub.substr(0, comma);
    StringRef bL = sub.substr(comma + 1, arrow - comma - 1);
    StringRef oL = sub.substr(arrow + 2);

    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = resTy.getElementType();
    if (!isa<FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "non-float contraction not implemented");

    // Iteration domain = output labels (parallel) ++ contracted labels
    // (reduction). Contracted = appears in an input but not the output.
    llvm::DenseMap<char, unsigned> dim;
    SmallVector<utils::IteratorType> iters;
    for (char c : oL)
      if (!dim.count(c)) {
        dim[c] = dim.size();
        iters.push_back(utils::IteratorType::parallel);
      }
    for (StringRef in : {aL, bL})
      for (char c : in)
        if (!dim.count(c)) {
          dim[c] = dim.size();
          iters.push_back(utils::IteratorType::reduction);
        }

    MLIRContext *ctx = op.getContext();
    auto mapFor = [&](StringRef labels) {
      SmallVector<AffineExpr> exprs;
      for (char c : labels)
        exprs.push_back(getAffineDimExpr(dim[c], ctx));
      return AffineMap::get(dim.size(), 0, exprs, ctx);
    };
    SmallVector<AffineMap> maps = {mapFor(aL), mapFor(bL), mapFor(oL)};

    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemTy));
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, resTy.getShape(), elemTy);
    Value init = rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                                 ValueRange{empty}).getResult(0);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getA(), op.getB()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value prod = b.create<arith::MulFOp>(l, args[0], args[1]);
          Value acc = b.create<arith::AddFOp>(l, args[2], prod);
          b.create<linalg::YieldOp>(l, acc);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower `intent.loop` to an `scf.for` over the trip count with the carries as
/// tensor `iter_args`. Each trip slices the scanned inputs (`tensor.extract_slice`
/// along `scan_axes`, `tile_hint` rows) and the body's intent ops are cloned in
/// (they lower to linalg via the patterns above). This is the natural tensor-
/// level lowering; the *fused* device kernel is a separate schedule (see
/// bench/flash_emit.py + the host-driven body path in test_flash_loop_*).
struct LoopLowering : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned nScan = op.getScanInputs().size();
    if (nScan == 0)
      return rewriter.notifyMatchFailure(op, "loop with no scan inputs");

    int64_t tile =
        op.getTileHint() ? static_cast<int64_t>(*op.getTileHint()) : 1;
    auto scan0Ty = cast<RankedTensorType>(op.getScanInputs()[0].getType());
    int64_t axis0 = cast<IntegerAttr>(op.getScanAxes()[0]).getInt();
    int64_t extent = scan0Ty.getDimSize(axis0);
    if (ShapedType::isDynamic(extent))
      return rewriter.notifyMatchFailure(op, "dynamic scan extent");
    if (tile <= 0 || extent % tile != 0)
      return rewriter.notifyMatchFailure(
          op, "scan extent not a multiple of tile_hint (tail tiles unsupported)");
    int64_t trip = extent / tile;

    Block &body = op.getBody().front();
    Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = rewriter.create<arith::ConstantIndexOp>(loc, trip);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value tileV = rewriter.create<arith::ConstantIndexOp>(loc, tile);
    SmallVector<Value> inits(op.getInits().begin(), op.getInits().end());

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lb, ub, step, inits,
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          IRMapping map;
          Value off = b.create<arith::MulIOp>(l, iv, tileV);
          for (unsigned i = 0; i < nScan; ++i) {
            Value src = op.getScanInputs()[i];
            auto srcTy = cast<RankedTensorType>(src.getType());
            int64_t ax = cast<IntegerAttr>(op.getScanAxes()[i]).getInt();
            unsigned rank = srcTy.getRank();
            SmallVector<OpFoldResult> offs(rank, b.getIndexAttr(0));
            SmallVector<OpFoldResult> sizes;
            for (unsigned d = 0; d < rank; ++d)
              sizes.push_back(b.getIndexAttr(srcTy.getDimSize(d)));
            SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
            offs[ax] = off;
            sizes[ax] = b.getIndexAttr(tile);
            auto sliceTy = cast<RankedTensorType>(body.getArgument(i).getType());
            Value slice = b.create<tensor::ExtractSliceOp>(
                l, sliceTy, src, offs, sizes, strides);
            map.map(body.getArgument(i), slice);
          }
          for (unsigned j = 0; j < iterArgs.size(); ++j)
            map.map(body.getArgument(nScan + j), iterArgs[j]);
          for (Operation &o : body.without_terminator())
            b.clone(o, map);
          auto yield = cast<YieldOp>(body.getTerminator());
          SmallVector<Value> yvals;
          for (Value v : yield.getResults())
            yvals.push_back(map.lookupOrDefault(v));
          b.create<scf::YieldOp>(l, yvals);
        });
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

/// Emit the scalar arith conversion for a (src -> dst) element-type pair.
/// Identity when equal; the natural arith op otherwise.
static Value emitScalarCast(OpBuilder &b, Location loc, Type dst, Value v) {
  Type src = v.getType();
  if (src == dst)
    return v;
  auto sf = dyn_cast<FloatType>(src);
  auto df = dyn_cast<FloatType>(dst);
  auto si = dyn_cast<IntegerType>(src);
  auto di = dyn_cast<IntegerType>(dst);
  if (sf && df)
    return sf.getWidth() < df.getWidth()
               ? b.create<arith::ExtFOp>(loc, dst, v).getResult()
               : b.create<arith::TruncFOp>(loc, dst, v).getResult();
  if (sf && di)
    return b.create<arith::FPToSIOp>(loc, dst, v);
  if (si && df)
    return b.create<arith::SIToFPOp>(loc, dst, v);
  if (si && di)
    return si.getWidth() < di.getWidth()
               ? b.create<arith::ExtSIOp>(loc, dst, v).getResult()
               : b.create<arith::TruncIOp>(loc, dst, v).getResult();
  return nullptr;
}

struct CastLowering : OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    auto inTy = cast<RankedTensorType>(op.getInput().getType());
    // Same element type -> identity alias (the f32-only corpus path).
    if (inTy.getElementType() == resTy.getElementType()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    unsigned rank = resTy.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Type dstElt = resTy.getElementType();
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getInput()}, ValueRange{init}, maps,
        iters, [&](OpBuilder &b, Location l, ValueRange args) {
          Value c = emitScalarCast(b, l, dstElt, args[0]);
          b.create<linalg::YieldOp>(l, c);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct ConvertIntentToLinalgPass
    : ::intent::impl::ConvertIntentToLinalgBase<ConvertIntentToLinalgPass> {
  using ConvertIntentToLinalgBase::ConvertIntentToLinalgBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MapUnaryLowering, MapBinaryLowering, ReduceLowering,
                 ContractionLowering, LoopLowering, CastLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
