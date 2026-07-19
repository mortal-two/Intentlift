//===- MapEmit.h - shared scalar emission for intent map lowerings --------===//
//
// Pure, stateless helpers shared by the block-cooperative intent->gpu lowerings
// (`convert-intent-row-reduction-to-gpu` and `convert-intent-elementwise-to-gpu`).
// Only the genuinely stateless pieces live here: the per-`fn` scalar arith/math
// emitters, the per-arg `Role` taxonomy (inferred from static shape vs (M,N)),
// and the launch-geometry default. The stateful read-once register-tile
// machinery (emitAt recursion / hoistUniforms / the per-arg SSA caches) stays
// self-contained in each pass — it is ~60 lines, intertwined with each pass's
// DenseMaps, and duplicating it keeps the M1 pass byte-for-byte unchanged in
// behavior (no shared base struct = no destabilization risk).
//
//===----------------------------------------------------------------------===//

#ifndef INTENT_CONVERSION_MAPEMIT_H
#define INTENT_CONVERSION_MAPEMIT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace intent {

/// Libdevice-safe scalar atan(x) for f32, expanded entirely in arith (no
/// `math.atan`, which lowers to a `__nv_atanf` libdevice extern = ptxas dead
/// end, the same failure mode as exp/sin/erf; and NVVM has NO atan.approx
/// hardware op to peephole to). Reciprocal argument reduction onto |t|<=1 with
/// atan(t)=pi/2 - atan(1/t) for |t|>1, then a 7-term odd minimax polynomial on
/// [0,1] (max abs err ~7.7e-7 vs numpy, far inside the 1e-3 PK tolerance):
///   atan(z) ~ z*(c0 + z2*(c1 + z2*(c2 + ... + z2*c6))),  z2=z*z.
inline mlir::Value emitAtan(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value x) {
  using namespace mlir;
  Type ty = x.getType();
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(ty, v)).getResult();
  };
  // Minimax coefficients (lstsq fit of atan(z)/z on z^2, z in [0,1]).
  static constexpr double kC[7] = {
      0.9999965891394812,  -0.33319009688008516, 0.1982329286937468,
      -0.13294171203737687, 0.08076283609048057, -0.03461290793780861,
      0.007151299050017444};
  Value one = cst(1.0), zero = cst(0.0), halfPi = cst(1.5707963267948966);
  Value negx = b.create<arith::NegFOp>(loc, x);
  Value t = b.create<arith::MaxNumFOp>(loc, x, negx); // |x|
  Value gt1 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, t, one);
  Value recip = b.create<arith::DivFOp>(loc, one, t);
  Value z = b.create<arith::SelectOp>(loc, gt1, recip, t);
  Value z2 = b.create<arith::MulFOp>(loc, z, z);
  // Horner in z2: p = c6; p = p*z2 + c5; ... ; p = p*z2 + c0.
  Value p = cst(kC[6]);
  for (int i = 5; i >= 0; --i)
    p = b.create<arith::AddFOp>(loc, cst(kC[i]),
                                b.create<arith::MulFOp>(loc, p, z2));
  Value poly = b.create<arith::MulFOp>(loc, z, p); // atan of |x| (or 1/|x|)
  // |x|>1: atan(|x|) = pi/2 - atan(1/|x|).
  Value atanAbs = b.create<arith::SelectOp>(
      loc, gt1, b.create<arith::SubFOp>(loc, halfPi, poly), poly);
  // Restore the sign: atan is odd.
  Value neg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, zero);
  return b.create<arith::SelectOp>(
      loc, neg, b.create<arith::NegFOp>(loc, atanAbs), atanAbs);
}

/// Libdevice-safe scalar atan2(y, x) for f32: emitAtan(y/x) + the quadrant
/// correction from the signs of (y, x) (add/sub pi for x<0; the ±pi/2 axes for
/// x==0). Matches numpy/torch.atan2 to ~7.7e-7 abs err across all quadrants.
/// `angle(z) = atan2(imag, real)` lowers to a `map_binary(imag, real, "atan2")`.
inline mlir::Value emitAtan2(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value y, mlir::Value x) {
  using namespace mlir;
  Type ty = y.getType();
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(ty, v)).getResult();
  };
  Value zero = cst(0.0), pi = cst(3.141592653589793),
        halfPi = cst(1.5707963267948966), negHalfPi = cst(-1.5707963267948966);
  Value r = b.create<arith::DivFOp>(loc, y, x);
  Value a = emitAtan(b, loc, r);
  Value xlt0 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, zero);
  Value yge0 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, y, zero);
  Value piCorr =
      b.create<arith::SelectOp>(loc, yge0, pi, b.create<arith::NegFOp>(loc, pi));
  Value corr = b.create<arith::SelectOp>(loc, xlt0, piCorr, zero);
  Value res = b.create<arith::AddFOp>(loc, a, corr);
  // x==0 axes: y>0 -> +pi/2, y<0 -> -pi/2, y==0 -> 0.
  Value xeq0 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, x, zero);
  Value ygt0 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, y, zero);
  Value ylt0 = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, y, zero);
  Value axis = b.create<arith::SelectOp>(
      loc, ygt0, halfPi,
      b.create<arith::SelectOp>(loc, ylt0, negHalfPi, zero));
  return b.create<arith::SelectOp>(loc, xeq0, axis, res);
}

/// Scalar arith/math op for one `map_unary` fn. nullptr = unsupported here.
inline mlir::Value emitScalarUnary(mlir::OpBuilder &b, mlir::Location loc,
                                   llvm::StringRef fn, mlir::Value x) {
  using namespace mlir;
  if (fn == "exp")
    return b.create<math::ExpOp>(loc, x);
  if (fn == "log")
    return b.create<math::LogOp>(loc, x);
  if (fn == "sqrt")
    return b.create<math::SqrtOp>(loc, x);
  if (fn == "rsqrt")
    return b.create<math::RsqrtOp>(loc, x);
  if (fn == "abs") {
    // |x| = max(x, -x). math.absf lowers to a `__nv_fabsf` libdevice extern
    // (ptxas dead end, same failure mode as exp/tanh/sqrt); the arith form is
    // resolved entirely on-device.
    Value negx = b.create<arith::NegFOp>(loc, x);
    return b.create<arith::MaxNumFOp>(loc, x, negx);
  }
  if (fn == "neg")
    return b.create<arith::NegFOp>(loc, x);
  if (fn == "tanh") {
    // tanh(x) = 1 - 2/(exp(2x)+1). Routed through math.exp (-> ex2.approx.ftz
    // hardware instruction by the `-lower-gpu-math-to-nvvm` peephole) instead of
    // math.tanh, whose stock GpuToNVVM lowering emits an unresolved `__nv_tanhf`
    // libdevice extern (ptxas dead end, same failure mode as exp/sqrt). This
    // form is overflow-safe: large +x -> exp->inf -> 1-0 = 1; large -x -> exp->0
    // -> 1-2 = -1 (the (exp-1)/(exp+1) form would give inf/inf = NaN).
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value two =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 2.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::MulFOp>(loc, x, two));
    Value den = b.create<arith::AddFOp>(loc, e, one);
    return b.create<arith::SubFOp>(loc, one,
                                   b.create<arith::DivFOp>(loc, two, den));
  }
  if (fn == "recip") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::DivFOp>(loc, one, x);
  }
  if (fn == "sigmoid") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
    return b.create<arith::DivFOp>(loc, one,
                                   b.create<arith::AddFOp>(loc, one, e));
  }
  // cos/sin route through math.cos/math.sin, which the `-lower-gpu-math-to-nvvm`
  // peephole turns into the `cos.approx.f`/`sin.approx.f` PTX hardware ops
  // (the stock GpuToNVVM path emits `__nv_cosf`/`__nv_sinf` libdevice externs =
  // ptxas dead end, the same failure mode as exp/tanh/sqrt).
  if (fn == "cos")
    return b.create<math::CosOp>(loc, x);
  if (fn == "sin")
    return b.create<math::SinOp>(loc, x);
  if (fn == "tan") {
    // tan(x) = sin(x)/cos(x): stay off `__nv_tanf` by composing the two approx
    // hardware ops the peephole already provides.
    Value s = b.create<math::SinOp>(loc, x);
    Value c = b.create<math::CosOp>(loc, x);
    return b.create<arith::DivFOp>(loc, s, c);
  }
  if (fn == "exp2")
    // 2^x directly on the `ex2.approx.ftz.f` hardware op (no log2(e) prescale,
    // unlike exp) via the math.exp2 peephole.
    return b.create<math::Exp2Op>(loc, x);
  if (fn == "atan")
    // Libdevice-safe: minimax poly + reciprocal reduction, all in arith (no
    // __nv_atanf extern; NVVM has no atan.approx hardware op).
    return emitAtan(b, loc, x);
  if (fn == "erf") {
    // Abramowitz & Stegun 7.1.26: erf(|x|) = 1 - poly(t)*exp(-x^2),
    // t = 1/(1+p|x|), max abs err ~1.5e-7 — far inside the 1e-3 tolerance. Uses
    // math.exp (-> ex2.approx peephole) + arith only; NVVM has no erf approx and
    // math.erf would lower to a `__nv_erff` libdevice extern (ptxas dead end).
    Type ty = x.getType();
    auto cst = [&](double v) {
      return b.create<arith::ConstantOp>(loc, b.getFloatAttr(ty, v)).getResult();
    };
    Value one = cst(1.0), zero = cst(0.0);
    Value negx = b.create<arith::NegFOp>(loc, x);
    Value absx = b.create<arith::MaxNumFOp>(loc, x, negx);
    Value t = b.create<arith::DivFOp>(
        loc, one,
        b.create<arith::AddFOp>(loc, one,
                                b.create<arith::MulFOp>(loc, cst(0.3275911), absx)));
    // Horner: t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    Value poly = cst(1.061405429);
    poly = b.create<arith::AddFOp>(loc, cst(-1.453152027),
                                   b.create<arith::MulFOp>(loc, t, poly));
    poly = b.create<arith::AddFOp>(loc, cst(1.421413741),
                                   b.create<arith::MulFOp>(loc, t, poly));
    poly = b.create<arith::AddFOp>(loc, cst(-0.284496736),
                                   b.create<arith::MulFOp>(loc, t, poly));
    poly = b.create<arith::AddFOp>(loc, cst(0.254829592),
                                   b.create<arith::MulFOp>(loc, t, poly));
    poly = b.create<arith::MulFOp>(loc, t, poly);
    Value ex = b.create<math::ExpOp>(
        loc, b.create<arith::NegFOp>(loc,
                                     b.create<arith::MulFOp>(loc, absx, absx)));
    Value y = b.create<arith::SubFOp>(
        loc, one, b.create<arith::MulFOp>(loc, poly, ex));
    // erf(-x) = -erf(x): flip the sign for negative inputs.
    Value isneg = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, zero);
    return b.create<arith::SelectOp>(loc, isneg, b.create<arith::NegFOp>(loc, y),
                                     y);
  }
  if (fn == "isnan")
    // NaN is the only value unordered with itself. -> i1 (feeds intent.select,
    // e.g. nan_to_num = select(isnan(x), 0, x)). No libdevice.
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, x, x);
  if (fn == "logical_not") {
    // i1 complement: x XOR true.
    Value t = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    return b.create<arith::XOrIOp>(loc, x, t);
  }
  return nullptr;
}

/// Scalar arith op for one `map_binary` fn. nullptr = unsupported here.
inline mlir::Value emitScalarBinary(mlir::OpBuilder &b, mlir::Location loc,
                                    llvm::StringRef fn, mlir::Value a,
                                    mlir::Value c) {
  using namespace mlir;
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
  if (fn == "atan2")
    // atan2(y=a, x=c): libdevice-safe atan of the ratio + quadrant correction.
    // angle(z) = atan2(imag, real) lowers to map_binary(imag, real, "atan2").
    return emitAtan2(b, loc, a, c);
  return nullptr;
}

/// Is `fn` a `map_binary` comparison (gt/ge/lt/le/eq/ne -> i1)?
inline bool isCmpFn(llvm::StringRef fn) {
  return fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" ||
         fn == "eq" || fn == "ne";
}

/// Scalar arith.cmpf for one `map_binary` comparison fn (-> i1). nullptr if not
/// a comparison fn. Ordered predicates (NaN-false), matching numpy/triton.
inline mlir::Value emitScalarCmp(mlir::OpBuilder &b, mlir::Location loc,
                                 llvm::StringRef fn, mlir::Value a,
                                 mlir::Value c) {
  using namespace mlir;
  arith::CmpFPredicate p;
  if (fn == "gt")
    p = arith::CmpFPredicate::OGT;
  else if (fn == "ge")
    p = arith::CmpFPredicate::OGE;
  else if (fn == "lt")
    p = arith::CmpFPredicate::OLT;
  else if (fn == "le")
    p = arith::CmpFPredicate::OLE;
  else if (fn == "eq")
    p = arith::CmpFPredicate::OEQ;
  else if (fn == "ne")
    p = arith::CmpFPredicate::UNE;
  else
    return nullptr;
  return b.create<arith::CmpFOp>(loc, p, a, c);
}

/// Scalar arith.cmpi for one `map_binary` comparison fn over INTEGER operands
/// (-> i1). Used by cross_entropy's `iota == target` one-hot condition (column
/// index vs the per-row target class), where the operands are integers and the
/// float `arith.cmpf` form does not apply. Signed predicates.
inline mlir::Value emitScalarCmpI(mlir::OpBuilder &b, mlir::Location loc,
                                  llvm::StringRef fn, mlir::Value a,
                                  mlir::Value c) {
  using namespace mlir;
  arith::CmpIPredicate p;
  if (fn == "gt")
    p = arith::CmpIPredicate::sgt;
  else if (fn == "ge")
    p = arith::CmpIPredicate::sge;
  else if (fn == "lt")
    p = arith::CmpIPredicate::slt;
  else if (fn == "le")
    p = arith::CmpIPredicate::sle;
  else if (fn == "eq")
    p = arith::CmpIPredicate::eq;
  else if (fn == "ne")
    p = arith::CmpIPredicate::ne;
  else
    return nullptr;
  return b.create<arith::CmpIOp>(loc, p, a, c);
}

/// Per-arg semantic role, inferred purely from static shape vs (M,N):
///   Row[M,N]  — a full feature row, register-tiled read-once per thread;
///   Col[N]    — a per-column vector (gamma/W), loaded at col[k];
///   Scalar[]  — a rank-0 uniform (alpha/eps/constant), loaded once per block;
///   RowInt[M] — a per-row INTEGER (cross_entropy's target class Y), loaded
///               once per block (uniform across the block's threads) as Y[bx].
///   ColSeg[C] — a "broadcast-segment" column vector (group_norm's W/B): a 1-D
///               f32 arg whose value at feature column `col` of a block in group
///               `g` is loaded at the channel index given by an affine
///               `intent.col_index_map = affine_map<(group, col) -> channel>`
///               arg-attr (e.g. col floordiv SPATIAL + group*CPG). A structured
///               data-independent gather (the index is affine in (group,col)),
///               general over any affine-broadcast Col — NOT a per-kernel branch.
///   ParamVector[P] — a small parameter vector addressed only through a
///                    structurally validated scalar/per-column intent.gather.
enum class Role { Row, Col, Scalar, RowInt, PerRow, ColSeg, ParamVector };

/// Default block size over the feature dim: largest power-of-two <= min(N,256)
/// dividing-friendly. The PK harness passes the library kernel's geometry.
inline int64_t autoThreads(int64_t N) {
  int64_t t = 1;
  while (t * 2 <= N && t * 2 <= 256)
    t *= 2;
  return t;
}

//===----------------------------------------------------------------------===//
// M1 shared high-performance emit primitives (R-CODEGEN-1/2, R3 §3, R5 P1/P2).
//
// The "flat grid-stride" occupancy fix + tail-mask: instead of one block per
// output row (sub-warp blocks that leave most SMs idle on NCHW pool/interp),
// map the WHOLE flattened output element space onto a full-width block-stride
// grid so every SM stays busy and any N (no divisibility) lowers. The grid is
// sized to a few waves of a numSM-SM device; the `scf.for i = gtid..total step
// grid*T` loop tail-masks the ragged remainder for free (threads past `total`
// just run zero iterations).
//===----------------------------------------------------------------------===//

/// Grid blocks for a flat grid-stride kernel over `total` output elements with
/// `T` threads/block: enough to fill a ~numSM-SM device several waves deep, but
/// never more than needed. numSM defaults to the RTX 5090D (GB202, sm_120a).
inline int64_t flatGridBlocks(int64_t total, int64_t T, int64_t numSM = 170) {
  if (T <= 0)
    return 1;
  int64_t need = (total + T - 1) / T;
  int64_t threadsPerSM = 2048, waves = 4;
  int64_t cap = (waves * numSM * threadsPerSM + T - 1) / T;
  int64_t g = need < cap ? need : cap;
  return g < 1 ? 1 : g;
}

/// Effective block size for a flat kernel: honor an explicit `blockThreads`
/// hint (the PK harness / autotuner), else a full 256-thread block (8 warps).
inline int64_t flatThreads(unsigned blockThreads) {
  return blockThreads ? (int64_t)blockThreads : 256;
}

/// Emit `gtid = bx*T + tx; scf.for iv = gtid to total step (G*T) { ... }` and
/// leave `b` positioned at the start of the loop body. `iv` receives the flat
/// output element index. The loop bound (static `total`) is the tail-mask: no
/// `total % (G*T) == 0` requirement. Returns the loop op.
inline mlir::scf::ForOp emitGridStride(mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::Value bx, mlir::Value tx, int64_t T,
                                       int64_t G, int64_t total,
                                       mlir::Value &iv) {
  using namespace mlir;
  Value cT = b.create<arith::ConstantIndexOp>(loc, T);
  Value gtid =
      b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bx, cT), tx);
  Value ub = b.create<arith::ConstantIndexOp>(loc, total);
  Value step = b.create<arith::ConstantIndexOp>(loc, G * T);
  auto forOp = b.create<scf::ForOp>(loc, gtid, ub, step);
  b.setInsertionPointToStart(forOp.getBody());
  iv = forOp.getInductionVar();
  return forOp;
}

/// Unravel a flat row-major index `flat` into per-dimension index coords for a
/// tensor of static shape `dims` (coords[i] in [0,dims[i])). coords is resized.
inline void unravelIndex(mlir::OpBuilder &b, mlir::Location loc,
                         mlir::Value flat, llvm::ArrayRef<int64_t> dims,
                         llvm::SmallVectorImpl<mlir::Value> &coords) {
  using namespace mlir;
  coords.assign(dims.size(), nullptr);
  Value rem = flat;
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    Value d = b.create<arith::ConstantIndexOp>(loc, dims[i]);
    coords[i] = b.create<arith::RemUIOp>(loc, rem, d);
    if (i > 0)
      rem = b.create<arith::DivUIOp>(loc, rem, d);
  }
}

} // namespace intent

#endif // INTENT_CONVERSION_MAPEMIT_H
