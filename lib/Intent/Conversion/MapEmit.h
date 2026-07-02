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
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

namespace intent {

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
  return nullptr;
}

/// Is `fn` a `map_binary` comparison (gt/ge/lt/le/eq -> i1)?
inline bool isCmpFn(llvm::StringRef fn) {
  return fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" || fn == "eq";
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
enum class Role { Row, Col, Scalar, RowInt, ColSeg };

/// Default block size over the feature dim: largest power-of-two <= min(N,256)
/// dividing-friendly. The PK harness passes the library kernel's geometry.
inline int64_t autoThreads(int64_t N) {
  int64_t t = 1;
  while (t * 2 <= N && t * 2 <= 256)
    t *= 2;
  return t;
}

} // namespace intent

#endif // INTENT_CONVERSION_MAPEMIT_H
