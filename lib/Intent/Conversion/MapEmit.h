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
  if (fn == "abs")
    return b.create<math::AbsFOp>(loc, x);
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

/// Per-arg semantic role, inferred purely from static shape vs (M,N):
///   Row[M,N]  — a full feature row, register-tiled read-once per thread;
///   Col[N]    — a per-column vector (gamma/W), loaded at col[k];
///   Scalar[]  — a rank-0 uniform (alpha/eps/constant), loaded once per block.
enum class Role { Row, Col, Scalar };

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
