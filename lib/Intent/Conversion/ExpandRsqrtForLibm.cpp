//===- ExpandRsqrtForLibm.cpp - host-safe reciprocal square root ---------===//
//
// CPU-only compatibility step for the ordinary host libm ABI.  The shared
// Intent-to-Linalg pass deliberately preserves math.rsqrt so GPU pipelines can
// still select their native target lowering.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace intent {
#define GEN_PASS_DEF_EXPANDRSQRTFORLIBM
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

struct ExpandRsqrtForLibmPass
    : ::intent::impl::ExpandRsqrtForLibmBase<ExpandRsqrtForLibmPass> {
  using ExpandRsqrtForLibmBase::ExpandRsqrtForLibmBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateExpandRsqrtPattern(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
