//===- LowerGpuMathToNVVM.cpp - math transcendentals -> NVVM intrinsics ---===//
//
// Backend codegen peephole for the NVIDIA target, run on the outlined device
// module just before `--gpu-lower-to-nvvm-pipeline`.
//
// The stock GpuToNVVM lowering turns `math.exp` into a `__nv_expf` libdevice
// call. On system MLIR 20.1.8 that path is unusable for us: without libdevice
// on the search path ptxas fails with "Unresolved extern function '__nv_expf'";
// with libdevice linked, the integrated serializer internalizes + DCEs the
// kernel entry and emits a cubin with no `.text` section (driver load rc 200).
//
// We sidestep libdevice entirely by computing exp with the native hardware
// instruction: exp(x) = 2^(x * log2(e)), where the 2^ is the PTX
// `ex2.approx.ftz.f32` op, emitted here as the `llvm.nvvm.ex2.approx.ftz.f`
// LLVM intrinsic. Verified on an RTX 5090 D (sm_120a): max rel err ~1.7e-7 vs
// torch.exp, well inside the softmax tolerance flash attention needs.
//
//===----------------------------------------------------------------------===//

#include "Target/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace intent {
#define GEN_PASS_DEF_LOWERGPUMATHTONVVM
#include "Target/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// log2(e) as an f32 — the change-of-base constant for exp(x) = 2^(x*log2e).
static constexpr double kLog2E = 1.4426950408889634;

/// `math.exp %x : f32`  ->  `ex2.approx.ftz.f32 (%x * log2e)`.
struct ExpToEx2Approx : OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    // ex2.approx.ftz.f is a scalar f32 instruction; only handle that form.
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");

    Location loc = op.getLoc();
    Value log2e = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(kLog2E));
    Value scaled =
        rewriter.create<arith::MulFOp>(loc, op.getOperand(), log2e);
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{scaled});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }
};

struct LowerGpuMathToNVVM
    : ::intent::impl::LowerGpuMathToNVVMBase<LowerGpuMathToNVVM> {
  using LowerGpuMathToNVVMBase::LowerGpuMathToNVVMBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpToEx2Approx>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
