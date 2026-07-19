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

#include <limits>

namespace intent {
#define GEN_PASS_DEF_LOWERGPUMATHTONVVM
#include "Target/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// log2(e) as an f32 — the change-of-base constant for exp(x) = 2^(x*log2e).
static constexpr double kLog2E = 1.4426950408889634;
/// ln(2) as an f32 — the change-of-base constant for log(x) = log2(x)*ln2.
static constexpr double kLn2 = 0.6931471805599453;

/// `math.log(%x) - %y` -> `fma(lg2.approx(%x), ln2, -%y)`.
///
/// The generic log rewrite would materialize `lg2 * ln2` followed by a
/// subtraction.  When the log has exactly this single consumer, one native FMA
/// preserves the same approximate-log policy with one fewer dependent FP32
/// instruction.  The match is purely SSA-structural and applies equally to any
/// row loss/divergence graph.
struct LogSubToLg2Fma : OpRewritePattern<arith::SubFOp> {
  explicit LogSubToLg2Fma(MLIRContext *ctx)
      : OpRewritePattern<arith::SubFOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter &rewriter) const override {
    auto log = op.getLhs().getDefiningOp<math::LogOp>();
    auto f32 = dyn_cast<Float32Type>(op.getType());
    if (!log || !log->hasOneUse() || !f32 || op.getRhs().getType() != f32)
      return rewriter.notifyMatchFailure(
          op, "requires a single-use scalar f32 log minuend");

    Location loc = op.getLoc();
    auto lg2 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.lg2.approx.f"),
        ValueRange{log.getOperand()});
    Value ln2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(kLn2));
    Value negativeRhs = rewriter.create<arith::NegFOp>(loc, op.getRhs());
    auto fused = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.fma"),
        ValueRange{lg2.getResult(0), ln2, negativeRhs});
    rewriter.replaceOp(op, fused.getResults());
    rewriter.eraseOp(log);
    return success();
  }
};

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

static bool isF32One(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto attr = dyn_cast<FloatAttr>(constant.getValue());
  return attr && attr.getType().isF32() && attr.getValueAsDouble() == 1.0;
}

/// `1 / (1 + ex2.approx(x))` -> `rcp.approx.ftz(1 + ex2.approx(x))`.
///
/// This is the normalized scalar shape of sigmoid after ExpToEx2Approx. The
/// matcher is deliberately structural and narrow: it does not inspect function
/// names, and it cannot rewrite unrelated reciprocal scales whose denominator
/// is not the native approximate exponential plus one.
struct SigmoidReciprocalToNVVMApprox : OpRewritePattern<arith::DivFOp> {
  explicit SigmoidReciprocalToNVVMApprox(MLIRContext *ctx)
      : OpRewritePattern<arith::DivFOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getType());
    auto add = op.getRhs().getDefiningOp<arith::AddFOp>();
    if (!f32 || !isF32One(op.getLhs()) || !add)
      return rewriter.notifyMatchFailure(
          op, "requires scalar f32 one divided by an addition");
    Value exponential;
    if (isF32One(add.getLhs()))
      exponential = add.getRhs();
    else if (isF32One(add.getRhs()))
      exponential = add.getLhs();
    else
      return rewriter.notifyMatchFailure(op, "denominator has no f32 one");
    auto intrinsic = exponential.getDefiningOp<LLVM::CallIntrinsicOp>();
    if (!intrinsic ||
        intrinsic.getIntrin() != "llvm.nvvm.ex2.approx.ftz.f")
      return rewriter.notifyMatchFailure(
          op, "denominator is not one plus native approximate exp2");
    auto reciprocal = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32,
        rewriter.getStringAttr("llvm.nvvm.rcp.approx.ftz.f"),
        ValueRange{op.getRhs()});
    rewriter.replaceOp(op, reciprocal.getResults());
    return success();
  }
};

/// `math.log %x : f32`  ->  `lg2.approx.f32(%x) * ln2`. The stock GpuToNVVM
/// lowering routes `math.log` to a `__nv_logf` libdevice extern (verified on
/// this path: ptxas dead end / DCE'd kernel, same failure mode as exp/sqrt).
/// `lg2.approx.f32` is the native PTX hardware op, emitted here as the
/// `llvm.nvvm.lg2.approx.f` intrinsic; log(x) = log2(x)*ln2 is the change of
/// base. kl_div/jsd's loss is a per-element log on log-probabilities, so the
/// approx rel err (~2^-23-class mantissa) is far inside the 1e-3 PK tolerance.
struct LogToLg2Approx : OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Location loc = op.getLoc();
    auto lg2 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.lg2.approx.f"),
        ValueRange{op.getOperand()});
    Value ln2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(kLn2));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lg2.getResult(0), ln2);
    return success();
  }
};

/// `math.log %x : f64` -> a self-contained range-reduced f64 polynomial.
///
/// Standalone ptxas cannot resolve the stock `__nv_log` libdevice call.  Unlike
/// f32, PTX has no native f64 lg2 instruction, so use IEEE-754 exponent/mantissa
/// decomposition and the atanh series
///
///   log(m) = 2 * (z + z^3/3 + ... + z^17/17), z = (m-1)/(m+1),
///
/// after reducing m to [1/sqrt(2), sqrt(2)].  On that interval |z|<=0.1716 and
/// nine terms are comfortably below double precision needs for the paper's f32
/// state promoted to an f64 LSE destination.  The implementation also preserves
/// zero, negative, infinity and NaN behavior without any external symbol.
struct LogF64Polynomial : OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    auto f64 = dyn_cast<Float64Type>(op.getOperand().getType());
    if (!f64)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f64");

    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    auto cF64 = [&](double value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF64FloatAttr(value));
    };
    auto cI64 = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, i64);
    };

    Value x = op.getOperand();
    Value zero = cF64(0.0);
    Value one = cF64(1.0);
    Value two = cF64(2.0);
    Value isZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, x, zero);
    Value isNegative = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, x, zero);

    // Normalize positive subnormals by 2^54, then compensate the exponent.
    Value originalBits = rewriter.create<arith::BitcastOp>(loc, i64, x);
    Value shift52 = cI64(52);
    Value exponentMask = cI64(0x7ff);
    Value originalExponent = rewriter.create<arith::AndIOp>(
        loc,
        rewriter.create<arith::ShRUIOp>(loc, originalBits, shift52),
        exponentMask);
    Value exponentZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, originalExponent, cI64(0));
    Value positive = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, x, zero);
    Value isSubnormal = rewriter.create<arith::AndIOp>(loc, exponentZero,
                                                       positive);
    Value scaledSubnormal = rewriter.create<arith::MulFOp>(
        loc, x, cF64(18014398509481984.0)); // 2^54
    Value normalized = rewriter.create<arith::SelectOp>(
        loc, isSubnormal, scaledSubnormal, x);

    Value bits = rewriter.create<arith::BitcastOp>(loc, i64, normalized);
    Value exponentBits = rewriter.create<arith::AndIOp>(
        loc, rewriter.create<arith::ShRUIOp>(loc, bits, shift52),
        exponentMask);
    Value isNonFinite = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, exponentBits, cI64(0x7ff));
    Value exponent = rewriter.create<arith::SubIOp>(loc, exponentBits,
                                                    cI64(1023));
    Value exponentSubnormal = rewriter.create<arith::SubIOp>(
        loc, exponent, cI64(54));
    exponent = rewriter.create<arith::SelectOp>(
        loc, isSubnormal, exponentSubnormal, exponent);

    constexpr uint64_t kMantissaMask = (uint64_t{1} << 52) - 1;
    constexpr uint64_t kOneExponent = uint64_t{1023} << 52;
    Value mantissaBits = rewriter.create<arith::OrIOp>(
        loc,
        rewriter.create<arith::AndIOp>(
            loc, bits, cI64(static_cast<int64_t>(kMantissaMask))),
        cI64(static_cast<int64_t>(kOneExponent)));
    Value mantissa = rewriter.create<arith::BitcastOp>(loc, f64, mantissaBits);
    Value aboveSqrt2 = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, mantissa,
        cF64(1.4142135623730950488));
    Value reducedMantissa = rewriter.create<arith::SelectOp>(
        loc, aboveSqrt2,
        rewriter.create<arith::MulFOp>(loc, mantissa, cF64(0.5)), mantissa);
    exponent = rewriter.create<arith::SelectOp>(
        loc, aboveSqrt2,
        rewriter.create<arith::AddIOp>(loc, exponent, cI64(1)), exponent);

    Value numerator = rewriter.create<arith::SubFOp>(loc, reducedMantissa, one);
    Value denominator = rewriter.create<arith::AddFOp>(loc, reducedMantissa, one);
    Value z = rewriter.create<arith::DivFOp>(loc, numerator, denominator);
    Value z2 = rewriter.create<arith::MulFOp>(loc, z, z);
    Value power = z;
    Value series = z;
    for (int denominatorValue = 3; denominatorValue <= 17;
         denominatorValue += 2) {
      power = rewriter.create<arith::MulFOp>(loc, power, z2);
      Value term = rewriter.create<arith::DivFOp>(
          loc, power, cF64(static_cast<double>(denominatorValue)));
      series = rewriter.create<arith::AddFOp>(loc, series, term);
    }
    Value logMantissa = rewriter.create<arith::MulFOp>(loc, two, series);
    Value exponentF64 =
        rewriter.create<arith::SIToFPOp>(loc, f64, exponent);
    Value exponentTerm = rewriter.create<arith::MulFOp>(
        loc, exponentF64, cF64(kLn2));
    Value approximate = rewriter.create<arith::AddFOp>(
        loc, exponentTerm, logMantissa);

    Value positiveOrSpecial = rewriter.create<arith::SelectOp>(
        loc, isNonFinite, x, approximate);
    Value negativeInfinity = cF64(-std::numeric_limits<double>::infinity());
    Value nan = cF64(std::numeric_limits<double>::quiet_NaN());
    Value zeroHandled = rewriter.create<arith::SelectOp>(
        loc, isZero, negativeInfinity, positiveOrSpecial);
    Value result = rewriter.create<arith::SelectOp>(
        loc, isNegative, nan, zeroHandled);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// `math.OP %x : f32` -> `llvm.call_intrinsic "<intr>"(%x)` for the NVVM
/// approximate transcendentals. Same libdevice-dead-end fix as exp: the stock
/// GpuToNVVM lowering routes `math.sqrt`/`math.rsqrt` to `__nv_sqrtf`/
/// `__nv_rsqrtf` externs, which ptxas cannot resolve (or the integrated
/// serializer DCEs the kernel). The `*.approx.f` PTX ops compute it in hardware;
/// measured rel vs fp32-ref is ~1.3e-7 (rms_norm's rstd is one value/row, error
/// negligible) — the same class of fix, applied to the row-reduction normalizer.
template <typename MathOp>
struct UnaryToNVVMApprox : OpRewritePattern<MathOp> {
  UnaryToNVVMApprox(MLIRContext *ctx, StringRef intrinsic)
      : OpRewritePattern<MathOp>(ctx), intrinsic(intrinsic) {}

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr(intrinsic),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }

  StringRef intrinsic;
};

/// `math.exp2 %x : f32` -> `ex2.approx.ftz.f32(%x)`. exp2 IS the native 2^x
/// hardware op with NO change-of-base prescale (unlike exp, which multiplies by
/// log2(e) first). The stock GpuToNVVM path routes math.exp2 to a `__nv_exp2f`
/// libdevice extern (ptxas dead end, same failure mode as exp/log/sqrt).
struct Exp2ToEx2Approx : OpRewritePattern<math::Exp2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Exp2Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32,
        rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }
};

/// `math.fma %a, %b, %c : f32` -> LLVM's target-independent fused multiply-add
/// intrinsic.  Leaving the math op to the stock GPU pipeline produces a
/// `__nv_fmaf` libdevice extern, which has the same unresolved-symbol failure
/// as the transcendental calls above.  `llvm.fma` lowers directly to PTX
/// `fma.rn.f32` and preserves the single-rounding semantics requested by the
/// row-reduction accumulator.
struct FmaToLLVMIntrinsic : OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getA().getType());
    if (!f32 || op.getB().getType() != f32 || op.getC().getType() != f32)
      return rewriter.notifyMatchFailure(op, "operands are not scalar f32");
    auto fused = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr("llvm.fma"),
        ValueRange{op.getA(), op.getB(), op.getC()});
    rewriter.replaceOp(op, fused.getResults());
    return success();
  }
};

struct LowerGpuMathToNVVM
    : ::intent::impl::LowerGpuMathToNVVMBase<LowerGpuMathToNVVM> {
  using LowerGpuMathToNVVMBase::LowerGpuMathToNVVMBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LogSubToLg2Fma>(&getContext());
    patterns.add<ExpToEx2Approx>(&getContext());
    patterns.add<SigmoidReciprocalToNVVMApprox>(&getContext());
    patterns.add<LogToLg2Approx>(&getContext());
    patterns.add<LogF64Polynomial>(&getContext());
    patterns.add<Exp2ToEx2Approx>(&getContext());
    patterns.add<FmaToLLVMIntrinsic>(&getContext());
    patterns.add<UnaryToNVVMApprox<math::SqrtOp>>(&getContext(),
                                                  "llvm.nvvm.sqrt.approx.f");
    patterns.add<UnaryToNVVMApprox<math::RsqrtOp>>(&getContext(),
                                                   "llvm.nvvm.rsqrt.approx.f");
    // sin/cos: the `sin.approx.f`/`cos.approx.f` PTX hardware ops (math.sin /
    // math.cos would otherwise become `__nv_sinf`/`__nv_cosf` libdevice externs).
    patterns.add<UnaryToNVVMApprox<math::SinOp>>(&getContext(),
                                                 "llvm.nvvm.sin.approx.f");
    patterns.add<UnaryToNVVMApprox<math::CosOp>>(&getContext(),
                                                 "llvm.nvvm.cos.approx.f");
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
