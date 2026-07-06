"""Stratified pilot spec registry for the C1 composability-measurement gate.

20-30 REAL FlagGems ops sampled across the coverage families (unary / binary /
reduce / norm / softmax / init / fused). Each entry names the real source module
(read by the LLM lifter) + a runnable ``flag_gems.<op>`` wrapper for the numeric
verify. The pilot MEASURES, per op:

    extracted?  (LLM produced valid IntentIR after bounded-repair)
    lowered?    (valid IntentIR -> zero residual intent.* via convert-intent-to-linalg
                 == the composability gate)
    verified?   (interpret(JSON) numerically == the real flag_gems kernel, rel<1e-3)

Extraction + lowering do NOT depend on the run()/inputs() wrappers being correct
(they read only source_file + shape_hint), so the composability signal is clean
even where a wrapper is best-effort; only the verify column is affected.
"""

from __future__ import annotations

import warnings

from .corpus import FG_OPS, KernelSpec

warnings.filterwarnings("ignore")

FG_FUSED = FG_OPS.parent / "fused"


def _f(name: str) -> str:
    return str(FG_OPS / f"{name}.py")


def _ff(name: str) -> str:
    return str(FG_FUSED / f"{name}.py")


def pilot_specs() -> list[KernelSpec]:
    import flag_gems
    import torch

    dev = "cuda"

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.float32)

    def pos(*shape):  # strictly-positive (safe divisor / log)
        return torch.rand(*shape, device=dev, dtype=torch.float32) + 0.5

    M, N = 64, 256
    S = KernelSpec
    specs: list[KernelSpec] = [
        # --- unary (validates the C0 scalar-fn/NVVM entries end-to-end) ---------
        S("cos", "flag_gems", _f("cos"),
          "single input X shape [M, N]; cosine elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.cos(t[0]), family="unary"),
        S("sin", "flag_gems", _f("sin"),
          "single input X shape [M, N]; sine elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.sin(t[0]), family="unary"),
        S("erf", "flag_gems", _f("erf"),
          "single input X shape [M, N]; error function erf elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.erf(t[0]), family="unary"),
        S("elu", "flag_gems", _f("elu"),
          "single input X shape [M, N]; ELU with alpha=1: "
          "select(x>0, x, exp(x)-1), elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.elu(t[0]), family="unary"),
        S("softplus", "flag_gems", _f("softplus"),
          "single input X shape [M, N]; softplus log(1+exp(x)) elementwise "
          "(beta=1).",
          lambda: [rnd(M, N)], lambda t: flag_gems.softplus(t[0]), family="unary"),
        # --- binary --------------------------------------------------------------
        S("addcdiv", "flag_gems", _f("addcdiv"),
          "inputs IN THIS ORDER: INP, T1, T2 all shape [M, N]; "
          "elementwise INP + value*(T1/T2) with value=0.5.",
          lambda: [rnd(M, N), rnd(M, N), pos(M, N)],
          lambda t: flag_gems.addcdiv(t[0], t[1], t[2], value=0.5),
          family="binary"),
        S("addcmul", "flag_gems", _f("addcmul"),
          "inputs IN THIS ORDER: INP, T1, T2 all shape [M, N]; "
          "elementwise INP + value*(T1*T2) with value=0.5.",
          lambda: [rnd(M, N), rnd(M, N), rnd(M, N)],
          lambda t: flag_gems.addcmul(t[0], t[1], t[2], value=0.5),
          family="binary"),
        S("mse_loss", "flag_gems", _f("mse_loss"),
          "inputs IN THIS ORDER: A, B both shape [M, N]; mean of (A-B)^2 over "
          "all elements -> a scalar (reduction='mean').",
          lambda: [rnd(M, N), rnd(M, N)],
          lambda t: flag_gems.mse_loss(t[0], t[1]), family="binary"),
        S("eq", "flag_gems", _f("eq"),
          "inputs IN THIS ORDER: A, B both shape [M, N]; elementwise A==B "
          "producing an i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.eq(t[0], t[1]),
          family="binary"),
        # --- reduce --------------------------------------------------------------
        S("amax", "flag_gems", _f("amax"),
          "single input X shape [M, N]; maximum over ALL elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.amax(t[0]), family="reduce"),
        S("vector_norm", "flag_gems", _f("vector_norm"),
          "single input X shape [M, N]; L2 norm sqrt(sum(X^2)) over ALL "
          "elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.vector_norm(t[0]),
          family="reduce"),
        S("std", "flag_gems", _f("std"),
          "single input X shape [M, N]; population/unbiased standard deviation "
          "over ALL elements -> scalar. Emit the std as the single output.",
          lambda: [rnd(M, N)],
          lambda t: flag_gems.std(t[0], correction=0), family="reduce"),
        S("var_mean", "flag_gems", _f("var_mean"),
          "single input X shape [M, N]; variance over ALL elements -> scalar. "
          "Emit the VARIANCE as the single output.",
          lambda: [rnd(M, N)],
          lambda t: flag_gems.var_mean(t[0], correction=0)[0], family="reduce"),
        S("count_nonzero", "flag_gems", _f("count_nonzero"),
          "single input X shape [M, N]; count of nonzero elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.count_nonzero(t[0]),
          family="reduce"),
        # --- norm ----------------------------------------------------------------
        S("batch_norm", "flag_gems", _f("batch_norm"),
          "INFERENCE batch-norm with PRE-COMPUTED running statistics supplied as "
          "inputs (NOT computed from X) -> a pure elementwise-with-broadcast op. "
          "inputs IN THIS ORDER: X shape [M, N] (N feature columns); weight [1, N]; "
          "bias [1, N]; running_mean [1, N]; running_var [1, N] (positive). "
          "For each feature column j: Y[i,j] = "
          "(X[i,j] - running_mean[0,j]) / sqrt(running_var[0,j] + eps) * "
          "weight[0,j] + bias[0,j], eps=1e-5. Emit Y as the single output.",
          lambda: [rnd(M, N), rnd(1, N), rnd(1, N), rnd(1, N), pos(1, N)],
          lambda t: flag_gems.batch_norm(
              t[0], weight=t[1].squeeze(0), bias=t[2].squeeze(0),
              running_mean=t[3].squeeze(0), running_var=t[4].squeeze(0),
              training=False, eps=1e-5),
          family="norm"),
        S("weight_norm", "flag_gems", _f("weightnorm"),
          "inputs IN THIS ORDER: V shape [M, N]; G shape [1, N]. For EACH column "
          "j independently: norm[j] = sqrt(sum over rows i of V[i,j]^2) (an L2 "
          "reduction over axis 0, keepdims -> [1, N]); output Y[i,j] = "
          "V[i,j] / norm[j] * G[0,j]. Emit Y as the single output.",
          lambda: [rnd(M, N), rnd(1, N)],
          lambda t: flag_gems.weight_norm(t[0], t[1], dim=1), family="norm"),
        S("instance_norm", "flag_gems", _ff("instance_norm"),
          "instance-norm over a 2-D input X shape [G, L] (G = Nn*C instances, "
          "L = spatial length). For EACH row g independently: mean[g] = "
          "average over the L columns; var[g] = average over columns of "
          "(X[g,:]-mean[g])^2; Y[g,l] = (X[g,l]-mean[g])/sqrt(var[g]+eps), "
          "eps=1e-5. mean/var are per-row reductions over axis 1 (keepdims "
          "-> [G, 1]). Emit Y as the single output.",
          lambda: [rnd(32, 64)],
          lambda t: (lambda x: (x - x.mean(1, keepdim=True))
                     / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-5)
                     )(t[0]),
          family="norm"),
        # --- softmax: covered by the 15-kernel corpus (softmax, log_softmax) --
        # deliberately NOT in the pilot: the pilot samples the UNTESTED universe,
        # and re-running flag_gems.softmax twice/process hits the known Triton-3.6
        # launcher arg-count bug (poisons the corpus test's JIT cache).
        # --- init ----------------------------------------------------------------
        S("arange", "flag_gems", _f("arange"),
          "no tensor input; produce the 1-D sequence [0,1,...,END-1] as f32 "
          "with END=64 (an iota).",
          lambda: [], lambda t: flag_gems.arange(64, device="cuda").float(),
          family="init"),
        S("full", "flag_gems", _f("full"),
          "no tensor input; produce a [M, N] tensor filled with the constant "
          "3.0 (a const splat).",
          lambda: [], lambda t: flag_gems.full((M, N), 3.0, device="cuda"),
          family="init"),
        # --- fused ---------------------------------------------------------------
        S("silu_and_mul", "flag_gems", _ff("silu_and_mul"),
          "inputs IN THIS ORDER: A, B both shape [M, N]; silu(A)*B elementwise "
          "(silu(x)=x*sigmoid(x)).",
          lambda: [rnd(M, N), rnd(M, N)],
          lambda t: flag_gems.silu_and_mul(t[0], t[1]), family="fused"),
        S("gelu_and_mul", "flag_gems", _ff("gelu_and_mul"),
          "inputs IN THIS ORDER: A, B both shape [M, N]; gelu(A)*B elementwise "
          "(use the exact/erf GELU 0.5*x*(1+erf(x/sqrt(2)))).",
          lambda: [rnd(M, N), rnd(M, N)],
          lambda t: flag_gems.gelu_and_mul(t[0], t[1]), family="fused"),
        # reglu is a SPLIT op: the kernel takes one tensor X=[A|B] and computes
        # relu(A)*B. The lifter naturally declares the two halves A,B as separate
        # inputs (a split has no pure-tensor "slice a contiguous half" primitive),
        # so the verify wrapper feeds the two halves and reconstructs X=cat([A,B])
        # for the real kernel -> the 2-input lift and the wrapper arity agree.
        S("reglu", "flag_gems", _ff("reglu"),
          "inputs IN THIS ORDER: A, B both shape [M, H] (H = N/2). Compute "
          "relu(A)*B elementwise -> [M, H]. (The real kernel receives the "
          "concatenation [A | B] along the last axis and splits it internally.)",
          lambda: [rnd(M, N // 2), rnd(M, N // 2)],
          lambda t: flag_gems.reglu(torch.cat([t[0], t[1]], dim=-1)),
          family="fused"),
    ]
    return specs


def bulk_specs() -> list[KernelSpec]:
    """C2b SCALE list: the extraction-only bulk across ALL families, disjoint
    from the 22-op pilot. Each op names its real FlagGems source module + a
    runnable ``flag_gems.<op>`` wrapper. Extraction + lowering are scored from
    source+shape_hint only; a best-effort wrapper affects only the verify column.
    """
    import flag_gems
    import torch

    dev = "cuda"

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.float32)

    def pos(*shape):  # strictly positive (safe log / sqrt / divisor / base)
        return torch.rand(*shape, device=dev, dtype=torch.float32) + 0.5

    def small(*shape):  # bounded magnitude (safe prod, safe pow exponent)
        return torch.rand(*shape, device=dev, dtype=torch.float32) * 0.5 + 0.75

    M, N, K = 64, 256, 128
    S = KernelSpec
    specs: list[KernelSpec] = [
        # ---- unary (scalar-fn / elementwise-composite lifts) ----------------
        S("tanh", "flag_gems", _f("tanh"),
          "single input X shape [M, N]; hyperbolic tangent elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.tanh(t[0]), family="unary"),
        S("sigmoid", "flag_gems", _f("sigmoid"),
          "single input X shape [M, N]; logistic sigmoid 1/(1+exp(-x)).",
          lambda: [rnd(M, N)], lambda t: flag_gems.sigmoid(t[0]), family="unary"),
        S("silu", "flag_gems", _f("silu"),
          "single input X shape [M, N]; SiLU x*sigmoid(x) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.silu(t[0]), family="unary"),
        S("gelu", "flag_gems", _f("gelu"),
          "single input X shape [M, N]; exact/erf GELU "
          "0.5*x*(1+erf(x/sqrt(2))) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.gelu(t[0]), family="unary"),
        S("relu", "flag_gems", _f("relu"),
          "single input X shape [M, N]; ReLU max(x, 0) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.relu(t[0]), family="unary"),
        S("abs", "flag_gems", _f("abs"),
          "single input X shape [M, N]; absolute value elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.abs(t[0]), family="unary"),
        S("neg", "flag_gems", _f("neg"),
          "single input X shape [M, N]; negation -x elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.neg(t[0]), family="unary"),
        S("exp", "flag_gems", _f("exp"),
          "single input X shape [M, N]; exp(x) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.exp(t[0]), family="unary"),
        S("sqrt", "flag_gems", _f("sqrt"),
          "single input X shape [M, N] (positive); sqrt(x) elementwise.",
          lambda: [pos(M, N)], lambda t: flag_gems.sqrt(t[0]), family="unary"),
        S("log", "flag_gems", _f("log"),
          "single input X shape [M, N] (positive); natural log(x) elementwise.",
          lambda: [pos(M, N)], lambda t: flag_gems.log(t[0]), family="unary"),
        S("reciprocal", "flag_gems", _f("reciprocal"),
          "single input X shape [M, N] (nonzero); reciprocal 1/x elementwise.",
          lambda: [pos(M, N)], lambda t: flag_gems.reciprocal(t[0]), family="unary"),
        S("rsqrt", "flag_gems", _f("rsqrt"),
          "single input X shape [M, N] (positive); rsqrt 1/sqrt(x) elementwise.",
          lambda: [pos(M, N)], lambda t: flag_gems.rsqrt(t[0]), family="unary"),
        S("atan", "flag_gems", _f("atan"),
          "single input X shape [M, N]; arctangent atan(x) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.atan(t[0]), family="unary"),
        S("tan", "flag_gems", _f("tan"),
          "single input X shape [M, N]; tangent tan(x) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.tan(t[0]), family="unary"),
        S("exp2", "flag_gems", _f("exp2"),
          "single input X shape [M, N]; base-2 exponential 2**x elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.exp2(t[0]), family="unary"),
        S("celu", "flag_gems", _f("celu"),
          "single input X shape [M, N]; CELU with alpha=1: "
          "max(0,x)+min(0,exp(x)-1) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.celu(t[0]), family="unary"),
        S("log_sigmoid", "flag_gems", _f("log_sigmoid"),
          "single input X shape [M, N]; log-sigmoid log(1/(1+exp(-x))) "
          "= -softplus(-x) elementwise.",
          lambda: [rnd(M, N)], lambda t: flag_gems.log_sigmoid(t[0]),
          family="unary"),
        # ---- binary elementwise ---------------------------------------------
        S("add", "flag_gems", _f("add"),
          "inputs A, B both shape [M, N]; elementwise A+B.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.add(t[0], t[1]),
          family="binary"),
        S("sub", "flag_gems", _f("sub"),
          "inputs A, B both shape [M, N]; elementwise A-B.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.sub(t[0], t[1]),
          family="binary"),
        S("mul", "flag_gems", _f("mul"),
          "inputs A, B both shape [M, N]; elementwise A*B.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.mul(t[0], t[1]),
          family="binary"),
        S("div", "flag_gems", _f("div"),
          "inputs A shape [M, N], B shape [M, N] (nonzero); elementwise A/B.",
          lambda: [rnd(M, N), pos(M, N)],
          lambda t: flag_gems.true_divide(t[0], t[1]), family="binary"),
        S("maximum", "flag_gems", _f("maximum"),
          "inputs A, B both shape [M, N]; elementwise max(A, B).",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.maximum(t[0], t[1]),
          family="binary"),
        S("minimum", "flag_gems", _f("minimum"),
          "inputs A, B both shape [M, N]; elementwise min(A, B).",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.minimum(t[0], t[1]),
          family="binary"),
        S("pow", "flag_gems", _f("pow"),
          "inputs A shape [M, N] (positive base), B shape [M, N] (exponent); "
          "elementwise A**B.",
          lambda: [pos(M, N), small(M, N)],
          lambda t: flag_gems.pow_tensor_tensor(t[0], t[1]), family="binary"),
        S("ge", "flag_gems", _f("ge"),
          "inputs A, B both shape [M, N]; elementwise A>=B -> i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.ge(t[0], t[1]),
          family="binary"),
        S("gt", "flag_gems", _f("gt"),
          "inputs A, B both shape [M, N]; elementwise A>B -> i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.gt(t[0], t[1]),
          family="binary"),
        S("lt", "flag_gems", _f("lt"),
          "inputs A, B both shape [M, N]; elementwise A<B -> i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.lt(t[0], t[1]),
          family="binary"),
        S("le", "flag_gems", _f("le"),
          "inputs A, B both shape [M, N]; elementwise A<=B -> i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.le(t[0], t[1]),
          family="binary"),
        S("ne", "flag_gems", _f("ne"),
          "inputs A, B both shape [M, N]; elementwise A!=B -> i1 mask.",
          lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.ne(t[0], t[1]),
          family="binary"),
        # ---- reduce (full reduction to scalar) ------------------------------
        S("sum", "flag_gems", _f("sum"),
          "single input X shape [M, N]; sum over ALL elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.sum(t[0]), family="reduce"),
        S("mean", "flag_gems", _f("mean"),
          "single input X shape [M, N]; mean over ALL elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.mean(t[0]), family="reduce"),
        S("prod", "flag_gems", _f("prod"),
          "single input X shape [M, N]; product over ALL elements -> scalar.",
          lambda: [small(M, N)], lambda t: flag_gems.prod(t[0]), family="reduce"),
        S("max", "flag_gems", _f("max"),
          "single input X shape [M, N]; maximum over ALL elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.max(t[0]), family="reduce"),
        S("min", "flag_gems", _f("min"),
          "single input X shape [M, N]; minimum over ALL elements -> scalar.",
          lambda: [rnd(M, N)], lambda t: flag_gems.min(t[0]), family="reduce"),
        # ---- blas / contraction ---------------------------------------------
        S("mm", "flag_gems", _f("mm"),
          "inputs A shape [M, K], B shape [K, N]; matrix product A@B -> [M, N].",
          lambda: [rnd(M, K), rnd(K, N)], lambda t: flag_gems.mm(t[0], t[1]),
          family="blas"),
        S("bmm", "flag_gems", _f("bmm"),
          "inputs A shape [B, M, K], B shape [B, K, N]; batched matmul -> "
          "[B, M, N]. Use B=4.",
          lambda: [rnd(4, M, K), rnd(4, K, N)],
          lambda t: flag_gems.bmm(t[0], t[1]), family="blas"),
        S("addmm", "flag_gems", _f("addmm"),
          "inputs IN THIS ORDER: C shape [M, N], A shape [M, K], B shape [K, N]; "
          "compute C + A@B (beta=alpha=1) -> [M, N].",
          lambda: [rnd(M, N), rnd(M, K), rnd(K, N)],
          lambda t: flag_gems.addmm(t[0], t[1], t[2]), family="blas"),
        S("mv", "flag_gems", _f("mv"),
          "inputs A shape [M, K], v shape [K]; matrix-vector product A@v -> [M].",
          lambda: [rnd(M, K), rnd(K)], lambda t: flag_gems.mv(t[0], t[1]),
          family="blas"),
        S("outer", "flag_gems", _ff("outer"),
          "inputs a shape [M], b shape [N]; outer product a[:,None]*b[None,:] "
          "-> [M, N].",
          lambda: [rnd(M), rnd(N)], lambda t: flag_gems.outer(t[0], t[1]),
          family="blas"),
        # ---- init / const-splat / iota --------------------------------------
        S("zeros", "flag_gems", _f("zeros"),
          "no tensor input; produce a [M, N] f32 tensor of all zeros.",
          lambda: [], lambda t: flag_gems.zeros((M, N), device="cuda"),
          family="init"),
        S("ones", "flag_gems", _f("ones"),
          "no tensor input; produce a [M, N] f32 tensor of all ones.",
          lambda: [], lambda t: flag_gems.ones((M, N), device="cuda"),
          family="init"),
        S("eye", "flag_gems", _f("eye"),
          "no tensor input; produce the [M, M] identity matrix (1 on diagonal, "
          "0 elsewhere) as f32 with M=64.",
          lambda: [], lambda t: flag_gems.eye(M, device="cuda"), family="init"),
        S("linspace", "flag_gems", _f("linspace"),
          "no tensor input; produce 64 evenly spaced f32 values from 0.0 to 1.0 "
          "inclusive (a 1-D length-64 tensor).",
          lambda: [], lambda t: flag_gems.linspace(0.0, 1.0, 64, device="cuda"),
          family="init"),
        S("full_like", "flag_gems", _f("full_like"),
          "single input X shape [M, N]; produce a [M, N] tensor filled with the "
          "constant 3.0 (same shape as X).",
          lambda: [rnd(M, N)], lambda t: flag_gems.full_like(t[0], 3.0),
          family="init"),
        # ---- fused / gated ---------------------------------------------------
        S("geglu", "flag_gems", _ff("geglu"),
          "inputs A, B both shape [M, N]; gelu(A)*B elementwise (exact/erf GELU).",
          lambda: [rnd(M, N), rnd(M, N)],
          lambda t: flag_gems.geglu(t[0], t[1]), family="fused"),
        S("glu", "flag_gems", _f("glu"),
          "single input X shape [M, N] with N even; split X into halves A,B "
          "along the last axis and compute A*sigmoid(B) -> [M, N/2].",
          lambda: [rnd(M, N)], lambda t: flag_gems.glu(t[0], -1), family="fused"),
    ]
    return specs
