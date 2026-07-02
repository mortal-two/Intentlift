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
          "inference batch-norm. input X shape [M, N] (N features); "
          "per-feature (X-running_mean)/sqrt(running_var+eps)*weight+bias.",
          lambda: [rnd(M, N)],
          lambda t: flag_gems.batch_norm(
              t[0], weight=torch.ones(N, device=dev),
              bias=torch.zeros(N, device=dev),
              running_mean=torch.zeros(N, device=dev),
              running_var=torch.ones(N, device=dev), training=False),
          family="norm"),
        S("weight_norm", "flag_gems", _f("weightnorm"),
          "inputs IN THIS ORDER: V shape [M, N]; G shape [1, N]; "
          "weight-norm along dim=0: V * G / ||V||_dim0.",
          lambda: [rnd(M, N), rnd(1, N)],
          lambda t: flag_gems.weight_norm(t[0], t[1], dim=0), family="norm"),
        S("instance_norm", "flag_gems", _ff("instance_norm"),
          "instance-norm. input X shape [Nn, C, L]; normalize each (n,c) over "
          "the spatial axis L: (X-mean)/sqrt(var+eps).",
          lambda: [rnd(4, 8, 32)],
          lambda t: flag_gems.instance_norm(t[0]), family="norm"),
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
        S("reglu", "flag_gems", _ff("reglu"),
          "single input X shape [M, N] with N even; split X into halves A,B "
          "along the last axis and compute relu(A)*B -> [M, N/2].",
          lambda: [rnd(M, N)], lambda t: flag_gems.reglu(t[0]), family="fused"),
    ]
    return specs
