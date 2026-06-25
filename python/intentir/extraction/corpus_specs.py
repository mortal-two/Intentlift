"""The concrete kernel registry (real FlagGems sources + runnable entries).

Input order is pinned by each kernel's ``shape_hint`` so the positional mapping
(provided arrays -> the LLM's declared inputs) is deterministic. eps is passed to
the library as a python float but lifted as a rank-0 scalar input.
"""

from __future__ import annotations

import warnings

from .corpus import FG_OPS, KernelSpec

warnings.filterwarnings("ignore")


def _f(name: str) -> str:
    return str(FG_OPS / f"{name}.py")


def kernel_specs() -> list[KernelSpec]:
    import flag_gems
    import torch

    dev = "cuda"

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.float32)

    def pos(*shape):  # positive inputs (for log)
        return torch.rand(*shape, device=dev, dtype=torch.float32) + 0.5

    M, N, K = 64, 256, 96
    specs: list[KernelSpec] = [
        KernelSpec(
            "softmax", "flag_gems", _f("softmax"),
            "single input X shape [M, N]; row softmax over the last axis (axis 1).",
            lambda: [rnd(M, N)], lambda t: flag_gems.softmax(t[0], dim=1),
            note="reduce/exp/div, numerically stabilized"),
        KernelSpec(
            "log_softmax", "flag_gems", _f("log_softmax"),
            "single input X shape [M, N]; log-softmax over the last axis (axis 1).",
            lambda: [rnd(M, N)], lambda t: flag_gems.log_softmax(t[0], dim=1),
            note="logsumexp"),
        KernelSpec(
            "silu", "flag_gems", _f("silu"),
            "single input X shape [M, N]; SiLU/swish x*sigmoid(x), elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.silu(t[0])),
        KernelSpec(
            "gelu_tanh", "flag_gems", _f("gelu"),
            "single input X shape [M, N]; lift the TANH approximation of GELU "
            "(0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))). Name it gelu_tanh.",
            lambda: [rnd(M, N)],
            lambda t: flag_gems.gelu(t[0], approximate="tanh")),
        KernelSpec(
            "relu", "flag_gems", _f("relu"),
            "single input X shape [M, N]; ReLU max(x, 0) elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.relu(t[0])),
        KernelSpec(
            "rms_norm", "flag_gems", _f("rms_norm"),
            "inputs IN THIS ORDER: X shape [M, N]; W shape [N]; eps scalar shape []. "
            "RMSNorm over the last axis: X * rsqrt(mean(X^2, axis=1) + eps) * W.",
            lambda: [rnd(M, N), rnd(N), torch.tensor(1e-5, device=dev)],
            lambda t: flag_gems.rms_norm(t[0], (N,), t[1], float(t[2].item())),
            note="mean/rsqrt + affine"),
        KernelSpec(
            "matmul", "flag_gems", _f("mm"),
            "inputs IN THIS ORDER: A shape [M, K]; B shape [K, N]. "
            "Matrix multiply A@B -> [M, N] (a contraction).",
            lambda: [rnd(M, K), rnd(K, N)], lambda t: flag_gems.mm(t[0], t[1])),
        KernelSpec(
            "sigmoid", "flag_gems", _f("sigmoid"),
            "single input X shape [M, N]; sigmoid elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.sigmoid(t[0])),
        KernelSpec(
            "tanh", "flag_gems", _f("tanh"),
            "single input X shape [M, N]; tanh elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.tanh(t[0])),
        KernelSpec(
            "exp", "flag_gems", _f("exp"),
            "single input X shape [M, N]; exp elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.exp(t[0])),
        KernelSpec(
            "abs", "flag_gems", _f("abs"),
            "single input X shape [M, N]; absolute value elementwise.",
            lambda: [rnd(M, N)], lambda t: flag_gems.abs(t[0])),
        KernelSpec(
            "log", "flag_gems", _f("log"),
            "single input X shape [M, N] (positive); natural log elementwise.",
            lambda: [pos(M, N)], lambda t: flag_gems.log(t[0])),
        KernelSpec(
            "add", "flag_gems", _f("add"),
            "inputs IN THIS ORDER: A, B both shape [M, N]; elementwise A+B.",
            lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.add(t[0], t[1])),
        KernelSpec(
            "mul", "flag_gems", _f("mul"),
            "inputs IN THIS ORDER: A, B both shape [M, N]; elementwise A*B.",
            lambda: [rnd(M, N), rnd(M, N)], lambda t: flag_gems.mul(t[0], t[1])),
        KernelSpec(
            "maximum", "flag_gems", _f("maximum"),
            "inputs IN THIS ORDER: A, B both shape [M, N]; elementwise max(A,B).",
            lambda: [rnd(M, N), rnd(M, N)],
            lambda t: flag_gems.maximum(t[0], t[1])),
    ]
    return specs
