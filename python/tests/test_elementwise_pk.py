"""PK the mature pure-elementwise lowering vs the REAL liger kernels on the 5090D.

dyt / swiglu / geglu are lowered from atomic IntentIR through the C++
``-convert-intent-elementwise-to-gpu`` pass to a read-once block-cooperative
sm_120a cubin, then launched and compared head-to-head with the real liger
forward launcher (the raw ``*_forward`` path, bypassing the autograd Function)
for (1) correctness (rel < 1e-3 vs an fp32 torch reference AND vs liger's own
output) and (2) kernel-only median latency.

These are read-once / write-once memory-bound kernels (read inputs once, write
the output once), so the honest bar is PARITY at peak HBM (~1507 GB/s) — there
are no bytes left to remove, and liger already hoists its per-program scalars
(dyt loads ``alpha`` once outside any column loop), so there is no named beat
mechanism here. The perf assertion is loose (sub-ms kernels carry ~8% clock
noise on the throttled "D" card); the real ratio + GB/s are printed.

The intent MLIR is hand-written here (M3 is lowering-side: the IR has the same
shape as test/lower-elementwise-to-gpu.mlir), not extracted from the corpus.
intent.select / intent.cast are deferred to M3b, so the fp32-no-op ``.cast`` in
swiglu/geglu's source is simply absent here (both sides run fp32, apples to
apples).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_elementwise_pk.py -q -s
"""

from __future__ import annotations

import math
import os
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]

_PASS = "convert-intent-elementwise-to-gpu"
SQRT_2_OVER_PI = 0.7978845608028654


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _T(N):
    """Block size over the feature dim, mirroring liger's thread geometry."""
    from liger_kernel.ops.utils import calculate_settings
    _, num_warps = calculate_settings(N)
    return num_warps * 32


# ---------------------------------------------------------------------------- #
# Hand-written intent MLIR (parameterized by M, N) — same shape as the lit test.
# ---------------------------------------------------------------------------- #

def _dyt_mlir(M, N):
    R = f"tensor<{M}x{N}xf32>"
    return f"""
func.func @dyt(%X: {R}, %Alpha: tensor<f32>, %Gamma: tensor<{N}xf32>) -> {R} {{
  %ax = intent.map_binary %Alpha, %X {{fn = "mul"}} : (tensor<f32>, {R}) -> {R}
  %t = intent.map_unary %ax {{fn = "tanh"}} : ({R}) -> {R}
  %y = intent.map_binary %t, %Gamma {{fn = "mul"}} : ({R}, tensor<{N}xf32>) -> {R}
  return %y : {R}
}}
"""


def _swiglu_mlir(M, N):
    R = f"tensor<{M}x{N}xf32>"
    return f"""
func.func @swiglu(%A: {R}, %B: {R}) -> {R} {{
  %s = intent.map_unary %A {{fn = "sigmoid"}} : ({R}) -> {R}
  %silu = intent.map_binary %A, %s {{fn = "mul"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %silu, %B {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %c : {R}
}}
"""


def _geglu_mlir(M, N):
    R = f"tensor<{M}x{N}xf32>"
    return f"""
func.func @geglu(%A: {R}, %B: {R}, %half: tensor<f32>, %k: tensor<f32>,
                 %sp: tensor<f32>, %one: tensor<f32>) -> {R} {{
  %a2 = intent.map_binary %A, %A {{fn = "mul"}} : ({R}, {R}) -> {R}
  %a3 = intent.map_binary %a2, %A {{fn = "mul"}} : ({R}, {R}) -> {R}
  %ka3 = intent.map_binary %k, %a3 {{fn = "mul"}} : (tensor<f32>, {R}) -> {R}
  %inner = intent.map_binary %A, %ka3 {{fn = "add"}} : ({R}, {R}) -> {R}
  %arg = intent.map_binary %sp, %inner {{fn = "mul"}} : (tensor<f32>, {R}) -> {R}
  %th = intent.map_unary %arg {{fn = "tanh"}} : ({R}) -> {R}
  %opt = intent.map_binary %one, %th {{fn = "add"}} : (tensor<f32>, {R}) -> {R}
  %ha = intent.map_binary %half, %A {{fn = "mul"}} : (tensor<f32>, {R}) -> {R}
  %gate = intent.map_binary %ha, %opt {{fn = "mul"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %gate, %B {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %c : {R}
}}
"""


def _kernel(name, M, N, mlir_text):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(name, {"M": M, "N": N}, _T(N),
                              pass_name=_PASS, mlir_text=mlir_text)


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


def _report(tag, M, N, gb, tm, tl):
    print(f"\n{tag} ({M},{N}) T={_T(N)}: mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f}")


# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_dyt_pk_vs_liger(M, N):
    from liger_kernel.ops.dyt import liger_dyt_fwd

    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    alpha = _scalar(0.5)
    gamma = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()

    k = _kernel("dyt", M, N, _dyt_mlir(M, N))
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    go = k.thunk([X, alpha, gamma, Y])  # sig order: X, Alpha, Gamma, then Y
    go()
    torch.cuda.synchronize()

    ref = torch.tanh(alpha * X) * gamma
    assert _rel(Y, ref) < 1e-3
    yl = liger_dyt_fwd(X, alpha, gamma, None)
    torch.cuda.synchronize()
    assert _rel(Y, yl) < 1e-3

    def liger():
        liger_dyt_fwd(X, alpha, gamma, None)

    gb = 2 * M * N * 4 / 1e9  # read X + write Y (gamma/alpha negligible)
    tm, tl = _bench(go), _bench(liger)
    _report("dyt", M, N, gb, tm, tl)
    assert tm <= tl * 1.5


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_swiglu_pk_vs_liger(M, N):
    from liger_kernel.ops.swiglu import swiglu_forward

    torch.manual_seed(0)
    A = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()

    k = _kernel("swiglu", M, N, _swiglu_mlir(M, N))
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    go = k.thunk([A, B, C])
    go()
    torch.cuda.synchronize()

    ref = (A * torch.sigmoid(A)) * B
    assert _rel(C, ref) < 1e-3
    _, _, cl = swiglu_forward(A.clone(), B.clone())
    torch.cuda.synchronize()
    assert _rel(C, cl) < 1e-3

    def liger():
        swiglu_forward(A, B)

    gb = 3 * M * N * 4 / 1e9  # read A + read B + write C
    tm, tl = _bench(go), _bench(liger)
    _report("swiglu", M, N, gb, tm, tl)
    assert tm <= tl * 1.5


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_geglu_pk_vs_liger(M, N):
    from liger_kernel.ops.geglu import geglu_forward

    torch.manual_seed(0)
    A = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()

    k = _kernel("geglu", M, N, _geglu_mlir(M, N))
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    go = k.thunk([A, B, _scalar(0.5), _scalar(0.044715),
                  _scalar(SQRT_2_OVER_PI), _scalar(1.0), C])
    go()
    torch.cuda.synchronize()

    ref = 0.5 * A * (1.0 + torch.tanh(SQRT_2_OVER_PI * (A + 0.044715 * A * A * A))) * B
    assert _rel(C, ref) < 1e-3
    _, _, cl = geglu_forward(A.clone(), B.clone())
    torch.cuda.synchronize()
    assert _rel(C, cl) < 1e-3

    def liger():
        geglu_forward(A, B)

    gb = 3 * M * N * 4 / 1e9  # read A + read B + write C
    tm, tl = _bench(go), _bench(liger)
    _report("geglu", M, N, gb, tm, tl)
    assert tm <= tl * 1.5
