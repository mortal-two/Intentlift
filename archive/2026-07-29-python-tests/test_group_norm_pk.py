"""PK the mature group_norm lowering vs the REAL liger _group_norm_forward_kernel.

group_norm is layer_norm's mean+var (here the NON-centered E[X^2]-E[X]^2 form
liger computes) applied per (batch, group): rows are flattened (B,G) -> M = B*G,
each of hidden = (C//G)*spatial elements; the reduce is EXACTLY layer_norm. The
ONLY new thing is the affine W/B index: gamma/beta are Col[C] gathered at
channel = group*CPG + col//SPATIAL (group = row % G). That gather is expressed
GENERALLY via a ColSeg arg carrying an `intent.col_index_map` affine-map arg-attr
(no per-kernel branch). The intent DAG is hand-written here (the corpus has no
group_norm) and lowered through the same C++ ``-convert-intent-row-reduction-to-
gpu`` pass to ONE read-once block-cooperative sm_120a cubin with THREE outputs
(Y[M,N] + per-row Mean[M]/RSTD[M]).

Memory-bound single-pass read-once -> honest PARITY floor (~1500 GB/s); a beat
would need a named mechanism. We assert Y/Mean/RSTD rel<1e-3 vs an fp32 torch
reference AND vs the real liger kernel.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_group_norm_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

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


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _choose_T(N: int) -> int:
    """Largest power-of-two T <= 256 dividing N with K = N/T <= 32 (single-tile,
    matching the M1 register-tile path the ColSeg gather rides on)."""
    best = 1
    t = 1
    while t <= 256:
        if N % t == 0 and N // t <= 32:
            best = t
        t *= 2
    return best


def _group_norm_intent(M: int, N: int, C: int, cpg: int, spatial: int) -> str:
    """Hand-written group_norm intent DAG (parameterized B,G,C,spatial). The W/B
    ColSeg args carry the affine channel map (group, col) -> col//spatial +
    group*cpg; the pass recovers G = C/cpg so group = row % G."""
    amap = (f"affine_map<(d0, d1) -> (d1 floordiv {spatial} + d0 * {cpg})>")
    return f"""
func.func @group_norm(
    %X: tensor<{M}x{N}xf32>,
    %W: tensor<{C}xf32> {{intent.col_index_map = {amap}}},
    %B: tensor<{C}xf32> {{intent.col_index_map = {amap}}},
    %eps: tensor<f32>) -> (tensor<{M}x{N}xf32>, tensor<{M}x1xf32>, tensor<{M}x1xf32>) {{
  %m = intent.reduce %X {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x1xf32>
  %xsq = intent.map_binary %X, %X {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %exsq = intent.reduce %xsq {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x1xf32>
  %m2 = intent.map_binary %m, %m {{fn = "mul"}} : (tensor<{M}x1xf32>, tensor<{M}x1xf32>) -> tensor<{M}x1xf32>
  %var = intent.map_binary %exsq, %m2 {{fn = "sub"}} : (tensor<{M}x1xf32>, tensor<{M}x1xf32>) -> tensor<{M}x1xf32>
  %vare = intent.map_binary %var, %eps {{fn = "add"}} : (tensor<{M}x1xf32>, tensor<f32>) -> tensor<{M}x1xf32>
  %rstd = intent.map_unary %vare {{fn = "rsqrt"}} : (tensor<{M}x1xf32>) -> tensor<{M}x1xf32>
  %xc = intent.map_binary %X, %m {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<{M}x1xf32>) -> tensor<{M}x{N}xf32>
  %xn = intent.map_binary %xc, %rstd {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<{M}x1xf32>) -> tensor<{M}x{N}xf32>
  %xnw = intent.map_binary %xn, %W {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<{C}xf32>) -> tensor<{M}x{N}xf32>
  %Y = intent.map_binary %xnw, %B {{fn = "add"}} : (tensor<{M}x{N}xf32>, tensor<{C}xf32>) -> tensor<{M}x{N}xf32>
  return %Y, %m, %rstd : tensor<{M}x{N}xf32>, tensor<{M}x1xf32>, tensor<{M}x1xf32>
}}
"""


@pytest.mark.parametrize("B,C,G,spatial", [(8, 256, 32, 64), (4, 128, 16, 128)])
def test_group_norm_pk_vs_liger(B, C, G, spatial):
    import torch.nn.functional as F

    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.group_norm import _group_norm_forward_kernel

    cpg = C // G               # channels per group
    hidden = cpg * spatial     # = N, the per-(batch,group) row length
    M = B * G
    N = hidden
    T = _choose_T(N)
    assert N % T == 0
    eps = 1e-5

    torch.manual_seed(0)
    # x is (batch, C, spatial); group_norm normalizes over (C//G, spatial).
    x = torch.randn(B, C, spatial, device="cuda", dtype=torch.float32).contiguous()
    W = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()
    Bias = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()

    # row-flat view: contiguous [B][C][spatial] -> [B*G, cpg*spatial] = [M, N]
    X = x.reshape(M, N).contiguous()

    # --- mine: compile + launch the read-once multi-output kernel ---------- #
    src = _group_norm_intent(M, N, C, cpg, spatial)
    k = RowReductionKernel("group_norm", {}, T, mlir_text=src)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    Mean = torch.empty(M, device="cuda", dtype=torch.float32)
    RSTD = torch.empty(M, device="cuda", dtype=torch.float32)
    epsd = torch.tensor(eps, device="cuda", dtype=torch.float32)
    # func-arg order after the pass: X, W, B, eps, then outputs Y, Mean, RSTD
    go = k.thunk([X, W, Bias, epsd, Y, Mean, RSTD])
    go()
    torch.cuda.synchronize()

    # --- fp32 torch reference --------------------------------------------- #
    Y_ref = F.group_norm(x, G, weight=W, bias=Bias, eps=eps)
    xr = x.reshape(B, G, hidden)
    mean_ref = xr.mean(-1).reshape(M)
    var_ref = (xr * xr).mean(-1).reshape(M) - mean_ref * mean_ref
    rstd_ref = torch.rsqrt(var_ref + eps)
    assert _rel(Y.reshape(B, C, spatial), Y_ref) < 1e-3
    assert _rel(Mean, mean_ref) < 1e-3
    assert _rel(RSTD, rstd_ref) < 1e-3

    # --- liger: call the RAW forward kernel directly (skip autograd) ------- #
    BLOCK_SIZE = min(65536, triton.next_power_of_2(hidden))
    Xl = X.reshape(B, G, hidden).contiguous()
    Yl = torch.empty((B, G, hidden), device="cuda", dtype=torch.float32)
    Meanl = torch.zeros((B, G), device="cuda", dtype=torch.float32)
    RSTDl = torch.zeros((B, G), device="cuda", dtype=torch.float32)

    def liger():
        _group_norm_forward_kernel[(B, G)](
            Yl, Yl.stride(0), Yl.stride(1),
            Xl, Xl.stride(0), Xl.stride(1),
            Meanl, Meanl.stride(0), Meanl.stride(1),
            RSTDl, RSTDl.stride(0), RSTDl.stride(1),
            W, Bias,
            hidden, cpg, eps, BLOCK_SIZE=BLOCK_SIZE)

    liger()
    torch.cuda.synchronize()
    # head-to-head vs liger (the real kernel), and liger sanity vs the ref.
    assert _rel(Yl.reshape(B, C, spatial), Y_ref) < 1e-3
    assert _rel(Y.reshape(B, G, hidden), Yl) < 1e-3
    assert _rel(Mean, Meanl.reshape(M)) < 1e-3
    assert _rel(RSTD, RSTDl.reshape(M)) < 1e-3

    # --- kernel-only median latency, no host sync in the timed region ------ #
    tm, tl = _bench(go), _bench(liger)
    gb = 2 * M * N * 4 / 1e9  # read X + write Y (the M-element stats negligible)
    print(f"\ngroup_norm B={B} C={C} G={G} spatial={spatial} (M={M},N={N}) T={T}: "
          f"mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * 1.5
