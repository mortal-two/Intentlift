"""PK the multi-output poly_norm lowering vs the REAL liger kernel on 5090D.

poly_norm (liger) normalizes ONE read-once x register tile at THREE powers and
sums the weighted norms:
    norm(u) = u * rsqrt(mean(u*u) + eps)
    Y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b
It is lowered from atomic IntentIR through the C++
``-convert-intent-row-reduction-to-gpu`` pass (ZERO new mechanism — three
last-axis mean reduces + multi-output epilogue, the exact same path as
rms_norm/layer_norm) to ONE read-once block-cooperative sm_120a cubin with FOUR
destination-passing outputs: Y[M,N] (from the register tile) plus the three
backward-cache rstd values as THREE separate per-row [M] outputs (rstd3, rstd2,
rstd1, matching liger's RSTD[:,0:3] columns), each written once per block by
thread 0. Compared head-to-head with the real liger Triton kernel for
correctness (rel < 1e-3 on every output, vs fp32 ref AND vs liger) and
kernel-only median latency.

poly_norm is a SINGLE-pass read-once single-block-per-row memory-bound kernel
(liger launches one program per row, reads X once, writes Y once) — so the
honest bar is PARITY (~1500 GB/s peak HBM, no HBM traffic left to remove). A
beat would require a named mechanism (none here: both already read-once), so we
state parity and assert mine <= liger * 1.5 within run-to-run clock noise.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_poly_norm_pk.py -q -s
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

_PASS = "convert-intent-row-reduction-to-gpu"


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _poly_mlir(M, N):
    """Atomic poly_norm intent DAG (parameterized M,N): one read-once X tile,
    three mean-square reduces (powers x^3, x^2, x), Y RowOut + three per-row
    rstd PerRowOut outputs (order rstd3, rstd2, rstd1 = liger RSTD columns)."""
    R = f"tensor<{M}x{N}xf32>"
    S = f"tensor<{M}x1xf32>"
    return f"""
func.func @poly_norm(%X: {R}, %w0: tensor<f32>, %w1: tensor<f32>, %w2: tensor<f32>, %b: tensor<f32>, %eps: tensor<f32>)
    -> ({R}, {S}, {S}, {S}) {{
  %xp2 = intent.map_binary %X, %X {{fn = "mul"}} : ({R}, {R}) -> {R}
  %xp3 = intent.map_binary %xp2, %X {{fn = "mul"}} : ({R}, {R}) -> {R}
  %sq3 = intent.map_binary %xp3, %xp3 {{fn = "mul"}} : ({R}, {R}) -> {R}
  %ms3 = intent.reduce %sq3 {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {S}
  %ms3e = intent.map_binary %ms3, %eps {{fn = "add"}} : ({S}, tensor<f32>) -> {S}
  %rstd3 = intent.map_unary %ms3e {{fn = "rsqrt"}} : ({S}) -> {S}
  %norm3 = intent.map_binary %xp3, %rstd3 {{fn = "mul"}} : ({R}, {S}) -> {R}
  %t3 = intent.map_binary %norm3, %w0 {{fn = "mul"}} : ({R}, tensor<f32>) -> {R}
  %sq2 = intent.map_binary %xp2, %xp2 {{fn = "mul"}} : ({R}, {R}) -> {R}
  %ms2 = intent.reduce %sq2 {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {S}
  %ms2e = intent.map_binary %ms2, %eps {{fn = "add"}} : ({S}, tensor<f32>) -> {S}
  %rstd2 = intent.map_unary %ms2e {{fn = "rsqrt"}} : ({S}) -> {S}
  %norm2 = intent.map_binary %xp2, %rstd2 {{fn = "mul"}} : ({R}, {S}) -> {R}
  %t2 = intent.map_binary %norm2, %w1 {{fn = "mul"}} : ({R}, tensor<f32>) -> {R}
  %ms1 = intent.reduce %xp2 {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {S}
  %ms1e = intent.map_binary %ms1, %eps {{fn = "add"}} : ({S}, tensor<f32>) -> {S}
  %rstd1 = intent.map_unary %ms1e {{fn = "rsqrt"}} : ({S}) -> {S}
  %norm1 = intent.map_binary %X, %rstd1 {{fn = "mul"}} : ({R}, {S}) -> {R}
  %t1 = intent.map_binary %norm1, %w2 {{fn = "mul"}} : ({R}, tensor<f32>) -> {R}
  %s01 = intent.map_binary %t3, %t2 {{fn = "add"}} : ({R}, {R}) -> {R}
  %s012 = intent.map_binary %s01, %t1 {{fn = "add"}} : ({R}, {R}) -> {R}
  %Y = intent.map_binary %s012, %b {{fn = "add"}} : ({R}, tensor<f32>) -> {R}
  return %Y, %rstd3, %rstd2, %rstd1 : {R}, {S}, {S}, {S}
}}
"""


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_poly_norm_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.poly_norm import _poly_norm_forward_kernel
    from liger_kernel.ops.utils import calculate_settings

    BLOCK_SIZE, num_warps = calculate_settings(N)
    T = num_warps * 32  # match liger's per-thread column count K = N/T
    assert N % T == 0, f"N={N} not divisible by T={T}"
    eps = 1e-6

    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    w0, w1, w2, bv = 0.7, -0.3, 1.1, 0.05

    # --- mine: compile + launch the read-once 4-output kernel -------------- #
    k = RowReductionKernel("poly_norm", {"M": M, "N": N}, T,
                           pass_name=_PASS, mlir_text=_poly_mlir(M, N))
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    rstd3 = torch.empty(M, device="cuda", dtype=torch.float32)
    rstd2 = torch.empty(M, device="cuda", dtype=torch.float32)
    rstd1 = torch.empty(M, device="cuda", dtype=torch.float32)
    w0d = torch.tensor(w0, device="cuda", dtype=torch.float32)
    w1d = torch.tensor(w1, device="cuda", dtype=torch.float32)
    w2d = torch.tensor(w2, device="cuda", dtype=torch.float32)
    bd = torch.tensor(bv, device="cuda", dtype=torch.float32)
    epsd = torch.tensor(eps, device="cuda", dtype=torch.float32)
    # sig order after the pass: X, w0, w1, w2, b, eps, then Y, rstd3, rstd2, rstd1
    go = k.thunk([X, w0d, w1d, w2d, bd, epsd, Y, rstd3, rstd2, rstd1])
    go()
    torch.cuda.synchronize()

    # --- fp32 torch reference --------------------------------------------- #
    def _norm(u):
        return u * torch.rsqrt((u * u).mean(-1, keepdim=True) + eps)

    def _rstd(u):
        return torch.rsqrt((u * u).mean(-1) + eps)

    Y_ref = w0 * _norm(X ** 3) + w1 * _norm(X ** 2) + w2 * _norm(X) + bv
    assert _rel(Y, Y_ref) < 1e-3
    assert _rel(rstd3, _rstd(X ** 3)) < 1e-3
    assert _rel(rstd2, _rstd(X ** 2)) < 1e-3
    assert _rel(rstd1, _rstd(X)) < 1e-3

    # --- liger: call the RAW forward kernel directly (skip autograd) ------- #
    W = torch.tensor([w0, w1, w2], device="cuda", dtype=torch.float32)
    B = torch.tensor([bv], device="cuda", dtype=torch.float32)
    Yl = torch.empty(M, N, device="cuda", dtype=torch.float32)
    RSTDl = torch.empty(M, 3, device="cuda", dtype=torch.float32)

    def liger():
        _poly_norm_forward_kernel[(M,)](
            Yl, Yl.stride(0), X, X.stride(0), W, B,
            RSTDl, RSTDl.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    liger()
    torch.cuda.synchronize()
    # head-to-head vs the real liger kernel (RSTD cols are rstd3, rstd2, rstd1)
    assert _rel(Y, Yl) < 1e-3
    assert _rel(rstd3, RSTDl[:, 0]) < 1e-3
    assert _rel(rstd2, RSTDl[:, 1]) < 1e-3
    assert _rel(rstd1, RSTDl[:, 2]) < 1e-3

    # --- kernel-only median latency, no host sync in the timed region ----- #
    tm, tl = _bench(go), _bench(liger)
    gb = 2 * M * N * 4 / 1e9  # read X + write Y (the 3 rstd[M] writes negligible)
    print(f"\npoly_norm ({M},{N}) T={T} num_warps={num_warps}: "
          f"mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f} "
          f"(single-pass read-once single-block-per-row -> honest PARITY floor)")
    assert tm <= tl * 1.5
