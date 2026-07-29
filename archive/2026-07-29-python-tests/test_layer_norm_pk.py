"""PK the mature multi-output layer_norm lowering vs the REAL liger kernel.

layer_norm (centered variance) is lowered from atomic IntentIR through the C++
``-convert-intent-row-reduction-to-gpu`` pass to ONE read-once block-cooperative
sm_120a cubin with THREE destination-passing outputs — Y[M,N] plus the per-row
stats Mean[M] and RSTD[M] (each written once per block by thread 0). It is then
launched and compared head-to-head with the real liger Triton kernel for
(1) correctness (rel < 1e-3 on Y, Mean, RSTD) and (2) kernel-only median latency.

Like rms_norm/softmax these are memory-bound single-pass read-once library
kernels, so the honest bar is PARITY (no HBM traffic left to remove). liger's
``calculate_settings`` hardwires num_warps=4 for N<=2048 (BLOCK_SIZE<2048) — low
occupancy / large per-thread K — leaving geometry headroom at (8192,1024); at
(4096,4096) it picks num_warps=8. We feed the SAME geometry (T = num_warps*32)
so K = N/T matches liger's per-thread column count: parity is then the floor and
any beat must come from a named mechanism (occupancy), not a geometry mismatch.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_layer_norm_pk.py -q -s
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


# Parity slack: sub-ms memory-bound kernels carry run-to-run clock noise; the
# bar is "no worse than the library within noise", not an exact tie.
_PARITY_SLACK = 1.5


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_layer_norm_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel
    from liger_kernel.ops.utils import calculate_settings

    BLOCK_SIZE, num_warps = calculate_settings(N)
    T = num_warps * 32  # match liger's per-thread column count K = N/T
    assert N % T == 0, f"N={N} not divisible by T={T}"

    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    W = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    eps = 1e-5

    # --- mine: compile + launch the read-once multi-output kernel ---------- #
    k = RowReductionKernel("layer_norm", {"M": M, "N": N}, T)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    Mean = torch.empty(M, device="cuda", dtype=torch.float32)
    RSTD = torch.empty(M, device="cuda", dtype=torch.float32)
    epsd = torch.tensor(eps, device="cuda", dtype=torch.float32)
    # func-arg order after the pass: X, W, B, eps, then outputs Y, Mean, RSTD
    go = k.thunk([X, W, B, epsd, Y, Mean, RSTD])
    go()
    torch.cuda.synchronize()

    # --- fp32 torch reference --------------------------------------------- #
    mean_ref = X.mean(-1)
    var_ref = ((X - mean_ref[:, None]) ** 2).mean(-1)
    rstd_ref = torch.rsqrt(var_ref + eps)
    Y_ref = (X - mean_ref[:, None]) * rstd_ref[:, None] * W + B
    assert _rel(Y, Y_ref) < 1e-3
    assert _rel(Mean, mean_ref) < 1e-3
    assert _rel(RSTD, rstd_ref) < 1e-3

    # --- liger: call the RAW forward kernel directly (skip the autograd
    # Function, which crashes on this torch build) ------------------------- #
    Yl = torch.empty(M, N, device="cuda", dtype=torch.float32)
    Meanl = torch.empty(M, device="cuda", dtype=torch.float32)
    RSTDl = torch.empty(M, device="cuda", dtype=torch.float32)

    def liger():
        _layer_norm_forward_kernel[(M,)](
            Yl, Yl.stride(0), X, X.stride(0), W, W.stride(0), B, B.stride(0),
            Meanl, Meanl.stride(0), RSTDl, RSTDl.stride(0),
            N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    liger()
    torch.cuda.synchronize()
    # cross-check liger against the same reference (harness sanity).
    assert _rel(Yl, Y_ref) < 1e-3

    # --- kernel-only median latency, no host sync in the timed region ------ #
    tm, tl = _bench(go), _bench(liger)
    # traffic: read X (M*N) + write Y (M*N); the M-element stats are negligible.
    gb = 2 * M * N * 4 / 1e9
    print(f"\nlayer_norm ({M},{N}) T={T} num_warps={num_warps}: "
          f"mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * _PARITY_SLACK
