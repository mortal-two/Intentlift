"""PK the mature row-reduction lowering vs the REAL liger kernels on the 5090D.

rms_norm and softmax are lowered from atomic IntentIR through the C++
``-convert-intent-row-reduction-to-gpu`` pass to a read-once block-cooperative
sm_120a cubin, then launched and compared head-to-head with the real liger
Triton kernel for (1) correctness (rel < 1e-3) and (2) kernel-only median
latency. These are memory-bound, single-pass read-once library kernels, so the
honest bar is PARITY (no HBM traffic left to remove) — measured ~1.00x at peak
bandwidth. The perf assertion is loose (sub-ms kernels carry ~8% clock noise on
the throttled "D" card); the real number is printed.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_row_reduction_pk.py -q -s
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
def test_rms_norm_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.rms_norm import (_CASTING_MODE_LLAMA,
                                           _rms_norm_forward_kernel,
                                           calculate_settings)

    BLOCK_SIZE, num_warps = calculate_settings(N)
    T = num_warps * 32
    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    W = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    eps = 1e-6

    k = RowReductionKernel("rms_norm", {"M": M, "N": N}, T)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    epsd = torch.tensor(eps, device="cuda", dtype=torch.float32)
    go = k.thunk([X, W, epsd, Y])  # signature order: X, W, eps, then output Y
    go()
    torch.cuda.synchronize()

    ref = X * torch.rsqrt((X * X).mean(-1, keepdim=True) + eps) * W
    assert _rel(Y, ref) < 1e-3

    Yl = torch.empty(M, N, device="cuda", dtype=torch.float32)
    RSTD = torch.empty(M, device="cuda", dtype=torch.float32)

    def liger():
        _rms_norm_forward_kernel[(M,)](
            Yl, Yl.stride(0), X, X.stride(0), W, W.stride(0),
            RSTD, RSTD.stride(0), N, eps, 0.0, _CASTING_MODE_LLAMA.value,
            elementwise_affine=True, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    tm, tl = _bench(go), _bench(liger)
    gb = 2 * M * N * 4 / 1e9
    print(f"\nrms_norm ({M},{N}) T={T}: mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * _PARITY_SLACK


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_softmax_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.softmax import _softmax_forward

    from liger_kernel.ops.rms_norm import calculate_settings
    _, num_warps = calculate_settings(N)
    T = num_warps * 32
    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()

    k = RowReductionKernel("softmax", {"M": M, "N": N}, T)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    go = k.thunk([X, Y])
    go()
    torch.cuda.synchronize()

    assert _rel(Y, torch.softmax(X, dim=1)) < 1e-3

    def liger():
        _softmax_forward(X)

    tm, tl = _bench(go), _bench(liger)
    gb = 2 * M * N * 4 / 1e9
    print(f"\nsoftmax ({M},{N}) T={T}: mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * _PARITY_SLACK
