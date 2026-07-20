"""PK the mature tvd lowering vs the REAL liger `_tv_distance_kernel` on the 5090D.

tvd (total-variation distance, reduction="none") is lowered from atomic IntentIR
through the C++ ``-convert-intent-elementwise-to-gpu`` pass (using the M3
``intent.select`` + a ``map_binary`` comparison fn) to a read-once block-
cooperative sm_120a cubin, then launched and compared head-to-head with the raw
liger Triton kernel (bypassing the autograd Function) for (1) correctness
(rel < 1e-3 on BOTH grad and loss, vs an fp32 torch reference AND vs liger's own
output) and (2) kernel-only median latency.

Lowered DAG (reduction="none", both outputs [M,N], pure elementwise):
  grad = where(p > q, +0.5*scale, -0.5*scale)   (intent.select over arith.cmpf)
  loss = 0.5 * |p - q|                           (|.| = max(x,-x), no libdevice)

PERF — honest PARITY floor. With reduction="none" there is NO host ``.sum()``
to fuse (that beat applies only to batchmean/mean/sum, where liger does a host
``output_tensor.sum()`` after the kernel). Here both sides read p,q once and
write grad,loss once — read-once / write-once, no HBM bytes left to remove. The
named-beat tvd story (fused loss row-sum) lives in the reduced modes; this clean
reduction="none" path is parity at peak HBM and is reported as such.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_tvd_pk.py -q -s
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

_PASS = "convert-intent-elementwise-to-gpu"


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _T(N):
    """Block size over the feature dim, mirroring liger's tvd thread geometry."""
    from liger_kernel.ops.tvd import get_num_warps
    bs = min(16384, triton.next_power_of_2(N))
    return get_num_warps(bs) * 32


def _tvd_mlir(M, N):
    R = f"tensor<{M}x{N}xf32>"
    S = "tensor<f32>"
    I = f"tensor<{M}x{N}xi1>"
    return f"""
func.func @tvd(%P: {R}, %Q: {R}, %half: {S}, %nhalf: {S}, %scale: {S}) -> ({R}, {R}) {{
  %pos = intent.map_binary %half, %scale {{fn = "mul"}} : ({S}, {S}) -> {S}
  %neg = intent.map_binary %nhalf, %scale {{fn = "mul"}} : ({S}, {S}) -> {S}
  %cond = intent.map_binary %P, %Q {{fn = "gt"}} : ({R}, {R}) -> {I}
  %grad = intent.select %cond, %pos, %neg : ({I}, {S}, {S}) -> {R}
  %diff = intent.map_binary %P, %Q {{fn = "sub"}} : ({R}, {R}) -> {R}
  %ad = intent.map_unary %diff {{fn = "abs"}} : ({R}) -> {R}
  %loss = intent.map_binary %half, %ad {{fn = "mul"}} : ({S}, {R}) -> {R}
  return %grad, %loss : {R}, {R}
}}
"""


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


def _liger_launch(P, Q):
    """Raw _tv_distance_kernel launch (reduction=none, scale=1.0, no label)."""
    from liger_kernel.ops.tvd import (_REDUCTION_MODE_NONE, _tv_distance_kernel,
                                       get_num_warps)
    BT, V = P.shape
    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)
    loss = torch.zeros((BT, V), device="cuda", dtype=torch.float32)
    grads = torch.empty_like(P)
    dummy = torch.empty(1, device="cuda")

    def go():
        _tv_distance_kernel[(BT,)](
            P, P.stride(0), Q, Q.stride(0), loss, loss.stride(0),
            grads, grads.stride(0), dummy, -100, V, 1.0,
            BLOCK_SIZE=BLOCK_SIZE, HAS_LABEL=False, num_warps=num_warps,
            reduction=_REDUCTION_MODE_NONE.value)
    return go, loss, grads


@pytest.mark.parametrize("M,N", [(8192, 1024), (4096, 4096)])
def test_tvd_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    torch.manual_seed(0)
    # Probability-like rows (softmax) so p>q is well populated; any finite f32
    # would do — the lowering is value-agnostic.
    P = torch.softmax(torch.randn(M, N, device="cuda"), dim=-1).contiguous()
    Q = torch.softmax(torch.randn(M, N, device="cuda"), dim=-1).contiguous()

    k = RowReductionKernel("tvd", {"M": M, "N": N}, _T(N),
                           pass_name=_PASS, mlir_text=_tvd_mlir(M, N))
    grad = torch.empty(M, N, device="cuda", dtype=torch.float32)
    loss = torch.empty(M, N, device="cuda", dtype=torch.float32)
    # sig order: P, Q, half, nhalf, scale, then outputs grad, loss
    go = k.thunk([P, Q, _scalar(0.5), _scalar(-0.5), _scalar(1.0), grad, loss])
    go()
    torch.cuda.synchronize()

    grad_ref = torch.where(P > Q, _scalar(0.5), _scalar(-0.5))
    loss_ref = 0.5 * torch.abs(P - Q)
    assert _rel(grad, grad_ref) < 1e-3
    assert _rel(loss, loss_ref) < 1e-3

    lgo, lloss, lgrad = _liger_launch(P, Q)
    lgo()
    torch.cuda.synchronize()
    assert _rel(grad, lgrad) < 1e-3
    assert _rel(loss, lloss) < 1e-3

    gb = 4 * M * N * 4 / 1e9  # read P,Q + write grad,loss
    tm, tl = _bench(go), _bench(lgo)
    print(f"\ntvd ({M},{N}) T={_T(N)}: mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/liger {tm/tl:.2f} "
          f"(parity floor: reduction=none read-once, no host .sum() to fuse)")
    assert tm <= tl * 1.5
