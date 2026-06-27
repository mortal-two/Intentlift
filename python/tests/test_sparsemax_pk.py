"""PK the M4b sparsemax fwd + bwd lowering vs the REAL liger raw kernels.

Both halves lower through ``-convert-intent-row-reduction-to-gpu`` to a read-once
block-cooperative sm_120a cubin (one block per row, BLOCK_SIZE >= n_cols so the
whole row is in ONE block, exactly liger's launch precondition):

  fwd  the new block-cooperative inclusive cumsum (``intent.scan``) path: a
       CONTIGUOUS column-to-thread tiling, per-thread sequential prefix +
       thread-0 exclusive scan of the segment totals through
       ``gpu.dynamic_shared_memory`` (the G3 gate), then t=(cssv-1)/(iota+1),
       support=z>t, k=#support, s=sum_support(z), tau=(s-1)/max(k,1), y=relu(x-tau).
       The k-count and s-sum reduces reuse the SAME dynamic-shared scratch as a
       manual tree (no gpu.all_reduce, which would clobber it).
  bwd  NO scan — the M1 reduce-then-broadcast + select template: go_sum/cnt over
       the support, gi = select(o>0, go - go_sum/cnt, 0).

torch.sort runs on the HOST (kernel-only PK: the sort is excluded both sides;
sorted_x is a 2nd Row input). Correctness: rel < 1e-3 vs fp32 torch ref AND vs
the real liger raw kernel. Timing: kernel-only no-sync ``do_bench``.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_sparsemax_pk.py -q -s
"""

from __future__ import annotations

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


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _threads(N):
    """Power-of-two T <= 1024 dividing N, register tile K=N/T <= 32, K as close to
    4 as possible. K~4 balances the per-thread sequential prefix against the
    log2(T) barrier count of the cross-thread scan + the reduce trees (empirically
    the block-cumsum sweet spot: too-large T => K=1 and a barrier-bound launch)."""
    best, bestcost = 32, 1 << 30
    t = 1
    while t <= 1024:
        if N % t == 0 and N // t <= 32:
            cost = abs(N // t - 4)
            if cost < bestcost:
                best, bestcost = t, cost
        t *= 2
    return best


def _fwd_mlir(M, N):
    return f"""
func.func @sparsemax_fwd(%x: tensor<{M}x{N}xf32>, %sx: tensor<{M}x{N}xf32>,
                         %zero: tensor<f32>, %one: tensor<f32>) -> tensor<{M}x{N}xf32> {{
  %cssv = intent.scan %sx {{fn = "sum", axis = 1}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %iota = intent.iota : tensor<{N}xi32>
  %r = intent.map_binary %iota, %one {{fn = "add"}} : (tensor<{N}xi32>, tensor<f32>) -> tensor<{N}xf32>
  %cm1 = intent.map_binary %cssv, %one {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %t = intent.map_binary %cm1, %r {{fn = "div"}} : (tensor<{M}x{N}xf32>, tensor<{N}xf32>) -> tensor<{M}x{N}xf32>
  %support = intent.map_binary %sx, %t {{fn = "gt"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xi1>
  %selone = intent.select %support, %one, %zero : (tensor<{M}x{N}xi1>, tensor<f32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %k = intent.reduce %selone {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %kc = intent.map_binary %k, %one {{fn = "max"}} : (tensor<{M}xf32>, tensor<f32>) -> tensor<{M}xf32>
  %selz = intent.select %support, %sx, %zero : (tensor<{M}x{N}xi1>, tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %s = intent.reduce %selz {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %sm1 = intent.map_binary %s, %one {{fn = "sub"}} : (tensor<{M}xf32>, tensor<f32>) -> tensor<{M}xf32>
  %tau = intent.map_binary %sm1, %kc {{fn = "div"}} : (tensor<{M}xf32>, tensor<{M}xf32>) -> tensor<{M}xf32>
  %xmt = intent.map_binary %x, %tau {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<{M}xf32>) -> tensor<{M}x{N}xf32>
  %y = intent.map_binary %xmt, %zero {{fn = "max"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  return %y : tensor<{M}x{N}xf32>
}}
"""


def _bwd_mlir(M, N):
    return f"""
func.func @sparsemax_bwd(%o: tensor<{M}x{N}xf32>, %go: tensor<{M}x{N}xf32>,
                         %zero: tensor<f32>, %one: tensor<f32>, %eps: tensor<f32>)
    -> tensor<{M}x{N}xf32> {{
  %supp = intent.map_binary %o, %zero {{fn = "gt"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xi1>
  %selgo = intent.select %supp, %go, %zero : (tensor<{M}x{N}xi1>, tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %gosum = intent.reduce %selgo {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %selone = intent.select %supp, %one, %zero : (tensor<{M}x{N}xi1>, tensor<f32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %cnt = intent.reduce %selone {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %cntc = intent.map_binary %cnt, %eps {{fn = "max"}} : (tensor<{M}xf32>, tensor<f32>) -> tensor<{M}xf32>
  %ratio = intent.map_binary %gosum, %cntc {{fn = "div"}} : (tensor<{M}xf32>, tensor<{M}xf32>) -> tensor<{M}xf32>
  %inner = intent.map_binary %go, %ratio {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<{M}xf32>) -> tensor<{M}x{N}xf32>
  %gi = intent.select %supp, %inner, %zero : (tensor<{M}x{N}xi1>, tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  return %gi : tensor<{M}x{N}xf32>
}}
"""


def _sparsemax_ref(x):
    """fp32 reference sparsemax (matches liger's formula)."""
    M, N = x.shape
    z, _ = torch.sort(x, dim=1, descending=True)
    cssv = torch.cumsum(z, dim=1)
    r = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).view(1, N)
    t = (cssv - 1.0) / r
    support = z > t
    k = support.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    s = torch.where(support, z, torch.zeros_like(z)).sum(dim=1, keepdim=True)
    tau = (s - 1.0) / k
    return torch.clamp(x - tau, min=0.0), z


@pytest.mark.parametrize("M,N", [(4096, 1024), (2048, 4096)])
def test_sparsemax_fwd_pk_vs_liger(M, N):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    from liger_kernel.ops.sparsemax import _sparsemax_forward_kernel
    from liger_kernel.ops.utils import calculate_settings

    T = _threads(N)
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").contiguous()
    sorted_x = torch.sort(x, dim=1, descending=True).values.contiguous()

    y_ref, _ = _sparsemax_ref(x)

    k = RowReductionKernel("sparsemax_fwd", {"M": M, "N": N}, T,
                           mlir_text=_fwd_mlir(M, N))
    zero = torch.zeros((), device="cuda")
    one = torch.ones((), device="cuda")
    out = torch.empty(M, N, device="cuda")
    go = k.thunk([x, sorted_x, zero, one, out])  # sig: x, sx, zero, one, y
    go(); torch.cuda.synchronize()
    rel_ref = _rel(out, y_ref)

    # real liger raw forward kernel (autograd bypassed), kernel-only.
    BLOCK, num_warps = calculate_settings(N)
    lig_out = torch.empty_like(x)
    grid = (M,)

    def liger():
        _sparsemax_forward_kernel[grid](
            x, x.stride(0), sorted_x, sorted_x.stride(0),
            lig_out, lig_out.stride(0), N,
            BLOCK_SIZE=BLOCK, num_warps=num_warps)

    liger(); torch.cuda.synchronize()
    rel_lig = _rel(out, lig_out)

    tm, tl = _bench(go), _bench(liger)
    print(f"\nsparsemax_fwd ({M},{N}) T={T}: mine {tm*1e3:.1f}us liger {tl*1e3:.1f}us "
          f"mine/liger {tm/tl:.2f}  rel_ref {rel_ref:.2e} rel_liger {rel_lig:.2e}")
    assert rel_ref < 1e-3 and rel_lig < 1e-3


@pytest.mark.parametrize("M,N", [(4096, 1024), (2048, 4096)])
def test_sparsemax_bwd_pk_vs_liger(M, N):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    from liger_kernel.ops.sparsemax import _sparsemax_backward_kernel
    from liger_kernel.ops.utils import calculate_settings

    T = _threads(N)
    torch.manual_seed(1)
    x = torch.randn(M, N, device="cuda").contiguous()
    o, _ = _sparsemax_ref(x)           # a valid sparsemax output (support meaningful)
    o = o.contiguous()
    go_grad = torch.randn(M, N, device="cuda").contiguous()

    # fp32 ref backward
    supp = o > 0.0
    cnt = supp.sum(dim=1, keepdim=True).to(x.dtype)
    go_sum = torch.where(supp, go_grad, torch.zeros_like(go_grad)).sum(dim=1, keepdim=True)
    gi_ref = torch.where(supp, go_grad - go_sum / cnt.clamp(min=1e-6),
                         torch.zeros_like(go_grad))

    k = RowReductionKernel("sparsemax_bwd", {"M": M, "N": N}, T,
                           mlir_text=_bwd_mlir(M, N))
    zero = torch.zeros((), device="cuda")
    one = torch.ones((), device="cuda")
    eps = torch.full((), 1e-6, device="cuda")
    gi = torch.empty(M, N, device="cuda")
    go = k.thunk([o, go_grad, zero, one, eps, gi])  # sig: o, go, zero, one, eps, gi
    go(); torch.cuda.synchronize()
    rel_ref = _rel(gi, gi_ref)

    # real liger raw backward kernel, kernel-only.
    BLOCK, num_warps = calculate_settings(N)
    lig_gi = torch.empty_like(o)
    grid = (M,)

    def liger():
        _sparsemax_backward_kernel[grid](
            o, go_grad, lig_gi, o.stride(0), N,
            BLOCK_SIZE=BLOCK, num_warps=num_warps)

    liger(); torch.cuda.synchronize()
    rel_lig = _rel(gi, lig_gi)

    tm, tl = _bench(go), _bench(liger)
    print(f"\nsparsemax_bwd ({M},{N}) T={T}: mine {tm*1e3:.1f}us liger {tl*1e3:.1f}us "
          f"mine/liger {tm/tl:.2f}  rel_ref {rel_ref:.2e} rel_liger {rel_lig:.2e}")
    assert rel_ref < 1e-3 and rel_lig < 1e-3
