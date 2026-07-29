"""PK the C6 order-statistics lowerings vs torch AND the real flag_gems kernels.

Two new mechanisms, both a single read-once block-cooperative sm_120a cubin (one
block per row, the feature axis over the block's threads):

  argmax/argmin  the C6 arg-reduce pass (`-convert-intent-arg-reduce-to-gpu`):
                 the block reduction carries a (best_val, best_idx) PAIR — a
                 per-thread local best over K contiguous columns (ascending +
                 STRICT compare => ties keep the lowest column, torch's rule),
                 then a power-of-two shared-memory tree over two dynamic-shared
                 buffers (val f32, idx i32). Integer index output. Verify: EXACT
                 index vs torch.argmax/argmin AND flag_gems.

  cummax/cummin  `intent.scan` fn=max/min through the SAME Hillis-Steele block
                 scan as cumsum (C5/M4), only the combine + identity change (max/
                 min, -inf/+inf) and the exclusive prefix is read from thread
                 (tx-1) instead of the sum-only incl-segTotal subtraction. Verify:
                 the VALUE (cummax also returns argmax indices; the value is the
                 tested half) rel < 1e-3 vs torch.cummax/cummin AND flag_gems.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_order_stats_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
flag_gems = pytest.importorskip("flag_gems")

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
    return float((a - b).float().norm() / b.float().norm())


def _threads(N):
    """Largest power-of-two T <= 1024 dividing N with K=N/T <= 32, K close to 4."""
    best, bestcost = 32, 1 << 30
    t = 1
    while t <= 1024:
        if N % t == 0 and N // t <= 32:
            cost = abs(N // t - 4)
            if cost < bestcost:
                best, bestcost = t, cost
        t *= 2
    return best


def _arg_mlir(M, N, fn):
    return f"""
func.func @{fn}_row(%x: tensor<{M}x{N}xf32>) -> tensor<{M}xi32> {{
  %i = intent.arg_reduce %x {{fn = "{fn}", axis = 1 : i64}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xi32>
  return %i : tensor<{M}xi32>
}}
"""


def _scan_mlir(M, N, fn):
    return f"""
func.func @cum{fn}_row(%x: tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %y = intent.scan %x {{fn = "{fn}", axis = 1}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  return %y : tensor<{M}x{N}xf32>
}}
"""


@pytest.mark.parametrize("fn", ["argmax", "argmin"])
@pytest.mark.parametrize("M,N", [(4096, 1024), (2048, 4096)])
def test_arg_reduce_pk(fn, M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    T = _threads(N)
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").contiguous()

    ref = (torch.argmax if fn == "argmax" else torch.argmin)(x, dim=1)
    gems = getattr(flag_gems, fn)(x, dim=1)

    k = RowReductionKernel(f"{fn}", {"M": M, "N": N}, T,
                           pass_name="convert-intent-arg-reduce-to-gpu",
                           mlir_text=_arg_mlir(M, N, fn))
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    go = k.thunk([x, out])
    go(); torch.cuda.synchronize()

    mine = out.to(torch.int64)
    match_ref = int((mine == ref).sum())
    match_gems = int((mine == gems.to(torch.int64)).sum())
    print(f"\n{fn} ({M},{N}) T={T}: match_torch {match_ref}/{M} "
          f"match_gems {match_gems}/{M}")
    assert match_ref == M          # EXACT index vs torch
    assert match_gems == M         # EXACT index vs the real flag_gems kernel


@pytest.mark.parametrize("fn", ["max", "min"])
@pytest.mark.parametrize("M,N", [(4096, 1024), (2048, 4096)])
def test_cumscan_pk(fn, M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    T = _threads(N)
    torch.manual_seed(1)
    x = torch.randn(M, N, device="cuda").contiguous()

    cum = torch.cummax if fn == "max" else torch.cummin
    ref = cum(x, dim=1).values
    gems = getattr(flag_gems, f"cum{fn}")(x, dim=1)[0]   # (values, indices)

    k = RowReductionKernel(f"cum{fn}", {"M": M, "N": N}, T,
                           mlir_text=_scan_mlir(M, N, fn))
    out = torch.empty(M, N, device="cuda")
    go = k.thunk([x, out])
    go(); torch.cuda.synchronize()

    rel_ref = _rel(out, ref)
    rel_gems = _rel(out, gems)
    print(f"\ncum{fn} ({M},{N}) T={T}: rel_torch {rel_ref:.2e} rel_gems {rel_gems:.2e}")
    assert rel_ref < 1e-3 and rel_gems < 1e-3
