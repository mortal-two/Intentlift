"""PK the relaxed row-reduction pass vs flash `_bwd_preprocess_do_o_dot` on 5090D.

flash's backward preprocess computes ``delta = sum(o * do, axis=1)`` — a per-row
[M] output with TWO Row inputs (o, do) and NO [M,N] RowOut. This exercises the M3
relaxation of ``-convert-intent-row-reduction-to-gpu``: all-PerRowOut funcs are
accepted (the ">=1 RowOut" requirement is dropped) and the feature dim N is
derived from a Row INPUT instead of a RowOut result. The cooperative read-once
register tile + ``gpu.all_reduce add`` + thread-0 scalar write are the existing
machinery; only match() was relaxed.

We PK the kernel-only inner work: a standalone bare-ptr launch over contiguous
``o, do : [M, headdim]`` (a single attention head — flash's 4-D BHQD strided
indexing collapses to stride_ob=stride_oh=0, stride_om=headdim, off_b=off_h=0,
seqlen_q_rounded=M). delta is compared vs ``torch.sum(o*do, dim=1)`` and vs the
flash kernel's own delta.

PERF — these are TINY tensors (1-4 MB), so both kernels are launch/latency-bound,
NOT HBM-bound (measured 236-819 GB/s, far below the ~1500 GB/s HBM peak). Our
kernel launches M blocks (one per row) vs flash's cdiv(M, BLOCK_M=128) blocks
(e.g. 32 for M=4096), so we get markedly higher SM occupancy and run faster
(mine/flash ~0.62-0.69). That is an honest launch-geometry win on a latency-bound
regime — NOT an HBM-traffic beat (both already read-once). On a bandwidth-bound
sizing this would collapse to parity at peak HBM.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_bwd_preprocess_pk.py -q -s
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

_PASS = "convert-intent-row-reduction-to-gpu"


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _bwd_mlir(M, N):
    R = f"tensor<{M}x{N}xf32>"
    return f"""
func.func @bwd_preprocess(%O: {R}, %DO: {R}) -> tensor<{M}xf32> {{
  %odo = intent.map_binary %O, %DO {{fn = "mul"}} : ({R}, {R}) -> {R}
  %delta = intent.reduce %odo {{fn = "sum", axes = [1], assoc = "associative"}} : ({R}) -> tensor<{M}xf32>
  return %delta : tensor<{M}xf32>
}}
"""


_FLASH_TRITON = ("/home/kingdom/phdworks/Dataset/flash-attention/"
                 "flash_attn/flash_attn_triton.py")


def _load_flash_kernel():
    """Load _bwd_preprocess_do_o_dot directly from the source file, bypassing
    flash_attn/__init__.py (which imports an uncompiled CUDA extension). The
    triton module itself imports only math/torch/triton."""
    import importlib.util
    if not pathlib.Path(_FLASH_TRITON).exists():
        pytest.skip("flash-attention source not present")
    spec = importlib.util.spec_from_file_location("flash_attn_triton_src",
                                                  _FLASH_TRITON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._bwd_preprocess_do_o_dot


def _flash_launch(O, DO, d):
    """Raw _bwd_preprocess_do_o_dot launch over a single contiguous head."""
    _bwd_preprocess_do_o_dot = _load_flash_kernel()
    M = O.shape[0]
    delta = torch.empty(M, device="cuda", dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = (triton.cdiv(M, BLOCK_M), 1)

    def go():
        _bwd_preprocess_do_o_dot[grid](
            O, DO, delta,
            0, 0, O.stride(0),        # stride_ob, stride_oh, stride_om
            0, 0, DO.stride(0),       # stride_dob, stride_doh, stride_dom
            1,                        # nheads
            M,                        # seqlen_q
            M,                        # seqlen_q_rounded (M is a multiple of 128)
            d,                        # headdim
            BLOCK_M=BLOCK_M,
            BLOCK_HEADDIM=BLOCK_HEADDIM)
    return go, delta


@pytest.mark.parametrize("M,d", [(4096, 64), (8192, 128)])
def test_bwd_preprocess_pk_vs_flash(M, d):
    from intentir.benchmarks.row_reduction import RowReductionKernel

    assert M % 128 == 0
    torch.manual_seed(0)
    O = torch.randn(M, d, device="cuda", dtype=torch.float32).contiguous()
    DO = torch.randn(M, d, device="cuda", dtype=torch.float32).contiguous()

    T = d if d <= 128 else 128       # one block per row, T threads over headdim
    k = RowReductionKernel("bwd_preprocess", {"M": M, "N": d}, T,
                           pass_name=_PASS, mlir_text=_bwd_mlir(M, d))
    delta = torch.empty(M, device="cuda", dtype=torch.float32)
    go = k.thunk([O, DO, delta])     # sig order: O, DO, then delta
    go()
    torch.cuda.synchronize()

    ref = torch.sum(O * DO, dim=1)
    assert _rel(delta, ref) < 1e-3

    fgo, fdelta = _flash_launch(O, DO, d)
    fgo()
    torch.cuda.synchronize()
    assert _rel(delta, fdelta) < 1e-3

    gb = 2 * M * d * 4 / 1e9  # read O,DO (delta write negligible)
    tm, tl = _bench(go), _bench(fgo)
    print(f"\nbwd_preprocess ({M},{d}) T={T}: mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s) "
          f"flash {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} GB/s)  mine/flash {tm/tl:.2f} "
          f"(latency-bound tiny tensor, both << peak HBM; win = M blocks vs "
          f"flash's M/128 -> higher occupancy, NOT an HBM-traffic beat)")
    assert tm <= tl * 1.5
