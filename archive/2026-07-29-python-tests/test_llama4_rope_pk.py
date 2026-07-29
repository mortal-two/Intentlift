"""PK llama4_rope vs the REAL liger ``_llama4_rope_kernel`` on the 5090D.

Llama4 RoPE is the same complex-rotation as rope, but the partner pairing is the
even/odd interleaved complex pair (2j, 2j+1) instead of the half-split. It lowers
through the SAME ``-convert-intent-elementwise-to-gpu`` pass with NO per-kernel
branch — the only change vs rope is a different ``intent.strides`` value
(col-stride 2 even/odd views instead of half-block). See
test/lower-llama4_rope-to-gpu.mlir. imag_sign is a scalar (1.0 fwd / -1.0 bwd).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_llama4_rope_pk.py -q -s
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
    t = 1
    while t * 2 <= N and t * 2 <= 256:
        t *= 2
    return t


def _llama4_mlir(M, hd, n_row):
    N = hd // 2
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{n_row}x{N}xf32>"
    s = f"array<i64: {hd}, 2>"
    return f"""
func.func @llama4_rope(%qe: {R} {{intent.strides = {s}}},
                       %qo: {R} {{intent.strides = {s}}},
                       %fr: {C} {{intent.strides = {s}}},
                       %fi: {C} {{intent.strides = {s}}},
                       %isign: tensor<f32>)
    -> ({R}, {R}) attributes {{intent.inplace = array<i64: 0, 1>}} {{
  %fis = intent.map_binary %fi, %isign {{fn = "mul"}} : ({C}, tensor<f32>) -> {C}
  %a = intent.map_binary %qe, %fr {{fn = "mul"}} : ({R}, {C}) -> {R}
  %b = intent.map_binary %qo, %fis {{fn = "mul"}} : ({R}, {C}) -> {R}
  %yr = intent.map_binary %a, %b {{fn = "sub"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %qe, %fis {{fn = "mul"}} : ({R}, {C}) -> {R}
  %d = intent.map_binary %qo, %fr {{fn = "mul"}} : ({R}, {C}) -> {R}
  %yi = intent.map_binary %c, %d {{fn = "add"}} : ({R}, {R}) -> {R}
  return %yr, %yi : {R}, {R}
}}
"""


@pytest.mark.parametrize("bsz,seq,nh,hd", [(8, 1024, 8, 128), (4, 2048, 16, 64)])
def test_llama4_rope_pk_vs_liger(bsz, seq, nh, hd):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.llama4_rope import _llama4_rope_kernel

    n_row = bsz * seq
    M = n_row * nh
    hd2 = hd // 2
    torch.manual_seed(0)
    q = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    k = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    freqs = torch.randn(seq, hd, device="cuda", dtype=torch.float32)
    freqs_exp = freqs.repeat(bsz, 1).contiguous()  # [n_row, hd]
    q0, k0 = q.clone(), k.clone()  # originals (mine rotates IN PLACE)
    qm, km = q.view(M, hd), k.view(M, hd)
    isign = torch.tensor(1.0, device="cuda")

    kern = RowReductionKernel("llama4_rope", {"M": M, "N": hd2}, _T(hd2),
                              pass_name=_PASS, mlir_text=_llama4_mlir(M, hd, n_row))
    fr, fi = freqs_exp[:, 0::2], freqs_exp[:, 1::2]
    # in-place: even/odd results alias qe/qo.
    go_q = kern.thunk([qm[:, 0::2], qm[:, 1::2], fr, fi, isign])
    go_k = kern.thunk([km[:, 0::2], km[:, 1::2], fr, fi, isign])

    def go():
        go_q()
        go_k()

    go()
    torch.cuda.synchronize()

    q0m = q0.view(M, hd)
    tok = torch.arange(M, device="cuda") // nh
    fre, fie = fr[tok], fi[tok]
    qe, qo = q0m[:, 0::2], q0m[:, 1::2]
    ref = torch.empty(M, hd, device="cuda")
    ref[:, 0::2] = qe * fre - qo * fie
    ref[:, 1::2] = qe * fie + qo * fre
    assert _rel(q.view(M, hd), ref) < 1e-3

    # vs raw liger kernel (in-place on fresh clones of the originals).
    BLOCK = 16
    q4 = q0.view(bsz, seq, nh, hd).clone()
    k4 = k0.view(bsz, seq, nh, hd).clone()
    _llama4_rope_kernel[(n_row, nh)](
        q4, k4, freqs, q4.stride(1), k4.stride(1), q4.stride(2), k4.stride(2),
        freqs.stride(0), seq, bsz, 1.0, hd2, nh, nh, BLOCK, num_warps=2,
        num_stages=2)
    torch.cuda.synchronize()
    assert _rel(q.view(M, hd), q4.reshape(M, hd)) < 1e-3
    assert _rel(k.view(M, hd), k4.reshape(M, hd)) < 1e-3

    q4 = q.view(bsz, seq, nh, hd)
    k4 = k.view(bsz, seq, nh, hd)

    def liger():
        _llama4_rope_kernel[(n_row, nh)](
            q4, k4, freqs, q4.stride(1), k4.stride(1), q4.stride(2),
            k4.stride(2), freqs.stride(0), seq, bsz, 1.0, hd2, nh, nh, BLOCK,
            num_warps=2, num_stages=2)

    gb = 4 * M * hd * 4 / 1e9
    tm, tl = _bench(go), _bench(liger)
    print(f"\nllama4_rope ({bsz},{seq},{nh},{hd}): mine {tm*1e3:.1f}us "
          f"({gb/(tm/1e3):.0f} GB/s) liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} "
          f"GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * 1.5
