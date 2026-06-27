"""PK the rope family (rope / qwen2vl_mrope / llama4_rope) vs the REAL liger
Triton kernels on the 5090D.

These are the M3c "index-remap" kernels: the rotation couples head element j with
its partner (j +/- hd/2 for rope/qwen2vl, the even/odd sibling for llama4). All
three lower through the SAME pure-elementwise pass
(``-convert-intent-elementwise-to-gpu``) with NO per-kernel branch — the partner
coupling is expressed as STRIDED HALF/EVEN-ODD VIEWS of one [M,hd] buffer (an
``intent.strides`` arg-attr), so at matching logical column the kernel is pure
elementwise. cos/sin are broadcast rows (one per token, shared by the token's
n_head blocks). See test/lower-{rope,qwen2vl_mrope,llama4_rope}-to-gpu.mlir.

Each is compared head-to-head against the RAW liger kernel (``_triton_rope`` /
``_triton_qwen2vl_mrope`` / ``_llama4_rope_kernel``, bypassing the autograd
Function), both sides on pre-contiguous [n_row, n_head*hd] fp32 inputs (same
input contract — no transpose in either timed region). rope is in-place, so q/k
are cloned before each correctness launch. These are read-once / write-once
memory-bound kernels: the honest bar is PARITY at peak HBM.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_rope_pk.py -q -s
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


def _kernel(name, M, N, mlir_text):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(name, {"M": M, "N": N}, _T(N),
                              pass_name=_PASS, mlir_text=mlir_text)


# ---------------------------------------------------------------------------- #
# rope: y_lo = q_lo*cos - q_hi*sin ; y_hi = q_hi*cos + q_lo*sin.
# q_lo/q_hi are half-views (stride [hd,1]); cos/sin broadcast rows [n_row, hd/2].
# ---------------------------------------------------------------------------- #

def _rope_mlir(M, hd, n_row):
    N = hd // 2
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{n_row}x{N}xf32>"
    s = f"array<i64: {hd}, 1>"
    return f"""
func.func @rope(%q_lo: {R} {{intent.strides = {s}}},
                %q_hi: {R} {{intent.strides = {s}}},
                %cos: {C} {{intent.strides = {s}}},
                %sin: {C} {{intent.strides = {s}}})
    -> ({R}, {R}) attributes {{intent.inplace = array<i64: 0, 1>}} {{
  %a = intent.map_binary %q_lo, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %b = intent.map_binary %q_hi, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %ylo = intent.map_binary %a, %b {{fn = "sub"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %q_hi, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %d = intent.map_binary %q_lo, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %yhi = intent.map_binary %c, %d {{fn = "add"}} : ({R}, {R}) -> {R}
  return %ylo, %yhi : {R}, {R}
}}
"""


def _rope_views(buf, hd):
    return buf[:, :hd // 2], buf[:, hd // 2:]


@pytest.mark.parametrize("bsz,seq,nh,hd", [(8, 1024, 8, 128), (4, 2048, 16, 64)])
def test_rope_pk_vs_liger(bsz, seq, nh, hd):
    from liger_kernel.ops.rope import _triton_rope

    n_row = bsz * seq
    M = n_row * nh
    hd2 = hd // 2
    torch.manual_seed(0)
    # one contiguous [n_row, nh*hd] buffer == [M, hd] reshape (same storage).
    q = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    k = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    cos = torch.randn(n_row, hd, device="cuda", dtype=torch.float32)
    sin = torch.randn(n_row, hd, device="cuda", dtype=torch.float32)
    q0, k0 = q.clone(), k.clone()  # originals (mine rotates q/k IN PLACE)
    qm, km = q.view(M, hd), k.view(M, hd)

    kern = _kernel("rope", M, hd2, _rope_mlir(M, hd, n_row))
    qlo, qhi = _rope_views(qm, hd)
    klo, khi = _rope_views(km, hd)
    clo, slo = cos[:, :hd2], sin[:, :hd2]
    go_q = kern.thunk([qlo, qhi, clo, slo])  # in-place: y_lo/y_hi alias q_lo/q_hi
    go_k = kern.thunk([klo, khi, clo, slo])

    def go():
        go_q()
        go_k()

    go()
    torch.cuda.synchronize()

    # fp32 torch reference (liger fwd math) from the pre-rotation originals.
    q0m = q0.view(M, hd)
    tok = torch.arange(M, device="cuda") // nh
    ce, se = clo[tok], slo[tok]
    ref_q = torch.cat([q0m[:, :hd2] * ce - q0m[:, hd2:] * se,
                       q0m[:, hd2:] * ce + q0m[:, :hd2] * se], dim=1)
    assert _rel(q.view(M, hd), ref_q) < 1e-3

    # vs raw liger kernel, in-place on fresh clones of the originals.
    ql, kl = q0.clone(), k0.clone()
    pad_hd = triton.next_power_of_2(hd)
    pad_nh = triton.next_power_of_2(nh)
    _triton_rope[(n_row,)](
        ql, ql.stride(0), kl, kl.stride(0), cos, cos.stride(0), sin,
        sin.stride(0), seq, bsz, bsz, nh, nh, hd, pad_nh, pad_nh, pad_hd,
        BLOCK_SIZE=max(pad_nh, pad_nh), BACKWARD_PASS=False)
    torch.cuda.synchronize()
    assert _rel(q.view(M, hd), ql.view(M, hd)) < 1e-3
    assert _rel(k.view(M, hd), kl.view(M, hd)) < 1e-3

    def liger():
        _triton_rope[(n_row,)](
            q, q.stride(0), k, k.stride(0), cos, cos.stride(0), sin,
            sin.stride(0), seq, bsz, bsz, nh, nh, hd, pad_nh, pad_nh, pad_hd,
            BLOCK_SIZE=max(pad_nh, pad_nh), BACKWARD_PASS=False)

    gb = 4 * M * hd * 4 / 1e9  # read+write q AND k
    tm, tl = _bench(go), _bench(liger)
    print(f"\nrope ({bsz},{seq},{nh},{hd}): mine {tm*1e3:.1f}us "
          f"({gb/(tm/1e3):.0f} GB/s) liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} "
          f"GB/s)  mine/liger {tm/tl:.2f}")
    assert tm <= tl * 1.5
