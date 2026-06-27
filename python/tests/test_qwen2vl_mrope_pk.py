"""PK qwen2vl_mrope vs the REAL liger ``_triton_qwen2vl_mrope`` on the 5090D.

Qwen2-VL M-RoPE is the same rope rotation, with cos/sin each assembled from three
position planes (temporal/height/width) selected by per-column section masks:
cos_row = tc*tm + hc*hm + wc*wm. It lowers through the SAME
``-convert-intent-elementwise-to-gpu`` pass with NO per-kernel branch — the
3-plane combination is just extra map ops over broadcast-row plane inputs + Col
masks, then the identical half-view rotation as rope. See
test/lower-qwen2vl_mrope-to-gpu.mlir.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_qwen2vl_mrope_pk.py -q -s
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


def _qwen_mlir(M, hd, n_row):
    N = hd // 2
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{n_row}x{N}xf32>"
    V = f"tensor<{N}xf32>"
    s = f"array<i64: {hd}, 1>"
    return f"""
func.func @qwen2vl_mrope(
    %q_lo: {R} {{intent.strides={s}}}, %q_hi: {R} {{intent.strides={s}}},
    %tc: {C} {{intent.strides={s}}}, %hc: {C} {{intent.strides={s}}}, %wc: {C} {{intent.strides={s}}},
    %ts: {C} {{intent.strides={s}}}, %hs: {C} {{intent.strides={s}}}, %ws: {C} {{intent.strides={s}}},
    %tm: {V}, %hm: {V}, %wm: {V})
    -> ({R}, {R}) attributes {{intent.inplace = array<i64: 0, 1>}} {{
  %tcm = intent.map_binary %tc, %tm {{fn="mul"}} : ({C},{V}) -> {C}
  %hcm = intent.map_binary %hc, %hm {{fn="mul"}} : ({C},{V}) -> {C}
  %wcm = intent.map_binary %wc, %wm {{fn="mul"}} : ({C},{V}) -> {C}
  %co1 = intent.map_binary %tcm, %hcm {{fn="add"}} : ({C},{C}) -> {C}
  %cos = intent.map_binary %co1, %wcm {{fn="add"}} : ({C},{C}) -> {C}
  %tsm = intent.map_binary %ts, %tm {{fn="mul"}} : ({C},{V}) -> {C}
  %hsm = intent.map_binary %hs, %hm {{fn="mul"}} : ({C},{V}) -> {C}
  %wsm = intent.map_binary %ws, %wm {{fn="mul"}} : ({C},{V}) -> {C}
  %si1 = intent.map_binary %tsm, %hsm {{fn="add"}} : ({C},{C}) -> {C}
  %sin = intent.map_binary %si1, %wsm {{fn="add"}} : ({C},{C}) -> {C}
  %a = intent.map_binary %q_lo, %cos {{fn="mul"}} : ({R},{C}) -> {R}
  %b = intent.map_binary %q_hi, %sin {{fn="mul"}} : ({R},{C}) -> {R}
  %ylo = intent.map_binary %a, %b {{fn="sub"}} : ({R},{R}) -> {R}
  %c = intent.map_binary %q_hi, %cos {{fn="mul"}} : ({R},{C}) -> {R}
  %d = intent.map_binary %q_lo, %sin {{fn="mul"}} : ({R},{C}) -> {R}
  %yhi = intent.map_binary %c, %d {{fn="add"}} : ({R},{R}) -> {R}
  return %ylo, %yhi : {R}, {R}
}}
"""


@pytest.mark.parametrize("bsz,seq,nh,hd", [(8, 1024, 8, 128), (4, 2048, 16, 64)])
def test_qwen2vl_mrope_pk_vs_liger(bsz, seq, nh, hd):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    from liger_kernel.ops.qwen2vl_mrope import _triton_qwen2vl_mrope

    n_row = bsz * seq
    M = n_row * nh
    hd2 = hd // 2
    t_sec, h_sec = hd2 // 4, hd2 // 4  # w_sec = hd2 - t - h
    torch.manual_seed(0)
    q = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    k = torch.randn(n_row, nh * hd, device="cuda", dtype=torch.float32)
    cos = torch.randn(3, n_row, hd, device="cuda", dtype=torch.float32)
    sin = torch.randn(3, n_row, hd, device="cuda", dtype=torch.float32)
    q0, k0 = q.clone(), k.clone()  # originals (mine rotates IN PLACE)
    qm, km = q.view(M, hd), k.view(M, hd)

    col = torch.arange(hd2, device="cuda")
    tm = (col < t_sec).float()
    hm = ((col >= t_sec) & (col < t_sec + h_sec)).float()
    wm = (col >= t_sec + h_sec).float()

    kern = RowReductionKernel("qwen2vl_mrope", {"M": M, "N": hd2}, _T(hd2),
                              pass_name=_PASS, mlir_text=_qwen_mlir(M, hd, n_row))

    def views(buf):
        return buf[:, :hd2], buf[:, hd2:]

    planes_c = [cos[0, :, :hd2], cos[1, :, :hd2], cos[2, :, :hd2]]
    planes_s = [sin[0, :, :hd2], sin[1, :, :hd2], sin[2, :, :hd2]]
    qlo, qhi = views(qm)
    klo, khi = views(km)
    # in-place: y_lo/y_hi alias q_lo/q_hi.
    go_q = kern.thunk([qlo, qhi, *planes_c, *planes_s, tm, hm, wm])
    go_k = kern.thunk([klo, khi, *planes_c, *planes_s, tm, hm, wm])

    def go():
        go_q()
        go_k()

    go()
    torch.cuda.synchronize()

    q0m = q0.view(M, hd)
    tok = torch.arange(M, device="cuda") // nh
    cr = planes_c[0] * tm + planes_c[1] * hm + planes_c[2] * wm
    sr = planes_s[0] * tm + planes_s[1] * hm + planes_s[2] * wm
    cre, sre = cr[tok], sr[tok]
    ref = torch.cat([q0m[:, :hd2] * cre - q0m[:, hd2:] * sre,
                     q0m[:, hd2:] * cre + q0m[:, :hd2] * sre], dim=1)
    assert _rel(q.view(M, hd), ref) < 1e-3

    pad_hd = triton.next_power_of_2(hd)
    pad_nh = triton.next_power_of_2(nh)
    cc, sc = cos.contiguous(), sin.contiguous()
    ql, kl = q0.clone(), k0.clone()
    _triton_qwen2vl_mrope[(n_row,)](
        ql, kl, cc, sc, seq, bsz, nh, nh, hd, pad_nh, pad_nh, pad_hd, t_sec,
        h_sec, BLOCK_SIZE=max(pad_nh, pad_nh), BACKWARD_PASS=False)
    torch.cuda.synchronize()
    assert _rel(q.view(M, hd), ql.view(M, hd)) < 1e-3
    assert _rel(k.view(M, hd), kl.view(M, hd)) < 1e-3

    def liger():
        _triton_qwen2vl_mrope[(n_row,)](
            q, k, cc, sc, seq, bsz, nh, nh, hd, pad_nh, pad_nh, pad_hd, t_sec,
            h_sec, BLOCK_SIZE=max(pad_nh, pad_nh), BACKWARD_PASS=False)

    gb = 4 * M * hd * 4 / 1e9
    tm_, tl = _bench(go), _bench(liger)
    print(f"\nqwen2vl_mrope ({bsz},{seq},{nh},{hd}): mine {tm_*1e3:.1f}us "
          f"({gb/(tm_/1e3):.0f} GB/s) liger {tl*1e3:.1f}us ({gb/(tl/1e3):.0f} "
          f"GB/s)  mine/liger {tm_/tl:.2f}")
    assert tm_ <= tl * 1.5
