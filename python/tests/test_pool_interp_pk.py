"""PK the C7 pooling / interpolation / polar lowerings vs torch + real flag_gems.

avg_pool2d / max_pool2d_with_indices / upsample_nearest2d / polar are lowered from
atomic IntentIR (`intent.pool` / `intent.upsample` / `intent.polar`) through the
C++ ``-convert-intent-{pool,upsample,polar}-to-gpu`` passes to a block-cooperative
sm_120a cubin, then launched and compared head-to-head against (1) a torch
reference and (2) the real ``flag_gems.<op>`` triton kernel.

Three distinct memory-bound mechanisms:
  * pool     — a BOUNDED per-output-cell reduce over the kHxkW window (avg = sum/
               (kH*kW) count_include_pad; max also emits the argmax FLAT index
               ih*W+iw as i64). Values rel<1e-3, indices EXACT.
  * upsample — a pure affine floor index-remap gather (out=in[floor(oh*IH/OH),..]),
               bit-EXACT.
  * polar    — elementwise abs*(cos,sin) reusing the C0 cos/sin approx ops; the
               complex output is the trailing real/imag pair, compared vs
               torch.view_as_real(torch.polar(...)) (rel<1e-3).

The intent MLIR is hand-written here (same shape as test/lower-pool-interp-to-gpu.
mlir). Destination-passing: the appended output memref(s) are the result.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_pool_interp_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")

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
    a, b = a.float(), b.float()
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _kernel(name, T, mlir_text, pass_name):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(name, {"M": 1, "N": 1}, T,
                              pass_name=pass_name, mlir_text=mlir_text)


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# avg_pool2d: kernel 2, stride 2, pad 0. out = window mean.
# ---------------------------------------------------------------------------- #

def test_avg_pool2d():
    N, C, H, W, T = 2, 3, 16, 16, 8
    OH, OW = 8, 8
    I = f"tensor<{N}x{C}x{H}x{W}xf32>"
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @avgp(%a: {I}) -> {O} {{
  %o = intent.pool %a {{fn = "avg", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}} : ({I}) -> {O}
  return %o : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(N, C, H, W, device="cuda").contiguous()
    out = torch.empty(N, C, OH, OW, device="cuda")

    k = _kernel("avgp", T, mlir, "convert-intent-pool-to-gpu")
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.nn.functional.avg_pool2d(a, 2, stride=2, padding=0)
    assert _rel(out, ref) < 1e-3

    fg = _flag_gems()
    if fg is not None:
        yg = fg.avg_pool2d(a, 2, stride=2, padding=0)
        torch.cuda.synchronize()
        assert _rel(out, yg) < 1e-3
    print(f"\navg_pool2d ({N},{C},{H},{W})->({N},{C},{OH},{OW}): rel<1e-3 vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# avg_pool2d with padding: kernel 3, stride 2, pad 1 (count_include_pad=True).
# ---------------------------------------------------------------------------- #

def test_avg_pool2d_pad():
    N, C, H, W, T = 2, 4, 16, 16, 8
    OH, OW = 8, 8  # (16 + 2 - 3)//2 + 1 = 8
    I = f"tensor<{N}x{C}x{H}x{W}xf32>"
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @avgpp(%a: {I}) -> {O} {{
  %o = intent.pool %a {{fn = "avg", kH = 3 : i64, kW = 3 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 1 : i64, padW = 1 : i64}} : ({I}) -> {O}
  return %o : {O}
}}
"""
    torch.manual_seed(1)
    a = torch.randn(N, C, H, W, device="cuda").contiguous()
    out = torch.empty(N, C, OH, OW, device="cuda")

    k = _kernel("avgpp", T, mlir, "convert-intent-pool-to-gpu")
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.nn.functional.avg_pool2d(a, 3, stride=2, padding=1,
                                         count_include_pad=True)
    assert _rel(out, ref) < 1e-3

    fg = _flag_gems()
    if fg is not None:
        yg = fg.avg_pool2d(a, 3, stride=2, padding=1, count_include_pad=True)
        torch.cuda.synchronize()
        assert _rel(out, yg) < 1e-3
    print(f"\navg_pool2d pad=1 k=3 s=2: rel<1e-3 vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# max_pool2d_with_indices: kernel 2, stride 2 — value + argmax FLAT index (i64).
# ---------------------------------------------------------------------------- #

def test_max_pool2d_with_indices():
    N, C, H, W, T = 2, 3, 16, 16, 8
    OH, OW = 8, 8
    I = f"tensor<{N}x{C}x{H}x{W}xf32>"
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    X = f"tensor<{N}x{C}x{OH}x{OW}xi64>"
    mlir = f"""
func.func @maxp(%a: {I}) -> ({O}, {X}) {{
  %o, %i = intent.pool %a {{fn = "max", kH = 2 : i64, kW = 2 : i64, sH = 2 : i64,
      sW = 2 : i64, padH = 0 : i64, padW = 0 : i64}} : ({I}) -> ({O}, {X})
  return %o, %i : {O}, {X}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(N, C, H, W, device="cuda").contiguous()
    out = torch.empty(N, C, OH, OW, device="cuda")
    idx = torch.empty(N, C, OH, OW, device="cuda", dtype=torch.int64)

    k = _kernel("maxp", T, mlir, "convert-intent-pool-to-gpu")
    k.thunk([a, out, idx])()
    torch.cuda.synchronize()

    rv, ri = torch.nn.functional.max_pool2d_with_indices(
        a, 2, stride=2, padding=0)
    assert _rel(out, rv) < 1e-3
    assert torch.equal(idx, ri)  # exact argmax flat index (ih*W+iw)

    fg = _flag_gems()
    if fg is not None:
        gv, gi = fg.max_pool2d_with_indices(a, 2, stride=2, padding=0)
        torch.cuda.synchronize()
        assert _rel(out, gv) < 1e-3
        assert torch.equal(idx, gi)
    print(f"\nmax_pool2d_with_indices ({N},{C},{H},{W}): value rel<1e-3, index "
          "EXACT vs torch" + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# upsample_nearest2d: scale 2 — nearest-neighbor floor index-remap gather.
# ---------------------------------------------------------------------------- #

def test_upsample_nearest2d():
    N, C, H, W, T = 2, 3, 8, 8, 16
    OH, OW = 16, 16
    I = f"tensor<{N}x{C}x{H}x{W}xf32>"
    O = f"tensor<{N}x{C}x{OH}x{OW}xf32>"
    mlir = f"""
func.func @up(%a: {I}) -> {O} {{
  %o = intent.upsample %a {{mode = "nearest"}} : ({I}) -> {O}
  return %o : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(N, C, H, W, device="cuda").contiguous()
    out = torch.empty(N, C, OH, OW, device="cuda")

    k = _kernel("up", T, mlir, "convert-intent-upsample-to-gpu")
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.nn.functional.interpolate(a, scale_factor=2, mode="nearest")
    assert torch.equal(out, ref)  # exact gather

    fg = _flag_gems()
    if fg is not None:
        yg = fg.upsample_nearest2d(a, (OH, OW), None, None)
        torch.cuda.synchronize()
        assert _rel(out, yg) < 1e-3
    print(f"\nupsample_nearest2d ({N},{C},{H},{W})->x2: exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# polar: abs*(cos angle, sin angle) — complex as a trailing real/imag pair.
# ---------------------------------------------------------------------------- #

def test_polar():
    M, Ncol, T = 128, 256, 256
    A = f"tensor<{M}x{Ncol}xf32>"
    mlir = f"""
func.func @pol(%a: {A}, %b: {A}) -> ({A}, {A}) {{
  %r, %i = intent.polar %a, %b : ({A}, {A}) -> ({A}, {A})
  return %r, %i : {A}, {A}
}}
"""
    torch.manual_seed(0)
    absv = torch.rand(M, Ncol, device="cuda").contiguous() + 0.1
    ang = (torch.rand(M, Ncol, device="cuda") * 6.0 - 3.0).contiguous()
    re = torch.empty(M, Ncol, device="cuda")
    im = torch.empty(M, Ncol, device="cuda")

    k = _kernel("pol", T, mlir, "convert-intent-polar-to-gpu")
    k.thunk([absv, ang, re, im])()
    torch.cuda.synchronize()

    ref = torch.view_as_real(torch.polar(absv, ang))  # [...,2]
    assert _rel(re, ref[..., 0]) < 1e-3
    assert _rel(im, ref[..., 1]) < 1e-3

    fg = _flag_gems()
    if fg is not None:
        gc = torch.view_as_real(fg.polar(absv, ang))
        torch.cuda.synchronize()
        assert _rel(re, gc[..., 0]) < 1e-3
        assert _rel(im, gc[..., 1]) < 1e-3
    print(f"\npolar ({M},{Ncol}): real+imag rel<1e-3 vs torch"
          + ("" if fg is None else " + flag_gems"))
