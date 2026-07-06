"""PK the C4 structural-layout lowering vs torch + the REAL flag_gems.

concat / cat / hstack / vstack / pad / repeat / tile / flip / triu are lowered
from atomic IntentIR (`intent.concat` / `intent.pad` / `intent.tile` /
`intent.flip` / `intent.triu`) through the C++ ``-convert-intent-layout-to-gpu``
pass to a block-cooperative sm_120a cubin, then launched and compared head-to-head
against (1) a torch reference and (2) the real ``flag_gems.<op>`` triton kernel.

These are pure DATA MOVEMENT (no arithmetic): each output element's SOURCE
coordinate is a STATIC structural function of the output coord (an affine index /
a piecewise-by-offset select) or the pad constant — so the bar is bit-EXACT
(rel < 1e-3; identical values). Read-once/write-once memory-bound kernels — parity
is the honest floor — but this suite is a CORRECTNESS gate (C4 is coverage).

The intent MLIR is hand-written here (same shape as test/lower-layout-to-gpu.mlir),
not extracted from the corpus. Destination-passing: the appended output memref is
the result.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_layout_pk.py -q -s
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

_PASS = "convert-intent-layout-to-gpu"


def _rel(a, b):
    b = b.float()
    n = b.norm()
    return float((a.float() - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _kernel(name, Ro, Co, T, mlir_text):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(name, {"M": Ro, "N": Co}, T,
                              pass_name=_PASS, mlir_text=mlir_text)


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# concat / cat / hstack (axis=1): out[i,j] placed at the cumulative col offset.
# ---------------------------------------------------------------------------- #

def test_concat_axis1():
    M, N, T = 2048, 512, 256
    A = f"tensor<{M}x{N}xf32>"
    O = f"tensor<{M}x{2 * N}xf32>"
    mlir = f"""
func.func @cat1(%a: {A}, %b: {A}) -> {O} {{
  %c = intent.concat %a, %b {{axis = 1 : i64}} : ({A}, {A}) -> {O}
  return %c : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    b = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(M, 2 * N, device="cuda")

    k = _kernel("cat1", M, 2 * N, T, mlir)
    k.thunk([a, b, out])()
    torch.cuda.synchronize()

    ref = torch.cat([a, b], dim=1)
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        yc = fg.cat([a, b], dim=1)
        torch.cuda.synchronize()
        assert _rel(out, yc) < 1e-3
    print(f"\nconcat axis=1 ({M},{N})x2->({M},{2 * N}): exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# concat / cat / vstack (axis=0): stacked along rows.
# ---------------------------------------------------------------------------- #

def test_concat_axis0():
    M, N, T = 1024, 1024, 256
    A = f"tensor<{M}x{N}xf32>"
    O = f"tensor<{2 * M}x{N}xf32>"
    mlir = f"""
func.func @cat0(%a: {A}, %b: {A}) -> {O} {{
  %c = intent.concat %a, %b {{axis = 0 : i64}} : ({A}, {A}) -> {O}
  return %c : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    b = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(2 * M, N, device="cuda")

    k = _kernel("cat0", 2 * M, N, T, mlir)
    k.thunk([a, b, out])()
    torch.cuda.synchronize()

    ref = torch.cat([a, b], dim=0)
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        yc = fg.vstack([a, b])
        torch.cuda.synchronize()
        assert _rel(out, yc) < 1e-3
    print(f"\nconcat axis=0 ({M},{N})x2->({2 * M},{N}): exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# pad (constant_pad_nd), lo=[2,2] hi=[2,2] value=0.
# ---------------------------------------------------------------------------- #

def test_pad_constant():
    M, N, T = 1020, 1020, 256
    A = f"tensor<{M}x{N}xf32>"
    O = f"tensor<{M + 4}x{N + 4}xf32>"
    mlir = f"""
func.func @padc(%a: {A}) -> {O} {{
  %p = intent.pad %a {{lo = [2, 2], hi = [2, 2], value = 0.0 : f64}} : ({A}) -> {O}
  return %p : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(M + 4, N + 4, device="cuda")

    k = _kernel("padc", M + 4, N + 4, T, mlir)
    k.thunk([a, out])()
    torch.cuda.synchronize()

    # torch F.pad order is last-dim-first: [left, right, top, bottom].
    ref = torch.nn.functional.pad(a, [2, 2, 2, 2], mode="constant", value=0.0)
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        yp = fg.constant_pad_nd(a, [2, 2, 2, 2], 0.0)
        torch.cuda.synchronize()
        assert _rel(out, yp) < 1e-3
    print(f"\npad ({M},{N})->({M + 4},{N + 4}) value=0: exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# tile / repeat, reps=[2,2]: out[i,j] = a[i mod M, j mod N].
# ---------------------------------------------------------------------------- #

def test_tile():
    M, N, T = 512, 512, 256
    A = f"tensor<{M}x{N}xf32>"
    O = f"tensor<{2 * M}x{2 * N}xf32>"
    mlir = f"""
func.func @tile0(%a: {A}) -> {O} {{
  %t = intent.tile %a {{reps = [2, 2]}} : ({A}) -> {O}
  return %t : {O}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(2 * M, 2 * N, device="cuda")

    k = _kernel("tile0", 2 * M, 2 * N, T, mlir)
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = a.repeat(2, 2)  # torch.Tensor.repeat == numpy.tile
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        yt = fg.tile(a, [2, 2])
        torch.cuda.synchronize()
        assert _rel(out, yt) < 1e-3
    print(f"\ntile reps=[2,2] ({M},{N})->({2 * M},{2 * N}): exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# flip axes=[1] (reverse the columns).
# ---------------------------------------------------------------------------- #

def test_flip_cols():
    M, N, T = 1024, 1024, 256
    A = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @flip1(%a: {A}) -> {A} {{
  %f = intent.flip %a {{axes = [1]}} : ({A}) -> {A}
  return %f : {A}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(M, N, device="cuda")

    k = _kernel("flip1", M, N, T, mlir)
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.flip(a, dims=[1])
    assert torch.equal(out, ref)

    fg = _flag_gems()
    if fg is not None:
        yf = fg.flip(a, [1])
        torch.cuda.synchronize()
        assert _rel(out, yf) < 1e-3
    print(f"\nflip axes=[1] ({M},{N}): exact vs torch"
          + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# flip axes=[0] (reverse the rows) — exercise the row-axis remap.
# ---------------------------------------------------------------------------- #

def test_flip_rows():
    M, N, T = 1024, 1024, 256
    A = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @flip0(%a: {A}) -> {A} {{
  %f = intent.flip %a {{axes = [0]}} : ({A}) -> {A}
  return %f : {A}
}}
"""
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(M, N, device="cuda")

    k = _kernel("flip0", M, N, T, mlir)
    k.thunk([a, out])()
    torch.cuda.synchronize()

    ref = torch.flip(a, dims=[0])
    assert torch.equal(out, ref)
    print(f"\nflip axes=[0] ({M},{N}): exact vs torch")


# ---------------------------------------------------------------------------- #
# triu k=0 (upper triangle), and a k>0 diagonal offset.
# ---------------------------------------------------------------------------- #

def test_triu():
    M, T = 1024, 256
    A = f"tensor<{M}x{M}xf32>"
    for kdiag in (0, 3, -2):
        mlir = f"""
func.func @triu(%a: {A}) -> {A} {{
  %u = intent.triu %a {{k = {kdiag} : i64}} : ({A}) -> {A}
  return %u : {A}
}}
"""
        torch.manual_seed(0)
        a = torch.randn(M, M, device="cuda").contiguous()
        out = torch.empty(M, M, device="cuda")

        k = _kernel("triu", M, M, T, mlir)
        k.thunk([a, out])()
        torch.cuda.synchronize()

        ref = torch.triu(a, diagonal=kdiag)
        assert torch.equal(out, ref)

        fg = _flag_gems()
        fg_ok = False
        if fg is not None:
            try:
                yu = fg.triu(a, kdiag)
                torch.cuda.synchronize()
                assert _rel(out, yu) < 1e-3
                fg_ok = True
            except TypeError:
                # flag_gems.triu's masked (nonzero-diagonal) triton kernel trips a
                # triton launcher ABI mismatch in this environment ("function
                # takes exactly 20 arguments") — unrelated to the intent kernel,
                # whose bit-exact torch gate above already passed. flag_gems.triu
                # works for k=0; skip its check for the masked variant.
                pass
        print(f"\ntriu k={kdiag} ({M},{M}): exact vs torch"
              + (" + flag_gems" if fg_ok else ""))
