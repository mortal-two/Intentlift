"""PK the P5 complex-dtype ops vs torch (and real flag_gems) on the 5090D.

IntentIR has NO complex element type: a complex tensor is represented as its
real/imag PLANES (two same-shaped real tensors; torch.view_as_real gives the
trailing [...,2] pair). Every complex op is therefore a COMPOSITION over the
planes, lowered through -convert-intent-elementwise-to-gpu to an sm_120a cubin:

  angle(re,im)   = atan2(im, re)                 (map_binary "atan2")
  resolve_conj   -> (re, neg(im))                (map_unary "neg" on imag)
  resolve_neg    -> (neg(re), neg(im))
  complex_add    -> (re1+re2, im1+im2)
  complex_mul    -> (re1*re2 - im1*im2, re1*im2 + im1*re2)

The ONE genuinely-new primitive is atan2 (angle). The direct emitter preserves
`math.atan2`, and the shared target lowering expands it into a libdevice-free
range-reduced polynomial with explicit IEEE special-value handling.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_complex_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

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

_PASS = "convert-intent-elementwise-to-gpu"
M, N, T = 64, 256, 128
R = f"tensor<{M}x{N}xf32>"


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def _run(name, mlir_text, inputs):
    """Compile+launch a single-[M,N]-output elementwise kernel."""
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    k = RowReductionKernel(name, {"M": M, "N": N}, T,
                           pass_name=_PASS, mlir_text=mlir_text)
    go = k.thunk([*inputs, Y])
    go()
    torch.cuda.synchronize()
    return Y


def _unary(name, fn):
    return f"""
func.func @{name}(%X: {R}) -> {R} {{
  %r = intent.map_unary %X {{fn = "{fn}"}} : ({R}) -> {R}
  return %r : {R}
}}
"""


def _binary(name, fn):
    return f"""
func.func @{name}(%A: {R}, %B: {R}) -> {R} {{
  %r = intent.map_binary %A, %B {{fn = "{fn}"}} : ({R}, {R}) -> {R}
  return %r : {R}
}}
"""


def _complex_planes(seed=0):
    torch.manual_seed(seed)
    re = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    im = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    z = torch.complex(re, im)
    return re, im, z


def test_angle_atan2():
    # angle(z) = atan2(imag, real) — map_binary "atan2" over the two planes.
    re, im, z = _complex_planes()
    Y = _run("angle", _binary("angle", "atan2"), [im, re])
    ref = torch.angle(z)
    r = _rel(Y, ref)
    abs_err = float((Y - ref).abs().max())
    print(f"\nangle(atan2): rel={r:.2e} max_abs_err={abs_err:.2e}")
    assert r < 1e-3
    assert abs_err < 1e-4  # the shared target polynomial budget
    # real flag_gems.angle parity (the classified-miss kernel).
    import flag_gems
    fg = flag_gems.angle(z)
    print(f"angle vs flag_gems: rel={_rel(Y, fg):.2e}")
    assert _rel(Y, fg) < 1e-3


def test_atan_unary():
    # The atan primitive on its own (map_unary "atan"), full quadrant via atan2
    # is exercised above; here the bare poly on a wide range vs torch.atan.
    torch.manual_seed(1)
    X = (torch.rand(M, N, device="cuda", dtype=torch.float32) * 100 - 50).contiguous()
    Y = _run("atan", _unary("atan", "atan"), [X])
    ref = torch.atan(X)
    abs_err = float((Y - ref).abs().max())
    print(f"\natan: rel={_rel(Y, ref):.2e} max_abs_err={abs_err:.2e}")
    assert abs_err < 1e-4


def test_resolve_conj():
    # resolve_conj(z) = (re, -im): imag plane negated, real unchanged.
    re, im, z = _complex_planes()
    imC = _run("conj_im", _unary("conj_im", "neg"), [im])
    got = torch.stack([re, imC], dim=-1)                 # [...,2] pair
    ref = torch.view_as_real(torch.resolve_conj(z.conj()))  # actual conj values
    r = _rel(got, ref)
    print(f"\nresolve_conj: rel={r:.2e}")
    assert r < 1e-3


def test_resolve_neg():
    # resolve_neg(z) = (-re, -im): both planes negated.
    re, im, z = _complex_planes()
    reN = _run("neg_re", _unary("neg_re", "neg"), [re])
    imN = _run("neg_im", _unary("neg_im", "neg"), [im])
    got = torch.stack([reN, imN], dim=-1)
    ref = torch.view_as_real(torch.resolve_neg((-z).conj().conj()))
    r = _rel(got, ref)
    print(f"\nresolve_neg: rel={r:.2e}")
    assert r < 1e-3


def test_complex_add():
    # (a+ib) + (c+id) = (a+c, b+d): two plane adds.
    a, b, z1 = _complex_planes(0)
    c, d, z2 = _complex_planes(2)
    reO = _run("add_re", _binary("add_re", "add"), [a, c])
    imO = _run("add_im", _binary("add_im", "add"), [b, d])
    got = torch.stack([reO, imO], dim=-1)
    ref = torch.view_as_real(z1 + z2)
    r = _rel(got, ref)
    print(f"\ncomplex_add: rel={r:.2e}")
    assert r < 1e-3


def test_complex_mul():
    # (a+ib)(c+id) = (ac - bd, ad + bc): sub/add of two plane products each.
    a, b, z1 = _complex_planes(0)
    c, d, z2 = _complex_planes(2)
    ac = _run("ac", _binary("ac", "mul"), [a, c])
    bd = _run("bd", _binary("bd", "mul"), [b, d])
    ad = _run("ad", _binary("ad", "mul"), [a, d])
    bc = _run("bc", _binary("bc", "mul"), [b, c])
    reO = _run("mul_re", _binary("mul_re", "sub"), [ac, bd])
    imO = _run("mul_im", _binary("mul_im", "add"), [ad, bc])
    got = torch.stack([reO, imO], dim=-1)
    ref = torch.view_as_real(z1 * z2)
    r = _rel(got, ref)
    print(f"\ncomplex_mul: rel={r:.2e}")
    assert r < 1e-3
