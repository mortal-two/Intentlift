"""PK the C0 scalar-unary unblockers vs a torch reference on the 5090D.

Each new fn is lowered from a tiny [M,N] intent.map_unary (or a small
composition) through -convert-intent-elementwise-to-gpu to an sm_120a cubin,
launched, and compared to torch at rel < 1e-3. This is what PROVES the
libdevice-safe lowering is numerically correct end-to-end:

  cos/sin/tan  -> nvvm.{cos,sin}.approx.f  (tan = sin/cos)
  exp2         -> ex2.approx.ftz.f (no log2e prescale)
  erf          -> shared target A&S polynomial x native ex2
  isnan        -> arith.cmpf uno (feeds select)

and the composable families (no new fn, built from primitives) lower through
the same pipeline:

  softplus = log(1+exp(x)) ; log_sigmoid = -softplus(-x) ;
  elu = select(x>0, x, a*(exp(x)-1)) ; clamp = min(max(x,lo),hi) ;
  threshold = select(x>t, x, v) ; nan_to_num = select(isnan(x), 0, x)

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_unary_fns_pk.py -q -s
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


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def _kernel(name, mlir_text):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    return RowReductionKernel(name, {"M": M, "N": N}, T,
                              pass_name=_PASS, mlir_text=mlir_text)


def _run(name, mlir_text, inputs):
    """Compile+launch; `inputs` are the func-sig inputs (Y output appended)."""
    Y = torch.empty(M, N, device="cuda", dtype=torch.float32)
    k = _kernel(name, mlir_text)
    go = k.thunk([*inputs, Y])
    go()
    torch.cuda.synchronize()
    return Y


R = f"tensor<{M}x{N}xf32>"


def _unary_mlir(name, fn):
    return f"""
func.func @{name}(%X: {R}) -> {R} {{
  %r = intent.map_unary %X {{fn = "{fn}"}} : ({R}) -> {R}
  return %r : {R}
}}
"""


@pytest.mark.parametrize("fn,ref,lo,hi", [
    ("cos", torch.cos, -3.0, 3.0),
    ("sin", torch.sin, -3.0, 3.0),
    ("tan", torch.tan, -1.0, 1.0),      # avoid the pi/2 pole
    ("exp2", torch.exp2, -4.0, 4.0),
    ("erf", torch.erf, -3.0, 3.0),
])
def test_unary_fn(fn, ref, lo, hi):
    torch.manual_seed(0)
    X = (torch.rand(M, N, device="cuda", dtype=torch.float32) * (hi - lo) + lo)
    X = X.contiguous()
    Y = _run(fn, _unary_mlir(fn, fn), [X])
    r = _rel(Y, ref(X))
    print(f"\n{fn}: rel={r:.2e}")
    assert r < 1e-3


def test_isnan_nan_to_num():
    # nan_to_num = select(isnan(x), 0, x): the isnan predicate feeds select.
    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    X.view(-1)[::7] = float("nan")
    Z = torch.zeros((), device="cuda", dtype=torch.float32)
    mlir = f"""
func.func @nan_to_num(%X: {R}, %Zero: tensor<f32>) -> {R} {{
  %m = intent.map_unary %X {{fn = "isnan"}} : ({R}) -> tensor<{M}x{N}xi1>
  %r = intent.select %m, %Zero, %X : (tensor<{M}x{N}xi1>, tensor<f32>, {R}) -> {R}
  return %r : {R}
}}
"""
    Y = _run("nan_to_num", mlir, [X, Z])
    ref = torch.nan_to_num(X, nan=0.0)
    r = _rel(Y, ref)
    print(f"\nnan_to_num: rel={r:.2e}")
    assert r < 1e-3


def test_softplus_composition():
    # softplus(x) = log(1 + exp(x)) — exp + add const 1 + log through the pipeline.
    torch.manual_seed(0)
    X = (torch.rand(M, N, device="cuda", dtype=torch.float32) * 8 - 4).contiguous()
    One = torch.ones((), device="cuda", dtype=torch.float32)
    mlir = f"""
func.func @softplus(%X: {R}, %One: tensor<f32>) -> {R} {{
  %e = intent.map_unary %X {{fn = "exp"}} : ({R}) -> {R}
  %s = intent.map_binary %e, %One {{fn = "add"}} : ({R}, tensor<f32>) -> {R}
  %r = intent.map_unary %s {{fn = "log"}} : ({R}) -> {R}
  return %r : {R}
}}
"""
    Y = _run("softplus", mlir, [X, One])
    ref = torch.nn.functional.softplus(X)
    r = _rel(Y, ref)
    print(f"\nsoftplus: rel={r:.2e}")
    assert r < 1e-3


def test_elu_composition():
    # elu(x) = select(x>0, x, exp(x)-1)  (alpha=1)
    torch.manual_seed(0)
    X = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    Zero = torch.zeros((), device="cuda", dtype=torch.float32)
    One = torch.ones((), device="cuda", dtype=torch.float32)
    mlir = f"""
func.func @elu(%X: {R}, %Zero: tensor<f32>, %One: tensor<f32>) -> {R} {{
  %c = intent.map_binary %X, %Zero {{fn = "gt"}} : ({R}, tensor<f32>) -> tensor<{M}x{N}xi1>
  %e = intent.map_unary %X {{fn = "exp"}} : ({R}) -> {R}
  %em1 = intent.map_binary %e, %One {{fn = "sub"}} : ({R}, tensor<f32>) -> {R}
  %r = intent.select %c, %X, %em1 : (tensor<{M}x{N}xi1>, {R}, {R}) -> {R}
  return %r : {R}
}}
"""
    Y = _run("elu", mlir, [X, Zero, One])
    ref = torch.nn.functional.elu(X)
    r = _rel(Y, ref)
    print(f"\nelu: rel={r:.2e}")
    assert r < 1e-3


def test_clamp_composition():
    # clamp(x, lo, hi) = min(max(x, lo), hi)
    torch.manual_seed(0)
    X = (torch.randn(M, N, device="cuda", dtype=torch.float32) * 3).contiguous()
    Lo = torch.full((), -1.0, device="cuda", dtype=torch.float32)
    Hi = torch.full((), 1.0, device="cuda", dtype=torch.float32)
    mlir = f"""
func.func @clamp(%X: {R}, %Lo: tensor<f32>, %Hi: tensor<f32>) -> {R} {{
  %a = intent.map_binary %X, %Lo {{fn = "max"}} : ({R}, tensor<f32>) -> {R}
  %r = intent.map_binary %a, %Hi {{fn = "min"}} : ({R}, tensor<f32>) -> {R}
  return %r : {R}
}}
"""
    Y = _run("clamp", mlir, [X, Lo, Hi])
    ref = torch.clamp(X, -1.0, 1.0)
    r = _rel(Y, ref)
    print(f"\nclamp: rel={r:.2e}")
    assert r < 1e-3
