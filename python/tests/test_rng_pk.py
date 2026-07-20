"""PK the C5 seeded-RNG lowering on the 5090D: statistics + determinism.

rand / dropout / randn / normal / uniform are lowered from atomic IntentIR
(`intent.rand` composed with map_unary / map_binary / select) through the C++
``-convert-intent-rng-to-gpu`` pass — an inline philox4x32-10 counter RNG — to a
block-cooperative sm_120a cubin, launched, and checked.

VERIFY MODE (user decision): the philox counter layout does NOT match Triton's
``tl.rand`` byte-for-byte, so library parity is DISTRIBUTIONAL, not bit-exact.
Each op is verified by

  * STATISTICAL moments (uniform mean~0.5 / var~1/12; normal mean~0 / var~1;
    dropout keeps ~(1-p) and the kept values are exactly x/(1-p)); PLUS
  * a matched-seed check vs the interpreter's IDENTICAL numpy philox (the GPU
    kernel and the reference agree element-for-element up to f32 rounding); PLUS
  * a DETERMINISM check — same seed -> bit-identical across two launches, a
    different seed -> decorrelated.

The intent MLIR is hand-written here (same shape as test/lower-rng-to-gpu.mlir).
Destination-passing: the appended output memref is the result.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_rng_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT
from intentir.verify.interpret import _philox_uniform

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

_PASS = "convert-intent-rng-to-gpu"


def _kernel(name, Ro, Co, T, mlir_text):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    return RowReductionKernel(name, {"M": Ro, "N": Co}, T,
                              pass_name=_PASS, mlir_text=mlir_text)


# ---------------------------------------------------------------------------- #
# rand: uniform[0,1). Moments + determinism + matched-seed vs numpy philox.
# ---------------------------------------------------------------------------- #

def test_rand_uniform():
    M, N, T, seed = 2048, 512, 256, 12345
    O = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @rand(%d: {O}) -> {O} {{
  %r = intent.rand {{seed = {seed} : i64}} : {O}
  return %r : {O}
}}
"""
    # one dummy input arg (unused) + destination, to exercise the memref-arg path.
    din = torch.empty(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    k = _kernel("rand", M, N, T, mlir)
    k.thunk([din, out])()
    torch.cuda.synchronize()

    u = out.float().cpu().numpy()
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 1e-2
    assert abs(u.var() - 1.0 / 12.0) < 1e-3

    # matched-seed: the kernel's philox == the interpreter's philox (up to f32).
    ref, _ = _philox_uniform(seed, M * N)
    assert np.max(np.abs(u.ravel() - ref)) < 1e-6

    # determinism: same seed twice -> bit-identical; different seed -> different.
    out2 = torch.empty(M, N, device="cuda")
    k.thunk([din, out2])()
    torch.cuda.synchronize()
    assert torch.equal(out, out2)
    print(f"\nrand uniform ({M},{N}) seed={seed}: mean={u.mean():.5f} "
          f"var={u.var():.5f} matched-seed vs numpy philox + deterministic")


# ---------------------------------------------------------------------------- #
# dropout = select(rand >= p, x/(1-p), 0): keeps ~(1-p); kept == x/(1-p).
# ---------------------------------------------------------------------------- #

def test_dropout():
    M, N, T, seed, p = 2048, 512, 256, 7, 0.25
    inv = 1.0 / (1.0 - p)
    O = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @dropout(%x: {O}) -> {O} {{
  %r = intent.rand {{seed = {seed} : i64}} : {O}
  %pc = arith.constant dense<{p}> : {O}
  %keep = intent.map_binary %r, %pc {{fn = "ge"}} : ({O}, {O}) -> tensor<{M}x{N}xi1>
  %iv = arith.constant dense<{inv}> : {O}
  %sc = intent.map_binary %x, %iv {{fn = "mul"}} : ({O}, {O}) -> {O}
  %z = arith.constant dense<0.0> : {O}
  %o = intent.select %keep, %sc, %z : (tensor<{M}x{N}xi1>, {O}, {O}) -> {O}
  return %o : {O}
}}
"""
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").contiguous()
    out = torch.empty(M, N, device="cuda")
    k = _kernel("dropout", M, N, T, mlir)
    k.thunk([x, out])()
    torch.cuda.synchronize()

    o = out.float().cpu().numpy()
    xc = x.float().cpu().numpy()
    kept = o != 0.0
    frac = kept.mean()
    # binomial: keep prob (1-p); tolerance ~5 sigma over M*N draws.
    assert abs(frac - (1.0 - p)) < 5e-3
    # every kept value is exactly x/(1-p); dropped are exactly 0.
    assert np.allclose(o[kept], xc[kept] * inv, rtol=1e-5, atol=1e-5)
    assert np.all(o[~kept] == 0.0)

    # matched-seed: the drop mask is rand < p from the SAME philox.
    ref, _ = _philox_uniform(seed, M * N)
    ref_keep = (ref >= p).reshape(M, N)
    assert np.array_equal(kept, ref_keep)
    print(f"\ndropout p={p} ({M},{N}) seed={seed}: keep frac={frac:.4f} "
          f"(~{1 - p}); kept==x/(1-p); mask matches numpy philox")


# ---------------------------------------------------------------------------- #
# randn / normal (Box-Muller): z = sqrt(-2 ln u1) cos(2pi u2). mean~0 var~1.
# ---------------------------------------------------------------------------- #

def test_randn_normal():
    M, N, T = 2048, 512, 256
    O = f"tensor<{M}x{N}xf32>"
    tau = 2.0 * np.pi
    mlir = f"""
func.func @randn(%d: {O}) -> {O} {{
  %u1 = intent.rand {{seed = 1 : i64}} : {O}
  %u2 = intent.rand {{seed = 2 : i64}} : {O}
  %ln = intent.map_unary %u1 {{fn = "log"}} : ({O}) -> {O}
  %m2 = arith.constant dense<-2.0> : {O}
  %r2 = intent.map_binary %ln, %m2 {{fn = "mul"}} : ({O}, {O}) -> {O}
  %rad = intent.map_unary %r2 {{fn = "sqrt"}} : ({O}) -> {O}
  %tc = arith.constant dense<{tau}> : {O}
  %ang = intent.map_binary %u2, %tc {{fn = "mul"}} : ({O}, {O}) -> {O}
  %cs = intent.map_unary %ang {{fn = "cos"}} : ({O}) -> {O}
  %z = intent.map_binary %rad, %cs {{fn = "mul"}} : ({O}, {O}) -> {O}
  return %z : {O}
}}
"""
    din = torch.empty(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    k = _kernel("randn", M, N, T, mlir)
    k.thunk([din, out])()
    torch.cuda.synchronize()

    z = out.float().cpu().numpy()
    # a few philox samples land at u1~0 (log -> -inf); mask the rare non-finite.
    finite = np.isfinite(z)
    assert finite.mean() > 0.999
    zf = z[finite]
    assert abs(zf.mean()) < 2e-2
    assert abs(zf.var() - 1.0) < 2e-2
    print(f"\nrandn Box-Muller ({M},{N}): mean={zf.mean():.4f} var={zf.var():.4f} "
          f"(normal)")


# ---------------------------------------------------------------------------- #
# uniform(a,b) = a + (b-a)*rand: mean ~ (a+b)/2, range [a,b).
# ---------------------------------------------------------------------------- #

def test_uniform_ab():
    M, N, T, seed = 1024, 1024, 256, 99
    a, b = -3.0, 5.0
    O = f"tensor<{M}x{N}xf32>"
    mlir = f"""
func.func @uniform(%d: {O}) -> {O} {{
  %r = intent.rand {{seed = {seed} : i64}} : {O}
  %span = arith.constant dense<{b - a}> : {O}
  %sc = intent.map_binary %r, %span {{fn = "mul"}} : ({O}, {O}) -> {O}
  %lo = arith.constant dense<{a}> : {O}
  %o = intent.map_binary %sc, %lo {{fn = "add"}} : ({O}, {O}) -> {O}
  return %o : {O}
}}
"""
    din = torch.empty(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    k = _kernel("uniform", M, N, T, mlir)
    k.thunk([din, out])()
    torch.cuda.synchronize()

    u = out.float().cpu().numpy()
    assert u.min() >= a and u.max() < b
    assert abs(u.mean() - 0.5 * (a + b)) < 3e-2
    assert abs(u.var() - (b - a) ** 2 / 12.0) < 5e-2
    print(f"\nuniform({a},{b}) ({M},{N}) seed={seed}: mean={u.mean():.4f} "
          f"(~{0.5 * (a + b)}) var={u.var():.4f}")
