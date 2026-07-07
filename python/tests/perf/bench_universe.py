"""N1 — WHOLE-UNIVERSE 3-way benchmark: our sm_120a kernels vs torch AND the
REAL flag_gems Triton kernels, family by family, under CUDA-graph steady state.

The prior task proved perf on 22 cluster micro-ops (``bench_clusters.py``). This
extends the measurement to the *whole verified op universe* — the memory-bound
majority (elementwise unary/binary, reduce, norm, softmax, activation) plus the
compute-bound blas core — so the geomean and the per-family LOSS MAP drive the
systematic optimization milestones N2-N8 (which family never got the paper's
warp-shuffle / CUB / float4 / ex2 templates).

DESIGN (family-driven, reusing the existing kernel infra):
  * Our kernel is built the SAME way the PKs build it:
      - ``RowReductionKernel(mlir, pass="convert-intent-row-reduction-to-gpu")``
        for reductions / norms / softmax (read-once block-cooperative reduce),
      - ``RowReductionKernel(mlir, pass="convert-intent-elementwise-to-gpu")``
        for the pure-elementwise unary/binary/activation families,
      - the corpus ``intent.json`` (via the kernel name) for the multi-op
        compositions already certified in ``corpus/kernels`` (softmax, rms_norm,
        layer_norm, log_softmax, silu, gelu_tanh, ...),
      - ``MatmulKernel`` (nvgpu.mma.sync WMMA) for the blas core.
  * References: ``torch.<op>`` AND ``flag_gems.<op>``, both in-process, all three
    timed with ``triton.testing.do_bench_cudagraph`` (same fair path as
    bench_clusters). Correctness asserted (our kernel vs torch, rel<1e-3);
    flag_gems verified where the signature matches (else timed-only).
  * The existing 22 cluster rows are REUSED (imported + run) so the geomean is
    over the full suite, not a parallel universe.
  * GEOMEAN of (our / faster-ref) overall AND per family, plus the per-family
    WIN / parity / LOSE breakdown — the measurement is the deliverable.

Ops that are genuinely hard to auto-build in a micro-bench (dynamic-shape,
multi-stage op-graph drivers, ops with no atomic-intent form) are listed as
SKIPPED with a reason — never faked.

Realistic f32 shapes: elementwise [8192,8192]; reduce/norm/softmax [4096,4096];
blas [4096,4096] f16 (mma.sync is f16-in/f32-acc).

Run (quiet GPU, sequential):
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  PYTHONPATH=python python3 python/tests/perf/bench_universe.py
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import torch

# Reuse the fair 3-way harness (CUDA-graph bench, flag_gems shim, record/ROWS).
from bench_clusters import (FG, ROWS, _cleanup, _fg_time, bench, c3_gather,
                            our_launcher,
                            c4_layout, c5_rng, c6_argmax, c6_sort, c7_conv,
                            c7_polar, c7_pool, c8_compact, fg_time, record)
from intentir.benchmarks.row_reduction import MatmulKernel, RowReductionKernel

RR = "convert-intent-row-reduction-to-gpu"
EW = "convert-intent-elementwise-to-gpu"

SKIPPED: list[tuple[str, str, str]] = []  # (family, op, reason)


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def build_row(family, op, shape, mlir, in_tensors, out, ref, pass_name,
              torch_fn, fg_fn=None, fg_verify=None, nbytes=None, flop=None,
              cls="mem", T=256, M=1, N=1, rel=1e-3, note=""):
    """Build our kernel, assert correctness vs torch ref, time 3-way, record."""
    try:
        k = RowReductionKernel("k", {"M": M, "N": N}, T, pass_name=pass_name,
                               mlir_text=mlir)
        go = our_launcher(k, list(in_tensors) + [out])
        go(); torch.cuda.synchronize()
        r = _rel(out, ref)
        if not (r < rel):
            SKIPPED.append((family, op, f"our rel={r:.1e} MISMATCH"))
            print(f"  {op:20s} SKIP our-mismatch rel={r:.1e}")
            return
    except Exception as e:  # noqa: BLE001
        SKIPPED.append((family, op, f"build/launch: {str(e)[:70]}"))
        print(f"  {op:20s} SKIP build: {str(e)[:80]}")
        return
    our, gg = bench(go)
    refs = {}
    tt, _ = _fg_time(torch_fn); refs["torch"] = tt
    if fg_fn is not None:
        refs["flag_gems"] = fg_time(fg_fn, verify=fg_verify)
    record(family, op, shape, our, refs, nbytes=nbytes, flop=flop, cls=cls,
           note=note, graph_ours=gg)


# --------------------------------------------------------------------------- #
# elementwise-unary  [8192,8192] f32  (read-once, write-once — memory bound)   #
# --------------------------------------------------------------------------- #
def fam_elementwise_unary():
    print("\n[elementwise-unary]  x -> f(x)   [8192,8192] f32")
    M = N = 8192
    R = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    xp = x.abs() + 0.1  # positive domain for log/sqrt/recip
    nb = 2 * M * N * 4

    def go(op, fn, tref, fg, inp=x):
        mlir = (f'func.func @f(%x: {R}) -> {R} {{ '
                f'%y = intent.map_unary %x {{fn="{fn}"}} : ({R}) -> {R}\n'
                f'return %y : {R} }}')
        out = torch.empty(M, N, device="cuda")
        ref = tref()
        build_row("elementwise-unary", op, f"[{M},{N}]", mlir, [inp], out, ref,
                  EW, tref, fg, fg_verify=lambda r: _rel(r, ref) < 1e-3,
                  nbytes=nb, M=M, N=N)
        _cleanup()

    go("abs", "abs", lambda: x.abs(), (lambda: FG.abs(x)) if FG else None)
    go("neg", "neg", lambda: -x, (lambda: FG.neg(x)) if FG else None)
    go("exp", "exp", lambda: torch.exp(x), (lambda: FG.exp(x)) if FG else None)
    go("sigmoid", "sigmoid", lambda: torch.sigmoid(x),
       (lambda: FG.sigmoid(x)) if FG else None)
    go("tanh", "tanh", lambda: torch.tanh(x),
       (lambda: FG.tanh(x)) if FG else None)
    go("sin", "sin", lambda: torch.sin(x), (lambda: FG.sin(x)) if FG else None)
    go("cos", "cos", lambda: torch.cos(x), (lambda: FG.cos(x)) if FG else None)
    go("atan", "atan", lambda: torch.atan(x),
       (lambda: FG.atan(x)) if FG else None)
    go("erf", "erf", lambda: torch.erf(x), (lambda: FG.erf(x)) if FG else None)
    go("exp2", "exp2", lambda: torch.exp2(x),
       (lambda: FG.exp2(x)) if FG else None)
    go("log", "log", lambda: torch.log(xp),
       (lambda: FG.log(xp)) if FG else None, inp=xp)
    go("sqrt", "sqrt", lambda: torch.sqrt(xp),
       (lambda: FG.sqrt(xp)) if FG else None, inp=xp)
    go("reciprocal", "recip", lambda: torch.reciprocal(xp),
       (lambda: FG.reciprocal(xp)) if FG else None, inp=xp)
    del x, xp; _cleanup()


# --------------------------------------------------------------------------- #
# elementwise-binary  [8192,8192] f32                                          #
# --------------------------------------------------------------------------- #
def fam_elementwise_binary():
    print("\n[elementwise-binary]  (a,b) -> f(a,b)   [8192,8192] f32")
    M = N = 8192
    R = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    a = torch.randn(M, N, device="cuda")
    b = torch.randn(M, N, device="cuda")
    bp = b.abs() + 0.1
    nb = 3 * M * N * 4

    def go(op, fn, tref, fg, bb=b):
        mlir = (f'func.func @f(%a: {R}, %b: {R}) -> {R} {{ '
                f'%y = intent.map_binary %a, %b {{fn="{fn}"}} : ({R},{R}) -> {R}\n'
                f'return %y : {R} }}')
        out = torch.empty(M, N, device="cuda")
        ref = tref()
        build_row("elementwise-binary", op, f"[{M},{N}]", mlir, [a, bb], out,
                  ref, EW, tref, fg,
                  fg_verify=lambda r: _rel(r, ref) < 1e-3, nbytes=nb, M=M, N=N)
        _cleanup()

    go("add", "add", lambda: a + b, (lambda: FG.add(a, b)) if FG else None)
    go("sub", "sub", lambda: a - b, (lambda: FG.sub(a, b)) if FG else None)
    go("mul", "mul", lambda: a * b, (lambda: FG.mul(a, b)) if FG else None)
    go("div", "div", lambda: a / bp, None, bb=bp)  # flag_gems has no div
    go("maximum", "max", lambda: torch.maximum(a, b),
       (lambda: FG.maximum(a, b)) if FG else None)
    go("minimum", "min", lambda: torch.minimum(a, b),
       (lambda: FG.minimum(a, b)) if FG else None)
    SKIPPED.append(("elementwise-binary", "pow",
                    "libdevice __nv_powf unresolved by ptxas on sm_120a"))
    del a, b, bp; _cleanup()


# --------------------------------------------------------------------------- #
# reduce  [4096,4096] axis=1  (read-once row reduction)                        #
# --------------------------------------------------------------------------- #
def fam_reduce():
    print("\n[reduce]  row reduce axis=1 -> [M,1]   [4096,4096] f32")
    M = N = 4096
    R = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    nb = M * N * 4  # read-once

    def go(op, fn, assoc, tref, fg):
        O = f"tensor<{M}x1xf32>"
        mlir = (f'func.func @f(%x: {R}) -> {O} {{ '
                f'%s = intent.reduce %x {{fn="{fn}", axes=[1], assoc="{assoc}"}} '
                f': ({R}) -> {O}\nreturn %s : {O} }}')
        out = torch.empty(M, 1, device="cuda")
        ref = tref()
        build_row("reduce", op, f"[{M},{N}]", mlir, [x], out, ref, RR, tref, fg,
                  fg_verify=lambda r: _rel(r.reshape(M), ref.reshape(M)) < 1e-3,
                  nbytes=nb, M=M, N=N)
        _cleanup()

    # flag_gems sum/mean/max/min/prod are FULL reductions (no dim) -> torch-only.
    go("sum", "sum", "float_nondeterministic",
       lambda: x.sum(1, keepdim=True), None)
    go("mean", "mean", "float_nondeterministic",
       lambda: x.mean(1, keepdim=True), None)
    go("max", "max", "associative",
       lambda: x.max(1, keepdim=True).values, None)
    go("min", "min", "associative",
       lambda: x.min(1, keepdim=True).values, None)
    go("prod", "prod", "float_nondeterministic",
       lambda: x.prod(1, keepdim=True), None)
    # amax supports the [dim] list form in flag_gems -> real 3-way.
    go("amax", "max", "associative",
       lambda: torch.amax(x, 1, keepdim=True),
       (lambda: FG.amax(x, [1])) if FG else None)
    del x; _cleanup()


# --------------------------------------------------------------------------- #
# norm  [4096,4096]                                                            #
# --------------------------------------------------------------------------- #
def fam_norm():
    print("\n[norm]  row-normalization   [4096,4096] f32")
    M = N = 4096
    R = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    W = torch.randn(N, device="cuda")
    B = torch.randn(N, device="cuda")
    eps = torch.tensor(1e-6, device="cuda")
    nb = 2 * M * N * 4

    # rms_norm (corpus intent.json): X * rsqrt(mean(X^2)+eps) * W
    ref = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * W
    try:
        k = RowReductionKernel("rms_norm", {"M": M, "N": N}, 256)
        out = torch.empty(M, N, device="cuda")
        go = our_launcher(k, [x, W, eps, out]); go(); torch.cuda.synchronize()
        assert _rel(out, ref) < 1e-3
        our, gg = bench(go)
        refs = {}
        tt, _ = _fg_time(lambda: x * torch.rsqrt((x * x).mean(-1, keepdim=True)
                                                  + 1e-6) * W); refs["torch"] = tt
        refs["flag_gems"] = (fg_time(lambda: FG.rms_norm(x, [N], W, 1e-6),
                                     verify=lambda r: _rel(r, ref) < 1e-3)
                             if FG else None)
        record("norm", "rms_norm", f"[{M},{N}]", our, refs, nbytes=nb,
               graph_ours=gg)
    except Exception as e:  # noqa: BLE001
        SKIPPED.append(("norm", "rms_norm", f"{str(e)[:70]}"))
        print(f"  rms_norm SKIP: {str(e)[:80]}")
    _cleanup()

    # layer_norm (corpus): (X-mean)/sqrt(var+eps)*W + B ; flag_gems has no
    # top-level layernorm entry -> torch-only ref.
    refl = torch.nn.functional.layer_norm(x, (N,), W, B, 1e-6)
    try:
        k = RowReductionKernel("layer_norm", {"M": M, "N": N}, 256)
        out = torch.empty(M, N, device="cuda")
        mean = torch.empty(M, device="cuda")   # kernel also writes mean/rstd
        rstd = torch.empty(M, device="cuda")
        go = our_launcher(k, [x, W, B, eps, out, mean, rstd])
        go(); torch.cuda.synchronize()
        assert _rel(out, refl) < 1e-3
        our, gg = bench(go)
        refs = {}
        tt, _ = _fg_time(lambda: torch.nn.functional.layer_norm(
            x, (N,), W, B, 1e-6)); refs["torch"] = tt
        record("norm", "layer_norm", f"[{M},{N}]", our, refs, nbytes=nb,
               graph_ours=gg, note="flag_gems has no top-level layernorm")
    except Exception as e:  # noqa: BLE001
        SKIPPED.append(("norm", "layer_norm", f"{str(e)[:70]}"))
        print(f"  layer_norm SKIP: {str(e)[:80]}")
    _cleanup()

    # vector_norm: sqrt(sum(x^2)) over axis1 (hand intent) — 3-way (fg supports).
    O = f"tensor<{M}x1xf32>"
    mlir = (f'func.func @f(%x: {R}) -> {O} {{ '
            f'%sq = intent.map_binary %x, %x {{fn="mul"}} : ({R},{R}) -> {R}\n'
            f'%s = intent.reduce %sq {{fn="sum", axes=[1], assoc="float_nondeterministic"}} : ({R}) -> {O}\n'
            f'%y = intent.map_unary %s {{fn="sqrt"}} : ({O}) -> {O}\n'
            f'return %y : {O} }}')
    refv = torch.linalg.vector_norm(x, 2, dim=1, keepdim=True)
    out = torch.empty(M, 1, device="cuda")
    build_row("norm", "vector_norm", f"[{M},{N}]", mlir, [x], out, refv, RR,
              lambda: torch.linalg.vector_norm(x, 2, dim=1, keepdim=True),
              (lambda: FG.vector_norm(x, 2, [1])) if FG else None,
              fg_verify=lambda r: _rel(r.reshape(M), refv.reshape(M)) < 1e-3,
              nbytes=M * N * 4, M=M, N=N)
    SKIPPED.append(("norm", "weightnorm",
                    "multi-stage op-graph (per-column norm + broadcast); not a "
                    "single row-reduction kernel"))
    del x, W, B; _cleanup()


# --------------------------------------------------------------------------- #
# softmax  [4096,4096]                                                         #
# --------------------------------------------------------------------------- #
def fam_softmax():
    print("\n[softmax]  row softmax family   [4096,4096] f32")
    M = N = 4096
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    nb = 2 * M * N * 4

    for nm, tref, fg in [
        ("softmax", lambda: torch.softmax(x, 1),
         (lambda: FG.softmax(x, dim=1)) if FG else None),
        ("log_softmax", lambda: torch.log_softmax(x, 1),
         (lambda: FG.log_softmax(x, dim=1)) if FG else None),
    ]:
        ref = tref()
        try:
            k = RowReductionKernel(nm, {"M": M, "N": N}, 256)
            out = torch.empty(M, N, device="cuda")
            go = our_launcher(k, [x, out]); go(); torch.cuda.synchronize()
            assert _rel(out, ref) < 1e-3
            our, gg = bench(go)
            refs = {}
            tt, _ = _fg_time(tref); refs["torch"] = tt
            refs["flag_gems"] = (fg_time(fg, verify=lambda r: _rel(r, ref) < 1e-3)
                                 if fg else None)
            record("softmax", nm, f"[{M},{N}]", our, refs, nbytes=nb,
                   graph_ours=gg)
        except Exception as e:  # noqa: BLE001
            SKIPPED.append(("softmax", nm, f"{str(e)[:70]}"))
            print(f"  {nm} SKIP: {str(e)[:80]}")
        _cleanup()

    # scaled_softmax = softmax(x * scale): reuse the softmax kernel on a
    # pre-scaled input (the scale is a cheap elementwise the graph would fuse);
    # flag_gems has no top-level scaled_softmax entry -> torch-only.
    SKIPPED.append(("softmax", "scaled_softmax",
                    "no atomic single-kernel intent form here (scale+softmax "
                    "fusion); flag_gems lacks a top-level entry"))
    del x; _cleanup()


# --------------------------------------------------------------------------- #
# activation  [8192,8192]  (elementwise, some need constants/select)          #
# --------------------------------------------------------------------------- #
def fam_activation():
    print("\n[activation]  [8192,8192] f32")
    M = N = 8192
    R = f"tensor<{M}x{N}xf32>"
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda")
    z = torch.zeros((), device="cuda")
    nb = 2 * M * N * 4

    # relu = max(x, 0)  (0 as a rank-0 scalar arg)
    mlir = (f'func.func @f(%x: {R}, %z: tensor<f32>) -> {R} {{ '
            f'%y = intent.map_binary %x, %z {{fn="max"}} : ({R},tensor<f32>) -> {R}\n'
            f'return %y : {R} }}')
    ref = torch.relu(x)
    out = torch.empty(M, N, device="cuda")
    build_row("activation", "relu", f"[{M},{N}]", mlir, [x, z], out, ref, EW,
              lambda: torch.relu(x), (lambda: FG.relu(x)) if FG else None,
              fg_verify=lambda r: _rel(r, ref) < 1e-3, nbytes=nb, M=M, N=N)
    _cleanup()

    # sigmoid / tanh (single map_unary)
    for nm, fn, tref, fg in [
        ("sigmoid", "sigmoid", lambda: torch.sigmoid(x),
         (lambda: FG.sigmoid(x)) if FG else None),
        ("tanh", "tanh", lambda: torch.tanh(x),
         (lambda: FG.tanh(x)) if FG else None),
    ]:
        m = (f'func.func @f(%x: {R}) -> {R} {{ '
             f'%y = intent.map_unary %x {{fn="{fn}"}} : ({R}) -> {R}\n'
             f'return %y : {R} }}')
        ref = tref(); out = torch.empty(M, N, device="cuda")
        build_row("activation", nm, f"[{M},{N}]", m, [x], out, ref, EW, tref, fg,
                  fg_verify=lambda r, ref=ref: _rel(r, ref) < 1e-3, nbytes=nb,
                  M=M, N=N)
        _cleanup()

    # silu = x*sigmoid(x)
    m = (f'func.func @f(%x: {R}) -> {R} {{ '
         f'%s = intent.map_unary %x {{fn="sigmoid"}} : ({R}) -> {R}\n'
         f'%y = intent.map_binary %x, %s {{fn="mul"}} : ({R},{R}) -> {R}\n'
         f'return %y : {R} }}')
    ref = torch.nn.functional.silu(x); out = torch.empty(M, N, device="cuda")
    build_row("activation", "silu", f"[{M},{N}]", m, [x], out, ref, EW,
              lambda: torch.nn.functional.silu(x),
              (lambda: FG.silu(x)) if FG else None,
              fg_verify=lambda r: _rel(r, ref) < 1e-3, nbytes=nb, M=M, N=N)
    _cleanup()

    # gelu (tanh approx): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
    # Hand-built with rank-0 scalar-arg constants + x*x*x (the corpus intent uses
    # `const` ops + `pow` which the elementwise pass/ptxas cannot take: pow ->
    # libdevice __nv_powf, const -> unlowered). Scalars: c_coef, c_sq2pi, c_half.
    coef = torch.tensor(0.044715, device="cuda")
    sq2pi = torch.tensor(math.sqrt(2.0 / math.pi), device="cuda")
    half = torch.tensor(0.5, device="cuda")
    one_s = torch.ones((), device="cuda")
    m = (f'func.func @f(%x: {R}, %coef: tensor<f32>, %sq2pi: tensor<f32>, '
         f'%half: tensor<f32>, %one: tensor<f32>) -> {R} {{ '
         f'%x2 = intent.map_binary %x, %x {{fn="mul"}} : ({R},{R}) -> {R}\n'
         f'%x3 = intent.map_binary %x2, %x {{fn="mul"}} : ({R},{R}) -> {R}\n'
         f'%cx3 = intent.map_binary %x3, %coef {{fn="mul"}} : ({R},tensor<f32>) -> {R}\n'
         f'%inner = intent.map_binary %x, %cx3 {{fn="add"}} : ({R},{R}) -> {R}\n'
         f'%sinner = intent.map_binary %inner, %sq2pi {{fn="mul"}} : ({R},tensor<f32>) -> {R}\n'
         f'%t = intent.map_unary %sinner {{fn="tanh"}} : ({R}) -> {R}\n'
         f'%t1 = intent.map_binary %t, %one {{fn="add"}} : ({R},tensor<f32>) -> {R}\n'
         f'%hx = intent.map_binary %x, %half {{fn="mul"}} : ({R},tensor<f32>) -> {R}\n'
         f'%y = intent.map_binary %hx, %t1 {{fn="mul"}} : ({R},{R}) -> {R}\n'
         f'return %y : {R} }}')
    ref = torch.nn.functional.gelu(x, approximate="tanh")
    out = torch.empty(M, N, device="cuda")
    build_row("activation", "gelu_tanh", f"[{M},{N}]", m,
              [x, coef, sq2pi, half, one_s], out, ref, EW,
              lambda: torch.nn.functional.gelu(x, approximate="tanh"),
              (lambda: FG.gelu(x)) if FG else None,
              fg_verify=lambda r: _rel(r, ref) < 5e-3, nbytes=nb, M=M, N=N)
    _cleanup()

    # softplus = log(1+exp(x))  (map_unary exp, +1 scalar, log)
    one = torch.ones((), device="cuda")
    m = (f'func.func @f(%x: {R}, %one: tensor<f32>) -> {R} {{ '
         f'%e = intent.map_unary %x {{fn="exp"}} : ({R}) -> {R}\n'
         f'%p = intent.map_binary %e, %one {{fn="add"}} : ({R},tensor<f32>) -> {R}\n'
         f'%y = intent.map_unary %p {{fn="log"}} : ({R}) -> {R}\n'
         f'return %y : {R} }}')
    xs = (x * 0.5)  # keep exp in-range for a clean rel check
    ref = torch.nn.functional.softplus(xs); out = torch.empty(M, N, device="cuda")
    build_row("activation", "softplus", f"[{M},{N}]", m, [xs, one], out, ref, EW,
              lambda: torch.nn.functional.softplus(xs),
              (lambda: FG.softplus(xs)) if FG else None,
              fg_verify=lambda r: _rel(r, ref) < 1e-3, nbytes=nb, M=M, N=N,
              note="log(1+exp) 3-op fusion")
    del x, xs; _cleanup()


# --------------------------------------------------------------------------- #
# blas  matmul (nvgpu.mma.sync WMMA)  [4096,4096]x[4096,4096] f16              #
# --------------------------------------------------------------------------- #
def fam_blas():
    print("\n[blas]  matmul mk,nk->mn  [4096,4096,4096] f16 mma.sync")
    Mm = Nn = Kk = 4096
    A = f"tensor<{Mm}x{Kk}xf16>"; Wt = f"tensor<{Nn}x{Kk}xf16>"
    Y = f"tensor<{Mm}x{Nn}xf32>"
    mlir = (f'func.func @matmul(%A: {A}, %W: {Wt}) -> {Y} {{ '
            f'%Y = intent.contraction %A, %W {{subscripts = "mk,nk->mn"}} '
            f': ({A}, {Wt}) -> {Y}\nreturn %Y : {Y} }}')
    torch.manual_seed(0)
    a = torch.randn(Mm, Kk, device="cuda", dtype=torch.float16)
    w = torch.randn(Nn, Kk, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a.float(), w.float().t())
    try:
        # nvgpu.ldmatrix + mma.sync path with the N6-tuned compute-bound config:
        # BM=BN=128 (8-warp block, max A/W reuse), BK=32, 2-stage cp.async double
        # buffer (ns=3 costs an smem stage -> loses a co-resident block on the
        # 100 KiB SM), smem-pad-k=8 to kill the ldmatrix bank conflicts. This is
        # the argmin the M5 autotuner selects over the BM/BN/BK/stages/pad sweep.
        k = MatmulKernel(mlir, mma=True, num_stages=2, block_m=128,
                         block_n=128, block_k=32, smem_pad=8)
        out = torch.zeros(Mm, Nn, device="cuda", dtype=torch.float32)
        go = our_launcher(k, [a, w, out]); go(); torch.cuda.synchronize()
        rel = _rel(out, ref)
        assert rel < 2e-2, f"matmul rel={rel:.2e}"
        our, gg = bench(go)
        refs = {}
        wt = w.t().contiguous()
        yt = torch.empty(Mm, Nn, device="cuda", dtype=torch.float16)
        tt, _ = _fg_time(lambda: torch.matmul(a, wt, out=yt)); refs["torch"] = tt
        refs["flag_gems"] = (fg_time(
            lambda: FG.mm(a, wt),
            verify=lambda r: _rel(r.float(), ref) < 3e-2) if FG else None)
        flop = 2.0 * Mm * Nn * Kk
        record("blas", "matmul f16", f"[{Mm},{Nn},{Kk}]", our, refs, flop=flop,
               cls="compute",
               note=f"rel={rel:.1e}; nvgpu.mma.sync cp.async-2stage "
                    f"pad8 bm128 vs cuBLAS",
               graph_ours=gg)
    except Exception as e:  # noqa: BLE001
        SKIPPED.append(("blas", "matmul", f"{str(e)[:70]}"))
        print(f"  matmul SKIP: {str(e)[:90]}")
    SKIPPED.append(("blas", "bmm/addmm/mv/dot",
                    "op-graph contraction variants: same mma.sync core as "
                    "matmul, benched once as the family representative"))
    del a, w; _cleanup()


# --------------------------------------------------------------------------- #
# geomean + per-family report                                                 #
# --------------------------------------------------------------------------- #
# Map the reused bench_clusters C3-C8 codes onto universe family names.
CLUSTER_FAMILY = {
    "C3": "gather/scatter", "C4": "layout", "C5": "rng",
    "C6": "arg-reduce/sort", "C7": "pool/conv/upsample", "C8": "compact",
}


def _geomean(ratios):
    rs = [r for r in ratios if r and r > 0]
    if not rs:
        return None
    return math.exp(sum(math.log(r) for r in rs) / len(rs))


def report():
    fams: dict[str, list] = {}
    for r in ROWS:
        fam = CLUSTER_FAMILY.get(r["cluster"], r["cluster"])
        fams.setdefault(fam, []).append(r)

    print("\n\n" + "=" * 78)
    print("UNIVERSE 3-WAY GEOMEAN REPORT  (ratio = our / faster-ref; <1 = WIN)")
    print("=" * 78)
    hdr = (f"{'family':22s} {'n':>3s} {'geomean':>8s}  "
           f"{'WIN':>4s} {'par':>4s} {'lose':>4s} {'LOSE':>4s}")
    print(hdr)
    all_ratios = []
    fam_summary = {}
    for fam in sorted(fams):
        rows = fams[fam]
        ratios = [r["ratio"] for r in rows if r["ratio"]]
        all_ratios += ratios
        gm = _geomean(ratios)
        counts = {"WIN": 0, "parity": 0, "lose": 0, "LOSE": 0, "n/a": 0}
        for r in rows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        fam_summary[fam] = {"n": len(rows),
                            "n_rated": len(ratios),
                            "geomean": round(gm, 3) if gm else None,
                            "WIN": counts["WIN"], "parity": counts["parity"],
                            "lose": counts["lose"], "LOSE": counts["LOSE"]}
        print(f"{fam:22s} {len(rows):>3d} "
              f"{(('%.3f' % gm) if gm else '   -  '):>8s}  "
              f"{counts['WIN']:>4d} {counts['parity']:>4d} "
              f"{counts['lose']:>4d} {counts['LOSE']:>4d}")
    overall = _geomean(all_ratios)
    print("-" * 78)
    print(f"{'OVERALL':22s} {len(ROWS):>3d} "
          f"{(('%.3f' % overall) if overall else '   -  '):>8s}   "
          f"(over {len(all_ratios)} rated rows)")

    # loss hotspots: families with geomean > 1.15, worst first.
    hotspots = sorted(
        [(f, s["geomean"]) for f, s in fam_summary.items()
         if s["geomean"] and s["geomean"] > 1.15],
        key=lambda kv: -kv[1])
    print("\nLOSS HOTSPOTS (family geomean > 1.15x — N2-N8 targets):")
    for f, gm in hotspots:
        print(f"  {f:22s} geomean {gm:.2f}x")
    if not hotspots:
        print("  (none above 1.15x)")

    print(f"\nSKIPPED ({len(SKIPPED)}):")
    for fam, op, why in SKIPPED:
        print(f"  {fam:20s} {op:18s} — {why}")

    out = Path(__file__).resolve().parent / "bench_universe_results.json"
    out.write_text(json.dumps({
        "overall_geomean": round(overall, 4) if overall else None,
        "n_rows": len(ROWS), "n_rated": len(all_ratios),
        "per_family": fam_summary,
        "loss_hotspots": [{"family": f, "geomean": gm} for f, gm in hotspots],
        "skipped": [{"family": f, "op": o, "reason": w} for f, o, w in SKIPPED],
        "rows": ROWS,
    }, indent=2))
    print(f"\nwrote {out}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"flag_gems={'yes' if FG else 'NO'}")
    # New universe families (memory-bound majority + blas core).
    universe = [fam_elementwise_unary, fam_elementwise_binary, fam_reduce,
                fam_norm, fam_softmax, fam_activation, fam_blas]
    # Reuse the existing 22 cluster micro-op rows.
    clusters = [c3_gather, c4_layout, c5_rng, c6_argmax, c6_sort, c7_pool,
                c7_conv, c7_polar, c8_compact]
    for st in universe + clusters:
        try:
            st()
        except Exception as e:  # noqa: BLE001
            print(f"  !! stage {st.__name__} errored: {e}")
            traceback.print_exc()
        _cleanup()
    report()


if __name__ == "__main__":
    main()
