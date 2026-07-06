"""PK the C7 convolution lowering vs torch + real flag_gems (conv1d/2d/3d, fwd).

Forward conv1d / conv2d / conv3d are lowered from atomic IntentIR (`intent.conv`)
through the C++ ``-convert-intent-conv-to-gpu`` pass to a block-cooperative
sm_120a cubin, then launched and compared head-to-head against (1) a torch
reference (``torch.nn.functional.conv{1,2,3}d``) and (2) the real
``flag_gems.conv{1,2,3}d`` triton kernel.

The mechanism is the `intent.pool` windowed per-output-cell reduce EXTENDED with
an input-channel sum and a per-tap weight multiply (direct windowed-weighted
accumulation, NO im2col): each output cell (n,co,*o) =
``bias[co] + sum_{ci,*k} in[n,ci,*(o*stride - pad + k*dil)] * w[co,ci_local,*k]``
with a single in-bounds scf.if per (ci,tap) and channel grouping. conv1d/2d/3d
share the accumulation core parameterized by the spatial rank R. rel<1e-3 (fp
accumulation order). Static shapes, O_last % T == 0.

Destination-passing: signature = (input, weight[, bias]) ++ [output].

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_conv_pk.py -q -s
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


def _kernel(name, T, mlir_text):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(name, {"M": 1, "N": 1}, T,
                              pass_name="convert-intent-conv-to-gpu",
                              mlir_text=mlir_text)


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


def _tt(shape):
    return "tensor<" + "x".join(str(s) for s in shape) + "xf32>"


def _mlir(name, xs, ws, bs, ost, arr):
    """A single-conv func. `arr` = (stride, padding, dilation, groups) with
    stride/padding/dilation as length-R lists."""
    stride, pad, dil, groups = arr
    st = ", ".join(str(v) for v in stride)
    pd = ", ".join(str(v) for v in pad)
    dl = ", ".join(str(v) for v in dil)
    I, W, O = _tt(xs), _tt(ws), _tt(ost)
    attrs = (f'stride = [{st}], padding = [{pd}], dilation = [{dl}], '
             f'groups = {groups} : i64')
    if bs is not None:
        B = _tt(bs)
        return f"""
func.func @{name}(%x: {I}, %w: {W}, %b: {B}) -> {O} {{
  %o = intent.conv %x, %w, %b {{{attrs}}} : ({I}, {W}, {B}) -> {O}
  return %o : {O}
}}
"""
    return f"""
func.func @{name}(%x: {I}, %w: {W}) -> {O} {{
  %o = intent.conv %x, %w {{{attrs}}} : ({I}, {W}) -> {O}
  return %o : {O}
}}
"""


def _outshape(xs, ws, stride, pad, dil):
    R = len(xs) - 2
    out = [xs[0], ws[0]]
    for j in range(R):
        out.append((xs[2 + j] + 2 * pad[j] - dil[j] * (ws[2 + j] - 1) - 1)
                   // stride[j] + 1)
    return out


def _run(name, R, Cin, Cout, sp, kk, stride, pad, dil, groups, bias, T, seed):
    N = 2
    xs = [N, Cin] + sp
    ws = [Cout, Cin // groups] + kk
    ost = _outshape(xs, ws, stride, pad, dil)
    assert ost[-1] % T == 0, f"{name}: O_last {ost[-1]} not divisible by T {T}"
    bs = [Cout] if bias else None

    torch.manual_seed(seed)
    x = torch.randn(*xs, device="cuda").contiguous()
    w = torch.randn(*ws, device="cuda").contiguous()
    b = torch.randn(Cout, device="cuda").contiguous() if bias else None
    out = torch.empty(*ost, device="cuda")

    mlir = _mlir(name, xs, ws, bs, ost, (stride, pad, dil, groups))
    k = _kernel(name, T, mlir)
    args = [x, w] + ([b] if bias else []) + [out]
    k.thunk(args)()
    torch.cuda.synchronize()

    convf = {1: torch.nn.functional.conv1d, 2: torch.nn.functional.conv2d,
             3: torch.nn.functional.conv3d}[R]
    sarg = stride[0] if R == 1 else tuple(stride)
    parg = pad[0] if R == 1 else tuple(pad)
    darg = dil[0] if R == 1 else tuple(dil)
    ref = convf(x, w, b, stride=sarg, padding=parg, dilation=darg, groups=groups)
    rel = _rel(out, ref)
    assert rel < 1e-3, f"{name}: rel {rel} vs torch"

    fg = _flag_gems()
    if fg is not None:
        gf = {1: fg.conv1d, 2: fg.conv2d, 3: fg.conv3d}[R]
        yg = gf(x, w, b, stride=sarg, padding=parg, dilation=darg, groups=groups)
        torch.cuda.synchronize()
        assert _rel(out, yg) < 1e-3, f"{name}: rel vs flag_gems"
    print(f"\n{name} in{tuple(xs)} w{tuple(ws)} -> {tuple(ost)}: rel<1e-3 vs "
          f"torch" + ("" if fg is None else " + flag_gems"))


# ---------------------------------------------------------------------------- #
# conv2d, no pad, stride 1, groups 1, with bias.  (16-3+1 = 14, 14 % 7 == 0)
# ---------------------------------------------------------------------------- #

def test_conv2d():
    _run("conv2d", 2, 4, 8, [16, 16], [3, 3], [1, 1], [0, 0], [1, 1], 1,
         bias=True, T=7, seed=0)


# ---------------------------------------------------------------------------- #
# conv2d, pad 1 (same size 16 % 8), stride 1, no bias.
# ---------------------------------------------------------------------------- #

def test_conv2d_pad():
    _run("conv2d_pad", 2, 4, 8, [16, 16], [3, 3], [1, 1], [1, 1], [1, 1], 1,
         bias=False, T=8, seed=1)


# ---------------------------------------------------------------------------- #
# conv2d, stride 2 pad 1 -> out 8 (8 % 8), with bias.
# ---------------------------------------------------------------------------- #

def test_conv2d_stride2():
    _run("conv2d_s2", 2, 4, 8, [16, 16], [3, 3], [2, 2], [1, 1], [1, 1], 1,
         bias=True, T=8, seed=2)


# ---------------------------------------------------------------------------- #
# conv2d, groups > 1 (grouped conv), no pad, no bias.  (14 % 7)
# ---------------------------------------------------------------------------- #

def test_conv2d_groups():
    _run("conv2d_g2", 2, 4, 8, [16, 16], [3, 3], [1, 1], [0, 0], [1, 1], 2,
         bias=False, T=7, seed=3)


# ---------------------------------------------------------------------------- #
# conv1d (spatial rank 1), with bias.  (16-3+1 = 14, 14 % 7)
# ---------------------------------------------------------------------------- #

def test_conv1d():
    _run("conv1d", 1, 4, 8, [16], [3], [1], [0], [1], 1, bias=True, T=7, seed=4)


# ---------------------------------------------------------------------------- #
# conv3d (spatial rank 3), no bias.  (8-3+1 = 6, 6 % 6)
# ---------------------------------------------------------------------------- #

def test_conv3d():
    _run("conv3d", 3, 2, 4, [8, 8, 8], [3, 3, 3], [1, 1, 1], [0, 0, 0],
         [1, 1, 1], 1, bias=False, T=6, seed=5)
