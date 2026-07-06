"""PK the conv2d implicit-GEMM tensor-core lowering vs torch + real flag_gems.

Forward conv2d is lowered from atomic IntentIR (`intent.conv`, R=2, groups=1)
through the C++ ``-convert-intent-conv-to-gpu-mma`` pass to an IMPLICIT GEMM on
the sm_120a nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16) tensor cores, then
launched and compared head-to-head against (1) a torch reference
(``torch.nn.functional.conv2d``, cuDNN) and (2) the real ``flag_gems.conv2d``
triton kernel.

  Y[N,Cout,OH,OW] = im2col(X)[N*OH*OW, Cin*kH*kW] . W[Cout, Cin*kH*kW]^T

The im2col matrix is NEVER materialized: each (m-row, k-col) of the A shared
tile is mapped back to an input pixel by the SAME OOB-guarded conv address math
as the naive pass (``ih = oh*stride - pad + kh*dil``; out-of-bounds = zero pad)
emitted as the f16 smem tile-fill load predicate. Both operands are cast
f32 -> f16 during the fill (fp16 input mode): reduced precision, rel ~1e-2 vs
fp32 torch (EXPECTED -- the cuDNN reference itself runs TF32), verified at
rel<2e-2. stride/padding/dilation are fully general; groups==1.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_conv_mma_pk.py -q -s
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

# fp16 tensor-core input mode: reduced precision vs fp32 torch (cuDNN runs TF32).
REL_TOL = 2e-2


def _rel(a, b):
    a, b = a.float(), b.float()
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _tt(shape):
    return "tensor<" + "x".join(str(s) for s in shape) + "xf32>"


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


def _mlir(name, xs, ws, bs, ost, arr):
    stride, pad, dil = arr
    st = ", ".join(str(v) for v in stride)
    pd = ", ".join(str(v) for v in pad)
    dl = ", ".join(str(v) for v in dil)
    I, W, O = _tt(xs), _tt(ws), _tt(ost)
    attrs = (f'stride = [{st}], padding = [{pd}], dilation = [{dl}], '
             f'groups = 1 : i64')
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
    out = [xs[0], ws[0]]
    for j in range(2):
        out.append((xs[2 + j] + 2 * pad[j] - dil[j] * (ws[2 + j] - 1) - 1)
                   // stride[j] + 1)
    return out


def _run(name, Cin, Cout, sp, kk, stride, pad, dil, bias, seed):
    from intentir.benchmarks.row_reduction import ConvMMAKernel
    N = 2
    xs = [N, Cin] + sp
    ws = [Cout, Cin] + kk
    ost = _outshape(xs, ws, stride, pad, dil)
    bs = [Cout] if bias else None

    torch.manual_seed(seed)
    x = torch.randn(*xs, device="cuda").contiguous()
    w = torch.randn(*ws, device="cuda").contiguous()
    b = torch.randn(Cout, device="cuda").contiguous() if bias else None
    out = torch.empty(*ost, device="cuda")

    mlir = _mlir(name, xs, ws, bs, ost, (stride, pad, dil))
    k = ConvMMAKernel(mlir)
    args = [x, w] + ([b] if bias else []) + [out]
    k.thunk(args)()
    torch.cuda.synchronize()

    ref = torch.nn.functional.conv2d(x, w, b, stride=tuple(stride),
                                     padding=tuple(pad), dilation=tuple(dil))
    rel = _rel(out, ref)
    assert rel < REL_TOL, f"{name}: rel {rel} vs torch (fp16 tensor-core)"

    fg = _flag_gems()
    if fg is not None:
        yg = fg.conv2d(x, w, b, stride=tuple(stride), padding=tuple(pad),
                       dilation=tuple(dil), groups=1)
        torch.cuda.synchronize()
        assert _rel(out, yg) < REL_TOL, f"{name}: rel vs flag_gems"
    print(f"\n{name} in{tuple(xs)} w{tuple(ws)} -> {tuple(ost)}: rel={rel:.1e} "
          f"< {REL_TOL:g} vs torch" + ("" if fg is None else " + flag_gems"))


# conv2d, no pad, stride 1, with bias.
def test_conv2d():
    _run("conv2d", 4, 8, [16, 16], [3, 3], [1, 1], [0, 0], [1, 1],
         bias=True, seed=0)


# conv2d, pad 1 (same-size), no bias.
def test_conv2d_pad():
    _run("conv2d_pad", 4, 8, [16, 16], [3, 3], [1, 1], [1, 1], [1, 1],
         bias=False, seed=1)


# conv2d, stride 2, pad 1, with bias.
def test_conv2d_stride2():
    _run("conv2d_s2", 8, 16, [16, 16], [3, 3], [2, 2], [1, 1], [1, 1],
         bias=True, seed=2)


# conv2d, dilation 2, pad 2, no bias.
def test_conv2d_dilation2():
    _run("conv2d_d2", 4, 8, [16, 16], [3, 3], [1, 1], [2, 2], [2, 2],
         bias=False, seed=3)


# Cout not a multiple of the mma n8 tile (exercises the N-pad guarded store).
def test_conv2d_cout10():
    _run("conv2d_c10", 4, 10, [16, 16], [3, 3], [1, 1], [1, 1], [1, 1],
         bias=True, seed=4)


# 1x1 conv (K = Cin), pointwise.
def test_conv2d_1x1():
    _run("conv2d_1x1", 32, 16, [8, 8], [1, 1], [1, 1], [0, 0], [1, 1],
         bias=False, seed=5)


# The benched ResNet-ish shape (larger, K=576 divides BK=32).
def test_conv2d_bench_shape():
    _run("conv2d_big", 64, 64, [56, 56], [3, 3], [1, 1], [1, 1], [1, 1],
         bias=False, seed=6)
