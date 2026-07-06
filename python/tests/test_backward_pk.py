"""PK the BACKWARD / gradient cluster vs torch autograd AND the real flag_gems
``<op>_backward`` on the 5090D.

The biggest deferred class (07-06 P4). flag_gems has 24 ``*_backward`` entrypoints;
almost all are compositions of mechanisms IntentIR already lowers, so this is
authoring + verify (like the extraction bulk), NOT new passes. ``backend.backward``
authors each gradient as an intent-MLIR func that ONE existing pass lowers
unchanged; three mechanism groups:

  1. pointwise activation backward = pure map (grad_in = grad_out * act'(x)):
     sigmoid/tanh/silu/gelu/elu/threshold/dropout  -> elementwise pass.
  2. softmax / norm backward = row-reduction + map (the norm grad reduces over the
     normalized axis then maps): softmax/log_softmax/rms_norm/layer_norm (dx)
     -> row-reduction pass.
  3. scatter-grad backward = C3 scatter reduce=add (scatter into a zeroed buffer):
     gather/embedding/nll_loss  -> gather pass.

Reference gradient = torch autograd (build the forward with requires_grad,
.backward(grad_out), compare grad). ALSO PK'd head-to-head vs the real
flag_gems.<op>_backward where its signature is callable on this arch. rel < 1e-3.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_backward_pk.py -q -s
"""

from __future__ import annotations

import math
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
    b = b.float()
    n = b.norm()
    return float((a.float() - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _scalar(v):
    return torch.tensor(float(v), device="cuda", dtype=torch.float32)


def _kernel(k):
    from intentir.benchmarks.row_reduction import RowReductionKernel
    return RowReductionKernel(k.name, {}, k.block_threads,
                              pass_name=k.pass_name, mlir_text=k.mlir_text)


def _run(k, inputs, out_shape):
    """Compile+launch a non-scatter backward kernel; return the appended output."""
    out = torch.empty(out_shape, device="cuda", dtype=torch.float32).contiguous()
    _kernel(k).thunk([*inputs, out])()
    torch.cuda.synchronize()
    return out


def _run_scatter(k, data, index, src):
    """Compile+launch a scatter backward kernel; in-place on `data`, return it."""
    _kernel(k).thunk([data, index, src])()
    torch.cuda.synchronize()
    return data


def _flag_gems():
    try:
        import flag_gems
        return flag_gems
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Group 1 — pointwise activation backward = pure map.                          #
# --------------------------------------------------------------------------- #
_M, _N, _T = 256, 1024, 128


def _autograd_grad(fwd, x, go):
    x = x.clone().detach().requires_grad_(True)
    y = fwd(x)
    y.backward(go)
    return x.grad.detach(), y.detach()


def test_sigmoid_backward():
    from intentir.backend.backward import sigmoid_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, y = _autograd_grad(torch.sigmoid, x, go)
    out = _run(sigmoid_backward(_M, _N, _T), [go, y], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.sigmoid_backward(go, y)) if fg else None
    print(f"\nsigmoid_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_tanh_backward():
    from intentir.backend.backward import tanh_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, y = _autograd_grad(torch.tanh, x, go)
    out = _run(tanh_backward(_M, _N, _T), [go, y], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.tanh_backward(go, y)) if fg else None
    print(f"\ntanh_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_silu_backward():
    from intentir.backend.backward import silu_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, _ = _autograd_grad(torch.nn.functional.silu, x, go)
    out = _run(silu_backward(_M, _N, _T), [go, x, _scalar(1.0)], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.silu_backward(go, x)) if fg else None
    print(f"\nsilu_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_gelu_backward():
    from intentir.backend.backward import gelu_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, _ = _autograd_grad(lambda t: torch.nn.functional.gelu(t, approximate="none"),
                           x, go)
    consts = [_scalar(0.5), _scalar(1.0), _scalar(1.0 / math.sqrt(2.0)),
              _scalar(1.0 / math.sqrt(2.0 * math.pi)), _scalar(-0.5)]
    out = _run(gelu_backward(_M, _N, _T), [go, x, *consts], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.gelu_backward(go, x, approximate="none")) if fg else None
    print(f"\ngelu_backward (erf): rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_elu_backward():
    from intentir.backend.backward import elu_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, _ = _autograd_grad(torch.nn.functional.elu, x, go)
    out = _run(elu_backward(_M, _N, _T), [go, x, _scalar(0.0), _scalar(1.0)],
               (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = None
    if fg is not None:
        # torch/flag_gems signature: (grad, alpha, scale, input_scale, is_result, self)
        try:
            g = fg.elu_backward(go, 1.0, 1.0, 1.0, False, x)
            rg = _rel(out, g)
        except Exception as e:  # noqa: BLE001 - arch/signature incompat
            rg = None
            print(f"  (flag_gems.elu_backward skipped: {type(e).__name__})")
    print(f"\nelu_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_threshold_backward():
    from intentir.backend.backward import threshold_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    thr = 0.0
    gx, _ = _autograd_grad(
        lambda t: torch.nn.functional.threshold(t, thr, 0.0), x, go)
    out = _run(threshold_backward(_M, _N, _T),
               [go, x, _scalar(thr), _scalar(0.0)], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.threshold_backward(go, x, thr)) if fg else None
    print(f"\nthreshold_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_dropout_backward():
    from intentir.backend.backward import dropout_backward
    torch.manual_seed(0)
    go = torch.randn(_M, _N, device="cuda").contiguous()
    p = 0.3
    scale = 1.0 / (1.0 - p)
    # A 0/1 keep-mask (the C5 philox mask semantic): dropped -> 0, kept -> 1.
    mask = (torch.rand(_M, _N, device="cuda") > p).float().contiguous()
    out = _run(dropout_backward(_M, _N, _T), [go, mask, _scalar(scale)], (_M, _N))
    ref = go * mask * scale
    rt = _rel(out, ref)
    fg = _flag_gems()
    rg = None
    if fg is not None:
        try:
            rg = _rel(out, fg.dropout_backward(go, mask.bool(), scale))
        except Exception as e:  # noqa: BLE001
            print(f"  (flag_gems.dropout_backward skipped: {type(e).__name__})")
    print(f"\ndropout_backward p={p}: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


# --------------------------------------------------------------------------- #
# Group 2 — softmax / norm backward = row-reduction + map.                     #
# --------------------------------------------------------------------------- #
def test_softmax_backward():
    from intentir.backend.backward import softmax_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, y = _autograd_grad(lambda t: torch.softmax(t, 1), x, go)
    out = _run(softmax_backward(_M, _N, _T), [go, y], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.softmax_backward(go, y, 1, torch.float32)) if fg else None
    print(f"\nsoftmax_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_log_softmax_backward():
    from intentir.backend.backward import log_softmax_backward
    torch.manual_seed(0)
    x = torch.randn(_M, _N, device="cuda").contiguous()
    go = torch.randn(_M, _N, device="cuda").contiguous()
    gx, y = _autograd_grad(lambda t: torch.log_softmax(t, 1), x, go)
    out = _run(log_softmax_backward(_M, _N, _T), [go, y], (_M, _N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = _rel(out, fg.log_softmax_backward(go, y, 1, torch.float32)) if fg else None
    print(f"\nlog_softmax_backward: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_rms_norm_backward():
    from intentir.backend.backward import rms_norm_backward
    torch.manual_seed(0)
    M, N, T = 256, 1024, 128
    eps = 1e-5
    x = torch.randn(M, N, device="cuda", requires_grad=True)
    w = torch.randn(N, device="cuda", requires_grad=True)
    r = torch.rsqrt((x * x).mean(1, keepdim=True) + eps)
    y = x * r * w
    go = torch.randn(M, N, device="cuda")
    y.backward(go)
    gx = x.grad.detach()
    out = _run(rms_norm_backward(M, N, T),
               [go.contiguous(), x.detach().contiguous(), w.detach().contiguous(),
                _scalar(eps), _scalar(1.0 / N)], (M, N))
    rt = _rel(out, gx)
    # flag_gems.rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps)
    fg = _flag_gems()
    rg = None
    if fg is not None:
        try:
            g = fg.rms_norm_backward(go.contiguous(), x.detach().contiguous(),
                                     r.detach().contiguous(), [N],
                                     w.detach().contiguous(), eps)
            g = g[0] if isinstance(g, (tuple, list)) else g
            rg = _rel(out, g)
        except Exception as e:  # noqa: BLE001
            print(f"  (flag_gems.rms_norm_backward skipped: {type(e).__name__})")
    print(f"\nrms_norm_backward (dx): rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_layer_norm_backward():
    from intentir.backend.backward import layer_norm_backward
    torch.manual_seed(0)
    M, N, T = 256, 1024, 128
    eps = 1e-5
    x = torch.randn(M, N, device="cuda", requires_grad=True)
    w = torch.randn(N, device="cuda", requires_grad=True)
    b = torch.randn(N, device="cuda", requires_grad=True)
    y = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
    go = torch.randn(M, N, device="cuda")
    y.backward(go)
    gx = x.grad.detach()
    out = _run(layer_norm_backward(M, N, T),
               [go.contiguous(), x.detach().contiguous(), w.detach().contiguous(),
                _scalar(eps), _scalar(1.0 / N)], (M, N))
    rt = _rel(out, gx)
    fg = _flag_gems()
    rg = None
    if fg is not None:
        try:
            mean = x.detach().mean(1, keepdim=True)
            rstd = torch.rsqrt(x.detach().var(1, unbiased=False, keepdim=True) + eps)
            g = fg.layer_norm_backward(go.contiguous(), x.detach().contiguous(),
                                       [N], mean, rstd,
                                       weight=w.detach().contiguous())
            g = g[0] if isinstance(g, (tuple, list)) else g
            rg = _rel(out, g)
        except Exception as e:  # noqa: BLE001
            print(f"  (flag_gems.layer_norm_backward skipped: {type(e).__name__})")
    print(f"\nlayer_norm_backward (dx): rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else ""))
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


# --------------------------------------------------------------------------- #
# Group 3 — scatter-grad backward = C3 scatter reduce=add.                     #
# --------------------------------------------------------------------------- #
def test_gather_backward():
    from intentir.backend.backward import gather_backward
    torch.manual_seed(0)
    M, N, T = 256, 1024, 128
    data = torch.randn(M, N, device="cuda", requires_grad=True)
    index = torch.randint(0, N, (M, N), device="cuda", dtype=torch.int64).contiguous()
    out = torch.gather(data, 1, index)
    go = torch.randn(M, N, device="cuda")
    out.backward(go)
    gx = data.grad.detach()
    gself = torch.zeros(M, N, device="cuda").contiguous()
    got = _run_scatter(gather_backward(M, N, T), gself, index, go.contiguous())
    rt = _rel(got, gx)
    print(f"\ngather_backward: rel_torch {rt:.2e} (scatter-add reduce=add)")
    assert rt < 1e-3


def test_embedding_backward():
    from intentir.backend.backward import embedding_backward
    torch.manual_seed(0)
    V, D, Nidx, T = 512, 128, 256, 64
    w = torch.randn(V, D, device="cuda", requires_grad=True)
    idx = torch.randint(0, V, (Nidx,), device="cuda", dtype=torch.int64)
    out = torch.nn.functional.embedding(idx, w)
    go = torch.randn(Nidx, D, device="cuda")
    out.backward(go)
    gw = w.grad.detach()
    # index broadcast across the D feature columns -> [Nidx, D] i64.
    idx2d = idx.view(-1, 1).expand(Nidx, D).contiguous()
    gwz = torch.zeros(V, D, device="cuda").contiguous()
    got = _run_scatter(embedding_backward(V, D, Nidx, T), gwz, idx2d,
                       go.contiguous())
    rt = _rel(got, gw)
    fg = _flag_gems()
    rg = None
    if fg is not None:
        try:
            g = fg.embedding_backward(go.contiguous(), idx, V)
            rg = _rel(got, g)
        except Exception as e:  # noqa: BLE001
            print(f"  (flag_gems.embedding_backward skipped: {type(e).__name__})")
    print(f"\nembedding_backward V={V} D={D} N={Nidx}: rel_torch {rt:.2e}"
          + (f" rel_gems {rg:.2e}" if rg is not None else "")
          + " (scatter-add axis=0)")
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


def test_nll_loss_backward():
    from intentir.backend.backward import nll_loss_backward
    torch.manual_seed(0)
    M, C = 256, 64
    x = torch.randn(M, C, device="cuda", requires_grad=True)
    target = torch.randint(0, C, (M,), device="cuda")
    loss = torch.nn.functional.nll_loss(x, target, reduction="mean")
    loss.backward()
    gx = x.grad.detach()
    gi = torch.zeros(M, C, device="cuda").contiguous()
    tgt2d = target.view(-1, 1).to(torch.int64).contiguous()
    src = torch.full((M, 1), -1.0 / M, device="cuda", dtype=torch.float32).contiguous()
    got = _run_scatter(nll_loss_backward(M, C), gi, tgt2d, src)
    rt = _rel(got, gx)
    print(f"\nnll_loss_backward (mean): rel_torch {rt:.2e} (scatter -1/M at target)")
    assert rt < 1e-3
