"""Exact one-column-block FlashAttention backward lowering slice.

The counted source symbol fixes ``start_n=0`` for the captured one-key-block
case (Sq=128, Sk=64, D=64, ATOMIC_ADD=false, no bias, non-causal).  Canonical
IntentIR removes that constant program coordinate and represents the same five
contractions as the production attention-backward graph.  This test executes
both the generated cubin and the raw Dataset Triton symbol on identical inputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import tempfile

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT
from test_flash_bwd_pk import _FlashBwdKernel, _bwd_intent, _rel


torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

pytestmark = [
    pytest.mark.skipif(
        not pathlib.Path(INTENT_OPT).exists(), reason="intent-opt not built"
    ),
    pytest.mark.skipif(
        not pathlib.Path(MLIR_OPT).exists(), reason="mlir-opt not present"
    ),
    pytest.mark.skipif(
        not pathlib.Path(CUDA_BIN).exists(), reason="CUDA toolkit not present"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]

_FLASH_TRITON = pathlib.Path(
    "/home/kingdom/phdworks/Dataset/flash-attention/"
    "flash_attn/flash_attn_triton.py"
)


def _load_one_col_kernel():
    if not _FLASH_TRITON.exists():
        pytest.skip("flash-attention source not present")
    source = _FLASH_TRITON.read_text(encoding="utf-8")
    patches = [
        ("qk += tl.dot(q, k, trans_b=True)", "qk += tl.dot(q, tl.trans(k))"),
        ("qk = tl.dot(q, k, trans_b=True)", "qk = tl.dot(q, tl.trans(k))"),
        (
            "dv += tl.dot(p.to(do.dtype), do, trans_a=True)",
            "dv += tl.dot(tl.trans(p.to(do.dtype)), do)",
        ),
        ("dp = tl.dot(do, v, trans_b=True)", "dp = tl.dot(do, tl.trans(v))"),
        ("dk += tl.dot(ds, q, trans_a=True)", "dk += tl.dot(tl.trans(ds), q)"),
    ]
    for old, new in patches:
        source = source.replace(old, new)
    tag = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    local = pathlib.Path(tempfile.gettempdir()) / f"fat_one_col_{tag}.py"
    if not local.exists():
        local.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"fat_one_col_{tag}", local)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bwd_kernel_one_col_block


def _reference_inputs(seed: int = 29):
    sq, sk, d = 128, 64, 64
    torch.manual_seed(seed)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float16).contiguous()
    do = torch.randn(sq, d, device="cuda", dtype=torch.float16).contiguous()
    scale = 1.0 / (d ** 0.5)
    with torch.no_grad():
        scores = (q.float() @ k.float().t()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output = probabilities @ v.float()
        lse = torch.logsumexp(scores, dim=-1).contiguous()
        delta = (output * do.float()).sum(dim=-1).contiguous()
    return q, k, v, do, lse, delta, scale


def _run_source(q, k, v, do, lse, delta, scale):
    kernel = _load_one_col_kernel()
    sq, d = q.shape
    sk = k.shape[0]
    dq = torch.zeros(sq, d, device="cuda", dtype=torch.float32)
    dk = torch.zeros(sk, d, device="cuda", dtype=torch.float32)
    dv = torch.zeros(sk, d, device="cuda", dtype=torch.float32)
    kernel[(1,)](
        0,
        q,
        k,
        v,
        None,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        float(scale),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        0,
        do.stride(0),
        dq.stride(0),
        dk.stride(0),
        dv.stride(0),
        sq,
        sk,
        d,
        ATOMIC_ADD=False,
        BIAS_TYPE="none",
        IS_CAUSAL=False,
        BLOCK_HEADDIM=64,
        EVEN_M=True,
        EVEN_N=True,
        EVEN_HEADDIM=True,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()
    return dq, dk, dv


def _run_generated(q, k, v, do, lse, delta, scale):
    kernel = _FlashBwdKernel(_bwd_intent(128, 64, 64))
    assert kernel.grid == (1, 1, 1)
    dq = torch.zeros(128, 64, device="cuda", dtype=torch.float32).contiguous()
    dk = torch.empty(64, 64, device="cuda", dtype=torch.float32).contiguous()
    dv = torch.empty(64, 64, device="cuda", dtype=torch.float32).contiguous()
    kernel.run(q, k, v, do, lse[:, None], delta[:, None], scale, dq, dk, dv)
    torch.cuda.synchronize()
    return dq, dk, dv


def test_flash_bwd_one_col_exact_symbol_lowering() -> None:
    q, k, v, do, lse, delta, scale = _reference_inputs()
    generated = _run_generated(q, k, v, do, lse, delta, scale)
    source = _run_source(q, k, v, do, lse, delta, scale)

    errors = tuple(_rel(left, right) for left, right in zip(generated, source))
    assert errors[0] < 2.0e-2, f"dQ rel={errors[0]:.3e}"
    assert errors[1] < 2.0e-2, f"dK rel={errors[1]:.3e}"
    assert errors[2] < 2.0e-2, f"dV rel={errors[2]:.3e}"
