from __future__ import annotations

import pytest

from intentir.verify.gate1._triton_worker import _normalize_deprecated_triton_dot
from intentir.verify.gate1.codegen import UnsafeReplayCodeError, validate_triton_module


def _module(call: str) -> str:
    return f"""import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def scrc_k0(p0, p1, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = {call}
    tl.store(p1 + v0, v1)
"""


def test_global_frontend_policy_allows_reviewed_public_math() -> None:
    validate_triton_module(
        _module("libdevice.tanh(tl.load(p0 + v0))"),
        required_function="scrc_k0",
    )


def test_global_frontend_policy_is_not_expandable_per_kernel() -> None:
    with pytest.raises(UnsafeReplayCodeError, match="forbidden"):
        validate_triton_module(
            _module("libdevice.private_canary(tl.load(p0 + v0))"),
            required_function="scrc_k0",
        )


def test_deprecated_static_dot_transpose_has_one_global_normalization() -> None:
    source = """import triton.language as tl
def kernel(a, b):
    return tl.dot(a, b, trans_a=False, trans_b=True)
"""
    normalized, evidence = _normalize_deprecated_triton_dot(source)
    assert "tl.dot(a, tl.trans(b))" in normalized
    assert "trans_b" not in normalized
    assert evidence["protocol"] == "triton_static_dot_transpose_v1"
    assert evidence["rewrite_count"] == 1
    assert len(evidence["normalized_sha256"]) == 64


def test_deprecated_dynamic_dot_transpose_fails_closed() -> None:
    source = """import triton.language as tl
def kernel(a, b, flag):
    return tl.dot(a, b, trans_b=flag)
"""
    with pytest.raises(ValueError, match="static boolean"):
        _normalize_deprecated_triton_dot(source)
