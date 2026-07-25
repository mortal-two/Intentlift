from __future__ import annotations

import pytest

from intentir.verify.gate1._triton_worker import _normalize_deprecated_triton_dot
from intentir.verify.gate1.codegen import (
    UnsafeReplayCodeError,
    validate_triton_artifact_uses,
    validate_triton_module,
)


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


@pytest.mark.parametrize(
    ("expression", "name"),
    [
        ("tl.tanh(tl.load(p0 + v0))", "tanh"),
        ("tl.tan(tl.load(p0 + v0))", "tan"),
        ("tl.atan(tl.load(p0 + v0))", "atan"),
        ("tl.atan2(tl.load(p0 + v0), tl.load(p0 + v0))", "atan2"),
        ("tl.pow(tl.load(p0 + v0), 2.0)", "pow"),
        ("tl.isnan(tl.load(p0 + v0))", "isnan"),
    ],
)
def test_unavailable_tl_math_is_rejected_before_triton_compile(
    expression: str,
    name: str,
) -> None:
    with pytest.raises(UnsafeReplayCodeError, match=rf"tl\.{name}.*unavailable"):
        validate_triton_module(
            _module(expression),
            required_function="scrc_k0",
        )


@pytest.mark.parametrize(
    "expression",
    [
        "libdevice.tanh(tl.load(p0 + v0))",
        "libdevice.tan(tl.load(p0 + v0))",
        "libdevice.atan(tl.load(p0 + v0))",
        "libdevice.atan2(tl.load(p0 + v0), tl.load(p0 + v0))",
        "tl.load(p0 + v0) ** 2.0",
        "tl.load(p0 + v0) != tl.load(p0 + v0)",
    ],
)
def test_pinned_triton_math_has_a_reviewed_public_spelling(
    expression: str,
) -> None:
    validate_triton_module(_module(expression), required_function="scrc_k0")


def test_global_frontend_policy_is_not_expandable_per_kernel() -> None:
    with pytest.raises(UnsafeReplayCodeError, match="forbidden"):
        validate_triton_module(
            _module("libdevice.private_canary(tl.load(p0 + v0))"),
            required_function="scrc_k0",
        )


def _artifact_use_module(expression: str | None) -> str:
    use = f"    v1 = {expression}\n" if expression is not None else ""
    return f"""import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
{use}    v2 = tl.load(p0 + v0)
    tl.store(p1 + v0, v2)
"""


def test_artifact_use_checker_accepts_an_unused_null_carrier() -> None:
    validate_triton_artifact_uses(
        _artifact_use_module(None),
        required_function="scrc_k0",
        forbidden_body_parameters=frozenset({"p2"}),
        unrefined_cast_parameters=frozenset({"p2"}),
    )


def test_artifact_use_checker_rejects_null_carrier_without_pi_route() -> None:
    with pytest.raises(UnsafeReplayCodeError, match="no routed pi realization"):
        validate_triton_artifact_uses(
            _artifact_use_module("p2 + v0"),
            required_function="scrc_k0",
            forbidden_body_parameters=frozenset({"p2"}),
        )


def test_artifact_use_checker_rejects_unrefined_direct_parameter_cast() -> None:
    with pytest.raises(UnsafeReplayCodeError, match="must not guess"):
        validate_triton_artifact_uses(
            _artifact_use_module("p2.to(tl.int64)"),
            required_function="scrc_k0",
            unrefined_cast_parameters=frozenset({"p2"}),
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
