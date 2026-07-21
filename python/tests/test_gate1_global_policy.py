from __future__ import annotations

import pytest

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
