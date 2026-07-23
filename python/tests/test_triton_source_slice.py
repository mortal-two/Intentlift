from __future__ import annotations

from pathlib import Path

import pytest

from intentir.verify.triton_source_slice import (
    TritonSourceSliceError,
    build_triton_source_slice,
    inline_imported_triton_helpers,
)
from intentir.verify.workload_manifest import build_workload_manifest


def test_slice_selects_transitive_jit_helper_and_strips_schedule_decorator() -> None:
    source = """import triton
import triton.language as tl

IGNORED = 99

@triton.jit
def helper(x):
    return x * x

@triton.autotune(configs=[], key=['N'])
@triton.heuristics({'EVEN': lambda args: True})
@triton.jit
def target(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = helper(tl.load(X + offsets))
    tl.store(Y + offsets, value)

def unrelated():
    return IGNORED
"""

    result = build_triton_source_slice(source, source_symbol="target")

    assert result.parameters == ("X", "Y", "N", "BLOCK")
    assert result.constexpr_parameters == ("BLOCK",)
    assert result.included_functions == ("helper", "target")
    assert result.removed_decorators == (
        "triton.autotune",
        "triton.heuristics",
    )
    assert "def unrelated" not in result.code
    assert "IGNORED" not in result.code
    assert "@triton.autotune" not in result.code
    assert "@triton.jit" in result.code


def test_slice_keeps_module_constant_and_conditional_import_closure() -> None:
    source = """import operator
import triton
import triton.language as tl
from package import enabled

if enabled(operator.ge):
    from triton.language.extra.libdevice import tanh
else:
    from triton.language.math import tanh

MODE: tl.constexpr = tl.constexpr(1)

@triton.jit
def target(X, Y, BLOCK: tl.constexpr):
    value = tanh(tl.load(X + tl.arange(0, BLOCK)))
    if MODE == 1:
        value += 1
    tl.store(Y + tl.arange(0, BLOCK), value)
"""

    result = build_triton_source_slice(source, source_symbol="target")

    assert "from package import enabled" in result.code
    assert "from triton.language.extra.libdevice import tanh" in result.code
    assert "MODE: tl.constexpr" in result.code
    assert "import operator" in result.code


def test_slice_abi_validation_fails_closed() -> None:
    source = """import triton
import triton.language as tl
@triton.jit
def target(X, BLOCK: tl.constexpr):
    pass
"""
    result = build_triton_source_slice(source, source_symbol="target")

    result.validate_abi(
        parameter_order=("X", "BLOCK"),
        constexpr_parameters=("BLOCK",),
    )
    with pytest.raises(TritonSourceSliceError, match="parameter order"):
        result.validate_abi(
            parameter_order=("BLOCK", "X"),
            constexpr_parameters=("BLOCK",),
        )


def test_slice_rejects_nested_symbol_and_unknown_decorator() -> None:
    nested = """import triton
class Holder:
    @triton.jit
    def target(X):
        pass
"""
    with pytest.raises(TritonSourceSliceError, match="module top level"):
        build_triton_source_slice(nested, source_symbol="target")

    decorated = """import triton
@custom_wrapper
@triton.jit
def target(X):
    pass
"""
    with pytest.raises(TritonSourceSliceError, match="unsupported"):
        build_triton_source_slice(decorated, source_symbol="target")


def test_imported_triton_helper_is_inlined_without_importing_package(
    tmp_path,
) -> None:
    helper = tmp_path / "fake_pkg" / "helpers.py"
    helper.parent.mkdir()
    helper.write_text(
        """import triton
import triton.language as tl
SCALE = 2.0
@triton.jit
def helper(x):
    return x * SCALE
""",
        encoding="utf-8",
    )
    source = """import triton
import triton.language as tl
from fake_pkg.helpers import helper
@triton.jit
def target(X, Y):
    tl.store(Y, helper(tl.load(X)))
"""
    selected = build_triton_source_slice(source, source_symbol="target")

    result = inline_imported_triton_helpers(
        selected, search_roots=(tmp_path,)
    )

    assert "from fake_pkg.helpers import helper" not in result.code
    assert "def helper" in result.code
    assert "SCALE = 2.0" in result.code
    assert result.inlined_imports == ("fake_pkg.helpers:helper",)


@pytest.mark.parametrize(
    "case_id",
    [
        "high_performance:liger_dyt",
        "high_performance:liger_swiglu",
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_preprocess_do_o_dot"
        ),
        "high_performance:liger_softmax",
        "high_performance:liger_rms_norm",
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
    ],
)
def test_six_real_representatives_form_a_fixed_c_slice(case_id: str) -> None:
    entry = build_workload_manifest().by_id[case_id]
    resolution = entry.resolve_source()
    assert resolution.path is not None

    result = build_triton_source_slice(
        Path(resolution.path).read_text(encoding="utf-8"),
        source_symbol=entry.symbol,
    )

    assert result.function == entry.symbol
    assert entry.symbol in result.included_functions
    assert "@triton.jit" in result.code
