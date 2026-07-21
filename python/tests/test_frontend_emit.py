"""Front-end: JSON -> intent-dialect MLIR -> (lower / execute).

These tests exercise the missing 前端 leg — :mod:`intentir.frontend.emit_mlir`
turning ``flashattention_fwd.intent.json`` into real `intent`-dialect text:

* ``test_emit_loop_module_lowers`` — the structured ``intent.loop`` module parses,
  verifies, and lowers to ``scf.for`` via ``-convert-intent-to-linalg`` (no GPU).
* ``test_emit_body_lowers`` — the executable body function lowers cleanly (no GPU).
* ``test_emitted_body_executes_vs_interpret`` — THE end-to-end front+back proof:
  the EMITTED body runs on the 5090D (host-driven trip loop, carries in CUDA
  buffers) and matches ``interpret(JSON)`` — i.e. JSON-sourced intent IR executes
  the FlashAttention recurrence on hardware, certified against our own spec.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT
from intentir.ir.model import IntentFunction

_TEST = pathlib.Path(__file__).resolve().parents[2] / "test"
_JSON = _TEST / "flashattention_fwd.intent.json"
_LIVE_JSON = (
    pathlib.Path(__file__).resolve().parents[2]
    / "artifacts/paper_pipeline/hp-hard-first-20260717"
    / "high_performance_flash_attn.flash_attn_triton._fwd_kernel/P1/intent.json"
)
_DIMS = {"B": 1, "H": 1, "Sq": 4, "Sk": 8, "D": 4, "BK": 2}


def _fn() -> IntentFunction:
    return IntentFunction.from_json(json.loads(_JSON.read_text()))


def _fn_with_f32_accumulators() -> IntentFunction:
    """The canonical fixture declares contraction precision in op.type."""

    return _fn()


def _flat_fn(name: str) -> IntentFunction:
    return IntentFunction.from_json({
        "intent_function": {
            "name": name,
            "inputs": [{"name": "X", "dtype": "f32", "shape": [4]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
            "body": [{
                "op": "map_unary",
                "result": "Y",
                "operands": ["X"],
                "attrs": {"fn": "exp"},
            }],
        }
    })


def _broadcast_binary_fn(
    name: str,
    operand_shape: list[int],
    output_shape: list[int],
) -> IntentFunction:
    return IntentFunction.from_json({
        "intent_function": {
            "name": name,
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": output_shape},
                {"name": "row", "dtype": "f32", "shape": operand_shape},
            ],
            "outputs": [
                {"name": "Y", "dtype": "f32", "shape": output_shape}
            ],
            "body": [{
                "op": "map_binary",
                "result": "Y",
                "operands": ["X", "row"],
                "attrs": {"fn": "mul"},
                "type": {"dtype": "f32", "shape": output_shape},
            }],
        }
    })


def _intent_opt(mlir: str, *passes: str) -> str:
    p = subprocess.run([INTENT_OPT, *passes], input=mlir, capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


_needs_opt = pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                                reason="intent-opt not built")


@pytest.mark.parametrize(("raw_name", "symbol"), [
    ("already_valid_name_1", "already_valid_name_1"),
    ("mlir.valid$but.ptx.unsafe", "mlir_valid_but_ptx_unsafe"),
    ("high_performance:flash_attn/kernel", "high_performance_flash_attn_kernel"),
    ("9attention", "_9attention"),
    ("λ attention", "__attention"),
    ("", "_"),
])
def test_emit_module_sanitizes_function_symbol(
    raw_name: str, symbol: str
) -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(_flat_fn(raw_name), {})
    assert f"func.func @{symbol}(" in mlir
    assert mlir == emit_module(_flat_fn(raw_name), {})  # stable across emissions


def test_emit_broadcast_keeps_existing_leading_singleton_axis() -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(
        _broadcast_binary_fn("existing_leading_axis", [1, 8], [4, 8]),
        {},
        broadcast_axes={"Y": {"row": (0,)}},
    )

    assert "tensor.expand_shape %row" not in mlir
    assert "tensor<1x8xf32>) -> tensor<4x8xf32>" in mlir


def test_emit_keeps_standard_trailing_vector_broadcast_rank_one() -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(
        _broadcast_binary_fn("trailing_vector", [8], [4, 8]),
        {},
        broadcast_axes={"Y": {"row": (0,)}},
    )

    assert "tensor.expand_shape %row" not in mlir
    assert "tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>" in mlir


def test_emit_nontrailing_mul_broadcast_canonicalizes_to_contraction() -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(
        _broadcast_binary_fn("row_scale", [4], [4, 8]),
        {},
        broadcast_axes={"Y": {"row": (1,)}},
    )

    assert "tensor.expand_shape %row" not in mlir
    assert (
        'intent.contraction %X, %row {subscripts = "ab,a->ab"} : '
        "(tensor<4x8xf32>, tensor<4xf32>) -> tensor<4x8xf32>"
    ) in mlir


def test_emit_broadcast_supports_mixed_existing_and_missing_axes() -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(
        _broadcast_binary_fn("mixed_broadcast_axes", [4, 1], [3, 4, 8]),
        {},
        broadcast_axes={"Y": {"row": (0, 2)}},
    )

    assert "tensor.expand_shape %row" in mlir
    assert "output_shape [1, 4, 1]" in mlir
    assert "tensor<1x4x1xf32>) -> tensor<3x4x8xf32>" in mlir


def test_emit_scalar_broadcast_uses_empty_affine_map_without_rank_insertion() -> None:
    from intentir.frontend import emit_module

    mlir = emit_module(
        _broadcast_binary_fn("scalar_broadcast", [], [4, 1]),
        {},
        broadcast_axes={"Y": {"row": (0,)}},
    )

    assert "tensor.expand_shape %row" not in mlir
    assert "tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>" in mlir


def test_emit_propagates_collapsed_output_shape_through_rms_chain() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "rms_chain_shape",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [4, 8]},
                    {"name": "eps", "dtype": "f32", "shape": []},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 8]},
                    {"name": "R", "dtype": "f32", "shape": [4]},
                ],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "square",
                        "operands": ["X", "X"],
                        "attrs": {"fn": "mul"},
                    },
                    {
                        "op": "reduce",
                        "result": "mean",
                        "operands": ["square"],
                        "attrs": {
                            "fn": "mean",
                            "axes": [1],
                            "assoc": "float_nondeterministic",
                        },
                    },
                    {
                        "op": "map_binary",
                        "result": "shifted",
                        "operands": ["mean", "eps"],
                        "attrs": {"fn": "add"},
                    },
                    {
                        "op": "map_unary",
                        "result": "root",
                        "operands": ["shifted"],
                        "attrs": {"fn": "sqrt"},
                    },
                    {
                        "op": "map_unary",
                        "result": "R",
                        "operands": ["root"],
                        "attrs": {"fn": "recip"},
                    },
                    {
                        "op": "contraction",
                        "result": "Y",
                        "operands": ["X", "R"],
                        "attrs": {"subscripts": "mn,m->mn"},
                    },
                ],
            }
        }
    )

    mlir = emit_module(
        fn,
        {},
        broadcast_axes={"shifted": {"eps": (0,)}},
    )

    assert "intent.reduce" in mlir
    assert "-> tensor<4xf32>" in mlir
    assert (
        '{subscripts = "mn,m->mn"} : '
        "(tensor<4x8xf32>, tensor<4xf32>) -> tensor<4x8xf32>"
    ) in mlir


def test_emit_select_keeps_existing_condition_broadcast_axis() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "select_existing_broadcast_axis",
            "inputs": [
                {"name": "cond", "dtype": "i1", "shape": [4, 1]},
                {"name": "X", "dtype": "f32", "shape": [4, 8]},
                {"name": "Y", "dtype": "f32", "shape": [4, 8]},
            ],
            "outputs": [
                {"name": "out", "dtype": "f32", "shape": [4, 8]}
            ],
            "body": [{
                "op": "select",
                "result": "out",
                "operands": ["cond", "X", "Y"],
                "attrs": {},
                "type": {"dtype": "f32", "shape": [4, 8]},
            }],
        }
    })
    mlir = emit_module(
        fn,
        {},
        broadcast_axes={"out": {"cond": (1,)}},
    )

    assert "tensor.expand_shape %cond" not in mlir
    assert "tensor<4x1xi1>, tensor<4x8xf32>, tensor<4x8xf32>" in mlir


def test_per_op_layer_b_is_rejected_instead_of_affecting_lowering() -> None:
    with pytest.raises(ValueError, match="layer_b"):
        IntentFunction.from_json({
            "intent_function": {
                "name": "forbidden_second_b_channel",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [4]}],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
                "body": [{
                    "op": "cast",
                    "result": "Y",
                    "operands": ["X"],
                    "attrs": {"to_dtype": "f32"},
                    "layer_b": {"invented_proof": True},
                }],
            }
        })


@_needs_opt
def test_bare_gather_emitter_never_invents_bounds_proof() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "unproven_gather",
            "inputs": [
                {"name": "W", "dtype": "f32", "shape": [3]},
                {"name": "I", "dtype": "i32", "shape": []},
            ],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
            "body": [{
                "op": "gather",
                "result": "Y",
                "operands": ["W", "I"],
                "attrs": {"axis": 0, "mode": "take"},
            }],
        }
    })

    mlir = emit_module(fn, {})
    assert "intent.index_bounds_proven" not in mlir
    assert "intent_layer_b_json" not in mlir
    assert "intent.inplace" not in mlir
    _intent_opt(mlir)


@_needs_opt
def test_all_emitters_sanitize_live_llm_function_name() -> None:
    from intentir.frontend import emit_body_function, emit_loop_module, emit_module

    raw_name = "high_performance:flash_attn.flash_attn_triton._fwd_kernel"
    symbol = "high_performance_flash_attn_flash_attn_triton__fwd_kernel"
    loop_fn = _fn()
    loop_fn.name = raw_name
    modules = [
        emit_module(_flat_fn(raw_name), {}),
        emit_loop_module(
            loop_fn, _DIMS, func_name=raw_name, schedule_tile=_DIMS["BK"]
        ),
        emit_body_function(
            loop_fn, _DIMS, func_name=raw_name, schedule_tile=_DIMS["BK"]
        ),
    ]

    for mlir in modules:
        assert f"func.func @{symbol}(" in mlir
        _intent_opt(mlir)


@_needs_opt
def test_emit_loop_module_lowers() -> None:
    from intentir.frontend import emit_loop_module

    mlir = emit_loop_module(
        _fn_with_f32_accumulators(), _DIMS, schedule_tile=_DIMS["BK"]
    )
    assert "intent.loop" in mlir and "intent.yield" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "scf.for" in lowered
    assert "tensor.extract_slice" in lowered
    assert "intent." not in lowered  # everything lowered to standard MLIR


@_needs_opt
def test_emit_body_lowers() -> None:
    from intentir.frontend import emit_body_function

    mlir = emit_body_function(
        _fn_with_f32_accumulators(), _DIMS, schedule_tile=_DIMS["BK"]
    )
    # body args: outer (Q, scale) + scan tiles + explicit loop state arguments.
    for name in (
        "%Q", "%scale", "%Kj", "%Vj", "%m_iter", "%l_iter", "%Oacc_iter",
    ):
        assert f"{name}:" in mlir, name
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "linalg.generic" in lowered
    assert "intent." not in lowered


@_needs_opt
def test_semantic_loop_uses_external_schedule_tile_and_typed_accumulators() -> None:
    """The MLIR tile comes from the explicit C projection, never from J."""
    from intentir.frontend import emit_body_function

    mlir = emit_body_function(_fn(), _DIMS, schedule_tile=_DIMS["BK"])
    assert "%Q: tensor<4x4xf16>" in mlir
    assert "%Kj: tensor<2x4xf16>" in mlir
    assert (
        "(tensor<4x4xf16>, tensor<2x4xf16>) -> tensor<4x2xf32>"
        in mlir
    )
    assert (
        "(tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>"
        in mlir
    )
    _intent_opt(mlir)  # real C++ parser/verifier; no lowering claim here


@_needs_opt
def test_flat_emitter_preserves_float_ssa_dtypes_and_predicates() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "dtype_graph",
            "inputs": [
                {"name": "A", "dtype": "f16", "shape": [2, 4]},
                {"name": "B", "dtype": "f16", "shape": [3, 4]},
                {"name": "X", "dtype": "bf16", "shape": [2]},
                {"name": "D", "dtype": "f64", "shape": [2]},
            ],
            "outputs": [
                {"name": "R", "dtype": "f32", "shape": [2]},
                {"name": "HT", "dtype": "f16", "shape": [3, 2]},
                {"name": "P", "dtype": "bf16", "shape": [2]},
                {"name": "Dn", "dtype": "f64", "shape": [2]},
                {"name": "pred", "dtype": "i1", "shape": [2]},
                {"name": "idx", "dtype": "i64", "shape": [2]},
            ],
            "body": [
                {
                    "op": "contraction",
                    "result": "C",
                    "operands": ["A", "B"],
                    "attrs": {"subscripts": "mk,nk->mn"},
                    "type": {"dtype": "f32"},
                },
                {
                    "op": "map_unary",
                    "result": "E",
                    "operands": ["C"],
                    "attrs": {"fn": "exp"},
                },
                {
                    "op": "reduce",
                    "result": "R",
                    "operands": ["E"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [1],
                        "assoc": "float_nondeterministic",
                        "keepdims": False,
                    },
                },
                {
                    "op": "cast",
                    "result": "H",
                    "operands": ["C"],
                    "attrs": {"to_dtype": "f16"},
                },
                {
                    "op": "transpose",
                    "result": "HT",
                    "operands": ["H"],
                    "attrs": {"perm": [1, 0]},
                },
                {
                    "op": "map_unary",
                    "result": "P",
                    "operands": ["X"],
                    "attrs": {"fn": "abs"},
                },
                {
                    "op": "map_unary",
                    "result": "Dn",
                    "operands": ["D"],
                    "attrs": {"fn": "neg"},
                },
                {
                    "op": "map_binary",
                    "result": "pred",
                    "operands": ["P", "P"],
                    "attrs": {"fn": "gt"},
                },
                {
                    "op": "iota",
                    "result": "idx",
                    "attrs": {"n": 2, "dtype": "i64"},
                },
            ],
        }
    })

    mlir = emit_module(fn, {})
    assert "%A: tensor<2x4xf16>" in mlir
    assert "%X: tensor<2xbf16>" in mlir
    assert "%D: tensor<2xf64>" in mlir
    assert "-> tensor<2x3xf32>" in mlir       # contraction accumulator
    assert 'to_dtype = "f16"' in mlir
    assert "-> tensor<3x2xf16>" in mlir       # transpose propagation
    assert "-> tensor<2xbf16>" in mlir        # unary propagation
    assert "-> tensor<2xf64>" in mlir
    assert "-> tensor<2xi1>" in mlir          # comparison predicate
    assert "intent.iota : tensor<2xi64>" in mlir
    _intent_opt(mlir)


def test_contraction_without_explicit_result_type_keeps_operand_dtype() -> None:
    from intentir.frontend import emit_module

    contraction = {
        "op": "contraction",
        "result": "C",
        "operands": ["A", "B"],
        "attrs": {"subscripts": "mk,nk->mn"},
    }
    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "same_type_dot",
            "inputs": [
                {"name": "A", "dtype": "f16", "shape": [2, 4]},
                {"name": "B", "dtype": "f16", "shape": [3, 4]},
            ],
            "outputs": [{"name": "C", "dtype": "f16", "shape": [2, 3]}],
            "body": [contraction],
        }
    })

    mlir = emit_module(fn, {})
    assert (
        "(tensor<2x4xf16>, tensor<3x4xf16>) -> tensor<2x3xf16>"
        in mlir
    )


def test_contraction_rejects_invalid_explicit_result_dtype() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "invalid_dot_type",
            "inputs": [
                {"name": "A", "dtype": "f16", "shape": [2, 4]},
                {"name": "B", "dtype": "f16", "shape": [3, 4]},
            ],
            "outputs": [{"name": "C", "dtype": "f16", "shape": [2, 3]}],
            "body": [{
                "op": "contraction",
                "result": "C",
                "operands": ["A", "B"],
                "attrs": {"subscripts": "mk,nk->mn"},
                "type": {"dtype": "not_a_dtype"},
            }],
        }
    })
    with pytest.raises(ValueError, match="invalid dtype"):
        emit_module(fn, {})


@pytest.mark.skipif(not _LIVE_JSON.exists(), reason="live P1 artifact unavailable")
@_needs_opt
def test_live_flash_artifact_preserves_dtype_contract_and_verifies() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json(json.loads(_LIVE_JSON.read_text()))
    mlir = emit_module(fn, {})
    assert (
        "func.func @high_performance_flash_attn_flash_attn_triton__fwd_kernel("
        in mlir
    )
    for name in ("Q", "K", "V"):
        assert f"%{name}: tensor<1x128x1x64xf16>" in mlir
    lines = mlir.splitlines()
    qk = next(line for line in lines if " = intent.contraction %Q, %K " in line)
    p_f32 = next(line for line in lines if "%p_f32_" in line and "map_unary" in line)
    p_f16 = next(line for line in lines if "%p_f16_" in line and "intent.cast" in line)
    weighted = next(
        line for line in lines
        if "%weighted_unscaled_" in line and "intent.contraction" in line
    )
    out = next(line for line in lines if "%Out_" in line and "intent.cast" in line)
    returned = next(line for line in lines if line.lstrip().startswith("return "))
    assert qk.endswith("-> tensor<128x1x1x128xf32>")
    assert p_f32.endswith("-> tensor<128x1x1x128xf32>")
    assert 'to_dtype = "f16"' in p_f16
    assert p_f16.endswith("-> tensor<128x1x1x128xf16>")
    assert weighted.endswith("-> tensor<64x1x1x128xf32>")
    assert 'to_dtype = "f16"' in out
    assert out.endswith("-> tensor<1x128x1x64xf16>")
    assert (
        "tensor<1x128x1x64xf16>, tensor<1x1x128xf32>, "
        "tensor<1x1x128xf32>" in returned
    )
    _intent_opt(mlir)  # P4 parser/verifier only; lowering is a separate status


@_needs_opt
def test_integer_reduce_preserves_dtype_and_lowers() -> None:
    """A compare/cast/count graph must keep i64 through reduce and the ABI."""
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "count_nonzero_like",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["M", "N"]}],
            "outputs": [{"name": "Y", "dtype": "i64", "shape": []}],
            "body": [
                {
                    "op": "cast",
                    "result": "nz",
                    "operands": ["X"],
                    "attrs": {"to_dtype": "i1"},
                },
                {
                    "op": "cast",
                    "result": "nz_i64",
                    "operands": ["nz"],
                    "attrs": {"to_dtype": "i64"},
                },
                {
                    "op": "reduce",
                    "result": "Y",
                    "operands": ["nz_i64"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [0, 1],
                        "assoc": "associative",
                    },
                },
            ],
        }
    })

    mlir = emit_module(fn, {"M": 4, "N": 17})
    assert "tensor<4x17xi64>) -> tensor<i64>" in mlir
    assert "-> (tensor<i64>)" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "arith.addi" in lowered
    assert "intent." not in lowered


@_needs_opt
def test_transpose_and_symbolic_pad_lower() -> None:
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "transpose_pad",
            "inputs": [
                {
                    "name": "X",
                    "dtype": "f32",
                    "shape": ["B", "M", "D"],
                }
            ],
            "outputs": [
                {
                    "name": "Y",
                    "dtype": "f32",
                    "shape": ["B", "D", "R"],
                }
            ],
            "body": [
                {
                    "op": "transpose",
                    "result": "T",
                    "operands": ["X"],
                    "attrs": {"perm": [0, 2, 1]},
                },
                {
                    "op": "pad",
                    "result": "Y",
                    "operands": ["T"],
                    "attrs": {
                        "lo": [0, 0, 0],
                        "hi": [0, 0, "R - M"],
                        "value": 0.0,
                    },
                },
            ],
        }
    })

    mlir = emit_module(fn, {"B": 1, "M": 2, "D": 3, "R": 4})
    assert "intent.transpose" in mlir
    assert "perm = [0, 2, 1]" in mlir
    assert "intent.pad" in mlir
    assert "hi = [0, 0, 2]" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "linalg.generic" in lowered
    assert "tensor.pad" in lowered or "tensor.empty" in lowered
    assert "intent." not in lowered


@_needs_opt
def test_reduce_to_non_self_inverse_transpose_uses_collapsed_shape() -> None:
    """A declared transposed output constrains an intermediate reduce shape."""
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "reduce_transpose",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": ["A", "B", "C", "D"]}
            ],
            "outputs": [
                {"name": "Y", "dtype": "f32", "shape": ["B", "C", "A"]}
            ],
            "body": [
                {
                    "op": "reduce",
                    "result": "R",
                    "operands": ["X"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [3],
                        "assoc": "float_nondeterministic",
                        "keepdims": False,
                    },
                },
                {
                    "op": "transpose",
                    "result": "Y",
                    "operands": ["R"],
                    "attrs": {"perm": [1, 2, 0]},
                },
            ],
        }
    })

    mlir = emit_module(fn, {"A": 2, "B": 3, "C": 5, "D": 7})
    assert "tensor<2x3x5x7xf32>) -> tensor<2x3x5xf32>" in mlir
    assert "perm = [1, 2, 0]" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "affine_map<(d0, d1, d2) -> (d2, d0, d1)>" in lowered
    assert "intent." not in lowered


@_needs_opt
def test_legacy_reduce_shape_is_inferred_backwards_through_transpose() -> None:
    """Old captures without keepdims still lower when the ABI disambiguates."""
    from intentir.frontend import emit_module

    fn = IntentFunction.from_json({
        "intent_function": {
            "name": "flash_delta_shape",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": [1, "M", "H", "D"]}
            ],
            "outputs": [
                {"name": "Delta", "dtype": "f32", "shape": [1, "H", "M"]}
            ],
            "body": [
                {
                    "op": "reduce",
                    "result": "Delta_bmh",
                    "operands": ["X"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [3],
                        "assoc": "float_nondeterministic",
                    },
                },
                {
                    "op": "transpose",
                    "result": "Delta",
                    "operands": ["Delta_bmh"],
                    "attrs": {"perm": [0, 2, 1]},
                },
            ],
        }
    })

    mlir = emit_module(fn, {"M": 128, "H": 1, "D": 64})
    assert "tensor<1x128x1x64xf32>) -> tensor<1x128x1xf32>" in mlir
    lowered = _intent_opt(mlir, "-convert-intent-to-linalg")
    assert "intent." not in lowered


def test_transpose_and_pad_reject_invalid_shape_contracts() -> None:
    from intentir.frontend import emit_module

    bad_perm = IntentFunction.from_json({
        "intent_function": {
            "name": "bad_perm",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [2, 3, 4]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [2, 3, 4]}],
            "body": [{
                "op": "transpose",
                "result": "Y",
                "operands": ["X"],
                "attrs": {"perm": [0, 1, 1]},
            }],
        }
    })
    with pytest.raises(ValueError, match="invalid"):
        emit_module(bad_perm, {})

    negative_pad = IntentFunction.from_json({
        "intent_function": {
            "name": "negative_pad",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [2, 3]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [2, 2]}],
            "body": [{
                "op": "pad",
                "result": "Y",
                "operands": ["X"],
                "attrs": {"lo": [0, 0], "hi": [0, -1], "value": 0.0},
            }],
        }
    })
    with pytest.raises(ValueError, match="non-negative"):
        emit_module(negative_pad, {})


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(), reason="intent-opt not built")
@pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(), reason="CUDA toolkit not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_emitted_body_executes_vs_interpret(tmp_path) -> None:
    """The whole front+back path: emit intent-dialect MLIR from the JSON, run the
    body once per K-tile on the 5090D with carries in persistent CUDA buffers,
    and check the result matches the numpy interpreter run on the same JSON IR."""
    from intentir.backend.cuda.orchestrate import run_program
    from intentir.frontend import emit_body_function
    from intentir.verify.back.legacy_v0.interpret import interpret

    fn = _fn_with_f32_accumulators()
    Sq, Sk, D, BK = _DIMS["Sq"], _DIMS["Sk"], _DIMS["D"], _DIMS["BK"]
    body_path = tmp_path / "flash_body_fwd.mlir"
    body_path.write_text(
        emit_body_function(fn, _DIMS, schedule_tile=BK)
    )
    nt = Sk // BK

    for seed, scale in [(0, 1.0), (1, 0.25), (2, 0.5)]:
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((1, 1, Sq, D)).astype(np.float16)
        K = rng.standard_normal((1, 1, Sk, D)).astype(np.float16)
        V = rng.standard_normal((1, 1, Sk, D)).astype(np.float16)
        o_interp = interpret(
            fn, {"Q": Q, "K": K, "V": V, "scale": np.float32(scale)}, {"BK": BK}
        )["O"][0, 0]

        def t(x):
            return torch.tensor(x, device="cuda").contiguous()

        Qd, Kd, Vd, sc = t(Q[0, 0]), t(K[0, 0]), t(V[0, 0]), t(np.float32(scale))
        m = torch.full((Sq, 1), float("-inf"), device="cuda").contiguous()
        l = torch.zeros((Sq, 1), device="cuda").contiguous()
        Oacc = torch.zeros((Sq, D), device="cuda").contiguous()
        for ti in range(nt):
            out = run_program(str(body_path), {
                "Q": Qd, "scale": sc,
                "Kj": Kd[ti * BK:(ti + 1) * BK].contiguous(),
                "Vj": Vd[ti * BK:(ti + 1) * BK].contiguous(),
                "m_iter": m, "l_iter": l, "Oacc_iter": Oacc})
            m = out["out0"].reshape(Sq, 1).contiguous()
            l = out["out1"].reshape(Sq, 1).contiguous()
            Oacc = out["out2"].reshape(Sq, D).contiguous()
        o_exec = (Oacc / l).cpu().numpy()
        rel = np.abs(o_exec - o_interp).max() / (np.abs(o_interp).max() + 1e-6)
        assert rel < 1e-3, f"seed={seed} scale={scale}: rel={rel}"
