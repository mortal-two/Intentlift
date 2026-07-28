"""Gate2 scalar backend: typed kernels execute through MLIR and a memref ABI.

Deterministic and GPU-free: runtime values cross a ranked-memref C interface and
never become MLIR constants.  All logical outputs are produced by one isolated
native call.  Skips if the MLIR CPU tools are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_HAVE = all(
    path.exists()
    for path in (
        ROOT / "build/tools/intent-opt/intent-opt",
        Path("/usr/lib/llvm-20/bin/mlir-opt"),
        Path("/usr/lib/llvm-20/bin/mlir-translate"),
        Path("/usr/lib/llvm-20/bin/clang-20"),
    )
)

pytestmark = pytest.mark.skipif(not _HAVE, reason="needs the scalar MLIR toolchain")


def _np_ref(name, arrs):
    x = arrs[0]
    if name == "add":
        return arrs[0] + arrs[1]
    if name == "silu":
        return x * (1.0 / (1.0 + np.exp(-x)))
    if name == "softmax":
        z = x - x.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    if name == "matmul":
        return arrs[0] @ arrs[1]
    raise KeyError(name)


@pytest.mark.parametrize("name,shapes,dims", [
    ("add", [(2, 3), (2, 3)], {"M": 2, "N": 3}),
    ("silu", [(2, 4)], {"M": 2, "N": 4}),
    ("softmax", [(3, 5)], {"M": 3, "N": 5}),
    ("matmul", [(3, 4), (4, 5)], {"M": 3, "K": 4, "N": 5}),
])
def test_kernel_runs_on_cpu(name, shapes, dims):
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus" / "kernels" / name / "intent.json").read_text()))
    y_cpu = run_on_cpu(fn, dims, arrs)
    y_ref = _np_ref(name, arrs)
    rel = np.abs(y_cpu - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
    assert rel < 1e-4, f"{name}: CPU exec rel {rel}"


def test_collapsed_row_sum_output_runs_on_cpu() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "row_sum_lowering",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["M", "N"]}
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": ["M"]}
                ],
                "body": [
                    {
                        "op": "reduce",
                        "result": "Y",
                        "operands": ["X"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [1],
                            "assoc": "float_nondeterministic",
                        },
                    }
                ],
            }
        }
    )
    x = np.arange(15, dtype=np.float32).reshape(3, 5)

    actual = run_on_cpu(fn, {"M": 3, "N": 5}, [x])

    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, x.sum(axis=1), rtol=1.0e-6, atol=1.0e-6)


def test_collapsed_intermediate_reduce_then_pad_runs_on_cpu() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "row_sum_then_pad",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [1, "M", 3, 4]}
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [1, "M", 5]}
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
                        },
                        "type": {"dtype": "f32", "shape": [1, "M", 3]},
                    },
                    {
                        "op": "pad",
                        "result": "Y",
                        "operands": ["R"],
                        "attrs": {
                            "lo": [0, 0, 0],
                            "hi": [0, 0, 2],
                            "value": 0.0,
                        },
                        "type": {"dtype": "f32", "shape": [1, "M", 5]},
                    },
                ],
            }
        }
    )
    value = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)

    actual = run_on_cpu(fn, {"M": 2}, [value])

    expected = np.pad(value.sum(axis=3), ((0, 0), (0, 0), (0, 2)))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_rsqrt_uses_standard_host_sqrt_abi() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "host_rsqrt",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [8]}],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [8]}],
                "body": [
                    {
                        "op": "map_unary",
                        "result": "Y",
                        "operands": ["X"],
                        "attrs": {"fn": "rsqrt"},
                    }
                ],
            }
        }
    )
    value = np.linspace(0.25, 4.0, 8, dtype=np.float32)

    actual = run_on_cpu(fn, {}, [value])

    np.testing.assert_allclose(
        actual,
        1.0 / np.sqrt(value),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_multiple_logical_outputs_run_on_cpu() -> None:
    from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "multi_result_scalar_lowering",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["N"]}
                ],
                "outputs": [
                    {"name": "Neg", "dtype": "f32", "shape": ["N"]},
                    {"name": "Square", "dtype": "f32", "shape": ["N"]},
                ],
                "body": [
                    {
                        "op": "map_unary",
                        "result": "Neg",
                        "operands": ["X"],
                        "attrs": {"fn": "neg"},
                    },
                    {
                        "op": "map_binary",
                        "result": "Square",
                        "operands": ["X", "X"],
                        "attrs": {"fn": "mul"},
                    },
                ],
                "returns": [
                    {"output": "Neg", "value": "Neg"},
                    {"output": "Square", "value": "Square"},
                ],
            }
        }
    )
    value = np.asarray([-2.0, 0.5, 3.0], dtype=np.float32)

    execution = execute_on_cpu_outputs(fn, {"N": 3}, [value])
    actual = execution.outputs

    assert tuple(actual) == ("Neg", "Square")
    np.testing.assert_array_equal(actual["Neg"], -value)
    np.testing.assert_array_equal(actual["Square"], value * value)
    assert execution.metadata["native_call_count"] == 1
    assert execution.metadata["output_count"] == 2
    assert execution.metadata["isolated_process"] is True


def test_cpu_runner_reads_back_exact_f32_bits() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "name": "identity_rank0",
            "inputs": [{"name": "X", "dtype": "f32", "shape": []}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
            "body": [
                {
                    "op": "cast",
                    "result": "Y",
                    "operands": ["X"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    value = np.asarray(91.181175, dtype=np.float32)

    actual = run_on_cpu(fn, {}, [value])

    assert actual.shape == ()
    assert actual.view(np.uint32) == value.view(np.uint32)


def test_cpu_runner_accepts_and_preserves_ieee_special_input_bits() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "name": "identity_ieee",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "Y",
                    "operands": ["X"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    values = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32
    )

    actual = run_on_cpu(fn, {"N": len(values)}, [values])

    np.testing.assert_array_equal(
        actual.view(np.uint32), values.view(np.uint32)
    )


def test_cpu_module_does_not_embed_runtime_input_values() -> None:
    from intentir.backend.cpu.pipeline import emit_cpu_module
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "name": "value_independent_identity",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "Y",
                    "operands": ["X"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )

    small = emit_cpu_module(fn, {"N": 7})
    large = emit_cpu_module(fn, {"N": 700_000})

    assert "dense<" not in small
    assert "91.181175" not in small
    assert "intentir_cpu_entry_v1" in small
    assert small.count("bufferization.materialize_in_destination") == 1
    assert abs(len(large) - len(small)) < 100


def test_structured_loop_is_explicitly_unsupported_not_reinterpreted() -> None:
    from intentir.backend.cpu.abi import CPUBackendUnsupportedError
    from intentir.backend.cpu.pipeline import emit_cpu_module
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "name": "unsupported_structured_loop",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
            "body": [{"op": "loop"}],
        }
    )

    with pytest.raises(CPUBackendUnsupportedError, match="structured IntentIR loops"):
        emit_cpu_module(fn, {"N": 4})


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("f16", np.asarray([-2.0, 0.5, 3.0], dtype=np.float16)),
        ("f64", np.asarray([-2.0, 0.5, 3.0], dtype=np.float64)),
        ("i1", np.asarray([True, False, True], dtype=np.bool_)),
        ("i8", np.asarray([-2, 0, 3], dtype=np.int8)),
        ("i16", np.asarray([-2, 0, 3], dtype=np.int16)),
        ("i32", np.asarray([-2, 0, 3], dtype=np.int32)),
        ("i64", np.asarray([-2, 0, 3], dtype=np.int64)),
    ],
)
def test_verified_memref_storage_types(dtype, values) -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu
    from intentir.ir.model import IntentFunction

    fn = IntentFunction.from_json(
        {
            "name": f"identity_{dtype}",
            "inputs": [{"name": "X", "dtype": dtype, "shape": ["N"]}],
            "outputs": [{"name": "Y", "dtype": dtype, "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "Y",
                    "operands": ["X"],
                    "attrs": {"dtype": dtype},
                }
            ],
        }
    )

    actual = run_on_cpu(fn, {"N": 3}, [values])

    np.testing.assert_array_equal(actual, values)
