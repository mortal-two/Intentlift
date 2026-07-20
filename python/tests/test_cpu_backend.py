"""Second backend: lifted kernels execute on CPU (intent -> linalg -> LLVM JIT).

Deterministic and GPU-free -- it uses the committed extraction artifacts, numpy
inputs, and mlir-runner. Proves intent->linalg is backend-neutral (the GPU tail
is the only specialized part). Skips if the MLIR CPU tools aren't present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_HAVE = (ROOT / "build/tools/intent-opt/intent-opt").exists() and \
        Path("/usr/lib/llvm-20/bin/mlir-runner").exists()

pytestmark = pytest.mark.skipif(not _HAVE, reason="needs intent-opt + mlir-runner")


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


def test_multiple_logical_outputs_run_on_cpu() -> None:
    from intentir.backend.cpu.pipeline import run_on_cpu_outputs
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

    actual = run_on_cpu_outputs(fn, {"N": 3}, [value])

    assert tuple(actual) == ("Neg", "Square")
    np.testing.assert_array_equal(actual["Neg"], -value)
    np.testing.assert_array_equal(actual["Square"], value * value)


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
