from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intentir.backend.cuda.compile import INTENT_OPT
from intentir.backend.cpu.pipeline import run_on_cpu
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.model import IntentFunction


def test_iota_accepts_symbolic_runtime_extent() -> None:
    function = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "symbolic_iota",
                "inputs": [],
                "outputs": [{"name": "Y", "dtype": "i32", "shape": ["N"]}],
                "body": [
                    {
                        "op": "iota",
                        "result": "Y",
                        "attrs": {"n": "N", "dtype": "i32"},
                    }
                ],
            }
        }
    )

    module = emit_module(function, {"N": 17})

    assert "intent.iota : tensor<17xi32>" in module
    assert "tensor<17xi32>" in module


@pytest.mark.skipif(not Path(INTENT_OPT).exists(), reason="intent-opt not built")
def test_symbolic_iota_arange_executes_through_cpu_lowering() -> None:
    function = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "arange_lowering_probe",
                "inputs": [
                    {"name": "start", "dtype": "f32", "shape": []},
                    {"name": "step", "dtype": "f32", "shape": []},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
                "body": [
                    {
                        "op": "iota",
                        "result": "idx",
                        "attrs": {"n": "N", "dtype": "i32"},
                    },
                    {
                        "op": "cast",
                        "result": "idx_f",
                        "operands": ["idx"],
                        "attrs": {"to_dtype": "f32"},
                    },
                    {
                        "op": "map_binary",
                        "result": "scaled",
                        "operands": ["idx_f", "step"],
                        "attrs": {"fn": "mul"},
                    },
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["scaled", "start"],
                        "attrs": {"fn": "add"},
                    },
                ],
            }
        }
    )
    start = np.asarray(-3.25, dtype=np.float32)
    step = np.asarray(0.5, dtype=np.float32)

    output = run_on_cpu(function, {"N": 259}, [start, step])
    expected = np.arange(259, dtype=np.float32) * step + start

    np.testing.assert_array_equal(output, expected)
