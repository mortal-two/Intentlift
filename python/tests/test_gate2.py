"""Gate2 three-way orchestration and one real add-kernel prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Lower2CRunner,
    Lower2CSpec,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
)


ROOT = Path(__file__).resolve().parents[2]
HAVE_CPU_LOWERING = (
    (ROOT / "build/tools/intent-opt/intent-opt").exists()
    and Path("/usr/lib/llvm-20/bin/mlir-runner").exists()
)


def _intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "identity",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [1]}],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [1]}],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["X", "X"],
                        "attrs": {"fn": "add"},
                    }
                ],
            }
        }
    )


class _FakeTritonRunner:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, leg, program, spec, values):
        return ExecutionResult(leg, True, {"Y": self.output})


class _FakeLowerRunner:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, spec, values):
        return ExecutionResult("lowering", True, {"Y": self.output})


def _case() -> Gate2Case:
    return Gate2Case(
        "classification",
        {"X": np.array([0.0], dtype=np.float32)},
        TritonExecutionSpec(
            (TritonArgument("out", ArgumentRole.OUTPUT, dtype="f32", shape=(1,), output_key="Y"),),
            (1,),
        ),
        Lower2CSpec(_intent(), {"N": 1}, {"X": "X"}, "Y"),
        ("Y",),
    )


@pytest.mark.parametrize(
    "source,replay,lowering,status",
    [
        (0.0, 0.0, 0.0, Gate2Status.PASS),
        (0.0, 0.0, 1.0, Gate2Status.LOWERING_DIVERGED),
        (0.0, 1.0, 0.0, Gate2Status.REPLAY_DIVERGED),
        (1.0, 0.0, 0.0, Gate2Status.LIFT_DIVERGED),
        (0.0, 1.0, 2.0, Gate2Status.ALL_DIVERGED),
    ],
)
def test_gate2_mechanical_attribution(source, replay, lowering, status) -> None:
    array = lambda value: np.array([value], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner(array(source)),
        replay_runner=_FakeTritonRunner(array(replay)),
        lowering_runner=_FakeLowerRunner(array(lowering)),
    )
    dummy = KernelSource.from_code("", "kernel")
    result = runner.run(dummy, dummy, _case())
    assert result.status is status


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_uses_real_intent_lowering_not_legacy_interpreter() -> None:
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/add/intent.json").read_text())
    )
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    result = Lower2CRunner().run(
        Lower2CSpec(fn, {"M": 2, "N": 3}, {"A": "A", "B": "B"}, "Y"),
        {"A": a, "B": b},
    )
    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["Y"], a + b)
    assert "mlir-runner" in result.metadata["path"]


def _have_cuda_triton() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_add_source_replay_lowering_three_way() -> None:
    # This proves the Gate2 wiring.  The saved Intent JSON is LLM-extracted;
    # the temporary five-kernel replay remains probe material, so this test is
    # not claimed as a full new lifter->Gate1->Gate2 corpus result.
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/add/intent.json").read_text())
    )
    n = 65
    rng = np.random.default_rng(7)
    values = {
        "A": rng.standard_normal((1, n)).astype(np.float32),
        "B": rng.standard_normal((1, n)).astype(np.float32),
        "n": n,
    }
    triton_spec = TritonExecutionSpec(
        (
            TritonArgument("a_ptr", ArgumentRole.INPUT, "A", "f32"),
            TritonArgument("b_ptr", ArgumentRole.INPUT, "B", "f32"),
            TritonArgument(
                "out_ptr",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(1, n),
                output_key="Y",
            ),
            TritonArgument("n_elements", ArgumentRole.SCALAR, "n"),
        ),
        ((n + 63) // 64,),
        {"BLOCK": 64, "num_warps": 4, "num_stages": 2},
    )
    case = Gate2Case(
        "add.random.tail65",
        values,
        triton_spec,
        Lower2CSpec(fn, {"M": 1, "N": n}, {"A": "A", "B": "B"}, "Y"),
        ("Y",),
        {"Y": Tolerance(rtol=1.0e-5, atol=1.0e-6)},
    )
    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py", "add_kernel", trusted=True
        ),
        KernelSource.from_file(
            ROOT / "test/scrc_probe_replays.py", "add_kernel"
        ),
        case,
    )
    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )
