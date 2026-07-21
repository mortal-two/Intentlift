"""Gate2 source/replay and source/scalar differential tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intentir.ir import (
    BoundaryContract,
    IntentFunction,
    IntentProgram,
    LogicalArgument,
    LogicalInterface,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Gate2Suite,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
    compare_outputs,
)


FINAL_ARTIFACT_DIGEST = "a" * 64
GATE1_J_DIGEST = "b" * 64
ROOT = Path(__file__).resolve().parents[2]
HAVE_SCALAR_TOOLS = (
    (ROOT / "build/tools/intent-opt/intent-opt").exists()
    and Path("/usr/lib/llvm-20/bin/mlir-opt").exists()
    and Path("/usr/lib/llvm-20/bin/mlir-translate").exists()
    and Path("/usr/lib/llvm-20/bin/clang-20").exists()
)


class _FakeTritonRunner:
    def __init__(
        self,
        outputs: dict[str, np.ndarray],
        *,
        ok: bool = True,
        error: str = "",
    ) -> None:
        self.outputs = outputs
        self.ok = ok
        self.error = error
        self.calls: list[str] = []

    def run(self, leg, program, spec, values):
        self.calls.append(str(values.get("case_name", leg)))
        return ExecutionResult(
            leg,
            self.ok,
            {name: value.copy() for name, value in self.outputs.items()},
            error=self.error,
        )


class _FakeScalarRunner:
    def __init__(
        self,
        outputs: dict[str, np.ndarray],
        *,
        ok: bool = True,
        unsupported: bool = False,
        error: str = "",
    ) -> None:
        self.outputs = outputs
        self.ok = ok
        self.unsupported = unsupported
        self.error = error
        self.calls: list[str] = []

    def run(self, spec, values):
        self.calls.append(str(values.get("case_name", "scalar")))
        return ExecutionResult(
            "scalar",
            self.ok,
            {name: value.copy() for name, value in self.outputs.items()},
            error=self.error,
            unsupported=self.unsupported,
        )


def _program() -> IntentProgram:
    function = IntentFunction.from_json(
        {
            "name": "gate2_identity",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "cast",
                    "result": "value",
                    "operands": ["X"],
                    "attrs": {"to_dtype": "f32"},
                    "type": {"dtype": "f32", "shape": ["N"]},
                }
            ],
            "returns": [{"output": "Y", "value": "value"}],
        }
    )
    interface = LogicalInterface(
        "gate2_identity",
        (
            LogicalArgument(
                "x_ptr",
                "tensor",
                "input",
                "f32",
                rank=1,
                symbolic_shape=("N",),
                tensor_ref="X",
            ),
            LogicalArgument(
                "y_ptr",
                "tensor",
                "output",
                "f32",
                rank=1,
                symbolic_shape=("N",),
                tensor_ref="Y",
            ),
        ),
        ("Y",),
    )
    return IntentProgram(function, BoundaryContract(), interface)


def _scalar_spec() -> ScalarLoweringSpec:
    return ScalarLoweringSpec.create(
        _program(),
        final_artifact_digest=FINAL_ARTIFACT_DIGEST,
        gate1_j_digest=GATE1_J_DIGEST,
        dims={"N": 1},
        input_bindings={"X": "X"},
        output_bindings={"Y": "Y"},
    )


def _case(
    *,
    case_id: str = "source-replay-scalar",
    values: dict[str, object] | None = None,
    tolerance: Tolerance | None = None,
) -> Gate2Case:
    return Gate2Case(
        case_id,
        values or {"X": np.asarray([0.0], dtype=np.float32)},
        TritonExecutionSpec(
            (
                TritonArgument(
                    "out",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(1,),
                    output_key="Y",
                ),
            ),
            (1,),
        ),
        _scalar_spec(),
        FINAL_ARTIFACT_DIGEST,
        ("Y",),
        {} if tolerance is None else {"Y": tolerance},
    )


def _dummy_kernel() -> KernelSource:
    return KernelSource.from_code("", "kernel")


def test_gate2_integer_and_boolean_outputs_use_exact_comparison() -> None:
    large = np.asarray([2**60], dtype=np.int64)
    adjacent = np.asarray([2**60 + 1], dtype=np.int64)
    integer = compare_outputs(
        "source",
        {"Y": large},
        "replay",
        {"Y": adjacent},
        ("Y",),
        {"Y": Tolerance(rtol=1.0, atol=1.0)},
    )
    assert not integer.passed
    assert integer.outputs[0].max_abs_error == 1.0

    boolean = compare_outputs(
        "source",
        {"Y": np.asarray([True, False])},
        "replay",
        {"Y": np.asarray([False, False])},
        ("Y",),
    )
    assert not boolean.passed
    assert boolean.outputs[0].max_abs_error == 1.0


@pytest.mark.parametrize(
    ("source", "replay", "scalar", "status"),
    [
        (0.0, 0.0, 0.0, Gate2Status.PASS),
        (0.0, 1.0, 0.0, Gate2Status.REPLAY_DIVERGED),
        (0.0, 0.0, 1.0, Gate2Status.SCALAR_DIVERGED),
        (0.0, 1.0, 2.0, Gate2Status.REPLAY_AND_SCALAR_DIVERGED),
    ],
)
def test_gate2_classifies_both_source_anchored_differentials(
    source: float,
    replay: float,
    scalar: float,
    status: Gate2Status,
) -> None:
    array = lambda value: np.asarray([value], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": array(source)}),
        replay_runner=_FakeTritonRunner({"Y": array(replay)}),
        scalar_runner=_FakeScalarRunner({"Y": array(scalar)}),
    )

    result = runner.run(_dummy_kernel(), _dummy_kernel(), _case())

    assert result.status is status
    assert result.source_replay is not None
    assert result.source_scalar is not None
    assert result.source_replay.passed is (
        status in {Gate2Status.PASS, Gate2Status.SCALAR_DIVERGED}
    )
    assert result.source_scalar.passed is (
        status in {Gate2Status.PASS, Gate2Status.REPLAY_DIVERGED}
    )
    assert result.input_digest


def test_gate2_execution_failure_is_not_a_semantic_mismatch() -> None:
    output = np.asarray([3.0], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": output}),
        replay_runner=_FakeTritonRunner(
            {}, ok=False, error="synthetic replay failure"
        ),
        scalar_runner=_FakeScalarRunner({"Y": output}),
    )

    result = runner.run(_dummy_kernel(), _dummy_kernel(), _case())

    assert result.status is Gate2Status.EXECUTION_FAILED
    assert result.source.ok
    assert not result.replay.ok
    assert result.scalar.ok
    assert result.source_replay is None
    assert result.source_scalar is not None
    assert result.source_scalar.passed


def test_gate2_scalar_unsupported_is_fail_closed() -> None:
    output = np.asarray([3.0], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": output}),
        replay_runner=_FakeTritonRunner({"Y": output}),
        scalar_runner=_FakeScalarRunner(
            {}, ok=False, unsupported=True, error="unsupported op"
        ),
    )

    result = runner.run(_dummy_kernel(), _dummy_kernel(), _case())

    assert result.status is Gate2Status.SCALAR_UNSUPPORTED
    assert result.source_replay is not None
    assert result.source_replay.passed
    assert result.source_scalar is None
    assert not result.passed


def test_scalar_unsupported_does_not_hide_source_replay_execution_failure() -> None:
    output = np.asarray([3.0], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": output}),
        replay_runner=_FakeTritonRunner(
            {}, ok=False, error="synthetic replay failure"
        ),
        scalar_runner=_FakeScalarRunner(
            {}, ok=False, unsupported=True, error="unsupported op"
        ),
    )

    result = runner.run(_dummy_kernel(), _dummy_kernel(), _case())

    assert result.status is Gate2Status.EXECUTION_FAILED
    assert result.source_replay is None
    assert result.source_scalar is None


def test_gate2_handoff_failure_does_not_suppress_source_replay() -> None:
    output = np.asarray([3.0], dtype=np.float32)
    scalar_runner = _FakeScalarRunner({"Y": output})
    unavailable = ScalarLoweringUnavailable(
        FINAL_ARTIFACT_DIGEST,
        GATE1_J_DIGEST,
        "GateProgramHandoffError: untyped predicate",
    )
    case = _case()
    case = Gate2Case(
        case.case_id,
        case.values,
        case.triton,
        unavailable,
        case.replay_artifact_digest,
        case.output_keys,
        case.tolerances,
    )
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": output}),
        replay_runner=_FakeTritonRunner({"Y": output}),
        scalar_runner=scalar_runner,
    )

    result = runner.run(_dummy_kernel(), _dummy_kernel(), case)

    assert result.status is Gate2Status.SCALAR_UNSUPPORTED
    assert result.source_replay is not None
    assert result.source_replay.passed
    assert result.source_scalar is None
    assert result.scalar.metadata["failure_class"] == "intent_program_handoff"
    assert scalar_runner.calls == []


@pytest.mark.skipif(not HAVE_SCALAR_TOOLS, reason="needs scalar MLIR toolchain")
def test_scalar_runner_executes_the_typed_program_not_a_cuda_fallback() -> None:
    spec = ScalarLoweringSpec.create(
        _program(),
        final_artifact_digest=FINAL_ARTIFACT_DIGEST,
        gate1_j_digest=GATE1_J_DIGEST,
        dims={"N": 3},
        input_bindings={"X": "input"},
        output_bindings={"Y": "result"},
    )
    value = np.asarray([-1.0, 2.0, 4.5], dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["result"], value)
    assert result.metadata["target_backend_in_verdict"] is False
    assert "CUDA" not in result.metadata["path"]
    execution = result.metadata["execution_evidence"]
    assert execution["abi_protocol"] == "intentir_cpu_ranked_memref_abi_v1"
    assert execution["native_call_count"] == 1
    assert execution["module_depends_on_runtime_values"] is False
    assert len(execution["scalar_module_digest"]) == 64
    assert len(execution["shared_object_digest"]) == 64


def test_gate2_uses_per_output_tolerance_for_floating_values() -> None:
    source = np.asarray([1.0], dtype=np.float32)
    replay = np.asarray([1.0005], dtype=np.float32)
    scalar = np.asarray([1.0005], dtype=np.float32)
    strict = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": source}),
        replay_runner=_FakeTritonRunner({"Y": replay}),
        scalar_runner=_FakeScalarRunner({"Y": scalar}),
    ).run(_dummy_kernel(), _dummy_kernel(), _case())
    tolerant = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": source}),
        replay_runner=_FakeTritonRunner({"Y": replay}),
        scalar_runner=_FakeScalarRunner({"Y": scalar}),
    ).run(
        _dummy_kernel(),
        _dummy_kernel(),
        _case(tolerance=Tolerance(rtol=1.0e-3, atol=0.0)),
    )

    assert strict.status is Gate2Status.REPLAY_AND_SCALAR_DIVERGED
    assert tolerant.status is Gate2Status.PASS


class _CaseAwareTritonRunner:
    def __init__(self, *, replay: bool) -> None:
        self.replay = replay
        self.calls: list[str] = []

    def run(self, leg, program, spec, values):
        name = str(values["case_name"])
        self.calls.append(name)
        delta = 1.0 if self.replay and values.get("replay_diverge") else 0.0
        return ExecutionResult(
            leg,
            True,
            {"Y": np.asarray(values["X"], dtype=np.float32) + delta},
        )


class _CaseAwareScalarRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, spec, values):
        name = str(values["case_name"])
        self.calls.append(name)
        delta = 1.0 if values.get("scalar_diverge") else 0.0
        output = np.asarray(values["X"], dtype=np.float32) + delta
        return ExecutionResult("scalar", True, {"Y": output})


def test_gate2_suite_runs_every_case_and_aggregates_strictly() -> None:
    first = _case(
        case_id="matrix.pass",
        values={
            "X": np.asarray([2.0], dtype=np.float32),
            "case_name": "pass",
            "replay_diverge": False,
            "scalar_diverge": False,
        },
    )
    second = _case(
        case_id="matrix.scalar-diverged",
        values={
            "X": np.asarray([2.0], dtype=np.float32),
            "case_name": "fail",
            "replay_diverge": False,
            "scalar_diverge": True,
        },
    )
    source_runner = _CaseAwareTritonRunner(replay=False)
    replay_runner = _CaseAwareTritonRunner(replay=True)
    scalar_runner = _CaseAwareScalarRunner()
    result = Gate2Runner(
        source_runner=source_runner,
        replay_runner=replay_runner,
        scalar_runner=scalar_runner,
    ).run_suite(
        _dummy_kernel(),
        _dummy_kernel(),
        Gate2Suite("strict.matrix", (first, second)),
    )

    assert not result.passed
    assert result.failed_case_ids == ("matrix.scalar-diverged",)
    assert result.status_counts == {
        "gate2_pass": 1,
        "scalar_diverged": 1,
    }
    assert source_runner.calls == ["pass", "fail"]
    assert replay_runner.calls == ["pass", "fail"]
    assert scalar_runner.calls == ["pass", "fail"]


def test_gate2_rejects_unaligned_replay_and_scalar_artifacts() -> None:
    with pytest.raises(ValueError, match="not bound"):
        Gate2Case(
            "misaligned",
            {"X": np.asarray([0.0], dtype=np.float32)},
            _case().triton,
            _scalar_spec(),
            "c" * 64,
            ("Y",),
        )


def test_gate2_suite_rejects_duplicate_case_ids() -> None:
    case = _case(case_id="duplicate")
    with pytest.raises(ValueError, match="case_id values must be unique"):
        Gate2Suite("bad", (case, case))
