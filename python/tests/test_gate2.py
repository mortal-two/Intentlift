"""Gate2 source/replay orchestration tests."""

from __future__ import annotations

import numpy as np
import pytest

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Gate2Suite,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
    compare_outputs,
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


def _case(
    *,
    case_id: str = "dynamic-equivalence",
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
        ("Y",),
        {} if tolerance is None else {"Y": tolerance},
    )


def _dummy_source() -> KernelSource:
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
    ("source", "replay", "status"),
    [
        (0.0, 0.0, Gate2Status.PASS),
        (0.0, 1.0, Gate2Status.REPLAY_DIVERGED),
    ],
)
def test_gate2_classifies_only_source_replay_agreement(
    source: float,
    replay: float,
    status: Gate2Status,
) -> None:
    array = lambda value: np.asarray([value], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": array(source)}),
        replay_runner=_FakeTritonRunner({"Y": array(replay)}),
    )

    result = runner.run(_dummy_source(), _dummy_source(), _case())

    assert result.status is status
    assert result.source_replay is not None
    assert result.source_replay.passed is (status is Gate2Status.PASS)
    assert not hasattr(result, "lowering")


def test_gate2_execution_failure_is_not_a_semantic_divergence() -> None:
    output = np.asarray([3.0], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": output}),
        replay_runner=_FakeTritonRunner(
            {}, ok=False, error="synthetic replay failure"
        ),
    )

    result = runner.run(_dummy_source(), _dummy_source(), _case())

    assert result.status is Gate2Status.EXECUTION_FAILED
    assert result.source.ok
    assert not result.replay.ok
    assert result.source_replay is None


def test_gate2_uses_per_output_tolerance_for_floating_values() -> None:
    source = np.asarray([1.0], dtype=np.float32)
    replay = np.asarray([1.0005], dtype=np.float32)
    strict = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": source}),
        replay_runner=_FakeTritonRunner({"Y": replay}),
    ).run(_dummy_source(), _dummy_source(), _case())
    tolerant = Gate2Runner(
        source_runner=_FakeTritonRunner({"Y": source}),
        replay_runner=_FakeTritonRunner({"Y": replay}),
    ).run(
        _dummy_source(),
        _dummy_source(),
        _case(tolerance=Tolerance(rtol=1.0e-3, atol=0.0)),
    )

    assert strict.status is Gate2Status.REPLAY_DIVERGED
    assert tolerant.status is Gate2Status.PASS


class _CaseAwareRunner:
    def __init__(self, *, replay: bool) -> None:
        self.replay = replay
        self.calls: list[str] = []

    def run(self, leg, program, spec, values):
        name = str(values["case_name"])
        self.calls.append(name)
        delta = 1.0 if self.replay and values.get("diverge") else 0.0
        output = np.asarray(values["X"], dtype=np.float32) + delta
        return ExecutionResult(leg, True, {"Y": output})


def test_gate2_suite_runs_every_case_and_aggregates_strictly() -> None:
    first = _case(
        case_id="matrix.pass",
        values={
            "X": np.asarray([2.0], dtype=np.float32),
            "case_name": "pass",
            "diverge": False,
        },
    )
    second = _case(
        case_id="matrix.replay-diverged",
        values={
            "X": np.asarray([2.0], dtype=np.float32),
            "case_name": "fail",
            "diverge": True,
        },
    )
    source_runner = _CaseAwareRunner(replay=False)
    replay_runner = _CaseAwareRunner(replay=True)
    result = Gate2Runner(
        source_runner=source_runner,
        replay_runner=replay_runner,
    ).run_suite(
        _dummy_source(),
        _dummy_source(),
        Gate2Suite("strict.matrix", (first, second)),
    )

    assert not result.passed
    assert result.failed_case_ids == ("matrix.replay-diverged",)
    assert result.status_counts == {
        "gate2_pass": 1,
        "replay_diverged": 1,
    }
    assert source_runner.calls == ["pass", "fail"]
    assert replay_runner.calls == ["pass", "fail"]


def test_gate2_suite_rejects_duplicate_case_ids() -> None:
    case = _case(case_id="duplicate")
    with pytest.raises(ValueError, match="case_id values must be unique"):
        Gate2Suite("bad", (case, case))
