"""SCRC Gate2 source/replay dynamic execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from intentir.verify.gate1.compiler import KernelSource

from .compare import compare_outputs
from .model import Gate2Result, Gate2Status, Gate2SuiteResult, Tolerance
from .triton import TritonExecutionSpec, TritonRunner


GATE2_EXECUTION_PROTOCOL = "source_replay_dynamic_equivalence_v1"


@dataclass(frozen=True)
class Gate2Case:
    case_id: str
    values: Mapping[str, Any]
    triton: TritonExecutionSpec
    output_keys: tuple[str, ...]
    tolerances: Mapping[str, Tolerance] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")


@dataclass(frozen=True)
class Gate2Suite:
    suite_id: str
    cases: tuple[Gate2Case, ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite case_id values must be unique")


class Gate2Runner:
    def __init__(
        self,
        *,
        source_runner: TritonRunner | None = None,
        replay_runner: TritonRunner | None = None,
    ) -> None:
        self.source_runner = source_runner or TritonRunner()
        self.replay_runner = replay_runner or TritonRunner()

    def run(
        self,
        source: KernelSource,
        replay: KernelSource,
        case: Gate2Case,
    ) -> Gate2Result:
        source_result = self.source_runner.run(
            "source", source, case.triton, case.values
        )
        replay_result = self.replay_runner.run(
            "replay", replay, case.triton, case.values
        )
        source_replay = (
            compare_outputs(
                "source",
                source_result.outputs,
                "replay",
                replay_result.outputs,
                case.output_keys,
                case.tolerances,
            )
            if source_result.ok and replay_result.ok
            else None
        )

        if not source_result.ok or not replay_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        else:
            assert source_replay is not None
            status = (
                Gate2Status.PASS
                if source_replay.passed
                else Gate2Status.REPLAY_DIVERGED
            )
        return Gate2Result(
            case.case_id,
            status,
            source_result,
            replay_result,
            source_replay,
            dict(case.metadata),
        )

    def run_suite(
        self,
        source: KernelSource,
        replay: KernelSource,
        suite: Gate2Suite,
    ) -> Gate2SuiteResult:
        """Run every matrix member; one failure keeps the suite from passing."""

        return Gate2SuiteResult(
            suite.suite_id,
            tuple(self.run(source, replay, case) for case in suite.cases),
        )


__all__ = [
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Runner",
    "Gate2Suite",
]
