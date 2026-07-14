"""SCRC Gate2: source/replay/lowering three-way execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from intentir.verify.gate1.compiler import KernelSource

from .compare import compare_outputs
from .lower2c import Lower2CRunner, Lower2CSpec
from .model import Gate2Result, Gate2Status, Tolerance
from .triton import TritonExecutionSpec, TritonRunner


@dataclass(frozen=True)
class Gate2Case:
    case_id: str
    values: Mapping[str, Any]
    triton: TritonExecutionSpec
    lowering: Lower2CSpec
    output_keys: tuple[str, ...]
    tolerances: Mapping[str, Tolerance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")


def _classify(source_replay: bool, source_lowering: bool, replay_lowering: bool) -> Gate2Status:
    pattern = (source_replay, source_lowering, replay_lowering)
    if pattern == (True, True, True):
        return Gate2Status.PASS
    if pattern == (True, False, False):
        return Gate2Status.LOWERING_DIVERGED
    if pattern == (False, True, False):
        return Gate2Status.REPLAY_DIVERGED
    if pattern == (False, False, True):
        return Gate2Status.LIFT_DIVERGED
    if pattern == (False, False, False):
        return Gate2Status.ALL_DIVERGED
    # With finite tolerances, approximate equality is not mathematically
    # transitive.  Do not force an attribution when two pairings pass and one
    # fails at a tolerance boundary.
    return Gate2Status.INCONCLUSIVE


class Gate2Runner:
    def __init__(
        self,
        *,
        source_runner: TritonRunner | None = None,
        replay_runner: TritonRunner | None = None,
        lowering_runner: Lower2CRunner | None = None,
    ) -> None:
        self.source_runner = source_runner or TritonRunner()
        self.replay_runner = replay_runner or TritonRunner()
        self.lowering_runner = lowering_runner or Lower2CRunner()

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
        lowering_result = self.lowering_runner.run(case.lowering, case.values)

        if lowering_result.unsupported:
            return Gate2Result(
                case.case_id,
                Gate2Status.LOWERING_UNSUPPORTED,
                source_result,
                replay_result,
                lowering_result,
            )
        if not source_result.ok or not replay_result.ok or not lowering_result.ok:
            return Gate2Result(
                case.case_id,
                Gate2Status.EXECUTION_FAILED,
                source_result,
                replay_result,
                lowering_result,
            )

        source_replay = compare_outputs(
            "source",
            source_result.outputs,
            "replay",
            replay_result.outputs,
            case.output_keys,
            case.tolerances,
        )
        source_lowering = compare_outputs(
            "source",
            source_result.outputs,
            "lowering",
            lowering_result.outputs,
            case.output_keys,
            case.tolerances,
        )
        replay_lowering = compare_outputs(
            "replay",
            replay_result.outputs,
            "lowering",
            lowering_result.outputs,
            case.output_keys,
            case.tolerances,
        )
        status = _classify(
            source_replay.passed,
            source_lowering.passed,
            replay_lowering.passed,
        )
        return Gate2Result(
            case.case_id,
            status,
            source_result,
            replay_result,
            lowering_result,
            source_replay,
            source_lowering,
            replay_lowering,
        )


__all__ = ["Gate2Case", "Gate2Runner"]
