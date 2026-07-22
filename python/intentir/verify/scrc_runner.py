"""One active SCRC Gate runner joining Lift, TypeCheck, Gate1 and Gate2.

The runner is deliberately frontend-neutral.  Frontend compilers and source
aware providers are injected as small adapters; the lifecycle and terminal
status rules live here exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from .gate1.model import Gate1Artifact, Gate1Status, JNecessityStatus
from .gate1.orchestrate import Gate1ExecutionResult
from .gate2.model import Gate2SuiteResult
from .typecheck import TypeCheckReport, TypeCheckStatus, type_check


MAX_GATE1_ROUNDS = 3
MAX_PRE_REPLAY_REJECTIONS = 3


class MainStatus(str, Enum):
    SCRC_CERTIFIED = "SCRC_CERTIFIED"
    NO_CLOSURE = "NO_CLOSURE"
    GATE2_FAILED = "GATE2_FAILED"


class FeedbackKind(str, Enum):
    LIFT = "lift"
    TYPECHECK = "typecheck"
    GATE1 = "gate1"


class SCRCInfrastructureError(RuntimeError):
    """An adapter/environment failure, never relabeled as a candidate failure."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.evidence = dict(evidence or {})
        super().__init__(message)


class CandidateLiftError(RuntimeError):
    """One model output was not a complete parseable artifact; relift is allowed."""

    def __init__(
        self,
        diagnostics: Mapping[str, Any],
        *,
        physical_calls: int = 1,
    ) -> None:
        self.diagnostics = dict(diagnostics)
        if not isinstance(physical_calls, int) or physical_calls < 1:
            raise ValueError("Lift physical_calls must be positive")
        self.physical_calls = physical_calls
        super().__init__(str(self.diagnostics.get("diagnostic", "Lift rejected")))


@dataclass(frozen=True)
class HarnessBindingInstance:
    """One actual c0 and its frontend execution coordinates.

    The object stays outside the proof artifact and Replay packet.  Adapters
    may read ``gate1_case``/``gate2_context`` only when compiling or executing.
    """

    c_values: tuple[tuple[str, Any], ...]
    gate1_case: Any
    gate2_context: Any = None

    @classmethod
    def create(
        cls,
        c_values: Mapping[str, Any],
        *,
        gate1_case: Any,
        gate2_context: Any = None,
    ) -> "HarnessBindingInstance":
        if not isinstance(c_values, Mapping):
            raise TypeError("Harness c0 values must be a mapping")
        ordered = tuple(sorted(c_values.items()))
        if any(not isinstance(name, str) or not name for name, _ in ordered):
            raise ValueError("Harness c0 slot names must be non-empty strings")
        return cls(ordered, gate1_case, gate2_context)

    @property
    def values(self) -> dict[str, Any]:
        return dict(self.c_values)

    def validate_artifact(self, artifact: Gate1Artifact) -> None:
        if artifact.c.values != self.values:
            raise SCRCInfrastructureError(
                "Lift artifact C differs from the Harness-provided c0",
                evidence={"artifact": artifact.to_json()},
            )


@dataclass(frozen=True)
class SourceProgram:
    """Privileged Source material shared by Lift and frontend execution only."""

    text: str
    gate_source: Any

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("SourceProgram.text must be non-empty")


@dataclass(frozen=True)
class LiftFeedback:
    kind: FeedbackKind
    diagnostics: Mapping[str, Any]
    formal_round: int | None = None
    previous_artifact: Gate1Artifact | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "formal_round": self.formal_round,
            "diagnostics": dict(self.diagnostics),
            "previous_artifact": (
                self.previous_artifact.to_json()
                if self.previous_artifact is not None else None
            ),
        }


@dataclass(frozen=True)
class LiftCallResult:
    """Execution sidecar for one complete artifact; not a proof object."""

    artifact: Gate1Artifact
    physical_calls: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, Gate1Artifact):
            raise TypeError("LiftCallResult requires complete canonical (J,C,pi)")
        if not isinstance(self.physical_calls, int) or self.physical_calls < 1:
            raise ValueError("LiftCallResult.physical_calls must be positive")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("LiftCallResult.metadata must be a mapping")


@dataclass(frozen=True)
class SCRCRunRequest:
    """Source and one real Harness binding instance (the same c0)."""

    source: SourceProgram
    binding: HarnessBindingInstance
    frontend: str
    max_gate1_rounds: int = MAX_GATE1_ROUNDS
    max_pre_replay_rejections: int = MAX_PRE_REPLAY_REJECTIONS

    def __post_init__(self) -> None:
        if not isinstance(self.frontend, str) or not self.frontend:
            raise ValueError("SCRC request frontend must be a non-empty id")
        if not isinstance(self.source, SourceProgram):
            raise TypeError("SCRC request requires one SourceProgram")
        if not isinstance(self.binding, HarnessBindingInstance):
            raise TypeError("SCRC request requires one HarnessBindingInstance")
        if not isinstance(self.max_gate1_rounds, int) or self.max_gate1_rounds < 1:
            raise ValueError("max_gate1_rounds must be positive")
        if (
            not isinstance(self.max_pre_replay_rejections, int)
            or self.max_pre_replay_rejections < 0
        ):
            raise ValueError("max_pre_replay_rejections must be non-negative")


class LiftAdapter(Protocol):
    def lift(
        self,
        request: SCRCRunRequest,
        feedback: LiftFeedback | None,
    ) -> LiftCallResult: ...


class Gate1AttemptAdapter(Protocol):
    def run(
        self,
        artifact: Gate1Artifact,
        request: SCRCRunRequest,
    ) -> Gate1ExecutionResult: ...


class Gate2ExecutionAdapter(Protocol):
    def execute(
        self,
        artifact: Gate1Artifact,
        gate1: Gate1ExecutionResult,
        request: SCRCRunRequest,
    ) -> Gate2SuiteResult: ...


@dataclass(frozen=True)
class CandidateAttemptReport:
    lift_call: int
    formal_round: int | None
    typecheck_status: TypeCheckStatus
    artifact: Gate1Artifact | None = None
    typecheck: TypeCheckReport | None = None
    gate1: Gate1ExecutionResult | None = None
    feedback: LiftFeedback | None = None
    lift_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "lift_call": self.lift_call,
            "formal_round": self.formal_round,
            "typecheck_status": self.typecheck_status.value,
            "artifact": self.artifact.to_json() if self.artifact is not None else None,
            "typecheck": (
                self.typecheck.to_json() if self.typecheck is not None else None
            ),
            "gate1": self.gate1.to_json() if self.gate1 is not None else None,
            "feedback": self.feedback.to_json() if self.feedback is not None else None,
            "lift_metadata": dict(self.lift_metadata),
        }


@dataclass(frozen=True)
class CallBudget:
    lift_calls: int
    lift_physical_calls: int
    typecheck_calls: int
    formal_gate1_rounds: int
    replay_normal_calls: int
    replay_masked_calls: int
    replay_physical_calls: int
    gate2_cases: int
    source_executions: int
    replay_executions: int
    refeval_executions: int

    def to_json(self) -> dict[str, int]:
        return {
            "lift_calls": self.lift_calls,
            "lift_physical_calls": self.lift_physical_calls,
            "typecheck_calls": self.typecheck_calls,
            "formal_gate1_rounds": self.formal_gate1_rounds,
            "replay_normal_calls": self.replay_normal_calls,
            "replay_masked_calls": self.replay_masked_calls,
            "replay_physical_calls": self.replay_physical_calls,
            "gate2_cases": self.gate2_cases,
            "source_executions": self.source_executions,
            "replay_executions": self.replay_executions,
            "refeval_executions": self.refeval_executions,
        }


@dataclass(frozen=True)
class SCRCRunReport:
    main_status: MainStatus
    attempts: tuple[CandidateAttemptReport, ...]
    accepted_attempt: int | None
    gate2: Gate2SuiteResult | None
    j_necessity_status: JNecessityStatus | None
    call_budget: CallBudget

    def to_json(self) -> dict[str, Any]:
        return {
            "main_status": self.main_status.value,
            "attempts": [item.to_json() for item in self.attempts],
            "accepted_attempt": self.accepted_attempt,
            "gate2": self.gate2.to_json() if self.gate2 is not None else None,
            "j_necessity_status": (
                self.j_necessity_status.value
                if self.j_necessity_status is not None else None
            ),
            "call_budget": self.call_budget.to_json(),
        }


@dataclass(frozen=True)
class SCRCRunDependencies:
    lift: LiftAdapter
    gate1_factory: Callable[[int], Gate1AttemptAdapter]
    gate2: Gate2ExecutionAdapter
    type_checker: Callable[[Any, Any, Any], TypeCheckReport] = type_check


class SCRCGateRunner:
    """Run one bounded lifecycle; no local runner may own a second lifecycle."""

    def __init__(self, dependencies: SCRCRunDependencies) -> None:
        self.dependencies = dependencies

    @staticmethod
    def _gate1_feedback(
        result: Gate1ExecutionResult,
        formal_round: int,
        artifact: Gate1Artifact,
    ) -> LiftFeedback:
        normal = result.normal
        diagnostics: dict[str, Any] = {
            "status": result.status.value,
            "diagnostic": normal.diagnostic,
        }
        if normal.comparison is not None:
            diagnostics["observer_diff"] = normal.comparison.to_dict()
        return LiftFeedback(
            FeedbackKind.GATE1,
            diagnostics,
            formal_round,
            artifact,
        )

    def run(self, request: SCRCRunRequest) -> SCRCRunReport:
        attempts: list[CandidateAttemptReport] = []
        feedback: LiftFeedback | None = None
        last_artifact: Gate1Artifact | None = None
        lift_calls = 0
        lift_physical_calls = 0
        typecheck_calls = 0
        pre_replay_rejections = 0
        formal_rounds = 0
        replay_normal_calls = 0
        replay_masked_calls = 0
        replay_physical_calls = 0
        gate1_adapters: list[Gate1AttemptAdapter] = []

        while formal_rounds < request.max_gate1_rounds:
            lift_calls += 1
            try:
                lift_result = self.dependencies.lift.lift(request, feedback)
            except CandidateLiftError as exc:
                lift_physical_calls += exc.physical_calls
                pre_replay_rejections += 1
                feedback = LiftFeedback(
                    FeedbackKind.LIFT,
                    exc.diagnostics,
                    None,
                    last_artifact,
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.NOT_RUN,
                        feedback=feedback,
                    )
                )
                if pre_replay_rejections >= request.max_pre_replay_rejections:
                    break
                continue
            if not isinstance(lift_result, LiftCallResult):
                raise SCRCInfrastructureError(
                    "Lift adapter must return LiftCallResult"
                )
            lift_physical_calls += lift_result.physical_calls
            artifact = lift_result.artifact
            last_artifact = artifact
            request.binding.validate_artifact(artifact)
            typecheck_calls += 1
            report = self.dependencies.type_checker(
                artifact.j, artifact.c, artifact.pi
            )
            if not report.passed:
                pre_replay_rejections += 1
                feedback = LiftFeedback(
                    FeedbackKind.TYPECHECK,
                    {"typecheck": report.to_json()},
                    None,
                    artifact,
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.FAIL,
                        artifact=artifact,
                        typecheck=report,
                        feedback=feedback,
                        lift_metadata=lift_result.metadata,
                    )
                )
                if pre_replay_rejections >= request.max_pre_replay_rejections:
                    break
                continue

            formal_rounds += 1
            gate1_adapter = self.dependencies.gate1_factory(formal_rounds)
            if any(gate1_adapter is previous for previous in gate1_adapters):
                raise SCRCInfrastructureError(
                    "gate1_factory reused an attempt adapter; Replay must be fresh"
                )
            gate1_adapters.append(gate1_adapter)
            gate1 = gate1_adapter.run(artifact, request)
            replay_normal_calls += gate1.normal_calls
            replay_masked_calls += gate1.masked_j_calls
            for arm in (gate1.normal, gate1.masked_j):
                if arm is not None and arm.generation is not None:
                    replay_physical_calls += arm.generation.attempts
            if gate1.status is Gate1Status.COMPILE_FAILED:
                raise SCRCInfrastructureError(
                    "Gate1 source compilation failed; it is not a closure verdict",
                    evidence={
                        "formal_round": formal_rounds,
                        "artifact": artifact.to_json(),
                        "typecheck": report.to_json(),
                        "gate1": gate1.to_json(),
                    },
                )
            if gate1.status is not Gate1Status.REPLAY_CLOSED:
                feedback = self._gate1_feedback(
                    gate1, formal_rounds, artifact
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=formal_rounds,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        gate1=gate1,
                        feedback=feedback,
                        lift_metadata=lift_result.metadata,
                    )
                )
                continue

            if gate1.normal.generation is None:
                raise SCRCInfrastructureError(
                    "Gate1 reported closure without the exact Replay generation",
                    evidence={
                        "formal_round": formal_rounds,
                        "artifact": artifact.to_json(),
                        "typecheck": report.to_json(),
                        "gate1": gate1.to_json(),
                    },
                )

            attempts.append(
                CandidateAttemptReport(
                    lift_call=lift_calls,
                    formal_round=formal_rounds,
                    typecheck_status=TypeCheckStatus.PASS,
                    artifact=artifact,
                    typecheck=report,
                    gate1=gate1,
                    lift_metadata=lift_result.metadata,
                )
            )
            gate2 = self.dependencies.gate2.execute(artifact, gate1, request)
            if gate2.passed:
                main_status = MainStatus.SCRC_CERTIFIED
            else:
                main_status = MainStatus.GATE2_FAILED
            return SCRCRunReport(
                main_status=main_status,
                attempts=tuple(attempts),
                accepted_attempt=formal_rounds,
                gate2=gate2,
                j_necessity_status=(
                    gate1.j_necessity_status
                    if gate1.masked_j is not None else None
                ),
                call_budget=CallBudget(
                    lift_calls=lift_calls,
                    lift_physical_calls=lift_physical_calls,
                    typecheck_calls=typecheck_calls,
                    formal_gate1_rounds=formal_rounds,
                    replay_normal_calls=replay_normal_calls,
                    replay_masked_calls=replay_masked_calls,
                    replay_physical_calls=replay_physical_calls,
                    gate2_cases=len(gate2.cases),
                    source_executions=len(gate2.cases),
                    replay_executions=len(gate2.cases),
                    refeval_executions=len(gate2.cases),
                ),
            )

        return SCRCRunReport(
            main_status=MainStatus.NO_CLOSURE,
            attempts=tuple(attempts),
            accepted_attempt=None,
            gate2=None,
            j_necessity_status=None,
            call_budget=CallBudget(
                lift_calls=lift_calls,
                lift_physical_calls=lift_physical_calls,
                typecheck_calls=typecheck_calls,
                formal_gate1_rounds=formal_rounds,
                replay_normal_calls=replay_normal_calls,
                replay_masked_calls=replay_masked_calls,
                replay_physical_calls=replay_physical_calls,
                gate2_cases=0,
                source_executions=0,
                replay_executions=0,
                refeval_executions=0,
            ),
        )


__all__ = [
    "CallBudget",
    "CandidateLiftError",
    "CandidateAttemptReport",
    "FeedbackKind",
    "Gate1AttemptAdapter",
    "Gate2ExecutionAdapter",
    "HarnessBindingInstance",
    "LiftAdapter",
    "LiftCallResult",
    "LiftFeedback",
    "MainStatus",
    "MAX_GATE1_ROUNDS",
    "MAX_PRE_REPLAY_REJECTIONS",
    "SCRCGateRunner",
    "SCRCInfrastructureError",
    "SCRCRunDependencies",
    "SCRCRunReport",
    "SCRCRunRequest",
    "SourceProgram",
]
