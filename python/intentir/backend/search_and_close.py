"""Target-neutral execution-domain search filtering and winner closure.

The core compiler never sees Source, Replay, Q, or test inputs.  This module is
the single outer controller that applies the fixed Gate2 comparison policy to
opaque executable runners.  It does not emit target code or add numeric
permissions to J.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from intentir.verify.gate1.model import SemanticArchive
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.model import ExecutionResult, PairwiseAgreement
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Suite,
    clone_gate2_values,
)
from intentir.verify.gate2.policy import ComparisonPolicy
from intentir.verify.gate2.scalar import ScalarLoweringSpec


SEARCH_AND_CLOSE_PROTOCOL = "intentir.backend.search-and-close.v1"
BACKEND_CLOSURE_PROTOCOL = "intentir.backend.execution-domain-closure.v1"
_STATE_ID = re.compile(r"^[A-Za-z0-9_.:+-]+$")


class BackendClosureError(ValueError):
    """Raised when Q/D closure cannot be established without guessing."""


class ObservationClass(str, Enum):
    HARD_PRESERVING = "hard_preserving"
    OBSERVATION_CHANGING = "observation_changing"


@dataclass(frozen=True)
class ConcreteLoweringState:
    """One concrete compiler state, never a semantic implementation family."""

    state_id: str
    lowering_path_kind: str
    transformation_trace: tuple[str, ...]
    schedule: Mapping[str, bool | int | float | str]
    observation_class: ObservationClass
    compiler_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise BackendClosureError("state_id must be non-empty")
        if (
            not isinstance(self.lowering_path_kind, str)
            or _STATE_ID.fullmatch(self.lowering_path_kind) is None
        ):
            raise BackendClosureError(
                "lowering_path_kind must be a stable identifier"
            )
        trace = tuple(self.transformation_trace)
        if not trace or any(
            not isinstance(item, str) or _STATE_ID.fullmatch(item) is None
            for item in trace
        ):
            raise BackendClosureError(
                "concrete lowering state requires a non-empty transformation trace"
            )
        object.__setattr__(self, "transformation_trace", trace)
        if not isinstance(self.observation_class, ObservationClass):
            object.__setattr__(
                self,
                "observation_class",
                ObservationClass(self.observation_class),
            )
        for name, value in self.schedule.items():
            if not isinstance(name, str) or not name:
                raise BackendClosureError("schedule keys must be non-empty strings")
            if not isinstance(value, (bool, int, float, str)):
                raise BackendClosureError("schedule values must be scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise BackendClosureError("schedule floats must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "lowering_path_kind": self.lowering_path_kind,
            "transformation_trace": list(self.transformation_trace),
            "schedule": dict(self.schedule),
            "observation_class": self.observation_class.value,
            "compiler_record": dict(self.compiler_record),
        }


@dataclass(frozen=True)
class ExecutionDomainCase:
    """Q-visible values plus an opaque harness context for one invocation."""

    case_id: str
    j: SemanticArchive
    values: Mapping[str, Any]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]
    opaque_context: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise BackendClosureError("execution-domain case_id must be non-empty")
        if not isinstance(self.j, SemanticArchive):
            raise BackendClosureError("execution-domain case requires canonical J")
        for label, bindings in (
            ("input", self.input_bindings),
            ("output", self.output_bindings),
        ):
            names = [name for name, _runtime in bindings]
            if names != sorted(names) or len(names) != len(set(names)):
                raise BackendClosureError(
                    f"execution-domain {label} bindings must be sorted and unique"
                )

    @classmethod
    def from_gate2_case(cls, case: Gate2Case) -> "ExecutionDomainCase":
        if not isinstance(case.refeval, ScalarLoweringSpec):
            raise BackendClosureError(
                "backend Q/D requires an executable canonical-J reference"
            )
        return cls(
            case_id=case.case_id,
            j=case.j,
            values=case.values,
            input_bindings=tuple(sorted(case.refeval.input_binding_map.items())),
            output_bindings=tuple(sorted(case.refeval.output_binding_map.items())),
            opaque_context=case,
        )

    @property
    def input_binding_map(self) -> dict[str, str]:
        return dict(self.input_bindings)

    @property
    def output_binding_map(self) -> dict[str, str]:
        return dict(self.output_bindings)

    def to_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_bindings": dict(self.input_bindings),
            "output_bindings": dict(self.output_bindings),
        }


@dataclass(frozen=True)
class ExecutionDomain:
    domain_id: str
    cases: tuple[ExecutionDomainCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain_id, str) or not self.domain_id:
            raise BackendClosureError("execution domain_id must be non-empty")
        cases = tuple(self.cases)
        if not cases:
            raise BackendClosureError("execution domain requires at least one case")
        ids = [item.case_id for item in cases]
        if len(ids) != len(set(ids)):
            raise BackendClosureError("execution domain has duplicate case_id values")
        object.__setattr__(self, "cases", cases)

    @classmethod
    def from_gate2_suite(
        cls, domain_id: str, suite: Gate2Suite
    ) -> "ExecutionDomain":
        return cls(
            domain_id,
            tuple(ExecutionDomainCase.from_gate2_case(case) for case in suite.cases),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "case_count": len(self.cases),
            "cases": [item.to_record() for item in self.cases],
        }


LegRunner = Callable[[ExecutionDomainCase, Mapping[str, Any]], ExecutionResult]
TargetRunner = Callable[
    [Any, ExecutionDomainCase, Mapping[str, Any]], ExecutionResult
]


@dataclass(frozen=True)
class OpaqueExecutionRunners:
    """Signature-driven runners; no Source/Replay text is exposed to search."""

    ref_j: LegRunner
    target: TargetRunner
    source: LegRunner | None = None
    replay: LegRunner | None = None


@dataclass(frozen=True)
class DirectEdgeEvidence:
    case_id: str
    edge: str
    passed: bool
    left_execution_ok: bool
    right_execution_ok: bool
    agreement: PairwiseAgreement | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "edge": self.edge,
            "passed": self.passed,
            "left_execution_ok": self.left_execution_ok,
            "right_execution_ok": self.right_execution_ok,
            "agreement": (
                self.agreement.to_json() if self.agreement is not None else None
            ),
            "error": self.error,
        }


@dataclass(frozen=True)
class CandidateQDResult:
    state: ConcreteLoweringState
    domain_id: str
    edges: tuple[DirectEdgeEvidence, ...]

    @property
    def passed(self) -> bool:
        return bool(self.edges) and all(item.passed for item in self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "domain_id": self.domain_id,
            "passed": self.passed,
            "edges": [item.to_dict() for item in self.edges],
        }


@dataclass(frozen=True)
class BackendClosureResult:
    target: Mapping[str, Any]
    q: Mapping[str, Any]
    d_search: Mapping[str, Any]
    d_cert: Mapping[str, Any]
    search_filter: CandidateQDResult
    winner: ConcreteLoweringState
    certification_edges: tuple[DirectEdgeEvidence, ...]
    protocol: str = BACKEND_CLOSURE_PROTOCOL

    @property
    def passed(self) -> bool:
        required = {"target_source", "target_replay", "target_ref_j"}
        observed = {item.edge for item in self.certification_edges}
        return (
            self.search_filter.passed
            and required <= observed
            and bool(self.certification_edges)
            and all(item.passed for item in self.certification_edges)
        )

    def _content(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "target": dict(self.target),
            "q": dict(self.q),
            "d_search": dict(self.d_search),
            "d_cert": dict(self.d_cert),
            "search_filter": self.search_filter.to_dict(),
            "winner": self.winner.to_dict(),
            "certification_edges": [
                item.to_dict() for item in self.certification_edges
            ],
            "passed": self.passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._content()


class SearchAndCloseController:
    """One generic Q/D mechanism shared by every target and every kernel."""

    def __init__(
        self,
        *,
        j: SemanticArchive,
        target: Mapping[str, Any],
        policy: ComparisonPolicy,
        d_search: ExecutionDomain,
        d_cert: ExecutionDomain,
        runners: OpaqueExecutionRunners,
    ) -> None:
        if not isinstance(j, SemanticArchive):
            raise BackendClosureError("SearchAndClose requires canonical J")
        if not isinstance(policy, ComparisonPolicy):
            raise BackendClosureError("SearchAndClose requires fixed Q")
        if not isinstance(d_search, ExecutionDomain) or not isinstance(
            d_cert, ExecutionDomain
        ):
            raise BackendClosureError("SearchAndClose requires D_search and D_cert")
        search_case_ids = {item.case_id for item in d_search.cases}
        cert_case_ids = {item.case_id for item in d_cert.cases}
        if search_case_ids & cert_case_ids:
            raise BackendClosureError(
                "D_search and D_cert must use disjoint case ids"
            )
        policy.validate_j(j)
        canonical_j = j.to_json()
        for domain in (d_search, d_cert):
            for case in domain.cases:
                if case.j.to_json() != canonical_j:
                    raise BackendClosureError(
                        "all Q/D cases must reference the same J"
                    )
        self.j = j
        self.target = dict(target)
        self.policy = policy
        self.d_search = d_search
        self.d_cert = d_cert
        self.runners = runners
        self._search_results: dict[str, CandidateQDResult] = {}

    @property
    def q_record(self) -> dict[str, Any]:
        return {"policy": self.policy.to_json()}

    @staticmethod
    def _execute(
        leg: str,
        runner: LegRunner,
        case: ExecutionDomainCase,
    ) -> ExecutionResult:
        try:
            result = runner(case, clone_gate2_values(case.values))
        except Exception as exc:  # runner failures are evidence, never permission
            return ExecutionResult(
                leg,
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"failure_class": "opaque_runner_exception"},
            )
        if not isinstance(result, ExecutionResult):
            return ExecutionResult(
                leg,
                False,
                error="opaque runner returned a non-ExecutionResult",
                metadata={"failure_class": "opaque_runner_protocol"},
            )
        return result

    @staticmethod
    def _execute_target(
        executable: Any,
        runner: TargetRunner,
        case: ExecutionDomainCase,
    ) -> ExecutionResult:
        try:
            result = runner(
                executable,
                case,
                clone_gate2_values(case.values),
            )
        except Exception as exc:
            return ExecutionResult(
                "target",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"failure_class": "target_runner_exception"},
            )
        if not isinstance(result, ExecutionResult):
            return ExecutionResult(
                "target",
                False,
                error="target runner returned a non-ExecutionResult",
                metadata={"failure_class": "target_runner_protocol"},
            )
        return result

    def _compare(
        self,
        *,
        case: ExecutionDomainCase,
        edge: str,
        left: ExecutionResult,
        right: ExecutionResult,
    ) -> DirectEdgeEvidence:
        if not left.ok or not right.ok:
            errors = [item.error for item in (left, right) if item.error]
            return DirectEdgeEvidence(
                case.case_id,
                edge,
                False,
                left.ok,
                right.ok,
                None,
                "; ".join(errors) or "execution failed without diagnostic",
            )
        try:
            agreement = compare_outputs(
                left.leg,
                left.outputs,
                right.leg,
                right.outputs,
                j=self.j,
                input_bindings=case.input_binding_map,
                output_bindings=case.output_binding_map,
                inputs=case.values,
                policy=self.policy,
            )
        except Exception as exc:
            return DirectEdgeEvidence(
                case.case_id,
                edge,
                False,
                True,
                True,
                None,
                f"{type(exc).__name__}: {exc}",
            )
        return DirectEdgeEvidence(
            case.case_id,
            edge,
            agreement.passed,
            True,
            True,
            agreement,
        )

    def filter_candidate(
        self,
        executable: Any,
        state: ConcreteLoweringState,
    ) -> CandidateQDResult:
        """Apply Q on D_search using only the direct target-to-RefJ edge."""

        edges: list[DirectEdgeEvidence] = []
        for case in self.d_search.cases:
            target = self._execute_target(executable, self.runners.target, case)
            reference = self._execute("ref_j", self.runners.ref_j, case)
            edges.append(
                self._compare(
                    case=case,
                    edge="target_ref_j",
                    left=target,
                    right=reference,
                )
            )
        result = CandidateQDResult(
            state,
            self.d_search.domain_id,
            tuple(edges),
        )
        self._search_results[state.state_id] = result
        return result

    def close_winner(
        self,
        executable: Any,
        state: ConcreteLoweringState,
    ) -> BackendClosureResult:
        """Certify one already-fixed winner by three independent direct edges."""

        search = self._search_results.get(state.state_id)
        if search is None or not search.passed:
            raise BackendClosureError(
                "winner must first pass target-to-RefJ filtering on D_search"
            )
        if self.runners.source is None or self.runners.replay is None:
            raise BackendClosureError(
                "formal backend closure requires opaque Source and Replay runners"
            )

        edges: list[DirectEdgeEvidence] = []
        for case in self.d_cert.cases:
            target = self._execute_target(executable, self.runners.target, case)
            source = self._execute("source", self.runners.source, case)
            replay = self._execute("replay", self.runners.replay, case)
            reference = self._execute("ref_j", self.runners.ref_j, case)
            for edge, right in (
                ("target_source", source),
                ("target_replay", replay),
                ("target_ref_j", reference),
            ):
                edges.append(
                    self._compare(
                        case=case,
                        edge=edge,
                        left=target,
                        right=right,
                    )
                )

        return BackendClosureResult(
            target=self.target,
            q=self.q_record,
            d_search=self.d_search.to_dict(),
            d_cert=self.d_cert.to_dict(),
            search_filter=search,
            winner=state,
            certification_edges=tuple(edges),
        )


__all__ = [
    "BACKEND_CLOSURE_PROTOCOL",
    "SEARCH_AND_CLOSE_PROTOCOL",
    "BackendClosureError",
    "BackendClosureResult",
    "CandidateQDResult",
    "ConcreteLoweringState",
    "DirectEdgeEvidence",
    "ExecutionDomain",
    "ExecutionDomainCase",
    "ObservationClass",
    "OpaqueExecutionRunners",
    "SearchAndCloseController",
]
