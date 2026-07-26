"""One active SCRC Gate runner joining Lift, TypeCheck, Gate1 and Gate2.

The runner is deliberately frontend-neutral.  Frontend compilers and source
aware providers are injected as small adapters; the lifecycle and terminal
status rules live here exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Mapping, Protocol

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .gate1.model import Gate1Artifact, Gate1Status, JNecessityStatus
from .gate1.orchestrate import (
    Gate1ExecutionResult,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
)
from .gate2.model import Gate2SuiteResult
from .typecheck import TypeCheckReport, TypeCheckStatus, type_check


MAX_GATE1_ROUNDS = 3
MAX_PRE_REPLAY_REJECTIONS = 3
UNRESOLVED_GATE1_DIAGNOSTIC = "unresolved_gate1"
UNRESOLVED_GATE2_DIAGNOSTIC = "unresolved_gate2"
MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY = 8
MAX_GATE1_FEEDBACK_PATTERNS_PER_CATEGORY = 16
MAX_GATE1_FEEDBACK_NODE_WITNESSES = 10
MAX_GATE1_FEEDBACK_EDGES_PER_NODE = 6
MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS = 512
_OBSERVER_NODE_FINGERPRINT = re.compile(
    r"(?<=[\[:])[0-9a-f]{12,64}(?=\])"
)

SCRCProgressCallback = Callable[
    [str, int | None, int | None, str],
    None,
]


def _notify_progress(
    callback: SCRCProgressCallback | None,
    stage: str,
    attempt: int | None,
    formal_round: int | None,
    status: str,
) -> None:
    """Best-effort runtime observation that can never alter a gate verdict."""

    if callback is None:
        return
    try:
        callback(stage, attempt, formal_round, status)
    except Exception:
        # Progress is deliberately outside the certificate and evidence TCB.
        # Broken output streams or observers cannot reject or accept a case.
        return


def _feedback_item(value: Any) -> Any:
    """Remove non-actionable observer node ids from relift-only diagnostics."""

    if isinstance(value, str):
        return _OBSERVER_NODE_FINGERPRINT.sub("node", value)
    if isinstance(value, list):
        return [_feedback_item(item) for item in value]
    if isinstance(value, tuple):
        return [_feedback_item(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _feedback_item(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _observer_fingerprints(value: Any) -> tuple[str, ...]:
    """Return observer-only recursive node fingerprints in deterministic order."""

    if isinstance(value, str):
        return tuple(_OBSERVER_NODE_FINGERPRINT.findall(value))
    if isinstance(value, (list, tuple)):
        return tuple(
            fingerprint
            for item in value
            for fingerprint in _observer_fingerprints(item)
        )
    if isinstance(value, Mapping):
        return tuple(
            fingerprint
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            for fingerprint in _observer_fingerprints(item)
        )
    return ()


def _observer_node_aliases(
    differences: list[Mapping[str, Any]],
) -> dict[str, str]:
    """Pseudonymize node identity while preserving within-feedback incidence."""

    sides: dict[str, set[str]] = {}
    for difference in differences:
        side = (
            "source"
            if difference.get("kind") == "missing"
            else "replay"
            if difference.get("kind") == "unexpected"
            else "shared"
        )
        for fingerprint in _observer_fingerprints(difference.get("item")):
            sides.setdefault(fingerprint, set()).add(side)

    counters = {"source": 0, "replay": 0, "shared": 0}
    aliases: dict[str, str] = {}
    for fingerprint in sorted(sides):
        observed_sides = sides[fingerprint]
        only_side = (
            next(iter(observed_sides)) if len(observed_sides) == 1 else None
        )
        side = (
            only_side
            if only_side in {"source", "replay"}
            else "shared"
        )
        aliases[fingerprint] = f"{side}_node_{counters[side]}"
        counters[side] += 1
    return aliases


def _pseudonymize_observer_nodes(
    value: Any,
    aliases: Mapping[str, str],
) -> Any:
    """Replace opaque hashes with repeatable side-qualified local aliases."""

    if isinstance(value, str):
        return _OBSERVER_NODE_FINGERPRINT.sub(
            lambda match: aliases.get(match.group(0), "node"),
            value,
        )
    if isinstance(value, list):
        return [_pseudonymize_observer_nodes(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_pseudonymize_observer_nodes(item, aliases) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _pseudonymize_observer_nodes(item, aliases)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _observer_node_witnesses(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
    diagnostic_signatures: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Expose bounded incident-edge witnesses for the first divergent nodes."""

    incident: dict[str, list[Mapping[str, Any]]] = {}
    labels: dict[str, str] = {}
    primary_consumers: list[str] = []
    ordered_tokens: list[str] = []
    for difference in differences:
        item = difference.get("item")
        if (
            difference.get("category") != "dataflow"
            or not isinstance(item, (list, tuple))
            or len(item) != 4
            or item[0] != "edge"
        ):
            continue
        # Prefer the consumer: one changed consumer hash otherwise causes a
        # cascade of indistinguishable family-only missing/unexpected edges.
        ordered_endpoints = (item[2], item[1])
        edge_tokens: list[str] = []
        for endpoint_index, endpoint in enumerate(ordered_endpoints):
            endpoint_tokens = _observer_fingerprints(endpoint)
            for token in endpoint_tokens:
                if token not in edge_tokens:
                    edge_tokens.append(token)
                if token not in ordered_tokens:
                    ordered_tokens.append(token)
                if (
                    endpoint_index == 0
                    and token not in primary_consumers
                    and len(primary_consumers) < 2
                ):
                    primary_consumers.append(token)
                labels.setdefault(
                    token,
                    str(_pseudonymize_observer_nodes(endpoint, aliases)),
                )
        for token in edge_tokens:
            incident.setdefault(token, []).append(difference)

    # Start with the first source/replay consumer pair, then walk its upstream
    # producers before spending the witness budget on an unrelated
    # lexicographically early edge.  Recursive observer hashes make the
    # earliest producer divergence much more useful than a broad family list;
    # each selected witness still retains both incoming and outgoing edges.
    seeds = list(primary_consumers)
    seed_index = 0
    while seed_index < len(seeds) and len(seeds) < MAX_GATE1_FEEDBACK_NODE_WITNESSES:
        token = seeds[seed_index]
        seed_index += 1
        incoming_neighbors: list[str] = []
        for edge in incident.get(token, ()):
            item = edge.get("item")
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            producers = _observer_fingerprints(item[1])
            consumers = _observer_fingerprints(item[2])
            if token in consumers:
                for neighbor in producers:
                    if neighbor != token and neighbor not in incoming_neighbors:
                        incoming_neighbors.append(neighbor)
        for neighbor in incoming_neighbors:
            if neighbor in seeds:
                continue
            seeds.append(neighbor)
            if len(seeds) >= MAX_GATE1_FEEDBACK_NODE_WITNESSES:
                break
    for token in ordered_tokens:
        if len(seeds) >= MAX_GATE1_FEEDBACK_NODE_WITNESSES:
            break
        if token not in seeds:
            seeds.append(token)

    witnesses: list[dict[str, Any]] = []
    for token in seeds:
        edges = incident.get(token, ())
        witness: dict[str, Any] = {
            "node": labels[token],
            "incident_differences": [
                {
                    key: _pseudonymize_observer_nodes(edge[key], aliases)
                    for key in (
                        "kind",
                        "item",
                        "expected_count",
                        "actual_count",
                    )
                    if key in edge
                }
                for edge in edges[:MAX_GATE1_FEEDBACK_EDGES_PER_NODE]
            ],
            "truncated_count": max(
                0, len(edges) - MAX_GATE1_FEEDBACK_EDGES_PER_NODE
            ),
        }
        signatures = diagnostic_signatures.get(token, {})
        source_signature = signatures.get("source")
        replay_signature = signatures.get("replay")
        if source_signature is not None and source_signature == replay_signature:
            witness["local_signature"] = source_signature
        else:
            if source_signature is not None:
                witness["source_local_signature"] = source_signature
            if replay_signature is not None:
                witness["replay_local_signature"] = replay_signature
        witnesses.append(witness)
    return witnesses


def _observer_diagnostic_signatures(
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> dict[str, dict[str, str]]:
    """Index ephemeral normalized signatures by their opaque node digest."""

    result: dict[str, dict[str, str]] = {}
    for side, fingerprint in (
        ("source", source_fingerprint),
        ("replay", replay_fingerprint),
    ):
        entries = getattr(fingerprint, "diagnostic_node_signatures", ())
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
            ):
                continue
            signature = entry[1][:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
            for token in _observer_fingerprints(entry[0]):
                result.setdefault(token, {}).setdefault(side, signature)
    return result


def _observer_edge_endpoint(value: Any) -> str:
    """Collapse an observer node label to a hash-free structural family."""

    normalized = str(_feedback_item(value))
    if normalized.startswith("region["):
        payload = normalized.removeprefix("region[").split("]", 1)[0]
        return "region[" + payload.split(":", 1)[0] + "]"
    if normalized.startswith("arg["):
        return normalized
    return normalized.split("[", 1)[0]


def _observer_edge_pattern(difference: Mapping[str, Any]) -> tuple[str, ...] | None:
    item = difference.get("item")
    if (
        not isinstance(item, (list, tuple))
        or len(item) != 4
        or item[0] != "edge"
    ):
        return None
    return (
        str(difference.get("kind", "unknown")),
        _observer_edge_endpoint(item[1]),
        _observer_edge_endpoint(item[2]),
        str(_feedback_item(item[3])),
    )


def _compact_gate1_observer_diff(
    comparison: Any,
    *,
    source_fingerprint: Any | None = None,
    replay_fingerprint: Any | None = None,
) -> dict[str, Any]:
    """Keep full evidence on disk while sending Lift a bounded actionable diff."""

    raw = comparison.to_dict()
    differences = raw.get("differences")
    if not isinstance(differences, list):
        return _feedback_item(raw)
    mapped_differences = [
        difference for difference in differences if isinstance(difference, Mapping)
    ]
    node_aliases = _observer_node_aliases(mapped_differences)
    diagnostic_signatures = _observer_diagnostic_signatures(
        source_fingerprint,
        replay_fingerprint,
    )

    grouped: dict[str, dict[str, Any]] = {}
    for difference in mapped_differences:
        category = str(difference.get("category", "unknown"))
        group = grouped.setdefault(
            category,
            {
                "category": category,
                "difference_count": 0,
                "examples": [],
                "_edge_patterns": {},
            },
        )
        group["difference_count"] += 1
        pattern = _observer_edge_pattern(difference)
        if pattern is not None:
            patterns = group["_edge_patterns"]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        examples = group["examples"]
        if len(examples) >= MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY:
            continue
        examples.append(
            {
                key: _pseudonymize_observer_nodes(
                    difference[key], node_aliases
                )
                for key in ("kind", "item", "expected_count", "actual_count")
                if key in difference
            }
        )

    categories = []
    for category in sorted(grouped):
        group = grouped[category]
        raw_patterns = group.pop("_edge_patterns")
        if raw_patterns:
            group["edge_patterns"] = [
                {
                    "kind": pattern[0],
                    "source": pattern[1],
                    "target": pattern[2],
                    "role": pattern[3],
                    "count": count,
                }
                for pattern, count in sorted(
                    raw_patterns.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:MAX_GATE1_FEEDBACK_PATTERNS_PER_CATEGORY]
            ]
        if category == "dataflow":
            witnesses = _observer_node_witnesses(
                mapped_differences,
                node_aliases,
                diagnostic_signatures,
            )
            if witnesses:
                group["node_witnesses"] = witnesses
                group["node_alias_scope"] = "this_feedback_only"
        group["truncated_count"] = max(
            0, group["difference_count"] - len(group["examples"])
        )
        categories.append(group)
    return {
        "hard_pass": bool(raw.get("hard_pass", False)),
        "hard_passes": _feedback_item(raw.get("hard_passes", {})),
        "primary_difference_category": raw.get("primary_difference_category"),
        "categories": categories,
    }


class MainStatus(str, Enum):
    SCRC_CERTIFIED = "SCRC_CERTIFIED"
    NO_CLOSURE = "NO_CLOSURE"
    GATE2_FAILED = "GATE2_FAILED"


class Gate1SearchStatus(str, Enum):
    GATE1_CLOSED = "GATE1_CLOSED"
    NO_CLOSURE = "NO_CLOSURE"
    SUPERSEDED = "SUPERSEDED"


class FeedbackKind(str, Enum):
    LIFT = "lift"
    TYPECHECK = "typecheck"
    GATE1 = "gate1"
    GATE2 = "gate2"


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
    selection_rank: tuple[int | float, ...] | None = None

    def __post_init__(self) -> None:
        if self.selection_rank is None:
            return
        if self.kind is not FeedbackKind.GATE1:
            raise ValueError("selection_rank is private Gate1 search state")
        if (
            len(self.selection_rank) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in self.selection_rank
            )
        ):
            raise ValueError("selection_rank must be one Gate1 rank tuple")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "formal_round": self.formal_round,
            "diagnostics": dict(self.diagnostics),
            "previous_artifact": (
                self.previous_artifact.to_json()
                if self.previous_artifact is not None else None
            ),
            "selection_rank": (
                list(self.selection_rank)
                if self.selection_rank is not None else None
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
class SCRCGate1SearchReport:
    """Pre-Gate2 branch result used by bounded candidate fan-out."""

    status: Gate1SearchStatus
    attempts: tuple[CandidateAttemptReport, ...]
    accepted_attempt: int | None
    j_necessity_status: JNecessityStatus | None
    call_budget: CallBudget
    stop_reason: str | None = None
    winner_branch_id: int | None = None
    stop_observed_at: str | None = None

    @property
    def closed(self) -> bool:
        return self.status is Gate1SearchStatus.GATE1_CLOSED

    @property
    def superseded(self) -> bool:
        return self.status is Gate1SearchStatus.SUPERSEDED

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "attempts": [item.to_json() for item in self.attempts],
            "accepted_attempt": self.accepted_attempt,
            "j_necessity_status": (
                self.j_necessity_status.value
                if self.j_necessity_status is not None else None
            ),
            "call_budget": self.call_budget.to_json(),
            "stop_reason": self.stop_reason,
            "winner_branch_id": self.winner_branch_id,
            "stop_observed_at": self.stop_observed_at,
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

    @classmethod
    def _gate1_feedback(
        cls,
        result: Gate1ExecutionResult,
        formal_round: int,
        artifact: Gate1Artifact,
        prior_feedback: LiftFeedback | None = None,
    ) -> LiftFeedback:
        normal = result.normal
        diagnostics: dict[str, Any] = {
            "status": result.status.value,
            "diagnostic": normal.diagnostic,
        }
        if normal.comparison is not None:
            diagnostics["observer_diff"] = _compact_gate1_observer_diff(
                normal.comparison,
                source_fingerprint=result.source_fingerprint,
                replay_fingerprint=normal.fingerprint,
            )
        unresolved_gate2 = cls._unresolved_gate2(prior_feedback)
        if unresolved_gate2 is not None:
            diagnostics[UNRESOLVED_GATE2_DIAGNOSTIC] = unresolved_gate2
        return LiftFeedback(
            FeedbackKind.GATE1,
            diagnostics,
            formal_round,
            artifact,
            cls._failed_gate1_rank(result),
        )

    @staticmethod
    def _failed_gate1_rank(
        result: Gate1ExecutionResult,
    ) -> tuple[int | float, ...] | None:
        """Return Gate1's diagnostic-only failed-candidate ordering.

        The rank never accepts a candidate.  It only chooses which complete,
        already TypeChecked LLM artifact and its exact counterexample become
        the parent of the next Lift call.  A result without a comparable
        observer verdict keeps the ordinary latest-candidate repair path.
        """

        comparison = result.normal.comparison
        if comparison is None:
            return None
        rank = getattr(comparison, "rank_tuple", None)
        if rank is None:
            raw = comparison.to_dict()
            rank = raw.get("rank_tuple") if isinstance(raw, Mapping) else None
        if (
            not isinstance(rank, (list, tuple))
            or len(rank) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in rank
            )
        ):
            return None
        return tuple(rank)

    @staticmethod
    def _unresolved_gate1(
        feedback: LiftFeedback | None,
    ) -> dict[str, Any] | None:
        if feedback is None:
            return None
        if feedback.kind is FeedbackKind.GATE1:
            return {
                "formal_round": feedback.formal_round,
                "diagnostics": dict(feedback.diagnostics),
            }
        retained = feedback.diagnostics.get(UNRESOLVED_GATE1_DIAGNOSTIC)
        if not isinstance(retained, Mapping):
            return None
        formal_round = retained.get("formal_round")
        diagnostics = retained.get("diagnostics")
        if (
            not isinstance(formal_round, int)
            or formal_round < 1
            or not isinstance(diagnostics, Mapping)
        ):
            return None
        return {
            "formal_round": formal_round,
            "diagnostics": dict(diagnostics),
        }

    @staticmethod
    def _unresolved_gate2(
        feedback: LiftFeedback | None,
    ) -> dict[str, Any] | None:
        if feedback is None:
            return None
        if feedback.kind is FeedbackKind.GATE2:
            return {"diagnostics": dict(feedback.diagnostics)}
        retained = feedback.diagnostics.get(UNRESOLVED_GATE2_DIAGNOSTIC)
        if not isinstance(retained, Mapping):
            return None
        diagnostics = retained.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return None
        return {"diagnostics": dict(diagnostics)}

    @classmethod
    def _retain_gate1_diagnostic(
        cls,
        diagnostics: Mapping[str, Any],
        feedback: LiftFeedback | None,
    ) -> dict[str, Any]:
        combined = dict(diagnostics)
        unresolved = cls._unresolved_gate1(feedback)
        if unresolved is not None:
            combined[UNRESOLVED_GATE1_DIAGNOSTIC] = unresolved
        unresolved_gate2 = cls._unresolved_gate2(feedback)
        if unresolved_gate2 is not None:
            combined[UNRESOLVED_GATE2_DIAGNOSTIC] = unresolved_gate2
        return combined

    def search_gate1(
        self,
        request: SCRCRunRequest,
        *,
        initial_feedback: LiftFeedback | None = None,
        stop_requested: Callable[[], int | None] | None = None,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCGate1SearchReport:
        """Search one isolated branch through TypeCheck and Gate1 only.

        ``initial_feedback`` resumes from an externally preserved TypeCheck,
        Gate1, or J-side Gate2 failure.  It is context for the first new Lift
        call only: the runner never accepts, repairs, or otherwise promotes the
        previous artifact, and all budgets count only work performed by this
        search.
        """

        if initial_feedback is not None:
            if not isinstance(initial_feedback, LiftFeedback):
                raise TypeError("initial_feedback must be LiftFeedback")
            if initial_feedback.kind not in {
                FeedbackKind.TYPECHECK,
                FeedbackKind.GATE1,
                FeedbackKind.GATE2,
            }:
                raise ValueError(
                    "initial_feedback must describe a TypeCheck, Gate1, or "
                    "J-side Gate2 failure"
                )
            if initial_feedback.kind is FeedbackKind.GATE1:
                if (
                    not isinstance(initial_feedback.formal_round, int)
                    or initial_feedback.formal_round < 1
                ):
                    raise ValueError(
                        "initial Gate1 feedback requires a positive formal_round"
                    )
            elif initial_feedback.formal_round is not None:
                raise ValueError(
                    "initial TypeCheck/Gate2 feedback has no formal Gate1 round"
                )
            if initial_feedback.previous_artifact is None:
                raise ValueError(
                    "initial Gate1 feedback requires the exact previous artifact"
                )
            request.binding.validate_artifact(
                initial_feedback.previous_artifact
            )

        attempts: list[CandidateAttemptReport] = []
        feedback = initial_feedback
        last_artifact = (
            initial_feedback.previous_artifact
            if initial_feedback is not None else None
        )
        best_failed_feedback: LiftFeedback | None = None
        best_failed_rank: tuple[int | float, ...] | None = None
        if (
            initial_feedback is not None
            and initial_feedback.kind is FeedbackKind.GATE1
            and initial_feedback.selection_rank is not None
        ):
            best_failed_feedback = initial_feedback
            best_failed_rank = initial_feedback.selection_rank
        lift_calls = 0
        lift_physical_calls = 0
        typecheck_calls = 0
        pre_replay_rejections = 0
        formal_rounds = 0
        replay_normal_calls = 0
        replay_masked_calls = 0
        replay_physical_calls = 0
        gate1_adapters: list[Gate1AttemptAdapter] = []

        def budget() -> CallBudget:
            return CallBudget(
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
            )

        def superseded(observed_at: str) -> SCRCGate1SearchReport | None:
            winner = stop_requested() if stop_requested is not None else None
            if winner is None:
                return None
            return SCRCGate1SearchReport(
                status=Gate1SearchStatus.SUPERSEDED,
                attempts=tuple(attempts),
                accepted_attempt=None,
                j_necessity_status=None,
                call_budget=budget(),
                stop_reason="first_gate1_closure",
                winner_branch_id=winner,
                stop_observed_at=observed_at,
            )

        while formal_rounds < request.max_gate1_rounds:
            stopped = superseded("before_lift")
            if stopped is not None:
                return stopped
            lift_calls += 1
            _notify_progress(
                progress_callback,
                "lift",
                lift_calls,
                None,
                "started",
            )
            try:
                lift_result = self.dependencies.lift.lift(request, feedback)
            except CandidateLiftError as exc:
                lift_physical_calls += exc.physical_calls
                pre_replay_rejections += 1
                _notify_progress(
                    progress_callback,
                    "lift",
                    lift_calls,
                    None,
                    "rejected",
                )
                feedback = LiftFeedback(
                    FeedbackKind.LIFT,
                    self._retain_gate1_diagnostic(exc.diagnostics, feedback),
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
                stopped = superseded("after_lift")
                if stopped is not None:
                    return stopped
                if pre_replay_rejections >= request.max_pre_replay_rejections:
                    break
                continue
            if not isinstance(lift_result, LiftCallResult):
                raise SCRCInfrastructureError(
                    "Lift adapter must return LiftCallResult"
                )
            lift_physical_calls += lift_result.physical_calls
            _notify_progress(
                progress_callback,
                "lift",
                lift_calls,
                None,
                "finished",
            )
            artifact = canonicalize_gate1_artifact(lift_result.artifact)
            last_artifact = artifact
            stopped = superseded("after_lift")
            if stopped is not None:
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.NOT_RUN,
                        artifact=artifact,
                        lift_metadata=lift_result.metadata,
                    )
                )
                return superseded("after_lift") or stopped
            request.binding.validate_artifact(artifact)
            typecheck_calls += 1
            _notify_progress(
                progress_callback,
                "typecheck",
                lift_calls,
                None,
                "started",
            )
            report = self.dependencies.type_checker(
                artifact.j, artifact.c, artifact.pi
            )
            _notify_progress(
                progress_callback,
                "typecheck",
                lift_calls,
                None,
                report.status.value,
            )
            if not report.passed:
                pre_replay_rejections += 1
                feedback = LiftFeedback(
                    FeedbackKind.TYPECHECK,
                    self._retain_gate1_diagnostic(
                        {"typecheck": report.to_json()}, feedback
                    ),
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
                stopped = superseded("after_typecheck")
                if stopped is not None:
                    return stopped
                if pre_replay_rejections >= request.max_pre_replay_rejections:
                    break
                continue

            stopped = superseded("before_replay")
            if stopped is not None:
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        lift_metadata=lift_result.metadata,
                    )
                )
                return superseded("before_replay") or stopped

            formal_rounds += 1
            _notify_progress(
                progress_callback,
                "gate1",
                None,
                formal_rounds,
                "started",
            )
            gate1_adapter = self.dependencies.gate1_factory(formal_rounds)
            if any(gate1_adapter is previous for previous in gate1_adapters):
                raise SCRCInfrastructureError(
                    "gate1_factory reused an attempt adapter; Replay must be fresh"
                )
            gate1_adapters.append(gate1_adapter)
            gate1 = gate1_adapter.run(artifact, request)
            _notify_progress(
                progress_callback,
                "gate1",
                None,
                formal_rounds,
                gate1.status.value,
            )
            replay_normal_calls += gate1.normal_calls
            replay_masked_calls += gate1.masked_j_calls
            for arm in (gate1.normal, gate1.masked_j):
                if arm is None:
                    continue
                replay_physical_calls += (
                    arm.physical_calls
                    if arm.physical_calls
                    else (
                        arm.generation.attempts
                        if arm.generation is not None
                        else 0
                    )
                )
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
                source_observer_unsupported = (
                    gate1.normal.diagnostic or ""
                ).startswith(SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX)
                if source_observer_unsupported:
                    attempts.append(
                        CandidateAttemptReport(
                            lift_call=lift_calls,
                            formal_round=formal_rounds,
                            typecheck_status=TypeCheckStatus.PASS,
                            artifact=artifact,
                            typecheck=report,
                            gate1=gate1,
                            feedback=None,
                            lift_metadata=lift_result.metadata,
                        )
                    )
                    stopped = superseded("after_gate1")
                    if stopped is not None:
                        return stopped
                    return SCRCGate1SearchReport(
                        status=Gate1SearchStatus.NO_CLOSURE,
                        attempts=tuple(attempts),
                        accepted_attempt=None,
                        j_necessity_status=None,
                        call_budget=budget(),
                        stop_reason="observer_unsupported",
                        stop_observed_at="after_gate1",
                    )

                candidate_feedback = self._gate1_feedback(
                    gate1,
                    formal_rounds,
                    artifact,
                    feedback,
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=lift_calls,
                        formal_round=formal_rounds,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        gate1=gate1,
                        feedback=candidate_feedback,
                        lift_metadata=lift_result.metadata,
                    )
                )
                candidate_rank = self._failed_gate1_rank(gate1)
                if candidate_rank is None:
                    # Compile/generation failures have no comparable observer
                    # rank.  Repair that exact latest artifact and diagnostic.
                    feedback = candidate_feedback
                    last_artifact = artifact
                else:
                    if (
                        best_failed_rank is None
                        or candidate_rank > best_failed_rank
                    ):
                        best_failed_rank = candidate_rank
                        best_failed_feedback = candidate_feedback
                    assert best_failed_feedback is not None
                    feedback = best_failed_feedback
                    assert feedback.previous_artifact is not None
                    last_artifact = feedback.previous_artifact
                stopped = superseded("after_gate1")
                if stopped is not None:
                    return stopped
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
            return SCRCGate1SearchReport(
                status=Gate1SearchStatus.GATE1_CLOSED,
                attempts=tuple(attempts),
                accepted_attempt=formal_rounds,
                j_necessity_status=(
                    gate1.j_necessity_status
                    if gate1.masked_j is not None else None
                ),
                call_budget=budget(),
            )

        return SCRCGate1SearchReport(
            status=Gate1SearchStatus.NO_CLOSURE,
            attempts=tuple(attempts),
            accepted_attempt=None,
            j_necessity_status=None,
            call_budget=budget(),
        )

    @staticmethod
    def _accepted_search_attempt(
        search: SCRCGate1SearchReport,
    ) -> CandidateAttemptReport:
        if not search.closed or search.accepted_attempt is None:
            raise ValueError("Gate2 requires one Gate1-closed search result")
        accepted = [
            attempt
            for attempt in search.attempts
            if attempt.formal_round == search.accepted_attempt
        ]
        if len(accepted) != 1:
            raise SCRCInfrastructureError(
                "Gate1 search does not identify exactly one accepted attempt"
            )
        attempt = accepted[0]
        if (
            attempt.artifact is None
            or attempt.typecheck is None
            or not attempt.typecheck.passed
            or attempt.gate1 is None
            or not attempt.gate1.passed
            or attempt.gate1.normal.generation is None
        ):
            raise SCRCInfrastructureError(
                "Gate1 search accepted attempt lacks certified pre-Gate2 evidence"
            )
        return attempt

    def complete_gate2(
        self,
        request: SCRCRunRequest,
        search: SCRCGate1SearchReport,
    ) -> SCRCRunReport:
        """Run Gate2 exactly once for a winner selected without Gate2 evidence."""

        accepted = self._accepted_search_attempt(search)
        artifact = accepted.artifact
        typecheck_report = accepted.typecheck
        gate1 = accepted.gate1
        assert artifact is not None
        assert typecheck_report is not None
        assert gate1 is not None
        try:
            gate2 = self.dependencies.gate2.execute(artifact, gate1, request)
        except Exception as exc:
            raise SCRCInfrastructureError(
                "Gate2 adapter failed after Gate1 closure: "
                f"{type(exc).__name__}: {exc}",
                evidence={
                    "artifact": artifact.to_json(),
                    "typecheck": typecheck_report.to_json(),
                    "gate1": gate1.to_json(),
                    "attempts": [item.to_json() for item in search.attempts],
                    "formal_round": search.accepted_attempt,
                },
            ) from exc
        budget = search.call_budget
        case_count = len(gate2.cases)
        return SCRCRunReport(
            main_status=(
                MainStatus.SCRC_CERTIFIED
                if gate2.passed else MainStatus.GATE2_FAILED
            ),
            attempts=search.attempts,
            accepted_attempt=search.accepted_attempt,
            gate2=gate2,
            j_necessity_status=search.j_necessity_status,
            call_budget=CallBudget(
                lift_calls=budget.lift_calls,
                lift_physical_calls=budget.lift_physical_calls,
                typecheck_calls=budget.typecheck_calls,
                formal_gate1_rounds=budget.formal_gate1_rounds,
                replay_normal_calls=budget.replay_normal_calls,
                replay_masked_calls=budget.replay_masked_calls,
                replay_physical_calls=budget.replay_physical_calls,
                gate2_cases=case_count,
                source_executions=case_count,
                replay_executions=case_count,
                refeval_executions=case_count,
            ),
        )

    def run(
        self,
        request: SCRCRunRequest,
        *,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCRunReport:
        search = self.search_gate1(
            request,
            progress_callback=progress_callback,
        )
        if search.closed:
            return self.complete_gate2(request, search)
        return SCRCRunReport(
            main_status=MainStatus.NO_CLOSURE,
            attempts=search.attempts,
            accepted_attempt=None,
            gate2=None,
            j_necessity_status=None,
            call_budget=search.call_budget,
        )


__all__ = [
    "CallBudget",
    "CandidateLiftError",
    "CandidateAttemptReport",
    "FeedbackKind",
    "Gate1SearchStatus",
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
    "SCRCGate1SearchReport",
    "SCRCInfrastructureError",
    "SCRCProgressCallback",
    "SCRCRunDependencies",
    "SCRCRunReport",
    "SCRCRunRequest",
    "SourceProgram",
]
