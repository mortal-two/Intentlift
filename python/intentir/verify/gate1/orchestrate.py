"""Bounded SCRC Gate1 orchestration.

The orchestrator keeps the trust split explicit:

* LLM/static writers propose replay source;
* :class:`TritonCompiler` compiles in an isolated process without launching;
* :mod:`.ttir` and :mod:`.compare` alone decide structural closure;
* diagnosis scopes the next Lift update but cannot override the verdict;
* paired J interventions compare a fresh mutant replay against the closed
  normal replay fingerprint; prior replay source never re-enters the writer;
* masked-J keeps the actual Lift-produced pi while hiding J semantics, then
  runs one fresh, unrepaired replay as an adversarial topology-leakage check.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Protocol

from intentir.extraction.providers.client import ProviderError

from .compare import TTIRComparator, TTIRComparison
from .composer import (
    ComposeError,
    ComposeResult,
    assert_replay_plan_matches,
    compose_replay_plan,
)
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .diagnose import (
    Diagnosis,
    Responsibility,
    diagnose_comparison,
    diagnose_compile_failure,
)
from .evaluation import FailureCause
from .leakage import (
    LeakagePolicy,
    LeakageReport,
    scan_static_leakage,
)
from .interventions import (
    INTERVENTION_SEMANTIC_FAILURE_STATUSES,
    PAIRED_INTERVENTION_FAMILY_POLICY,
    InterventionProfileRecord,
    InterventionRecord,
    InterventionStatus,
    build_semantic_interventions,
    evaluate_semantic_intervention,
)
from .model import (
    Frontend,
    Gate1Artifact,
    Gate1Status,
    ParameterKind,
    ReplayPlan,
)
from .replay import (
    ReplayGeneration,
    ReplayWriter,
    restore_masked_replay_module,
)
from .rules import RuleRegistry
from .ttir import (
    TTIRFingerprint,
    TTIRParseError,
    fingerprint_ttir,
    runtime_argument_names,
)


MAX_EXACT_COMPOSED_PLAN_STEPS = 512


def _exact_composed_leakage_policy(
    base: LeakagePolicy,
    plan: ReplayPlan,
) -> tuple[LeakagePolicy, bool]:
    """Admit trusted deterministic expansion without enlarging pi freedom.

    The caller must first prove that the candidate plan exactly equals a fresh
    server-side composition.  Only the step-count budget may then grow to the
    trusted mechanical size; every literal/API/source-ngram budget remains
    unchanged.  An independent absolute limit still rejects pathological
    compositions.
    """
    deterministic_steps = len(plan.steps)
    if deterministic_steps > MAX_EXACT_COMPOSED_PLAN_STEPS:
        return replace(
            base,
            max_steps=MAX_EXACT_COMPOSED_PLAN_STEPS,
        ), False
    if deterministic_steps <= base.max_steps:
        return base, False
    return replace(base, max_steps=deterministic_steps), True


@dataclass(frozen=True)
class Gate1Config:
    """Bounded Gate1 budgets.

    ``max_relifts`` is retained as the historical public field name, but its
    value is the total number of artifact evaluation rounds including the
    initial Lift.  Therefore the maximum number of source-aware relift calls is
    ``max_relifts - 1``.  Normal replay has no inner repair loop: every failed
    generation/compile/interface/fingerprint result is returned to Lift and the
    next artifact round restarts the complete pipeline.
    """

    max_relifts: int = 5
    stable_closures: int = 1
    run_semantic_interventions: bool = True
    max_semantic_interventions: int = 1
    run_masked_ablation: bool = True
    # Compatibility name: exact composition is always a verdict condition;
    # this flag controls only whether detailed comparison records are emitted.
    run_composer_shadow: bool = True

    def __post_init__(self) -> None:
        if self.max_relifts < 1:
            raise ValueError("max_relifts must be positive")
        if self.stable_closures < 1:
            raise ValueError("stable_closures must be positive")
        if self.max_semantic_interventions < 0:
            raise ValueError("max_semantic_interventions must be non-negative")


@dataclass(frozen=True)
class CandidateRecord:
    relift_round: int
    generation_round: int
    generation: ReplayGeneration | None
    compile: CompileResult | None
    comparison: TTIRComparison | None
    diagnosis: Diagnosis | None
    adopted: bool = False
    regressed: bool = False
    specialization_evidence: Mapping[str, Any] | None = None
    specialization_hard_pass: bool | None = None
    effective_hard_difference_digest: str | None = None
    effective_hard_difference_categories: tuple[str, ...] = ()
    relift_guidance: ReliftGuidance | None = None
    relift_trigger: ReliftTrigger | None = None
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        if self.specialization_evidence is not None:
            normalized = json.loads(
                json.dumps(
                    self.specialization_evidence,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            object.__setattr__(self, "specialization_evidence", normalized)
        categories = tuple(self.effective_hard_difference_categories)
        if len(categories) != len(set(categories)):
            raise ValueError(
                "effective_hard_difference_categories must be unique"
            )
        object.__setattr__(
            self, "effective_hard_difference_categories", categories
        )

    @property
    def compiled(self) -> bool:
        return bool(self.compile and self.compile.ok)

    @property
    def overall_hard_pass(self) -> bool:
        return bool(
            self.comparison is not None
            and self.comparison.hard_pass
            and self.specialization_hard_pass is not False
        )

    @property
    def rank_tuple(
        self,
    ) -> tuple[int, int, int, int, int, float, int, int, int]:
        """Best-candidate order: compile > interface > A > B > score > C."""

        if self.comparison is None:
            return (
                int(self.compiled),
                0,
                0,
                0,
                0,
                0.0,
                -10**9,
                0,
                -10**9,
            )
        by_category = {
            component.category: component
            for component in self.comparison.components
        }
        a_categories = ("op_multiset", "dataflow", "reduction")
        hard_difference_count = sum(
            len(component.differences)
            for component in self.comparison.components
            if component.hard
        )
        schedule = by_category["schedule-only"]
        return (
            int(self.compiled),
            int(self.overall_hard_pass),
            int(by_category["interface"].passed),
            sum(int(by_category[name].passed) for name in a_categories),
            int(by_category["predicate_effect"].passed),
            round(self.comparison.score, 12),
            -hard_difference_count,
            int(schedule.passed),
            -len(schedule.differences),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "generation_round": self.generation_round,
            "generation": None
            if self.generation is None
            else {
                "prompt_sha256": self.generation.prompt_sha256,
                "requested_model": self.generation.requested_model,
                "generation_id": self.generation.generation_id,
                "code_sha256": self.generation.code_sha256,
                "response_model": self.generation.response_model,
                "response_id": self.generation.response_id,
                "masked": self.generation.masked,
                "ablation_kind": self.generation.ablation_kind,
            },
            "compile": None
            if self.compile is None
            else {
                "ok": self.compile.ok,
                "diagnostic": self.compile.diagnostic,
                "cache_key": self.compile.cache_key,
                "cache_hit": self.compile.cache_hit,
                "elapsed_ms": self.compile.elapsed_ms,
                "compiler": dict(self.compile.compiler),
                "target": dict(self.compile.target),
            },
            "comparison": None
            if self.comparison is None
            else self.comparison.to_dict(),
            "diagnosis": None
            if self.diagnosis is None
            else self.diagnosis.to_json(),
            "adopted": self.adopted,
            "regressed": self.regressed,
            "overall_hard_pass": self.overall_hard_pass,
            "specialization_evidence": self.specialization_evidence,
            "specialization_hard_pass": self.specialization_hard_pass,
            "effective_hard_difference_digest": (
                self.effective_hard_difference_digest
            ),
            "effective_hard_difference_categories": list(
                self.effective_hard_difference_categories
            ),
            "relift_guidance": (
                None
                if self.relift_guidance is None
                else self.relift_guidance.to_json()
            ),
            "relift_trigger": (
                None
                if self.relift_trigger is None
                else self.relift_trigger.value
            ),
            "artifact_digest": self.artifact_digest,
        }


class ComposerShadowStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    FAILED = "failed"


@dataclass(frozen=True)
class ComposerShadowRecord:
    """Compatibility report for materialized-versus-recomputed pi."""

    relift_round: int
    status: ComposerShadowStatus
    composed: ComposeResult | None = None
    materialized_plan_digest: str | None = None
    missing_steps: tuple[tuple[str, str | None, tuple[str, ...]], ...] = ()
    extra_steps: tuple[tuple[str, str | None, tuple[str, ...]], ...] = ()
    order_drift: bool = False
    diagnostic: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "status": self.status.value,
            "materialized_plan_digest": self.materialized_plan_digest,
            "composed_plan": (
                None if self.composed is None else self.composed.plan.to_json()
            ),
            "compose_provenance": (
                None
                if self.composed is None
                else self.composed.provenance.to_json()
            ),
            "missing_steps": [
                {"rule": rule, "target": target, "refs": list(refs)}
                for rule, target, refs in self.missing_steps
            ],
            "extra_steps": [
                {"rule": rule, "target": target, "refs": list(refs)}
                for rule, target, refs in self.extra_steps
            ],
            "order_drift": self.order_drift,
            "diagnostic": self.diagnostic,
        }


class ReliftTrigger(str, Enum):
    """Why the source-aware Lift agent is being invoked again.

    Every normal replay failure returns to the source-aware Lift agent. Static
    or masked-J leakage remains a distinct anti-leakage failure and is not
    silently converted into the same optimization loop.
    """

    REPLAY_GENERATION_FAILED = "replay_generation_failed"
    REPLAY_COMPILE_FAILED = "replay_compile_failed"
    REPLAY_EVALUATION_FAILED = "replay_evaluation_failed"
    NORMAL_NO_CLOSURE = "normal_no_closure"


RELIFT_GUIDANCE_POLICY = "deterministic_gate1_hard_diff_guidance_v2"
RELIFT_STRUCTURAL_GUIDANCE_POLICY = (
    "deterministic_scoped_fingerprint_structure_guidance_v2"
)


@dataclass(frozen=True)
class ReliftGuidance:
    """Deterministic semantic worklist for the next source-aware Lift.

    This is emitted only after a complete source/replay hard comparison.  It
    compresses low-level fingerprint deltas into auditable inspection targets
    without asserting that a TTIR operation maps one-to-one onto a J op.
    """

    hard_difference_digest: str
    failed_hard_categories: tuple[str, ...]
    confirmed_hard_categories: tuple[str, ...]
    category_guidance: tuple[Mapping[str, Any], ...]
    current_j_summary: Mapping[str, Any]
    decision_contract: tuple[str, ...]
    structural_guidance: Mapping[str, Any] | None = None
    policy: str = RELIFT_GUIDANCE_POLICY

    def __post_init__(self) -> None:
        normalized_categories = tuple(
            json.loads(
                json.dumps(
                    item,
                    sort_keys=True,
                    allow_nan=False,
                    default=str,
                )
            )
            for item in self.category_guidance
        )
        normalized_j = json.loads(
            json.dumps(
                self.current_j_summary,
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
        )
        object.__setattr__(self, "category_guidance", normalized_categories)
        object.__setattr__(self, "current_j_summary", normalized_j)
        if self.structural_guidance is not None:
            normalized_structural = json.loads(
                json.dumps(
                    self.structural_guidance,
                    sort_keys=True,
                    allow_nan=False,
                    default=str,
                )
            )
            object.__setattr__(
                self, "structural_guidance", normalized_structural
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "hard_difference_digest": self.hard_difference_digest,
            "failed_hard_categories": list(self.failed_hard_categories),
            "confirmed_hard_categories": list(
                self.confirmed_hard_categories
            ),
            "category_guidance": [
                dict(item) for item in self.category_guidance
            ],
            "current_j_summary": dict(self.current_j_summary),
            "structural_guidance": self.structural_guidance,
            "decision_contract": list(self.decision_contract),
        }


@dataclass(frozen=True)
class ReliftRequest:
    """Stable, auditable reason for one outer-loop semantic relift."""

    trigger: ReliftTrigger
    evaluation_round: int
    candidate_count: int
    compiled_candidate_count: int
    best_generation_round: int | None
    hard_difference_categories: tuple[str, ...]
    semantic_guidance: ReliftGuidance | None = None

    def __post_init__(self) -> None:
        if self.evaluation_round < 0:
            raise ValueError("evaluation_round must be non-negative")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0 <= self.compiled_candidate_count <= self.candidate_count:
            raise ValueError(
                "compiled_candidate_count must be within candidate_count"
            )
        if self.best_generation_round is not None and (
            self.best_generation_round < 0
        ):
            raise ValueError("best_generation_round must be non-negative")
        if len(set(self.hard_difference_categories)) != len(
            self.hard_difference_categories
        ):
            raise ValueError("hard_difference_categories must be unique")

    def to_json(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "evaluation_round": self.evaluation_round,
            "candidate_count": self.candidate_count,
            "compiled_candidate_count": self.compiled_candidate_count,
            "best_generation_round": self.best_generation_round,
            "hard_difference_categories": list(
                self.hard_difference_categories
            ),
            "semantic_guidance": (
                None
                if self.semantic_guidance is None
                else self.semantic_guidance.to_json()
            ),
        }


@dataclass(frozen=True)
class ReliftEvidence:
    """Source-free evidence exposed across the outer Relift boundary.

    ``CandidateRecord`` remains the internal Gate1 audit record and therefore
    retains the immutable replay generation and compiler result.  Relift must
    not receive either object: it sees only deterministic comparison output,
    a sanitized diagnosis, and bookkeeping needed to reason about progress.
    """

    relift_round: int
    generation_round: int
    generation_available: bool
    compiled: bool
    comparison: Mapping[str, Any] | None
    diagnosis: Mapping[str, Any] | None
    adopted: bool
    regressed: bool
    specialization_evidence: Mapping[str, Any] | None = None
    overall_hard_pass: bool = False
    effective_hard_difference_digest: str | None = None
    effective_hard_difference_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.relift_round < 0 or self.generation_round < 0:
            raise ValueError("relift evidence rounds must be non-negative")
        categories = tuple(self.effective_hard_difference_categories)
        if len(categories) != len(set(categories)):
            raise ValueError(
                "effective_hard_difference_categories must be unique"
            )
        object.__setattr__(
            self, "effective_hard_difference_categories", categories
        )
        # Normalize to plain JSON values.  This prevents a caller from hiding
        # a ReplayGeneration/CompileResult object behind a custom Mapping.
        for field_name in (
            "comparison",
            "diagnosis",
            "specialization_evidence",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = json.loads(
                json.dumps(value, allow_nan=False, sort_keys=True)
            )
            object.__setattr__(self, field_name, normalized)

    @classmethod
    def from_candidate(cls, candidate: CandidateRecord) -> "ReliftEvidence":
        return cls(
            relift_round=candidate.relift_round,
            generation_round=candidate.generation_round,
            generation_available=candidate.generation is not None,
            compiled=candidate.compiled,
            comparison=(
                None
                if candidate.comparison is None
                else candidate.comparison.to_dict()
            ),
            diagnosis=(
                None
                if candidate.diagnosis is None
                else candidate.diagnosis.to_json()
            ),
            adopted=candidate.adopted,
            regressed=candidate.regressed,
            specialization_evidence=_specialization_relift_view(
                candidate.specialization_evidence
            ),
            overall_hard_pass=candidate.overall_hard_pass,
            effective_hard_difference_digest=(
                candidate.effective_hard_difference_digest
            ),
            effective_hard_difference_categories=(
                candidate.effective_hard_difference_categories
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "generation_round": self.generation_round,
            "generation_available": self.generation_available,
            "compiled": self.compiled,
            "comparison": self.comparison,
            "diagnosis": self.diagnosis,
            "adopted": self.adopted,
            "regressed": self.regressed,
            "specialization_evidence": self.specialization_evidence,
            "overall_hard_pass": self.overall_hard_pass,
            "effective_hard_difference_digest": (
                self.effective_hard_difference_digest
            ),
            "effective_hard_difference_categories": list(
                self.effective_hard_difference_categories
            ),
        }

    def to_prompt_json(self) -> dict[str, Any]:
        """Return a bounded source-aware Lift view without duplicate raw diff.

        The immutable audit record keeps the complete comparator payload.  The
        Relift prompt already receives the exact score-free worklist in
        ``ReliftRequest.semantic_guidance``; repeating every low-level edge in
        history dilutes that signal and grows linearly with family size.
        """

        comparison_summary: dict[str, Any] | None = None
        if self.comparison is not None:
            components = self.comparison.get("components")
            component_items = (
                components if isinstance(components, list) else []
            )
            comparison_summary = {
                "hard_pass": self.comparison.get("hard_pass"),
                "primary_difference_category": self.comparison.get(
                    "primary_difference_category"
                ),
                "hard_passes": self.comparison.get("hard_passes"),
                "active_categories": self.comparison.get(
                    "active_categories"
                ),
                "inactive_categories": self.comparison.get(
                    "inactive_categories"
                ),
                "difference_count": len(
                    self.comparison.get("differences", [])
                    if isinstance(
                        self.comparison.get("differences"), list
                    )
                    else []
                ),
                "components": [
                    {
                        "category": component.get("category"),
                        "applicable": component.get("applicable"),
                        "passed": component.get("passed"),
                        "difference_count": len(
                            component.get("differences", [])
                            if isinstance(
                                component.get("differences"), list
                            )
                            else []
                        ),
                    }
                    for component in component_items
                    if isinstance(component, Mapping)
                    and component.get("hard") is True
                ],
            }
        return {
            "relift_round": self.relift_round,
            "generation_round": self.generation_round,
            "generation_available": self.generation_available,
            "compiled": self.compiled,
            "comparison_summary": comparison_summary,
            "diagnosis": self.diagnosis,
            "adopted": self.adopted,
            "regressed": self.regressed,
            "specialization_evidence": self.specialization_evidence,
            "overall_hard_pass": self.overall_hard_pass,
            "effective_hard_difference_digest": (
                self.effective_hard_difference_digest
            ),
            "effective_hard_difference_categories": list(
                self.effective_hard_difference_categories
            ),
        }


def _specialization_relift_view(
    evidence: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Bound family history before it crosses into source-aware Relift.

    Exact grouped hard differences are carried by ``ReliftGuidance``.  The
    history view preserves member identity/stage/pass constraints but omits
    raw per-profile comparator payloads and their internal node digests.
    """

    if evidence is None:
        return None
    records = evidence.get("records")
    items = records if isinstance(records, list) else []
    return {
        key: evidence.get(key)
        for key in (
            "policy",
            "diff_policy",
            "family_id",
            "primary_profile_id",
            "fixed_c",
            "status",
            "all_passed",
            "hard_comparison_complete",
            "failed_profile_ids",
            "family_hard_difference_digest",
        )
    } | {
        "records": [
            {
                key: item.get(key)
                for key in (
                    "profile_id",
                    "passed",
                    "failure_stage",
                    "source_compile_ok",
                    "replay_compile_ok",
                    "diagnostic",
                )
            }
            for item in items
            if isinstance(item, Mapping)
        ]
    }


def _relift_request_for_round(
    evaluation_round: int,
    records: tuple[CandidateRecord, ...],
    best: CandidateRecord | None,
) -> ReliftRequest:
    if not records:
        raise ValueError(
            "semantic relift requires at least one immutable replay observation"
        )
    hard_categories: tuple[str, ...] = ()
    if best is not None:
        hard_categories = best.effective_hard_difference_categories
        if not hard_categories and best.comparison is not None:
            hard_categories = tuple(
                component.category
                for component in best.comparison.components
                if component.hard and not component.passed
            )
    explicit_triggers = [
        record.relift_trigger
        for record in records
        if record.relift_trigger is not None
    ]
    if explicit_triggers:
        trigger = explicit_triggers[-1]
    elif any(record.comparison is not None for record in records):
        trigger = ReliftTrigger.NORMAL_NO_CLOSURE
    elif any(record.compiled for record in records):
        trigger = ReliftTrigger.REPLAY_EVALUATION_FAILED
    elif any(record.generation is not None for record in records):
        trigger = ReliftTrigger.REPLAY_COMPILE_FAILED
    else:
        trigger = ReliftTrigger.REPLAY_GENERATION_FAILED
    return ReliftRequest(
        trigger=trigger,
        evaluation_round=evaluation_round,
        candidate_count=len(records),
        compiled_candidate_count=sum(record.compiled for record in records),
        best_generation_round=(
            None if best is None else best.generation_round
        ),
        hard_difference_categories=hard_categories,
    )


class Relifter(Protocol):
    def relift(
        self,
        artifact: Gate1Artifact,
        *,
        source_fingerprint: TTIRFingerprint,
        history: tuple[ReliftEvidence, ...],
        request: ReliftRequest,
    ) -> Gate1Artifact: ...


GATE1_STAGNATION_POLICY = "exact_artifact_and_hard_diff_v1"
GATE1_TERMINATION_POLICY = "structured_no_closure_termination_v1"


@dataclass(frozen=True)
class StagnationEvidence:
    """Deterministic evidence that outer Relift stopped making progress.

    Stagnation is observed only after two complete normal replay attempts.  It
    therefore cannot turn an unchanged Lift response into a pre-writer reject,
    and it never hashes replay source, fuzzy scores, prose diagnostics, or raw
    compiler output.
    """

    first_relift_round: int
    repeated_relift_round: int
    artifact_digest: str
    hard_difference_digest: str
    hard_difference_categories: tuple[str, ...]
    policy: str = GATE1_STAGNATION_POLICY
    consecutive_observations: int = 2

    def __post_init__(self) -> None:
        if self.first_relift_round < 0:
            raise ValueError("first_relift_round must be non-negative")
        if self.repeated_relift_round <= self.first_relift_round:
            raise ValueError(
                "repeated_relift_round must follow first_relift_round"
            )
        if self.consecutive_observations < 2:
            raise ValueError("stagnation needs at least two observations")

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "first_relift_round": self.first_relift_round,
            "repeated_relift_round": self.repeated_relift_round,
            "artifact_digest": self.artifact_digest,
            "hard_difference_digest": self.hard_difference_digest,
            "hard_difference_categories": list(
                self.hard_difference_categories
            ),
            "consecutive_observations": self.consecutive_observations,
        }


class Gate1TerminationReason(str, Enum):
    """Deterministic reason why Gate1 ended without replay closure."""

    INTERVENTION_EVIDENCE_INCOMPLETE = (
        "intervention_evidence_incomplete"
    )
    MASKED_EVIDENCE_INCOMPLETE = "masked_evidence_incomplete"
    REPLAY_PROVIDER_UNAVAILABLE = "replay_provider_unavailable"
    DETERMINISTIC_STAGNATION = "deterministic_stagnation"
    RELIFTER_UNAVAILABLE = "relifter_unavailable"
    RELIFT_BUDGET_EXHAUSTED = "relift_budget_exhausted"
    RELIFT_FAILED = "relift_failed"


@dataclass(frozen=True)
class Gate1Termination:
    """Source-free, score-free audit record for one non-closure exit."""

    reason: Gate1TerminationReason
    artifact_round: int
    normal_candidate_count: int
    remaining_artifact_round_budget: int
    hard_difference_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reason, Gate1TerminationReason):
            object.__setattr__(
                self,
                "reason",
                Gate1TerminationReason(self.reason),
            )
        if self.artifact_round < 0:
            raise ValueError("artifact_round must be non-negative")
        if self.normal_candidate_count < 0:
            raise ValueError("normal_candidate_count must be non-negative")
        if self.remaining_artifact_round_budget < 0:
            raise ValueError(
                "remaining_artifact_round_budget must be non-negative"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "artifact_round": self.artifact_round,
            "normal_candidate_count": self.normal_candidate_count,
            "remaining_artifact_round_budget": (
                self.remaining_artifact_round_budget
            ),
            "hard_difference_categories": list(
                self.hard_difference_categories
            ),
        }


@dataclass(frozen=True)
class Gate1ExecutionResult:
    status: Gate1Status
    artifact: Gate1Artifact
    source_compile: CompileResult | None
    source_fingerprint: TTIRFingerprint | None
    candidates: tuple[CandidateRecord, ...] = ()
    best_candidate: CandidateRecord | None = None
    leakage_report: LeakageReport | None = None
    masked_compile: CompileResult | None = None
    masked_comparison: TTIRComparison | None = None
    masked_outcome: str | None = None
    diagnostics: tuple[str, ...] = ()
    masked_generation: ReplayGeneration | None = None
    interventions: tuple[InterventionRecord, ...] = ()
    anti_leakage_basis: str = "compose_equality"
    composer_shadow: tuple[ComposerShadowRecord, ...] = ()
    stagnation: StagnationEvidence | None = None
    termination: Gate1Termination | None = None
    masked_family_evidence: Mapping[str, Any] | None = None

    @property
    def replay_closed(self) -> bool:
        return self.status is Gate1Status.REPLAY_CLOSED

    @property
    def termination_reason(self) -> str:
        if self.termination is not None:
            return self.termination.reason.value
        if self.status is Gate1Status.NO_CLOSURE:
            # Compatibility for callers constructing historical results.  The
            # production runner always supplies a structured reason.
            return "unspecified_no_closure"
        return self.status.value

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "replay_closed": self.replay_closed,
            "artifact": self.artifact.to_json(),
            "source_compile": None
            if self.source_compile is None
            else {
                "ok": self.source_compile.ok,
                "diagnostic": self.source_compile.diagnostic,
                "cache_key": self.source_compile.cache_key,
                "cache_hit": self.source_compile.cache_hit,
                "elapsed_ms": self.source_compile.elapsed_ms,
                "compiler": dict(self.source_compile.compiler),
                "target": dict(self.source_compile.target),
            },
            "source_fingerprint": None
            if self.source_fingerprint is None
            else self.source_fingerprint.to_dict(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "best_candidate_index": None
            if self.best_candidate is None
            else list(self.candidates).index(self.best_candidate),
            "leakage_report": None
            if self.leakage_report is None
            else self.leakage_report.to_json(),
            "masked_compile": None
            if self.masked_compile is None
            else {
                "ok": self.masked_compile.ok,
                "diagnostic": self.masked_compile.diagnostic,
            },
            "masked_generation": None
            if self.masked_generation is None
            else {
                "prompt_sha256": self.masked_generation.prompt_sha256,
                "requested_model": self.masked_generation.requested_model,
                "generation_id": self.masked_generation.generation_id,
                "code_sha256": self.masked_generation.code_sha256,
                "response_model": self.masked_generation.response_model,
                "response_id": self.masked_generation.response_id,
                "masked": self.masked_generation.masked,
                "ablation_kind": self.masked_generation.ablation_kind,
            },
            "masked_comparison": None
            if self.masked_comparison is None
            else self.masked_comparison.to_dict(),
            "masked_outcome": self.masked_outcome,
            "masked_family_evidence": self.masked_family_evidence,
            "interventions": [item.to_json() for item in self.interventions],
            "anti_leakage_basis": self.anti_leakage_basis,
            "composer_shadow": [item.to_json() for item in self.composer_shadow],
            "stagnation": (
                None if self.stagnation is None else self.stagnation.to_json()
            ),
            "termination_reason": self.termination_reason,
            "termination": (
                None if self.termination is None else self.termination.to_json()
            ),
            "diagnostics": list(self.diagnostics),
        }


def _replay_plan_digest(plan: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            plan.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_artifact_digest(composed: ComposeResult) -> str:
    """Hash only canonical replay-bearing Lift content.

    ``ComposeProvenance`` already contains deterministic digests for J,
    interface, frozen C.schedule, editable C.strategy, the selected rule
    registry, and canonical pi.  Location/provenance/model metadata are
    deliberately excluded because they cannot change the writer packet.
    """

    provenance = composed.provenance
    payload = {
        "digest_schema": "scrc-artifact-semantic-v1",
        "composer_version": provenance.composer_version,
        "j_digest": provenance.j_digest,
        "interface_digest": provenance.interface_digest,
        "schedule_digest": provenance.schedule_digest,
        "strategy_digest": provenance.strategy_digest,
        "registry_digest": provenance.registry_digest,
        "plan_digest": provenance.plan_digest,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_artifact_digest(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> str:
    """Return the canonical semantic identity of a complete Lift artifact.

    The helper first verifies that the materialized pi still equals the
    deterministic J+C+rule-library composition.  Callers therefore cannot use
    it to bless an incomplete or drifted artifact merely by hashing its JSON.
    """

    return _canonical_artifact_digest(
        assert_replay_plan_matches(artifact, registry)
    )


def _hard_difference_digest(comparison: TTIRComparison) -> str:
    """Hash the exact hard comparator state without score or prose."""

    components: list[dict[str, Any]] = []
    for component in sorted(
        (item for item in comparison.components if item.hard),
        key=lambda item: item.category,
    ):
        differences = [
            {
                "kind": difference.kind,
                "item": difference.item,
                "expected_count": difference.expected_count,
                "actual_count": difference.actual_count,
            }
            for difference in component.differences
        ]
        differences.sort(
            key=lambda value: json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            )
        )
        components.append(
            {
                "category": component.category,
                "applicable": component.applicable,
                "passed": component.passed,
                "differences": differences,
            }
        )
    payload = {
        "signature_schema": "gate1-hard-diff-v1",
        "hard_pass": comparison.hard_pass,
        "components": components,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


_FINGERPRINT_NODE_DIGEST_RE = re.compile(r"\[[0-9a-f]{20}\]$")

_RELIFT_CATEGORY_TARGETS: dict[str, dict[str, Any]] = {
    "interface": {
        "inspect": (
            "j.intent.inputs/outputs",
            "j.boundary.output_mappings",
            "j.boundary.aliases/effects",
        ),
        "strategy_families": (),
        "semantic_question": (
            "Does J bind the frozen ABI to the intended whole-tensor inputs, "
            "outputs, aliases and effects? Interface/C.schedule stay frozen."
        ),
    },
    "op_multiset": {
        "inspect": (
            "j.intent.body[*].op/attrs",
            "j.intent.loop.body/carry/yield",
            "j.boundary.output_mappings",
        ),
        "strategy_families": (),
        "semantic_question": (
            "Re-derive the whole-tensor algebra from SOURCE and compare it "
            "with J. A lowering-only TTIR op count is not by itself a J op."
        ),
    },
    "dataflow": {
        "inspect": (
            "j.intent.body[*].operands/results",
            "j.intent.loop.carry/yield/scan",
            "j.boundary.output_mappings",
            "j.scopes captures/definitions",
            "j.storage lifetime",
        ),
        "strategy_families": (
            "address_association",
            "program_coordinate_width",
        ),
        "semantic_question": (
            "Does J preserve semantic producer/consumer dependencies, loop "
            "carry and final output values? If yes, check only an already "
            "reviewed association strategy; never copy source topology into J."
        ),
    },
    "reduction": {
        "inspect": (
            "j.intent reduce/loop nodes",
            "j.boundary.reductions identity/masked_contribution/tile_axis",
            "j.intent.loop.carry/yield",
        ),
        "strategy_families": ("reduction_call_shape",),
        "semantic_question": (
            "Check reducer semantics, axis, identity/fill, carried state and "
            "finalized value before considering a reviewed call-shape strategy."
        ),
    },
    "predicate_effect": {
        "inspect": (
            "j.boundary.accesses predicate/fill/index",
            "j.boundary.effects",
            "j.boundary.aliases",
            "j.storage init/last-use/escape",
        ),
        "strategy_families": (
            "mask_materialization",
            "address_association",
        ),
        "semantic_question": (
            "Check observable read/write/atomic effects, masks, fill values, "
            "aliasing and lifetime. Use only reviewed strategy choices after "
            "J semantics are confirmed."
        ),
    },
}


def _semantic_difference_item(category: str, item: Any) -> Any:
    """Remove internal node hashes while retaining exact edge roles."""

    if (
        category == "dataflow"
        and isinstance(item, (list, tuple))
        and len(item) == 4
        and item[0] == "edge"
    ):
        return [
            "edge",
            _FINGERPRINT_NODE_DIGEST_RE.sub("", str(item[1])),
            _FINGERPRINT_NODE_DIGEST_RE.sub("", str(item[2])),
            item[3],
        ]
    return item


def _group_semantic_differences(component: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for difference in component.differences:
        normalized_item = _semantic_difference_item(
            component.category, difference.item
        )
        key = json.dumps(
            {"kind": difference.kind, "item": normalized_item},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        entry = grouped.setdefault(
            key,
            {
                "kind": difference.kind,
                "item": normalized_item,
                "expected_count": 0,
                "actual_count": 0,
            },
        )
        entry["expected_count"] += difference.expected_count
        entry["actual_count"] += difference.actual_count
    return [grouped[key] for key in sorted(grouped)]


def _current_j_summary(artifact: Gate1Artifact) -> dict[str, Any]:
    op_counts: Counter[str] = Counter()
    loop_carry_count = 0

    def visit(ops: Any) -> None:
        nonlocal loop_carry_count
        for op in ops:
            op_counts[str(op.op)] += 1
            loop_carry_count += len(getattr(op, "carry", ()))
            visit(getattr(op, "body", ()))

    visit(artifact.j.intent.body)
    return {
        "intent_op_multiset": dict(sorted(op_counts.items())),
        "intent_loop_count": op_counts.get("loop", 0),
        "loop_carry_count": loop_carry_count,
        "boundary": {
            "domains": len(artifact.j.boundary.domains),
            "accesses": [
                {
                    "kind": access.kind.value,
                    "has_predicate": access.predicate is not None,
                    "has_fill": access.fill_value is not None,
                    "has_index": access.index_expression is not None,
                }
                for access in artifact.j.boundary.accesses
            ],
            "reductions": [
                reduction.to_json()
                for reduction in artifact.j.boundary.reductions
            ],
            "broadcast_count": len(artifact.j.boundary.broadcasts),
            "alias_count": len(artifact.j.boundary.aliases),
            "effect_count": len(artifact.j.boundary.effects),
            "output_mapping_count": len(
                artifact.j.boundary.output_mappings
            ),
        },
        "scope_kinds": dict(
            sorted(Counter(scope.kind.value for scope in artifact.j.scopes).items())
        ),
        "storage_mutability": dict(
            sorted(
                Counter(
                    storage.mutability.value for storage in artifact.j.storage
                ).items()
            )
        ),
        "strategy_bindings": [
            binding.to_json() for binding in artifact.strategy.bindings
        ],
    }


def _fingerprint_structure_summary(
    fingerprint: TTIRFingerprint,
) -> dict[str, Any]:
    """Project a fingerprint onto compact semantic structure coordinates."""

    effect_counts: Counter[str] = Counter()
    for effect in fingerprint.memory_effects:
        effect_counts[
            json.dumps(
                {
                    "kind": effect.kind,
                    "masked": effect.masked,
                    "other": (
                        None
                        if effect.other is None
                        else _FINGERPRINT_NODE_DIGEST_RE.sub(
                            "", effect.other
                        )
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ] += 1
    return {
        "loop_count": len(fingerprint.loops),
        "loops": [
            {
                "lower": _FINGERPRINT_NODE_DIGEST_RE.sub("", loop.lower),
                "upper": _FINGERPRINT_NODE_DIGEST_RE.sub("", loop.upper),
                "step": _FINGERPRINT_NODE_DIGEST_RE.sub("", loop.step),
                "init_values": [
                    _FINGERPRINT_NODE_DIGEST_RE.sub("", value)
                    for value in loop.init_values
                ],
                "yielded_values": [
                    _FINGERPRINT_NODE_DIGEST_RE.sub("", value)
                    for value in loop.yielded_values
                ],
                "carry_pairs": [
                    {
                        "init": _FINGERPRINT_NODE_DIGEST_RE.sub("", init),
                        "yield": _FINGERPRINT_NODE_DIGEST_RE.sub("", yielded),
                    }
                    for init, yielded in zip(
                        loop.init_values, loop.yielded_values
                    )
                ],
                "block_argument_count": loop.block_argument_count,
                "induction_use_count": loop.induction_use_count,
                "carried_argument_use_counts": list(
                    loop.carried_argument_use_counts
                ),
                "captured_values": [
                    _FINGERPRINT_NODE_DIGEST_RE.sub("", value)
                    for value in loop.captured_values
                ],
                "body_op_multiset": dict(loop.body_op_multiset),
                "body_effects": dict(loop.body_effects),
            }
            for loop in fingerprint.loops
        ],
        "reduction_count": len(fingerprint.reductions),
        "reductions": [
            reduction.to_dict() for reduction in fingerprint.reductions
        ],
        "memory_effects": [
            {
                **json.loads(key),
                "count": count,
            }
            for key, count in sorted(effect_counts.items())
        ],
        "mask_predicate_count": len(fingerprint.mask_predicates),
        "dot_count": len(fingerprint.dots),
        "dots": [dot.to_dict() for dot in fingerprint.dots],
    }


def _build_structural_relift_guidance(
    source_fingerprint: TTIRFingerprint,
    replay_fingerprint: TTIRFingerprint,
) -> dict[str, Any]:
    """Return exact high-level structure deltas without raw TTIR or prose."""

    source = _fingerprint_structure_summary(source_fingerprint)
    replay = _fingerprint_structure_summary(replay_fingerprint)
    coordinate_keys = (
        ("loop_structure", "loops"),
        ("reduction_structure", "reductions"),
        ("memory_effect_structure", "memory_effects"),
        ("mask_predicate_count", "mask_predicate_count"),
        ("dot_structure", "dots"),
    )
    mismatches = [
        coordinate
        for coordinate, key in coordinate_keys
        if source[key] != replay[key]
    ]
    return {
        "policy": RELIFT_STRUCTURAL_GUIDANCE_POLICY,
        "mismatch_coordinates": mismatches,
        "source": source,
        "replay": replay,
    }


def _build_relift_guidance(
    artifact: Gate1Artifact,
    comparison: TTIRComparison,
    *,
    source_fingerprint: TTIRFingerprint | None = None,
    replay_fingerprint: TTIRFingerprint | None = None,
) -> ReliftGuidance:
    failed = tuple(
        component.category
        for component in comparison.components
        if component.hard and not component.passed
    )
    confirmed = tuple(
        component.category
        for component in comparison.components
        if component.hard and component.passed
    )
    category_guidance: list[dict[str, Any]] = []
    for component in comparison.components:
        if not component.hard or component.passed:
            continue
        target = _RELIFT_CATEGORY_TARGETS[component.category]
        grouped = _group_semantic_differences(component)
        category_guidance.append(
            {
                "category": component.category,
                "raw_difference_count": len(component.differences),
                "grouped_differences": grouped,
                "inspect_j_surfaces": list(target["inspect"]),
                "reviewed_strategy_families": list(
                    target["strategy_families"]
                ),
                "semantic_question": target["semantic_question"],
            }
        )
    return ReliftGuidance(
        hard_difference_digest=_hard_difference_digest(comparison),
        failed_hard_categories=failed,
        confirmed_hard_categories=confirmed,
        category_guidance=tuple(category_guidance),
        current_j_summary=_current_j_summary(artifact),
        structural_guidance=(
            None
            if source_fingerprint is None or replay_fingerprint is None
            else _build_structural_relift_guidance(
                source_fingerprint, replay_fingerprint
            )
        ),
        decision_contract=(
            "rederive_whole_tensor_semantics_from_source",
            "classify_each_failed_category_as_j_gap_or_reviewed_strategy_gap_or_no_expressible_change",
            "edit_only_j_strategy_location_map",
            "preserve_confirmed_hard_categories",
            "never_patch_or_describe_replay_code",
            "return_unchanged_editable_fields_when_no_semantic_change_is_justified",
        ),
    )


def _composer_shadow_record(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    relift_round: int,
) -> ComposerShadowRecord:
    try:
        composed = compose_replay_plan(artifact, registry)
    except Exception as exc:
        code = exc.code if isinstance(exc, ComposeError) else type(exc).__name__
        return ComposerShadowRecord(
            relift_round,
            ComposerShadowStatus.FAILED,
            diagnostic=f"{code}: {exc}",
        )

    def marker(step: Any) -> tuple[str, str | None, tuple[str, ...]]:
        return step.rule, step.node_ref, tuple(step.refs)

    materialized = [marker(step) for step in artifact.plan.steps]
    expected = [marker(step) for step in composed.plan.steps]
    materialized_count = Counter(materialized)
    expected_count = Counter(expected)
    missing = tuple(
        sorted((expected_count - materialized_count).elements(), key=repr)
    )
    extra = tuple(
        sorted((materialized_count - expected_count).elements(), key=repr)
    )
    exact = artifact.plan.to_json() == composed.plan.to_json()
    return ComposerShadowRecord(
        relift_round,
        ComposerShadowStatus.MATCH if exact else ComposerShadowStatus.MISMATCH,
        composed=composed,
        materialized_plan_digest=_replay_plan_digest(artifact.plan),
        missing_steps=missing,
        extra_steps=extra,
        order_drift=(not missing and not extra and materialized != expected),
        diagnostic=None if exact else "materialized pi differs from deterministic compose",
    )


def _case_with_artifact_schedule(
    case: CompileCase, artifact: Gate1Artifact
) -> CompileCase:
    values: dict[str, int | float | bool | str] = {}
    for name, value in artifact.schedule.values.items():
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(
                f"Triton schedule value {name!r} must be a JSON scalar, "
                f"got {value!r}"
            )
        values[name] = value
    missing = artifact.schedule.missing_required_values()
    if missing:
        raise ValueError(f"missing required C value(s): {', '.join(missing)}")
    if case.schedule and dict(case.schedule) != values:
        raise ValueError(
            "CompileCase.schedule differs from the privileged artifact C values"
        )
    return replace(case, schedule=values)


_DTYPE_ALIASES = {
    "bool": "i1",
    "float16": "f16",
    "half": "f16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "uint8": "u8",
    "int16": "i16",
    "uint16": "u16",
    "int32": "i32",
    "uint32": "u32",
    "int64": "i64",
    "uint64": "u64",
}


def _canonical_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPE_ALIASES.get(spelling, spelling)


def _validate_artifact_compile_contract(
    source: KernelSource,
    artifact: Gate1Artifact,
    case: CompileCase,
) -> None:
    """Fail closed when privileged C/ABI inputs contradict the artifact."""

    interface = artifact.interface
    if interface.frontend is not Frontend.TRITON:
        raise ValueError(
            f"Gate1Runner currently supports Triton, got {interface.frontend.value!r}"
        )
    if source.function != interface.function_name:
        raise ValueError(
            "source function differs from KernelInterface.function_name: "
            f"{source.function!r} != {interface.function_name!r}"
        )
    if len(interface.grid) != len(case.grid):
        raise ValueError(
            "CompileCase grid rank differs from KernelInterface.grid contract"
        )
    parameter_by_name = {parameter.name: parameter for parameter in interface.parameters}
    interface_alias_pairs: set[frozenset[str]] = set()
    for parameter in interface.parameters:
        if parameter.alias_of is None:
            continue
        other = parameter_by_name[parameter.alias_of]
        if parameter.tensor_ref is None or other.tensor_ref is None:
            raise ValueError(
                "aliased interface parameters must both reference J tensors"
            )
        interface_alias_pairs.add(
            frozenset((parameter.tensor_ref, other.tensor_ref))
        )
    boundary_must_alias_pairs = {
        frozenset((alias.left, alias.right))
        for alias in artifact.j.boundary.aliases
        if alias.kind.value == "must_alias"
    }
    has_alias_contract = bool(
        artifact.j.boundary.aliases or interface_alias_pairs
    )
    if has_alias_contract:
        if not artifact.j.scopes or not artifact.j.storage:
            raise ValueError(
                "alias contracts require explicit portable scopes and semantic storage"
            )
        missing_boundary_aliases = sorted(
            tuple(sorted(pair))
            for pair in interface_alias_pairs - boundary_must_alias_pairs
        )
        if missing_boundary_aliases:
            raise ValueError(
                "KernelInterface.alias_of is not backed by Boundary must_alias: "
                f"{missing_boundary_aliases!r}"
            )
        tensor_parameter_refs = {
            parameter.tensor_ref
            for parameter in interface.parameters
            if parameter.tensor_ref is not None
        }
        visible_boundary_pairs = {
            pair
            for pair in boundary_must_alias_pairs
            if set(pair).issubset(tensor_parameter_refs)
        }
        missing_interface_aliases = sorted(
            tuple(sorted(pair))
            for pair in visible_boundary_pairs - interface_alias_pairs
        )
        if missing_interface_aliases:
            raise ValueError(
                "Boundary must_alias is absent from the trusted interface alias "
                f"contract: {missing_interface_aliases!r}"
            )

    runtime_parameters = [
        parameter
        for parameter in interface.parameters
        if parameter.kind is not ParameterKind.CONSTEXPR
    ]
    constexpr_parameters = [
        parameter
        for parameter in interface.parameters
        if parameter.kind is ParameterKind.CONSTEXPR
    ]
    runtime_names = [parameter.name for parameter in runtime_parameters]
    if list(case.arguments) != runtime_names:
        raise ValueError(
            "CompileCase runtime argument order/names differ from KernelInterface: "
            f"{list(case.arguments)!r} != {runtime_names!r}"
        )
    constexpr_names = [parameter.name for parameter in constexpr_parameters]
    missing_constexpr = sorted(set(constexpr_names) - set(case.schedule))
    if missing_constexpr:
        raise ValueError(
            "CompileCase is missing constexpr value(s): "
            + ", ".join(missing_constexpr)
        )
    missing_compiler = sorted(set(interface.compiler_slots) - set(case.schedule))
    if missing_compiler:
        raise ValueError(
            "CompileCase is missing compiler schedule value(s): "
            + ", ".join(missing_compiler)
        )

    for parameter in runtime_parameters:
        argument = case.arguments[parameter.name]
        expects_pointer = parameter.kind in {
            ParameterKind.POINTER,
            ParameterKind.TENSOR,
        }
        if expects_pointer != (argument.kind == "tensor"):
            raise ValueError(
                f"CompileCase argument {parameter.name!r} kind contradicts "
                f"KernelInterface kind {parameter.kind.value!r}"
            )
        if parameter.rank is not None:
            actual_rank = len(argument.shape or ()) if argument.kind == "tensor" else 0
            if actual_rank != parameter.rank:
                raise ValueError(
                    f"CompileCase argument {parameter.name!r} rank {actual_rank} "
                    f"!= interface rank {parameter.rank}"
                )
        if parameter.dtype is not None and argument.dtype is not None:
            if _canonical_dtype(parameter.dtype) != _canonical_dtype(argument.dtype):
                raise ValueError(
                    f"CompileCase argument {parameter.name!r} dtype "
                    f"{argument.dtype!r} != interface dtype {parameter.dtype!r}"
                )


def _validate_compiled_interface(
    result: CompileResult,
    fingerprint: TTIRFingerprint,
    artifact: Gate1Artifact,
) -> None:
    """Check the source/replay Python+TTIR interface against the declared ABI."""

    declared = artifact.interface
    expected_names = [parameter.name for parameter in declared.parameters]
    actual_names = list(result.python_interface.get("arg_names", ()))
    if actual_names != expected_names:
        raise ValueError(
            f"compiled arg_names {actual_names!r} != interface {expected_names!r}"
        )
    expected_constexprs = [
        parameter.name
        for parameter in declared.parameters
        if parameter.kind is ParameterKind.CONSTEXPR
    ]
    actual_constexprs = list(result.python_interface.get("constexprs", ()))
    if actual_constexprs != expected_constexprs:
        raise ValueError(
            f"compiled constexprs {actual_constexprs!r} != interface "
            f"{expected_constexprs!r}"
        )
    runtime_parameters = [
        parameter
        for parameter in declared.parameters
        if parameter.kind is not ParameterKind.CONSTEXPR
    ]
    if result.ttir is None:
        raise ValueError("compiled result has no TTIR")
    actual_names = runtime_argument_names(result.ttir)
    if len(actual_names) != len(fingerprint.interface.arguments):
        raise ValueError("compiled TTIR argument parser disagrees with fingerprint")
    by_name = {parameter.name: parameter for parameter in runtime_parameters}
    unknown = sorted(set(actual_names) - set(by_name))
    if unknown:
        raise ValueError(
            "compiled TTIR contains unknown runtime argument(s): "
            + ", ".join(unknown)
        )
    expected_surviving_order = [
        parameter.name
        for parameter in runtime_parameters
        if parameter.name in actual_names
    ]
    if list(actual_names) != expected_surviving_order:
        raise ValueError("compiled TTIR runtime arguments are not an ABI subsequence")
    omitted = [
        parameter
        for parameter in runtime_parameters
        if parameter.name not in actual_names
    ]
    binding_omissions = [
        parameter.name
        for parameter in omitted
        if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}
        or parameter.tensor_ref is not None
    ]
    if binding_omissions:
        raise ValueError(
            "compiled TTIR omitted binding runtime argument(s): "
            + ", ".join(binding_omissions)
        )
    for name, actual in zip(
        actual_names, fingerprint.interface.arguments, strict=True
    ):
        parameter = by_name[name]
        expected_kind = (
            "pointer"
            if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}
            else "scalar"
        )
        if actual.kind != expected_kind:
            raise ValueError(
                f"compiled argument {parameter.name!r} kind {actual.kind!r} "
                f"!= {expected_kind!r}"
            )
        if parameter.dtype is not None and (
            _canonical_dtype(actual.dtype) != _canonical_dtype(parameter.dtype)
        ):
            raise ValueError(
                f"compiled argument {parameter.name!r} dtype {actual.dtype!r} "
                f"!= {parameter.dtype!r}"
            )
        if actual.access != parameter.mode.value:
            raise ValueError(
                f"compiled argument {parameter.name!r} access {actual.access!r} "
                f"!= interface mode {parameter.mode.value!r}"
            )


def _passed_hard_categories(record: CandidateRecord | None) -> frozenset[str]:
    if record is None:
        return frozenset()
    if record.relift_guidance is not None:
        return frozenset(record.relift_guidance.confirmed_hard_categories)
    if record.comparison is None:
        return frozenset()
    return frozenset(
        component.category
        for component in record.comparison.components
        if component.hard and component.passed
    )


def _interface_for_fingerprint(
    result: CompileResult, case: CompileCase, artifact: Gate1Artifact
) -> dict[str, Any]:
    interface = dict(result.python_interface)
    # TTIR does not contain the launch grid.  It is nevertheless a hard
    # interface fact for row kernels whose M parameter may be unused in-body.
    interface["grid"] = list(case.grid)
    interface["grid_contract"] = list(artifact.interface.grid)
    return interface


MASKED_J_POLICY = "actual_lift_pi_masked_j_hard_nonclosure_v3"
MASKED_J_FAMILY_POLICY = (
    "actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1"
)
MASKED_J_FAMILY_DIGEST_SCHEMA = (
    "actual_lift_pi_masked_j_fixed_c_family_evidence_v2"
)
GATE1_FAMILY_ANTI_LEAKAGE_POLICY = (
    "normal_paired_and_masked_fixed_c_family_all_required_v1"
)


@dataclass(frozen=True)
class _MaskedJProfileProbe:
    profile_id: str
    source_compile: CompileResult | None
    compile: CompileResult | None
    comparison: TTIRComparison | None
    outcome: str
    diagnostic: str
    source_compile_reused: bool = False
    source_fingerprint_digest: str | None = None
    replay_fingerprint_digest: str | None = None

    @property
    def hard_closed(self) -> bool:
        return bool(self.comparison and self.comparison.hard_pass)

    @property
    def hard_mismatched(self) -> bool:
        return bool(
            self.comparison is not None
            and not self.comparison.hard_pass
            and self.outcome == "hard_mismatch"
        )

    @staticmethod
    def _compile_json(
        result: CompileResult | None,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "ok": result.ok,
            "diagnostic": result.diagnostic,
            "cache_key": result.cache_key,
            "cache_hit": result.cache_hit,
            "elapsed_ms": result.elapsed_ms,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "outcome": self.outcome,
            "hard_closed": self.hard_closed,
            "hard_mismatched": self.hard_mismatched,
            "source_compile": self._compile_json(self.source_compile),
            "source_compile_reused": self.source_compile_reused,
            "source_fingerprint_digest": self.source_fingerprint_digest,
            "replay_fingerprint_digest": self.replay_fingerprint_digest,
            "compile": self._compile_json(self.compile),
            "comparison": (
                None
                if self.comparison is None
                else self.comparison.to_dict()
            ),
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class _MaskedJProbe:
    generation: ReplayGeneration | None
    compile: CompileResult | None
    comparison: TTIRComparison | None
    outcome: str
    diagnostic: str
    profile_records: tuple[_MaskedJProfileProbe, ...] = ()
    family_policy: str | None = None
    actual_pi_digest: str | None = None
    family_id: str | None = None
    family_contract_digest: str | None = None
    restored_code_sha256: str | None = None

    @property
    def hard_closed(self) -> bool:
        return bool(
            (self.comparison and self.comparison.hard_pass)
            or any(item.hard_closed for item in self.profile_records)
        )

    @property
    def family_complete_hard_mismatch(self) -> bool:
        return bool(
            self.profile_records
            and all(item.hard_mismatched for item in self.profile_records)
        )

    def family_to_json(self) -> dict[str, Any] | None:
        if not self.profile_records:
            return None
        ordered = tuple(
            sorted(self.profile_records, key=lambda item: item.profile_id)
        )
        evidence_complete = all(
            item.comparison is not None
            and item.source_fingerprint_digest is not None
            and item.replay_fingerprint_digest is not None
            for item in ordered
        )
        family_digest = (
            hashlib.sha256(
                json.dumps(
                    {
                        "digest_schema": MASKED_J_FAMILY_DIGEST_SCHEMA,
                        "policy": self.family_policy,
                        "actual_pi_digest": self.actual_pi_digest,
                        "family_id": self.family_id,
                        "family_contract_digest": (
                            self.family_contract_digest
                        ),
                        "masked_code_sha256": (
                            None
                            if self.generation is None
                            else self.generation.code_sha256
                        ),
                        "restored_code_sha256": self.restored_code_sha256,
                        "profiles": [
                            {
                                "profile_id": item.profile_id,
                                "outcome": item.outcome,
                                "source_fingerprint_digest": (
                                    item.source_fingerprint_digest
                                ),
                                "replay_fingerprint_digest": (
                                    item.replay_fingerprint_digest
                                ),
                                "hard_difference_digest": (
                                    _hard_difference_digest(
                                        item.comparison
                                    )
                                ),
                            }
                            for item in ordered
                            if item.comparison is not None
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if evidence_complete
            else None
        )
        return {
            "policy": self.family_policy,
            "actual_pi_digest": self.actual_pi_digest,
            "family_id": self.family_id,
            "family_contract_digest": self.family_contract_digest,
            "masked_code_sha256": (
                None
                if self.generation is None
                else self.generation.code_sha256
            ),
            "restored_code_sha256": self.restored_code_sha256,
            "outcome": self.outcome,
            "evidence_complete": evidence_complete,
            "family_digest": family_digest,
            "all_profiles_hard_mismatch": (
                self.family_complete_hard_mismatch
            ),
            "hard_closed_profile_ids": sorted(
                item.profile_id
                for item in ordered
                if item.hard_closed
            ),
            "incomplete_profile_ids": sorted(
                item.profile_id
                for item in ordered
                if not item.hard_closed and not item.hard_mismatched
            ),
            "records": [
                item.to_json() for item in ordered
            ],
        }


def _run_masked_j_probe(
    *,
    source: KernelSource,
    artifact: Gate1Artifact,
    source_compile: CompileResult,
    source_fingerprint: TTIRFingerprint,
    compile_case: CompileCase,
    registry: RuleRegistry,
    writer: ReplayWriter,
    compiler: TritonCompiler,
    comparator: TTIRComparator,
    seen_generation_ids: set[str],
    specialization_family: Any | None = None,
) -> _MaskedJProbe:
    """Replay masked J against the actual Lift-produced pi exactly once.

    This packet is an intentional adversarial counterfactual, not a valid Lift
    artifact: J semantic fields are hidden by ``build_rewriter_payload`` while
    the saved pi rule sequence and reference topology remain unchanged.  The
    writer receives no normal replay candidate and no repair feedback.
    """

    family_id: str | None = None
    family_contract_digest: str | None = None
    actual_pi_digest: str | None = None
    validate_specialization_profile = None
    if specialization_family is not None:
        from .specialization import (
            Gate1SpecializationFamily,
            specialization_family_contract_digest,
            validate_specialization_profile as validate_profile,
        )

        if not isinstance(
            specialization_family, Gate1SpecializationFamily
        ):
            raise TypeError(
                "specialization_family must be Gate1SpecializationFamily"
            )
        family_id = specialization_family.family_id
        family_contract_digest = specialization_family_contract_digest(
            specialization_family
        )
        actual_pi_digest = _replay_plan_digest(artifact.plan)
        validate_specialization_profile = validate_profile

    generation: ReplayGeneration | None = None
    try:
        generation = writer.generate(
            artifact,
            registry,
            masked=True,
        )
        if not generation.masked:
            raise RuntimeError("masked-J writer returned an unmasked generation")
        if generation.generation_id in seen_generation_ids:
            raise RuntimeError(
                "masked-J writer reused a generation_id from another replay"
            )
        seen_generation_ids.add(generation.generation_id)
    except Exception as exc:
        diagnostic = (
            "masked-J/full-pi attack could not produce an auditable fresh "
            f"candidate: {type(exc).__name__}: {exc}"
        )
        family_records = (
            tuple(
                _MaskedJProfileProbe(
                    profile.profile_id,
                    None,
                    None,
                    None,
                    "generation_incomplete",
                    diagnostic,
                )
                for profile in specialization_family.profiles
            )
            if specialization_family is not None
            else ()
        )
        return _MaskedJProbe(
            generation,
            None,
            None,
            "generation_unavailable",
            diagnostic,
            family_records,
            (
                MASKED_J_FAMILY_POLICY
                if specialization_family is not None
                else None
            ),
            (
                actual_pi_digest
            ),
            family_id=family_id,
            family_contract_digest=family_contract_digest,
        )

    try:
        restored_code = restore_masked_replay_module(
            generation.code,
            artifact,
            registry,
        )
    except Exception as exc:
        diagnostic = (
            "masked-J/full-pi attack produced no legal source-ABI replay and "
            "is therefore unevaluable: "
            f"{type(exc).__name__}: {exc}"
        )
        family_records = (
            tuple(
                _MaskedJProfileProbe(
                    profile.profile_id,
                    None,
                    None,
                    None,
                    "invalid_replay",
                    diagnostic,
                )
                for profile in specialization_family.profiles
            )
            if specialization_family is not None
            else ()
        )
        return _MaskedJProbe(
            generation,
            None,
            None,
            "invalid_replay",
            diagnostic,
            family_records,
            (
                MASKED_J_FAMILY_POLICY
                if specialization_family is not None
                else None
            ),
            (
                actual_pi_digest
            ),
            family_id=family_id,
            family_contract_digest=family_contract_digest,
        )

    replay_source = KernelSource.from_replay(
        restored_code,
        artifact.interface.function_name,
        registry,
    )
    restored_code_sha256 = hashlib.sha256(
        restored_code.encode("utf-8")
    ).hexdigest()
    if specialization_family is not None:
        assert validate_specialization_profile is not None
        records: list[_MaskedJProfileProbe] = []
        for profile in specialization_family.profiles:
            validate_specialization_profile(artifact, profile)
            source_reused = (
                profile.profile_id
                == specialization_family.primary_profile_id
            )
            if (
                profile.profile_id
                == specialization_family.primary_profile_id
            ):
                profile_source_compile = source_compile
                profile_source_fp = source_fingerprint
            else:
                try:
                    profile_source_compile = compiler.compile(
                        source, profile.case
                    )
                except Exception as exc:
                    records.append(
                        _MaskedJProfileProbe(
                            profile.profile_id,
                            None,
                            None,
                            None,
                            "source_compile_incomplete",
                            "trusted source compilation raised during the "
                            "masked-J family sweep: "
                            f"{type(exc).__name__}: {exc}",
                            source_compile_reused=source_reused,
                        )
                    )
                    continue
                if (
                    not profile_source_compile.ok
                    or not profile_source_compile.ttir
                ):
                    records.append(
                        _MaskedJProfileProbe(
                            profile.profile_id,
                            profile_source_compile,
                            None,
                            None,
                            "source_compile_incomplete",
                            "trusted source did not compile during the "
                            "masked-J family sweep",
                            source_compile_reused=source_reused,
                        )
                    )
                    continue
                try:
                    profile_source_fp = fingerprint_ttir(
                        profile_source_compile.ttir,
                        python_interface=_interface_for_fingerprint(
                            profile_source_compile,
                            profile.case,
                            artifact,
                        ),
                        schedule=profile.case.schedule,
                    )
                    _validate_compiled_interface(
                        profile_source_compile,
                        profile_source_fp,
                        artifact,
                    )
                except (TTIRParseError, ValueError) as exc:
                    records.append(
                        _MaskedJProfileProbe(
                            profile.profile_id,
                            profile_source_compile,
                            None,
                            None,
                            "source_fingerprint_incomplete",
                            "trusted source produced no comparable fingerprint "
                            "during the masked-J family sweep: "
                            f"{type(exc).__name__}: {exc}",
                            source_compile_reused=source_reused,
                        )
                    )
                    continue

            try:
                profile_compile = compiler.compile(
                    replay_source, profile.case
                )
            except Exception as exc:
                records.append(
                    _MaskedJProfileProbe(
                        profile.profile_id,
                        profile_source_compile,
                        None,
                        None,
                        "compile_incomplete",
                        "masked-J replay compilation raised: "
                        f"{type(exc).__name__}: {exc}",
                        source_compile_reused=source_reused,
                        source_fingerprint_digest=profile_source_fp.digest,
                    )
                )
                continue
            if not profile_compile.ok or not profile_compile.ttir:
                records.append(
                    _MaskedJProfileProbe(
                        profile.profile_id,
                        profile_source_compile,
                        profile_compile,
                        None,
                        "compile_incomplete",
                        "masked-J replay did not compile; fixed-C family "
                        "evidence requires a comparable fingerprint for every "
                        "profile",
                        source_compile_reused=source_reused,
                        source_fingerprint_digest=profile_source_fp.digest,
                    )
                )
                continue
            try:
                profile_replay_fp = fingerprint_ttir(
                    profile_compile.ttir,
                    python_interface=_interface_for_fingerprint(
                        profile_compile,
                        profile.case,
                        artifact,
                    ),
                    schedule=profile.case.schedule,
                )
            except (TTIRParseError, ValueError) as exc:
                records.append(
                    _MaskedJProfileProbe(
                        profile.profile_id,
                        profile_source_compile,
                        profile_compile,
                        None,
                        "fingerprint_incomplete",
                        "masked-J replay produced no comparable fingerprint: "
                        f"{type(exc).__name__}: {exc}",
                        source_compile_reused=source_reused,
                        source_fingerprint_digest=profile_source_fp.digest,
                    )
                )
                continue
            profile_comparison = comparator.compare(
                profile_source_fp, profile_replay_fp
            )
            if profile_comparison.hard_pass:
                outcome = "hard_closure"
                diagnostic = (
                    "masked-J/full-pi attack hard-closed for this fixed-C "
                    "profile"
                )
            else:
                outcome = "hard_mismatch"
                categories = sorted(
                    component.category
                    for component in profile_comparison.components
                    if component.hard and not component.passed
                )
                diagnostic = (
                    "masked-J/full-pi attack produced a comparable hard "
                    f"mismatch; categories={categories}"
                )
            records.append(
                _MaskedJProfileProbe(
                    profile.profile_id,
                    profile_source_compile,
                    profile_compile,
                    profile_comparison,
                    outcome,
                    diagnostic,
                    source_compile_reused=source_reused,
                    source_fingerprint_digest=profile_source_fp.digest,
                    replay_fingerprint_digest=profile_replay_fp.digest,
                )
            )

        hard_closed_ids = [
            item.profile_id for item in records if item.hard_closed
        ]
        incomplete_ids = [
            item.profile_id
            for item in records
            if not item.hard_closed and not item.hard_mismatched
        ]
        if hard_closed_ids:
            family_outcome = "family_hard_closure"
            family_diagnostic = (
                "masked-J/full-pi attack hard-closed in fixed-C profile(s): "
                + ", ".join(hard_closed_ids)
            )
        elif records and all(item.hard_mismatched for item in records):
            family_outcome = "family_hard_mismatch"
            family_diagnostic = (
                "masked-J/full-pi attack produced comparable hard mismatch "
                "for every fixed-C profile"
            )
        else:
            family_outcome = "family_incomplete"
            family_diagnostic = (
                "masked-J/full-pi fixed-C family evidence is incomplete for "
                "profile(s): "
                + ", ".join(incomplete_ids)
            )
        primary = next(
            item
            for item in records
            if item.profile_id
            == specialization_family.primary_profile_id
        )
        return _MaskedJProbe(
            generation,
            primary.compile,
            primary.comparison,
            family_outcome,
            family_diagnostic,
            tuple(records),
            MASKED_J_FAMILY_POLICY,
            actual_pi_digest,
            family_id=family_id,
            family_contract_digest=family_contract_digest,
            restored_code_sha256=restored_code_sha256,
        )

    compile_result = compiler.compile(replay_source, compile_case)
    if not compile_result.ok or not compile_result.ttir:
        return _MaskedJProbe(
            generation,
            compile_result,
            None,
            "compile_nonclosure",
            "masked-J/full-pi attack produced a legal candidate but the "
            "compiler rejected it; the attack did not close",
        )

    try:
        replay_fingerprint = fingerprint_ttir(
            compile_result.ttir,
            python_interface=_interface_for_fingerprint(
                compile_result, compile_case, artifact
            ),
            schedule=compile_case.schedule,
        )
    except (TTIRParseError, ValueError) as exc:
        return _MaskedJProbe(
            generation,
            compile_result,
            None,
            "fingerprint_invalid",
            "masked-J/full-pi attack produced no comparable fingerprint "
            "and is therefore unevaluable: "
            f"{type(exc).__name__}: {exc}",
        )

    comparison = comparator.compare(source_fingerprint, replay_fingerprint)
    if comparison.hard_pass:
        return _MaskedJProbe(
            generation,
            compile_result,
            comparison,
            "hard_closure",
            "masked-J/full-pi attack hard-closed against the source fingerprint; "
            "the non-J packet still exposes enough structure for replay",
        )
    hard_categories = sorted(
        component.category
        for component in comparison.components
        if component.hard and not component.passed
    )
    return _MaskedJProbe(
        generation,
        compile_result,
        comparison,
        "hard_mismatch",
        "masked-J/full-pi attack compiled but did not hard-close; mismatched "
        f"categories={hard_categories}",
    )


def _run_semantic_interventions(
    *,
    artifact: Gate1Artifact,
    baseline_record: CandidateRecord,
    compile_case: CompileCase,
    registry: RuleRegistry,
    writer: ReplayWriter,
    compiler: TritonCompiler,
    seen_generation_ids: set[str],
    limit: int,
    specialization_family: Any | None = None,
) -> tuple[InterventionRecord, ...]:
    """Run source-blind paired interventions against a closed replay."""

    if (
        limit == 0
        or baseline_record.compile is None
        or not baseline_record.compile.ok
        or not baseline_record.compile.ttir
    ):
        return ()
    specs = build_semantic_interventions(
        artifact, limit=limit, registry=registry
    )
    if not specs:
        return ()
    if specialization_family is not None:
        return _run_semantic_interventions_family(
            artifact=artifact,
            baseline_record=baseline_record,
            specs=specs,
            registry=registry,
            writer=writer,
            compiler=compiler,
            seen_generation_ids=seen_generation_ids,
            specialization_family=specialization_family,
        )
    baseline_fp = fingerprint_ttir(
        baseline_record.compile.ttir,
        python_interface=_interface_for_fingerprint(
            baseline_record.compile, compile_case, artifact
        ),
        schedule=compile_case.schedule,
    )
    records: list[InterventionRecord] = []
    for spec in specs:
        try:
            generation = writer.generate(
                spec.artifact,
                registry,
                masked=False,
            )
            if generation.generation_id in seen_generation_ids:
                raise RuntimeError(
                    "semantic intervention reused an existing generation_id"
                )
            seen_generation_ids.add(generation.generation_id)
        except Exception as exc:
            records.append(
                InterventionRecord(
                    spec,
                    InterventionStatus.GENERATION_FAILED,
                    diagnostic=(
                        "semantic intervention generation failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue

        function_name = spec.artifact.interface.function_name
        try:
            compiled = compiler.compile(
                KernelSource.from_replay(
                    generation.code,
                    function_name,
                    registry,
                ),
                compile_case,
            )
        except Exception as exc:
            records.append(
                InterventionRecord(
                    spec,
                    InterventionStatus.COMPILE_FAILED,
                    generation=generation,
                    diagnostic=(
                        "semantic intervention compiler raised: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue
        if not compiled.ok or not compiled.ttir:
            records.append(
                InterventionRecord(
                    spec,
                    InterventionStatus.COMPILE_FAILED,
                    generation=generation,
                    compile=compiled,
                    diagnostic=(
                        "semantic intervention replay did not compile: "
                        f"{compiled.diagnostic}"
                    ),
                )
            )
            continue
        try:
            intervention_fp = fingerprint_ttir(
                compiled.ttir,
                python_interface=_interface_for_fingerprint(
                    compiled, compile_case, spec.artifact
                ),
                schedule=compile_case.schedule,
            )
            _validate_compiled_interface(
                compiled, intervention_fp, spec.artifact
            )
            verdict = evaluate_semantic_intervention(
                spec, baseline_fp, intervention_fp
            )
            records.append(
                InterventionRecord(
                    spec,
                    verdict.status,
                    generation=generation,
                    compile=compiled,
                    verdict=verdict,
                    diagnostic=verdict.diagnostic,
                )
            )
        except Exception as exc:
            records.append(
                InterventionRecord(
                    spec,
                    InterventionStatus.EVALUATION_FAILED,
                    generation=generation,
                    compile=compiled,
                    diagnostic=(
                        "semantic intervention evaluation failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
    return tuple(records)


def _run_semantic_interventions_family(
    *,
    artifact: Gate1Artifact,
    baseline_record: CandidateRecord,
    specs: tuple[Any, ...],
    registry: RuleRegistry,
    writer: ReplayWriter,
    compiler: TritonCompiler,
    seen_generation_ids: set[str],
    specialization_family: Any,
) -> tuple[InterventionRecord, ...]:
    """Evaluate each single mutant replay over the complete fixed-C family."""

    from .specialization import (
        Gate1SpecializationFamily,
        specialization_family_contract_digest,
        validate_specialization_profile,
    )

    if not isinstance(
        specialization_family, Gate1SpecializationFamily
    ):
        raise TypeError(
            "specialization_family must be Gate1SpecializationFamily"
        )
    family_id = specialization_family.family_id
    family_contract_digest = specialization_family_contract_digest(
        specialization_family
    )
    baseline_code_sha256 = (
        None
        if baseline_record.generation is None
        else baseline_record.generation.code_sha256
    )
    if (
        baseline_record.generation is None
        or baseline_record.compile is None
        or not baseline_record.compile.ok
        or not baseline_record.compile.ttir
    ):
        return tuple(
            InterventionRecord(
                spec,
                InterventionStatus.EVALUATION_FAILED,
                diagnostic=(
                    "fixed-C family intervention has no auditable normal "
                    "baseline replay"
                ),
                family_policy=PAIRED_INTERVENTION_FAMILY_POLICY,
                baseline_code_sha256=baseline_code_sha256,
                family_id=family_id,
                family_contract_digest=family_contract_digest,
                family_outcome="incomplete",
            )
            for spec in specs
        )

    baseline_source = KernelSource.from_replay(
        baseline_record.generation.code,
        artifact.interface.function_name,
        registry,
    )
    baseline_by_profile: dict[
        str, tuple[CompileResult | None, TTIRFingerprint | None, str]
    ] = {}
    for profile in specialization_family.profiles:
        validate_specialization_profile(artifact, profile)
        if (
            profile.profile_id
            == specialization_family.primary_profile_id
        ):
            compiled = baseline_record.compile
        else:
            try:
                compiled = compiler.compile(baseline_source, profile.case)
            except Exception as exc:
                baseline_by_profile[profile.profile_id] = (
                    None,
                    None,
                    "normal baseline replay compilation raised: "
                    f"{type(exc).__name__}: {exc}",
                )
                continue
        if not compiled.ok or not compiled.ttir:
            baseline_by_profile[profile.profile_id] = (
                compiled,
                None,
                "normal baseline replay did not compile for the fixed-C "
                "intervention profile",
            )
            continue
        try:
            fingerprint = fingerprint_ttir(
                compiled.ttir,
                python_interface=_interface_for_fingerprint(
                    compiled, profile.case, artifact
                ),
                schedule=profile.case.schedule,
            )
            _validate_compiled_interface(compiled, fingerprint, artifact)
        except (TTIRParseError, ValueError) as exc:
            baseline_by_profile[profile.profile_id] = (
                compiled,
                None,
                "normal baseline replay produced no comparable fingerprint: "
                f"{type(exc).__name__}: {exc}",
            )
            continue
        baseline_by_profile[profile.profile_id] = (
            compiled,
            fingerprint,
            "",
        )

    records: list[InterventionRecord] = []
    for spec in specs:
        try:
            generation = writer.generate(
                spec.artifact,
                registry,
                masked=False,
            )
            if generation.generation_id in seen_generation_ids:
                raise RuntimeError(
                    "semantic intervention reused an existing generation_id"
                )
            seen_generation_ids.add(generation.generation_id)
        except Exception as exc:
            records.append(
                InterventionRecord(
                    spec,
                    InterventionStatus.GENERATION_FAILED,
                    diagnostic=(
                        "fixed-C family semantic intervention generation "
                        f"failed: {type(exc).__name__}: {exc}"
                    ),
                    family_policy=PAIRED_INTERVENTION_FAMILY_POLICY,
                    baseline_code_sha256=baseline_code_sha256,
                    family_id=family_id,
                    family_contract_digest=family_contract_digest,
                    family_outcome="incomplete",
                )
            )
            continue

        mutant_source = KernelSource.from_replay(
            generation.code,
            spec.artifact.interface.function_name,
            registry,
        )
        profile_records: list[InterventionProfileRecord] = []
        for profile in specialization_family.profiles:
            baseline_reused = (
                profile.profile_id
                == specialization_family.primary_profile_id
            )
            baseline_compile, baseline_fp, baseline_diagnostic = (
                baseline_by_profile[profile.profile_id]
            )
            if baseline_fp is None:
                profile_records.append(
                    InterventionProfileRecord(
                        profile.profile_id,
                        InterventionStatus.EVALUATION_FAILED,
                        baseline_compile=baseline_compile,
                        diagnostic=baseline_diagnostic,
                        baseline_compile_reused=baseline_reused,
                    )
                )
                continue
            try:
                compiled = compiler.compile(mutant_source, profile.case)
            except Exception as exc:
                profile_records.append(
                    InterventionProfileRecord(
                        profile.profile_id,
                        InterventionStatus.COMPILE_FAILED,
                        baseline_compile=baseline_compile,
                        diagnostic=(
                            "semantic intervention compiler raised: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        baseline_compile_reused=baseline_reused,
                    )
                )
                continue
            if not compiled.ok or not compiled.ttir:
                profile_records.append(
                    InterventionProfileRecord(
                        profile.profile_id,
                        InterventionStatus.COMPILE_FAILED,
                        baseline_compile=baseline_compile,
                        compile=compiled,
                        diagnostic=(
                            "semantic intervention replay did not compile: "
                            f"{compiled.diagnostic}"
                        ),
                        baseline_compile_reused=baseline_reused,
                    )
                )
                continue
            try:
                intervention_fp = fingerprint_ttir(
                    compiled.ttir,
                    python_interface=_interface_for_fingerprint(
                        compiled, profile.case, spec.artifact
                    ),
                    schedule=profile.case.schedule,
                )
                _validate_compiled_interface(
                    compiled, intervention_fp, spec.artifact
                )
                verdict = evaluate_semantic_intervention(
                    spec, baseline_fp, intervention_fp
                )
            except Exception as exc:
                profile_records.append(
                    InterventionProfileRecord(
                        profile.profile_id,
                        InterventionStatus.EVALUATION_FAILED,
                        baseline_compile=baseline_compile,
                        compile=compiled,
                        diagnostic=(
                            "semantic intervention evaluation failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        baseline_compile_reused=baseline_reused,
                    )
                )
                continue
            profile_records.append(
                InterventionProfileRecord(
                    profile.profile_id,
                    verdict.status,
                    baseline_compile=baseline_compile,
                    compile=compiled,
                    verdict=verdict,
                    diagnostic=verdict.diagnostic,
                    baseline_compile_reused=baseline_reused,
                )
            )

        failed = sorted(
            (item for item in profile_records if not item.passed),
            key=lambda item: item.profile_id,
        )
        semantic_failures = [
            item
            for item in failed
            if item.status in INTERVENTION_SEMANTIC_FAILURE_STATUSES
        ]
        incomplete_failures = [
            item
            for item in failed
            if item.status not in INTERVENTION_SEMANTIC_FAILURE_STATUSES
        ]
        if semantic_failures:
            semantic_precedence = (
                InterventionStatus.IGNORED_J,
                InterventionStatus.WRONG_OPERATOR_DELTA,
                InterventionStatus.WRONG_CONSTANT_DELTA,
                InterventionStatus.WRONG_FILL_DELTA,
                InterventionStatus.OFF_TARGET_DRIFT,
            )
            aggregate_status = next(
                status
                for status in semantic_precedence
                if any(
                    item.status is status for item in semantic_failures
                )
            )
            family_outcome = (
                "semantic_failure_and_incomplete"
                if incomplete_failures
                else "semantic_failure"
            )
        elif incomplete_failures:
            incomplete_precedence = (
                InterventionStatus.GENERATION_FAILED,
                InterventionStatus.COMPILE_FAILED,
                InterventionStatus.EVALUATION_FAILED,
            )
            aggregate_status = next(
                status
                for status in incomplete_precedence
                if any(
                    item.status is status for item in incomplete_failures
                )
            )
            family_outcome = "incomplete"
        else:
            aggregate_status = InterventionStatus.PASS
            family_outcome = "pass"
        primary = next(
            item
            for item in profile_records
            if item.profile_id
            == specialization_family.primary_profile_id
        )
        representative = (
            primary
            if aggregate_status is InterventionStatus.PASS
            else next(
                item
                for item in sorted(
                    profile_records, key=lambda value: value.profile_id
                )
                if item.status is aggregate_status
            )
        )
        diagnostic = (
            "registered semantic intervention exact delta passed for every "
            "fixed-C profile"
            if not failed
            else "fixed-C semantic intervention failed profile(s): "
            + ", ".join(
                f"{item.profile_id}:{item.status.value}" for item in failed
            )
        )
        records.append(
            InterventionRecord(
                spec,
                aggregate_status,
                generation=generation,
                compile=representative.compile,
                verdict=representative.verdict,
                diagnostic=diagnostic,
                profile_records=tuple(profile_records),
                family_policy=PAIRED_INTERVENTION_FAMILY_POLICY,
                baseline_code_sha256=baseline_code_sha256,
                family_id=family_id,
                family_contract_digest=family_contract_digest,
                family_outcome=family_outcome,
            )
        )
    return tuple(records)


def _termination_for_round(
    reason: Gate1TerminationReason,
    *,
    relift_round: int,
    history: list[CandidateRecord],
    config: Gate1Config,
    hard_difference_categories: tuple[str, ...] = (),
) -> Gate1Termination:
    return Gate1Termination(
        reason=reason,
        artifact_round=relift_round,
        normal_candidate_count=len(history),
        remaining_artifact_round_budget=max(
            config.max_relifts - relift_round - 1, 0
        ),
        hard_difference_categories=hard_difference_categories,
    )


@dataclass(frozen=True)
class _SpecializationEvaluation:
    evidence: Mapping[str, Any]
    hard_pass: bool
    diagnosis: Diagnosis | None = None
    hard_difference_digest: str | None = None
    hard_difference_categories: tuple[str, ...] = ()
    relift_guidance: ReliftGuidance | None = None
    relift_trigger: ReliftTrigger | None = None
    source_compile_failed: bool = False
    leakage_rejected: bool = False


def _evaluate_specialization_family(
    *,
    source: KernelSource,
    artifact: Gate1Artifact,
    family: Any,
    registry: RuleRegistry,
    generation: ReplayGeneration,
    primary_source_compile: CompileResult,
    primary_replay_compile: CompileResult,
    primary_comparison: TTIRComparison | None,
    compiler: TritonCompiler,
    comparator: TTIRComparator,
    leakage_policy: LeakagePolicy,
) -> _SpecializationEvaluation:
    """Run the complete fixed-C sweep for one immutable replay generation."""

    from .specialization import (
        Gate1FamilyRunner,
        Gate1SpecializationRecord,
        build_family_relift_guidance,
        diagnose_family_failure,
        family_hard_difference_digest,
    )

    result = Gate1FamilyRunner(
        compiler=compiler,
        comparator=comparator,
        leakage_policy=leakage_policy,
    ).run(
        source,
        artifact,
        family,
        registry,
        generation=generation,
        primary_record=Gate1SpecializationRecord(
            family.primary_profile_id,
            primary_source_compile,
            primary_replay_compile,
            primary_comparison,
            (
                ""
                if primary_comparison is not None
                and primary_comparison.hard_pass
                else primary_replay_compile.diagnostic
                or "hard TTIR mismatch"
            ),
            source_compile_reused=True,
            replay_compile_reused=True,
        ),
    )
    evidence = result.relift_evidence_json()
    if result.status is Gate1Status.COMPILE_FAILED:
        return _SpecializationEvaluation(
            evidence,
            False,
            diagnosis=Diagnosis(
                FailureCause.REPLAY_POLICY,
                Responsibility.UNKNOWN,
                ("at least one trusted source specialization did not compile",),
                (),
                (),
                "reviewed fixed-C source profile 编译失败；不能把 harness/"
                "contract 问题伪装成 J 语义 Relift。",
            ),
            source_compile_failed=True,
        )
    if result.status is Gate1Status.LEAKAGE_REJECTED:
        return _SpecializationEvaluation(
            evidence,
            False,
            leakage_rejected=True,
        )
    if result.all_passed:
        return _SpecializationEvaluation(evidence, True)

    guidance = build_family_relift_guidance(artifact, result)
    return _SpecializationEvaluation(
        evidence,
        False,
        diagnosis=diagnose_family_failure(result),
        hard_difference_digest=family_hard_difference_digest(result),
        hard_difference_categories=(
            () if guidance is None else guidance.failed_hard_categories
        ),
        relift_guidance=guidance,
        relift_trigger=(
            ReliftTrigger.REPLAY_COMPILE_FAILED
            if guidance is None
            else ReliftTrigger.NORMAL_NO_CLOSURE
        ),
    )


class Gate1Runner:
    def __init__(
        self,
        *,
        compiler: TritonCompiler | None = None,
        comparator: TTIRComparator | None = None,
        config: Gate1Config | None = None,
        leakage_policy: LeakagePolicy | None = None,
    ) -> None:
        self.compiler = compiler or TritonCompiler()
        self.comparator = comparator or TTIRComparator()
        self.config = config or Gate1Config()
        self.leakage_policy = leakage_policy or LeakagePolicy()

    def run(
        self,
        source: KernelSource,
        artifact: Gate1Artifact,
        case: CompileCase,
        registry: RuleRegistry,
        writer: ReplayWriter,
        *,
        relifter: Relifter | None = None,
        specialization_family: Any | None = None,
    ) -> Gate1ExecutionResult:
        if not isinstance(registry, RuleRegistry):
            registry = RuleRegistry.from_json(registry)
        registry = registry.frozen_snapshot()
        history: list[CandidateRecord] = []
        diagnostics: list[str] = []
        best: CandidateRecord | None = None
        current = artifact
        source_compile: CompileResult | None = None
        source_fp: TTIRFingerprint | None = None
        leakage_report: LeakageReport | None = None
        composer_shadow_records: list[ComposerShadowRecord] = []
        previous_stagnation_state: tuple[
            str, str, int, tuple[str, ...]
        ] | None = None
        stagnation: StagnationEvidence | None = None
        termination: Gate1Termination | None = None

        for relift_round in range(self.config.max_relifts):
            artifact_digest: str | None = None
            try:
                compile_case = _case_with_artifact_schedule(case, current)
                if specialization_family is not None:
                    from .specialization import Gate1SpecializationFamily

                    if not isinstance(
                        specialization_family, Gate1SpecializationFamily
                    ):
                        raise TypeError(
                            "specialization_family must be "
                            "Gate1SpecializationFamily"
                        )
                    primary_profile = next(
                        profile
                        for profile in specialization_family.profiles
                        if profile.profile_id
                        == specialization_family.primary_profile_id
                    )
                    if (
                        primary_profile.case.to_payload()
                        != compile_case.to_payload()
                    ):
                        raise ValueError(
                            "primary specialization profile differs from the "
                            "Gate1 CompileCase"
                        )
                _validate_artifact_compile_contract(source, current, compile_case)
                registry.validate_strict_plan(
                    current.plan,
                    frontend=current.interface.frontend,
                    artifact=current,
                )
                # The complete artifact, including pi, is owned by Lift.
                # Gate1 only recomputes the deterministic expectation to reject
                # corruption or drift; it never materializes a missing plan.
                composed = assert_replay_plan_matches(current, registry)
                artifact_digest = _canonical_artifact_digest(composed)
                if (
                    current.compose_provenance is not None
                    and current.compose_provenance
                    != composed.provenance.to_json()
                ):
                    raise ComposeError(
                        "stored compose_provenance does not equal fresh "
                        "deterministic composition"
                    )
                if self.config.run_composer_shadow:
                    shadow = _composer_shadow_record(
                        current, registry, relift_round
                    )
                    composer_shadow_records.append(shadow)
                diagnostics.append(
                    "deterministic pi equality and leaf provenance recomputation "
                    f"passed at relift round {relift_round}"
                )
                source_text, _ = source.read_code()
                effective_leakage_policy, step_budget_extended = (
                    _exact_composed_leakage_policy(
                        self.leakage_policy,
                        composed.plan,
                    )
                )
                leakage_report = scan_static_leakage(
                    composed.plan,
                    source=source_text,
                    j=current.j,
                    policy=effective_leakage_policy,
                    registry=registry,
                )
                deterministic_steps = len(composed.plan.steps)
                leakage_report.metrics.update(
                    {
                        "configured_step_budget": self.leakage_policy.max_steps,
                        "effective_step_budget": effective_leakage_policy.max_steps,
                        "deterministic_plan_steps": deterministic_steps,
                        "absolute_exact_composed_step_budget": (
                            MAX_EXACT_COMPOSED_PLAN_STEPS
                        ),
                        "exact_composer_step_extension": int(step_budget_extended),
                    }
                )
                if step_budget_extended:
                    diagnostics.append(
                        "exact composer attestation extended only the mechanical "
                        "plan step budget from "
                        f"{self.leakage_policy.max_steps} to "
                        f"{effective_leakage_policy.max_steps}"
                    )
                elif deterministic_steps > MAX_EXACT_COMPOSED_PLAN_STEPS:
                    diagnostics.append(
                        "exact composed plan step count "
                        f"{deterministic_steps} exceeds absolute trusted limit "
                        f"{MAX_EXACT_COMPOSED_PLAN_STEPS}"
                    )
            except Exception as exc:
                diagnostics.append(f"artifact validation failed: {type(exc).__name__}: {exc}")
                return Gate1ExecutionResult(
                    Gate1Status.LIFT_REJECTED,
                    current,
                    source_compile,
                    source_fp,
                    tuple(history),
                    best,
                    leakage_report,
                    diagnostics=tuple(diagnostics),
                    composer_shadow=tuple(composer_shadow_records),
                )
            if not leakage_report.passed:
                diagnostics.extend(v.detail for v in leakage_report.violations)
                return Gate1ExecutionResult(
                    Gate1Status.LEAKAGE_REJECTED,
                    current,
                    source_compile,
                    source_fp,
                    tuple(history),
                    best,
                    leakage_report,
                    diagnostics=tuple(diagnostics),
                    composer_shadow=tuple(composer_shadow_records),
                )

            source_compile = self.compiler.compile(source, compile_case)
            if not source_compile.ok or not source_compile.ttir:
                diagnostics.append(source_compile.diagnostic)
                return Gate1ExecutionResult(
                    Gate1Status.COMPILE_FAILED,
                    current,
                    source_compile,
                    None,
                    tuple(history),
                    best,
                    leakage_report,
                    diagnostics=tuple(diagnostics),
                    composer_shadow=tuple(composer_shadow_records),
                )
            try:
                source_fp = fingerprint_ttir(
                    source_compile.ttir,
                    python_interface=_interface_for_fingerprint(
                        source_compile, compile_case, current
                    ),
                    schedule=compile_case.schedule,
                )
                _validate_compiled_interface(source_compile, source_fp, current)
            except (TTIRParseError, ValueError) as exc:
                diagnostics.append(f"source interface/TTIR validation failed: {exc}")
                return Gate1ExecutionResult(
                    Gate1Status.LIFT_REJECTED,
                    current,
                    source_compile,
                    None,
                    tuple(history),
                    best,
                    leakage_report,
                    diagnostics=tuple(diagnostics),
                    composer_shadow=tuple(composer_shadow_records),
                )

            stable = 0
            closed_record: CandidateRecord | None = None
            seen_generation_ids: set[str] = set()
            generation_round = 0
            round_best: CandidateRecord | None = None
            failed_record: CandidateRecord | None = None
            round_transport_failure = False

            # One fresh source-blind replay proposal is evaluated for this
            # artifact. Additional proposals exist only when an independent
            # stability confirmation is explicitly requested. A failure is
            # never repaired in-place; it returns to the outer Lift loop.
            while stable < self.config.stable_closures:
                generation: ReplayGeneration | None = None
                compile_result: CompileResult | None = None
                comparison: TTIRComparison | None = None
                diagnosis: Diagnosis | None = None
                specialization_evidence: Mapping[str, Any] | None = None
                specialization_hard_pass: bool | None = None
                effective_hard_difference_digest: str | None = None
                effective_hard_difference_categories: tuple[str, ...] = ()
                relift_guidance: ReliftGuidance | None = None
                relift_trigger: ReliftTrigger | None = None
                family_source_compile_failed = False
                family_setup_failed = False
                family_leakage_rejected = False
                try:
                    generation = writer.generate(
                        current,
                        registry,
                        masked=False,
                    )
                    if generation.generation_id in seen_generation_ids:
                        raise RuntimeError(
                            "replay writer reused generation_id; stability evidence "
                            "must come from an independent invocation"
                        )
                    seen_generation_ids.add(generation.generation_id)
                    function_name = getattr(
                        current.interface,
                        "function_name",
                        current.interface.kernel_id,
                    )
                    replay_source = KernelSource.from_replay(
                        generation.code,
                        function_name,
                        registry,
                    )
                    compile_result = self.compiler.compile(
                        replay_source, compile_case
                    )
                    if not compile_result.ok or not compile_result.ttir:
                        diagnosis = diagnose_compile_failure(
                            compile_result.diagnostic
                        )
                        relift_trigger = ReliftTrigger.REPLAY_COMPILE_FAILED
                        if specialization_family is not None:
                            try:
                                family_evaluation = (
                                    _evaluate_specialization_family(
                                        source=source,
                                        artifact=current,
                                        family=specialization_family,
                                        registry=registry,
                                        generation=generation,
                                        primary_source_compile=source_compile,
                                        primary_replay_compile=compile_result,
                                        primary_comparison=None,
                                        compiler=self.compiler,
                                        comparator=self.comparator,
                                        leakage_policy=self.leakage_policy,
                                    )
                                )
                            except Exception as exc:
                                family_setup_failed = True
                                diagnosis = Diagnosis(
                                    FailureCause.ARTIFACT_SCHEMA,
                                    Responsibility.LIFT_ARTIFACT,
                                    (f"{type(exc).__name__}: {exc}",),
                                    (),
                                    ("j", "strategy", "location_map"),
                                    "fixed-C family 无法按冻结 interface/C 建立"
                                    "完整 specialization sweep；不允许修 replay。",
                                )
                            else:
                                specialization_evidence = (
                                    family_evaluation.evidence
                                )
                                specialization_hard_pass = False
                                family_source_compile_failed = (
                                    family_evaluation.source_compile_failed
                                )
                                family_leakage_rejected = (
                                    family_evaluation.leakage_rejected
                                )
                                if family_evaluation.diagnosis is not None:
                                    diagnosis = family_evaluation.diagnosis
                                relift_guidance = (
                                    family_evaluation.relift_guidance
                                )
                                effective_hard_difference_digest = (
                                    family_evaluation.hard_difference_digest
                                )
                                effective_hard_difference_categories = (
                                    family_evaluation.hard_difference_categories
                                )
                    else:
                        replay_fp = fingerprint_ttir(
                            compile_result.ttir,
                            python_interface=_interface_for_fingerprint(
                                compile_result, compile_case, current
                            ),
                            schedule=compile_case.schedule,
                        )
                        try:
                            _validate_compiled_interface(
                                compile_result, replay_fp, current
                            )
                        except ValueError as exc:
                            diagnosis = Diagnosis(
                                FailureCause.INTERFACE,
                                Responsibility.LIFT_ARTIFACT,
                                (str(exc),),
                                (),
                                ("j", "strategy", "location_map"),
                                "重放接口不符合冻结 interface；禁止原地修签名，"
                                "回流 Lift 后从头重放。",
                            )
                        else:
                            comparison = self.comparator.compare(
                                source_fp, replay_fp
                            )
                            diagnosis = diagnose_comparison(comparison)
                            if (
                                specialization_family is None
                                and not comparison.hard_pass
                            ):
                                relift_guidance = _build_relift_guidance(
                                    current,
                                    comparison,
                                    source_fingerprint=source_fp,
                                    replay_fingerprint=replay_fp,
                                )
                                effective_hard_difference_digest = (
                                    relift_guidance.hard_difference_digest
                                )
                                effective_hard_difference_categories = (
                                    relift_guidance.failed_hard_categories
                                )
                                relift_trigger = ReliftTrigger.NORMAL_NO_CLOSURE
                            if specialization_family is not None:
                                try:
                                    family_evaluation = (
                                        _evaluate_specialization_family(
                                            source=source,
                                            artifact=current,
                                            family=specialization_family,
                                            registry=registry,
                                            generation=generation,
                                            primary_source_compile=source_compile,
                                            primary_replay_compile=compile_result,
                                            primary_comparison=comparison,
                                            compiler=self.compiler,
                                            comparator=self.comparator,
                                            leakage_policy=self.leakage_policy,
                                        )
                                    )
                                except Exception as exc:
                                    family_setup_failed = True
                                    diagnosis = Diagnosis(
                                        FailureCause.ARTIFACT_SCHEMA,
                                        Responsibility.LIFT_ARTIFACT,
                                        (f"{type(exc).__name__}: {exc}",),
                                        (),
                                        ("j", "strategy", "location_map"),
                                        "fixed-C family 无法按冻结 interface/C 建立完整"
                                        "specialization sweep；该失败属于 Lift artifact/"
                                        "family contract，不允许修 replay。",
                                    )
                                else:
                                    specialization_evidence = (
                                        family_evaluation.evidence
                                    )
                                    specialization_hard_pass = (
                                        family_evaluation.hard_pass
                                    )
                                    family_source_compile_failed = (
                                        family_evaluation.source_compile_failed
                                    )
                                    family_leakage_rejected = (
                                        family_evaluation.leakage_rejected
                                    )
                                    if family_evaluation.diagnosis is not None:
                                        diagnosis = family_evaluation.diagnosis
                                    relift_guidance = (
                                        family_evaluation.relift_guidance
                                    )
                                    effective_hard_difference_digest = (
                                        family_evaluation.hard_difference_digest
                                    )
                                    effective_hard_difference_categories = (
                                        family_evaluation.hard_difference_categories
                                    )
                                    relift_trigger = (
                                        family_evaluation.relift_trigger
                                    )
                                    if (
                                        family_evaluation.hard_pass
                                        and not comparison.hard_pass
                                    ):
                                        relift_guidance = _build_relift_guidance(
                                            current,
                                            comparison,
                                            source_fingerprint=source_fp,
                                            replay_fingerprint=replay_fp,
                                        )
                                        effective_hard_difference_digest = (
                                            relift_guidance.hard_difference_digest
                                        )
                                        effective_hard_difference_categories = (
                                            relift_guidance.failed_hard_categories
                                        )
                                        relift_trigger = (
                                            ReliftTrigger.NORMAL_NO_CLOSURE
                                        )
                except ProviderError as exc:
                    round_transport_failure = True
                    diagnosis = Diagnosis(
                        FailureCause.REPLAY_POLICY,
                        Responsibility.UNKNOWN,
                        (str(exc)[-2000:],),
                        (),
                        (),
                        "重写服务传输/配额不可用，尚未形成 replay candidate 或"
                        "Gate1 语义差分；该失败不能指导 J/C.strategy，因此不调用"
                        "Relift，也不把它伪装成语义修复。",
                    )
                except Exception as exc:
                    if diagnosis is None:
                        if generation is None or "generation_id" in str(exc):
                            diagnosis = Diagnosis(
                                FailureCause.REPLAY_POLICY,
                                Responsibility.LIFT_ARTIFACT,
                                (f"{type(exc).__name__}: {exc}",),
                                (),
                                ("j", "strategy", "location_map"),
                                "重写服务未产生可审计候选；将失败类别回流 Lift，"
                                "重新生成完整 artifact 后从头重放。",
                            )
                        else:
                            diagnosis = diagnose_compile_failure(
                                f"{type(exc).__name__}: {exc}"
                            )

                provisional = CandidateRecord(
                    relift_round,
                    generation_round,
                    generation,
                    compile_result,
                    comparison,
                    diagnosis,
                    specialization_evidence=specialization_evidence,
                    specialization_hard_pass=specialization_hard_pass,
                    effective_hard_difference_digest=(
                        effective_hard_difference_digest
                    ),
                    effective_hard_difference_categories=(
                        effective_hard_difference_categories
                    ),
                    relift_guidance=relift_guidance,
                    relift_trigger=relift_trigger,
                    artifact_digest=artifact_digest,
                )
                prior_passes = _passed_hard_categories(round_best)
                new_passes = _passed_hard_categories(provisional)
                regressed = bool(prior_passes - new_passes)
                adopted = (
                    round_best is None
                    or (
                        not regressed
                        and provisional.rank_tuple > round_best.rank_tuple
                    )
                )
                record = replace(
                    provisional,
                    adopted=adopted,
                    regressed=regressed,
                )
                history.append(record)
                if adopted:
                    round_best = record
                    if best is None or record.rank_tuple > best.rank_tuple:
                        best = record

                if family_setup_failed:
                    diagnostics.append(
                        "fixed-C specialization family setup failed before a "
                        "complete comparable sweep"
                    )
                    return Gate1ExecutionResult(
                        Gate1Status.LIFT_REJECTED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        diagnostics=tuple(diagnostics),
                        composer_shadow=tuple(composer_shadow_records),
                    )
                if family_source_compile_failed:
                    diagnostics.append(
                        "at least one trusted source specialization failed; "
                        "stopping without semantic Relift"
                    )
                    return Gate1ExecutionResult(
                        Gate1Status.COMPILE_FAILED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        diagnostics=tuple(diagnostics),
                        composer_shadow=tuple(composer_shadow_records),
                    )
                if family_leakage_rejected:
                    diagnostics.append(
                        "fixed-C specialization family static leakage check "
                        "rejected the artifact"
                    )
                    return Gate1ExecutionResult(
                        Gate1Status.LEAKAGE_REJECTED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        diagnostics=tuple(diagnostics),
                        composer_shadow=tuple(composer_shadow_records),
                    )

                if record.overall_hard_pass:
                    assert generation is not None
                    stable += 1
                    closed_record = record
                    generation_round += 1
                    continue

                # Any normal replay failure invalidates this artifact round.
                # The diagnosis/history are passed only to the source-aware
                # Lift agent; the failed replay is never a writer repair base.
                stable = 0
                closed_record = None
                failed_record = record
                if diagnosis is not None and diagnosis.cause is FailureCause.LOWERING:
                    diagnostics.append(diagnosis.explanation)
                break

            if closed_record is not None and stable >= self.config.stable_closures:
                intervention_records: tuple[InterventionRecord, ...] = ()
                paired_intervention_passed = False
                if self.config.run_semantic_interventions:
                    intervention_records = _run_semantic_interventions(
                        artifact=current,
                        baseline_record=closed_record,
                        compile_case=compile_case,
                        registry=registry,
                        writer=writer,
                        compiler=self.compiler,
                        seen_generation_ids=seen_generation_ids,
                        limit=self.config.max_semantic_interventions,
                        specialization_family=specialization_family,
                    )
                    if intervention_records:
                        paired_intervention_passed = all(
                            item.passed for item in intervention_records
                        )
                        if not paired_intervention_passed:
                            diagnostics.extend(
                                "paired semantic intervention "
                                f"{item.spec.intervention_id} failed: "
                                f"{item.status.value}: {item.diagnostic}"
                                for item in intervention_records
                                if not item.passed
                            )
                            family_semantic_failure = any(
                                item.status
                                in INTERVENTION_SEMANTIC_FAILURE_STATUSES
                                or any(
                                    profile.status
                                    in INTERVENTION_SEMANTIC_FAILURE_STATUSES
                                    for profile in item.profile_records
                                )
                                for item in intervention_records
                            )
                            if (
                                specialization_family is not None
                                and not family_semantic_failure
                            ):
                                diagnostics.append(
                                    "fixed-C paired intervention evidence is "
                                    "incomplete; writer/compile/fingerprint "
                                    "failure is not a causal leakage verdict"
                                )
                                return Gate1ExecutionResult(
                                    Gate1Status.NO_CLOSURE,
                                    current,
                                    source_compile,
                                    source_fp,
                                    tuple(history),
                                    best,
                                    leakage_report,
                                    diagnostics=tuple(diagnostics),
                                    composer_shadow=tuple(
                                        composer_shadow_records
                                    ),
                                    interventions=intervention_records,
                                    anti_leakage_basis=(
                                        "paired_intervention_fixed_c_family_incomplete"
                                    ),
                                    termination=_termination_for_round(
                                        Gate1TerminationReason.INTERVENTION_EVIDENCE_INCOMPLETE,
                                        relift_round=relift_round,
                                        history=history,
                                        config=self.config,
                                    ),
                                )
                            return Gate1ExecutionResult(
                                Gate1Status.LEAKAGE_REJECTED,
                                current,
                                source_compile,
                                source_fp,
                                tuple(history),
                                best,
                                leakage_report,
                                diagnostics=tuple(diagnostics),
                                composer_shadow=tuple(composer_shadow_records),
                                interventions=intervention_records,
                                anti_leakage_basis=(
                                    "paired_intervention_fixed_c_family"
                                    if specialization_family is not None
                                    else "paired_intervention"
                                ),
                            )
                        diagnostics.append(
                            "paired semantic intervention passed with an exact "
                            "registered TTIR delta"
                        )
                masked_probe: _MaskedJProbe | None = None
                if self.config.run_masked_ablation:
                    assert source_compile is not None
                    masked_probe = _run_masked_j_probe(
                        source=source,
                        artifact=current,
                        source_compile=source_compile,
                        source_fingerprint=source_fp,
                        compile_case=compile_case,
                        registry=registry,
                        writer=writer,
                        compiler=self.compiler,
                        comparator=self.comparator,
                        seen_generation_ids=seen_generation_ids,
                        specialization_family=specialization_family,
                    )
                    diagnostics.append(masked_probe.diagnostic)
                    if specialization_family is not None:
                        if masked_probe.hard_closed:
                            return Gate1ExecutionResult(
                                Gate1Status.LEAKAGE_REJECTED,
                                current,
                                source_compile,
                                source_fp,
                                tuple(history),
                                best,
                                leakage_report,
                                masked_compile=masked_probe.compile,
                                masked_comparison=masked_probe.comparison,
                                masked_outcome=masked_probe.outcome,
                                diagnostics=tuple(diagnostics),
                                masked_generation=masked_probe.generation,
                                interventions=intervention_records,
                                anti_leakage_basis=(
                                    "masked_j_full_pi_fixed_c_family_hard_closure"
                                ),
                                composer_shadow=tuple(
                                    composer_shadow_records
                                ),
                                masked_family_evidence=(
                                    masked_probe.family_to_json()
                                ),
                            )
                        if not masked_probe.family_complete_hard_mismatch:
                            diagnostics.append(
                                "Gate1 fixed-C family anti-leakage evidence "
                                "is incomplete; every declared profile must "
                                "produce a legal comparable masked-J hard "
                                "mismatch"
                            )
                            return Gate1ExecutionResult(
                                Gate1Status.NO_CLOSURE,
                                current,
                                source_compile,
                                source_fp,
                                tuple(history),
                                best,
                                leakage_report,
                                masked_compile=masked_probe.compile,
                                masked_comparison=masked_probe.comparison,
                                masked_outcome=masked_probe.outcome,
                                diagnostics=tuple(diagnostics),
                                masked_generation=masked_probe.generation,
                                interventions=intervention_records,
                                anti_leakage_basis=(
                                    "masked_j_full_pi_fixed_c_family_incomplete"
                                ),
                                composer_shadow=tuple(
                                    composer_shadow_records
                                ),
                                termination=_termination_for_round(
                                    Gate1TerminationReason.MASKED_EVIDENCE_INCOMPLETE,
                                    relift_round=relift_round,
                                    history=history,
                                    config=self.config,
                                ),
                                masked_family_evidence=(
                                    masked_probe.family_to_json()
                                ),
                            )
                    if (
                        specialization_family is None
                        and masked_probe.outcome not in {
                            "hard_closure",
                            "hard_mismatch",
                            "compile_nonclosure",
                        }
                    ):
                        diagnostics.append(
                            "Gate1 anti-leakage evidence is incomplete; a "
                            "normal closure cannot substitute for an unevaluated "
                            "masked-J/full-pi attack"
                        )
                        return Gate1ExecutionResult(
                            Gate1Status.NO_CLOSURE,
                            current,
                            source_compile,
                            source_fp,
                            tuple(history),
                            best,
                            leakage_report,
                            masked_compile=masked_probe.compile,
                            masked_comparison=masked_probe.comparison,
                            masked_outcome=masked_probe.outcome,
                            diagnostics=tuple(diagnostics),
                            masked_generation=masked_probe.generation,
                            interventions=intervention_records,
                            anti_leakage_basis="masked_j_full_pi_unevaluated",
                            composer_shadow=tuple(composer_shadow_records),
                            termination=_termination_for_round(
                                Gate1TerminationReason.MASKED_EVIDENCE_INCOMPLETE,
                                relift_round=relift_round,
                                history=history,
                                config=self.config,
                            ),
                            masked_family_evidence=(
                                masked_probe.family_to_json()
                            ),
                        )
                    if (
                        specialization_family is None
                        and masked_probe.hard_closed
                    ):
                        return Gate1ExecutionResult(
                            Gate1Status.LEAKAGE_REJECTED,
                            current,
                            source_compile,
                            source_fp,
                            tuple(history),
                            best,
                            leakage_report,
                            masked_compile=masked_probe.compile,
                            masked_comparison=masked_probe.comparison,
                            masked_outcome=masked_probe.outcome,
                            diagnostics=tuple(diagnostics),
                            masked_generation=masked_probe.generation,
                            interventions=intervention_records,
                            anti_leakage_basis="masked_j_full_pi_hard_closure",
                            composer_shadow=tuple(composer_shadow_records),
                            masked_family_evidence=(
                                masked_probe.family_to_json()
                            ),
                        )
                anti_leakage_parts = ["compose_equality"]
                if paired_intervention_passed:
                    anti_leakage_parts.append(
                        "paired_intervention_fixed_c_family"
                        if specialization_family is not None
                        else "paired_intervention"
                    )
                if masked_probe is not None:
                    anti_leakage_parts.append(
                        "masked_j_full_pi_fixed_c_family_hard_mismatch"
                        if specialization_family is not None
                        else "masked_j_full_pi_nonclosure"
                    )
                return Gate1ExecutionResult(
                    Gate1Status.REPLAY_CLOSED,
                    current,
                    source_compile,
                    source_fp,
                    tuple(history),
                    best,
                    leakage_report,
                    masked_compile=(
                        None if masked_probe is None else masked_probe.compile
                    ),
                    masked_comparison=(
                        None if masked_probe is None else masked_probe.comparison
                    ),
                    masked_outcome=(
                        None if masked_probe is None else masked_probe.outcome
                    ),
                    diagnostics=tuple(diagnostics),
                    masked_generation=(
                        None if masked_probe is None else masked_probe.generation
                    ),
                    interventions=intervention_records,
                    anti_leakage_basis="+".join(anti_leakage_parts),
                    composer_shadow=tuple(composer_shadow_records),
                    masked_family_evidence=(
                        None
                        if masked_probe is None
                        else masked_probe.family_to_json()
                    ),
                )

            if round_transport_failure:
                diagnostics.append(
                    "replay provider unavailable before an auditable candidate; "
                    "stopping without semantic Relift because no Gate1 hard "
                    "difference exists"
                )
                termination = _termination_for_round(
                    Gate1TerminationReason.REPLAY_PROVIDER_UNAVAILABLE,
                    relift_round=relift_round,
                    history=history,
                    config=self.config,
                )
                break

            # This is an outer-loop progress check, not replay repair.  A
            # state becomes comparable only after a fresh normal replay has
            # compiled and produced the deterministic hard comparison.  An
            # unchanged artifact alone never stops the next writer call.
            if (
                artifact_digest is not None
                and failed_record is not None
                and failed_record.effective_hard_difference_digest is not None
            ):
                hard_difference_digest = (
                    failed_record.effective_hard_difference_digest
                )
                hard_categories = (
                    failed_record.effective_hard_difference_categories
                )
                state = (artifact_digest, hard_difference_digest)
                if (
                    previous_stagnation_state is not None
                    and state == previous_stagnation_state[:2]
                ):
                    stagnation = StagnationEvidence(
                        first_relift_round=previous_stagnation_state[2],
                        repeated_relift_round=relift_round,
                        artifact_digest=artifact_digest,
                        hard_difference_digest=hard_difference_digest,
                        hard_difference_categories=hard_categories,
                    )
                    diagnostics.append(
                        "deterministic outer-loop stagnation: canonical "
                        "Lift artifact and exact hard-difference signature "
                        f"were unchanged in rounds "
                        f"{stagnation.first_relift_round} and "
                        f"{stagnation.repeated_relift_round}; stopping without "
                        "another Relift or replay generation"
                    )
                    termination = _termination_for_round(
                        Gate1TerminationReason.DETERMINISTIC_STAGNATION,
                        relift_round=relift_round,
                        history=history,
                        config=self.config,
                        hard_difference_categories=hard_categories,
                    )
                    break
                previous_stagnation_state = (
                    artifact_digest,
                    hard_difference_digest,
                    relift_round,
                    hard_categories,
                )
            else:
                # Generation, compile, interface, and parser failures do not
                # provide a deterministic hard-diff state.  They break the
                # consecutive stagnation chain rather than hashing error text.
                previous_stagnation_state = None

            if relift_round + 1 >= self.config.max_relifts:
                termination = _termination_for_round(
                    Gate1TerminationReason.RELIFT_BUDGET_EXHAUSTED,
                    relift_round=relift_round,
                    history=history,
                    config=self.config,
                    hard_difference_categories=(
                        ()
                        if failed_record is None
                        else failed_record.effective_hard_difference_categories
                    ),
                )
                break
            if relifter is None:
                termination = _termination_for_round(
                    Gate1TerminationReason.RELIFTER_UNAVAILABLE,
                    relift_round=relift_round,
                    history=history,
                    config=self.config,
                )
                break
            assert source_fp is not None
            round_records = tuple(
                record
                for record in history
                if record.relift_round == relift_round
            )
            request = _relift_request_for_round(
                relift_round, round_records, round_best
            )
            if (
                failed_record is not None
                and failed_record.relift_guidance is not None
            ):
                guidance = failed_record.relift_guidance
                request = replace(
                    request,
                    hard_difference_categories=(
                        guidance.failed_hard_categories
                    ),
                    semantic_guidance=guidance,
                )
            diagnostics.append(
                "normal replay failed; restarting from source-aware Lift with "
                f"trigger={request.trigger.value}"
            )
            try:
                relift_evidence = tuple(
                    ReliftEvidence.from_candidate(record)
                    for record in history
                )
                current = relifter.relift(
                    current,
                    source_fingerprint=source_fp,
                    history=relift_evidence,
                    request=request,
                )
            except Exception as exc:
                diagnostics.append(
                    "semantic relift failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                termination = _termination_for_round(
                    Gate1TerminationReason.RELIFT_FAILED,
                    relift_round=relift_round,
                    history=history,
                    config=self.config,
                    hard_difference_categories=(
                        request.hard_difference_categories
                    ),
                )
                break

        if termination is None:
            termination = _termination_for_round(
                Gate1TerminationReason.RELIFT_BUDGET_EXHAUSTED,
                relift_round=max(self.config.max_relifts - 1, 0),
                history=history,
                config=self.config,
            )
        return Gate1ExecutionResult(
            Gate1Status.NO_CLOSURE,
            current,
            source_compile,
            source_fp,
            tuple(history),
            best,
            leakage_report,
            diagnostics=tuple(diagnostics),
            composer_shadow=tuple(composer_shadow_records),
            stagnation=stagnation,
            termination=termination,
        )


__all__ = [
    "CandidateRecord",
    "ComposerShadowRecord",
    "ComposerShadowStatus",
    "Gate1Config",
    "Gate1ExecutionResult",
    "Gate1Runner",
    "Gate1Termination",
    "Gate1TerminationReason",
    "GATE1_STAGNATION_POLICY",
    "GATE1_TERMINATION_POLICY",
    "GATE1_FAMILY_ANTI_LEAKAGE_POLICY",
    "MASKED_J_FAMILY_POLICY",
    "MASKED_J_POLICY",
    "RELIFT_GUIDANCE_POLICY",
    "RELIFT_STRUCTURAL_GUIDANCE_POLICY",
    "ReliftEvidence",
    "ReliftGuidance",
    "ReliftRequest",
    "ReliftTrigger",
    "Relifter",
    "StagnationEvidence",
    "canonical_artifact_digest",
]
