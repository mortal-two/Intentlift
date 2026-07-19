"""Constraint-shaped schedule candidate planning for the CUDA backend.

The backend separates three decisions that must not be conflated:

* backend legality preserves the binding A/B program and target constraints;
* accepted A/B/C constraints and Layer-C hints reduce the legal schedule space;
* measured tuning and host dispatch choose among the materialized variants.

This module implements only the first two decisions.  It does not validate an
IntentIR program, compile a kernel, benchmark a candidate, or select a runtime
variant.  Keeping those boundaries explicit makes the
``constraints_off`` ablation well-defined: it bypasses constraint pruning while
retaining the same raw candidates and backend-legality predicate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


ScheduleScalar = bool | int | float | str
LegalityPredicate = Callable[["ScheduleCandidate"], bool | tuple[bool, str]]


class VariantPlanningError(ValueError):
    """Raised when a candidate space or constraints policy is malformed."""


def _require_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VariantPlanningError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_scalar(value: Any, *, field_name: str) -> ScheduleScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)) and not isinstance(value, complex):
        if isinstance(value, str) and not value:
            raise VariantPlanningError(f"{field_name} must not be empty")
        return value
    raise VariantPlanningError(
        f"{field_name} must be a bool, int, float, or non-empty string"
    )


def _canonical_parameters(
    parameters: Mapping[str, ScheduleScalar],
) -> tuple[tuple[str, ScheduleScalar], ...]:
    if not isinstance(parameters, Mapping):
        raise VariantPlanningError("schedule parameters must be a mapping")
    canonical: list[tuple[str, ScheduleScalar]] = []
    for raw_name, raw_value in parameters.items():
        name = _require_identifier(raw_name, field_name="parameter name")
        value = _require_scalar(raw_value, field_name=f"parameter {name!r}")
        canonical.append((name, value))
    if not canonical:
        raise VariantPlanningError("schedule parameters must not be empty")
    canonical.sort(key=lambda item: item[0])
    return tuple(canonical)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class ScheduleCandidate:
    """One schedule-only variant of an unchanged binding A/B program."""

    variant_id: str
    parameters: Mapping[str, ScheduleScalar]
    source_seed: bool = False
    tags: tuple[str, ...] = ()
    _canonical: tuple[tuple[str, ScheduleScalar], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_id",
            _require_identifier(self.variant_id, field_name="variant_id"),
        )
        canonical = _canonical_parameters(self.parameters)
        object.__setattr__(self, "_canonical", canonical)
        object.__setattr__(self, "parameters", dict(canonical))
        normalized_tags = tuple(
            sorted(
                {
                    _require_identifier(tag, field_name="candidate tag")
                    for tag in self.tags
                }
            )
        )
        object.__setattr__(self, "tags", normalized_tags)

    @property
    def parameter_key(self) -> tuple[tuple[str, ScheduleScalar], ...]:
        return self._canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parameters": dict(self._canonical),
            "source_seed": self.source_seed,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class SelectionConstraints:
    """Auditable restrictions on schedule parameters.

    A Gate or frontend adapter derives these restrictions from
    access witnesses, stride/layout facts, and Layer-C hints.  The planner does
    not infer semantic facts itself.  Every policy must therefore carry the
    constraints references that justify it.
    """

    allowed_parameter_values: Mapping[str, Sequence[ScheduleScalar]]
    required_tags: tuple[str, ...] = ()
    constraint_references: tuple[str, ...] = ()
    policy_id: str = "backend.constraint-search.v1"
    _allowed: tuple[tuple[str, tuple[ScheduleScalar, ...]], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_id",
            _require_identifier(self.policy_id, field_name="policy_id"),
        )
        if not isinstance(self.allowed_parameter_values, Mapping):
            raise VariantPlanningError("allowed_parameter_values must be a mapping")
        allowed: list[tuple[str, tuple[ScheduleScalar, ...]]] = []
        for raw_name, raw_values in self.allowed_parameter_values.items():
            name = _require_identifier(raw_name, field_name="constraints parameter")
            if isinstance(raw_values, (str, bytes)) or not isinstance(
                raw_values, Sequence
            ):
                raise VariantPlanningError(
                    f"allowed values for {name!r} must be a sequence"
                )
            values: list[ScheduleScalar] = []
            for value in raw_values:
                normalized = _require_scalar(
                    value, field_name=f"allowed value for {name!r}"
                )
                if normalized not in values:
                    values.append(normalized)
            if not values:
                raise VariantPlanningError(
                    f"allowed values for {name!r} must not be empty"
                )
            allowed.append((name, tuple(values)))
        if not allowed:
            raise VariantPlanningError(
                "constraints pruning requires at least one parameter restriction"
            )
        allowed.sort(key=lambda item: item[0])
        object.__setattr__(self, "_allowed", tuple(allowed))
        object.__setattr__(
            self, "allowed_parameter_values", dict(self._allowed)
        )

        tags = tuple(
            sorted(
                {
                    _require_identifier(tag, field_name="required constraints tag")
                    for tag in self.required_tags
                }
            )
        )
        object.__setattr__(self, "required_tags", tags)
        references = tuple(
            sorted(
                {
                    _require_identifier(
                        reference, field_name="constraints reference"
                    )
                    for reference in self.constraint_references
                }
            )
        )
        if not references:
            raise VariantPlanningError(
                "constraints pruning policy requires source constraints references"
            )
        object.__setattr__(self, "constraint_references", references)

    def rejection_reasons(self, candidate: ScheduleCandidate) -> tuple[str, ...]:
        reasons: list[str] = []
        for name, allowed in self._allowed:
            if name not in candidate.parameters:
                reasons.append(f"missing_parameter:{name}")
                continue
            if candidate.parameters[name] not in allowed:
                rendered = ",".join(str(value) for value in allowed)
                reasons.append(f"{name}:not_in:[{rendered}]")
        missing_tags = sorted(set(self.required_tags) - set(candidate.tags))
        reasons.extend(f"missing_tag:{tag}" for tag in missing_tags)
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "allowed_parameter_values": {
                name: list(values) for name, values in self._allowed
            },
            "required_tags": list(self.required_tags),
            "constraint_references": list(self.constraint_references),
        }


@dataclass(frozen=True)
class CandidateRejection:
    variant_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant_id, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class VariantPlan:
    """The materialization input and its complete pruning audit."""

    schema_version: str
    family: str
    constraints_enabled: bool
    source_seed_variant_id: str
    raw_candidates: tuple[ScheduleCandidate, ...]
    legal_candidates: tuple[ScheduleCandidate, ...]
    materialization_candidates: tuple[ScheduleCandidate, ...]
    backend_rejections: tuple[CandidateRejection, ...]
    constraint_rejections: tuple[CandidateRejection, ...]
    constraint_policy: SelectionConstraints | None
    planning_time_ms: float
    deterministic_digest: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            "raw": len(self.raw_candidates),
            "backend_legal": len(self.legal_candidates),
            "materialize": len(self.materialization_candidates),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "constraints_enabled": self.constraints_enabled,
            "source_seed_variant_id": self.source_seed_variant_id,
            "counts": self.counts,
            "raw_candidates": [item.to_dict() for item in self.raw_candidates],
            "legal_candidates": [
                item.to_dict() for item in self.legal_candidates
            ],
            "materialization_candidates": [
                item.to_dict() for item in self.materialization_candidates
            ],
            "backend_rejections": [
                item.to_dict() for item in self.backend_rejections
            ],
            "constraint_rejections": [
                item.to_dict() for item in self.constraint_rejections
            ],
            "constraint_policy": (
                self.constraint_policy.to_dict()
                if self.constraint_policy is not None
                else None
            ),
            "planning_time_ms": self.planning_time_ms,
            "deterministic_digest": self.deterministic_digest,
        }


def _stable_variant_id(
    family: str, parameters: Mapping[str, ScheduleScalar]
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {"family": family, "parameters": dict(_canonical_parameters(parameters))}
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{family}.variant.{digest}"


def enumerate_schedule_candidates(
    *,
    family: str,
    axes: Mapping[str, Sequence[ScheduleScalar]],
    source_schedule: Mapping[str, ScheduleScalar],
    tags: Sequence[str] = (),
) -> tuple[ScheduleCandidate, ...]:
    """Build a deterministic Cartesian space plus an explicit freeze seed."""

    family = _require_identifier(family, field_name="family")
    if not isinstance(axes, Mapping) or not axes:
        raise VariantPlanningError("candidate axes must be a non-empty mapping")
    normalized_axes: list[tuple[str, tuple[ScheduleScalar, ...]]] = []
    for raw_name, raw_values in axes.items():
        name = _require_identifier(raw_name, field_name="candidate axis")
        if isinstance(raw_values, (str, bytes)) or not isinstance(
            raw_values, Sequence
        ):
            raise VariantPlanningError(f"candidate axis {name!r} must be a sequence")
        values: list[ScheduleScalar] = []
        for value in raw_values:
            normalized = _require_scalar(value, field_name=f"axis {name!r}")
            if normalized not in values:
                values.append(normalized)
        if not values:
            raise VariantPlanningError(f"candidate axis {name!r} must not be empty")
        normalized_axes.append((name, tuple(values)))
    normalized_axes.sort(key=lambda item: item[0])

    source_key = _canonical_parameters(source_schedule)
    axis_names = tuple(name for name, _ in normalized_axes)
    if tuple(name for name, _ in source_key) != axis_names:
        raise VariantPlanningError(
            "source_schedule keys must exactly match the candidate axes"
        )
    normalized_tags = tuple(tags)
    source = ScheduleCandidate(
        variant_id=f"{family}.source-seed",
        parameters=dict(source_key),
        source_seed=True,
        tags=normalized_tags,
    )

    candidates: list[ScheduleCandidate] = [source]
    seen = {source.parameter_key}
    for combination in itertools.product(
        *(values for _, values in normalized_axes)
    ):
        parameters = dict(zip(axis_names, combination, strict=True))
        key = _canonical_parameters(parameters)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            ScheduleCandidate(
                variant_id=_stable_variant_id(family, parameters),
                parameters=parameters,
                tags=normalized_tags,
            )
        )
    return tuple(candidates)


def _legality_result(
    predicate: LegalityPredicate, candidate: ScheduleCandidate
) -> tuple[bool, str]:
    result = predicate(candidate)
    if isinstance(result, bool):
        return result, "backend_predicate_rejected" if not result else ""
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], bool)
        and isinstance(result[1], str)
    ):
        accepted, reason = result
        if not accepted and not reason.strip():
            raise VariantPlanningError(
                "backend legality rejection must include a reason"
            )
        return accepted, reason.strip()
    raise VariantPlanningError(
        "backend legality predicate must return bool or (bool, reason)"
    )


def plan_variant_space(
    *,
    family: str,
    axes: Mapping[str, Sequence[ScheduleScalar]],
    source_schedule: Mapping[str, ScheduleScalar],
    backend_legal: LegalityPredicate,
    constraints_enabled: bool,
    constraints: SelectionConstraints | None,
    tags: Sequence[str] = (),
) -> VariantPlan:
    """Create the backend candidate set without compiling or benchmarking it."""

    started = time.perf_counter_ns()
    if not callable(backend_legal):
        raise VariantPlanningError("backend_legal must be callable")
    if constraints_enabled and constraints is None:
        raise VariantPlanningError(
            "constraints-on planning requires an auditable constraints policy"
        )

    raw = enumerate_schedule_candidates(
        family=family,
        axes=axes,
        source_schedule=source_schedule,
        tags=tags,
    )
    legal: list[ScheduleCandidate] = []
    backend_rejections: list[CandidateRejection] = []
    for candidate in raw:
        accepted, reason = _legality_result(backend_legal, candidate)
        if accepted:
            legal.append(candidate)
        else:
            backend_rejections.append(
                CandidateRejection(candidate.variant_id, (reason,))
            )

    source_seed_id = raw[0].variant_id
    if not any(candidate.source_seed for candidate in legal):
        raise VariantPlanningError(
            "the source freeze schedule is backend-illegal; freeze/retune comparison "
            "cannot use this candidate space"
        )

    materialize: list[ScheduleCandidate] = []
    constraint_rejections: list[CandidateRejection] = []
    for candidate in legal:
        reasons = (
            constraints.rejection_reasons(candidate)
            if constraints_enabled
            and constraints is not None
            and not candidate.source_seed
            else ()
        )
        if reasons:
            constraint_rejections.append(
                CandidateRejection(candidate.variant_id, reasons)
            )
        else:
            materialize.append(candidate)

    if not materialize:
        raise VariantPlanningError("candidate planning produced no materializable variant")

    deterministic_payload = {
        "schema_version": "intentir.backend.variant-plan.v1",
        "family": family,
        "constraints_enabled": constraints_enabled,
        "source_seed_variant_id": source_seed_id,
        "raw_candidates": [candidate.to_dict() for candidate in raw],
        "legal_candidates": [candidate.to_dict() for candidate in legal],
        "materialization_candidates": [
            candidate.to_dict() for candidate in materialize
        ],
        "backend_rejections": [item.to_dict() for item in backend_rejections],
        "constraint_rejections": [
            item.to_dict() for item in constraint_rejections
        ],
        "constraint_policy": (
            constraints.to_dict()
            if constraints_enabled and constraints is not None
            else None
        ),
    }
    digest = hashlib.sha256(
        _canonical_json(deterministic_payload).encode("utf-8")
    ).hexdigest()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return VariantPlan(
        schema_version="intentir.backend.variant-plan.v1",
        family=family,
        constraints_enabled=constraints_enabled,
        source_seed_variant_id=source_seed_id,
        raw_candidates=raw,
        legal_candidates=tuple(legal),
        materialization_candidates=tuple(materialize),
        backend_rejections=tuple(backend_rejections),
        constraint_rejections=tuple(constraint_rejections),
        constraint_policy=constraints if constraints_enabled else None,
        planning_time_ms=elapsed_ms,
        deterministic_digest=digest,
    )
