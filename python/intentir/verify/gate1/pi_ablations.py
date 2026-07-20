"""Executable π-necessity probes and C.strategy causal interventions.

Canonical π equality remains an admission invariant. Necessity probes are
deliberate post-admission counterfactuals: an incomplete plan reaches a fresh
source-blind writer, compilation, and the same score-free hard comparator.
Writer-front canonicality rejection is never counted as causal evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .compare import TTIRComparator, TTIRComparison
from .composer import assert_replay_plan_matches, materialize_replay_plan
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .model import (
    Gate1Artifact,
    ReplayPlan,
    StrategyBinding,
    StrategyChoice,
    StrategyFamily,
    StrategySpec,
)
from .replay import ReplayGeneration, ReplayWriter
from .rules import RuleAblationRole, RuleRegistry, RuleSelectionMode
from .ttir import TTIRFingerprint, fingerprint_ttir


class PiAblationKind(str, Enum):
    J_ONLY = "j_only"
    PLAN_STEP_DROPOUT = "plan_step_dropout"
    PLAN_MECHANISM_SUBSTITUTION = "plan_mechanism_substitution"


class PiAblationStatus(str, Enum):
    DEGRADED = "degraded"
    UNEXPECTED_CLOSURE = "unexpected_closure"
    CANONICALITY_REJECTED = "canonicality_rejected"
    SEMANTICS_PRESERVED = "semantics_preserved"
    OFF_TARGET_DRIFT = "off_target_drift"
    GENERATION_FAILED = "generation_failed"
    COMPILE_FAILED = "compile_failed"
    EVALUATION_FAILED = "evaluation_failed"


@dataclass(frozen=True)
class PiStepClassification:
    step_index: int
    rule: str
    target: str | None
    role: RuleAblationRole

    def to_json(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "rule": self.rule,
            "target": self.target,
            "role": self.role.value,
        }


@dataclass(frozen=True)
class PiAblationSpec:
    ablation_id: str
    kind: PiAblationKind
    plan: ReplayPlan
    dropped_step_index: int | None = None
    dropped_rule: str | None = None
    dropped_target: str | None = None
    replacement_rule: str | None = None
    artifact: Gate1Artifact | None = None
    strategy_before: dict[str, Any] | None = None
    strategy_after: dict[str, Any] | None = None
    dropped_rule_instances_remaining: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "kind": self.kind.value,
            "plan": self.plan.to_json(),
            "dropped_step_index": self.dropped_step_index,
            "dropped_rule": self.dropped_rule,
            "dropped_target": self.dropped_target,
            "replacement_rule": self.replacement_rule,
            "strategy_before": self.strategy_before,
            "strategy_after": self.strategy_after,
            "dropped_rule_instances_remaining": (
                self.dropped_rule_instances_remaining
            ),
        }


@dataclass(frozen=True)
class PiAblationRecord:
    spec: PiAblationSpec
    status: PiAblationStatus
    generation: ReplayGeneration | None = None
    compile: CompileResult | None = None
    comparison: TTIRComparison | None = None
    diagnostic: str = ""
    attempts: int = 1

    @property
    def supports_pi_necessity(self) -> bool:
        return self.status is PiAblationStatus.DEGRADED

    @property
    def necessity_evidence_complete(self) -> bool:
        if self.spec.kind not in {
            PiAblationKind.J_ONLY,
            PiAblationKind.PLAN_STEP_DROPOUT,
        }:
            return True
        return self.status in {
            PiAblationStatus.DEGRADED,
            PiAblationStatus.UNEXPECTED_CLOSURE,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_json(),
            "status": self.status.value,
            "supports_pi_necessity": self.supports_pi_necessity,
            "necessity_evidence_complete": self.necessity_evidence_complete,
            "generation": None
            if self.generation is None
            else {
                "prompt_sha256": self.generation.prompt_sha256,
                "generation_id": self.generation.generation_id,
                "code_sha256": self.generation.code_sha256,
                "response_model": self.generation.response_model,
                "response_id": self.generation.response_id,
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
            },
            "comparison": None
            if self.comparison is None
            else self.comparison.to_dict(),
            "diagnostic": self.diagnostic,
            "attempts": self.attempts,
        }


_MECHANISM_SUBSTITUTIONS = {
    "materialize_index_tile": "materialize_pointer_tile",
    "materialize_pointer_tile": "materialize_index_tile",
    "bind_program_domain": "bind_wide_program_domain",
    "bind_wide_program_domain": "bind_program_domain",
    "realize_reduce": "realize_flattened_reduce",
    "realize_flattened_reduce": "realize_reduce",
    "realize_multi_tile_reduce": "realize_reduce",
}


def classify_pi_steps(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> tuple[PiStepClassification, ...]:
    """Classify every canonical π step using reviewed RuleCard metadata."""

    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )
    assert_replay_plan_matches(artifact, registry)
    return tuple(
        PiStepClassification(
            step_index=index,
            rule=step.rule,
            target=step.target_ref,
            role=registry.get(
                step.rule, frontend=artifact.interface.frontend
            ).ablation_role,
        )
        for index, step in enumerate(artifact.plan.steps)
    )


def build_pi_ablations(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    limit: int | None = None,
    kinds: tuple[PiAblationKind, ...] | None = None,
) -> tuple[PiAblationSpec, ...]:
    """Build J-only and exhaustive causal-step counterfactuals.

    The limit exists only for explicitly bounded diagnostics. A formal Gate1
    call leaves it unset so every RuleCard-classified causal step is tested.
    Administrative steps remain visible through classify_pi_steps but are not
    misrepresented as semantic necessity probes.
    """

    if limit is not None and limit < 0:
        raise ValueError("pi ablation limit must be non-negative")
    if limit == 0:
        return ()
    requested = tuple(
        kinds
        if kinds is not None
        else (
            PiAblationKind.J_ONLY,
            PiAblationKind.PLAN_STEP_DROPOUT,
        )
    )
    if len(set(requested)) != len(requested):
        raise ValueError("pi ablation kinds must be unique")
    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    classifications = classify_pi_steps(artifact, registry)

    specs: list[PiAblationSpec] = []

    def has_capacity() -> bool:
        return limit is None or len(specs) < limit

    has_causal_steps = any(
        item.role is RuleAblationRole.CAUSAL for item in classifications
    )
    if (
        PiAblationKind.J_ONLY in requested
        and has_causal_steps
        and has_capacity()
    ):
        specs.append(
            PiAblationSpec(
                "j_only",
                PiAblationKind.J_ONLY,
                ReplayPlan([]),
            )
        )

    if PiAblationKind.PLAN_STEP_DROPOUT in requested:
        for classification in classifications:
            if not has_capacity():
                break
            if classification.role is RuleAblationRole.ADMINISTRATIVE:
                continue
            dropped_index = classification.step_index
            dropped = artifact.plan.steps[dropped_index]
            remaining_steps = [
                step
                for index, step in enumerate(artifact.plan.steps)
                if index != dropped_index
            ]
            specs.append(
                PiAblationSpec(
                    (
                        f"drop_step:{dropped_index}:{dropped.rule}:"
                        f"{dropped.target_ref}"
                    ),
                    PiAblationKind.PLAN_STEP_DROPOUT,
                    ReplayPlan(remaining_steps),
                    dropped_step_index=dropped_index,
                    dropped_rule=dropped.rule,
                    dropped_target=dropped.target_ref,
                    dropped_rule_instances_remaining=sum(
                        step.rule == dropped.rule for step in remaining_steps
                    ),
                )
            )

    if (
        PiAblationKind.PLAN_MECHANISM_SUBSTITUTION in requested
        and has_capacity()
    ):
        candidates = [
            index
            for index, step in enumerate(artifact.plan.steps)
            if step.rule in _MECHANISM_SUBSTITUTIONS
        ]
        if candidates:
            substitution_index = min(
                candidates,
                key=lambda index: (
                    0
                    if artifact.plan.steps[index].rule
                    in {"materialize_index_tile", "materialize_pointer_tile"}
                    else 1,
                    index,
                ),
            )
            original = artifact.plan.steps[substitution_index]
            replacement = _MECHANISM_SUBSTITUTIONS[original.rule]
            family_choice = {
                "bind_program_domain": (
                    StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                    StrategyChoice.WIDE,
                ),
                "materialize_index_tile": (
                    StrategyFamily.ADDRESS_ASSOCIATION,
                    StrategyChoice.POINTER_STAGED,
                ),
                "realize_reduce": (
                    StrategyFamily.REDUCTION_CALL_SHAPE,
                    StrategyChoice.FLATTENED,
                ),
            }.get(original.rule)
            bindings = list(artifact.strategy.bindings)
            if family_choice is not None:
                family, choice = family_choice
                bindings = [
                    binding
                    for binding in bindings
                    if not (
                        binding.target_ref == original.target_ref
                        and binding.family is family
                    )
                ]
                bindings.append(
                    StrategyBinding(original.target_ref, family, choice)
                )
            else:
                reverse_family = {
                    "bind_wide_program_domain": (
                        StrategyFamily.PROGRAM_COORDINATE_WIDTH
                    ),
                    "materialize_pointer_tile": (
                        StrategyFamily.ADDRESS_ASSOCIATION
                    ),
                    "realize_flattened_reduce": (
                        StrategyFamily.REDUCTION_CALL_SHAPE
                    ),
                    "realize_multi_tile_reduce": (
                        StrategyFamily.REDUCTION_TILING
                    ),
                }.get(original.rule)
                if reverse_family is not None:
                    bindings = [
                        binding
                        for binding in bindings
                        if not (
                            binding.target_ref == original.target_ref
                            and binding.family is reverse_family
                        )
                    ]
                else:
                    candidates = []
            if candidates:
                changed, _ = materialize_replay_plan(
                    replace(
                        artifact,
                        strategy=StrategySpec(bindings),
                    ),
                    registry,
                )
                specs.append(
                    PiAblationSpec(
                        (
                            f"substitute_step:{substitution_index}:"
                            f"{original.rule}:{replacement}:"
                            f"{original.target_ref}"
                        ),
                        PiAblationKind.PLAN_MECHANISM_SUBSTITUTION,
                        changed.plan,
                        dropped_step_index=substitution_index,
                        dropped_rule=original.rule,
                        dropped_target=original.target_ref,
                        replacement_rule=replacement,
                        artifact=changed,
                        strategy_before=artifact.strategy.to_json(),
                        strategy_after=changed.strategy.to_json(),
                    )
                )
    return tuple(specs)

def _interface_for_fingerprint(
    result: CompileResult,
    case: CompileCase,
    artifact: Gate1Artifact,
) -> dict[str, Any]:
    interface = dict(result.python_interface)
    interface["grid"] = list(case.grid)
    interface["grid_contract"] = list(artifact.interface.grid)
    return interface


def _case_with_artifact_schedule(
    case: CompileCase, artifact: Gate1Artifact
) -> CompileCase:
    """Compile every ablation with the same privileged C as its control."""

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
            "pi ablation CompileCase.schedule differs from the frozen "
            "artifact C values"
        )
    return replace(case, schedule=values)


def run_pi_ablations(
    artifact: Gate1Artifact,
    source_fingerprint: TTIRFingerprint,
    compile_case: CompileCase,
    registry: RuleRegistry,
    writer: ReplayWriter,
    *,
    compiler: TritonCompiler | None = None,
    comparator: TTIRComparator | None = None,
    limit: int | None = None,
    kinds: tuple[PiAblationKind, ...] | None = None,
    seen_generation_ids: set[str] | None = None,
) -> tuple[PiAblationRecord, ...]:
    """Execute π counterfactuals through writer, compiler, and hard compare."""

    compiler = compiler or TritonCompiler()
    comparator = comparator or TTIRComparator()
    compile_case = _case_with_artifact_schedule(compile_case, artifact)
    seen_ids = seen_generation_ids if seen_generation_ids is not None else set()
    records: list[PiAblationRecord] = []
    for spec in build_pi_ablations(
        artifact, registry, limit=limit, kinds=kinds
    ):
        necessity_probe = spec.kind in {
            PiAblationKind.J_ONLY,
            PiAblationKind.PLAN_STEP_DROPOUT,
        }
        intervention_artifact = spec.artifact or artifact
        if not necessity_probe:
            assert_replay_plan_matches(intervention_artifact, registry)

        try:
            generation = writer.generate(
                intervention_artifact,
                registry,
                masked=False,
                plan_override=spec.plan if necessity_probe else None,
                ablation_kind=spec.kind.value if necessity_probe else None,
            )
            if generation.generation_id in seen_ids:
                raise RuntimeError(
                    "pi ablation writer reused generation_id; every "
                    "counterfactual requires a fresh source-blind invocation"
                )
            seen_ids.add(generation.generation_id)
        except Exception as exc:
            records.append(
                PiAblationRecord(
                    spec,
                    PiAblationStatus.GENERATION_FAILED,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    attempts=1,
                )
            )
            continue

        try:
            compiled = compiler.compile(
                KernelSource.from_replay(
                    generation.code,
                    intervention_artifact.interface.function_name,
                    registry,
                ),
                compile_case,
            )
        except Exception as exc:
            records.append(
                PiAblationRecord(
                    spec,
                    PiAblationStatus.EVALUATION_FAILED,
                    generation=generation,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    attempts=1,
                )
            )
            continue

        if not compiled.ok or not compiled.ttir:
            status = (
                PiAblationStatus.DEGRADED
                if necessity_probe
                else PiAblationStatus.COMPILE_FAILED
            )
            records.append(
                PiAblationRecord(
                    spec,
                    status,
                    generation=generation,
                    compile=compiled,
                    diagnostic=(
                        "fresh ablated replay did not compile"
                        if necessity_probe
                        else compiled.diagnostic
                    ),
                    attempts=1,
                )
            )
            continue

        try:
            fingerprint = fingerprint_ttir(
                compiled.ttir,
                python_interface=_interface_for_fingerprint(
                    compiled, compile_case, intervention_artifact
                ),
                schedule=compile_case.schedule,
            )
            comparison = comparator.compare(source_fingerprint, fingerprint)
        except Exception as exc:
            records.append(
                PiAblationRecord(
                    spec,
                    PiAblationStatus.EVALUATION_FAILED,
                    generation=generation,
                    compile=compiled,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    attempts=1,
                )
            )
            continue

        if necessity_probe:
            status = (
                PiAblationStatus.UNEXPECTED_CLOSURE
                if comparison.hard_pass
                else PiAblationStatus.DEGRADED
            )
            diagnostic = (
                "ablated π unexpectedly retained score-free hard closure"
                if comparison.hard_pass
                else "ablated π produced a score-free hard mismatch"
            )
        else:
            status = (
                PiAblationStatus.SEMANTICS_PRESERVED
                if comparison.hard_pass
                else PiAblationStatus.OFF_TARGET_DRIFT
            )
            diagnostic = (
                "C.strategy intervention preserved all J semantic hard components"
                if comparison.hard_pass
                else "C.strategy intervention changed J semantic hard components"
            )
        records.append(
            PiAblationRecord(
                spec,
                status,
                generation=generation,
                compile=compiled,
                comparison=comparison,
                diagnostic=diagnostic,
                attempts=1,
            )
        )
    return tuple(records)



__all__ = [
    "PiAblationKind",
    "PiAblationRecord",
    "PiAblationSpec",
    "PiAblationStatus",
    "PiStepClassification",
    "build_pi_ablations",
    "classify_pi_steps",
    "run_pi_ablations",
]
