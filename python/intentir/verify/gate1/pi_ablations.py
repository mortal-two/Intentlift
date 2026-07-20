"""Canonical pi negative tests and C.strategy causal interventions.

Direct pi deletion/addition/substitution is rejected by deterministic compose
equality before any writer call.  A legal implementation-mechanism change is
expressed only through C.strategy, followed by a fresh deterministic compose.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .compare import TTIRComparator, TTIRComparison
from .composer import (
    ComposeMismatchError,
    assert_replay_plan_matches,
    materialize_replay_plan,
)
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
from .rules import RuleRegistry, RuleSelectionMode
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
        return self.status in {
            PiAblationStatus.DEGRADED,
            PiAblationStatus.CANONICALITY_REJECTED,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_json(),
            "status": self.status.value,
            "supports_pi_necessity": self.supports_pi_necessity,
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


_COMPUTE_RULES = {
    "realize_elementwise",
    "realize_const",
    "realize_iota",
    "realize_reduce",
    "realize_flattened_reduce",
    "realize_broadcast",
    "realize_k_tiled_dot",
    "realize_outer_product",
    "realize_indexed_access",
    "realize_transpose",
    "realize_concat",
    "realize_cast",
    "realize_select",
    "realize_loop_carry",
}

_MEMORY_RULES = {
    "materialize_index_tile",
    "materialize_pointer_tile",
    "materialize_mask_tile",
    "realize_masked_access",
    "realize_unmasked_access",
    "realize_output_mapping",
}

_MECHANISM_SUBSTITUTIONS = {
    "materialize_index_tile": "materialize_pointer_tile",
    "materialize_pointer_tile": "materialize_index_tile",
    "bind_program_domain": "bind_wide_program_domain",
    "bind_wide_program_domain": "bind_program_domain",
    "realize_reduce": "realize_flattened_reduce",
    "realize_flattened_reduce": "realize_reduce",
}


def build_pi_ablations(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    limit: int = 2,
    kinds: tuple[PiAblationKind, ...] | None = None,
) -> tuple[PiAblationSpec, ...]:
    """Build deterministic J-only and one-step-dropout diagnostics."""

    if limit < 0:
        raise ValueError("pi ablation limit must be non-negative")
    if limit == 0:
        return ()
    requested = tuple(
        kinds
        if kinds is not None
        else (
            PiAblationKind.PLAN_STEP_DROPOUT,
            PiAblationKind.PLAN_MECHANISM_SUBSTITUTION,
        )
    )
    if len(set(requested)) != len(requested):
        raise ValueError("pi ablation kinds must be unique")
    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )
    assert_replay_plan_matches(artifact, registry)

    specs: list[PiAblationSpec] = []
    if PiAblationKind.J_ONLY in requested:
        specs.append(
            PiAblationSpec(
                "j_only",
                PiAblationKind.J_ONLY,
                ReplayPlan([]),
            )
        )
    if len(specs) >= limit or not artifact.plan.steps:
        return tuple(specs[:limit])

    def priority(index: int) -> tuple[int, int]:
        step = artifact.plan.steps[index]
        card = registry.get(step.rule, frontend=artifact.interface.frontend)
        if card.selection_mode is RuleSelectionMode.OVERRIDE:
            return (0, index)
        if step.rule in _COMPUTE_RULES:
            return (1, index)
        if step.rule in _MEMORY_RULES:
            return (2, index)
        return (3, index)

    if PiAblationKind.PLAN_STEP_DROPOUT in requested:
        dropped_index = min(range(len(artifact.plan.steps)), key=priority)
        dropped = artifact.plan.steps[dropped_index]
        remaining = ReplayPlan(
            [
                step
                for index, step in enumerate(artifact.plan.steps)
                if index != dropped_index
            ]
        )
        specs.append(
            PiAblationSpec(
                (
                    f"drop_step:{dropped_index}:{dropped.rule}:"
                    f"{dropped.target_ref}"
                ),
                PiAblationKind.PLAN_STEP_DROPOUT,
                remaining,
                dropped_step_index=dropped_index,
                dropped_rule=dropped.rule,
                dropped_target=dropped.target_ref,
            )
        )
    if (
        len(specs) < limit
        and PiAblationKind.PLAN_MECHANISM_SUBSTITUTION in requested
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
                    "bind_wide_program_domain": StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                    "materialize_pointer_tile": StrategyFamily.ADDRESS_ASSOCIATION,
                    "realize_flattened_reduce": StrategyFamily.REDUCTION_CALL_SHAPE,
                }.get(original.rule)
                if reverse_family is None:
                    return tuple(specs[:limit])
                bindings = [
                    binding
                    for binding in bindings
                    if not (
                        binding.target_ref == original.target_ref
                        and binding.family is reverse_family
                    )
                ]
            changed, _ = materialize_replay_plan(
                replace(artifact, strategy=StrategySpec(bindings)), registry
            )
            specs.append(
                PiAblationSpec(
                    (
                        f"substitute_step:{substitution_index}:{original.rule}:"
                        f"{replacement}:{original.target_ref}"
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
    return tuple(specs[:limit])


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
    limit: int = 2,
    kinds: tuple[PiAblationKind, ...] | None = None,
) -> tuple[PiAblationRecord, ...]:
    """Generate, compile and compare diagnostic incomplete-pi replays."""

    compiler = compiler or TritonCompiler()
    comparator = comparator or TTIRComparator()
    compile_case = _case_with_artifact_schedule(compile_case, artifact)
    records: list[PiAblationRecord] = []
    for spec in build_pi_ablations(
        artifact, registry, limit=limit, kinds=kinds
    ):
        if spec.kind in {
            PiAblationKind.J_ONLY,
            PiAblationKind.PLAN_STEP_DROPOUT,
        }:
            try:
                assert_replay_plan_matches(
                    artifact, registry, plan=spec.plan
                )
            except ComposeMismatchError as exc:
                records.append(
                    PiAblationRecord(
                        spec,
                        PiAblationStatus.CANONICALITY_REJECTED,
                        diagnostic=str(exc),
                        attempts=0,
                    )
                )
            else:
                records.append(
                    PiAblationRecord(
                        spec,
                        PiAblationStatus.UNEXPECTED_CLOSURE,
                        diagnostic="tampered pi unexpectedly passed exact compose equality",
                        attempts=0,
                    )
                )
            continue

        intervention_artifact = spec.artifact or artifact
        # A mechanism substitution is not a direct pi mutation.  It changes
        # C.strategy, deterministically composes a new canonical pi, and then
        # invokes the writer through the ordinary replay path.  Keeping an old
        # plan-ablation flag here would incorrectly couple the legal C path to
        # the legacy plan_override diagnostic interface.
        assert_replay_plan_matches(intervention_artifact, registry)
        try:
            generation = writer.generate(
                intervention_artifact,
                registry,
                masked=False,
                plan_override=None,
                ablation_kind=None,
            )
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
                    PiAblationStatus.COMPILE_FAILED,
                    generation=generation,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    attempts=1,
                )
            )
            continue
        if not compiled.ok or not compiled.ttir:
            records.append(
                PiAblationRecord(
                spec,
                PiAblationStatus.COMPILE_FAILED,
                generation=generation,
                compile=compiled,
                diagnostic=compiled.diagnostic,
                attempts=1,
                )
            )
            continue

        try:
            fingerprint = fingerprint_ttir(
                compiled.ttir,
                python_interface=_interface_for_fingerprint(
                    compiled, compile_case, artifact
                ),
                schedule=compile_case.schedule,
            )
            comparison = comparator.compare(source_fingerprint, fingerprint)
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
    return tuple(records)


__all__ = [
    "PiAblationKind",
    "PiAblationRecord",
    "PiAblationSpec",
    "PiAblationStatus",
    "build_pi_ablations",
    "run_pi_ablations",
]
