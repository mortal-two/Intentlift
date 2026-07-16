"""Plan-aligned diagnostic ablations for the strict replay plan ``pi``.

These experiments intentionally run an incomplete replay packet after the
original artifact has passed strict-plan validation.  They do not change the
Gate1 verdict yet; they provide deterministic evidence for the original SCRC
J-only and step-dropout experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .compare import TTIRComparator, TTIRComparison
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .model import Gate1Artifact, ReplayPlan
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

    def to_json(self) -> dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "kind": self.kind.value,
            "plan": self.plan.to_json(),
            "dropped_step_index": self.dropped_step_index,
            "dropped_rule": self.dropped_rule,
            "dropped_target": self.dropped_target,
            "replacement_rule": self.replacement_rule,
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
    "realize_reduce",
    "realize_flattened_reduce",
    "realize_broadcast",
    "realize_k_tiled_dot",
    "realize_outer_product",
    "realize_indexed_access",
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
    requested = tuple(kinds or tuple(PiAblationKind))
    if len(set(requested)) != len(requested):
        raise ValueError("pi ablation kinds must be unique")
    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )

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
            substituted = ReplayPlan(
                [
                    type(step)(
                        replacement,
                        node_ref=step.node_ref,
                        refs=list(step.refs),
                    )
                    if index == substitution_index
                    else step
                    for index, step in enumerate(artifact.plan.steps)
                ]
            )
            registry.validate_strict_plan(
                substituted,
                frontend=artifact.interface.frontend,
                artifact=artifact,
            )
            specs.append(
                PiAblationSpec(
                    (
                        f"substitute_step:{substitution_index}:{original.rule}:"
                        f"{replacement}:{original.target_ref}"
                    ),
                    PiAblationKind.PLAN_MECHANISM_SUBSTITUTION,
                    substituted,
                    dropped_step_index=substitution_index,
                    dropped_rule=original.rule,
                    dropped_target=original.target_ref,
                    replacement_rule=replacement,
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

    values: dict[str, int | float | bool] = {}
    for name, value in artifact.schedule.values.items():
        if not isinstance(value, (bool, int, float)):
            raise ValueError(
                f"Triton schedule value {name!r} must be bool/int/float, "
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
    control_candidate: ReplayGeneration | None = None,
) -> tuple[PiAblationRecord, ...]:
    """Generate, compile and compare diagnostic incomplete-pi replays."""

    compiler = compiler or TritonCompiler()
    comparator = comparator or TTIRComparator()
    compile_case = _case_with_artifact_schedule(compile_case, artifact)
    records: list[PiAblationRecord] = []
    for spec in build_pi_ablations(
        artifact, registry, limit=limit, kinds=kinds
    ):
        base_candidate = (
            control_candidate
            if spec.kind is PiAblationKind.PLAN_MECHANISM_SUBSTITUTION
            else None
        )
        try:
            generation = writer.generate(
                artifact,
                registry,
                base_candidate=base_candidate,
                masked=False,
                plan_override=spec.plan,
                ablation_kind=spec.kind.value,
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
                KernelSource.from_code(
                    generation.code, artifact.interface.function_name
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
                PiAblationStatus.UNEXPECTED_CLOSURE
                if comparison.hard_pass
                else PiAblationStatus.DEGRADED
            )
            diagnostic = (
                "pi ablation replay still hard-closes"
                if comparison.hard_pass
                else "pi ablation replay loses at least one hard component"
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
