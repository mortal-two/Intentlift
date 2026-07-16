"""Bounded SCRC Gate1 orchestration.

The orchestrator keeps the trust split explicit:

* LLM/static writers propose replay source;
* :class:`TritonCompiler` compiles in an isolated process without launching;
* :mod:`.ttir` and :mod:`.compare` alone decide structural closure;
* diagnosis scopes the next Lift update but cannot override the verdict;
* paired J interventions reuse the normal model-generated replay as a
  controlled structural baseline and deterministically test whether semantics
  obey J without unrelated regeneration drift;
* masked-J remains a diagnostic/fallback for artifacts without an intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from .compare import TTIRComparator, TTIRComparison
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
    leakage_ratio,
    scan_static_leakage,
)
from .interventions import (
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
    # Retained for configuration/artifact compatibility.  The ratio is now
    # diagnostic only; masked-J fallback uses exact semantic hard components.
    leakage_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.max_relifts < 1:
            raise ValueError("max_relifts must be positive")
        if self.stable_closures < 1:
            raise ValueError("stable_closures must be positive")
        if self.max_semantic_interventions < 0:
            raise ValueError("max_semantic_interventions must be non-negative")
        if self.leakage_threshold < 0:
            raise ValueError("leakage_threshold must be non-negative")


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

    @property
    def compiled(self) -> bool:
        return bool(self.compile and self.compile.ok)

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
            int(self.comparison.hard_pass),
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
                "base_generation_id": self.generation.base_generation_id,
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


@dataclass(frozen=True)
class ReliftRequest:
    """Stable, auditable reason for one outer-loop semantic relift."""

    trigger: ReliftTrigger
    evaluation_round: int
    candidate_count: int
    compiled_candidate_count: int
    best_generation_round: int | None
    hard_difference_categories: tuple[str, ...]

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
        }


def _relift_request_for_round(
    evaluation_round: int,
    records: tuple[CandidateRecord, ...],
    best: CandidateRecord | None,
) -> ReliftRequest:
    if not records:
        raise ValueError("semantic relift requires at least one replay candidate")
    hard_categories: tuple[str, ...] = ()
    if best is not None and best.comparison is not None:
        hard_categories = tuple(
            component.category
            for component in best.comparison.components
            if component.hard and not component.passed
        )
    if any(record.comparison is not None for record in records):
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
        history: tuple[CandidateRecord, ...],
        request: ReliftRequest,
    ) -> Gate1Artifact: ...


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
    leakage_ratio: float | None = None
    diagnostics: tuple[str, ...] = ()
    masked_generation: ReplayGeneration | None = None
    interventions: tuple[InterventionRecord, ...] = ()
    anti_leakage_basis: str = "masked_j_fallback"

    @property
    def replay_closed(self) -> bool:
        return self.status is Gate1Status.REPLAY_CLOSED

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
                "base_generation_id": self.masked_generation.base_generation_id,
                "ablation_kind": self.masked_generation.ablation_kind,
            },
            "masked_comparison": None
            if self.masked_comparison is None
            else self.masked_comparison.to_dict(),
            "leakage_ratio": self.leakage_ratio,
            "interventions": [item.to_json() for item in self.interventions],
            "anti_leakage_basis": self.anti_leakage_basis,
            "diagnostics": list(self.diagnostics),
        }


def _case_with_artifact_schedule(
    case: CompileCase, artifact: Gate1Artifact
) -> CompileCase:
    values: dict[str, int | float | bool] = {}
    for name, value in artifact.schedule.values.items():
        if not isinstance(value, (bool, int, float)):
            raise ValueError(
                f"Triton schedule value {name!r} must be bool/int/float, got {value!r}"
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
    if artifact.j.boundary.aliases or any(
        parameter.alias_of is not None for parameter in interface.parameters
    ):
        raise ValueError(
            "Gate1 Triton MVP cannot statically validate alias contracts; "
            "rejecting instead of silently treating them as closed"
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
    if record is None or record.comparison is None:
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


_MASKED_SEMANTIC_CATEGORIES = (
    "op_multiset",
    "dataflow",
    "reduction",
    "predicate_effect",
)
MASKED_J_FALLBACK_POLICY = "exact_semantic_hard_degradation_v1"


def _masked_semantic_degradation(
    comparison: TTIRComparison,
) -> tuple[bool, tuple[str, ...]]:
    """Return exact A/B categories lost by a compiled masked-J replay."""

    failed = tuple(
        category
        for category in _MASKED_SEMANTIC_CATEGORIES
        if not comparison.component(category).passed
    )
    return bool(failed), failed


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
) -> tuple[InterventionRecord, ...]:
    """Run source-blind paired interventions against a closed replay."""

    if (
        limit == 0
        or baseline_record.compile is None
        or not baseline_record.compile.ok
        or not baseline_record.compile.ttir
    ):
        return ()
    specs = build_semantic_interventions(artifact, limit=limit)
    if not specs:
        return ()
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
                base_candidate=baseline_record.generation,
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
                KernelSource.from_code(generation.code, function_name),
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
    ) -> Gate1ExecutionResult:
        history: list[CandidateRecord] = []
        diagnostics: list[str] = []
        best: CandidateRecord | None = None
        current = artifact
        source_compile: CompileResult | None = None
        source_fp: TTIRFingerprint | None = None
        leakage_report: LeakageReport | None = None

        for relift_round in range(self.config.max_relifts):
            try:
                compile_case = _case_with_artifact_schedule(case, current)
                _validate_artifact_compile_contract(source, current, compile_case)
                registry.validate_strict_plan(
                    current.plan,
                    frontend=current.interface.frontend,
                    artifact=current,
                )
                source_text, _ = source.read_code()
                leakage_report = scan_static_leakage(
                    current.plan,
                    source=source_text,
                    j=current.j,
                    policy=self.leakage_policy,
                    registry=registry,
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
                )

            stable = 0
            closed_record: CandidateRecord | None = None
            seen_generation_ids: set[str] = set()
            generation_round = 0
            round_best: CandidateRecord | None = None

            # One fresh source-blind replay proposal is evaluated for this
            # artifact. Additional proposals exist only when an independent
            # stability confirmation is explicitly requested. A failure is
            # never repaired in-place; it returns to the outer Lift loop.
            while stable < self.config.stable_closures:
                generation: ReplayGeneration | None = None
                compile_result: CompileResult | None = None
                comparison: TTIRComparison | None = None
                diagnosis: Diagnosis | None = None
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
                    replay_source = KernelSource.from_code(
                        generation.code, function_name
                    )
                    compile_result = self.compiler.compile(
                        replay_source, compile_case
                    )
                    if not compile_result.ok or not compile_result.ttir:
                        diagnosis = diagnose_compile_failure(
                            compile_result.diagnostic
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
                                Responsibility.REPLAY_PLAN,
                                (str(exc),),
                                (),
                                ("j", "pi", "location_map"),
                                "重放接口不符合冻结 interface；禁止原地修签名，"
                                "回流 Lift 后从头重放。",
                            )
                        else:
                            comparison = self.comparator.compare(
                                source_fp, replay_fp
                            )
                            diagnosis = diagnose_comparison(comparison)
                except Exception as exc:
                    if diagnosis is None:
                        if generation is None or "generation_id" in str(exc):
                            diagnosis = Diagnosis(
                                FailureCause.REPLAY_POLICY,
                                Responsibility.REPLAY_PLAN,
                                (f"{type(exc).__name__}: {exc}",),
                                (),
                                ("j", "pi", "location_map"),
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

                if comparison is not None and comparison.hard_pass:
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
                            return Gate1ExecutionResult(
                                Gate1Status.LEAKAGE_REJECTED,
                                current,
                                source_compile,
                                source_fp,
                                tuple(history),
                                best,
                                leakage_report,
                                diagnostics=tuple(diagnostics),
                                interventions=intervention_records,
                                anti_leakage_basis="paired_intervention",
                            )
                        diagnostics.append(
                            "paired semantic intervention passed with an exact "
                            "registered TTIR delta"
                        )
                if not self.config.run_masked_ablation:
                    return Gate1ExecutionResult(
                        Gate1Status.REPLAY_CLOSED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        leakage_ratio=0.0,
                        diagnostics=tuple(diagnostics),
                        interventions=intervention_records,
                        anti_leakage_basis=(
                            "paired_intervention"
                            if paired_intervention_passed
                            else "disabled"
                        ),
                    )

                masked_compile: CompileResult | None = None
                masked_comparison: TTIRComparison | None = None
                masked_generation: ReplayGeneration | None = None
                masked_score = 0.0
                masked_attempted = False
                try:
                    masked_generation = writer.generate(
                        current, registry, masked=True
                    )
                    if masked_generation.generation_id in seen_generation_ids:
                        raise RuntimeError(
                            "masked ablation reused a normal generation_id"
                        )
                    masked_code = restore_masked_replay_module(
                        masked_generation.code, current
                    )
                    masked_attempted = True
                    function_name = getattr(
                        current.interface,
                        "function_name",
                        current.interface.kernel_id,
                    )
                    masked_compile = self.compiler.compile(
                        KernelSource.from_code(
                            masked_code, function_name
                        ),
                        compile_case,
                    )
                    if masked_compile.ok and masked_compile.ttir:
                        masked_fp = fingerprint_ttir(
                            masked_compile.ttir,
                            python_interface=_interface_for_fingerprint(
                                masked_compile, compile_case, current
                            ),
                            schedule=compile_case.schedule,
                        )
                        _validate_compiled_interface(
                            masked_compile, masked_fp, current
                        )
                        masked_comparison = self.comparator.compare(
                            source_fp, masked_fp
                        )
                        masked_score = masked_comparison.score
                except Exception as exc:  # ablation failure is the expected safe outcome
                    diagnostics.append(
                        f"masked ablation could not replay: {type(exc).__name__}: {exc}"
                    )
                if not masked_attempted:
                    diagnostics.append(
                        "masked-J ablation was unavailable"
                    )
                    if paired_intervention_passed:
                        diagnostics.append(
                            "paired intervention remains the deterministic "
                            "anti-leakage basis"
                        )
                        return Gate1ExecutionResult(
                            Gate1Status.REPLAY_CLOSED,
                            current,
                            source_compile,
                            source_fp,
                            tuple(history),
                            best,
                            leakage_report,
                            masked_compile,
                            masked_comparison,
                            None,
                            tuple(diagnostics),
                            masked_generation=masked_generation,
                            interventions=intervention_records,
                            anti_leakage_basis="paired_intervention",
                        )
                    diagnostics.append(
                        "no paired intervention passed; Gate1 fails closed"
                    )
                    return Gate1ExecutionResult(
                        Gate1Status.LEAKAGE_REJECTED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        masked_compile,
                        masked_comparison,
                        None,
                        tuple(diagnostics),
                        masked_generation=masked_generation,
                        interventions=intervention_records,
                        anti_leakage_basis="masked_j_fallback",
                    )
                normal_score = closed_record.comparison.score  # type: ignore[union-attr]
                ratio = leakage_ratio(normal_score, masked_score)
                if masked_comparison is None:
                    masked_semantically_degraded = True
                    masked_failed_categories: tuple[str, ...] = ()
                    diagnostics.append(
                        "masked-J could not produce a compiled, interface-valid "
                        "semantic fingerprint"
                    )
                else:
                    (
                        masked_semantically_degraded,
                        masked_failed_categories,
                    ) = _masked_semantic_degradation(masked_comparison)

                if not masked_semantically_degraded:
                    diagnostics.append(
                        "masked-J preserved every semantic hard component; "
                        f"diagnostic score ratio={ratio:.6f}"
                    )
                    if paired_intervention_passed:
                        diagnostics.append(
                            "blank-J guessability is diagnostic only because "
                            "the paired intervention proved causal J control"
                        )
                        return Gate1ExecutionResult(
                            Gate1Status.REPLAY_CLOSED,
                            current,
                            source_compile,
                            source_fp,
                            tuple(history),
                            best,
                            leakage_report,
                            masked_compile,
                            masked_comparison,
                            ratio,
                            tuple(diagnostics),
                            masked_generation=masked_generation,
                            interventions=intervention_records,
                            anti_leakage_basis="paired_intervention",
                        )
                    return Gate1ExecutionResult(
                        Gate1Status.LEAKAGE_REJECTED,
                        current,
                        source_compile,
                        source_fp,
                        tuple(history),
                        best,
                        leakage_report,
                        masked_compile,
                        masked_comparison,
                        ratio,
                        tuple(diagnostics),
                        masked_generation=masked_generation,
                        interventions=intervention_records,
                        anti_leakage_basis="masked_j_fallback",
                    )
                if masked_failed_categories:
                    diagnostics.append(
                        "masked-J deterministically lost semantic hard "
                        "component(s): " + ", ".join(masked_failed_categories)
                    )
                diagnostics.append(
                    f"masked-J score ratio={ratio:.6f} is diagnostic only"
                )
                return Gate1ExecutionResult(
                    Gate1Status.REPLAY_CLOSED,
                    current,
                    source_compile,
                    source_fp,
                    tuple(history),
                    best,
                    leakage_report,
                    masked_compile,
                    masked_comparison,
                    ratio,
                    tuple(diagnostics),
                    masked_generation=masked_generation,
                    interventions=intervention_records,
                    anti_leakage_basis=(
                        "paired_intervention"
                        if paired_intervention_passed
                        else "masked_j_fallback"
                    ),
                )

            if relifter is None or relift_round + 1 >= self.config.max_relifts:
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
            diagnostics.append(
                "normal replay failed; restarting from source-aware Lift with "
                f"trigger={request.trigger.value}"
            )
            try:
                current = relifter.relift(
                    current,
                    source_fingerprint=source_fp,
                    history=tuple(history),
                    request=request,
                )
            except Exception as exc:
                diagnostics.append(
                    "semantic relift failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

        return Gate1ExecutionResult(
            Gate1Status.NO_CLOSURE,
            current,
            source_compile,
            source_fp,
            tuple(history),
            best,
            leakage_report,
            diagnostics=tuple(diagnostics),
        )


__all__ = [
    "CandidateRecord",
    "Gate1Config",
    "Gate1ExecutionResult",
    "Gate1Runner",
    "ReliftRequest",
    "ReliftTrigger",
    "Relifter",
]
