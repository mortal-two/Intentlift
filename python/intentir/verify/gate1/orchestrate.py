"""Bounded SCRC Gate1 orchestration.

The orchestrator keeps the trust split explicit:

* LLM/static writers propose replay source;
* :class:`TritonCompiler` compiles in an isolated process without launching;
* :mod:`.ttir` and :mod:`.compare` alone decide structural closure;
* diagnosis scopes repairs but cannot override the verdict;
* the final masked-J ablation measures whether pi/interface channels leaked J.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from .compare import TTIRComparator, TTIRComparison
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .diagnose import (
    Diagnosis,
    RepairFeedback,
    Responsibility,
    diagnose_comparison,
    diagnose_compile_failure,
)
from .evaluation import FailureCause
from .leakage import (
    LeakagePolicy,
    LeakageReport,
    leakage_ratio,
    masked_ablation_passes,
    scan_static_leakage,
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
    max_repairs: int = 2
    max_relifts: int = 5
    stable_closures: int = 1
    run_masked_ablation: bool = True
    leakage_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        if self.max_relifts < 1:
            raise ValueError("max_relifts must be positive")
        if self.stable_closures < 1:
            raise ValueError("stable_closures must be positive")
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


class Relifter(Protocol):
    def relift(
        self,
        artifact: Gate1Artifact,
        *,
        source_fingerprint: TTIRFingerprint,
        history: tuple[CandidateRecord, ...],
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
            },
            "masked_comparison": None
            if self.masked_comparison is None
            else self.masked_comparison.to_dict(),
            "leakage_ratio": self.leakage_ratio,
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
                registry.resolve(current.plan, frontend=current.interface.frontend)
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

            feedback: RepairFeedback | None = None
            repair_base: ReplayGeneration | None = None
            stable = 0
            closed_record: CandidateRecord | None = None
            seen_generation_ids: set[str] = set()
            repairs_generated = 0
            generation_round = 0
            round_best: CandidateRecord | None = None

            # ``max_repairs`` counts generations that receive repair feedback;
            # the initial proposal and independent stability proposals do not
            # consume that budget.
            while True:
                if feedback is not None:
                    if repairs_generated >= self.config.max_repairs:
                        break
                    repairs_generated += 1
                generation: ReplayGeneration | None = None
                compile_result: CompileResult | None = None
                comparison: TTIRComparison | None = None
                diagnosis: Diagnosis | None = None
                try:
                    generation = writer.generate(
                        current,
                        registry,
                        feedback=feedback,
                        base_candidate=repair_base,
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
                                Responsibility.INTERFACE_SCHEDULE,
                                (str(exc),),
                                (),
                                ("replay signature",),
                                "重放编译接口不符合 artifact 声明；只修签名。",
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
                                Responsibility.REPLAY_CODE,
                                (f"{type(exc).__name__}: {exc}",),
                                (),
                                (),
                                "重写服务未产生可审计候选；记录失败并停止该版 artifact。",
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
                    if stable >= self.config.stable_closures:
                        break
                    # A stability rerun is an independent source-blind proposal,
                    # not a repair of the just-closed code.
                    feedback = None
                    repair_base = None
                    generation_round += 1
                    continue

                stable = 0
                # If generation itself failed there is no candidate source on
                # which to base a bounded repair. Provider/model fallback is
                # already handled inside the writer, so stop this artifact.
                if generation is None or diagnosis is None:
                    break
                if diagnosis.cause is FailureCause.LOWERING:
                    diagnostics.append(diagnosis.explanation)
                    break
                feedback_record = round_best or record
                if feedback_record.generation is None:
                    break
                repair_base = feedback_record.generation
                if (
                    feedback_record.comparison is not None
                    and feedback_record.comparison.hard_pass
                    and record is not feedback_record
                ):
                    feedback = RepairFeedback(
                        FailureCause.REPLAY_POLICY,
                        Responsibility.REPLAY_CODE,
                        tuple(feedback_record.comparison.hard_passes),
                        ("replay code",),
                        "稳定性重写发生回退；以给定的已闭合候选为基准，"
                        "保持 J/π 与接口不变并重新发出完整模块。",
                    )
                else:
                    feedback_diagnosis = feedback_record.diagnosis or diagnosis
                    feedback = feedback_diagnosis.to_rewriter_feedback()
                generation_round += 1

            if closed_record is not None and stable >= self.config.stable_closures:
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
                        "masked-J ablation was unavailable; Gate1 fails closed"
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
                    )
                normal_score = closed_record.comparison.score  # type: ignore[union-attr]
                ratio = leakage_ratio(normal_score, masked_score)
                if not masked_ablation_passes(
                    normal_score,
                    masked_score,
                    threshold=self.config.leakage_threshold,
                ):
                    diagnostics.append(
                        f"masked-J leakage ratio {ratio:.6f} exceeds "
                        f"threshold {self.config.leakage_threshold:.6f}"
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
                )

            if relifter is None or relift_round + 1 >= self.config.max_relifts:
                break
            assert source_fp is not None
            current = relifter.relift(
                current,
                source_fingerprint=source_fp,
                history=tuple(history),
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
        )


__all__ = [
    "CandidateRecord",
    "Gate1Config",
    "Gate1ExecutionResult",
    "Gate1Runner",
    "Relifter",
]
