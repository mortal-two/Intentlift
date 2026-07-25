"""Real SCRC lifecycle helpers for declarative Triton workload adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.candidate_fanout import CandidateFanoutSearchReport
from intentir.verify.gate1.model import (
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)
from intentir.extraction.lifters.miniswe import MiniSWELiftAdapter
from intentir.extraction.lifters.scrc import ResponsesSCRCLiftAdapter
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.adapters import (
    BoundGate1Adapter,
    TritonGate2ExecutionAdapter,
)
from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.gate1.orchestrate import (
    Gate1Config,
    Gate1ExecutionResult,
    Gate1Runner,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
    canonical_artifact_digest,
)
from intentir.verify.gate1.replay import (
    MiniSWEReplayWriter,
    RecordedReplayWriter,
    ReplayAttempt,
    ReplayGeneration,
    ResponsesReplayWriter,
)
from intentir.verify.gate2.model import Gate2Status
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from intentir.verify.gate2.scalar import ScalarLoweringSpec
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import (
    MainStatus,
    SCRCGateRunner,
    SCRCGate1SearchReport,
    SCRCProgressCallback,
    SCRCRunDependencies,
    SCRCRunReport,
)
from intentir.verify.typecheck import TypeCheckReport, type_check
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    NoneBinding,
    PreparedTritonWorkload,
)


RecordedLiftAdapter = ResponsesSCRCLiftAdapter | MiniSWELiftAdapter
GATE1_DIAGNOSTIC_RECHECK_PROTOCOL = (
    "intentir.scrc.gate1-diagnostic-recheck.v1"
)


def _new_replay_writer(
    replay_backend: str,
    *,
    provider: ProviderConfig,
    effort: str | None,
) -> RecordedReplayWriter:
    """Create one writer for exactly one formal Gate1 round."""

    if replay_backend == "responses":
        return ResponsesReplayWriter(provider=provider, effort=effort)
    if replay_backend == "miniswe":
        return MiniSWEReplayWriter(provider=provider, effort=effort)
    raise ValueError(f"unknown Replay backend: {replay_backend!r}")


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    normalized = aliases.get(dtype.removeprefix("torch."), dtype)
    if normalized == "bf16":
        # NumPy has no portable bfloat16; Gate2's torch leg performs the final
        # dtype conversion from finite float32 Harness inputs.
        normalized = "float32"
    try:
        return np.dtype(normalized)
    except TypeError as exc:
        raise ValueError(f"unsupported Harness input dtype {dtype!r}") from exc


def _input_value(
    rng: np.random.Generator,
    buffer: BufferBinding,
) -> np.ndarray:
    dtype = _numpy_dtype(buffer.dtype)
    if buffer.initializer is BufferInitializer.ZEROS:
        return np.zeros(buffer.shape, dtype=dtype)
    if buffer.initializer is not BufferInitializer.RANDOM:
        raise ValueError("input/inout buffer lacks a supported initializer")
    if dtype.kind == "b":
        return rng.integers(0, 2, size=buffer.shape).astype(dtype)
    if dtype.kind in {"i", "u"}:
        return rng.integers(-3, 4, size=buffer.shape).astype(dtype)
    return rng.normal(0.0, 0.25, size=buffer.shape).astype(dtype)


def _bind_shape(
    declared: list[int | str],
    actual: tuple[int, ...],
    dimensions: dict[str, int],
    *,
    tensor: str,
) -> None:
    if len(declared) != len(actual):
        raise ValueError(f"J rank for {tensor!r} differs from the Harness buffer")
    for expected, extent in zip(declared, actual, strict=True):
        if isinstance(expected, int):
            if expected != extent:
                raise ValueError(
                    f"J static extent for {tensor!r} differs from Harness"
                )
            continue
        previous = dimensions.setdefault(expected, extent)
        if previous != extent:
            raise ValueError(f"J symbolic extent {expected!r} is inconsistent")


def _bind_logical_scalar_input(
    *,
    parameter_name: str,
    logical_ref: str | None,
    parameter_dtype: str | None,
    value: bool | int | float | str,
    j_inputs: Mapping[str, Any],
    j_outputs: Mapping[str, Any],
    values: dict[str, Any],
    input_bindings: dict[str, str],
    dimensions: dict[str, int],
) -> None:
    if logical_ref is None:
        return
    if logical_ref in j_inputs:
        tensor = j_inputs[logical_ref]
        if tensor.shape:
            raise ValueError("scalar parameter must bind a rank-0 J input")
        if parameter_dtype is not None and tensor.dtype != parameter_dtype:
            raise ValueError("scalar parameter dtype differs from its J input")
        key = f"logical_{parameter_name}"
        values[key] = np.asarray(value, dtype=_numpy_dtype(tensor.dtype))
        if input_bindings.setdefault(logical_ref, key) != key:
            raise ValueError("multiple Harness scalars bind one J input")
        return
    if logical_ref in j_outputs:
        raise ValueError("value scalar cannot bind a J output tensor")
    if not isinstance(value, (bool, int)):
        raise ValueError("dimension-bound scalar must be integral")
    previous = dimensions.setdefault(logical_ref, int(value))
    if previous != int(value):
        raise ValueError("scalar dimension is inconsistent")


def build_workload_gate2_suite(
    prepared: PreparedTritonWorkload,
    artifact: Gate1Artifact,
) -> Gate2Suite:
    """Generate the hidden finite case only after Gate1 accepts the Replay."""

    adapter = prepared.adapter
    mapping = dict(prepared.source_to_opaque)
    buffers = {mapping[item.formal]: item for item in adapter.buffers}
    scalars = {mapping[item.formal]: item for item in adapter.scalars}
    none_arguments = {
        mapping[item.formal]: item for item in adapter.none_arguments
    }
    semantics = {
        mapping[name]: value for name, value in adapter.semantic_constexprs
    }
    j_inputs = {item.name: item for item in artifact.j.intent.inputs}
    j_outputs = {item.name: item for item in artifact.j.intent.outputs}
    rng = np.random.default_rng(adapter.seed)
    values: dict[str, Any] = {}
    dimensions: dict[str, int] = {}
    input_bindings: dict[str, str] = {}
    output_bindings: dict[str, str] = {}
    arguments: list[TritonArgument] = []

    for parameter in artifact.j.signature.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            if parameter.name not in semantics:
                raise ValueError(
                    f"J constexpr {parameter.name!r} lacks a Harness binding"
                )
            value = semantics[parameter.name]
            _bind_logical_scalar_input(
                parameter_name=parameter.name,
                logical_ref=parameter.logical_ref,
                parameter_dtype=parameter.dtype,
                value=value,
                j_inputs=j_inputs,
                j_outputs=j_outputs,
                values=values,
                input_bindings=input_bindings,
                dimensions=dimensions,
            )
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value=value,
                )
            )
            continue
        if parameter.kind is ParameterKind.SCALAR:
            binding = scalars.get(parameter.name)
            if binding is None:
                raise ValueError(
                    f"J scalar {parameter.name!r} lacks a Harness binding"
                )
            value_key = f"scalar_{parameter.name}"
            values[value_key] = binding.value
            _bind_logical_scalar_input(
                parameter_name=parameter.name,
                logical_ref=parameter.logical_ref,
                parameter_dtype=parameter.dtype,
                value=binding.value,
                j_inputs=j_inputs,
                j_outputs=j_outputs,
                values=values,
                input_bindings=input_bindings,
                dimensions=dimensions,
            )
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=value_key,
                )
            )
            continue
        if parameter.kind is ParameterKind.OPAQUE:
            binding = none_arguments.get(parameter.name)
            if binding is None:
                raise ValueError(
                    f"J opaque {parameter.name!r} lacks a Harness None binding"
                )
            if parameter.dtype != binding.dtype:
                raise ValueError(
                    "J opaque dtype differs from the Harness None binding"
                )
            arguments.append(
                TritonArgument(parameter.name, ArgumentRole.NONE)
            )
            continue
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            raise ValueError(
                f"unsupported real-workload J parameter kind {parameter.kind.value}"
            )
        buffer = buffers.get(parameter.name)
        if buffer is None:
            raise ValueError(
                f"J pointer {parameter.name!r} lacks a Harness buffer binding"
            )
        logical_ref = parameter.logical_ref
        if logical_ref is None:
            raise ValueError("real-workload pointer must anchor a J tensor")
        tensor = j_inputs.get(logical_ref) or j_outputs.get(logical_ref)
        if tensor is None:
            raise ValueError("J pointer logical_ref is not an input/output tensor")
        if tensor.dtype != buffer.dtype:
            raise ValueError("J tensor dtype differs from the Harness buffer")
        _bind_shape(
            tensor.shape,
            buffer.shape,
            dimensions,
            tensor=logical_ref,
        )
        if buffer.role is BufferRole.INPUT:
            if logical_ref not in j_inputs or parameter.mode is ParameterMode.OUTPUT:
                raise ValueError("Harness input buffer is not a J input")
            key = f"input_{parameter.name}"
            values[key] = _input_value(rng, buffer)
            if input_bindings.setdefault(logical_ref, key) != key:
                raise ValueError("multiple Harness inputs bind one J tensor")
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=key,
                    dtype=buffer.dtype,
                )
            )
        elif buffer.role is BufferRole.OUTPUT:
            if logical_ref not in j_outputs or parameter.mode is ParameterMode.INPUT:
                raise ValueError("Harness output buffer is not a J output")
            key = buffer.output_key or f"output_{parameter.name}"
            if output_bindings.setdefault(logical_ref, key) != key:
                raise ValueError("multiple Harness outputs bind one J tensor")
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=buffer.dtype,
                    shape=buffer.shape,
                    output_key=key,
                )
            )
        else:
            if logical_ref not in j_inputs or logical_ref not in j_outputs:
                raise ValueError("Harness inout buffer must anchor J input and output")
            input_key = f"input_{parameter.name}"
            output_key = buffer.output_key or f"output_{parameter.name}"
            values[input_key] = _input_value(rng, buffer)
            input_bindings[logical_ref] = input_key
            output_bindings[logical_ref] = output_key
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INOUT,
                    value_key=input_key,
                    dtype=buffer.dtype,
                    output_key=output_key,
                )
            )

    if set(input_bindings) != set(j_inputs):
        raise ValueError("Harness input bindings differ from accepted J inputs")
    if set(output_bindings) != set(j_outputs):
        raise ValueError("Harness output bindings differ from accepted J outputs")
    refeval = ScalarLoweringSpec.create(
        artifact.j,
        dims=dimensions,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    execution = TritonExecutionSpec(
        tuple(arguments),
        adapter.grid,
        artifact.c.values,
    )
    output_keys = tuple(
        output_bindings[item.name] for item in artifact.j.intent.outputs
    )
    case = Gate2Case(
        "finite-hidden-1",
        refeval.j,
        values,
        execution,
        refeval,
        output_keys,
        {
            "test_mode": "finite_hidden_after_gate1",
            "adapter_protocol": adapter.protocol,
        },
    )
    return Gate2Suite(adapter.case_id, (case,), prepared.request.binding.c_values)


def classify_scrc_report(report: SCRCRunReport) -> BatchClassification:
    """Project one canonical lifecycle report into the batch taxonomy."""

    if report.main_status is MainStatus.SCRC_CERTIFIED:
        return BatchClassification.SCRC_CERTIFIED
    if report.main_status is MainStatus.GATE2_FAILED:
        statuses = {
            case.status for case in (report.gate2.cases if report.gate2 else ())
        }
        if statuses == {Gate2Status.REFERENCE_UNSUPPORTED}:
            return BatchClassification.REFERENCE_UNSUPPORTED
        return BatchClassification.GATE2_FAILED

    if not report.attempts:
        return BatchClassification.LIFT_RESPONSE_REJECTED
    terminal = report.attempts[-1]
    if terminal.formal_round is None:
        if terminal.typecheck is not None:
            return BatchClassification.TYPECHECK_FAILED
        return BatchClassification.LIFT_RESPONSE_REJECTED
    gate1 = terminal.gate1
    if gate1 is None:
        return BatchClassification.NO_CLOSURE
    diagnostic = gate1.normal.diagnostic or ""
    if diagnostic.startswith(
        SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX
    ):
        return BatchClassification.OBSERVER_UNSUPPORTED
    if diagnostic.startswith("harness binding failed"):
        return BatchClassification.HARNESS_INVALID
    if gate1.normal.generation is None:
        return BatchClassification.REPLAY_RESPONSE_REJECTED
    if (
        gate1.normal.compile_result is not None
        and not gate1.normal.compile_result.ok
    ):
        return BatchClassification.REPLAY_COMPILE_FAILED
    return BatchClassification.NO_CLOSURE


def _attempts_diagnostic(attempts: tuple[Any, ...]) -> str:
    if attempts:
        terminal = attempts[-1]
        if terminal.feedback is not None:
            diagnostics = terminal.feedback.diagnostics
            if isinstance(diagnostics.get("diagnostic"), str):
                return diagnostics["diagnostic"]
            typecheck = diagnostics.get("typecheck")
            if isinstance(typecheck, Mapping):
                items = typecheck.get("diagnostics")
                if isinstance(items, list):
                    return "; ".join(
                        f"{item.get('code')}@{item.get('path')}: "
                        f"{item.get('detail')}"
                        for item in items
                        if isinstance(item, Mapping)
                    )
        if terminal.gate1 is not None:
            if terminal.gate1.normal.diagnostic:
                return terminal.gate1.normal.diagnostic
            comparison = terminal.gate1.normal.comparison
            if comparison is not None and not comparison.hard_pass:
                failed = [
                    name
                    for name, passed in comparison.hard_passes.items()
                    if not passed
                ]
                return "Gate1 hard mismatch: " + ", ".join(failed)
    return ""


def scrc_report_diagnostic(report: SCRCRunReport) -> str:
    if report.gate2 is not None:
        failures = [
            case
            for case in report.gate2.cases
            if case.status is not Gate2Status.PASS
        ]
        if failures:
            case = failures[0]
            return case.source.error or case.replay.error or case.reference.error
    return _attempts_diagnostic(report.attempts)


def scrc_gate1_search_diagnostic(report: SCRCGate1SearchReport) -> str:
    return _attempts_diagnostic(report.attempts)


def scrc_attempt_stage_counts(
    report: SCRCRunReport | SCRCGate1SearchReport,
) -> dict[str, int]:
    stages: Counter[str] = Counter()
    for attempt in report.attempts:
        if attempt.formal_round is None:
            stages[
                "typecheck_failed"
                if attempt.typecheck is not None
                else "lift_response_rejected"
            ] += 1
            continue
        gate1 = attempt.gate1
        if gate1 is None:
            stages["formal_gate1_missing"] += 1
        elif (gate1.normal.diagnostic or "").startswith(
            SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX
        ):
            stages["observer_unsupported"] += 1
        elif (gate1.normal.diagnostic or "").startswith(
            "harness binding failed"
        ):
            stages["harness_invalid"] += 1
        elif gate1.normal.generation is None:
            stages["replay_response_rejected"] += 1
        elif (
            gate1.normal.compile_result is not None
            and not gate1.normal.compile_result.ok
        ):
            stages["replay_compile_failed"] += 1
        elif gate1.passed:
            stages["replay_closed"] += 1
        else:
            stages["gate1_hard_mismatch"] += 1
    return dict(sorted(stages.items()))


@dataclass(frozen=True)
class WorkloadExecution:
    report: SCRCRunReport
    lift: RecordedLiftAdapter
    replay_writers: tuple[RecordedReplayWriter, ...]


@dataclass(frozen=True)
class WorkloadGate1Search:
    report: SCRCGate1SearchReport
    runner: SCRCGateRunner
    lift: RecordedLiftAdapter
    replay_writers: tuple[RecordedReplayWriter, ...]


@dataclass(frozen=True)
class WorkloadGate1Recheck:
    """One diagnostic-only Gate1 run over an already accepted artifact.

    This result is deliberately outside the SCRC candidate/winner lifecycle.
    It cannot select a candidate, change a prior verdict, or enter Gate2.
    """

    artifact: Gate1Artifact
    artifact_digest: str
    typecheck: TypeCheckReport
    gate1: Gate1ExecutionResult | None
    replay_writer: RecordedReplayWriter | None
    protocol: str = GATE1_DIAGNOSTIC_RECHECK_PROTOCOL
    diagnostic_only: bool = True

    @property
    def passed(self) -> bool:
        return self.gate1 is not None and self.gate1.passed


class WorkloadExecutionError(RuntimeError):
    def __init__(
        self,
        cause: BaseException,
        lift: RecordedLiftAdapter,
        replay_writers: tuple[RecordedReplayWriter, ...],
    ) -> None:
        self.cause = cause
        self.lift = lift
        self.replay_writers = replay_writers
        evidence = getattr(cause, "evidence", None)
        self.evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        super().__init__(f"{type(cause).__name__}: {cause}")


def recheck_prepared_workload_gate1(
    prepared: PreparedTritonWorkload,
    artifact: Gate1Artifact,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    replay_backend: str = "responses",
) -> WorkloadGate1Recheck:
    """Re-run normal Gate1 for one fixed artifact without Lift or Gate2.

    The input remains the ordinary ``Gate1Artifact``; this function creates no
    repair artifact or alternate certificate object.  Canonicalization and the
    authoritative TypeCheck run before any provider call.  A TypeCheck failure
    is returned as diagnostic evidence and never reaches Replay.
    """

    canonical = canonicalize_gate1_artifact(artifact)
    digest = canonical_artifact_digest(canonical)
    checked = type_check(canonical.j, canonical.c, canonical.pi)
    if not checked.passed:
        return WorkloadGate1Recheck(
            canonical,
            digest,
            checked,
            None,
            None,
        )

    writer = _new_replay_writer(
        replay_backend,
        provider=provider,
        effort=effort,
    )
    gate1 = BoundGate1Adapter(
        Gate1Runner(
            compiler=TritonCompiler(timeout=240.0),
            writer=writer,
            config=Gate1Config(run_masked_j=False),
        )
    ).run(canonical, prepared.request)
    if gate1.artifact_digest != digest:
        raise RuntimeError("fixed-artifact Gate1 digest changed during recheck")
    if gate1.masked_j is not None or gate1.masked_j_calls != 0:
        raise RuntimeError("diagnostic Gate1 recheck must not run masked-J")
    return WorkloadGate1Recheck(
        canonical,
        digest,
        checked,
        gate1,
        writer,
    )


def _build_workload_runtime(
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
) -> tuple[SCRCGateRunner, RecordedLiftAdapter, list[RecordedReplayWriter]]:
    lift_class: type[RecordedLiftAdapter]
    if lift_backend == "responses":
        lift_class = ResponsesSCRCLiftAdapter
    elif lift_backend == "miniswe":
        lift_class = MiniSWELiftAdapter
    else:
        raise ValueError(f"unknown Lift backend: {lift_backend!r}")
    lift = lift_class(
        provider=provider,
        compiler_view=prepared.compiler_view,
        effort=effort,
        source_symbol=prepared.entry.symbol,
    )
    writers: list[RecordedReplayWriter] = []
    compiler = TritonCompiler(timeout=240.0)

    def gate1_factory(_formal_round: int) -> BoundGate1Adapter:
        writer = _new_replay_writer(
            replay_backend,
            provider=provider,
            effort=effort,
        )
        writers.append(writer)
        return BoundGate1Adapter(
            Gate1Runner(
                compiler=compiler,
                writer=writer,
                config=Gate1Config(run_masked_j=run_masked_j),
            )
        )

    gate2 = TritonGate2ExecutionAdapter(
        Gate2Runner(),
        lambda artifact, _gate1, _binding: build_workload_gate2_suite(
            prepared, artifact
        ),
    )
    runner = SCRCGateRunner(SCRCRunDependencies(lift, gate1_factory, gate2))
    return runner, lift, writers


def search_prepared_workload_gate1(
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
    max_gate1_rounds: int | None = None,
    max_pre_replay_rejections: int | None = None,
    stop_requested: Callable[[], int | None] | None = None,
    progress_callback: SCRCProgressCallback | None = None,
) -> WorkloadGate1Search:
    """Run one independent branch only through TypeCheck and Gate1."""

    runner, lift, writers = _build_workload_runtime(
        prepared,
        provider=provider,
        effort=effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
        replay_backend=replay_backend,
    )
    request = prepared.request
    if max_gate1_rounds is not None:
        request = replace(request, max_gate1_rounds=max_gate1_rounds)
    if max_pre_replay_rejections is not None:
        request = replace(
            request,
            max_pre_replay_rejections=max_pre_replay_rejections,
        )
    try:
        report = runner.search_gate1(
            request,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise WorkloadExecutionError(exc, lift, tuple(writers)) from exc
    return WorkloadGate1Search(report, runner, lift, tuple(writers))


def complete_prepared_workload_gate2(
    prepared: PreparedTritonWorkload,
    search: WorkloadGate1Search,
) -> WorkloadExecution:
    """Execute hidden Gate2 once for the preselected Gate1-closed branch."""

    if not search.report.closed:
        raise ValueError("cannot execute Gate2 for a branch without Gate1 closure")
    try:
        report = search.runner.complete_gate2(
            prepared.request, search.report
        )
    except Exception as exc:
        raise WorkloadExecutionError(
            exc, search.lift, search.replay_writers
        ) from exc
    return WorkloadExecution(report, search.lift, search.replay_writers)


def execute_prepared_workload(
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
    progress_callback: SCRCProgressCallback | None = None,
) -> WorkloadExecution:
    """Run one independent real case through the canonical full lifecycle."""

    search = search_prepared_workload_gate1(
        prepared,
        provider=provider,
        effort=effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
        replay_backend=replay_backend,
        progress_callback=progress_callback,
    )
    if search.report.closed:
        return complete_prepared_workload_gate2(prepared, search)
    report = SCRCRunReport(
        MainStatus.NO_CLOSURE,
        search.report.attempts,
        None,
        None,
        None,
        search.report.call_budget,
    )
    return WorkloadExecution(report, search.lift, search.replay_writers)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _case_slug(case_id: str) -> str:
    identity = case_id.removeprefix("high_performance:")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", identity).strip("_").lower()
    if not slug:
        raise ValueError("workload id does not form a run directory name")
    return slug


def _next_evidence_dir(
    output_root: str | Path,
    case_id: str,
    *,
    prefix: str,
) -> Path:
    parent = Path(output_root) / _case_slug(case_id)
    parent.mkdir(parents=True, exist_ok=True)
    indexes = [
        int(path.name.removeprefix(prefix))
        for path in parent.glob(f"{prefix}[0-9][0-9][0-9]")
        if path.name.removeprefix(prefix).isdigit()
    ]
    index = max(indexes, default=0) + 1
    while True:
        destination = parent / f"{prefix}{index:03d}"
        try:
            # mkdir is the allocation primitive.  If another worker/process
            # wins the same number after our directory scan, advance without
            # reusing or overwriting any evidence root.
            destination.mkdir()
        except FileExistsError:
            index += 1
            continue
        return destination


def next_workload_run_dir(output_root: str | Path, case_id: str) -> Path:
    return _next_evidence_dir(output_root, case_id, prefix="run-")


def next_candidate_fanout_dir(
    output_root: str | Path,
    case_id: str,
) -> Path:
    return _next_evidence_dir(output_root, case_id, prefix="fanout-")


def next_gate1_recheck_dir(
    output_root: str | Path,
    case_id: str,
) -> Path:
    """Atomically allocate evidence isolated from canonical ``run-*`` data."""

    return _next_evidence_dir(output_root, case_id, prefix="recheck-")


def _read_recheck_lineage_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"parent Gate1 evidence lacks {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parent Gate1 {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"parent Gate1 {label} must be a JSON object")
    return value


def validate_workload_gate1_recheck_parent(
    parent_evidence: str | Path,
    prepared: PreparedTritonWorkload,
    artifact: Gate1Artifact,
) -> Path:
    """Bind a diagnostic recheck to one exact parent case/source/artifact.

    This is lineage validation only, not a Gate or certificate predicate.  It
    fails closed before any provider call when an old evidence directory lacks
    the request/source/digest receipts needed to identify its parent run.
    """

    parent = Path(parent_evidence).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent Gate1 evidence directory does not exist")

    digest = canonical_artifact_digest(canonicalize_gate1_artifact(artifact))
    gate1 = _read_recheck_lineage_object(parent / "gate1.json", "gate1.json")
    if gate1.get("artifact_digest") != digest:
        raise ValueError("parent Gate1 artifact digest differs from input artifact")

    lineage_root: Path | None = None
    required_names = ("request.json", "source_projection.json", "source.py")
    for candidate in (parent, *tuple(parent.parents)[:4]):
        present = tuple((candidate / name).is_file() for name in required_names)
        if not any(present):
            continue
        if not all(present):
            raise ValueError("parent Gate1 lineage receipts are incomplete")
        lineage_root = candidate
        break
    if lineage_root is None:
        raise ValueError("parent Gate1 evidence lacks request/source lineage")

    request = _read_recheck_lineage_object(
        lineage_root / "request.json", "request.json"
    )
    expected_request = {
        "case_id": prepared.entry.case_id,
        "frontend": prepared.entry.frontend,
        "c0": prepared.request.binding.values,
        "grid": list(prepared.adapter.grid),
        "source_to_opaque": dict(prepared.source_to_opaque),
    }
    missing_request = sorted(set(expected_request) - set(request))
    if missing_request:
        raise ValueError(
            "parent Gate1 request lacks lineage field(s): "
            + ", ".join(missing_request)
        )
    mismatched_request = sorted(
        key for key, value in expected_request.items() if request[key] != value
    )
    if mismatched_request:
        raise ValueError(
            "parent Gate1 request differs from selected workload: "
            + ", ".join(mismatched_request)
        )
    if artifact.c.values != request["c0"]:
        raise ValueError("input artifact C differs from parent Gate1 request c0")

    projection = _read_recheck_lineage_object(
        lineage_root / "source_projection.json", "source_projection.json"
    )
    expected_projection = {
        **prepared.source_slice.to_json(),
        "source_location": prepared.source_location,
    }
    if projection != expected_projection:
        raise ValueError(
            "parent Gate1 source projection differs from selected workload"
        )
    try:
        parent_source = (lineage_root / "source.py").read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"parent Gate1 source.py is unreadable: {exc}") from exc
    if parent_source != prepared.source_slice.code:
        raise ValueError("parent Gate1 source differs from selected workload")
    return lineage_root


def initialize_workload_gate1_recheck_evidence(
    recheck_dir: Path,
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None,
    parent_evidence: str | Path,
    replay_backend: str = "responses",
) -> None:
    """Initialize an immutable diagnostic recheck directory.

    ``parent_evidence`` is a reference only.  No file below it is opened for
    writing, and this evidence cannot replace a prior SCRC verdict.
    """

    parent = Path(parent_evidence).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent Gate1 evidence directory does not exist")
    if not recheck_dir.is_dir():
        raise ValueError("Gate1 recheck evidence directory was not allocated")
    if any(recheck_dir.iterdir()):
        raise ValueError("Gate1 recheck evidence directory must be empty")

    (recheck_dir / "source.py").write_text(
        prepared.source_slice.code, encoding="utf-8"
    )
    _write_json(recheck_dir / "adapter.json", prepared.adapter.to_json())
    _write_json(
        recheck_dir / "source_projection.json",
        {
            **prepared.source_slice.to_json(),
            "source_location": prepared.source_location,
        },
    )
    _write_json(
        recheck_dir / "request.json",
        {
            "protocol": GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
            "diagnostic_only": True,
            "enters_gate2": False,
            "changes_prior_verdict": False,
            "case_id": prepared.entry.case_id,
            "parent_evidence": str(parent),
            "provider": provider.name,
            "model": provider.model,
            "reasoning_effort": effort or provider.reasoning_effort,
            "replay_backend": replay_backend,
            "frontend": prepared.entry.frontend,
            "observer_profile": prepared.entry.observer_profile,
            "c0": prepared.request.binding.values,
            "grid": list(prepared.adapter.grid),
            "run_masked_j": False,
            "source_to_opaque": dict(prepared.source_to_opaque),
        },
    )


def initialize_workload_evidence(
    run_dir: Path,
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None,
    run_masked_j: bool,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
    search_metadata: Mapping[str, Any] | None = None,
) -> None:
    (run_dir / "source.py").write_text(
        prepared.source_slice.code, encoding="utf-8"
    )
    _write_json(run_dir / "adapter.json", prepared.adapter.to_json())
    _write_json(
        run_dir / "source_projection.json",
        {
            **prepared.source_slice.to_json(),
            "source_location": prepared.source_location,
        },
    )
    request_record = {
        "case_id": prepared.entry.case_id,
        "provider": provider.name,
        "model": provider.model,
        "reasoning_effort": effort or provider.reasoning_effort,
        "lift_backend": lift_backend,
        "replay_backend": replay_backend,
        "frontend": prepared.entry.frontend,
        "observer_profile": prepared.entry.observer_profile,
        "q_policy": prepared.entry.q_policy,
        "c0": prepared.request.binding.values,
        "grid": list(prepared.adapter.grid),
        "run_masked_j": run_masked_j,
        "source_to_opaque": dict(prepared.source_to_opaque),
    }
    if search_metadata is not None:
        request_record["candidate_search"] = dict(search_metadata)
    _write_json(run_dir / "request.json", request_record)


def _replay_receipt(record: ReplayGeneration | ReplayAttempt) -> dict[str, Any]:
    result: dict[str, Any] = {
        "recorded": True,
        "backend_protocol": record.backend_protocol,
        "episode_contract_sha256": record.episode_contract_sha256,
        "model": record.model,
        "response_model": record.response_model,
        "response_id": record.response_id,
        "usage": dict(record.usage),
        "attempts": record.attempts,
        "physical_provider_calls": record.physical_provider_calls,
        "transport_retries": record.transport_retries,
        "elapsed_ms": record.elapsed_ms,
        "request_sha256": record.request_sha256,
        "response_sha256": record.response_sha256,
        "raw_response_sha256": record.raw_response_sha256,
    }
    if isinstance(record, ReplayGeneration):
        result.update(
            {
                "status": "generated",
                "packet_digest": record.packet_digest,
                "prompt_digest": record.prompt_digest,
                "context_receipt": dict(record.context_receipt),
                "function_name": record.function_name,
            }
        )
    else:
        result.update(
            {
                "status": "rejected",
                "diagnostic": record.diagnostic,
            }
        )
    return result


_TRAJECTORY_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    }
)


def _canonical_replay_trajectory_bytes(
    trajectory: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _assert_sanitized_replay_trajectory(
    value: Any,
    *,
    key: str = "",
) -> None:
    normalized = key.casefold().replace("-", "_")
    if (
        normalized in _TRAJECTORY_SECRET_KEYS
        or normalized.endswith("_api_key")
    ) and value != "<redacted>":
        raise ValueError("Replay trajectory contains an unredacted secret field")
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            _assert_sanitized_replay_trajectory(item, key=str(item_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_sanitized_replay_trajectory(item)


def _verified_replay_trajectory_bytes(
    record: ReplayGeneration | ReplayAttempt,
) -> bytes | None:
    if not record.trajectory:
        return None
    _assert_sanitized_replay_trajectory(record.trajectory)
    encoded = _canonical_replay_trajectory_bytes(record.trajectory)
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != record.raw_response_sha256:
        raise ValueError(
            "Replay trajectory canonical hash differs from raw_response_sha256"
        )
    if record.raw_response and record.raw_response != encoded:
        raise ValueError("Replay raw_response differs from canonical trajectory bytes")
    return encoded


def _save_replay_record_evidence(
    directory: Path,
    *,
    prefix: str,
    exact_source_name: str,
    record: ReplayGeneration | ReplayAttempt,
) -> None:
    """Persist exact source, one canonical receipt, and verified trajectory."""

    trajectory_bytes = _verified_replay_trajectory_bytes(record)
    (directory / exact_source_name).write_text(
        record.response_text,
        encoding="utf-8",
    )
    if isinstance(record, ReplayGeneration) and record.code != record.response_text:
        (directory / f"{prefix}_normalized.py").write_text(
            record.code,
            encoding="utf-8",
        )
    _write_json(directory / f"{prefix}_receipt.json", _replay_receipt(record))
    if trajectory_bytes is not None:
        # Write the parsed sanitized value only after its canonical byte hash has
        # been bound to the record receipt.
        _write_json(directory / f"{prefix}_trajectory.json", record.trajectory)


def save_workload_gate1_recheck(
    recheck_dir: Path,
    prepared: PreparedTritonWorkload,
    result: WorkloadGate1Recheck,
) -> None:
    """Persist fixed-artifact diagnostic evidence without a new verdict."""

    request_path = recheck_dir / "request.json"
    if not request_path.is_file():
        raise ValueError("Gate1 recheck evidence was not initialized")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != GATE1_DIAGNOSTIC_RECHECK_PROTOCOL:
        raise ValueError("Gate1 recheck evidence protocol differs")
    if request.get("case_id") != prepared.entry.case_id:
        raise ValueError("Gate1 recheck evidence case differs")
    if not result.diagnostic_only:
        raise ValueError("fixed-artifact recheck result must be diagnostic-only")
    if canonical_artifact_digest(result.artifact) != result.artifact_digest:
        raise ValueError("fixed-artifact recheck digest differs from artifact")

    _write_json(recheck_dir / "artifact.json", result.artifact.to_json())
    _write_json(recheck_dir / "typecheck.json", result.typecheck.to_json())
    if result.gate1 is not None:
        _write_json(recheck_dir / "gate1.json", result.gate1.to_json())

    records = (
        tuple(result.replay_writer.records)
        if result.replay_writer is not None
        else ()
    )
    if len(records) > 1:
        raise ValueError("normal-only Gate1 recheck produced multiple Replay records")
    record: ReplayGeneration | ReplayAttempt | None = (
        records[0] if records else None
    )
    if record is None and result.gate1 is not None:
        record = result.gate1.normal.generation
    if isinstance(record, ReplayGeneration):
        _save_replay_record_evidence(
            recheck_dir,
            prefix="replay",
            exact_source_name="replay.py",
            record=record,
        )
    elif isinstance(record, ReplayAttempt):
        _save_replay_record_evidence(
            recheck_dir,
            prefix="replay",
            exact_source_name="replay_rejected.txt",
            record=record,
        )
    else:
        _write_json(
            recheck_dir / "replay_receipt.json",
            {
                "recorded": False,
                "status": "not_run" if result.gate1 is None else "unavailable",
                "diagnostic": (
                    result.gate1.normal.diagnostic
                    if result.gate1 is not None
                    else "TypeCheck rejected the fixed artifact before Replay"
                ),
            },
        )

    status = (
        "typecheck_failed"
        if result.gate1 is None
        else result.gate1.status.value
    )
    _write_json(
        recheck_dir / "recheck.json",
        {
            "protocol": GATE1_DIAGNOSTIC_RECHECK_PROTOCOL,
            "diagnostic_only": True,
            "enters_gate2": False,
            "changes_prior_verdict": False,
            "case_id": prepared.entry.case_id,
            "parent_evidence": request["parent_evidence"],
            "artifact_digest": result.artifact_digest,
            "artifact_schema": result.artifact.wire_schema,
            "typecheck_protocol": result.typecheck.protocol,
            "typecheck_status": result.typecheck.status.value,
            "gate1_protocol": (
                result.gate1.protocol if result.gate1 is not None else None
            ),
            "status": status,
        },
    )
    (recheck_dir / "SUMMARY.md").write_text(
        f"# {prepared.entry.case_id} {recheck_dir.name}\n\n"
        f"- protocol: `{GATE1_DIAGNOSTIC_RECHECK_PROTOCOL}`\n"
        "- diagnostic_only: `true`\n"
        "- enters_gate2: `false`\n"
        "- changes_prior_verdict: `false`\n"
        f"- parent_evidence: `{request['parent_evidence']}`\n"
        f"- artifact_digest: `{result.artifact_digest}`\n"
        f"- status: `{status}`\n",
        encoding="utf-8",
    )


def save_workload_execution(
    run_dir: Path,
    prepared: PreparedTritonWorkload,
    execution: WorkloadExecution,
) -> None:
    report = execution.report
    _write_json(run_dir / "report.json", report.to_json())
    _save_attempt_evidence(
        run_dir,
        report.attempts,
        execution.lift,
        execution.replay_writers,
    )
    if report.gate2 is not None:
        _write_json(run_dir / "gate2.json", report.gate2.to_json())
    classification = classify_scrc_report(report)
    summary = [
        f"# {prepared.entry.case_id} {run_dir.name}",
        "",
        f"- classification: `{classification.value}`",
        f"- main_status: `{report.main_status.value}`",
        f"- accepted_attempt: `{report.accepted_attempt}`",
        "- j_necessity_status: `"
        + (
            report.j_necessity_status.value
            if report.j_necessity_status is not None
            else "NOT_RUN"
        )
        + "`",
        "- call_budget: `"
        + json.dumps(report.call_budget.to_json(), sort_keys=True)
        + "`",
    ]
    if report.gate2 is not None:
        summary.append(
            "- gate2_status_counts: `"
            + json.dumps(report.gate2.status_counts, sort_keys=True)
            + "`"
        )
    summary.extend(
        [
            "",
            "本目录中的 digest/receipt 仅是普通运行追溯信息，不参与 TypeCheck、Gate1 或 Gate2 判定。",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def _save_attempt_evidence(
    run_dir: Path,
    attempts: tuple[Any, ...],
    lift: RecordedLiftAdapter,
    replay_writers: tuple[RecordedReplayWriter, ...],
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        attempt_dir = run_dir / f"attempt-{index:02d}"
        attempt_dir.mkdir()
        if attempt.artifact is not None:
            _write_json(attempt_dir / "artifact.json", attempt.artifact.to_json())
        if attempt.typecheck is not None:
            _write_json(
                attempt_dir / "typecheck.json", attempt.typecheck.to_json()
            )
        if attempt.gate1 is not None:
            _write_json(attempt_dir / "gate1.json", attempt.gate1.to_json())
        if index <= len(lift.records):
            record = lift.records[index - 1]
            if record.extraction is not None:
                provenance = record.extraction.provenance
                _write_json(
                    attempt_dir / "lift_receipt.json",
                    {
                        "model": provenance.model,
                        "response_model": provenance.response_model,
                        "response_id": provenance.response_id,
                        "usage": dict(provenance.usage),
                        "attempts": provenance.attempts,
                        "elapsed_ms": provenance.elapsed_ms,
                        "backend": provenance.backend,
                        "termination_status": provenance.termination_status,
                    },
                )
                (attempt_dir / "lift_response.txt").write_text(
                    provenance.response_text, encoding="utf-8"
                )
                if provenance.trajectory is not None:
                    _write_json(
                        attempt_dir / "lift_trajectory.json",
                        provenance.trajectory,
                    )
            else:
                (attempt_dir / "lift_rejection.txt").write_text(
                    (record.diagnostic or "Lift response rejected") + "\n",
                    encoding="utf-8",
                )
                for rejected_index, rejected in enumerate(
                    record.rejected_attempts, start=1
                ):
                    (attempt_dir / f"lift_rejected_{rejected_index:02d}.txt").write_text(
                        rejected.response_text, encoding="utf-8"
                    )
                    _write_json(
                        attempt_dir
                        / f"lift_rejected_{rejected_index:02d}_receipt.json",
                        {
                            "model": rejected.model,
                            "response_model": rejected.response_model,
                            "response_id": rejected.response_id,
                            "usage": dict(rejected.usage),
                            "attempts": rejected.attempts,
                            "elapsed_ms": rejected.elapsed_ms,
                            "backend": rejected.backend,
                            "termination_status": rejected.termination_status,
                            "diagnostic": rejected.diagnostic,
                        },
                    )
                    if rejected.trajectory is not None:
                        _write_json(
                            attempt_dir
                            / f"lift_rejected_{rejected_index:02d}_trajectory.json",
                            rejected.trajectory,
                        )
            if record.backend_evidence is not None:
                worker = dict(record.backend_evidence)
                trajectory = worker.pop("trajectory", None)
                _write_json(attempt_dir / "lift_worker.json", worker)
                if trajectory is not None and not (
                    record.extraction is not None
                    and record.extraction.provenance.trajectory is not None
                ):
                    _write_json(attempt_dir / "lift_trajectory.json", trajectory)
            if record.continuation_candidate is not None:
                _write_json(
                    attempt_dir / "unaccepted_lift_continuation.json",
                    dict(record.continuation_candidate),
                )
    formal_attempts = [
        (run_dir / f"attempt-{index:02d}", attempt)
        for index, attempt in enumerate(attempts, start=1)
        if attempt.formal_round is not None
    ]
    for (attempt_dir, attempt), writer in zip(
        formal_attempts, replay_writers, strict=True
    ):
        records = list(writer.records)
        if not records and attempt.gate1 is not None:
            normal = attempt.gate1.normal.generation
            masked = (
                attempt.gate1.masked_j.generation
                if attempt.gate1.masked_j is not None
                else None
            )
            records.extend(
                record for record in (normal, masked) if record is not None
            )
        for record_index, record in enumerate(records, start=1):
            if record_index == 1:
                prefix = "replay"
            elif record_index == 2:
                prefix = "masked_j_replay"
            else:
                prefix = f"replay_extra_{record_index:02d}"
            exact_name = (
                f"{prefix}.py"
                if isinstance(record, ReplayGeneration)
                else f"{prefix}_rejected.txt"
            )
            _save_replay_record_evidence(
                attempt_dir,
                prefix=prefix,
                exact_source_name=exact_name,
                record=record,
            )


def save_workload_gate1_search(
    run_dir: Path,
    prepared: PreparedTritonWorkload,
    search: WorkloadGate1Search,
    *,
    selected_for_gate2: bool,
) -> None:
    report = search.report
    _write_json(run_dir / "gate1_search.json", report.to_json())
    _save_attempt_evidence(
        run_dir,
        report.attempts,
        search.lift,
        search.replay_writers,
    )
    summary = [
        f"# {prepared.entry.case_id} {run_dir.name}",
        "",
        "- phase: `pre_gate2_candidate_search`",
        f"- status: `{report.status.value}`",
        f"- accepted_attempt: `{report.accepted_attempt}`",
        f"- selected_for_gate2: `{str(selected_for_gate2).lower()}`",
        "- call_budget: `"
        + json.dumps(report.call_budget.to_json(), sort_keys=True)
        + "`",
        "",
        "该目录是候选搜索 evidence，不是第四个证明 artifact。未被选择的 Gate1-closed "
        "candidate 不执行 Gate2。",
        "",
    ]
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def save_candidate_fanout_report(
    fanout_dir: Path,
    report: CandidateFanoutSearchReport,
    *,
    winner_result: Mapping[str, Any] | None,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
) -> None:
    payload = report.to_json()
    payload["lift_backend"] = lift_backend
    payload["replay_backend"] = replay_backend
    payload["winner_result"] = (
        dict(winner_result) if winner_result is not None else None
    )
    _write_json(fanout_dir / "fanout-report.json", payload)
    summary = [
        f"# {report.case_id} {fanout_dir.name}",
        "",
        f"- candidate_count: `{len(report.branches)}`",
        f"- selection_rule: `{report.selection_rule}`",
        f"- lift_backend: `{lift_backend}`",
        f"- replay_backend: `{replay_backend}`",
        "- winner_branch_id: `"
        + (str(report.winner.branch_id) if report.winner is not None else "None")
        + "`",
        "- pre_gate2_status_counts: `"
        + json.dumps(report.counts, sort_keys=True)
        + "`",
    ]
    if winner_result is not None:
        summary.append(
            f"- winner_main_status: `{winner_result.get('main_status')}`"
        )
    summary.extend(
        [
            "",
            "所有 branch 使用同一 Source/Harness/c0，但拥有独立 Lift、Replay、TypeCheck、"
            "Gate1 与 relift history。Gate2 只对预先选定的 winner 执行。",
            "",
        ]
    )
    (fanout_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def save_workload_failure(
    run_dir: Path,
    error: BaseException,
    *,
    evidence: Mapping[str, Any] | None = None,
    runtime: WorkloadExecutionError | None = None,
) -> None:
    cause = runtime.cause if runtime is not None else error
    _write_json(
        run_dir / "failure.json",
        {
            "failure_class": type(cause).__name__,
            "diagnostic": str(cause),
            "partial_evidence": dict(evidence or {}),
        },
    )
    if runtime is not None:
        for index, record in enumerate(runtime.lift.records, start=1):
            call_dir = run_dir / f"lift-call-{index:02d}"
            call_dir.mkdir()
            if record.extraction is not None:
                extraction = record.extraction
                _write_json(
                    call_dir / "artifact.json", extraction.artifact.to_json()
                )
                (call_dir / "response.txt").write_text(
                    extraction.provenance.response_text, encoding="utf-8"
                )
                _write_json(
                    call_dir / "receipt.json",
                    {
                        "model": extraction.provenance.model,
                        "response_model": extraction.provenance.response_model,
                        "response_id": extraction.provenance.response_id,
                        "usage": dict(extraction.provenance.usage),
                        "attempts": extraction.provenance.attempts,
                        "elapsed_ms": extraction.provenance.elapsed_ms,
                        "backend": extraction.provenance.backend,
                        "termination_status": extraction.provenance.termination_status,
                    },
                )
                if extraction.provenance.trajectory is not None:
                    _write_json(
                        call_dir / "trajectory.json",
                        extraction.provenance.trajectory,
                    )
            else:
                (call_dir / "rejection.txt").write_text(
                    (record.diagnostic or "Lift response rejected") + "\n",
                    encoding="utf-8",
                )
                for rejected_index, rejected in enumerate(
                    record.rejected_attempts, start=1
                ):
                    (call_dir / f"response-{rejected_index:02d}.txt").write_text(
                        rejected.response_text, encoding="utf-8"
                    )
                    if rejected.trajectory is not None:
                        _write_json(
                            call_dir / f"trajectory-{rejected_index:02d}.json",
                            rejected.trajectory,
                        )
            if record.backend_evidence is not None:
                worker = dict(record.backend_evidence)
                trajectory = worker.pop("trajectory", None)
                _write_json(call_dir / "worker.json", worker)
                if trajectory is not None and not (call_dir / "trajectory.json").exists():
                    _write_json(call_dir / "trajectory.json", trajectory)
            if record.continuation_candidate is not None:
                _write_json(
                    call_dir / "unaccepted_lift_continuation.json",
                    dict(record.continuation_candidate),
                )
        for writer_index, writer in enumerate(
            runtime.replay_writers, start=1
        ):
            call_dir = run_dir / f"gate1-call-{writer_index:02d}"
            call_dir.mkdir()
            for record_index, record in enumerate(writer.records, start=1):
                prefix = (
                    "replay"
                    if record_index == 1
                    else "masked_j_replay"
                    if record_index == 2
                    else f"replay_extra_{record_index:02d}"
                )
                exact_name = (
                    f"{prefix}.py"
                    if isinstance(record, ReplayGeneration)
                    else f"{prefix}_rejected.txt"
                )
                _save_replay_record_evidence(
                    call_dir,
                    prefix=prefix,
                    exact_source_name=exact_name,
                    record=record,
                )
    (run_dir / "SUMMARY.md").write_text(
        f"# {run_dir.parent.name} {run_dir.name}\n\n"
        "- classification: `infrastructure_failed`\n"
        f"- exception: `{type(cause).__name__}: {cause}`\n",
        encoding="utf-8",
    )


__all__ = [
    "GATE1_DIAGNOSTIC_RECHECK_PROTOCOL",
    "WorkloadExecution",
    "WorkloadExecutionError",
    "WorkloadGate1Search",
    "WorkloadGate1Recheck",
    "build_workload_gate2_suite",
    "classify_scrc_report",
    "complete_prepared_workload_gate2",
    "execute_prepared_workload",
    "initialize_workload_gate1_recheck_evidence",
    "initialize_workload_evidence",
    "next_candidate_fanout_dir",
    "next_gate1_recheck_dir",
    "next_workload_run_dir",
    "recheck_prepared_workload_gate1",
    "save_candidate_fanout_report",
    "save_workload_execution",
    "save_workload_failure",
    "save_workload_gate1_search",
    "save_workload_gate1_recheck",
    "scrc_gate1_search_diagnostic",
    "scrc_attempt_stage_counts",
    "scrc_report_diagnostic",
    "search_prepared_workload_gate1",
    "validate_workload_gate1_recheck_parent",
]
