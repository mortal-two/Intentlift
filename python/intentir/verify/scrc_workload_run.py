"""Real SCRC lifecycle helpers for declarative Triton workload adapters."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

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
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import (
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
    SCRCRunDependencies,
    SCRCRunReport,
)
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferRole,
    PreparedTritonWorkload,
)


RecordedLiftAdapter = ResponsesSCRCLiftAdapter | MiniSWELiftAdapter


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
    replay_writers: tuple[ResponsesReplayWriter, ...]


@dataclass(frozen=True)
class WorkloadGate1Search:
    report: SCRCGate1SearchReport
    runner: SCRCGateRunner
    lift: RecordedLiftAdapter
    replay_writers: tuple[ResponsesReplayWriter, ...]


class WorkloadExecutionError(RuntimeError):
    def __init__(
        self,
        cause: BaseException,
        lift: RecordedLiftAdapter,
        replay_writers: tuple[ResponsesReplayWriter, ...],
    ) -> None:
        self.cause = cause
        self.lift = lift
        self.replay_writers = replay_writers
        evidence = getattr(cause, "evidence", None)
        self.evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        super().__init__(f"{type(cause).__name__}: {cause}")


def _build_workload_runtime(
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
) -> tuple[SCRCGateRunner, RecordedLiftAdapter, list[ResponsesReplayWriter]]:
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
    writers: list[ResponsesReplayWriter] = []
    compiler = TritonCompiler(timeout=240.0)

    def gate1_factory(_formal_round: int) -> BoundGate1Adapter:
        writer = ResponsesReplayWriter(provider=provider, effort=effort)
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
    stop_requested: Callable[[], int | None] | None = None,
) -> WorkloadGate1Search:
    """Run one independent branch only through TypeCheck and Gate1."""

    runner, lift, writers = _build_workload_runtime(
        prepared,
        provider=provider,
        effort=effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
    )
    try:
        report = runner.search_gate1(
            prepared.request,
            stop_requested=stop_requested,
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
) -> WorkloadExecution:
    """Run one independent real case through the canonical full lifecycle."""

    search = search_prepared_workload_gate1(
        prepared,
        provider=provider,
        effort=effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
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


def next_workload_run_dir(output_root: str | Path, case_id: str) -> Path:
    parent = Path(output_root) / _case_slug(case_id)
    parent.mkdir(parents=True, exist_ok=True)
    indexes = [
        int(path.name.removeprefix("run-"))
        for path in parent.glob("run-[0-9][0-9][0-9]")
        if path.name.removeprefix("run-").isdigit()
    ]
    run_dir = parent / f"run-{max(indexes, default=0) + 1:03d}"
    run_dir.mkdir()
    return run_dir


def next_candidate_fanout_dir(
    output_root: str | Path,
    case_id: str,
) -> Path:
    parent = Path(output_root) / _case_slug(case_id)
    parent.mkdir(parents=True, exist_ok=True)
    indexes = [
        int(path.name.removeprefix("fanout-"))
        for path in parent.glob("fanout-[0-9][0-9][0-9]")
        if path.name.removeprefix("fanout-").isdigit()
    ]
    fanout_dir = parent / f"fanout-{max(indexes, default=0) + 1:03d}"
    fanout_dir.mkdir()
    return fanout_dir


def initialize_workload_evidence(
    run_dir: Path,
    prepared: PreparedTritonWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None,
    run_masked_j: bool,
    lift_backend: str = "responses",
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
    replay_writers: tuple[ResponsesReplayWriter, ...],
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
            normal = attempt.gate1.normal.generation
            if normal is not None:
                (attempt_dir / "replay.py").write_text(
                    normal.code, encoding="utf-8"
                )
            masked = (
                attempt.gate1.masked_j.generation
                if attempt.gate1.masked_j is not None
                else None
            )
            if masked is not None:
                (attempt_dir / "masked_j_replay.py").write_text(
                    masked.code, encoding="utf-8"
                )
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
    formal_attempt_dirs = [
        run_dir / f"attempt-{index:02d}"
        for index, attempt in enumerate(attempts, start=1)
        if attempt.formal_round is not None
    ]
    for attempt_dir, writer in zip(
        formal_attempt_dirs, replay_writers, strict=True
    ):
        for record_index, record in enumerate(writer.records, start=1):
            if not isinstance(record, ReplayAttempt):
                continue
            (attempt_dir / f"replay_rejected_{record_index:02d}.txt").write_text(
                record.response_text, encoding="utf-8"
            )
            _write_json(
                attempt_dir / f"replay_rejected_{record_index:02d}_receipt.json",
                {
                    "model": record.model,
                    "response_model": record.response_model,
                    "response_id": record.response_id,
                    "usage": dict(record.usage),
                    "attempts": record.attempts,
                    "elapsed_ms": record.elapsed_ms,
                    "diagnostic": record.diagnostic,
                },
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
) -> None:
    payload = report.to_json()
    payload["winner_result"] = (
        dict(winner_result) if winner_result is not None else None
    )
    _write_json(fanout_dir / "fanout-report.json", payload)
    summary = [
        f"# {report.case_id} {fanout_dir.name}",
        "",
        f"- candidate_count: `{len(report.branches)}`",
        f"- selection_rule: `{report.selection_rule}`",
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
        for writer_index, writer in enumerate(
            runtime.replay_writers, start=1
        ):
            call_dir = run_dir / f"gate1-call-{writer_index:02d}"
            call_dir.mkdir()
            for record_index, record in enumerate(writer.records, start=1):
                if isinstance(record, ReplayGeneration):
                    (call_dir / f"replay-{record_index:02d}.py").write_text(
                        record.code, encoding="utf-8"
                    )
                    diagnostic = ""
                else:
                    (call_dir / f"rejected-{record_index:02d}.txt").write_text(
                        record.response_text, encoding="utf-8"
                    )
                    diagnostic = record.diagnostic
                _write_json(
                    call_dir / f"receipt-{record_index:02d}.json",
                    {
                        "model": record.model,
                        "response_model": record.response_model,
                        "response_id": record.response_id,
                        "usage": dict(record.usage),
                        "attempts": record.attempts,
                        "elapsed_ms": record.elapsed_ms,
                        "diagnostic": diagnostic,
                    },
                )
    (run_dir / "SUMMARY.md").write_text(
        f"# {run_dir.parent.name} {run_dir.name}\n\n"
        "- classification: `infrastructure_failed`\n"
        f"- exception: `{type(cause).__name__}: {cause}`\n",
        encoding="utf-8",
    )


__all__ = [
    "WorkloadExecution",
    "WorkloadExecutionError",
    "WorkloadGate1Search",
    "build_workload_gate2_suite",
    "classify_scrc_report",
    "complete_prepared_workload_gate2",
    "execute_prepared_workload",
    "initialize_workload_evidence",
    "next_candidate_fanout_dir",
    "next_workload_run_dir",
    "save_candidate_fanout_report",
    "save_workload_execution",
    "save_workload_failure",
    "save_workload_gate1_search",
    "scrc_gate1_search_diagnostic",
    "scrc_attempt_stage_counts",
    "scrc_report_diagnostic",
    "search_prepared_workload_gate1",
]
