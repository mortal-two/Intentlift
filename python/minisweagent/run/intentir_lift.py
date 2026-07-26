"""Compose the in-repository mini-swe components for one IntentIR Lift round."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from minisweagent import Environment, Model, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.intentir_lift import (
    IntentIRLiftEnvironment,
    IntentIRLiftWorkspace,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.utils.serialize import recursive_merge


@dataclass(frozen=True)
class IntentIRLiftRunResult:
    candidate_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float
    physical_provider_calls: int | None = None
    transport_retries: int = 0
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.physical_provider_calls is None:
            object.__setattr__(
                self, "physical_provider_calls", self.model_calls
            )


class IntentIRLiftRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: IntentIRLiftRunResult,
        cause: BaseException | None = None,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(message)


@dataclass(frozen=True)
class _IntentIRLiftStageResult:
    output_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float
    physical_provider_calls: int
    transport_retries: int
    caught: Exception | None = None


def load_intentir_lift_config() -> dict[str, Any]:
    path = Path(package_dir) / "config" / "intentir_lift.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("IntentIR Lift config must be a mapping")
    return value


def create_intentir_lift_model(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    reasoning_effort: str,
    timeout_seconds: float,
) -> LitellmResponseModel:
    """Configure the existing Responses model for an OpenAI-compatible CPA."""

    config = load_intentir_lift_config()["model"]
    model_kwargs = {
        **config.pop("model_kwargs", {}),
        "api_base": api_base,
        "api_key": api_key,
        "reasoning_effort": reasoning_effort,
        "timeout": timeout_seconds,
    }
    qualified_name = model_name if "/" in model_name else f"openai/{model_name}"
    return LitellmResponseModel(
        model_name=qualified_name,
        model_kwargs=model_kwargs,
        **config,
    )


def _run_agent_stage(
    *,
    model: Model,
    environment: Environment,
    agent_config: Mapping[str, Any],
    output_reader: Callable[[], str],
) -> _IntentIRLiftStageResult:
    """Run one fresh LLM context and read only its declared opaque output."""

    physical_before = getattr(model, "physical_provider_calls", None)
    retries_before = getattr(model, "transport_retries", None)
    agent = DefaultAgent(model, environment, **dict(agent_config))
    started = time.monotonic()
    caught: Exception | None = None
    try:
        terminal = agent.run()
    except Exception as exc:
        caught = exc
        terminal = {"exit_status": type(exc).__name__, "submission": ""}
    exit_status = str(terminal.get("exit_status", ""))
    output_text: str | None = None
    if exit_status == "Submitted":
        try:
            output_text = output_reader()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    elif caught is None:
        activate_fallback = getattr(
            environment, "activate_committed_fallback", None
        )
        if callable(activate_fallback):
            try:
                output_text = activate_fallback(exit_status)
            except Exception as exc:
                caught = exc
                exit_status = "InvalidSubmission"
            else:
                if output_text is not None:
                    exit_status = "SubmittedCheckpoint"
    elapsed_ms = (time.monotonic() - started) * 1000.0
    trajectory = agent.save(None)
    physical_after = getattr(model, "physical_provider_calls", None)
    retries_after = getattr(model, "transport_retries", None)
    physical_calls = (
        physical_after - physical_before
        if isinstance(physical_before, int)
        and isinstance(physical_after, int)
        and physical_after >= physical_before
        else agent.n_calls
    )
    transport_retries = (
        retries_after - retries_before
        if isinstance(retries_before, int)
        and isinstance(retries_after, int)
        and retries_after >= retries_before
        else 0
    )
    return _IntentIRLiftStageResult(
        output_text=output_text,
        trajectory=trajectory,
        exit_status=exit_status,
        model_calls=agent.n_calls,
        elapsed_ms=elapsed_ms,
        physical_provider_calls=physical_calls,
        transport_retries=transport_retries,
        caught=caught,
    )


def _public_result(
    stage: _IntentIRLiftStageResult,
) -> IntentIRLiftRunResult:
    return IntentIRLiftRunResult(
        candidate_text=stage.output_text,
        trajectory=stage.trajectory,
        exit_status=stage.exit_status,
        model_calls=stage.model_calls,
        elapsed_ms=stage.elapsed_ms,
        physical_provider_calls=stage.physical_provider_calls,
        transport_retries=stage.transport_retries,
    )


def _tag_stage_message(message: Any, stage_name: str) -> Any:
    if not isinstance(message, Mapping):
        return message
    tagged = dict(message)
    extra = tagged.get("extra")
    tagged["extra"] = {
        **(dict(extra) if isinstance(extra, Mapping) else {}),
        "intentir_lift_stage": stage_name,
    }
    return tagged


def _aggregate_staged_result(
    stages: Mapping[str, _IntentIRLiftStageResult],
    *,
    plan_sha256: str | None,
) -> IntentIRLiftRunResult:
    ordered = list(stages.items())
    last = ordered[-1][1]
    messages: list[Any] = []
    stage_metrics: dict[str, dict[str, Any]] = {}
    stage_info: dict[str, Any] = {}
    for stage_name, stage in ordered:
        raw_messages = stage.trajectory.get("messages", ())
        if isinstance(raw_messages, list):
            messages.extend(
                _tag_stage_message(message, stage_name)
                for message in raw_messages
            )
        info = stage.trajectory.get("info")
        stage_info[stage_name] = dict(info) if isinstance(info, Mapping) else {}
        stage_metrics[stage_name] = {
            "exit_status": stage.exit_status,
            "model_calls": stage.model_calls,
            "physical_provider_calls": stage.physical_provider_calls,
            "transport_retries": stage.transport_retries,
            "elapsed_ms": stage.elapsed_ms,
        }
    final_info = last.trajectory.get("info")
    aggregate_info = dict(final_info) if isinstance(final_info, Mapping) else {}
    aggregate_info["model_stats"] = {
        **(
            dict(aggregate_info.get("model_stats", {}))
            if isinstance(aggregate_info.get("model_stats"), Mapping)
            else {}
        ),
        "api_calls": sum(stage.model_calls for _name, stage in ordered),
    }
    aggregate_info["exit_status"] = last.exit_status
    aggregate_info["lift_workflow"] = {
        "protocol": "intentir.lift-workflow.v1",
        "plan_sha256": plan_sha256,
        "stages": stage_metrics,
        "stage_info": stage_info,
    }
    trajectory = {
        "trajectory_format": "intentir-mini-swe-staged-1.0",
        "messages": messages,
        "info": aggregate_info,
    }
    return IntentIRLiftRunResult(
        candidate_text=last.output_text,
        trajectory=trajectory,
        exit_status=last.exit_status,
        model_calls=sum(stage.model_calls for _name, stage in ordered),
        elapsed_ms=sum(stage.elapsed_ms for _name, stage in ordered),
        physical_provider_calls=sum(
            stage.physical_provider_calls for _name, stage in ordered
        ),
        transport_retries=sum(
            stage.transport_retries for _name, stage in ordered
        ),
        stage_metrics=stage_metrics,
    )


def run_intentir_lift(
    *,
    model: Model,
    workspace: IntentIRLiftWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Environment | None = None,
    candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
) -> IntentIRLiftRunResult:
    """Run a fresh DefaultAgent and return its sole candidate plus trajectory."""

    config = recursive_merge(
        load_intentir_lift_config(), dict(config_overrides or {})
    )
    if environment is not None and candidate_checker is not None:
        raise ValueError(
            "candidate_checker must be installed by the supplied environment"
        )
    environment = environment or IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        candidate_checker=candidate_checker,
        **config.get("environment", {}),
    )
    stage = _run_agent_stage(
        model=model,
        environment=environment,
        agent_config=config["agent"],
        output_reader=workspace.read_candidate,
    )
    result = _public_result(stage)
    if stage.caught is not None:
        raise IntentIRLiftRunError(
            "IntentIR Lift worker failed: "
            f"{type(stage.caught).__name__}: {stage.caught}",
            result=result,
            cause=stage.caught,
        ) from stage.caught
    if stage.exit_status not in {"Submitted", "SubmittedCheckpoint"} or stage.output_text is None:
        raise IntentIRLiftRunError(
            "IntentIR Lift worker terminated without a candidate: "
            f"{stage.exit_status}",
            result=result,
        )
    return result


def run_intentir_staged_lift(
    *,
    model: Model,
    workspace: IntentIRLiftWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
) -> IntentIRLiftRunResult:
    """Run three source-aware LLM stages; fixed code only transports exact bytes."""

    config = recursive_merge(
        load_intentir_lift_config(), dict(config_overrides or {})
    )
    staged = config.get("staged")
    if (
        not isinstance(staged, Mapping)
        or staged.get("protocol") != "intentir.lift-workflow.v1"
    ):
        raise ValueError("missing closed staged Lift workflow configuration")
    completed: dict[str, _IntentIRLiftStageResult] = {}
    plan_sha256: str | None = None

    def stage_config(name: str) -> dict[str, Any]:
        specific = staged.get(name)
        if not isinstance(specific, Mapping):
            raise ValueError(f"missing staged Lift configuration for {name!r}")
        return recursive_merge(
            {"agent": config["agent"], "environment": config["environment"]},
            dict(specific),
        )

    def run_stage(
        name: str,
        *,
        checker: Callable[[str], Mapping[str, Any]] | None,
        reader: Callable[[], str],
    ) -> _IntentIRLiftStageResult:
        selected = stage_config(name)
        environment = IntentIRLiftEnvironment(
            input_dir=workspace.input_dir,
            output_dir=workspace.output_dir,
            scratch_dir=workspace.scratch_dir,
            candidate_checker=checker,
            **selected["environment"],
        )
        result = _run_agent_stage(
            model=model,
            environment=environment,
            agent_config=selected["agent"],
            output_reader=reader,
        )
        completed[name] = result
        return result

    planner = run_stage(
        "planner", checker=None, reader=workspace.read_lift_plan
    )
    if planner.caught is not None or planner.output_text is None:
        result = _aggregate_staged_result(
            completed, plan_sha256=plan_sha256
        )
        raise IntentIRLiftRunError(
            "IntentIR staged Lift planner failed: "
            f"{planner.exit_status}",
            result=result,
            cause=planner.caught,
        ) from planner.caught
    plan_bytes = planner.output_text.encode("utf-8")
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan_input = workspace.input_dir / "knowledge" / "lift_plan.json"
    if plan_input.exists():
        raise ValueError("staged Lift plan input path already exists")
    plan_input.parent.mkdir(parents=True, exist_ok=True)
    plan_input.write_bytes(plan_bytes)
    workspace.reset_writable()

    builder = run_stage(
        "builder", checker=candidate_checker, reader=workspace.read_candidate
    )
    if builder.caught is not None or builder.output_text is None:
        result = _aggregate_staged_result(
            completed, plan_sha256=plan_sha256
        )
        raise IntentIRLiftRunError(
            "IntentIR staged Lift builder failed: "
            f"{builder.exit_status}",
            result=result,
            cause=builder.caught,
        ) from builder.caught
    workspace.reset_writable(preserve_candidate=True)

    reviewer = run_stage(
        "reviewer", checker=candidate_checker, reader=workspace.read_candidate
    )
    result = _aggregate_staged_result(completed, plan_sha256=plan_sha256)
    if reviewer.caught is not None:
        raise IntentIRLiftRunError(
            "IntentIR staged Lift reviewer failed: "
            f"{type(reviewer.caught).__name__}: {reviewer.caught}",
            result=result,
            cause=reviewer.caught,
        ) from reviewer.caught
    if reviewer.exit_status not in {
        "Submitted",
        "SubmittedCheckpoint",
    } or reviewer.output_text is None:
        raise IntentIRLiftRunError(
            "IntentIR staged Lift reviewer terminated without a candidate: "
            f"{reviewer.exit_status}",
            result=result,
        )
    return result


__all__ = [
    "create_intentir_lift_model",
    "IntentIRLiftRunError",
    "IntentIRLiftRunResult",
    "load_intentir_lift_config",
    "run_intentir_lift",
    "run_intentir_staged_lift",
]
