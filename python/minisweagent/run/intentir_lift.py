"""Compose the in-repository mini-swe components for one IntentIR Lift round."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    agent = DefaultAgent(model, environment, **config["agent"])
    started = time.monotonic()
    caught: Exception | None = None
    try:
        terminal = agent.run()
    except Exception as exc:
        caught = exc
        terminal = {"exit_status": type(exc).__name__, "submission": ""}
    elapsed_ms = (time.monotonic() - started) * 1000.0
    trajectory = agent.save(None)
    exit_status = str(terminal.get("exit_status", ""))
    candidate_text: str | None = None
    if exit_status == "Submitted":
        try:
            candidate_text = workspace.read_candidate()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    result = IntentIRLiftRunResult(
        candidate_text=candidate_text,
        trajectory=trajectory,
        exit_status=exit_status,
        model_calls=agent.n_calls,
        elapsed_ms=elapsed_ms,
        physical_provider_calls=getattr(
            model, "physical_provider_calls", agent.n_calls
        ),
        transport_retries=getattr(model, "transport_retries", 0),
    )
    if caught is not None:
        raise IntentIRLiftRunError(
            f"IntentIR Lift worker failed: {type(caught).__name__}: {caught}",
            result=result,
            cause=caught,
        ) from caught
    if exit_status != "Submitted" or candidate_text is None:
        raise IntentIRLiftRunError(
            f"IntentIR Lift worker terminated without a candidate: {exit_status}",
            result=result,
        )
    return result


__all__ = [
    "create_intentir_lift_model",
    "IntentIRLiftRunError",
    "IntentIRLiftRunResult",
    "load_intentir_lift_config",
    "run_intentir_lift",
]
