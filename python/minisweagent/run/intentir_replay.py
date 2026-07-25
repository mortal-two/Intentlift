"""Compose a fresh mini-swe episode for one source-blind Replay generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from minisweagent import Environment, Model, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.intentir_replay import (
    IntentIRReplayEnvironment,
    IntentIRReplayWorkspace,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.utils.serialize import recursive_merge


@dataclass(frozen=True)
class IntentIRReplayRunResult:
    source_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float


class IntentIRReplayRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: IntentIRReplayRunResult,
        cause: BaseException | None = None,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(message)


def load_intentir_replay_config() -> dict[str, Any]:
    path = Path(package_dir) / "config" / "intentir_replay.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("IntentIR Replay config must be a mapping")
    return value


def create_intentir_replay_model(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    reasoning_effort: str,
    timeout_seconds: float,
) -> LitellmResponseModel:
    config = load_intentir_replay_config()["model"]
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


def run_intentir_replay(
    *,
    model: Model,
    workspace: IntentIRReplayWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Environment | None = None,
    replay_checker: Callable[[str], Mapping[str, Any]] | None = None,
) -> IntentIRReplayRunResult:
    """Run one new DefaultAgent; no messages or workspace survive this call."""

    config = recursive_merge(
        load_intentir_replay_config(), dict(config_overrides or {})
    )
    if environment is not None and replay_checker is not None:
        raise ValueError("replay_checker must be installed by the supplied environment")
    environment = environment or IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=replay_checker,
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
    source_text: str | None = None
    if exit_status == "Submitted":
        try:
            source_text = workspace.read_replay()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    result = IntentIRReplayRunResult(
        source_text=source_text,
        trajectory=trajectory,
        exit_status=exit_status,
        model_calls=agent.n_calls,
        elapsed_ms=elapsed_ms,
    )
    if caught is not None:
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker failed: {type(caught).__name__}: {caught}",
            result=result,
            cause=caught,
        ) from caught
    if exit_status != "Submitted" or source_text is None:
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker terminated without source: {exit_status}",
            result=result,
        )
    return result


__all__ = [
    "IntentIRReplayRunError",
    "IntentIRReplayRunResult",
    "create_intentir_replay_model",
    "load_intentir_replay_config",
    "run_intentir_replay",
]
