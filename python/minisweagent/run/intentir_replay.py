"""Compose a fresh mini-swe episode for one source-blind Replay generation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from minisweagent import Environment, Model, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.intentir_replay import (
    MAX_ACTION_OUTPUT_BYTES,
    MAX_REPLAY_BYTES,
    MAX_WORKSPACE_BYTES,
    MAX_WORKSPACE_DEPTH,
    MAX_WORKSPACE_ENTRIES,
    REPLAY_CHECK_PROTOCOL,
    REPLAY_BUSYBOX,
    REPLAY_BWRAP,
    REPLAY_INPUT_FILES,
    REPLAY_SANDBOX_HOSTNAME,
    REPLAY_SANDBOX_PROTOCOL,
    REPLAY_TOOL_APPLETS,
    IntentIRReplayEnvironment,
    IntentIRReplayWorkspace,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.utils.serialize import recursive_merge


_ALLOWED_OVERRIDE_KEYS = {
    "agent": {"step_limit", "wall_time_limit_seconds"},
    "environment": {
        "timeout",
        "replay_check_limit",
        "replay_submission_check_limit",
        "replay_compile_check_limit",
        "replay_source_inspect_limit",
    },
}
_OVERRIDE_BOUNDS = {
    ("agent", "step_limit"): (1, 64),
    ("agent", "wall_time_limit_seconds"): (1, 7200),
    ("environment", "timeout"): (1, 300),
    ("environment", "replay_check_limit"): (0, 32),
    ("environment", "replay_submission_check_limit"): (0, 4),
    ("environment", "replay_compile_check_limit"): (0, 16),
    ("environment", "replay_source_inspect_limit"): (0, 32),
}


@dataclass(frozen=True)
class IntentIRReplayRunResult:
    source_text: str | None
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


def validated_intentir_replay_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only bounded budget overrides to the fixed Replay config."""

    supplied = dict(overrides or {})
    unknown_sections = sorted(set(supplied) - set(_ALLOWED_OVERRIDE_KEYS))
    if unknown_sections:
        raise ValueError(
            "Replay worker overrides contain forbidden section(s): "
            + ", ".join(unknown_sections)
        )
    normalized: dict[str, dict[str, int]] = {}
    for section, raw_values in supplied.items():
        if not isinstance(raw_values, Mapping):
            raise TypeError(f"Replay worker override {section} must be a mapping")
        unknown_keys = sorted(
            set(raw_values) - _ALLOWED_OVERRIDE_KEYS[section]
        )
        if unknown_keys:
            raise ValueError(
                f"Replay worker override {section} has forbidden field(s): "
                + ", ".join(unknown_keys)
            )
        normalized[section] = {}
        for key, value in raw_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"Replay worker override {section}.{key} must be an integer"
                )
            lower, upper = _OVERRIDE_BOUNDS[(section, key)]
            if not lower <= value <= upper:
                raise ValueError(
                    f"Replay worker override {section}.{key} must be in "
                    f"[{lower}, {upper}]"
                )
            normalized[section][key] = value
    config = recursive_merge(load_intentir_replay_config(), normalized)
    if config.get("agent", {}).get("output_path") is not None:
        raise ValueError("Replay agent output_path must remain disabled")
    return config


def intentir_replay_episode_contract_sha256(
    files: Mapping[str, str | bytes],
    config: Mapping[str, Any],
) -> str:
    """Bind exact inputs, prompts, budgets, protocols, and isolation policy."""

    if set(files) != REPLAY_INPUT_FILES:
        raise ValueError("Replay episode digest requires the exact input allowlist")
    agent = config.get("agent")
    environment = config.get("environment")
    model = config.get("model")
    if (
        not isinstance(agent, Mapping)
        or not isinstance(environment, Mapping)
        or not isinstance(model, Mapping)
    ):
        raise TypeError("Replay episode digest requires a validated config")
    policy = {
        "protocol": "intentir.miniswe-replay-episode-contract.v1",
        "replay_check_protocol": REPLAY_CHECK_PROTOCOL,
        "sandbox_protocol": REPLAY_SANDBOX_PROTOCOL,
        "hostname": REPLAY_SANDBOX_HOSTNAME,
        "network": "new-unshared-namespace",
        "proc": "not-mounted",
        "mounted_busybox_entrypoints": list(REPLAY_TOOL_APPLETS),
        "busybox_sha256": hashlib.sha256(REPLAY_BUSYBOX.read_bytes()).hexdigest(),
        "bubblewrap_sha256": hashlib.sha256(REPLAY_BWRAP.read_bytes()).hexdigest(),
        "environment": {
            "HOME": "/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/bin",
            "TMPDIR": "/tmp",
        },
        "output": {
            "path": "/output/replay.source",
            "max_bytes": MAX_REPLAY_BYTES,
            "max_action_output_bytes": MAX_ACTION_OUTPUT_BYTES,
        },
        "workspace": {
            "max_bytes": MAX_WORKSPACE_BYTES,
            "max_entries": MAX_WORKSPACE_ENTRIES,
            "max_depth": MAX_WORKSPACE_DEPTH,
        },
        "system_template": agent.get("system_template"),
        "instance_template": agent.get("instance_template"),
        "budgets": {
            "step_limit": agent.get("step_limit"),
            "wall_time_limit_seconds": agent.get("wall_time_limit_seconds"),
            "action_timeout_seconds": environment.get("timeout"),
            "replay_check_limit": environment.get("replay_check_limit"),
            "replay_submission_check_limit": environment.get(
                "replay_submission_check_limit"
            ),
            "replay_compile_check_limit": environment.get(
                "replay_compile_check_limit"
            ),
            "replay_source_inspect_limit": environment.get(
                "replay_source_inspect_limit"
            ),
            "transport_attempts_per_turn": model.get("retry_attempts"),
            "max_tool_calls_per_response": model.get(
                "max_tool_calls_per_response"
            ),
            "parallel_tool_calls": (
                model.get("model_kwargs", {}).get("parallel_tool_calls")
                if isinstance(model.get("model_kwargs"), Mapping)
                else None
            ),
        },
    }
    hasher = hashlib.sha256()

    def frame(label: bytes, payload: bytes) -> None:
        hasher.update(len(label).to_bytes(8, "big"))
        hasher.update(label)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

    frame(
        b"policy.json",
        json.dumps(
            policy,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    for name in sorted(files):
        content = files[name]
        payload = content if isinstance(content, bytes) else content.encode("utf-8")
        frame(name.encode("utf-8"), payload)
    return hasher.hexdigest()


def run_intentir_replay(
    *,
    model: Model,
    workspace: IntentIRReplayWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Environment | None = None,
    replay_checker: Callable[[str], Mapping[str, Any]] | None = None,
    replay_compile_checker: Callable[[str], Mapping[str, Any]] | None = None,
) -> IntentIRReplayRunResult:
    """Run one new DefaultAgent; no messages or workspace survive this call."""

    config = validated_intentir_replay_config(config_overrides)
    if environment is not None and (
        replay_checker is not None or replay_compile_checker is not None
    ):
        raise ValueError(
            "Replay checkers must be installed by the supplied environment"
        )
    environment = environment or IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=replay_checker,
        replay_compile_checker=replay_compile_checker,
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
    exit_status = str(terminal.get("exit_status", ""))
    source_text: str | None = None
    if exit_status == "Submitted":
        try:
            source_text = workspace.read_replay()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    elif caught is None:
        activate_fallback = getattr(
            environment, "activate_committed_fallback", None
        )
        if callable(activate_fallback):
            try:
                source_text = activate_fallback(exit_status)
            except Exception as exc:
                caught = exc
                exit_status = "InvalidSubmission"
            else:
                if source_text is not None:
                    exit_status = "SubmittedCheckpoint"
    elapsed_ms = (time.monotonic() - started) * 1000.0
    trajectory = agent.save(None)
    result = IntentIRReplayRunResult(
        source_text=source_text,
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
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker failed: {type(caught).__name__}: {caught}",
            result=result,
            cause=caught,
        ) from caught
    if exit_status not in {"Submitted", "SubmittedCheckpoint"} or source_text is None:
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker terminated without source: {exit_status}",
            result=result,
        )
    return result


__all__ = [
    "IntentIRReplayRunError",
    "IntentIRReplayRunResult",
    "create_intentir_replay_model",
    "intentir_replay_episode_contract_sha256",
    "load_intentir_replay_config",
    "run_intentir_replay",
    "validated_intentir_replay_config",
]
