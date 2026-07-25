"""Source-blind Replay writers and the shared compiler-free ingress."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import extract_python_module
from .cuda_codegen import extract_cuda_module
from .cute_codegen import extract_cute_module
from .model import Frontend, Gate1Artifact
from .prompt import (
    assert_source_isolated_payload,
    build_public_frontend_context_receipt,
    build_rewriter_prompt,
    build_rewriter_payload,
    replay_packet_digest,
)
from .tilelang_codegen import extract_tilelang_module


MINISWE_REPLAY_BACKEND = "minisweagent.intentir-replay.v1"
RESPONSES_REPLAY_BACKEND = "openai.responses.intentir-replay.v1"
STATIC_REPLAY_BACKEND = "intentir.static-replay-test.v1"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v1"
_ANONYMOUS_HANDLE = re.compile(r"^[pc][0-9]+$")


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    function_name: str
    model: str
    response_id: str | None
    packet_digest: str
    prompt_digest: str
    context_receipt: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    response_model: str | None = None
    entry_function_name: str | None = None
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    episode_contract_sha256: str = ""


@dataclass(frozen=True)
class ReplayAttempt:
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    episode_contract_sha256: str = ""


class ReplayGenerationError(RuntimeError):
    def __init__(self, attempt: ReplayAttempt) -> None:
        self.attempt = attempt
        super().__init__(f"Replay response rejected: {attempt.diagnostic}")


class ReplayWriter(Protocol):
    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration: ...


class RecordedReplayWriter(ReplayWriter, Protocol):
    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]: ...


@dataclass(frozen=True)
class ReplayIngressResult:
    """Result of fixed, compiler-free Replay ingress validation."""

    status: str
    code: str | None = field(default=None, repr=False)
    entry_function_name: str | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def to_check_response(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": self.status,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


class _ReplayABIViolation(ValueError):
    def __init__(
        self,
        rule_code: str,
        *,
        path: str,
        handles: tuple[str, ...] = (),
        line: int | None = None,
    ) -> None:
        self.rule_code = rule_code
        self.path = path
        self.handles = handles
        self.line = line
        super().__init__(rule_code)


def _check_triton_exact_abi(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> None:
    """Check the exact packet-derived ABI where Triton syntax is unambiguous."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    j_parameters = packet["j"]["signature"]["parameters"]
    c_slots = packet["c"]["slots"]
    expected_names = [
        *(str(item["name"]) for item in j_parameters),
        *(str(item["name"]) for item in c_slots),
    ]
    actual_names = [item.arg for item in function.args.args]
    if len(actual_names) != len(expected_names):
        raise _ReplayABIViolation(
            "abi_parameter_count",
            path="<entry-signature>",
            handles=tuple(expected_names[:32]),
            line=function.lineno,
        )
    if actual_names != expected_names:
        visible = tuple(
            dict.fromkeys(
                item
                for item in [*expected_names, *actual_names]
                if _ANONYMOUS_HANDLE.fullmatch(item)
            )
        )[:32]
        raise _ReplayABIViolation(
            "abi_parameter_order",
            path="<entry-signature>",
            handles=visible,
            line=function.lineno,
        )
    j_constexpr = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
    }
    expected_constexpr = j_constexpr | {str(item["name"]) for item in c_slots}
    for argument in function.args.args:
        is_constexpr = (
            isinstance(argument.annotation, ast.Attribute)
            and isinstance(argument.annotation.value, ast.Name)
            and argument.annotation.value.id == "tl"
            and argument.annotation.attr == "constexpr"
        )
        if argument.arg in expected_constexpr and not is_constexpr:
            raise _ReplayABIViolation(
                "constexpr_annotation_required",
                path=f"<parameter:{argument.arg}>",
                handles=(argument.arg,),
                line=argument.lineno,
            )
        if argument.arg not in expected_constexpr and is_constexpr:
            raise _ReplayABIViolation(
                "constexpr_annotation_forbidden",
                path=f"<parameter:{argument.arg}>",
                handles=(argument.arg,),
                line=argument.lineno,
            )


def _closed_ingress_diagnostic(exc: Exception) -> dict[str, Any]:
    """Project an exception to bounded source-free rule/path metadata."""

    if isinstance(exc, _ReplayABIViolation):
        return {
            "rule_code": exc.rule_code,
            "path": exc.path,
            "anonymous_handles": list(exc.handles),
            "line": exc.line,
        }
    line_match = re.search(r"\bline\s+(\d+)\b", str(exc))
    line = int(line_match.group(1)) if line_match is not None else None
    return {
        "rule_code": "replay_static_policy_rejected",
        "path": f"<line:{line}>" if line is not None else "<replay>",
        "anonymous_handles": [],
        "line": line,
    }


def _extract_replay_code(
    response: str,
    *,
    frontend: Frontend,
    function_name: str,
) -> tuple[str, str | None]:
    if frontend is Frontend.TRITON:
        return (
            extract_python_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUTE:
        entry = "scrc_entry"
        return (
            extract_cute_module(
                response,
                required_kernel=function_name,
                required_entry=entry,
            ),
            entry,
        )
    if frontend is Frontend.TILELANG:
        return (
            extract_tilelang_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUDA:
        return (
            extract_cuda_module(
                response,
                required_kernel=function_name,
            ),
            None,
        )
    raise ValueError(
        f"source-blind Replay has no audited validator for {frontend.value!r}"
    )


def replay_source_ingress(
    source: str,
    *,
    artifact: Gate1Artifact,
    frontend: Frontend | str,
    function_name: str = "scrc_k0",
    packet: Mapping[str, Any] | None = None,
) -> ReplayIngressResult:
    """Run the one host-owned static ingress used by check and submission.

    This path deliberately does not invoke a frontend compiler, Gate1 observer,
    execution harness, or any source-side view.
    """

    try:
        selected_frontend = Frontend(frontend)
    except (TypeError, ValueError):
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "frontend_not_supported",
                    "path": "<frontend>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    if selected_frontend is not Frontend.TRITON:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "frontend_not_supported",
                    "path": "<frontend>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    if packet is None:
        packet = build_rewriter_payload(
            artifact,
            frontend=selected_frontend,
            masked=False,
        )
    assert_source_isolated_payload(packet)
    try:
        code, entry = _extract_replay_code(
            source,
            artifact=artifact,
            frontend=selected_frontend,
            function_name=function_name,
        )
        _check_triton_exact_abi(code, packet, function_name)
    except Exception as exc:
        return ReplayIngressResult(
            "fail", diagnostics=(_closed_ingress_diagnostic(exc),)
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=entry,
    )


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    value = raw.get("usage")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _context_receipt(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    trace: list[dict[str, str]] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and "search" in item_type:
                trace.append(
                    {
                        "type": item_type,
                        "id": item.get("id") if isinstance(item.get("id"), str) else "",
                    }
                )
    receipt = build_public_frontend_context_receipt(context)
    return {
        "replay_input_protocol": str(context.get("protocol", "")),
        **receipt,
        "retrieval_trace": trace,
    }


def _agent_usage_and_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = False
    response_id: str | None = None
    response_model: str | None = None
    messages = trajectory.get("messages", ())
    if not isinstance(messages, list):
        return {}, None, None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("object") != "response":
            continue
        if isinstance(message.get("id"), str):
            response_id = message["id"]
        if isinstance(message.get("model"), str):
            response_model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        seen = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen else {}), response_id, response_model


def _sanitize_agent_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
    """Apply the same recursive secret-key and literal redaction as Lift."""

    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    } or normalized_key.endswith("_api_key"):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_agent_trajectory(
                item, secret, key=str(item_key)
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replay_input_files(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend,
    masked: bool,
) -> tuple[dict[str, str], dict[str, Any], str]:
    """Build the complete, closed allowlist mounted for one agent episode."""

    instructions, user = build_rewriter_prompt(
        artifact, frontend=frontend, masked=masked
    )
    packet = json.loads(user)
    assert_source_isolated_payload(packet)
    packet_text = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    readme = """# IntentIR source-blind Replay input

This is the complete immutable input allowlist for one fresh Replay episode.

- `packet.json`: exactly source-isolated J + pi + MaskC(C) + U_f.
- `REPLAY_CONTRACT.md`: fixed public Replay construction contract.
- `TOOL.md`: fixed bounded check and submission interface.

No Source or source slice, Harness/compiler view, actual C values, CompileCase,
grid, Gate diff/TTIR, previous Replay, sibling trajectory, or network is
available. Generate the entire source yourself. Fixed code only checks and
rejects; it never generates, patches, or certifies source.
"""
    tool = """# Replay tool contract

Write exactly `/output/replay.source`. To request bounded static diagnostics,
run exactly `intentir_replay_check /output/replay.source` as the complete shell
action. The seven ordinary evaluations share one budget; submission has one
reserved evaluation when exact current bytes have no prior PASS. Results contain
only a closed protocol/status/rule/path plus anonymous handles or line number.
No compiler, Gate1 observer, execution, or source oracle is run. A PASS has no
acceptance authority. Submit with a separate `printf 'INTENTIR_REPLAY_SUBMIT\\n'`.
The writer independently reruns the same ingress callable after submission.
"""
    return (
        {
            "README.md": readme,
            "packet.json": packet_text,
            "REPLAY_CONTRACT.md": instructions + "\n",
            "TOOL.md": tool,
        },
        packet,
        instructions,
    )


@dataclass
class ResponsesReplayWriter:
    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        if frontend not in {
            Frontend.TRITON,
            Frontend.CUTE,
            Frontend.TILELANG,
            Frontend.CUDA,
        }:
            raise ValueError(
                f"source-blind Replay has no audited validator for {frontend.value!r}"
            )
        reply = call_responses(
            instructions,
            user,
            provider=self.provider,
            effort=self.effort,
        )
        prompt_digest = hashlib.sha256(
            (instructions + "\0" + user).encode()
        ).hexdigest()
        try:
            code, entry_function_name = _extract_replay_code(
                reply.text,
                frontend=frontend,
                function_name=function_name,
            )
        except Exception as exc:
            attempt = ReplayAttempt(
                model=reply.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=reply.text,
                response_id=(
                    reply.raw.get("id")
                    if isinstance(reply.raw.get("id"), str)
                    else None
                ),
                response_model=(
                    reply.raw.get("model")
                    if isinstance(reply.raw.get("model"), str)
                    else None
                ),
                usage=_usage(reply.raw),
                request_sha256=reply.request_sha256,
                response_sha256=(
                    reply.response_sha256
                    or hashlib.sha256(reply.text.encode()).hexdigest()
                ),
                raw_response_sha256=reply.raw_response_sha256,
                raw_response=reply.raw_response,
                attempts=reply.attempts,
                elapsed_ms=reply.elapsed_ms,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        generation = ReplayGeneration(
            code=code,
            function_name=function_name,
            model=reply.model,
            response_id=(
                reply.raw.get("id")
                if isinstance(reply.raw.get("id"), str) else None
            ),
            packet_digest=replay_packet_digest(packet),
            prompt_digest=prompt_digest,
            context_receipt=_context_receipt(
                reply.raw, packet.get("u_f", {})
            ),
            usage=_usage(reply.raw),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            entry_function_name=entry_function_name,
            request_sha256=reply.request_sha256,
            response_sha256=(
                reply.response_sha256
                or hashlib.sha256(reply.text.encode()).hexdigest()
            ),
            raw_response_sha256=reply.raw_response_sha256,
            response_text=reply.text,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            elapsed_ms=reply.elapsed_ms,
        )
        self._records.append(generation)
        return generation


@dataclass
class MiniSWEReplayWriter:
    """Fresh bounded multi-turn Replay agent; v1 is Triton-only."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        if self.frontend is not Frontend.TRITON:
            raise ValueError(
                "MiniSWE Replay v1 fails closed for non-Triton frontends"
            )
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        if timeout is None:
            timeout = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        files, packet, instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        packet_digest = replay_packet_digest(packet)
        function_name = str(packet["u_f"]["replay_function"])
        prompt_digest = hashlib.sha256(
            (
                instructions
                + "\0"
                + json.dumps(
                    packet,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ).encode()
        ).hexdigest()
        try:
            from minisweagent.environments.intentir_replay import (
                IntentIRReplayWorkspace,
            )
            from minisweagent.run.intentir_replay import (
                IntentIRReplayRunError,
                create_intentir_replay_model,
                intentir_replay_episode_contract_sha256,
                run_intentir_replay,
                validated_intentir_replay_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MiniSWE Replay backend requires the lift-agent dependencies"
            ) from exc

        effective_worker_config = validated_intentir_replay_config(
            self.worker_config
        )
        episode_contract_sha256 = intentir_replay_episode_contract_sha256(
            files,
            effective_worker_config,
        )

        # This exact callable is used by every virtual check and once again by
        # the writer after submission. It has no compiler or Source dependency.
        def ingress(source_text: str) -> ReplayIngressResult:
            return replay_source_ingress(
                source_text,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        workspace = IntentIRReplayWorkspace.create(files)
        result = None
        formal_ingress: ReplayIngressResult | None = None
        try:
            model = create_intentir_replay_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
            )
            result = run_intentir_replay(
                model=model,
                workspace=workspace,
                config_overrides=self.worker_config,
                replay_checker=lambda source_text: ingress(
                    source_text
                ).to_check_response(),
            )
            assert result.source_text is not None
            formal_ingress = ingress(result.source_text)
        except IntentIRReplayRunError as exc:
            result = exc.result
            safe_trajectory = _sanitize_agent_trajectory(
                result.trajectory, provider.api_key
            )
            raw_trajectory = _trajectory_bytes(safe_trajectory)
            usage, response_id, response_model = _agent_usage_and_identity(
                safe_trajectory
            )
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay agent terminated without a formal submission: "
                    + result.exit_status
                ),
                response_text=result.source_text or "",
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                request_sha256=prompt_digest,
                response_sha256=(
                    hashlib.sha256((result.source_text or "").encode()).hexdigest()
                    if result.source_text is not None
                    else ""
                ),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        except Exception as exc:
            infrastructure_trajectory = {
                "info": {
                    "backend": MINISWE_REPLAY_BACKEND,
                    "fresh_session": True,
                    "exit_status": "InfrastructureError",
                    "exception_type": type(exc).__name__,
                }
            }
            raw_trajectory = _trajectory_bytes(infrastructure_trajectory)
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay infrastructure failure: " + type(exc).__name__
                ),
                response_text=(
                    result.source_text
                    if result is not None and result.source_text is not None
                    else ""
                ),
                request_sha256=prompt_digest,
                response_sha256=(
                    hashlib.sha256(result.source_text.encode()).hexdigest()
                    if result is not None and result.source_text is not None
                    else ""
                ),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls if result is not None else 0,
                elapsed_ms=result.elapsed_ms if result is not None else 0.0,
                trajectory=infrastructure_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise
        finally:
            workspace.cleanup()

        assert result is not None and result.source_text is not None
        assert formal_ingress is not None
        safe_trajectory = _sanitize_agent_trajectory(
            result.trajectory, provider.api_key
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        usage, response_id, response_model = _agent_usage_and_identity(
            safe_trajectory
        )
        if formal_ingress.status != "pass" or formal_ingress.code is None:
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=json.dumps(
                    formal_ingress.to_check_response(), sort_keys=True
                ),
                response_text=result.source_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                request_sha256=prompt_digest,
                response_sha256=hashlib.sha256(
                    result.source_text.encode()
                ).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt)
        info = safe_trajectory.get("info", {})
        replay_check = (
            info.get("replay_check", {}) if isinstance(info, Mapping) else {}
        )
        context_receipt = _context_receipt({}, packet["u_f"])
        context_receipt.update(
            {
                "backend_protocol": MINISWE_REPLAY_BACKEND,
                "fresh_session": True,
                "model_calls": result.model_calls,
                "physical_provider_calls": result.model_calls,
                "transport_retries": 0,
                "packet_digest": packet_digest,
                "replay_check": (
                    dict(replay_check)
                    if isinstance(replay_check, Mapping)
                    else {}
                ),
                "formal_ingress_recheck": True,
                "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                "episode_contract_sha256": episode_contract_sha256,
            }
        )
        generation = ReplayGeneration(
            code=formal_ingress.code,
            function_name=function_name,
            model=provider.model,
            response_id=response_id,
            packet_digest=packet_digest,
            prompt_digest=prompt_digest,
            context_receipt=context_receipt,
            usage=usage,
            response_model=response_model,
            entry_function_name=formal_ingress.entry_function_name,
            request_sha256=prompt_digest,
            response_sha256=hashlib.sha256(
                result.source_text.encode()
            ).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=result.source_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            elapsed_ms=result.elapsed_ms,
            trajectory=safe_trajectory,
            backend_protocol=MINISWE_REPLAY_BACKEND,
            episode_contract_sha256=episode_contract_sha256,
        )
        self._records.append(generation)
        return generation


@dataclass
class StaticReplayWriter:
    code: str
    model: str = "static-test-writer"
    frontend: Frontend = Frontend.TRITON
    calls: list[bool] = field(default_factory=list, init=False)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(
            artifact, frontend=self.frontend, masked=masked
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        code, entry_function_name = _extract_replay_code(
            self.code,
            frontend=frontend,
            function_name=function_name,
        )
        self.calls.append(masked)
        return ReplayGeneration(
            code=code,
            function_name=function_name,
            model=self.model,
            response_id=None,
            entry_function_name=entry_function_name,
            packet_digest=replay_packet_digest(packet),
            prompt_digest=hashlib.sha256(
                (instructions + "\0" + user).encode()
            ).hexdigest(),
            context_receipt=_context_receipt({}, packet.get("u_f", {})),
            backend_protocol=STATIC_REPLAY_BACKEND,
        )


__all__ = [
    "MINISWE_REPLAY_BACKEND", "RESPONSES_REPLAY_BACKEND",
    "STATIC_REPLAY_BACKEND", "MiniSWEReplayWriter", "RecordedReplayWriter",
    "ReplayAttempt", "ReplayGeneration", "ReplayGenerationError",
    "ReplayIngressResult", "ReplayWriter", "ResponsesReplayWriter",
    "StaticReplayWriter", "replay_source_ingress",
]
