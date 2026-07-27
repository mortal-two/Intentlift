"""Fresh source-blind Replay episodes driven by the isolated Codex CLI runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .model import Frontend, Gate1Artifact
from .prompt import replay_packet_digest
from .replay import (
    REPLAY_CHECK_PROTOCOL,
    ReplayAttempt,
    ReplayGeneration,
    ReplayGenerationError,
    _context_receipt,
    _replay_input_files,
    _trajectory_bytes,
    replay_source_ingress,
    source_blind_replay_compile_check,
)


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v8"

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
The outer runtime exposes only this isolated /workspace. input/ is immutable;
work/, output/, and tmp/ are writable. There is no Source, source slice,
Harness/compiler view, actual C value, Gate result, earlier Replay, sibling
trajectory, or network tool. Do not search for any of them.

Start with exactly one call to
`python .agents/intentir_agent_tool.py replay_workspace_view`. Do not first dump
the whole input/packet.json: the tool is a lossless, compact projection of this
packet's complete anonymous ABI, J graph and boundary, MaskC(C), U_f, and a
per-instance pi obligation ledger joining authored routes, order, sharing,
temporal roles, refinements, and terminal control. It does not derive a
schedule or see Gate evidence. Use each J graph row and pi obligation row as a
construction checklist. Read a named field from input/packet.json only when the
workspace view points to an exact field that remains unclear; never re-dump the
whole packet or redundantly reconstruct routes/order already joined by the
ledger. Preserve every authored effect and realization ordering relation;
never replace pi topology with a guessed source recipe. Use REPLAY_CONTRACT.md
only for a focused unclear rule.
Author the entire Replay as
exactly output/replay.source and no other output file. You may keep
proof-irrelevant notes or helper scripts under work/. You own every semantic
choice in the Replay. Run
`python .agents/intentir_agent_tool.py replay_check` on your complete source,
repair it until PASS, then run
`python .agents/intentir_agent_tool.py replay_compile_check` for the public
synthetic construction profile. Fixed host code only checks exact bytes and
never writes or repairs Replay. Both checks must pass before this episode ends;
neither has Gate or certification authority.
Do not claim Gate1, Gate2, or certification.
"""

_REPAIR_PROMPT = """The same fresh source-blind Replay episode continues in the same thread and workspace.
The host rejected the current exact Replay bytes using only the packet and the
fixed source-blind ingress or public synthetic construction-compile contract.
Read work/host_replay_check.json and the current output/replay.source, then
repair the complete source yourself. The diagnostic is rejection-only and has
no Gate or certification authority. End only after the current complete
output/replay.source passes both replay_check and replay_compile_check.
"""


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = usage.get(key, 0) + value
    return usage


def _episode_contract_sha256(files: Mapping[str, str | bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(CODEX_CLI_REPLAY_BACKEND.encode())
    for name in sorted(files):
        content = files[name]
        raw = content if isinstance(content, bytes) else content.encode()
        digest.update(b"\0")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _replay_realization_view(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return an exact, source-isolated focus over authored realization facts."""

    c = packet.get("c")
    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(c, Mapping) or not isinstance(j, Mapping) or not isinstance(
        pi, Mapping
    ):
        raise ValueError("Replay packet lacks closed C/J/pi objects")
    boundary = j.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Replay packet lacks a closed J boundary")
    accesses = boundary.get("accesses")
    effects = boundary.get("effects")
    if not isinstance(accesses, list) or not isinstance(effects, list):
        raise ValueError("Replay packet J boundary lacks access/effect lists")
    return {
        "protocol": "intentir.scrc.replay-realization-view.v1",
        "status": "pass",
        "c": dict(c),
        "j_boundary": {
            "accesses": accesses,
            "effects": effects,
        },
        "pi": dict(pi),
        "authority": (
            "lossless source-isolated packet projection only; no schedule, code, "
            "repair, Source, Harness value, or Gate evidence"
        ),
    }


def _replay_workspace_view(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return one compact, lossless current-protocol Replay construction view.

    The view reorganizes only this packet's authored J/C/pi/U_f facts.  It joins
    pi edges around declared instances to avoid repeated model-side paging, but
    never invents an edge, operation, value, or realization choice.
    """

    c = packet.get("c")
    j = packet.get("j")
    pi = packet.get("pi")
    u_f = packet.get("u_f")
    if not all(isinstance(item, Mapping) for item in (c, j, pi, u_f)):
        raise ValueError("Replay packet lacks closed J/C/pi/U_f objects")
    intent = j.get("intent")
    signature = j.get("signature")
    boundary = j.get("boundary")
    if not all(
        isinstance(item, Mapping) for item in (intent, signature, boundary)
    ):
        raise ValueError("Replay packet lacks a closed J interface/intent/boundary")
    function = intent.get("intent_function")
    if not isinstance(function, Mapping):
        raise ValueError("Replay packet lacks a closed J intent function")

    from minisweagent.environments.intentir_replay import (
        _packet_graph,
        _packet_obligations,
    )

    graph: list[Any] = []
    start = 0
    while True:
        page = _packet_graph(packet, start, 48)
        graph.extend(page["items"])
        next_start = page["next_start"]
        if next_start is None:
            break
        start = next_start

    obligation_items: list[Any] = []
    obligation_columns: list[str] | None = None
    obligation_total = 0
    start = 0
    while True:
        page = _packet_obligations(packet, start, 48)
        obligation_columns = list(page["columns"])
        obligation_total = int(page["total"])
        obligation_items.extend(page["items"])
        next_start = page["next_start"]
        if next_start is None:
            break
        start = next_start

    return {
        "protocol": "intentir.scrc.replay-workspace-view.v1",
        "status": "pass",
        "abi": {
            "u_f": dict(u_f),
            "signature": dict(signature),
            "intent_io": {
                key: function[key]
                for key in ("name", "inputs", "outputs", "returns")
                if key in function
            },
            "mask_c": dict(c),
        },
        "j": {
            "kernel_id": j.get("kernel_id"),
            "node_ids": j.get("node_ids"),
            "graph": graph,
            "boundary": dict(boundary),
            "scopes": j.get("scopes"),
            "storage": j.get("storage"),
            "dimension_constraints": j.get("dimension_constraints"),
            "value_dimension_relations": j.get("value_dimension_relations"),
        },
        "pi": {
            "regions": pi.get("regions"),
            "terminal_controls": pi.get("terminal_controls"),
            "obligations": {
                "protocol": "intentir.replay-instance-obligations.v2",
                "total": obligation_total,
                "columns": obligation_columns or [],
                "items": obligation_items,
            },
        },
        "authority": (
            "lossless source-isolated packet projection only; joins authored "
            "facts but derives no schedule, code, repair, Source, Harness value, "
            "or Gate evidence"
        ),
    }


@dataclass
class CodexCLIReplayWriter:
    """Time-governed Codex writer; every ``generate`` is a fresh episode."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    model_timeout_seconds: float | None = None
    session_wall_time_limit_seconds: float = 900.0
    repair_effort: str | None = None
    compile_checker: Callable[[str, Gate1Artifact, str], Any] | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
    session_factory: Callable[..., Any] = field(
        default=CodexCLISession,
        repr=False,
    )
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        if self.session_wall_time_limit_seconds <= 0:
            raise ValueError("Replay episode wall-time limit must be positive")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        turn_limit = self.model_timeout_seconds
        if turn_limit is None:
            turn_limit = (
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
        episode_contract_sha256 = _episode_contract_sha256(files)

        try:
            from minisweagent.environments.intentir_replay import (
                IntentIRReplayWorkspace,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Codex CLI Replay requires the replay-agent workspace dependency"
            ) from exc

        workspace = IntentIRReplayWorkspace.create(files)
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        compile_checks: list[dict[str, Any]] = []
        source_text = ""
        formal_ingress = None
        formal_compile = None
        prompt = _INITIAL_PROMPT
        selected_effort = effective_effort
        active_elapsed_ms = 0.0
        thread_id: str | None = None
        session: Any | None = None

        def ingress_current() -> Any:
            current = workspace.read_replay()
            return replay_source_ingress(
                current,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        def replay_check_tool() -> Mapping[str, Any]:
            try:
                return ingress_current().to_check_response()
            except Exception as exc:
                return {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_unavailable",
                            "path": "<replay>",
                            "observed_refs_or_types": [
                                f"{type(exc).__name__}: {exc}"[:2048]
                            ],
                        }
                    ],
                }

        def replay_compile_tool() -> Mapping[str, Any]:
            ingress = ingress_current()
            if ingress.status != "pass" or ingress.code is None:
                return ingress.to_check_response()
            checked = (
                self.compile_checker(ingress.code, artifact, function_name)
                if self.compile_checker is not None
                else source_blind_replay_compile_check(
                    ingress.code,
                    artifact,
                    function_name,
                )
            )
            return checked.to_check_response()

        def replay_realization_view_tool() -> Mapping[str, Any]:
            return _replay_realization_view(packet)

        def replay_workspace_view_tool() -> Mapping[str, Any]:
            return _replay_workspace_view(packet)

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "replay_workspace_view": replay_workspace_view_tool,
                "replay_realization_view": replay_realization_view_tool,
                "replay_check": replay_check_tool,
                "replay_compile_check": replay_compile_tool,
            },
        )
        try:
            bridge.start()
            session = self.session_factory(
                provider=provider,
                input_dir=workspace.input_dir,
                output_dir=workspace.output_dir,
                scratch_dir=workspace.scratch_dir,
                tmp_dir=workspace.tmp_dir,
                model=provider.model,
                effort=effective_effort,
            )
            while True:
                remaining = (
                    self.session_wall_time_limit_seconds
                    - active_elapsed_ms / 1000.0
                )
                if remaining <= 0:
                    raise RuntimeError("Replay owner episode wall-time exhausted")
                turn = session.run(
                    prompt,
                    timeout_seconds=min(remaining, turn_limit),
                    effort=selected_effort,
                    stop_requested=self.stop_requested,
                )
                turns.append(turn)
                thread_id = turn.thread_id
                active_elapsed_ms += turn.elapsed_ms
                try:
                    source_text = workspace.read_replay()
                except Exception as exc:
                    check = {
                        "protocol": REPLAY_CHECK_PROTOCOL,
                        "status": "fail",
                        "diagnostics": [
                            {
                                "rule_code": "replay_unavailable",
                                "path": "<replay>",
                                "observed_refs_or_types": [
                                    f"{type(exc).__name__}: {exc}"[:2048]
                                ],
                            }
                        ],
                    }
                    formal_ingress = None
                else:
                    formal_ingress = replay_source_ingress(
                        source_text,
                        artifact=artifact,
                        frontend=self.frontend,
                        function_name=function_name,
                        packet=packet,
                    )
                    check = formal_ingress.to_check_response()
                checks.append(
                    {
                        "status": check["status"],
                        "diagnostics": [
                            dict(item) for item in check["diagnostics"]
                        ],
                        "source_sha256": (
                            hashlib.sha256(source_text.encode()).hexdigest()
                            if source_text
                            else None
                        ),
                    }
                )
                if (
                    formal_ingress is not None
                    and formal_ingress.status == "pass"
                    and formal_ingress.code is not None
                ):
                    formal_compile = (
                        self.compile_checker(
                            formal_ingress.code,
                            artifact,
                            function_name,
                        )
                        if self.compile_checker is not None
                        else source_blind_replay_compile_check(
                            formal_ingress.code,
                            artifact,
                            function_name,
                        )
                    )
                    compile_check = formal_compile.to_check_response()
                    compile_checks.append(
                        {
                            "status": compile_check["status"],
                            "diagnostics": [
                                dict(item)
                                for item in compile_check["diagnostics"]
                            ],
                            "source_sha256": hashlib.sha256(
                                source_text.encode()
                            ).hexdigest(),
                        }
                    )
                    if formal_compile.status == "pass":
                        break
                    check = compile_check
                (workspace.scratch_dir / "host_replay_check.json").write_text(
                    json.dumps(
                        check,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                prompt = _REPAIR_PROMPT
                selected_effort = self.repair_effort or effective_effort

            assert formal_ingress is not None and formal_ingress.code is not None
            assert formal_compile is not None and formal_compile.status == "pass"
            usage = _aggregate_usage(turns)
            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_CLI_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "tool_bridge": bridge.evidence,
                "formal_ingress_recheck": True,
                "isolation": {
                    "source_blind": True,
                    "input": "read-only",
                    "output_work_tmp": "write-only-scope",
                    "tool_network": "disabled",
                    "provider_secret_in_tool_env": False,
                },
            }
            raw_trajectory = _trajectory_bytes(trajectory)
            context_receipt = _context_receipt({}, packet["u_f"])
            context_receipt.update(
                {
                    "backend_protocol": CODEX_CLI_REPLAY_BACKEND,
                    "fresh_session": True,
                    "thread_id": thread_id,
                    "cli_turns": len(turns),
                    "model_segments_lower_bound": sum(
                        turn.model_segments for turn in turns
                    ),
                    "tool_actions": sum(turn.tool_actions for turn in turns),
                    "packet_digest": packet_digest,
                    "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                    "formal_compile_recheck": True,
                    "episode_contract_sha256": episode_contract_sha256,
                }
            )
            generation = ReplayGeneration(
                code=formal_ingress.code,
                function_name=function_name,
                model=provider.model,
                response_id=thread_id,
                packet_digest=packet_digest,
                prompt_digest=prompt_digest,
                context_receipt=context_receipt,
                usage=usage,
                response_model=provider.model,
                entry_function_name=formal_ingress.entry_function_name,
                request_sha256=prompt_digest,
                response_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                response_text=source_text,
                raw_response=raw_trajectory,
                attempts=len(turns),
                physical_provider_calls=sum(turn.model_segments for turn in turns),
                transport_retries=0,
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(generation)
            return generation
        except Exception as exc:
            cancelled_calls = 0
            if isinstance(exc, CodexCLICancelled):
                active_elapsed_ms += exc.elapsed_ms
                cancelled_calls = exc.physical_calls
            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_CLI_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "tool_bridge": bridge.evidence,
                "failure_class": type(exc).__name__,
                **(
                    {
                        "winner_branch_id": exc.winner_branch_id,
                        "cancelled_elapsed_ms": exc.elapsed_ms,
                        "cancelled_stdout_sha256": exc.stdout_sha256,
                        "cancelled_stderr_sha256": exc.stderr_sha256,
                    }
                    if isinstance(exc, CodexCLICancelled)
                    else {}
                ),
            }
            raw_trajectory = _trajectory_bytes(trajectory)
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=source_text,
                response_id=thread_id,
                response_model=provider.model,
                usage=_aggregate_usage(turns),
                request_sha256=prompt_digest,
                response_sha256=(
                    hashlib.sha256(source_text.encode()).hexdigest()
                    if source_text
                    else ""
                ),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=len(turns),
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
                physical_provider_calls=sum(
                    turn.model_segments for turn in turns
                ) + cancelled_calls,
                transport_retries=0,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        finally:
            bridge.close()
            if session is not None:
                session.close()
            workspace.cleanup()


__all__ = ["CODEX_CLI_REPLAY_BACKEND", "CodexCLIReplayWriter"]
