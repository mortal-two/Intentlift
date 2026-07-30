"""Fresh source-blind Replay episodes driven by the isolated Codex CLI runtime."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIError,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.ttir import fingerprint_ttir

from .model import Frontend, Gate1Artifact
from .replay_composer import ComposerFragment, render_route_composer
from .replay import (
    REPLAY_CHECK_PROTOCOL,
    ReplayAttempt,
    ReplayGeneration,
    ReplayGenerationError,
    _context_receipt,
    build_source_blind_replay_compile_case,
    _replay_input_files,
    _trajectory_bytes,
    replay_source_ingress,
    source_blind_replay_compile_check,
)


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v51"
REPLAY_COMPOSER_PROTOCOL = "intentir.replay-episode-composer.v2"
REPLAY_PI_WIRE_PROTOCOL = "intentir.pi-min.relational-column-wire.v1"
_COMPOSER_FRAGMENT = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}\.source")
_MAX_COMPOSER_FRAGMENTS = 128
_MAX_REPLAY_BYTES = 1024 * 1024

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
Work only under the isolated /workspace. input/ is immutable; output/, work/, and
tmp/ are writable. There is no Source, Harness/compiler view, actual C value, Gate
result, earlier Replay, sibling state, repository, or network.

J is the only semantic authority; pi supplies realization constraints only. Read
input/packet.json by file path and inspect only the sections you need. Begin with
`python .agents/intentir_agent_tool.py packet_query requirements`; it is an exact,
bounded declaration-order view of J nodes and pi control/time relations, not a plan
or lowering rule. Preserve every listed J node's operands, attrs and result before
choosing target syntax. Do not infer a
Source recipe or invent a missing fact. input/REPLAY_CONTRACT.md defines the compact
ownership boundary; input/TOOL.md defines the exact tools.

Act like a coding agent. The primary path is to author one normal complete source
file and repair it locally after checks. Use the optional route-bound fragment
composer in input/TOOL.md only when it genuinely makes a cross-reference-heavy edit
easier; do not split a small kernel into one file per occurrence merely to satisfy a
host convention. The composer never chooses semantics.

After a meaningful complete draft, run
`python .agents/intentir_agent_tool.py submit_replay`. It transactionally runs the
source-blind ingress and synthetic compile checks on your exact bytes. Repair your
own source until it passes. If compilation is unclear, use `replay_ttir_inspect`
only on your current Replay. Fixed code only projects packet facts, checks, and
rejects; composer_render only renders your exact syntax templates through explicit
pi route references and never chooses or repairs semantics. Finish only with the
transaction passing
and do not claim Gate1, Gate2, or certification.
"""

_REPAIR_PROMPT = """The same fresh source-blind Replay episode continues in the same thread and workspace.
Read work/host_replay_check.json and the current output/replay.source, repair the
complete source or its composer fragments yourself, render again when fragments
changed, and rerun submit_replay. Diagnostics reject only; they do not supply a
semantic mapping or repair. Finish only when ingress and compile both pass.
"""


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(text.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = usage.get(key, 0) + value
    return usage


def _composer_fragments(composer_dir: Path) -> list[dict[str, Any]]:
    """Read an episode-local, syntax-only source-fragment workspace.

    Fragment names and bytes are entirely model-authored.  The host only checks a
    closed filesystem shape and concatenates exact UTF-8 bytes in lexical order.
    """

    if composer_dir.is_symlink() or not composer_dir.is_dir():
        raise ValueError("work/composer must be a real directory")
    paths = sorted(composer_dir.iterdir(), key=lambda path: path.name)
    if not 1 <= len(paths) <= _MAX_COMPOSER_FRAGMENTS:
        raise ValueError(
            "composer needs between 1 and "
            f"{_MAX_COMPOSER_FRAGMENTS} source fragments"
        )
    result: list[dict[str, Any]] = []
    total = 0
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or _COMPOSER_FRAGMENT.fullmatch(path.name) is None
        ):
            raise ValueError(
                "composer entries must be regular NAME.source fragments"
            )
        raw = path.read_bytes()
        if not raw or not raw.endswith(b"\n") or b"\x00" in raw:
            raise ValueError(
                f"composer fragment {path.name} must be non-empty UTF-8 text "
                "ending in one newline"
            )
        text = raw.decode("utf-8")
        total += len(raw)
        if total > _MAX_REPLAY_BYTES:
            raise ValueError("composed Replay exceeds the public source-size bound")
        result.append({"name": path.name, "bytes": len(raw), "text": text})
    return result


def _composer_response(
    composer_dir: Path,
    *,
    manifest_path: Path | None = None,
    pi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fragments = _composer_fragments(composer_dir)
    if manifest_path is not None and manifest_path.exists():
        if pi is None:
            raise ValueError("route-bound composer requires canonical pi")
        rendered = render_route_composer(
            tuple(
                ComposerFragment(
                    str(item["name"]),
                    str(item["text"]),
                    int(item["bytes"]),
                )
                for item in fragments
            ),
            manifest_path=manifest_path,
            pi=pi,
        )
        return rendered.to_response()
    return {
        "protocol": REPLAY_COMPOSER_PROTOCOL,
        "status": "pass",
        "fragments": [
            {"name": item["name"], "bytes": item["bytes"]}
            for item in fragments
        ],
        "total_bytes": sum(int(item["bytes"]) for item in fragments),
        "authority": (
            "episode-local syntax bytes only; no packet read, semantic mapping, "
            "lowering rule, repair, Gate, or certification"
        ),
    }


_PI_ROW_COLUMNS: dict[str, tuple[str, ...]] = {
    "regions": ("id", "parent", "placement"),
    "instances": ("id", "semantic_ref", "mechanism", "region"),
    "routes": ("source", "target", "slots"),
    "order": ("before", "after"),
    "sharing": ("source", "consumers"),
    "refinements": ("instance", "kind", "auth_ref"),
    "program_axes": ("domain_ref", "axis"),
    "program_linearizations": ("domain_ref", "dimension_refs"),
    "program_partitions": (
        "domain_ref",
        "target_dimension_ref",
        "block_dimension_ref",
    ),
}


def _pi_min_agent_wire(pi: Mapping[str, Any]) -> dict[str, Any]:
    """Encode exact pi relations without repeated JSON object keys.

    This is a bijective wire codec, not a semantic projection.  Lists not covered by
    the fixed row columns remain byte-for-value JSON relations.  A round-trip check
    below fails closed whenever the canonical pi schema evolves.
    """

    wire: dict[str, Any] = {
        "protocol": REPLAY_PI_WIRE_PROTOCOL,
        "columns": {name: list(columns) for name, columns in _PI_ROW_COLUMNS.items()},
    }
    for name, value in pi.items():
        if name not in _PI_ROW_COLUMNS:
            wire[name] = value
            continue
        if not isinstance(value, list):
            raise TypeError(f"pi.{name} must be a relation list")
        columns = _PI_ROW_COLUMNS[name]
        rows: list[list[Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping) or set(item) != set(columns):
                raise ValueError(
                    f"pi.{name}[{index}] does not match the closed relational columns"
                )
            rows.append([item[column] for column in columns])
        wire[name] = rows

    expanded: dict[str, Any] = {}
    for name, value in wire.items():
        if name in {"protocol", "columns"}:
            continue
        columns = _PI_ROW_COLUMNS.get(name)
        if columns is None:
            expanded[name] = value
            continue
        expanded[name] = [
            {column: row[index] for index, column in enumerate(columns)}
            for row in value
        ]
    if expanded != dict(pi):
        raise RuntimeError("pi-min relational wire failed exact round-trip")
    return wire


def _agent_replay_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact Replay packet with only pi's JSON wire made compact."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping):
        raise TypeError("Replay packet pi must be an object")
    result = dict(packet)
    result["pi"] = _pi_min_agent_wire(pi)
    return result



def _replay_startup_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return a path-only receipt; Codex reads the exact packet as a local file."""

    encoded = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "protocol": "intentir.scrc.replay-startup-view.v2",
        "packet_bytes": len(encoded),
        "paths": {
            "packet": "input/packet.json",
            "public_contract": "input/REPLAY_CONTRACT.md",
            "tool_contract": "input/TOOL.md",
            "composer": "work/composer/*.source",
            "composer_manifest": "work/replay.compose.json",
            "replay": "output/replay.source",
            "host_check": "work/host_replay_check.json",
        },
        "authority": (
            "path and byte-count receipt only; exact source-isolated packet bytes "
            "remain in input/packet.json and no code or schedule is generated"
        ),
    }

@dataclass
class CodexCLIReplayWriter:
    """Time-governed Codex writer; every ``generate`` is a fresh episode."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    model_timeout_seconds: float | None = None
    # Time, not a semantic-attempt count, bounds one fresh Replay episode.  High
    # gets three full provider windows; xhigh gets two of its longer windows.
    # This prevents a bad direction from cycling for an hour while leaving every
    # useful coding-agent tool action inside those turns unrestricted.
    session_wall_time_limit_seconds: float | None = None
    # Codex owns its thread compaction.  Zero disables host-side thread rollover;
    # cumulative API input-token counters are not a measure of live context size.
    context_rollover_input_tokens: int = 0
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
        if self.context_rollover_input_tokens < 0:
            raise ValueError("Replay context rollover token threshold cannot be negative")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        episode_limit = self.session_wall_time_limit_seconds
        if episode_limit is None:
            episode_limit = (
                2.0 * XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else 3.0 * DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        if episode_limit <= 0:
            raise ValueError("Replay episode wall-time limit must be positive")
        turn_limit = self.model_timeout_seconds
        if turn_limit is None:
            turn_limit = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        files, packet, _instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        agent_packet = _agent_replay_packet(packet)
        files["packet.json"] = json.dumps(
            agent_packet,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
        function_name = str(packet["u_f"]["replay_function"])

        try:
            from minisweagent.environments.intentir_replay import (
                INTENTIR_PACKET_QUERY,
                IntentIRReplayWorkspace,
                _query_packet,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Codex CLI Replay requires the replay-agent workspace dependency"
            ) from exc

        workspace = IntentIRReplayWorkspace.create(files)
        composer_dir = workspace.scratch_dir / "composer"
        composer_dir.mkdir()
        composer_manifest = workspace.scratch_dir / "replay.compose.json"
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        compile_checks: list[dict[str, Any]] = []
        ttir_inspections: list[dict[str, Any]] = []
        packet_queries: list[dict[str, Any]] = []
        composer_events: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        source_text = ""
        formal_ingress = None
        formal_compile = None
        last_checked_good: str | None = None
        startup_context = _replay_startup_context(agent_packet)
        synthetic_compiler = (
            None
            if self.compile_checker is not None
            else TritonCompiler(
                timeout=120.0,
                cache_dir=workspace.tmp_dir / "synthetic-replay-compile-cache",
            )
        )
        ttir_observations: dict[str, dict[str, Any]] = {}
        prompt = (
            _INITIAL_PROMPT
            + "\n\n<source-blind-startup>\n"
            + json.dumps(
                startup_context,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n</source-blind-startup>\n"
        )
        selected_effort = effective_effort
        active_elapsed_ms = 0.0
        timed_out_calls = 0
        thread_id: str | None = None
        thread_ids: list[str] = []
        session: Any | None = None
        last_replay_check_source: str | None = None
        last_replay_check_signature: tuple[tuple[str, str], ...] | None = None
        repeated_replay_diagnostic = 0
        explicit_submission_accepted = False

        def write_host_check(check: Mapping[str, Any]) -> None:
            _atomic_write_text(
                workspace.scratch_dir / "host_replay_check.json",
                json.dumps(
                    check,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
            )

        def new_session() -> Any:
            return self.session_factory(
                provider=provider,
                input_dir=workspace.input_dir,
                output_dir=workspace.output_dir,
                scratch_dir=workspace.scratch_dir,
                tmp_dir=workspace.tmp_dir,
                model=provider.model,
                effort=effective_effort,
            )

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
            nonlocal last_replay_check_source
            nonlocal last_replay_check_signature
            nonlocal repeated_replay_diagnostic
            try:
                current = workspace.read_replay()
                response = dict(ingress_current().to_check_response())
            except Exception:
                current = None
                response = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_unavailable",
                            "path": "<replay>",
                            "anonymous_handles": [],
                            "line": None,
                        }
                    ],
                }
            diagnostics = response.get("diagnostics")
            signature = tuple(
                (
                    str(item.get("rule_code", "")),
                    str(item.get("path", "")),
                )
                for item in diagnostics
                if isinstance(item, Mapping)
            ) if isinstance(diagnostics, list) else ()
            source_changed = (
                current is not None
                and last_replay_check_source is not None
                and current != last_replay_check_source
            )
            if signature and signature == last_replay_check_signature:
                repeated_replay_diagnostic += 1
            else:
                repeated_replay_diagnostic = 1 if signature else 0
            response["diagnostic_progress"] = {
                "same_rule_path_streak": repeated_replay_diagnostic,
                "replay_source_changed_since_previous_check": source_changed,
                "authority": (
                    "mechanical episode-local observation only; it neither edits "
                    "Replay nor changes ingress or Gate1 membership"
                ),
                **(
                    {
                        "next_inspection": (
                            "The same closed obligation persists. Read its expected/"
                            "observed fields, inspect the exact packet path and search "
                            "the relation name in input/REPLAY_CONTRACT.md before "
                            "another structural edit."
                        )
                    }
                    if repeated_replay_diagnostic >= 2
                    else {}
                ),
            }
            last_replay_check_source = current
            last_replay_check_signature = signature
            return response

        def replay_compile_tool() -> Mapping[str, Any]:
            nonlocal last_checked_good
            current = workspace.read_replay()
            ingress = replay_source_ingress(
                current,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )
            if ingress.status != "pass" or ingress.code is None:
                return ingress.to_check_response()
            checked = (
                self.compile_checker(ingress.code, artifact, function_name)
                if self.compile_checker is not None
                else source_blind_replay_compile_check(
                    ingress.code,
                    artifact,
                    function_name,
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            response = checked.to_check_response()
            if checked.status == "pass":
                # The checkpoint is the exact Replay-LLM-authored text that
                # passed both source-blind construction checks in this fresh
                # episode.  Fixed code may restore only these bytes.
                last_checked_good = current
            return response

        def composer_status_tool() -> Mapping[str, Any]:
            try:
                response = _composer_response(
                    composer_dir,
                    manifest_path=composer_manifest,
                    pi=packet["pi"],
                )
            except (OSError, UnicodeError, ValueError) as exc:
                response = {
                    "protocol": REPLAY_COMPOSER_PROTOCOL,
                    "status": "fail",
                    "diagnostic": f"{type(exc).__name__}: {exc}",
                    "authority": (
                        "syntax-workspace validation only; no packet read, semantic "
                        "mapping, repair, Gate, or certification"
                    ),
                }
            composer_events.append(
                {
                    "action": "status",
                    "status": response.get("status"),
                    "mode": response.get("mode", "plain"),
                    "consumed_routes": response.get("consumed_routes"),
                    "total_routes": response.get("total_routes"),
                }
            )
            return response

        def composer_render_tool() -> Mapping[str, Any]:
            try:
                fragments = _composer_fragments(composer_dir)
                if composer_manifest.exists():
                    rendered = render_route_composer(
                        tuple(
                            ComposerFragment(
                                str(item["name"]),
                                str(item["text"]),
                                int(item["bytes"]),
                            )
                            for item in fragments
                        ),
                        manifest_path=composer_manifest,
                        pi=packet["pi"],
                    )
                    source = rendered.source
                    response = rendered.to_response()
                else:
                    source = "".join(str(item["text"]) for item in fragments)
                    response = _composer_response(composer_dir)
                _atomic_write_text(workspace.replay_path, source)
                response["rendered_path"] = "output/replay.source"
            except (OSError, UnicodeError, ValueError) as exc:
                response = {
                    "protocol": REPLAY_COMPOSER_PROTOCOL,
                    "status": "fail",
                    "diagnostic": f"{type(exc).__name__}: {exc}",
                    "authority": (
                        "syntax-only concatenation; no packet read, semantic mapping, "
                        "repair, Gate, or certification"
                    ),
                }
            composer_events.append(
                {
                    "action": "render",
                    "status": response.get("status"),
                    "mode": response.get("mode", "plain"),
                    "total_bytes": response.get("total_bytes"),
                    "consumed_routes": response.get("consumed_routes"),
                    "total_routes": response.get("total_routes"),
                }
            )
            return response

        def submit_replay_tool() -> Mapping[str, Any]:
            """Run both construction checks as one exact-source transaction."""

            nonlocal explicit_submission_accepted

            ingress = dict(replay_check_tool())
            if ingress.get("status") != "pass":
                return {
                    "protocol": "intentir.codex-cli.replay-submit.v2",
                    "status": "fail",
                    "phase": "ingress",
                    "ingress": ingress,
                    "compile": None,
                    "authority": "diagnostic-only; no Replay source is generated or repaired",
                }
            compiled = dict(replay_compile_tool())
            if compiled.get("status") == "pass":
                explicit_submission_accepted = True
            return {
                "protocol": "intentir.codex-cli.replay-submit.v2",
                "status": compiled.get("status", "fail"),
                "phase": "compile",
                "ingress": ingress,
                "compile": compiled,
                "authority": "diagnostic-only; no Replay source is generated or repaired",
            }

        def packet_query_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            """Project one bounded exact view from the immutable source-blind packet."""

            value = _query_packet(packet, [INTENTIR_PACKET_QUERY, *arguments])
            packet_queries.append({"query": list(arguments)})
            return {
                "protocol": "intentir.codex-cli.replay-packet-query.v1",
                "status": "pass",
                "query": list(arguments),
                "value": value,
                "authority": (
                    "exact bounded packet projection only; no Source, semantic "
                    "inference, realization choice, code generation, or Gate authority"
                ),
            }

        def replay_ttir_inspect_tool(
            arguments: tuple[str, ...]
        ) -> Mapping[str, Any]:
            if synthetic_compiler is None:
                return {
                    "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                    "status": "unavailable",
                    "diagnostic": "custom compile checker exposes no Replay TTIR",
                }
            current = workspace.read_replay()
            observation = ttir_observations.get(current)
            if observation is None:
                ingress = replay_source_ingress(
                    current,
                    artifact=artifact,
                    frontend=self.frontend,
                    function_name=function_name,
                    packet=packet,
                )
                if ingress.status != "pass" or ingress.code is None:
                    return ingress.to_check_response()
                case = build_source_blind_replay_compile_case(artifact)
                result = synthetic_compiler.compile(
                    KernelSource.from_replay(ingress.code, function_name),
                    case,
                )
                if not result.ok or not isinstance(result.ttir, str):
                    observation = {
                        "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                        "status": "fail",
                        "diagnostic": result.diagnostic or "compiler returned no TTIR",
                    }
                else:
                    fingerprint = fingerprint_ttir(
                        result.ttir,
                        python_interface=result.python_interface,
                        schedule=case.schedule,
                    )
                    observation = {
                        "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                        "status": "pass",
                        "ttir": result.ttir,
                        "fingerprint": fingerprint.to_dict(),
                        "compiler": dict(result.compiler),
                        "target": dict(result.target),
                    }
                ttir_observations[current] = observation

            mode = arguments[0] if arguments else "summary"
            if mode not in {"summary", "raw", "find", "observer"}:
                raise ValueError(
                    "replay_ttir_inspect mode must be summary/raw/find/observer"
                )
            if observation.get("status") != "pass":
                result_view = dict(observation)
            else:
                ttir = str(observation["ttir"])
                lines = ttir.splitlines()
                if mode == "summary":
                    fingerprint = observation.get("fingerprint")
                    result_view = {
                        key: value
                        for key, value in observation.items()
                        if key not in {"ttir", "fingerprint"}
                    }
                    result_view["ttir_lines"] = len(lines)
                    result_view["fingerprint_sections"] = (
                        list(fingerprint)
                        if isinstance(fingerprint, Mapping)
                        else []
                    )
                elif mode == "raw":
                    if len(arguments) > 3:
                        raise ValueError("raw accepts START [COUNT]")
                    start = int(arguments[1]) if len(arguments) >= 2 else 1
                    count = int(arguments[2]) if len(arguments) >= 3 else 80
                    if start < 1 or count < 1 or count > 160:
                        raise ValueError("raw START/COUNT must be positive; COUNT <= 160")
                    selected = lines[start - 1 : start - 1 + count]
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "start": start,
                        "total": len(lines),
                        "lines": [
                            [ordinal, line]
                            for ordinal, line in enumerate(selected, start=start)
                        ],
                        "next_start": (
                            start + len(selected)
                            if start - 1 + len(selected) < len(lines)
                            else None
                        ),
                    }
                elif mode == "find":
                    if len(arguments) not in {2, 3}:
                        raise ValueError("find accepts TOKEN [CONTEXT]")
                    token = arguments[1]
                    context = int(arguments[2]) if len(arguments) == 3 else 2
                    if not token or len(token) > 256 or not 0 <= context <= 12:
                        raise ValueError("find TOKEN/CONTEXT is outside the closed bound")
                    matches = [index for index, line in enumerate(lines) if token in line]
                    rows: list[list[Any]] = []
                    seen: set[int] = set()
                    for index in matches[:32]:
                        for ordinal in range(
                            max(0, index - context),
                            min(len(lines), index + context + 1),
                        ):
                            if ordinal not in seen:
                                rows.append([ordinal + 1, lines[ordinal]])
                                seen.add(ordinal)
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "token": token,
                        "match_count": len(matches),
                        "lines": rows,
                    }
                else:
                    if len(arguments) != 2:
                        raise ValueError("observer accepts one dotted PATH")
                    value: Any = observation.get("fingerprint")
                    for component in arguments[1].split("."):
                        if not component:
                            raise ValueError("observer PATH contains an empty component")
                        if isinstance(value, Mapping) and component in value:
                            value = value[component]
                        elif isinstance(value, list) and component.isdigit():
                            value = value[int(component)]
                        else:
                            raise KeyError("observer PATH is absent")
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "path": arguments[1],
                        "value": value,
                    }
            ttir_inspections.append(
                {
                    "query": list(arguments) if arguments else ["summary"],
                    "status": result_view.get("status"),
                }
            )
            return {
                **result_view,
                "authority": (
                    "exact TTIR/fingerprint of this episode's current Replay under "
                    "the public synthetic profile only; no Source, Gate, inference, "
                    "or repair"
                ),
            }

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "replay_check": replay_check_tool,
                "replay_compile_check": replay_compile_tool,
                "submit_replay": submit_replay_tool,
                "composer_status": composer_status_tool,
                "composer_render": composer_render_tool,
            },
            argument_handlers={
                "packet_query": packet_query_tool,
                "replay_ttir_inspect": replay_ttir_inspect_tool,
            },
        )

        def evaluate_current(*, trigger: str) -> tuple[bool, Mapping[str, Any]]:
            nonlocal source_text, formal_ingress, formal_compile
            try:
                source_text = workspace.read_replay()
            except Exception:
                source_text = ""
                formal_ingress = None
                formal_compile = None
                check: Mapping[str, Any] = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_unavailable",
                            "path": "<replay>",
                            "anonymous_handles": [],
                            "line": None,
                        }
                    ],
                }
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
                    "trigger": trigger,
                    "status": check["status"],
                    "diagnostics": [dict(item) for item in check["diagnostics"]],
                }
            )
            if (
                formal_ingress is None
                or formal_ingress.status != "pass"
                or formal_ingress.code is None
            ):
                return False, check
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
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            compile_check = formal_compile.to_check_response()
            compile_checks.append(
                {
                    "trigger": trigger,
                    "status": compile_check["status"],
                    "diagnostics": [
                        dict(item) for item in compile_check["diagnostics"]
                    ],
                }
            )
            return formal_compile.status == "pass", compile_check

        try:
            bridge.start()
            session = new_session()
            while True:
                remaining = (
                    episode_limit
                    - active_elapsed_ms / 1000.0
                )
                if remaining <= 0:
                    raise RuntimeError("Replay owner episode wall-time exhausted")
                turn_timeout = min(remaining, turn_limit)
                try:
                    turn = session.run(
                        prompt,
                        timeout_seconds=turn_timeout,
                        effort=selected_effort,
                        stop_requested=self.stop_requested,
                    )
                except CodexCLIError as exc:
                    if "wall-time limit" not in str(exc):
                        raise
                    elapsed = turn_timeout * 1000.0
                    active_elapsed_ms += elapsed
                    timed_out_calls += 1
                    thread_id = getattr(session, "thread_id", None)
                    timeout_event = {
                        "turn_timeout_seconds": turn_timeout,
                        "thread_id": thread_id,
                        "recovery": None,
                    }
                    timeout_events.append(timeout_event)
                    passed, check = evaluate_current(
                        trigger="host_timeout_recheck"
                    )
                    remaining_after_timeout = (
                        episode_limit
                        - active_elapsed_ms / 1000.0
                    )
                    if passed and (
                        explicit_submission_accepted
                        or remaining_after_timeout <= 0
                    ):
                        timeout_event["recovery"] = (
                            "current_exact_replay"
                            if explicit_submission_accepted
                            else "deadline_host_validated_current_exact_replay"
                        )
                        break
                    if last_checked_good is not None:
                        _atomic_write_text(
                            workspace.replay_path, last_checked_good
                        )
                        passed, check = evaluate_current(
                            trigger="host_timeout_checkpoint_restore"
                        )
                        if passed and (
                            explicit_submission_accepted
                            or remaining_after_timeout <= 0
                        ):
                            timeout_event["recovery"] = (
                                "same_episode_checked_checkpoint"
                                if explicit_submission_accepted
                                else "deadline_host_validated_checked_checkpoint"
                            )
                            break
                    if remaining_after_timeout <= 0:
                        timeout_event["recovery"] = "episode_time_exhausted"
                        raise RuntimeError(
                            "Replay owner episode wall-time exhausted"
                        ) from exc
                    if thread_id is None:
                        timeout_event["recovery"] = "no_resumable_thread"
                        raise CodexCLIError(
                            "initial Codex Replay turn timed out without a "
                            "resumable thread or checked Replay"
                        ) from exc
                    write_host_check(check)
                    timeout_event["recovery"] = "same_thread_continuation"
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
                turns.append(turn)
                thread_id = turn.thread_id
                if thread_id not in thread_ids:
                    thread_ids.append(thread_id)
                active_elapsed_ms += turn.elapsed_ms
                passed, check = evaluate_current(trigger="host_post_turn")
                if passed and explicit_submission_accepted:
                    break
                write_host_check(check)
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
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "composer_events": composer_events,
                "timeout_events": timeout_events,
                "explicit_submission_accepted": explicit_submission_accepted,
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
                    "thread_ids": list(thread_ids),
                    "context_rollover_input_tokens": (
                        self.context_rollover_input_tokens
                    ),
                    "episode_wall_time_limit_seconds": (
                        episode_limit
                    ),
                    "cli_turns": len(turns) + timed_out_calls,
                    "model_segments_lower_bound": sum(
                        turn.model_segments for turn in turns
                    ) + timed_out_calls,
                    "tool_actions": sum(turn.tool_actions for turn in turns),
                    "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                    "formal_compile_recheck": True,
                }
            )
            generation = ReplayGeneration(
                code=formal_ingress.code,
                function_name=function_name,
                model=provider.model,
                response_id=thread_id,
                context_receipt=context_receipt,
                usage=usage,
                response_model=provider.model,
                entry_function_name=formal_ingress.entry_function_name,
                response_text=source_text,
                raw_response=raw_trajectory,
                attempts=len(turns) + timed_out_calls,
                physical_provider_calls=(
                    sum(turn.model_segments for turn in turns) + timed_out_calls
                ),
                transport_retries=0,
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
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
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "composer_events": composer_events,
                "timeout_events": timeout_events,
                "tool_bridge": bridge.evidence,
                "failure_class": type(exc).__name__,
                **(
                    {
                        "winner_branch_id": exc.winner_branch_id,
                        "cancelled_elapsed_ms": exc.elapsed_ms,
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
                raw_response=raw_trajectory,
                attempts=len(turns) + timed_out_calls,
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                physical_provider_calls=sum(
                    turn.model_segments for turn in turns
                ) + cancelled_calls + timed_out_calls,
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
