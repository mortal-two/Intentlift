"""Fresh source-blind Replay episodes driven by the isolated Codex CLI runtime."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIDiagnosticReady,
    CodexCLIError,
    CodexCLISubmissionAccepted,
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
from .replay_semantic_core import (
    materialize_replay_semantics,
    query_replay_semantics,
)
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


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v64"

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
Work only under the isolated /workspace. input/ is immutable; output/, work/, and
tmp/ are writable. There is no Source, Harness/compiler view, actual C value, Gate
result, earlier Replay, sibling state, repository, or network.

J is the only semantic authority. Read input/compiler-index.intent and the complete
read-only input/semantic-program.intent before drafting, then read input/frontend.json
and input/mask-c.json for the callable ABI. S_J is a deterministic parameterized compiler
IR derived only from this J, not a target-source skeleton. Use `semantic_query` or
`packet_query` only to cross-reference one named J/C handle after reading those complete
small views. Pi, Source realization, Gate observations, and prior Replay are absent.

Act as the complete Replay compiler. Write one full target program to
output/replay.source yourself. You own instruction selection, addressing, program
mapping, control, loops/carriers, layout, memory, synchronization, target APIs and
whole-program restructuring while preserving all J-owned value semantics. Do not edit
S_J or infer a Source recipe. input/REPLAY_CONTRACT.md defines
the ownership boundary and input/TOOL.md defines the tools.

`compiler-index.intent` is only a compact cross-reference of facts already present in
S_J and MaskC. Its dependency subsets are syntactic and neither select nor authorize a
program axis, loop, lane, guard, or target API. Use it to avoid repeated JSON joins,
then compile `boundary_ownership` as one whole-program contract. Preserve every
authored access/effect and keep logical domain guards distinct from access-local
predicates; you still own every target realization choice.

Compile, inspect only your own Replay IR, and patch the complete program in this same
episode. First run `replay_compile_check`; once construction passes, call
`replay_ttir_inspect jc` for the bounded J/C structural summary. Compare its exact
typed J obligations against your own dot/reduction/loop/cast signatures before submit;
matching only operation counts is insufficient. Inspect your own IR
again when a concrete J obligation remains unclear; use targeted `find`, `raw`, or
`observer` queries instead of dumping the whole fingerprint. Keep computations for inactive constexpr
alternatives inside their branch and do not add unmentioned `volatile`, cache,
eviction, or other target attributes. Then run
`submit_replay` on the reviewed exact bytes and repair until it
passes. Fixed code materializes S_J, projects exact packet facts, checks and rejects;
it never constructs or repairs target source. Finish only with the transaction passing
and do not claim Gate1, Gate2, or certification.
"""

_REPAIR_PROMPT = """The same fresh source-blind Replay episode continues in the same thread and workspace.
Read work/host_replay_check.json and the current output/replay.source. Repair the
complete program using the same read-only S_J, then rerun submit_replay.
Diagnostics reject only; they do not supply a semantic mapping or repair. Finish only
when ingress and compile both pass.
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


def _replay_startup_context(
    packet: Mapping[str, Any], files: Mapping[str, str]
) -> dict[str, Any]:
    """Return a path-only receipt; Codex reads the exact packet as a local file."""

    encoded = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "protocol": "intentir.scrc.replay-startup-view.v6",
        "packet_bytes": len(encoded),
        "semantic_program_text_bytes": len(
            files["semantic-program.intent"].encode("utf-8")
        ),
        "semantic_program_json_bytes": len(
            files["semantic-program.json"].encode("utf-8")
        ),
        "compiler_index_text_bytes": len(
            files["compiler-index.intent"].encode("utf-8")
        ),
        "compiler_index_json_bytes": len(
            files["compiler-index.json"].encode("utf-8")
        ),
        "paths": {
            "packet": "input/packet.json",
            "semantic_program": "input/semantic-program.intent",
            "semantic_program_json": "input/semantic-program.json",
            "compiler_index": "input/compiler-index.intent",
            "compiler_index_json": "input/compiler-index.json",
            "mask_c": "input/mask-c.json",
            "frontend": "input/frontend.json",
            "public_contract": "input/REPLAY_CONTRACT.md",
            "tool_contract": "input/TOOL.md",
            "replay": "output/replay.source",
            "host_check": "work/host_replay_check.json",
        },
        "authority": (
            "path and byte-count receipt only; exact source-isolated packet bytes "
            "remain in input/packet.json and no code or schedule is generated"
        ),
    }


def _jc_fingerprint_summary(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Project one bounded, source-blind structural view of Replay's own TTIR."""

    result: dict[str, Any] = {}
    for key in (
        "interface",
        "op_multiset",
        "effect_multiset",
        "reductions",
        "dots",
        "structured_ifs",
        "terminal_forks",
    ):
        if key in fingerprint:
            result[key] = fingerprint[key]

    loops = fingerprint.get("loops")
    if isinstance(loops, list):
        kept = {
            "block_argument_count",
            "body_effects",
            "body_op_multiset",
            "carried_operand_count",
            "result_count",
        }
        result["loops"] = [
            {key: row[key] for key in kept if key in row}
            for row in loops
            if isinstance(row, Mapping)
        ]

    for key in ("dataflow_edges", "memory_effects", "mask_predicates"):
        value = fingerprint.get(key)
        if isinstance(value, (list, tuple, dict)):
            result[f"{key}_count"] = len(value)
    return result


def _j_contract_summary(semantic_program: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact J-only checklist beside Replay's own TTIR summary.

    Counts here are logical authored facts, not required raw TTIR multiplicities:
    target lowering may fuse or split representation.  The view gives the coding
    agent a bounded omission checklist without reading Source, pi, or Gate state.
    """

    nodes = semantic_program.get("nodes", [])
    node_ops = Counter(
        str(item.get("op"))
        for item in nodes
        if isinstance(item, Mapping) and isinstance(item.get("op"), str)
    ) if isinstance(nodes, list) else Counter()
    value_types_value = semantic_program.get("value_types", [])
    value_types = {
        str(item["value"]): {
            "dtype": item.get("dtype"),
            "shape": item.get("shape"),
        }
        for item in value_types_value
        if isinstance(item, Mapping) and isinstance(item.get("value"), str)
    } if isinstance(value_types_value, list) else {}

    # These operations are exactly the places where a superficially plausible
    # target program most often changes a J-owned type, axis, temporal state, or
    # representation boundary.  The list is a read-only projection of S_J; it is
    # not a target recipe and it neither accepts nor rejects Replay.
    typed_kinds = {
        "cast",
        "contraction",
        "gather",
        "loop",
        "reduce",
        "scan",
        "transpose",
    }
    typed_obligations: list[dict[str, Any]] = []
    for item in nodes if isinstance(nodes, list) else []:
        if not isinstance(item, Mapping) or item.get("op") not in typed_kinds:
            continue
        raw_operands = item.get("operands", [])
        operands = raw_operands if isinstance(raw_operands, list) else []
        raw_result = item.get("result")
        results = raw_result if isinstance(raw_result, list) else [raw_result]
        record = {
            "id": item.get("id"),
            "op": item.get("op"),
            "operands": [
                {"value": value, **value_types.get(str(value), {})}
                for value in operands
            ],
            "results": [
                {"value": value, **value_types.get(str(value), {})}
                for value in results
                if value is not None
            ],
        }
        for key in ("attrs", "trip", "carry", "scan", "yield"):
            if key in item:
                record[key] = item[key]
        typed_obligations.append(record)

    ownership = semantic_program.get("boundary_ownership", {})
    accesses = (
        ownership.get("access_effects", [])
        if isinstance(ownership, Mapping)
        else []
    )
    access_kinds: Counter[str] = Counter()
    effect_kinds: Counter[str] = Counter()
    accesses_by_tensor: Counter[str] = Counter()
    for item in accesses if isinstance(accesses, list) else []:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("kind"), str):
            access_kinds[str(item["kind"])] += 1
        if isinstance(item.get("tensor"), str):
            accesses_by_tensor[str(item["tensor"])] += 1
        effects = item.get("effects", [])
        for effect in effects if isinstance(effects, list) else []:
            if isinstance(effect, Mapping) and isinstance(effect.get("kind"), str):
                effect_kinds[str(effect["kind"])] += 1
    return {
        "authority": (
            "J-only authored-fact checklist; logical counts are not raw TTIR "
            "occurrence requirements and do not accept or reject Replay"
        ),
        "node_ops": dict(sorted(node_ops.items())),
        "typed_obligations": typed_obligations[:96],
        "typed_obligations_omitted": max(0, len(typed_obligations) - 96),
        "access_kinds": dict(sorted(access_kinds.items())),
        "effect_kinds": dict(sorted(effect_kinds.items())),
        "accesses_by_tensor": dict(sorted(accesses_by_tensor.items())),
        "shared_write_axis_refs": (
            ownership.get("shared_write_axis_refs", [])
            if isinstance(ownership, Mapping)
            else []
        ),
        "shared_write_domain_axis_refs": (
            ownership.get("shared_write_domain_axis_refs", [])
            if isinstance(ownership, Mapping)
            else []
        ),
        "domain_axis_refs": (
            ownership.get("domain_axis_refs", {})
            if isinstance(ownership, Mapping)
            else {}
        ),
        "outputs": semantic_program.get("output_bindings", []),
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
        agent_packet = dict(packet)
        semantic_program = materialize_replay_semantics(packet["j"]).to_json()
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
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        compile_checks: list[dict[str, Any]] = []
        ttir_inspections: list[dict[str, Any]] = []
        packet_queries: list[dict[str, Any]] = []
        semantic_queries: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        accepted_submission_interrupts: list[dict[str, Any]] = []
        diagnostic_ready_interrupts: list[dict[str, Any]] = []
        live_checkpoint_events: list[dict[str, Any]] = []
        accepted_submission_event = threading.Event()
        diagnostic_ready_event = threading.Event()
        live_checkpoint_stop = threading.Event()
        evaluation_lock = threading.RLock()
        source_text = ""
        formal_ingress = None
        formal_compile = None
        last_checked_good: str | None = None
        startup_context = _replay_startup_context(agent_packet, files)
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
                accepted_submission_event.set()
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

        def semantic_query_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            value = query_replay_semantics(semantic_program, arguments)
            semantic_queries.append({"query": list(arguments)})
            return {
                **value,
                "authority": value.get(
                    "authority", semantic_program.get("authority")
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
            if mode not in {"jc", "summary", "raw", "find", "observer"}:
                raise ValueError(
                    "replay_ttir_inspect mode must be jc/summary/raw/find/observer"
                )
            if observation.get("status") != "pass":
                result_view = dict(observation)
            else:
                ttir = str(observation["ttir"])
                lines = ttir.splitlines()
                if mode == "jc":
                    if len(arguments) != 1:
                        raise ValueError("jc accepts no additional arguments")
                    fingerprint = observation.get("fingerprint")
                    if not isinstance(fingerprint, Mapping):
                        raise ValueError("compiled Replay exposes no fingerprint")
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "ttir_lines": len(lines),
                        "j_contract": _j_contract_summary(semantic_program),
                        "jc": _jc_fingerprint_summary(fingerprint),
                    }
                elif mode == "summary":
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

        def _locked_tool_call(
            callback: Callable[..., Mapping[str, Any]],
            *arguments: Any,
        ) -> Mapping[str, Any]:
            with evaluation_lock:
                return callback(*arguments)

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "replay_check": lambda: _locked_tool_call(replay_check_tool),
                "replay_compile_check": lambda: _locked_tool_call(
                    replay_compile_tool
                ),
                "submit_replay": lambda: _locked_tool_call(submit_replay_tool),
            },
            argument_handlers={
                "packet_query": packet_query_tool,
                "semantic_query": semantic_query_tool,
                "replay_ttir_inspect": lambda arguments: _locked_tool_call(
                    replay_ttir_inspect_tool, arguments
                ),
            },
        )

        def _evaluate_current_unlocked(
            *, trigger: str
        ) -> tuple[bool, Mapping[str, Any]]:
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

        def evaluate_current(*, trigger: str) -> tuple[bool, Mapping[str, Any]]:
            with evaluation_lock:
                return _evaluate_current_unlocked(trigger=trigger)

        try:
            initial_replay = workspace.read_replay()
        except Exception:
            initial_replay = None
        live_checkpoint_attempted = (
            {initial_replay} if initial_replay is not None else set()
        )

        def live_checkpoint_watch() -> None:
            """Accept only stable exact Replay bytes passing both host checks."""

            nonlocal last_checked_good
            observed_replay: str | None = None
            stable_since = 0.0
            while not live_checkpoint_stop.wait(0.25):
                try:
                    observed_text = workspace.read_replay()
                except Exception:
                    observed_replay = None
                    stable_since = 0.0
                    continue
                if observed_text in live_checkpoint_attempted:
                    continue
                if observed_text != observed_replay:
                    observed_replay = observed_text
                    stable_since = time.monotonic()
                    continue
                if time.monotonic() - stable_since < 3.0:
                    continue
                live_checkpoint_attempted.add(observed_text)
                passed, check = evaluate_current(trigger="live_checkpoint")
                with evaluation_lock:
                    checked_text = source_text
                    accepted = passed and checked_text == observed_text
                    if accepted:
                        last_checked_good = checked_text
                live_checkpoint_events.append(
                    {
                        "replay_bytes": len(checked_text.encode()) if checked_text else 0,
                        "status": check.get("status", "fail"),
                        "accepted": accepted,
                    }
                )
                if checked_text != observed_text:
                    live_checkpoint_attempted.discard(observed_text)
                if accepted:
                    accepted_submission_event.set()
                    return
                if checked_text == observed_text:
                    write_host_check(check)
                    diagnostic_ready_event.set()
                    while (
                        diagnostic_ready_event.is_set()
                        and not live_checkpoint_stop.wait(0.25)
                    ):
                        pass
                    observed_replay = None
                    stable_since = 0.0

        live_checkpoint_thread: threading.Thread | None = None
        try:
            bridge.start()
            live_checkpoint_thread = threading.Thread(
                target=live_checkpoint_watch,
                name="intentir-codex-replay-live-checkpoint",
                daemon=True,
            )
            live_checkpoint_thread.start()
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
                        submission_accepted=accepted_submission_event.is_set,
                        diagnostic_ready=diagnostic_ready_event.is_set,
                    )
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                except CodexCLISubmissionAccepted as exc:
                    active_elapsed_ms += exc.elapsed_ms
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                    thread_id = exc.thread_id
                    if thread_id not in thread_ids:
                        thread_ids.append(thread_id)
                    accepted_submission_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "pending_exact_checkpoint_recheck",
                        }
                    )
                    if last_checked_good is None:
                        raise CodexCLIError(
                            "accepted Replay submission has no exact checked checkpoint"
                        ) from exc
                    _atomic_write_text(workspace.replay_path, last_checked_good)
                    passed, _ = evaluate_current(
                        trigger="host_accepted_submission_restore"
                    )
                    if not passed:
                        raise CodexCLIError(
                            "accepted Replay failed independent host recheck"
                        ) from exc
                    accepted_submission_interrupts[-1]["recovery"] = (
                        "exact_checkpoint_restored"
                    )
                    break
                except CodexCLIDiagnosticReady as exc:
                    active_elapsed_ms += exc.elapsed_ms
                    diagnostic_ready_event.clear()
                    accepted_submission_event.clear()
                    thread_id = exc.thread_id
                    if thread_id not in thread_ids:
                        thread_ids.append(thread_id)
                    diagnostic_ready_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "same_thread_closed_diagnostic",
                        }
                    )
                    passed, check = evaluate_current(
                        trigger="host_live_diagnostic_recheck"
                    )
                    if passed:
                        break
                    write_host_check(check)
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
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
                    if passed:
                        timeout_event["recovery"] = (
                            "current_exact_replay"
                            if explicit_submission_accepted
                            else "turn_deadline_host_validated_current_exact_replay"
                        )
                        break
                    if last_checked_good is not None:
                        _atomic_write_text(
                            workspace.replay_path, last_checked_good
                        )
                        passed, check = evaluate_current(
                            trigger="host_timeout_checkpoint_restore"
                        )
                        if passed:
                            timeout_event["recovery"] = (
                                "same_episode_checked_checkpoint"
                                if explicit_submission_accepted
                                else "turn_deadline_host_validated_checked_checkpoint"
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
                if passed:
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
                "semantic_queries": semantic_queries,
                "timeout_events": timeout_events,
                "accepted_submission_interrupts": accepted_submission_interrupts,
                "diagnostic_ready_interrupts": diagnostic_ready_interrupts,
                "live_checkpoint": live_checkpoint_events,
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
                    "cli_turns": (
                        len(turns)
                        + timed_out_calls
                        + len(accepted_submission_interrupts)
                        + len(diagnostic_ready_interrupts)
                    ),
                    "model_segments_lower_bound": (
                        sum(turn.model_segments for turn in turns)
                        + timed_out_calls
                        + len(accepted_submission_interrupts)
                        + len(diagnostic_ready_interrupts)
                    ),
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
                attempts=(
                    len(turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                ),
                physical_provider_calls=(
                    sum(turn.model_segments for turn in turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
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
                "semantic_queries": semantic_queries,
                "timeout_events": timeout_events,
                "accepted_submission_interrupts": accepted_submission_interrupts,
                "diagnostic_ready_interrupts": diagnostic_ready_interrupts,
                "live_checkpoint": live_checkpoint_events,
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
                attempts=(
                    len(turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                ),
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                physical_provider_calls=sum(
                    turn.model_segments for turn in turns
                ) + cancelled_calls + timed_out_calls + len(
                    accepted_submission_interrupts
                ) + len(diagnostic_ready_interrupts),
                transport_retries=0,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        finally:
            live_checkpoint_stop.set()
            if live_checkpoint_thread is not None:
                live_checkpoint_thread.join()
            bridge.close()
            if session is not None:
                session.close()
            workspace.cleanup()


__all__ = ["CODEX_CLI_REPLAY_BACKEND", "CodexCLIReplayWriter"]
