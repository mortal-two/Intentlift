"""Fresh source-blind Replay episodes driven by the isolated Codex CLI runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
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


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v30"

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
The outer runtime exposes only this isolated /workspace. input/ is immutable;
work/, output/, and tmp/ are writable. There is no Source, source slice,
Harness/compiler view, actual C value, Gate result, earlier Replay, sibling
trajectory, or network tool. Do not search for any of them.

The final `<source-blind-startup>` block contains the exact immutable packet when
`exact_packet_embedded=true`. In that normal fast path, do not call
`replay_manifest_view` or `replay_packet_query requirements` before drafting:
consume the embedded J/C/pi/U_f directly. When the packet exceeds the fixed
transport bound, the block instead contains one compact manifest with the same
authored-requirements checklist and focused continuation queries; call only the
named query needed for an omitted page. Up to four independent read-only focused
queries may be issued together. Packet views contain only exact anonymous ABI,
J graph/boundary, MaskC(C), U_f, and authored pi obligations; they do not derive
a schedule or see Gate evidence.

Before writing, reconcile every declared J op occurrence (including fma), every
boundary access/effect, every pi instance mechanism (including transfer_direct
versus atomic realization), and every authored route/order/time/control relation.
Use `critical_facts.lexical_regions` as the exact region-by-region construction
ledger and `critical_facts.cross_region_order` as the exact ordering checklist;
use `critical_facts.broadcast_obligations` as the exact operand/access/consumer
rank-projection checklist; use `critical_facts.identity_materializations` as the
exact representation-stage/adjacent-transfer checklist. These views only
reindex the same J/pi and add no
realization decision. For each authored J.boundary.broadcasts item, preserve the
operand's closed type at its producer and boundary access, then materialize the
declared axes at that named consumer. Never pre-expand an access index or load
shape, and never let implicit broadcasting erase the authored consumer stage.
Every J.signature p* is one already-existing callable ABI carrier. Repeated pi
instances anchored to that same p* only scope or route uses of that one carrier;
they never create another value or program coordinate. Placement=`program` does
not turn p* into `tl.program_id`, and duplicate p* instances never authorize
duplicate carrier reads or coordinates. Only a distinct typed J-domain
realization authorized by the public frontend contract may create a program
coordinate.
This is a consumption checklist, not permission to add a missing operator or
edge. Use `section PATH`, `obligation HANDLE`, or `find HANDLE` only for one
named omitted fact. Preserve every authored effect and realization ordering
relation; never replace pi topology with a guessed source recipe. Use
REPLAY_CONTRACT.md only for a focused unclear rule.
Author the entire Replay as
exactly output/replay.source and no other output file. You may keep
proof-irrelevant notes or helper scripts under work/. You own the source
spelling and low-level realization of facts already authored by J and pi; you
do not invent semantic operators, dataflow edges, predicates, effects, result
relations, or physical routes. If a required fact is absent from the packet,
keep the replay fail-closed rather than guessing it. Run
`python .agents/intentir_agent_tool.py replay_obligation_audit` after writing a
complete source whenever the critical ledger contains broadcast obligations,
identity materializations, lexical regions, or cross-region order. Compare the returned exact ledger with the numbered current
bytes yourself; after any edit, audit those new bytes again. The audit neither
maps instances nor diagnoses or repairs source. Then run
`python .agents/intentir_agent_tool.py replay_check` on your complete source,
repair it until PASS, then run
`python .agents/intentir_agent_tool.py replay_compile_check` for the public
synthetic construction profile. Fixed host code only checks exact bytes and
never writes or repairs Replay. The compile check also preserves the exact
pointer-address operation count already determined by J.signature carrier kinds
and authored pi address routes; it derives no address formula. Both checks must
pass before this episode ends;
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


def _replay_manifest_view(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return one exact fallback index for an oversized packet."""

    from minisweagent.environments.intentir_replay import (
        _packet_overview,
        _packet_requirements,
    )

    overview = _packet_overview(packet)
    return {
        "protocol": "intentir.scrc.replay-manifest-view.v1",
        "status": "pass",
        "entry": overview.get("entry"),
        "frontend": overview.get("frontend"),
        "j_parameter_total": overview.get("j_parameter_total"),
        "operation_count": overview.get("operation_count"),
        "boundary_counts": overview.get("boundary_counts"),
        "pi_counts": overview.get("pi_counts"),
        "root_sections": overview.get("root_sections"),
        "operations_preview": overview.get("operations", []),
        "operations_omitted": overview.get("operations_omitted", 0),
        "authored_requirements": _packet_requirements(packet),
        "query_plan": [
            "python .agents/intentir_agent_tool.py replay_packet_query abi 0 48",
            "python .agents/intentir_agent_tool.py replay_packet_query graph 0 12",
            "python .agents/intentir_agent_tool.py replay_packet_query realization",
            "python .agents/intentir_agent_tool.py replay_packet_query obligations 0 12",
        ],
        "authority": (
            "exact source-isolated packet index only; no Source, Harness value, "
            "Gate evidence, inference, repair, or schedule"
        ),
    }


def _replay_startup_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Embed one exact normal-size packet, otherwise one exact compact manifest."""

    from minisweagent.environments.intentir_replay import (
        REPLAY_STARTUP_PACKET_EMBED_LIMIT_BYTES,
        _packet_critical_facts,
    )

    encoded = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    result: dict[str, Any] = {
        "protocol": "intentir.scrc.replay-startup-view.v1",
        "packet_bytes": len(encoded),
        "embed_limit_bytes": REPLAY_STARTUP_PACKET_EMBED_LIMIT_BYTES,
        "exact_packet_embedded": (
            len(encoded) <= REPLAY_STARTUP_PACKET_EMBED_LIMIT_BYTES
        ),
        "authority": (
            "exact source-isolated packet transport only; no Source, Harness "
            "value, Gate evidence, inference, repair, code, or schedule"
        ),
        "critical_facts": _packet_critical_facts(packet),
    }
    if result["exact_packet_embedded"]:
        result["packet"] = packet
    else:
        result["manifest"] = _replay_manifest_view(packet)
    return result


def _replay_packet_query(
    packet: Mapping[str, Any], arguments: tuple[str, ...]
) -> dict[str, Any]:
    """Run the existing bounded packet projection with a closed argument list."""

    from minisweagent.environments.intentir_replay import (
        INTENTIR_PACKET_QUERY,
        _query_packet,
    )

    if not arguments:
        raise ValueError("replay_packet_query needs a query kind")
    result = _query_packet(packet, [INTENTIR_PACKET_QUERY, *arguments])
    return {
        "protocol": "intentir.scrc.replay-packet-query.v1",
        "status": "pass",
        "query": list(arguments),
        "view": result,
        "authority": (
            "exact bounded source-isolated packet projection only; no Source, "
            "Harness value, Gate evidence, inference, repair, or schedule"
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

    reductions = _replay_reduction_obligations(packet)
    return {
        "protocol": "intentir.scrc.replay-workspace-view.v2",
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
            "reduction_obligations": reductions,
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


def _replay_reduction_obligations(
    packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Colocate only already-authored J/C/pi reduction facts.

    This is a lossless lookup/join for model attention.  It does not select a
    physical rank, tile, reshape, loop, or frontend spelling.
    """

    j = packet.get("j")
    c = packet.get("c")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(c, Mapping) or not isinstance(
        pi, Mapping
    ):
        return []
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary")
    if not isinstance(function, Mapping) or not isinstance(boundary, Mapping):
        return []
    node_ids = j.get("node_ids")
    path_to_node = (
        {str(path): str(node) for node, path in node_ids.items()}
        if isinstance(node_ids, Mapping)
        else {}
    )
    boundary_rows = boundary.get("reductions")
    reduction_by_ref: dict[str, list[Mapping[str, Any]]] = {}
    if isinstance(boundary_rows, list):
        for item in boundary_rows:
            if isinstance(item, Mapping) and isinstance(item.get("node_ref"), str):
                reduction_by_ref.setdefault(str(item["node_ref"]), []).append(item)
    slots = c.get("slots")
    instances = pi.get("instances")
    slot_rows = slots if isinstance(slots, list) else []
    instance_rows = instances if isinstance(instances, list) else []

    value_types: dict[str, Any] = {}
    for collection in ("inputs", "outputs"):
        tensors = function.get(collection)
        if not isinstance(tensors, list):
            continue
        for tensor in tensors:
            if isinstance(tensor, Mapping) and isinstance(tensor.get("name"), str):
                value_types[str(tensor["name"])] = {
                    key: tensor.get(key) for key in ("dtype", "shape")
                }

    result: list[dict[str, Any]] = []

    def visit(body: Any, prefix: str, local_types: dict[str, Any]) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            node_ref = (
                str(operation["id"])
                if isinstance(operation.get("id"), str)
                else path_to_node.get(path)
            )
            operands = [
                str(item)
                for item in operation.get("operands", [])
                if isinstance(item, str)
            ]
            if operation.get("op") == "reduce":
                result.append(
                    {
                        "node_ref": node_ref,
                        "operands": operands,
                        "operand_types": [local_types.get(item) for item in operands],
                        "semantic_axes": (
                            operation.get("attrs", {}).get("axes")
                            if isinstance(operation.get("attrs"), Mapping)
                            else None
                        ),
                        "result": operation.get("result"),
                        "result_type": operation.get("type"),
                        "boundary": [
                            dict(item)
                            for item in reduction_by_ref.get(str(node_ref), [])
                        ],
                        "c_slots": [
                            dict(item)
                            for item in slot_rows
                            if isinstance(item, Mapping)
                            and item.get("subject_ref") == node_ref
                        ],
                        "pi_instances": [
                            dict(item)
                            for item in instance_rows
                            if isinstance(item, Mapping)
                            and item.get("semantic_ref") == node_ref
                        ],
                    }
                )
            produced = operation.get("result")
            produced_names = (
                [produced]
                if isinstance(produced, str)
                else [item for item in produced if isinstance(item, str)]
                if isinstance(produced, list)
                else []
            )
            for name in produced_names:
                local_types[name] = operation.get("type")
            visit(operation.get("body"), f"{path}.body", dict(local_types))
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if isinstance(arm, Mapping):
                    visit(
                        arm.get("body"),
                        f"{path}.{polarity}.body",
                        dict(local_types),
                    )

    visit(function.get("body"), "body", value_types)
    return result


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
        if self.frontend not in {
            Frontend.TRITON,
            Frontend.CUDA,
            Frontend.TILELANG,
        }:
            raise ValueError(
                f"Codex CLI Replay has no audited ingress for {self.frontend.value}"
            )
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
        obligation_audits: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        source_text = ""
        formal_ingress = None
        formal_compile = None
        last_checked_good: str | None = None
        startup_context = _replay_startup_context(packet)
        from minisweagent.environments.intentir_replay import (
            _packet_critical_facts,
            _replay_obligation_audit_view,
        )

        critical_facts = _packet_critical_facts(packet)
        obligation_audit_required = bool(
            critical_facts["broadcast_obligations"]
            or critical_facts["identity_materializations"]
            or critical_facts["lexical_regions"]
            or critical_facts["cross_region_order"]
        )
        obligation_audited_source_digests: set[str] = set()
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
            except Exception:
                return {
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

        def replay_compile_tool() -> Mapping[str, Any]:
            nonlocal last_checked_good
            current = workspace.read_replay()
            current_digest = hashlib.sha256(current.encode()).hexdigest()
            if (
                obligation_audit_required
                and current_digest not in obligation_audited_source_digests
            ):
                return {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_obligation_audit_required",
                            "path": "<current-replay-bytes>",
                            "anonymous_handles": [],
                            "line": None,
                        }
                    ],
                }
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
                    frontend=self.frontend,
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

        def replay_obligation_audit_tool() -> Mapping[str, Any]:
            current = workspace.read_replay()
            view = _replay_obligation_audit_view(packet, current)
            source_digest = str(view["source_sha256"])
            obligation_audited_source_digests.add(source_digest)
            obligation_audits.append(
                {
                    "source_sha256": source_digest,
                    "broadcast_obligation_count": len(
                        view["broadcast_obligations"]
                    ),
                    "identity_materialization_count": len(
                        view["identity_materializations"]
                    ),
                    "lexical_region_count": len(view["lexical_regions"]),
                    "cross_region_order_count": len(view["cross_region_order"]),
                }
            )
            return view

        def replay_manifest_view_tool() -> Mapping[str, Any]:
            return _replay_manifest_view(packet)

        def replay_packet_query_tool(
            arguments: tuple[str, ...]
        ) -> Mapping[str, Any]:
            return _replay_packet_query(packet, arguments)

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "replay_manifest_view": replay_manifest_view_tool,
                "replay_check": replay_check_tool,
                "replay_compile_check": replay_compile_tool,
                "replay_obligation_audit": replay_obligation_audit_tool,
            },
            argument_handlers={
                "replay_packet_query": replay_packet_query_tool,
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
                    "source_sha256": (
                        hashlib.sha256(source_text.encode()).hexdigest()
                        if source_text
                        else None
                    ),
                }
            )
            if (
                formal_ingress is None
                or formal_ingress.status != "pass"
                or formal_ingress.code is None
            ):
                return False, check
            source_digest = hashlib.sha256(source_text.encode()).hexdigest()
            if (
                obligation_audit_required
                and source_digest not in obligation_audited_source_digests
            ):
                audit_check: Mapping[str, Any] = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_obligation_audit_required",
                            "path": "<current-replay-bytes>",
                            "anonymous_handles": [],
                            "line": None,
                        }
                    ],
                }
                compile_checks.append(
                    {
                        "trigger": trigger,
                        "status": "fail",
                        "diagnostics": [
                            dict(item) for item in audit_check["diagnostics"]
                        ],
                        "source_sha256": source_digest,
                    }
                )
                return False, audit_check
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
                    frontend=self.frontend,
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
                    "source_sha256": hashlib.sha256(
                        source_text.encode()
                    ).hexdigest(),
                }
            )
            return formal_compile.status == "pass", compile_check

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
                    if passed:
                        timeout_event["recovery"] = "current_exact_replay"
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
                            )
                            break
                    remaining_after_timeout = (
                        self.session_wall_time_limit_seconds
                        - active_elapsed_ms / 1000.0
                    )
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
                    timeout_event["recovery"] = "same_thread_continuation"
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
                turns.append(turn)
                thread_id = turn.thread_id
                active_elapsed_ms += turn.elapsed_ms
                passed, check = evaluate_current(trigger="host_post_turn")
                if passed:
                    break
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
                "replay_obligation_audits": obligation_audits,
                "timeout_events": timeout_events,
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
                    "cli_turns": len(turns) + timed_out_calls,
                    "model_segments_lower_bound": sum(
                        turn.model_segments for turn in turns
                    ) + timed_out_calls,
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
                attempts=len(turns) + timed_out_calls,
                physical_provider_calls=(
                    sum(turn.model_segments for turn in turns) + timed_out_calls
                ),
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
                "replay_obligation_audits": obligation_audits,
                "timeout_events": timeout_events,
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
                attempts=len(turns) + timed_out_calls,
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
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
