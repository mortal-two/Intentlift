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
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.ttir import fingerprint_ttir

from .model import Frontend, Gate1Artifact
from .replay_composer import (
    REPLAY_COMPOSER_BUILDER,
    REPLAY_COMPOSER_MODES,
    REPLAY_COMPOSER_PLAN,
    REPLAY_COMPOSER_NODE_DRAFT,
    REPLAY_COMPOSER_PROTOCOL,
    REPLAY_COMPOSER_SOURCE_MAP,
    REPLAY_COMPOSER_STATE,
    REPLAY_COMPOSER_TOOL,
    checkpoint_episode_composer,
    composer_diagnostic_map,
    install_episode_composer,
    load_episode_composer_evidence,
    restore_episode_composer_checkpoint,
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


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v36"
PI_MIN_OWNERSHIP_RECEIPT_PROTOCOL = "intentir.pi-min.ownership-receipt.v1"

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
Work only under the isolated /workspace. input/ is immutable; output/, work/, and
tmp/ are writable. There is no Source, Harness/compiler view, actual C value, Gate
result, earlier Replay, sibling state, repository, or network.

Act like a coding agent. The exact source-blind packet is J + pi + MaskC(C) + public
U_f in input/packet.json. Read that file once with ordinary file tools, form one
coherent implementation, and write a complete output/replay.source early. Do not
walk the packet one leaf at a time. The packet_query tool is only a fallback for one
precise path or handle that remains unclear; do not begin with overview/requirements/
realization query chains. input/REPLAY_CONTRACT.md is a public reference to inspect
only when one exact rule is unclear. You choose source spelling and realization,
but every semantic operation, operand edge, predicate, effect, ABI carrier, region,
route, order, sharing, and time/control obligation must come from the packet. Do not
guess a Source recipe or invent missing facts.

Before writing, reconcile every declared J op occurrence (including fma), every
boundary access/effect, every pi instance mechanism (including transfer_direct
versus atomic realization), and every authored route/order/time/control relation.
Use `critical_facts.lexical_regions` as the exact region-by-region construction
ledger and `critical_facts.cross_region_order` as the exact ordering checklist;
use `critical_facts.broadcast_obligations` as the exact operand/access/consumer
rank-projection checklist; use `critical_facts.identity_materializations` as the
exact representation-stage/adjacent-transfer checklist; consume every
`critical_facts.route_extent_transforms` item at its exact existing route slot.
Consume every `critical_facts.participant_mappings` item by resolving its
domain_instance to exact J before selecting the public participant spelling.
These views only reindex the same J/pi and add no
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
identity materializations, lexical regions, cross-region order, or route extent
transforms. Compare the
returned exact ledger with the numbered current
bytes yourself; after any edit, audit those new bytes again. The audit neither
maps instances nor diagnoses or repairs source. After the required audit, run
`python .agents/intentir_agent_tool.py submit_replay`. It transactionally runs the
source-blind ingress and synthetic compile checks on your exact bytes. Repair your
own source until it passes. If compilation is unclear, use `replay_ttir_inspect`
only on a current Triton Replay; TileLang uses TIR and CUDA uses its native compile
path. Fixed code only projects packet facts, checks, and rejects; it never writes
or repairs Replay. The compile check preserves the exact pointer-address operation
count already determined by J.signature carrier kinds and authored pi address
routes; it derives no address formula. Finish only with the transaction passing.
Neither check has Gate or certification authority. Do not claim Gate1, Gate2, or
certification.
"""

_REPAIR_PROMPT = """The same fresh source-blind Replay episode continues in the same thread and workspace.
Read work/host_replay_check.json and the current output/replay.source, repair the
complete source yourself, and rerun submit_replay. The
diagnostic is rejection-only. Finish only when both pass.
"""

_COMPOSER_AVAILABLE_PROMPT = """

This fresh episode also provides a tiny syntax-only composer. It never reads
input/packet.json and contains no J/pi-to-target mapping. The lowest-friction path
is to author work/build_replay.py with its generic line/block/raw_block constructors
and run it; this atomically renders output/replay.source and records stable
node-to-line spans. You may instead edit work/replay.composer.json and run
`python work/replay_composer.py render`. For local edits, put one node in
work/replay.composer.node.json and use insert-before, insert-after, append-child, or
replace; delete/move commands need no node draft.
Use `checkpoint`, `rollback`, or `status` on the same helper when useful. Directly
authoring output/replay.source remains a valid raw escape. Do not spend a turn
reimplementing the composer or treating it as a semantic planner: you still choose
every target API, expression, control structure, and realization from exact J/pi.
See input/TOOL.md for the compact plan grammar.
"""

_COMPOSER_PREFERRED_PROMPT = """

This episode is the composer-preferred efficiency arm. Start the first complete
Replay by editing work/build_replay.py with the provided generic
line/block/raw_block constructors and run it. Keep stable IDs for regions you may
repair, then use the local edit commands when diagnostics identify a node. This is
only an authoring treatment: it receives no extra semantic information and changes
no ingress, compile, Gate1, or Gate2 rule. Unknown target syntax belongs in
raw_block. Direct output/replay.source authoring remains a last-resort raw escape;
use it when the syntax helper would genuinely add friction, not merely by default.
You still derive every API, expression, control structure, and realization from
exact J/pi yourself.
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


def _pi_min_ownership_receipt(
    packet: Mapping[str, Any], *, artifact_schema: str, masked: bool
) -> dict[str, Any]:
    """Summarize already-validated ownership without becoming another validator."""

    pi = packet.get("pi")
    relation_counts = {
        key: len(value)
        for key, value in (pi.items() if isinstance(pi, Mapping) else ())
        if isinstance(value, list)
    }
    return {
        "protocol": PI_MIN_OWNERSHIP_RECEIPT_PROTOCOL,
        "status": "pass",
        "artifact_schema": artifact_schema,
        "masked_j": masked,
        "semantic_authority": "J only",
        "pi_authority": "J/C-dependent realization constraints only",
        "pi_closed_fields": sorted(pi) if isinstance(pi, Mapping) else [],
        "relation_counts": relation_counts,
        "validation_basis": [
            "canonical Gate1Artifact closed-schema decode",
            "TypeCheck(J,C,pi)=PASS before Replay packet construction",
            "closed source-isolated Replay payload grammar",
        ],
        "claim_scope": (
            "proof-irrelevant ownership receipt; it neither proves zero statistical "
            "information nor changes TypeCheck, Replay, Gate1, or Gate2 acceptance"
        ),
    }
def _replay_startup_context(
    packet: Mapping[str, Any], *, composer_mode: str
) -> dict[str, Any]:
    """Return paths plus a proof-irrelevant index of the exact local packet."""

    from minisweagent.environments.intentir_replay import _packet_critical_facts

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
            "replay": "output/replay.source",
            "host_check": "work/host_replay_check.json",
            **(
                {
                    "composer_tool": f"work/{REPLAY_COMPOSER_TOOL}",
                    "composer_builder": f"work/{REPLAY_COMPOSER_BUILDER}",
                    "composer_plan": f"work/{REPLAY_COMPOSER_PLAN}",
                    "composer_node_draft": f"work/{REPLAY_COMPOSER_NODE_DRAFT}",
                    "composer_state": f"work/{REPLAY_COMPOSER_STATE}",
                    "composer_source_map": f"work/{REPLAY_COMPOSER_SOURCE_MAP}",
                }
                if composer_mode != "off"
                else {}
            ),
        },
        "syntax_composer": {
            "available": composer_mode != "off",
            "offer_mode": composer_mode,
            "protocol": REPLAY_COMPOSER_PROTOCOL if composer_mode != "off" else None,
            "authority": (
                "episode-local syntax rendering of agent-authored nodes only; "
                "no packet read, semantic mapping, target recipe, or Gate authority"
            ),
        },
        "critical_facts": _packet_critical_facts(packet),
        "authority": (
            "path, byte-count, and exact packet projection only; source-isolated "
            "packet bytes remain in input/packet.json and no code or schedule is "
            "generated"
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
    # gets two full provider windows; xhigh retains its full transport window.
    # This prevents a bad direction from cycling for an hour while leaving every
    # useful coding-agent tool action inside those turns unrestricted.
    session_wall_time_limit_seconds: float | None = None
    # Codex owns its thread compaction.  Zero disables host-side thread rollover;
    # cumulative API input-token counters are not a measure of live context size.
    context_rollover_input_tokens: int = 0
    repair_effort: str | None = None
    compile_checker: Callable[[str, Gate1Artifact, str], Any] | None = None
    composer_mode: str = "available"
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
        if self.composer_mode not in REPLAY_COMPOSER_MODES:
            raise ValueError("composer_mode must be off/available/preferred")
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
        episode_limit = self.session_wall_time_limit_seconds
        if episode_limit is None:
            episode_limit = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else 2.0 * DEFAULT_RESPONSES_TIMEOUT_SECONDS
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
        files, packet, instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        function_name = str(packet["u_f"]["replay_function"])

        try:
            from minisweagent.environments.intentir_replay import (
                INTENTIR_PACKET_QUERY,
                IntentIRReplayWorkspace,
                _packet_critical_facts,
                _query_packet,
                _replay_obligation_audit_required,
                _replay_obligation_audit_view,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Codex CLI Replay requires the replay-agent workspace dependency"
            ) from exc

        workspace = IntentIRReplayWorkspace.create(files)
        composer_installation: Mapping[str, Any] | None = None
        if self.composer_mode != "off":
            try:
                composer_installation = install_episode_composer(
                    workspace.scratch_dir,
                    self.frontend.value,
                    offer_mode=self.composer_mode,
                )
            except Exception:
                workspace.cleanup()
                raise
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        compile_checks: list[dict[str, Any]] = []
        obligation_audits: list[dict[str, Any]] = []
        ttir_inspections: list[dict[str, Any]] = []
        packet_queries: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        composer_host_events: list[dict[str, Any]] = []
        source_text = ""
        formal_ingress = None
        formal_compile = None
        last_checked_good: str | None = None
        startup_context = _replay_startup_context(
            packet, composer_mode=self.composer_mode
        )
        pi_min_ownership = _pi_min_ownership_receipt(
            packet,
            artifact_schema=artifact.wire_schema,
            masked=masked,
        )
        synthetic_compiler = (
            None
            if self.compile_checker is not None or self.frontend is not Frontend.TRITON
            else TritonCompiler(
                timeout=120.0,
                cache_dir=workspace.tmp_dir / "synthetic-replay-compile-cache",
            )
        )
        critical_facts = _packet_critical_facts(packet)
        obligation_audit_required = _replay_obligation_audit_required(
            critical_facts
        )
        obligation_audited_source_digests: set[str] = set()
        ttir_observations: dict[str, dict[str, Any]] = {}
        prompt = (
            _INITIAL_PROMPT
            + (
                _COMPOSER_PREFERRED_PROMPT
                if self.composer_mode == "preferred"
                else _COMPOSER_AVAILABLE_PROMPT
                if composer_installation is not None
                else ""
            )
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

        def current_composer_evidence() -> Mapping[str, Any]:
            if composer_installation is None:
                return {
                    "protocol": REPLAY_COMPOSER_PROTOCOL,
                    "status": "disabled",
                    "installed": False,
                    "builder_installed": False,
                    "offer_mode": "off",
                    "used": False,
                    "selected_path": "raw_escape",
                }
            evidence = dict(
                load_episode_composer_evidence(
                    workspace.scratch_dir, workspace.replay_path
                )
            )
            evidence["host_events"] = list(composer_host_events)
            return evidence

        def checkpoint_current_composer(expected_source: str) -> bool:
            if composer_installation is None:
                return False
            try:
                return checkpoint_episode_composer(
                    workspace.scratch_dir,
                    workspace.replay_path,
                    expected_source=expected_source,
                )
            except Exception as exc:
                composer_host_events.append(
                    {
                        "action": "checkpoint",
                        "status": "unavailable",
                        "diagnostic_class": type(exc).__name__,
                    }
                )
                return False

        def restore_current_composer(expected_source: str) -> bool:
            if composer_installation is None:
                return False
            try:
                return restore_episode_composer_checkpoint(
                    workspace.scratch_dir,
                    workspace.replay_path,
                    expected_source=expected_source,
                )
            except Exception as exc:
                composer_host_events.append(
                    {
                        "action": "rollback",
                        "status": "unavailable",
                        "diagnostic_class": type(exc).__name__,
                    }
                )
                return False

        def attach_composer_map(
            response: Mapping[str, Any], current: str | None
        ) -> dict[str, Any]:
            result = dict(response)
            diagnostics = result.get("diagnostics")
            if (
                composer_installation is not None
                and current is not None
                and isinstance(diagnostics, list)
            ):
                mapped = composer_diagnostic_map(
                    workspace.scratch_dir,
                    current,
                    [item for item in diagnostics if isinstance(item, Mapping)],
                )
                if mapped:
                    result["composer_diagnostic_map"] = mapped
            return result

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
            return attach_composer_map(response, current)

        def replay_compile_tool() -> Mapping[str, Any]:
            nonlocal last_checked_good
            current = workspace.read_replay()
            current_digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
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
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            response = attach_composer_map(checked.to_check_response(), current)
            if checked.status == "pass":
                # The checkpoint is the exact Replay-LLM-authored text that
                # passed both source-blind construction checks in this fresh
                # episode.  Fixed code may restore only these bytes.
                last_checked_good = current
                checkpoint_current_composer(current)
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
                    "program_axis_count": len(view["program_axes"]),
                    "route_extent_transform_count": len(
                        view["route_extent_transforms"]
                    ),
                    "participant_mapping_count": len(
                        view["participant_mappings"]
                    ),
                }
            )
            return view

        def submit_replay_tool() -> Mapping[str, Any]:
            """Run both construction checks as one exact-source transaction."""

            ingress = dict(replay_check_tool())
            if ingress.get("status") != "pass":
                return {
                    "protocol": "intentir.codex-cli.replay-submit.v1",
                    "status": "fail",
                    "phase": "ingress",
                    "ingress": ingress,
                    "compile": None,
                    "authority": "diagnostic-only; no Replay source is generated or repaired",
                }
            compiled = dict(replay_compile_tool())
            return {
                "protocol": "intentir.codex-cli.replay-submit.v1",
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
                "replay_obligation_audit": replay_obligation_audit_tool,
                "submit_replay": submit_replay_tool,
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
                check = attach_composer_map(
                    formal_ingress.to_check_response(), source_text
                )
            checks.append(
                {
                    "trigger": trigger,
                    "status": check["status"],
                    "diagnostics": [dict(item) for item in check["diagnostics"]],
                    "source_sha256": (
                        hashlib.sha256(source_text.encode("utf-8")).hexdigest()
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
            source_digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
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
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            compile_check = attach_composer_map(
                formal_compile.to_check_response(), source_text
            )
            compile_checks.append(
                {
                    "trigger": trigger,
                    "status": compile_check["status"],
                    "diagnostics": [
                        dict(item) for item in compile_check["diagnostics"]
                    ],
                    "source_sha256": source_digest,
                }
            )
            if formal_compile.status == "pass":
                checkpoint_current_composer(source_text)
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
                    if passed:
                        timeout_event["recovery"] = "current_exact_replay"
                        break
                    if last_checked_good is not None:
                        restored_composer = restore_current_composer(
                            last_checked_good
                        )
                        if not restored_composer:
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
                        episode_limit
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
                "replay_obligation_audits": obligation_audits,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "timeout_events": timeout_events,
                "syntax_composer": current_composer_evidence(),
                "pi_min_ownership": pi_min_ownership,
                "tool_bridge": bridge.evidence,
                "formal_ingress_recheck": True,
                "isolation": {
                    "source_blind": True,
                    "input": "read-only",
                    "output_work_tmp": "write-only-scope",
                    "tool_network": "disabled",
                    "provider_secret_in_tool_env": False,
                    "composer_reads_packet": False,
                    "composer_episode_local": True,
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
                    "syntax_composer": current_composer_evidence(),
                    "pi_min_ownership": pi_min_ownership,
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
                "replay_obligation_audits": obligation_audits,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "timeout_events": timeout_events,
                "syntax_composer": current_composer_evidence(),
                "pi_min_ownership": pi_min_ownership,
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
