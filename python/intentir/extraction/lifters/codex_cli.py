"""Codex CLI Lift backend with external SCRC verification boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from minisweagent.config.intentir_context_query import (
    _candidate_control_value,
    _feedback_category_value,
    _harness_abi_value,
)

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIError,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.scrc_runner import CandidateLiftError

from .miniswe import MiniSWELiftAdapter, _input_files, _trajectory_bytes
from .compiler_observation import observe_anonymous_source_ttir
from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCProvenance,
    SCRCReliftRecord,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)


CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v13"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v71"
CODEX_CLI_TURN_TIMEOUT_SECONDS = 900.0

_EAGER_DIRECT_FEEDBACK_CATEGORIES = frozenset(
    {"interface", "reduction", "predicate_effect", "op_multiset"}
)

_INITIAL_PROMPT = """You are the sole source-aware IntentIR Lift owner for this branch.
The outer runtime exposes only this isolated workspace. Work only under /workspace:
input/ is immutable; work/, output/, and tmp/ are writable. Never inspect any
other path and never use network tools.

This is a deliberately small authoring workspace, not a repository checkout.
Read input/source.py, input/harness_view.json, and the single closed
input/wire_contract.json. It contains the concise semantic ownership rules and
the exact annotation-free artifact.v11 JSON Schema. Do not search for another
schema, tutorial, example, preflight, or operation catalog: none is authoritative
here. Author every field of one complete output/candidate.json containing exactly
J, C, and pi. You may keep proof-irrelevant notes or a model-authored build script
under work/.

Use input/source_ttir.json only when it helps resolve one concrete Source control,
time, def-use, effect, or materialization uncertainty. Read only the relevant
local TTIR fragment; do not load the complete observer fingerprint as a default
phase. This observation describes the same anonymous Source specialization. It
never prescribes or constructs J/C/pi.

Treat Harness `shape` as one valuation: map every `shape_contract` h* to a
distinct/repeated symbolic J dimension as declared, preserve scalar and bound
constexpr shape-coordinate equality in authored J, and keep integer contract axes
literal. Trace predicated loads to every sum/max/min consumer and author each
required per-consumer outside-domain relation; fills are not a substitute.
For every Source index, domain, and predicate expression already representable
by the closed ContractExpr grammar, preserve its integer operator tree (`//`,
`%`, `ceil_div`, `+`, `-`, `*`) rather than replacing it with an algebraically
equivalent formula; Gate1 observes the resulting realization topology.
Before CandidateCheck, reconcile every loop, if, and return in Source with the
complete candidate. Decide from Source and
the selected specialization whether each item belongs to logical J control,
pi-owned physical time/control, C-owned compile-time choice, or has no
compiler-visible event. Do not treat a loop region, a large route set, or syntax
alone as evidence that the corresponding control/time obligation was authored;
and do not invent an obligation merely because it appears in syntax. If you query
the syntax-only outline, treat each `lexically_preexisting_writes` row only as a
checklist: inspect the Source
and decide every real loop carrier yourself, then author the complete carrier
set and frontend tuple order. For each selected reachable helper, preserve the
literal Source statement order of writes in J scope.effects and pi transfer/order
topology; J node declaration order is not a substitute for effect order.
After the first meaningful complete edit run:

python .agents/intentir_agent_tool.py candidate_check

Use `lift_harness_abi`, or
`lift_candidate_control overview|outputs|operands|boundary|pi`, only when a focused
self-read of your current candidate would resolve a named uncertainty; it is not
a mandatory pre-check phase. These tools only project exact Harness facts or
your current authored candidate; they never infer or edit J/C/pi.

This invokes the same diagnostic-only CandidateCheck used by the host over your
exact current bytes. It never edits or generates a field. Repair the current
complete file yourself until it reports PASS, then end this Codex turn. The host
independently reruns CandidateCheck after the turn and may resume this same
thread with closed diagnostics. Do not claim TypeCheck, Replay, Gate1, Gate2, or
certification. Do not create any other file under output/.
"""


_COMPACT_WIRE_RULES = (
    "The only accepted envelope is exactly schema,j,c,pi with schema="
    "intentir.scrc.artifact.v11. Lift authors every required field, including "
    "required empty arrays; fixed code never fills, renames, reorders, or infers "
    "artifact content.",
    "J=A+B is the sole logical semantic object. J owns typed logical values and "
    "state, dataflow/control, domains, predicates/fills, logical indices, effects, "
    "aliases, output mappings, result-relation classes, logical signature, scopes, "
    "storage identity, and value/dimension relations.",
    "C owns only implementation-selection slots and their values. A Source value, "
    "shape, predicate, effect, or operation never moves to C merely because it is "
    "compile-time in the selected specialization.",
    "pi is a realization witness for the authored J+C only. It may preserve regions, "
    "instances, routes, route-local extent transforms, order, sharing, refinements, "
    "logical-loop realization, physical-only iteration/carriers, and terminal "
    "topology; it must not introduce a logical op, predicate, shape, dtype, value, "
    "bound formula, or frontend spelling.",
    "Use opaque declaration-ordered handles only. n* maps to an exact structural "
    "IntentOp path; v* is an SSA result. pi route/order/sharing endpoints are i* "
    "instances, never raw J/C handles.",
    "Harness concrete tensor shapes are valuations, not permission to literalize a "
    "parametric Source dimension. Preserve every declared h* shape contract as x*, "
    "author value_dimension_relations when one integral rank-0 logical value equals "
    "an extent, and author every Source-grounded dimension_relation explicitly.",
    "Preserve Source loops, branches, returns, carries, output multiplicity, access "
    "identity, and literal effect order in J and the corresponding realization "
    "topology in pi. Node declaration order is not effect order.",
    "A non-null access fill requires a predicated load. Bind ambiguous access views "
    "to their actual node_ref. Predicated sum/max/min consumers require their exact "
    "consumer_extensions relation; computed access guards require an authored "
    "access_guard_relation and explicit Layer-A select/state semantics where needed.",
    "Choose each output result-relation class from the authored floating dependency "
    "graph. Concrete tolerance values are external and never appear in J.",
    "CandidateCheck is source-free parse/Decode/dependent-TypeCheck diagnostics over "
    "the exact current bytes. It neither judges Source semantic completeness nor "
    "repairs or certifies a candidate; Gate1 feedback is the semantic counterexample.",
)


def _strip_schema_annotations(value: Any) -> Any:
    """Keep the exact structural schema while removing tutorial-sized prose."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip_schema_annotations(item)
            for key, item in value.items()
            if key not in {"description", "title", "examples", "default"}
        }
    if isinstance(value, list):
        return [_strip_schema_annotations(item) for item in value]
    return value


def _codex_input_files(
    request: Any,
    *,
    source_ttir_observation: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Project one minimal Codex authoring surface from the shared Lift packet."""

    complete = _input_files(
        request,
        source_ttir_observation=source_ttir_observation,
    )
    schema = json.loads(complete["schema/artifact.schema.json"])
    wire_contract = {
        "protocol": "intentir.codex-lift-wire-contract.v1",
        "authority": (
            "source-neutral authoring reference only; the host independently "
            "runs the full schema, Decode, and dependent TypeCheck"
        ),
        "semantic_rules": list(_COMPACT_WIRE_RULES),
        "wire_schema": _strip_schema_annotations(schema),
    }
    files = {
        "source.py": complete["source.py"],
        "harness_view.json": complete["harness_view.json"],
        "feedback.json": complete["feedback.json"],
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "wire_contract.json": json.dumps(
            wire_contract,
            indent=0,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        "candidate_check.json": "null\n",
    }
    if "source_ttir.json" in complete:
        files["source_ttir.json"] = complete["source_ttir.json"]
    return files

_EXTERNAL_CONTINUATION_PROMPT = """The same branch-local source-aware Lift task continues in the same workspace and thread.
The final `<source-aware-startup>` block contains the exact bounded feedback
frontier and eagerly expands every available direct interface, reduction,
predicate/effect, and op-multiset category. Inspect the current
output/candidate.json and resolve those direct categories together before paging
recursive dataflow witnesses. A compiler-visible representation-only operation
in op-multiset feedback is a reason to recheck the existing pi representation
refinement grammar and Source realization, not permission for fixed code to add
a field or for you to copy a non-semantic compiler operation into J. Use
`lift_feedback_category CATEGORY` only for one named omitted category. Use
the Source control-flow outline again when the frontier names a loop, branch,
terminal, or carrier difference; reconcile the complete J/C/pi ownership rather
than only adding routes around the existing flat graph. When an effect-order or
terminal-fork witness appears, reread the selected reachable helper bodies and
compare their literal memory-call sequence with each branch scope.effects and pi
transfer/order edges; do not infer effect order from J node declaration order. Use
the bounded exact category/path query only when the frontier names an unresolved
section. Repair the complete model-authored (J,C,pi) against the closed external
feedback and Source, run `python .agents/intentir_agent_tool.py candidate_check`,
and end only after it
passes for the complete current output/candidate.json. This is not a new writer
and not a patch-only artifact. The host alone will rerun CandidateCheck and the
formal SCRC gates.
"""

_CANDIDATE_CHECK_CONTINUATION_PROMPT = """The host CandidateCheck rejected the current exact candidate bytes.
Read input/candidate_check.json, then repair output/candidate.json yourself in
this same thread. Make coordinated J/C/pi changes when required; do not merely
silence the first path. Rerun `python .agents/intentir_agent_tool.py candidate_check`
and end only after the
complete candidate passes. CandidateCheck is
diagnostic-only and has no certification authority.
"""


def _source_aware_startup_context(workspace: Any, *, fresh_owner: bool) -> dict[str, Any]:
    """Return only paths on first use, or bounded closed feedback later."""

    if fresh_owner:
        return {
            "protocol": "intentir.codex-lift-startup-view.v1",
            "fresh_owner": True,
            "paths": {
                "source": "input/source.py",
                "harness": "input/harness_view.json",
                "source_ttir": "input/source_ttir.json",
                "wire_contract": "input/wire_contract.json",
                "candidate": "output/candidate.json",
            },
            "authority": (
                "path manifest only; exact bytes remain in mounted files and no "
                "logical_ref, J/C/pi field, semantic classification, or repair "
                "is generated"
            ),
        }

    frontier = json.loads(
        (workspace.input_dir / "feedback_frontier.json").read_text(
            encoding="utf-8"
        )
    )
    primary: str | None = None
    if isinstance(frontier, Mapping):
        candidate = frontier.get("primary_difference_category")
        if isinstance(candidate, str):
            primary = candidate
        elif isinstance(frontier.get("categories"), list):
            for item in frontier["categories"]:
                if isinstance(item, Mapping) and isinstance(
                    item.get("category"), str
                ):
                    primary = str(item["category"])
                    break
    direct_category_views: dict[str, Any] = {}
    deferred_categories: list[str] = []
    categories = frontier.get("categories") if isinstance(frontier, Mapping) else None
    if isinstance(categories, list):
        for item in categories:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("category"), str
            ):
                continue
            name = str(item["category"])
            if name not in _EAGER_DIRECT_FEEDBACK_CATEGORIES:
                deferred_categories.append(name)
                continue
            try:
                direct_category_views[name] = _feedback_category_value(
                    workspace.input_dir, name
                )
            except (KeyError, TypeError, ValueError):
                # The bounded frontier remains authoritative and visible even if
                # an older feedback envelope cannot serve the focused view.
                continue
    return {
        "protocol": "intentir.codex-lift-continuation-view.v2",
        "fresh_owner": False,
        "feedback_frontier": frontier,
        "primary_category": primary,
        "direct_category_views": direct_category_views,
        "deferred_categories": deferred_categories,
        "candidate_path": "/output/candidate.json",
        "authority": (
            "exact bounded closed-feedback projection only; no semantic "
            "inference, artifact field, or repair is generated"
        ),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any] | None) -> None:
    text = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        if value is not None
        else "null\n"
    )
    _atomic_write_text(path, text)


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return totals


@dataclass
class CodexCLILiftAdapter(MiniSWELiftAdapter):
    """Persistent source-aware Codex owner across formal Gate1 continuations."""

    session_factory: Callable[..., Any] = field(
        default=CodexCLISession,
        repr=False,
    )
    repair_effort: str | None = None
    _codex_session: Any = field(default=None, init=False, repr=False)
    _source_ttir_observation: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def close(self) -> None:
        session = self._codex_session
        self._codex_session = None
        if session is not None:
            session.close()
        self._source_ttir_observation = None
        super().close()

    def _remaining_seconds(self) -> float:
        if self.session_wall_time_limit_seconds <= 0:
            raise ValueError("Lift owner session wall-time limit must be positive")
        return self.session_wall_time_limit_seconds - (
            self._session_active_elapsed_ms / 1000.0
        )

    def _terminal_time_error(self, round_index: int) -> Exception:
        diagnostic = {
            "diagnostic": "branch-local Codex Lift owner wall-time exhausted",
            "failure_class": "lift_owner_time_exhausted",
            "session_id": self._session_id,
            "active_elapsed_ms": self._session_active_elapsed_ms,
        }
        self._records.append(
            SCRCReliftRecord(
                round_index,
                diagnostic=diagnostic["diagnostic"],
                backend_evidence={
                    "backend": CODEX_CLI_LIFT_BACKEND,
                    "runtime": CODEX_CLI_AGENT_RUNTIME,
                    "owner_session_id": self._session_id,
                    "thread_id": getattr(self._codex_session, "thread_id", None),
                    "session_active_elapsed_ms": self._session_active_elapsed_ms,
                },
                continuation_candidate=(
                    self._read_continuation_candidate(self._session_workspace)
                    if self._session_workspace is not None
                    else None
                ),
            )
        )
        return CandidateLiftError(
            diagnostic,
            physical_calls=0,
            terminal=True,
        )

    @staticmethod
    def _candidate_missing_check(diagnostic: str) -> dict[str, Any]:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_unavailable",
                    "path": "<candidate>",
                    "observed_refs_or_types": [diagnostic[:2048]],
                }
            ],
        }

    def lift(self, request: Any, feedback: Any) -> Any:
        if self.workflow != "single":
            raise ValueError("Codex CLI Lift supports only the single-owner workflow")
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Codex CLI Lift requires SourceProgram.text")
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        if self._session_source_sha256 is None:
            self._session_source_sha256 = source_sha256
        elif self._session_source_sha256 != source_sha256:
            raise ValueError("one Codex Lift owner cannot cross Source programs")

        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=self._relift_context(feedback),
            source_symbol=self.source_symbol,
        )
        candidate_check_context = scrc_candidate_check_context(prepared)
        round_index = len(self._records) + 1
        if self._source_ttir_observation is None:
            prepared_packet = json.loads(prepared.user_input)
            anonymous_source = prepared_packet.get("anonymous_source")
            anonymous_view = prepared_packet.get("harness_compiler_view")
            self._source_ttir_observation = observe_anonymous_source_ttir(
                anonymous_source,
                anonymous_view,
            )
        files = _codex_input_files(
            prepared,
            source_ttir_observation=self._source_ttir_observation,
        )

        try:
            fresh_workspace = self._create_or_refresh_session_workspace(
                files,
                mutable_paths=frozenset(
                    {
                        "feedback.json",
                        "feedback_frontier.json",
                        "candidate_check.json",
                    }
                ),
            )
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "Codex CLI Lift requires the lift-agent dependencies",
                evidence={"backend": CODEX_CLI_LIFT_BACKEND},
            ) from exc
        workspace = self._session_workspace
        fresh_owner = self._codex_session is None
        call_turns: list[CodexCLITurn] = []
        check_results: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        candidate_text: str | None = None
        last_checked_good: str | None = None
        timed_out_calls = 0
        timed_out_elapsed_ms = 0.0

        def check_current_candidate(*, trigger: str) -> dict[str, Any]:
            nonlocal candidate_text, last_checked_good
            try:
                candidate_text = workspace.read_candidate()
            except Exception as exc:
                candidate_text = None
                check = self._candidate_missing_check(
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                check = check_scrc_lift_candidate(
                    candidate_text,
                    candidate_check_context,
                )
            check_results.append(
                {
                    "trigger": trigger,
                    "status": check["status"],
                    "diagnostics": [
                        dict(item) for item in check["diagnostics"]
                    ],
                    "candidate_sha256": (
                        hashlib.sha256(candidate_text.encode()).hexdigest()
                        if candidate_text is not None
                        else None
                    ),
                }
            )
            if (
                trigger == "agent_tool"
                and check["status"] == "pass"
                and candidate_text is not None
            ):
                # Exact LLM-authored bytes only.  This is a transactional
                # checkpoint for this one Lift invocation, never a previous
                # formal-round parent and never a synthesized artifact.
                last_checked_good = candidate_text
            return check

        def candidate_check_tool() -> Mapping[str, Any]:
            return check_current_candidate(trigger="agent_tool")

        def harness_abi_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            try:
                if len(arguments) > 2:
                    raise ValueError("lift_harness_abi accepts [START [COUNT]]")
                start = int(arguments[0]) if arguments else 1
                count = int(arguments[1]) if len(arguments) >= 2 else 24
                view = _harness_abi_value(workspace.input_dir, start, count)
                if view.get("parameter_next_command") is not None:
                    next_start = start + len(view["parameters"])
                    view["parameter_next_command"] = (
                        "python .agents/intentir_agent_tool.py "
                        f"lift_harness_abi {next_start} {count}"
                    )
            except (TypeError, ValueError) as exc:
                return {
                    "protocol": "intentir.lift-harness-abi-view.v1",
                    "status": "fail",
                    "diagnostic": str(exc),
                }
            return {"status": "pass", **view}

        def candidate_control_tool(
            arguments: tuple[str, ...],
        ) -> Mapping[str, Any]:
            try:
                if len(arguments) > 3:
                    raise ValueError(
                        "lift_candidate_control accepts SECTION [START [COUNT]]"
                    )
                section = arguments[0] if arguments else "overview"
                if section not in {
                    "overview", "outputs", "operands", "boundary", "pi"
                }:
                    raise ValueError(
                        "SECTION must be overview, outputs, operands, boundary, or pi"
                    )
                start = int(arguments[1]) if len(arguments) >= 2 else 1
                count = int(arguments[2]) if len(arguments) >= 3 else 20
                view = _candidate_control_value(
                    workspace.output_dir, section, start, count
                )
                if section == "overview" and view.get("candidate_found"):
                    view["next_queries"] = [
                        "python .agents/intentir_agent_tool.py "
                        f"lift_candidate_control {name} 1 20"
                        for name in ("outputs", "operands", "boundary", "pi")
                    ]
                elif view.get("next_command") is not None:
                    next_start = start + int(view.get("returned", 0))
                    view["next_command"] = (
                        "python .agents/intentir_agent_tool.py "
                        f"lift_candidate_control {section} {next_start} {count}"
                    )
            except (TypeError, ValueError) as exc:
                return {
                    "protocol": "intentir.lift-candidate-control-view.v2",
                    "status": "fail",
                    "diagnostic": str(exc),
                }
            return {"status": "pass", **view}

        def feedback_category_tool(
            arguments: tuple[str, ...],
        ) -> Mapping[str, Any]:
            try:
                if len(arguments) != 1:
                    raise ValueError("lift_feedback_category accepts exactly CATEGORY")
                view = _feedback_category_value(workspace.input_dir, arguments[0])
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "protocol": "intentir.lift-feedback-category-view.v1",
                    "status": "fail",
                    "diagnostic": str(exc),
                }
            return {
                "protocol": "intentir.lift-feedback-category-view.v1",
                "status": "pass",
                "category": view,
                "authority": (
                    "exact bounded projection of closed external feedback; "
                    "diagnostic-only, no semantic inference or artifact repair"
                ),
            }

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "feedback_frontier": lambda: {
                    "protocol": "intentir.codex-feedback-frontier.v1",
                    "status": "pass",
                    "frontier": json.loads(
                        (workspace.input_dir / "feedback_frontier.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                },
                "candidate_check": candidate_check_tool,
            },
            argument_handlers={
                "lift_harness_abi": harness_abi_tool,
                "lift_candidate_control": candidate_control_tool,
                "lift_feedback_category": feedback_category_tool,
            },
        )
        try:
            bridge.start()
            if self._codex_session is None:
                self._codex_session = self.session_factory(
                    provider=provider,
                    input_dir=workspace.input_dir,
                    output_dir=workspace.output_dir,
                    scratch_dir=workspace.scratch_dir,
                    tmp_dir=workspace.tmp_dir,
                    model=provider.model,
                    effort=effective_effort,
                )
            startup_context = _source_aware_startup_context(
                workspace, fresh_owner=fresh_owner
            )
            prompt = (
                (_INITIAL_PROMPT if fresh_owner else _EXTERNAL_CONTINUATION_PROMPT)
                + "\n\n<source-aware-startup>\n"
                + json.dumps(
                    startup_context,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n</source-aware-startup>\n"
            )
            selected_effort = effective_effort
            while True:
                remaining = self._remaining_seconds()
                if remaining <= 0:
                    raise self._terminal_time_error(round_index)
                turn_timeout = min(
                    remaining,
                    self.model_timeout_seconds or CODEX_CLI_TURN_TIMEOUT_SECONDS,
                )
                try:
                    turn = self._codex_session.run(
                        prompt,
                        timeout_seconds=turn_timeout,
                        effort=selected_effort,
                        stop_requested=self.stop_requested,
                    )
                except CodexCLICancelled as exc:
                    self._session_active_elapsed_ms += exc.elapsed_ms
                    self._session_model_turns += 1
                    self._session_physical_provider_calls += exc.physical_calls
                    cancellation_evidence = {
                        "backend": CODEX_CLI_LIFT_BACKEND,
                        "runtime": CODEX_CLI_AGENT_RUNTIME,
                        "owner_session_id": self._session_id,
                        "thread_id": getattr(
                            self._codex_session, "thread_id", None
                        ),
                        "winner_branch_id": exc.winner_branch_id,
                        "cancelled_elapsed_ms": exc.elapsed_ms,
                        "cancelled_stdout_sha256": exc.stdout_sha256,
                        "cancelled_stderr_sha256": exc.stderr_sha256,
                        "turns": [item.to_evidence() for item in call_turns],
                        "candidate_check": list(check_results),
                        "tool_bridge": bridge.evidence,
                    }
                    self._records.append(
                        SCRCReliftRecord(
                            round_index,
                            diagnostic=str(exc),
                            backend_evidence=cancellation_evidence,
                            continuation_candidate=(
                                self._read_continuation_candidate(workspace)
                            ),
                        )
                    )
                    raise CandidateLiftError(
                        {
                            "diagnostic": str(exc),
                            "failure_class": "candidate_superseded",
                            "winner_branch_id": exc.winner_branch_id,
                        },
                        physical_calls=exc.physical_calls,
                    ) from exc
                except CodexCLIError as exc:
                    if "wall-time limit" in str(exc):
                        elapsed = turn_timeout * 1000.0
                        self._session_active_elapsed_ms += elapsed
                        self._session_model_turns += 1
                        self._session_physical_provider_calls += 1
                        timed_out_calls += 1
                        timed_out_elapsed_ms += elapsed
                        timeout_event = {
                            "turn_timeout_seconds": turn_timeout,
                            "thread_id": getattr(
                                self._codex_session, "thread_id", None
                            ),
                            "recovery": None,
                        }
                        timeout_events.append(timeout_event)

                        check = check_current_candidate(
                            trigger="host_timeout_recheck"
                        )
                        if (
                            check["status"] == "pass"
                            and candidate_text is not None
                        ):
                            timeout_event["recovery"] = "current_exact_candidate"
                            break
                        if last_checked_good is not None:
                            _atomic_write_text(
                                workspace.candidate_path, last_checked_good
                            )
                            check = check_current_candidate(
                                trigger="host_timeout_checkpoint_restore"
                            )
                            if (
                                check["status"] == "pass"
                                and candidate_text is not None
                            ):
                                timeout_event["recovery"] = (
                                    "same_invocation_checked_checkpoint"
                                )
                                break

                        remaining_after_timeout = self._remaining_seconds()
                        if remaining_after_timeout <= 0:
                            timeout_event["recovery"] = "owner_time_exhausted"
                            raise self._terminal_time_error(round_index) from exc
                        if getattr(self._codex_session, "thread_id", None) is None:
                            timeout_event["recovery"] = "no_resumable_thread"
                            raise CodexCLIError(
                                "initial Codex Lift turn timed out without a "
                                "resumable thread or checked candidate"
                            ) from exc
                        _atomic_write_json(
                            workspace.input_dir / "candidate_check.json",
                            check,
                        )
                        timeout_event["recovery"] = "same_thread_continuation"
                        prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                        selected_effort = self.repair_effort or effective_effort
                        continue
                    raise
                call_turns.append(turn)
                self._session_active_elapsed_ms += turn.elapsed_ms
                self._session_model_turns += 1
                self._session_physical_provider_calls += turn.model_segments

                check = check_current_candidate(trigger="host_post_turn")
                if check["status"] == "pass" and candidate_text is not None:
                    break
                _atomic_write_json(
                    workspace.input_dir / "candidate_check.json",
                    check,
                )
                prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                selected_effort = self.repair_effort or effective_effort
        except Exception as exc:
            if isinstance(exc, CandidateLiftError):
                raise
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            evidence = {
                "backend": CODEX_CLI_LIFT_BACKEND,
                "runtime": CODEX_CLI_AGENT_RUNTIME,
                "owner_session_id": self._session_id,
                "thread_id": getattr(self._codex_session, "thread_id", None),
                "fresh_session": fresh_owner,
                "turns": [turn.to_evidence() for turn in call_turns],
                "candidate_check": list(check_results),
                "timeout_events": list(timeout_events),
                "tool_bridge": bridge.evidence,
            }
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence=evidence,
                    continuation_candidate=self._read_continuation_candidate(
                        workspace
                    ),
                )
            )
            raise SCRCInfrastructureError(
                f"Codex CLI Lift infrastructure failed: {type(exc).__name__}: {exc}",
                evidence={
                    key: value
                    for key, value in evidence.items()
                    if key != "turns"
                },
            ) from exc
        finally:
            bridge.close()

        assert candidate_text is not None
        artifact = decode_scrc_lift_artifact(candidate_text, prepared)
        usage = _aggregate_usage(call_turns)
        trajectory = {
            "protocol": CODEX_CLI_AGENT_RUNTIME,
            "backend": CODEX_CLI_LIFT_BACKEND,
            "thread_id": self._codex_session.thread_id,
            "turns": [turn.to_evidence() for turn in call_turns],
            "candidate_check": list(check_results),
            "timeout_events": list(timeout_events),
            "tool_bridge": bridge.evidence,
            "source_ttir_observation": {
                key: value
                for key, value in self._source_ttir_observation.items()
                if key not in {"ttir", "fingerprint"}
            },
            "isolation": {
                "outer": "bubblewrap-host-fs-v1",
                "input": "read-only",
                "output_work_tmp": "write-only-scope",
                "tool_network": "disabled",
                "provider_secret_in_tool_env": False,
            },
        }
        raw_trajectory = _trajectory_bytes(trajectory)
        elapsed_ms = (
            sum(turn.elapsed_ms for turn in call_turns) + timed_out_elapsed_ms
        )
        physical_calls = (
            sum(turn.model_segments for turn in call_turns) + timed_out_calls
        )
        provenance = SCRCProvenance(
            model=provider.model,
            prompt_sha256=prepared.prompt_sha256,
            response_id=self._codex_session.thread_id,
            response_model=provider.model,
            usage=usage,
            response_sha256=hashlib.sha256(candidate_text.encode()).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=candidate_text,
            raw_response=raw_trajectory,
            attempts=len(call_turns) + timed_out_calls,
            elapsed_ms=elapsed_ms,
            backend=CODEX_CLI_LIFT_BACKEND,
            termination_status="CandidateCheckPass",
            trajectory=trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
        )
        backend_evidence = {
            "backend": CODEX_CLI_LIFT_BACKEND,
            "knowledge_profile": CODEX_CLI_LIFT_KNOWLEDGE,
            "runtime": CODEX_CLI_AGENT_RUNTIME,
            "fresh_session": fresh_owner,
            "fresh_workspace": fresh_workspace,
            "persistent_owner": True,
            "persistent_workspace": True,
            "owner_session_id": self._session_id,
            "thread_id": self._codex_session.thread_id,
            "cli_turns": len(call_turns) + timed_out_calls,
            "model_segments_lower_bound": physical_calls,
            "tool_actions": sum(turn.tool_actions for turn in call_turns),
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "candidate_check": list(check_results),
            "timeout_events": list(timeout_events),
            "tool_bridge": bridge.evidence,
            "trajectory": trajectory,
        }
        self._records.append(
            SCRCReliftRecord(
                round_index,
                extraction=extraction,
                backend_evidence=backend_evidence,
            )
        )

        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            artifact,
            physical_calls=physical_calls,
            metadata={
                key: value
                for key, value in backend_evidence.items()
                if key != "trajectory"
            }
            | {
                "model": provider.model,
                "artifact_schema": artifact.wire_schema,
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_cli_turns": self._session_model_turns,
                "session_model_segments_lower_bound": (
                    self._session_physical_provider_calls
                ),
                "physical_provider_call_observability": (
                    "Codex CLI exposes model segments, not raw HTTP request count"
                ),
                "workflow": "single",
            },
        )


__all__ = [
    "CODEX_CLI_LIFT_BACKEND",
    "CODEX_CLI_LIFT_KNOWLEDGE",
    "CodexCLILiftAdapter",
]
