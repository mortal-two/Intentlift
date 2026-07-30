"""Codex CLI Lift backend with external SCRC verification boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from minisweagent.config.intentir_schema_query import schema_sections

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
from .compiler_observation import (
    SOURCE_TTIR_RECONCILE_PROTOCOL,
    observe_anonymous_source_ttir,
    reconcile_candidate_source_ttir,
)
from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCProvenance,
    SCRCReliftRecord,
    SourceIdentifierPrivacyError,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
from .prompt import SCHEMA as _INTENTIR_OPERATION_REFERENCE
from .scrc_schema import scrc_lift_core_json_schema


CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v20"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v78"
CODEX_CLI_TURN_TIMEOUT_SECONDS = 900.0

_FEEDBACK_CATEGORY_NAMES = (
    "interface",
    "predicate_effect",
    "op_multiset",
    "dataflow",
    "reduction",
)

_INITIAL_PROMPT = """You are the sole source-aware IntentIR Lift owner for this branch.
Work only under the isolated /workspace; input/ is immutable and output/, work/,
and tmp/ are writable. There is no repository or network.

Act like a coding agent, not a staged JSON generator. Read input/source.py,
input/harness_view.json, and input/README.md; inspect only useful ranges of
input/source.ttir. input/artifact_contract.json is the complete high-density closed
wire reference projected from the host Schema, and input/intent_ops.md gives exact
IntentOp spelling. Query only `.object_shapes` and the leaf `.sections.NAME` values
you actually need; do not pretty-print the whole contract or read the entire op
catalog into the thread. Do not search for another schema, tutorial, or tool source.

You alone author the complete output/candidate.json. J owns all logical semantics;
C owns only implementation choices; pi owns only source-realization relations for
the authored J+C. Fixed code must not choose or fill any field, handle, relation,
operation, shape, or topology. Preserve anonymous ABI order, parametric shape
identities, control/time, effects, and realization structure visible in Source and
the selected anonymous specialization. Use dense declaration-ordered opaque
handles and never copy Source identifiers into the artifact.

Organize your own scratch files or helper edits under work/ however you find
efficient; no host-prescribed planner, fragment layout, ledger, or builder is
required. The final candidate bytes must be entirely your work. Run
`python .agents/intentir_agent_tool.py candidate_check` after a meaningful complete
draft, then run `python .agents/intentir_agent_tool.py source_reconcile`. Repair the
exact diagnostics and finish only when both report PASS. CandidateCheck remains
source-free; source_reconcile only compares your already-authored closed compiled
obligations with the immutable Source TTIR observation. The host independently
reruns Decode and TypeCheck, then starts fresh
source-blind Replay and the formal gates. Do not claim certification.
"""



def _compact_intent_op_reference() -> str:
    """Reuse the existing op vocabulary without exposing its legacy top level."""

    start_marker = "Atomic ops (each is one JSON object in a \"body\" list):"
    end_marker = "Hard rules:"
    start = _INTENTIR_OPERATION_REFERENCE.index(start_marker)
    end = _INTENTIR_OPERATION_REFERENCE.index(end_marker, start)
    return _INTENTIR_OPERATION_REFERENCE[start:end].strip()


def _compact_authoring_schema(value: Any) -> Any:
    """Drop prose only; preserve the exact closed JSON Schema structure."""

    if isinstance(value, Mapping):
        return {
            str(key): _compact_authoring_schema(item)
            for key, item in value.items()
            if key not in {"description", "title", "examples"}
        }
    if isinstance(value, list):
        return [_compact_authoring_schema(item) for item in value]
    return value


def _contract_object_shapes(schema: Mapping[str, Any]) -> dict[str, Any]:
    root = schema["properties"]
    j = root["j"]
    j_properties = j["properties"]
    intent = j_properties["intent"]
    function = intent["properties"]["intent_function"]
    objects = {
        "root": schema,
        "j": j,
        "j.intent": intent,
        "j.intent.intent_function": function,
        "j.boundary": j_properties["boundary"],
        "j.signature": j_properties["signature"],
        "c": root["c"],
        "pi": root["pi"],
    }
    return {
        name: {
            "required": list(item.get("required", [])),
            "fields": list(item.get("properties", {})),
            "additional_properties": item.get("additionalProperties"),
        }
        for name, item in objects.items()
    }


def _observer_diff_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    observer = diagnostics.get("observer_diff")
    if isinstance(observer, Mapping):
        return observer
    unresolved = diagnostics.get("unresolved_gate1")
    nested = (
        unresolved.get("diagnostics")
        if isinstance(unresolved, Mapping)
        else None
    )
    observer = nested.get("observer_diff") if isinstance(nested, Mapping) else None
    return observer if isinstance(observer, Mapping) else None


def _codex_feedback_files(feedback_text: str) -> dict[str, str]:
    """Partition exact closed feedback without exposing one bulk relift object.

    Category names already belong to the Gate1 diagnostic protocol.  This only
    separates bytes by that existing label and removes the duplicate previous
    artifact, which the persistent owner already has in output/candidate.json.
    """

    feedback = json.loads(feedback_text)
    categories: dict[str, Any] = {}
    if isinstance(feedback, Mapping):
        observer = _observer_diff_from_diagnostics(
            feedback.get("diagnostics")
            if isinstance(feedback.get("diagnostics"), Mapping)
            else None
        )
        raw_categories = observer.get("categories") if isinstance(observer, Mapping) else None
        if isinstance(raw_categories, list):
            for category in raw_categories:
                if not isinstance(category, Mapping):
                    continue
                name = category.get("category")
                if isinstance(name, str) and name in _FEEDBACK_CATEGORY_NAMES:
                    categories[name] = dict(category)

    detail: Any = None
    if isinstance(feedback, Mapping):
        detail = {
            str(key): value
            for key, value in feedback.items()
            if key not in {"previous_artifact", "diagnostics"}
        }
        diagnostics = feedback.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            pruned = {
                str(key): value
                for key, value in diagnostics.items()
                if key not in {"observer_diff", "unresolved_gate1"}
            }
            unresolved = diagnostics.get("unresolved_gate1")
            if isinstance(unresolved, Mapping):
                unresolved_copy = dict(unresolved)
                nested = unresolved_copy.get("diagnostics")
                if isinstance(nested, Mapping):
                    unresolved_copy["diagnostics"] = {
                        str(key): value
                        for key, value in nested.items()
                        if key != "observer_diff"
                    }
                pruned["unresolved_gate1"] = unresolved_copy
            if pruned:
                detail["diagnostics"] = pruned

    values: dict[str, Any] = {"feedback_detail.json": detail}
    values.update(
        {
            f"feedback_category_{name}.json": categories.get(name)
            for name in _FEEDBACK_CATEGORY_NAMES
        }
    )
    return {
        name: json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for name, value in values.items()
    }



def _codex_input_files(
    request: Any,
    *,
    source_ttir_observation: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Build a small ordinary coding workspace for the source-aware owner."""

    complete = _input_files(
        request,
        source_ttir_observation=source_ttir_observation,
    )
    schema = scrc_lift_core_json_schema()
    contract = {
        "protocol": "intentir.codex-lift-authoring-contract.v1",
        "authority": (
            "source-neutral exact projection of the authoritative JSON Schema; "
            "no artifact field, handle, semantic choice, or repair is generated"
        ),
        "object_shapes": _contract_object_shapes(schema),
        "sections": {
            name: _compact_authoring_schema(section)
            for name, section in schema_sections(schema).items()
        },
    }
    files = {
        "source.py": complete["source.py"],
        "harness_view.json": complete["harness_view.json"],
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "README.md": (
            "# IntentIR Lift workspace\n\n"
            "Author exactly `output/candidate.json` from the anonymous Source and "
            "Harness. J is the sole logical semantic carrier; C contains only "
            "implementation selections; pi contains only anonymous realization "
            "relations indexed by J+C. Every artifact field and opaque handle is "
            "model-authored. Fixed code only checks and rejects. Harness shapes are "
            "valuations unless an explicit shape contract binds a parametric extent. "
            "Use the exact contract and operation reference locally; inspect raw TTIR "
            "only where it clarifies Source control, def-use, effects, or time.\n\n"
            "Before the first CandidateCheck, enforce these closed language "
            "invariants in your own artifact:\n"
            "- Every required array is present. Each handle family is dense from zero "
            "in declaration order. Tensor handles follow first declaration across "
            "inputs then outputs; an inout output reuses its input handle. Symbolic "
            "x* order is first appearance in tensor shapes.\n"
            "- Signature p* exactly follows anonymous ABI order. logical_ref names a "
            "declared t*/x* or null. A scalar used only for representation/address may "
            "be null; active logical data, predicate, index, shape, or result may not. "
            "Null removes the carrier only from J semantics: if the selected Source "
            "TTIR still uses it for address, control, or realization, pi must author "
            "its occurrence and wiring so source-blind Replay never guesses its role; "
            "if that selected TTIR has no body use, do not invent a pi occurrence.\n"
            "- C subject_ref names an existing J node/domain/access/scope/storage. "
            "axis_ref is present only with subject_ref and must be one of that "
            "subject's actual closed axes. C never owns logical values or operations.\n"
            "- Layer-B domains and access indices are logical, not pointer arithmetic. "
            "Use explicit logical index variables (k0, k1, ...) and author every "
            "zero-based bound that makes an access/predicate closed; x* denotes an "
            "extent and never substitutes for the varying index. An ordered direct "
            "index lists its logical coordinates. General affine indices remain legal "
            "only over authored logical indices/extents. Program ids, tile rounding, "
            "strides, and flattened address construction stay in C/pi realization.\n"
            "- Preserve each Source load/store predicate as J logical validity when it "
            "guards the logical access, including conjuncts that happen to be true for "
            "the current valuation. C tile bounds may realize that predicate but may "
            "not replace, weaken, or invent it.\n"
            "- A predicated load that reaches sum/max/min has one typed, consumer-local "
            "outside-domain identity relation. A numeric-cast authorization anchors "
            "the explicit cast node and uses its operand/result dtypes.\n"
            "- Every authored semantic loop state has one loop-scoped StorageObject "
            "covering state/body-argument/next/result and non-literal init, with "
            "crosses_iteration=true and escapes=true. Physical-only source loops stay "
            "in pi rather than becoming false J recurrence.\n"
            "- Every physical carrier has type-consistent init/state/next/result, "
            "authored init->state and state->next routes, and a route from state to an "
            "observable transfer/update/synchronization instance. Array order is the "
            "Source frontend carrier-tuple order.\n"
            "- Every direct/async transfer is reachable through authored pi routes "
            "from the matching pointer/tensor signature carrier. Route slots match "
            "the target J operand type; sharing consumers each have a direct route.\n"
            "- Canonical outputs, returns/output_mapping, effects, result-relation "
            "class, scope captures/effect order, and all J/C/pi cross-references are "
            "complete and mutually consistent. CandidateCheck only diagnoses these "
            "facts; it never supplies them.\n"
            "On a relift, read feedback_frontier.json first. Exact bulk Gate1 "
            "feedback is not mounted: open only the already-labelled "
            "feedback_category_NAME.json required by the current hypothesis; "
            "feedback_detail.json contains non-observer diagnostics and never "
            "duplicates the previous artifact.\n"
            "After CandidateCheck passes, run the separate source_reconcile tool. It "
            "only compares occurrence obligations already authored by your pi with "
            "the immutable Source TTIR observer; it never supplies a field or repair.\n"
        ),
        "artifact_contract.json": json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n",
        "intent_ops.md": _compact_intent_op_reference() + "\n",
        "candidate_check.json": "null\n",
        "source_reconcile.json": "null\n",
    }
    files.update(_codex_feedback_files(complete["feedback.json"]))
    if "source_ttir.json" in complete:
        observation = json.loads(complete["source_ttir.json"])
        ttir = observation.get("ttir") if isinstance(observation, Mapping) else None
        if isinstance(ttir, str) and ttir:
            files["source.ttir"] = ttir.rstrip() + "\n"
    return files

_EXTERNAL_CONTINUATION_PROMPT = """The same branch-local source-aware Lift task continues in the same workspace and thread.
Read input/feedback_frontier.json first. Open only a named
input/feedback_category_NAME.json when its bounded frontier is insufficient;
input/feedback_detail.json contains exact non-observer diagnostics. The previous
artifact already exists at output/candidate.json and is never duplicated in input.
Inspect Source/TTIR only where needed, then revise the complete model-authored
J/C/pi in place. Treat feedback as a compiler counterexample, not as a semantic patch.
Run `python .agents/intentir_agent_tool.py candidate_check` and finish only after
the current exact candidate passes. The host alone reruns the checks and gates.
"""

_CANDIDATE_CHECK_CONTINUATION_PROMPT = """The host submission preflight rejected the current exact candidate bytes.
Read input/candidate_check.json and repair output/candidate.json in this same
thread. Make coordinated J/C/pi changes when required. Rerun
`python .agents/intentir_agent_tool.py candidate_check` and finish only after the
complete candidate passes. CandidateCheck is diagnostic-only.
"""

_SOURCE_RECONCILE_CONTINUATION_PROMPT = """The source-aware occurrence preflight rejected the current exact candidate bytes.
Read input/source_reconcile.json and revise only the model-authored J/C/pi facts whose
closed compiled obligations disagree with the immutable Source TTIR observation.
This diagnostic supplies no artifact field or repair. Rerun candidate_check and
source_reconcile, and finish only after both pass. Formal TypeCheck, Replay, and
Gate1 remain external.
"""


def _source_aware_startup_context(workspace: Any, *, fresh_owner: bool) -> dict[str, Any]:
    """Return a path-only receipt; exact task state stays in ordinary files."""

    return {
        "protocol": "intentir.codex-lift-startup-view.v2",
        "fresh_owner": fresh_owner,
        "paths": {
            "source": "input/source.py",
            "harness": "input/harness_view.json",
            "source_ttir": "input/source.ttir",
            "artifact_contract": "input/artifact_contract.json",
            "intent_ops": "input/intent_ops.md",
            "feedback_frontier": "input/feedback_frontier.json",
            "feedback_detail": "input/feedback_detail.json",
            "feedback_categories": [
                f"input/feedback_category_{name}.json"
                for name in _FEEDBACK_CATEGORY_NAMES
            ],
            "candidate_check": "input/candidate_check.json",
            "source_reconcile": "input/source_reconcile.json",
            "candidate": "output/candidate.json",
        },
        "authority": (
            "path manifest only; exact bytes remain in files and no artifact "
            "field, semantic classification, or repair is generated"
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
                        "feedback_frontier.json",
                        "feedback_detail.json",
                        *{
                            f"feedback_category_{name}.json"
                            for name in _FEEDBACK_CATEGORY_NAMES
                        },
                        "candidate_check.json",
                        "source_reconcile.json",
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
        source_reconcile_results: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        candidate_text: str | None = None
        last_checked_good: str | None = None
        timed_out_calls = 0
        timed_out_elapsed_ms = 0.0

        def check_current_candidate(*, trigger: str) -> dict[str, Any]:
            nonlocal candidate_text
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
                if check["status"] == "pass":
                    try:
                        # CandidateCheck intentionally remains source-free.  The
                        # adjacent submission boundary additionally enforces
                        # anonymity before a passing draft may leave this owner.
                        decode_scrc_lift_artifact(candidate_text, prepared)
                    except SourceIdentifierPrivacyError as exc:
                        check = {
                            "protocol": CANDIDATE_CHECK_PROTOCOL,
                            "status": "fail",
                            "diagnostics": [
                                {
                                    "rule_code": "source_identifier_privacy",
                                    "path": exc.path,
                                    "observed_refs_or_types": [
                                        "replace this free-form Source spelling "
                                        "with the correct anonymous handle or a "
                                        "source-neutral closed protocol value"
                                    ],
                                }
                            ],
                        }
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
            return check

        def candidate_check_tool() -> Mapping[str, Any]:
            return check_current_candidate(trigger="agent_tool")

        def reconcile_current_candidate(*, trigger: str) -> dict[str, Any]:
            nonlocal candidate_text, last_checked_good
            if candidate_text is None:
                try:
                    candidate_text = workspace.read_candidate()
                except Exception as exc:
                    result = {
                        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
                        "status": "fail",
                        "applicable": False,
                        "diagnostics": [
                            {
                                "rule_code": "candidate_unavailable",
                                "path": "<candidate>",
                                "anonymous_handles": [],
                                "observed_refs_or_types": [type(exc).__name__],
                                "line": None,
                            }
                        ],
                    }
                else:
                    result = reconcile_candidate_source_ttir(
                        candidate_text,
                        self._source_ttir_observation,
                    )
            else:
                result = reconcile_candidate_source_ttir(
                    candidate_text,
                    self._source_ttir_observation,
                )
            source_reconcile_results.append(
                {
                    "trigger": trigger,
                    "status": result["status"],
                    "applicable": result.get("applicable"),
                    "diagnostics": [
                        dict(item) for item in result.get("diagnostics", [])
                    ],
                    "candidate_sha256": (
                        hashlib.sha256(candidate_text.encode()).hexdigest()
                        if candidate_text is not None
                        else None
                    ),
                }
            )
            if result["status"] == "pass" and candidate_text is not None:
                # Exact LLM-authored bytes only.  This checkpoint has passed both
                # source-free CandidateCheck (enforced by callers) and the
                # source-aware reject-only occurrence preflight.
                last_checked_good = candidate_text
            return result

        def source_reconcile_tool() -> Mapping[str, Any]:
            candidate_check = check_current_candidate(
                trigger="source_reconcile_candidate_check"
            )
            if candidate_check["status"] != "pass":
                return {
                    "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
                    "status": "fail",
                    "applicable": False,
                    "diagnostics": [
                        {
                            "rule_code": "candidate_check_required",
                            "path": "<candidate>",
                            "anonymous_handles": [],
                            "observed_refs_or_types": [],
                            "line": None,
                        }
                    ],
                }
            return reconcile_current_candidate(trigger="agent_tool")

        def host_preflight(
            *, trigger: str
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
            candidate_check = check_current_candidate(trigger=trigger)
            if candidate_check["status"] != "pass" or candidate_text is None:
                return candidate_check, None
            source_reconcile = reconcile_current_candidate(trigger=trigger)
            return candidate_check, source_reconcile

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "candidate_check": candidate_check_tool,
                "source_reconcile": source_reconcile_tool,
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
                        "source_reconcile": list(source_reconcile_results),
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

                        check, source_reconcile = host_preflight(
                            trigger="host_timeout_recheck"
                        )
                        if (
                            check["status"] == "pass"
                            and source_reconcile is not None
                            and source_reconcile["status"] == "pass"
                            and candidate_text is not None
                        ):
                            timeout_event["recovery"] = "current_exact_candidate"
                            break
                        if last_checked_good is not None:
                            _atomic_write_text(
                                workspace.candidate_path, last_checked_good
                            )
                            check, source_reconcile = host_preflight(
                                trigger="host_timeout_checkpoint_restore"
                            )
                            if (
                                check["status"] == "pass"
                                and source_reconcile is not None
                                and source_reconcile["status"] == "pass"
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
                        if check["status"] != "pass":
                            _atomic_write_json(
                                workspace.input_dir / "candidate_check.json",
                                check,
                            )
                            prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                        else:
                            assert source_reconcile is not None
                            _atomic_write_json(
                                workspace.input_dir / "source_reconcile.json",
                                source_reconcile,
                            )
                            prompt = _SOURCE_RECONCILE_CONTINUATION_PROMPT
                        timeout_event["recovery"] = "same_thread_continuation"
                        selected_effort = self.repair_effort or effective_effort
                        continue
                    raise
                call_turns.append(turn)
                self._session_active_elapsed_ms += turn.elapsed_ms
                self._session_model_turns += 1
                self._session_physical_provider_calls += turn.model_segments

                check, source_reconcile = host_preflight(trigger="host_post_turn")
                if (
                    check["status"] == "pass"
                    and source_reconcile is not None
                    and source_reconcile["status"] == "pass"
                    and candidate_text is not None
                ):
                    break
                if check["status"] != "pass":
                    _atomic_write_json(
                        workspace.input_dir / "candidate_check.json",
                        check,
                    )
                    prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                else:
                    assert source_reconcile is not None
                    _atomic_write_json(
                        workspace.input_dir / "source_reconcile.json",
                        source_reconcile,
                    )
                    prompt = _SOURCE_RECONCILE_CONTINUATION_PROMPT
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
                "source_reconcile": list(source_reconcile_results),
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
            "source_reconcile": list(source_reconcile_results),
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
            termination_status="CandidateCheckAndSourceReconcilePass",
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
            "source_reconcile": list(source_reconcile_results),
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
