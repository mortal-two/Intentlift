"""Codex CLI Lift backend with external SCRC verification boundaries."""

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
CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v30"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v88"
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

Act like a coding agent, not a staged JSON generator. Start with the one compact
`source-brief` command in input/README.md, then read input/source.py. The brief is
only exact Source/Harness/compiler navigation; you decide all meaning. Query an
exact compiler-observer section, TTIR slice, schema leaf, operation, or contract
term only when the current draft needs it. Do not dump input/source_ttir.json, the
whole Schema, the whole operation catalog, or the whole Lift contract into the
thread.

You alone author the complete output/candidate.json. J owns all logical semantics;
C owns only implementation choices; pi owns only source-realization relations for
the authored J+C. Fixed code must not choose or fill any field, handle, relation,
operation, shape, or topology. Preserve anonymous ABI order, parametric shape
identities, control/time, effects, and realization structure visible in Source and
the selected anonymous specialization. Use dense declaration-ordered opaque
handles and never copy Source identifiers into the artifact.

Write a meaningful complete output/candidate.json early, then inspect and patch
that persistent file in place; do not postpone all authoring until after exhaustive
reading. Organize scratch under work/ however useful. The final J/C/pi and every
semantic field remain entirely your work. Run
`python .agents/intentir_agent_tool.py submit_candidate` after a meaningful complete
draft. This transaction first runs the source-free CandidateCheck and then the
reject-only SourceReconcile on the same exact bytes. Repair the returned diagnostics
and invoke it again. The host then runs the public Source-vs-J semantic probe on
those exact passing bytes and returns a same-thread diagnostic if they diverge.
That probe never certifies or edits the candidate: repair only when the evidence
shows your J is wrong, rerun submit_candidate, and leave small floating deltas or
unsupported execution as diagnostics. The host independently reruns Decode and
TypeCheck, then starts fresh source-blind Replay and the formal gates. Do not claim
certification.
"""


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


def _replay_counterexample_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Find the exact prior Replay evidence through retained Gate1 wrappers."""

    if not isinstance(diagnostics, Mapping):
        return None
    evidence = diagnostics.get("replay_counterexample")
    if isinstance(evidence, Mapping):
        return evidence
    unresolved = diagnostics.get("unresolved_gate1")
    nested = (
        unresolved.get("diagnostics")
        if isinstance(unresolved, Mapping)
        else None
    )
    evidence = (
        nested.get("replay_counterexample")
        if isinstance(nested, Mapping)
        else None
    )
    return evidence if isinstance(evidence, Mapping) else None


def _codex_feedback_files(feedback_text: str) -> dict[str, str]:
    """Partition exact closed feedback without exposing one bulk relift object.

    Category names already belong to the Gate1 diagnostic protocol.  This only
    separates bytes by that existing label and removes the duplicate previous
    artifact, which the persistent owner already has in output/candidate.json.
    """

    feedback = json.loads(feedback_text)
    categories: dict[str, Any] = {}
    replay_counterexample: Mapping[str, Any] | None = None
    if isinstance(feedback, Mapping):
        diagnostics = (
            feedback.get("diagnostics")
            if isinstance(feedback.get("diagnostics"), Mapping)
            else None
        )
        replay_counterexample = _replay_counterexample_from_diagnostics(
            diagnostics
        )
        observer = _observer_diff_from_diagnostics(
            diagnostics
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
                if key not in {
                    "observer_diff",
                    "replay_counterexample",
                    "unresolved_gate1",
                }
            }
            unresolved = diagnostics.get("unresolved_gate1")
            if isinstance(unresolved, Mapping):
                unresolved_copy = dict(unresolved)
                nested = unresolved_copy.get("diagnostics")
                if isinstance(nested, Mapping):
                    unresolved_copy["diagnostics"] = {
                        str(key): value
                        for key, value in nested.items()
                        if key not in {
                            "observer_diff",
                            "replay_counterexample",
                        }
                    }
                pruned["unresolved_gate1"] = unresolved_copy
            if pruned:
                detail["diagnostics"] = pruned

    replay_source = (
        replay_counterexample.get("replay_source")
        if isinstance(replay_counterexample, Mapping)
        else None
    )
    replay_ttir = (
        replay_counterexample.get("replay_ttir")
        if isinstance(replay_counterexample, Mapping)
        else None
    )
    replay_metadata = (
        {
            str(key): value
            for key, value in replay_counterexample.items()
            if key not in {"replay_source", "replay_ttir"}
        }
        if isinstance(replay_counterexample, Mapping)
        else None
    )
    values: dict[str, Any] = {
        "feedback_detail.json": detail,
        "feedback_replay.json": replay_metadata,
    }
    values.update(
        {
            f"feedback_category_{name}.json": categories.get(name)
            for name in _FEEDBACK_CATEGORY_NAMES
        }
    )
    rendered = {
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
    rendered["feedback_replay.py"] = (
        replay_source.rstrip() + "\n"
        if isinstance(replay_source, str) and replay_source
        else "# No formal Replay source is available before a Gate1 counterexample.\n"
    )
    rendered["feedback_replay.ttir"] = (
        replay_ttir.rstrip() + "\n"
        if isinstance(replay_ttir, str) and replay_ttir
        else "// No formal Replay TTIR is available before a Gate1 counterexample.\n"
    )
    return rendered



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
    files = {
        "source.py": complete["source.py"],
        "harness_view.json": complete["harness_view.json"],
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "README.md": (
            "# IntentIR Lift workspace\n\n"
            "J owns logical semantics, C only implementation choices, and pi only "
            "anonymous Source realization of the authored J+C. You author every "
            "semantic field; fixed code only projects evidence, checks, and rejects.\n\n"
            "Start once with:\n"
            "`python input/tools/context_query.py --input-root input --output-root output source-brief`\n\n"
            "Then read `input/source.py`. For an exact compiler fact, use the same tool "
            "with `source-observer --path fingerprint.NAME`, `source-ttir-outline`, "
            "`source-ttir-find`, or `source-ttir-def-use`. For wire syntax, query only "
            "the needed leaves, for example:\n"
            "`python input/tools/schema_query.py authoring-brief`\n"
            "Then request only the exact `section NAME` or `op NAME` needed by the draft.\n\n"
            "Search one exact term in `input/knowledge/lift_contract.md` only when its "
            "ownership is unclear. Write `output/candidate.json` early and patch it in "
            "place. Run `python .agents/intentir_agent_tool.py submit_candidate`; the "
            "host supplies the public semantic probe after a mechanically passing draft.\n"
        ),
        "schema/artifact.schema.json": complete["schema/artifact.schema.json"],
        "knowledge/lift_contract.md": complete["knowledge/lift_contract.md"],
        "knowledge/intent_ops.md": complete["knowledge/intent_ops.md"],
        "knowledge/preflight.md": complete["knowledge/preflight.md"],
        "tools/context_query.py": complete["tools/context_query.py"],
        "tools/schema_query.py": complete["tools/schema_query.py"],
        "candidate_check.json": "null\n",
        "source_reconcile.json": "null\n",
        "semantic_probe.json": "null\n",
    }
    files.update(_codex_feedback_files(complete["feedback.json"]))
    replay_source = complete.get("gate1_counterexample/replay.py", "")
    replay_ttir = complete.get("gate1_counterexample/replay.ttir", "")
    files["feedback_replay.py"] = (
        replay_source.rstrip() + "\n"
        if replay_source
        else "# No formal Replay source is available before a Gate1 counterexample.\n"
    )
    files["feedback_replay.ttir"] = (
        replay_ttir.rstrip() + "\n"
        if replay_ttir
        else "// No formal Replay TTIR is available before a Gate1 counterexample.\n"
    )
    if "source_ttir.json" in complete:
        files["source_ttir.json"] = complete["source_ttir.json"]
    return files

_EXTERNAL_CONTINUATION_PROMPT = """The same branch-local source-aware Lift task continues in the same workspace and thread.
Read input/feedback_frontier.json first. Open only a named
input/feedback_category_NAME.json when its bounded frontier is insufficient; that
file contains the exact differences for the named category.
input/feedback_detail.json contains exact non-observer diagnostics. The previous
artifact already exists at output/candidate.json and is never duplicated in input.
After a formal Gate1 failure, input/feedback_replay.py and
input/feedback_replay.ttir contain the exact preceding source-blind Replay output;
use them only as read-only counterexample evidence.
Your own prior formal candidates are immutable under work/candidate_history/;
consult or restore them when a repair regresses, but decide every edit yourself.
Use the compact source query only where the named feedback needs exact Source or
compiler evidence, then revise the complete model-authored J/C/pi in place. If a
named relation's ownership is unclear, search only that name in
input/knowledge/lift_contract.md. Treat feedback as a compiler counterexample, not
as a semantic patch. Run `python .agents/intentir_agent_tool.py submit_candidate`;
the host automatically probes the exact passing J and returns any public divergence
to this same thread. The host alone runs formal gates.
"""

_CANDIDATE_CHECK_CONTINUATION_PROMPT = """The host submission preflight rejected the current exact candidate bytes.
Read input/candidate_check.json and repair output/candidate.json in this same
thread. Make coordinated J/C/pi changes when required. Rerun
`python .agents/intentir_agent_tool.py submit_candidate` and finish immediately
after the complete candidate passes. Search a rejected relation's exact field name
in input/knowledge/lift_contract.md when its ownership is unclear. The transaction is diagnostic-only.
"""

_SOURCE_RECONCILE_CONTINUATION_PROMPT = """The source-aware occurrence preflight rejected the current exact candidate bytes.
Read input/source_reconcile.json and revise only the model-authored J/C/pi facts whose
closed compiled obligations disagree with the immutable Source TTIR observation.
This diagnostic supplies no artifact field or repair. Rerun
`python .agents/intentir_agent_tool.py submit_candidate` and finish immediately
after it passes. Formal TypeCheck, Replay, and Gate1 remain external.
"""

_SEMANTIC_PROBE_CONTINUATION_PROMPT = """The current exact candidate passed CandidateCheck and SourceReconcile, and the host ran the public Source-vs-J semantic probe.
Read input/semantic_probe.json in this same thread. It is diagnostic-only and does
not accept, reject, certify, or edit J. If the observed divergence exposes an error
in your authored J, patch output/candidate.json and rerun submit_candidate. If the
probe is unsupported or the remaining delta is only an expected floating reduction
difference, leave the candidate unchanged and finish. Do not claim Gate membership.
"""


def _source_aware_startup_context(workspace: Any, *, fresh_owner: bool) -> dict[str, Any]:
    """Return a path-only receipt; exact task state stays in ordinary files."""

    return {
        "protocol": "intentir.codex-lift-startup-view.v2",
        "fresh_owner": fresh_owner,
        "paths": {
            "source": "input/source.py",
            "harness": "input/harness_view.json",
            "source_observer": "input/source_ttir.json",
            "source_query": "input/tools/context_query.py",
            "schema_query": "input/tools/schema_query.py",
            "artifact_schema": "input/schema/artifact.schema.json",
            "intent_ops": "input/knowledge/intent_ops.md",
            "lift_contract": "input/knowledge/lift_contract.md",
            "feedback_frontier": "input/feedback_frontier.json",
            "feedback_detail": "input/feedback_detail.json",
            "feedback_replay_source": "input/feedback_replay.py",
            "feedback_replay_ttir": "input/feedback_replay.ttir",
            "feedback_replay_metadata": "input/feedback_replay.json",
            "feedback_categories": [
                f"input/feedback_category_{name}.json"
                for name in _FEEDBACK_CATEGORY_NAMES
            ],
            "candidate_check": "input/candidate_check.json",
            "source_reconcile": "input/source_reconcile.json",
            "semantic_probe": "input/semantic_probe.json",
            "candidate": "output/candidate.json",
            "candidate_history": "work/candidate_history/",
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
    semantic_probe: Callable[[Any], Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
    )
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

    @staticmethod
    def _archive_formal_candidate(workspace: Any, feedback: Any) -> None:
        """Preserve this owner's exact formal candidates without selecting one."""

        formal_round = getattr(feedback, "formal_round", None)
        artifact = getattr(feedback, "previous_artifact", None)
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        if (
            kind != "gate1"
            or not isinstance(formal_round, int)
            or isinstance(formal_round, bool)
            or formal_round < 1
            or artifact is None
        ):
            return
        payload = getattr(artifact, "to_json", None)
        if not callable(payload):
            return
        encoded = (
            json.dumps(
                payload(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        history_dir = workspace.scratch_dir / "candidate_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        destination = history_dir / f"gate1_round_{formal_round:03d}.json"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    "branch-local formal candidate history changed for one round"
                )
            return
        _atomic_write_text(destination, encoded)

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
                        "feedback_replay.py",
                        "feedback_replay.ttir",
                        "feedback_replay.json",
                        *{
                            f"feedback_category_{name}.json"
                            for name in _FEEDBACK_CATEGORY_NAMES
                        },
                        "candidate_check.json",
                        "source_reconcile.json",
                        "semantic_probe.json",
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
        if workspace is None:
            raise RuntimeError("Codex Lift workspace was not created")
        self._archive_formal_candidate(workspace, feedback)
        fresh_owner = self._codex_session is None
        call_turns: list[CodexCLITurn] = []
        check_results: list[dict[str, Any]] = []
        source_reconcile_results: list[dict[str, Any]] = []
        semantic_probe_results: list[dict[str, Any]] = []
        semantic_probe_by_candidate: dict[str, dict[str, Any]] = {}
        semantic_probe_feedback_delivered: set[str] = set()
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

        def reconcile_current_candidate(
            *, trigger: str, checkpoint_allowed: bool = True
        ) -> dict[str, Any]:
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
            if (
                checkpoint_allowed
                and result["status"] == "pass"
                and candidate_text is not None
            ):
                # Exact LLM-authored bytes only.  This checkpoint has passed both
                # source-free CandidateCheck (enforced by callers) and the
                # source-aware reject-only occurrence preflight.
                last_checked_good = candidate_text
            return result

        def submit_candidate_tool() -> Mapping[str, Any]:
            """Run both reject-only preflights as one exact-byte transaction.

            CandidateCheck and SourceReconcile retain separate authority and
            diagnostics.  This wrapper only removes the model-facing sequencing
            overhead; the host still reruns both checks after the CLI turn.
            """

            candidate_check = check_current_candidate(trigger="submit_candidate")
            source_reconcile = reconcile_current_candidate(
                trigger="submit_candidate",
                checkpoint_allowed=candidate_check["status"] == "pass",
            )
            passed = (
                candidate_check["status"] == "pass"
                and source_reconcile["status"] == "pass"
            )
            return {
                "protocol": "intentir.codex-cli.lift-submit.v1",
                "status": "pass" if passed else "fail",
                "phase": (
                    "complete"
                    if passed
                    else (
                        "candidate_check"
                        if candidate_check["status"] != "pass"
                        else "source_reconcile"
                    )
                ),
                "candidate_check": candidate_check,
                "source_reconcile": source_reconcile,
                "authority": (
                    "diagnostic-only independent phase results over the same exact "
                    "candidate bytes; no artifact field is generated or repaired"
                ),
            }

        def host_preflight(
            *, trigger: str
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
            candidate_check = check_current_candidate(trigger=trigger)
            if candidate_check["status"] != "pass" or candidate_text is None:
                return candidate_check, None
            source_reconcile = reconcile_current_candidate(trigger=trigger)
            return candidate_check, source_reconcile

        def semantic_probe_tool() -> Mapping[str, Any]:
            """Execute one public Source-vs-J diagnostic over exact candidate bytes."""

            candidate_check, source_reconcile = host_preflight(
                trigger="semantic_probe"
            )
            if candidate_check["status"] != "pass":
                result: dict[str, Any] = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "candidate_check",
                    "diagnostic": "current candidate has not passed CandidateCheck",
                }
            elif source_reconcile is None or source_reconcile["status"] != "pass":
                result = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "source_reconcile",
                    "diagnostic": "current candidate has not passed SourceReconcile",
                }
            elif candidate_text is None or self.semantic_probe is None:
                result = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "execution",
                    "diagnostic": "this Lift invocation has no public probe runner",
                }
            else:
                artifact = decode_scrc_lift_artifact(candidate_text, prepared)
                result = dict(self.semantic_probe(artifact))
            result.setdefault(
                "authority",
                "public diagnostic only; not CandidateCheck or certification",
            )
            candidate_sha256 = (
                hashlib.sha256(candidate_text.encode()).hexdigest()
                if candidate_text is not None
                else None
            )
            result["candidate_sha256"] = candidate_sha256
            semantic_probe_results.append(dict(result))
            if candidate_sha256 is not None:
                semantic_probe_by_candidate[candidate_sha256] = dict(result)
            _atomic_write_json(
                workspace.input_dir / "semantic_probe.json",
                result,
            )
            return result

        def semantic_probe_needs_owner_turn() -> bool:
            """Run one public probe per exact passing draft and surface divergence."""

            if candidate_text is None:
                return False
            candidate_sha256 = hashlib.sha256(candidate_text.encode()).hexdigest()
            result = semantic_probe_by_candidate.get(candidate_sha256)
            if result is None:
                result = dict(semantic_probe_tool())
            if (
                result.get("status") == "diverged"
                and candidate_sha256 not in semantic_probe_feedback_delivered
            ):
                semantic_probe_feedback_delivered.add(candidate_sha256)
                return True
            return False

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "submit_candidate": submit_candidate_tool,
                "semantic_probe": semantic_probe_tool,
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
                        "semantic_probe": list(semantic_probe_results),
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
                            if semantic_probe_needs_owner_turn():
                                timeout_event["recovery"] = (
                                    "current_exact_candidate_semantic_probe"
                                )
                                prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                                selected_effort = self.repair_effort or effective_effort
                                continue
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
                                if semantic_probe_needs_owner_turn():
                                    timeout_event["recovery"] = (
                                        "checkpoint_semantic_probe"
                                    )
                                    prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                                    selected_effort = self.repair_effort or effective_effort
                                    continue
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
                    if semantic_probe_needs_owner_turn():
                        prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                        selected_effort = self.repair_effort or effective_effort
                        continue
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
                "semantic_probe": list(semantic_probe_results),
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
            "semantic_probe": list(semantic_probe_results),
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
            "semantic_probe": list(semantic_probe_results),
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
