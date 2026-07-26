"""mini-swe-agent Lift backend; SCRC verification remains entirely external."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.provider import (
    ProviderConfig,
    get_provider,
)

from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftRequest,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    decode_scrc_lift_artifact,
    check_scrc_lift_candidate,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
from .scrc_schema import scrc_lift_core_json_schema
from .prompt import SCHEMA as INTENTIR_OPERATION_REFERENCE


MINISWE_LIFT_BACKEND = "minisweagent.intentir-lift.v1"
MINISWE_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v47"
MINISWE_LIFT_TURN_TIMEOUT_SECONDS = 900.0
MAX_CONTINUATION_FEEDBACK_BYTES = 64 * 1024
MAX_RELIFT_CHECK_FEEDBACK_ITEMS = 20
MAX_RELIFT_CHECK_FEEDBACK_CHARS = 2048


def _usage_and_response_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen_usage = False
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
        seen_usage = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen_usage else {}), response_id, response_model


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _continued_lift_history(result: Any) -> list[dict[str, Any]]:
    """Keep valid conversation context while dropping the local terminal record."""

    messages = result.trajectory.get("messages", ())
    if not isinstance(messages, list):
        raise TypeError("Lift trajectory messages must be an array")
    history = copy.deepcopy(messages)
    while history and history[-1].get("role") == "exit":
        history.pop()
    if result.exit_status in {"Submitted", "InvalidSubmission"} and history:
        # Submission is represented by a final tool call whose environment-side
        # success interrupted the old worker before a tool result was appended.
        # It cannot be replayed as an unmatched call in the continued Responses
        # conversation. The exact candidate remains in the branch-local
        # persistent Lift workspace.
        terminal_response = history[-1]
        extra = terminal_response.get("extra")
        actions = extra.get("actions", ()) if isinstance(extra, Mapping) else ()
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                "submitted Lift history lacks its unmatched terminal tool call"
            )
        history.pop()
    if not history or any(item.get("role") == "exit" for item in history):
        raise ValueError("Lift continuation history is empty or contains exit state")
    return history


def _round_trajectory(
    trajectory: Mapping[str, Any],
    continued_from_messages: int,
) -> dict[str, Any]:
    """Return per-invocation evidence while the private owner keeps full context."""

    value = copy.deepcopy(dict(trajectory))
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Lift trajectory messages must be an array")
    if not 0 <= continued_from_messages <= len(messages):
        raise ValueError("invalid Lift continuation prefix length")
    value["messages"] = messages[continued_from_messages:]
    return value


def _sanitize_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
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
            str(item_key): _sanitize_trajectory(
                item,
                secret,
                key=str(item_key),
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _worker_evidence(
    result: Any,
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    info = trajectory.get("info", {})
    candidate_check = (
        info.get("candidate_check", {}) if isinstance(info, Mapping) else {}
    )
    agent_runtime = (
        info.get("agent_runtime", {}) if isinstance(info, Mapping) else {}
    )
    return {
        "backend": MINISWE_LIFT_BACKEND,
        "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
        "exit_status": result.exit_status,
        "model_turns": result.model_calls,
        "physical_provider_calls": result.physical_provider_calls,
        "transport_retries": result.transport_retries,
        "elapsed_ms": result.elapsed_ms,
        "fresh_session": True,
        "workflow": (
            "staged" if bool(getattr(result, "stage_metrics", {})) else "single"
        ),
        "stage_metrics": dict(getattr(result, "stage_metrics", {})),
        "candidate_check": (
            dict(candidate_check)
            if isinstance(candidate_check, Mapping)
            else {}
        ),
        "agent_runtime": (
            dict(agent_runtime) if isinstance(agent_runtime, Mapping) else {}
        ),
        "trajectory": trajectory,
    }


def _last_candidate_check_feedback(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the last evaluated bounded, source-free diagnostic summary."""

    candidate_check = evidence.get("candidate_check")
    if not isinstance(candidate_check, Mapping):
        return None
    results = candidate_check.get("results")
    if not isinstance(results, list):
        return None
    for result in reversed(results):
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "fail"
            or result.get("evaluated") is not True
        ):
            continue
        protocol = result.get("protocol")
        rule_diagnostics = result.get("rule_diagnostics")
        if not isinstance(rule_diagnostics, list):
            rule_diagnostics = result.get("rule_paths")
        if protocol != CANDIDATE_CHECK_PROTOCOL or not isinstance(
            rule_diagnostics, list
        ):
            continue
        summary: list[dict[str, Any]] = []
        for item in rule_diagnostics[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            rule_code = item.get("rule_code")
            path = item.get("path")
            if (
                not isinstance(rule_code, str)
                or not isinstance(path, str)
                or rule_code == "check_budget_exhausted"
            ):
                continue
            bounded_item: dict[str, Any] = {
                "rule_code": rule_code[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
                "path": path[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
            }
            observed = item.get("observed_refs_or_types")
            if isinstance(observed, list):
                bounded_observed = [
                    value[:MAX_RELIFT_CHECK_FEEDBACK_CHARS]
                    for value in observed[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]
                    if isinstance(value, str)
                ]
                if bounded_observed:
                    bounded_item["observed_refs_or_types"] = bounded_observed
            summary.append(bounded_item)
        if summary:
            return {"protocol": protocol, "diagnostics": summary}
    return None


def _input_files(request: SCRCLiftRequest) -> dict[str, str]:
    packet = json.loads(request.user_input)
    anonymous_source = packet.get("anonymous_source")
    if not isinstance(anonymous_source, str) or not anonymous_source:
        raise ValueError("prepared Lift packet lacks anonymous Source")
    harness_view = packet.get("harness_compiler_view", {})
    feedback = packet.get("relift")
    from minisweagent import package_dir

    validator = (
        Path(package_dir) / "config" / "intentir_validate_candidate.py"
    ).read_text(encoding="utf-8")
    preflight = (
        Path(package_dir) / "config" / "intentir_lift_preflight.md"
    ).read_text(encoding="utf-8")
    schema_query = (
        Path(package_dir) / "config" / "intentir_schema_query.py"
    ).read_text(encoding="utf-8")
    context_query = (
        Path(package_dir) / "config" / "intentir_context_query.py"
    ).read_text(encoding="utf-8")
    minimal_example = (
        Path(package_dir) / "config" / "intentir_lift_example.json"
    ).read_text(encoding="utf-8")
    readme = """# IntentIR Lift input

This directory is the complete allowlisted input for one branch-local Lift
owner. Its conversation and persistent workspace survive across formal Gate1
continuations for the same Source; Replay remains a separate fresh process.

- `source.py`: alpha-renamed Source available only to Lift.
- `harness_view.json`: trusted anonymous compiler/Harness view.
- `feedback.json`: null on the first round, otherwise only the explicitly
  allowed previous parser/TypeCheck/Gate1/Gate2 feedback, previous candidate, or the
  last worker's bounded source-free CandidateCheck diagnostic summary.
- `schema/artifact.schema.json`: closed static JSON envelope for `(J,C,pi)`.
- `knowledge/lift_contract.md`: semantic ownership and construction rules.
- `knowledge/intent_ops.md`: global source-neutral IntentIR operation reference.
- `knowledge/preflight.md`: short source-neutral handle/ownership checklist.
- `knowledge/minimal_example.json`: source-neutral wire/ownership example for one
  integral rank-0 scalar explicitly equated to a logical dimension by a J-owned
  value_dimension_relations entry, explicit sum-then-div stages, one exact cast
  refinement, and a local reduction axis; replace every case-specific semantic,
  handle, shape, value, topology, and multiplicity.
- `tools/validate_candidate.py`: syntax/static-schema validator only.
- `intentir_candidate_check /output/candidate.json`: read-only
  Decode/TypeCheck diagnostics for the complete candidate; calls are audited
  telemetry and are governed by the owner session's active wall time;
  it never repairs or certifies the artifact.
- `intentir_candidate_commit /output/candidate.json`: record only exact bytes
  already returned PASS by CandidateCheck as the LLM owner's conditional
  fallback; fixed code does not parse or modify fields.
- `intentir_candidate_rollback /output/candidate.json`: restore only those exact
  LLM-committed bytes.
- `intentir_lift_submit /output/candidate.json` or
  `intentir_lift_submit --committed`: direct structured delivery without a
  stdout marker. Direct submit is the normal fast path: it runs CandidateCheck
  on exact current bytes when needed and exits immediately on PASS; external
  Decode/TypeCheck still rerun independently.
- `apply_patch`: replace one exact, uniquely occurring text span in the existing
  `/output/candidate.json`. The model supplies both strings; fixed code performs
  no JSON parsing, handle allocation, reordering, inference, or semantic edit.
- `tools/schema_query.py`: compact schema lookup (`top`, `catalog`, `op`,
  `section`) so raw 1400-line schema never needs to be paged end-to-end.
- `tools/context_query.py`: bounded lookup for bootstrap, exact external
  feedback, paged Source, Harness JSON paths, and focused contract/preflight
  matches. Its `candidate-abi` command compares the current draft with only
  Harness-authored mechanical ABI facts and never proposes semantic fields.
  Its syntax-only `source-outline`, literal `source-find` search, and paged `candidate-nodes`/single
  `candidate-node` views expose only Source and J content already visible to
  this worker; they never parse semantics, construct fields, or edit the draft.
  Its `feedback --path`, `candidate --path`, and `candidate-find` commands are
  bounded read-only inspection over already-mounted data and the current draft.
  `bootstrap` copies an explicitly supplied previous artifact only when no
  candidate exists. In the persistent owner workspace it preserves the current
  candidate unchanged; no command constructs or repairs artifact fields.

When the initial user packet says `exact_inputs_embedded=false`, start with this
single bounded command so long inputs cannot hide feedback. Otherwise the exact
Source/Harness/contract/preflight/example bytes are already present and this
inventory call is optional:

`python /input/tools/context_query.py bootstrap`

Then inspect Source, Harness, and the syntax-only outline with independent calls
that may share one model response:
`python /input/tools/context_query.py source` and
`python /input/tools/context_query.py harness`, plus
`python /input/tools/context_query.py source-outline --start 1 --count 32`.
On repair rounds, inspect
`python /input/tools/context_query.py feedback` before editing the candidate
already retained in the workspace. Use its `feedback-category`, paged `source`, Harness-path,
and `knowledge` subcommands for narrower views.
For a long Source, run
`python /input/tools/context_query.py source-outline --start 1 --count 32`
and page it before deciding that no loop, branch, or early return exists. The
outline contains syntax locations only; it never classifies or constructs J.
Use `feedback --path diagnostics.KEY` for an exact unresolved Gate1 or
CandidateCheck section; do not probe category names or parse the full file with
ad-hoc scripts. Use `candidate --path j.PATH` and `candidate-find HANDLE` to
inspect a large current draft without rewriting it.
Gate1 `dataflow.node_witnesses` use `source_node_*`, `replay_node_*`, and
`shared_node_*` aliases only to preserve repeated observer identity within that
one feedback. Compare their normalized local signatures and incident
producer/consumer roles to localize the earliest wiring divergence; never copy
an observer alias into J/C/pi.
When present, inspect `dataflow.local_frontier` first. It removes recursive
observer-hash fallout and lists bounded immediate producer/consumer profile
differences; it is diagnostic-only, cannot accept a candidate, and does not
prescribe any J/C/pi field.
After writing or editing a complete draft, inspect the bounded public-ABI diff
without paging the whole Harness again:

`python /input/tools/context_query.py candidate-abi`

Do not page through the raw schema. Use the example plus compact queries such as:

`python /input/tools/schema_query.py /input/schema/artifact.schema.json top`

`python /input/tools/schema_query.py /input/schema/artifact.schema.json catalog`

Query exact `op NAME` or leaf `section NAME` only when needed. Useful leaf
sections include `access`, `effect`, `result_relation`, `authorization`,
`scope`, `c_slot`, `pi_instance`, `pi_route`, `pi_iteration`, and
`pi_refinement`. Create the
first complete candidate early, then let validator errors drive further lookup.
On a continuation, keep editing the current complete candidate in the persistent
workspace; `feedback.json.previous_artifact` is an external receipt and bootstrap
fallback, not a command to overwrite newer working bytes. Use `repair_progress` when present and make the
Source-justified coordinated J/C/pi changes needed by the external
counterexample; do not blindly rebuild or mechanically resubmit the same
artifact. Run `python /input/tools/context_query.py candidate-diff` before
submission. This is only a read-only diff of two LLM-authored candidates, never
a repair. The repaired full object is still a new Lift submission and will be
checked externally again.
If current Lift/TypeCheck diagnostics contain `unresolved_gate1`, the enclosed
Gate1 counterexample remains an open obligation across the intermediate
construction failure; address both without inventing fields.
Use `python /input/tools/context_query.py source-find TEXT --context 3` to
locate literal Source occurrences without paging the whole file. For a large
draft, use `candidate-nodes --start 1 --count 40` and `candidate-node n0` to
inspect the graph that you authored. These commands are read-only discovery,
not semantic extraction or repair.
If bootstrap copied a `continuation_draft`, it is only the adjacent prior
worker's complete LLM-written JSON draft, not a decoded or accepted artifact.
Continue its focused repair and rerun validator/CandidateCheck yourself.
Write it to `/output/candidate.json`. The normal next action is:

`intentir_lift_submit /output/candidate.json`

Submission runs the source-free CandidateCheck preflight on unchecked exact
bytes. It exits this Lift invocation on PASS and returns short diagnostics on
failure. The static validator and explicit diagnostic command remain available
when useful, but are not mandatory network turns:

`intentir_candidate_check /output/candidate.json`

This host virtual command must be the entire shell action exactly as shown. Do
not prepend `set`, an assignment, a pipe, or another shell command; such a
prefixed action runs inside the sandbox, where it is not an executable.
If it fails, edit the candidate yourself from the closed diagnostics and rerun
within the branch-owner active wall-time budget. The formal Runner independently repeats Decode
and TypeCheck after submission; CandidateCheck is not an acceptance path.
If the exact submitted bytes have no prior CandidateCheck PASS, submission
automatically invokes the same read-only preflight and returns FAIL diagnostics
to this session instead of delivering the candidate. It still never edits the
object, and a PASS still has no authority outside this worker.
Old tool calls and observations may appear as deterministic hash receipts in the
model packet. Re-read the persistent files for exact content; the runtime never
summarizes or interprets J/C/pi semantics during compaction.
"""
    return {
        "README.md": readme,
        "source.py": anonymous_source,
        "harness_view.json": json.dumps(
            harness_view, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "feedback.json": json.dumps(
            feedback, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "schema/artifact.schema.json": json.dumps(
            scrc_lift_core_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        "knowledge/lift_contract.md": request.instructions,
        "knowledge/intent_ops.md": INTENTIR_OPERATION_REFERENCE,
        "knowledge/preflight.md": preflight,
        "knowledge/minimal_example.json": minimal_example,
        "tools/validate_candidate.py": validator,
        "tools/schema_query.py": schema_query,
        "tools/context_query.py": context_query,
    }


@dataclass
class MiniSWELiftAdapter:
    """One branch-local source-aware owner across successive LiftAdapter calls."""

    provider: ProviderConfig | None = None
    compiler_view: Any = None
    effort: str | None = None
    source_symbol: str | None = None
    workflow: str = "single"
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    session_wall_time_limit_seconds: float = 3600.0
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    _session_history: list[dict[str, Any]] | None = field(default=None, init=False)
    _session_source_sha256: str | None = field(default=None, init=False)
    _session_active_elapsed_ms: float = field(default=0.0, init=False)
    _session_model_turns: int = field(default=0, init=False)
    _session_physical_provider_calls: int = field(default=0, init=False)
    _session_transport_retries: int = field(default=0, init=False)
    _session_workspace: Any = field(default=None, init=False, repr=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def close(self) -> None:
        workspace = self._session_workspace
        self._session_workspace = None
        if workspace is not None:
            workspace.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _latest_rejected_candidate(
        self,
        diagnostics: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Return only the adjacent JSON object named by parser feedback."""

        if (
            not self._records
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("failure_class") != "lift_response"
        ):
            return None
        record = self._records[-1]
        if (
            record.relift_round != len(self._records)
            or record.extraction is not None
            or len(record.rejected_attempts) != 1
        ):
            return None
        attempt = record.rejected_attempts[0]
        if diagnostics.get("diagnostic") != attempt.diagnostic:
            return None

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                attempt.response_text,
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict):
                return None
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return None
        return candidate

    def _latest_continuation_candidate(
        self,
        diagnostics: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Return one adjacent LLM-written draft without accepting or editing it."""

        if (
            not self._records
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("failure_class") != "lift_worker_no_candidate"
        ):
            return None
        receipt = diagnostics.get("continuation_draft")
        if not isinstance(receipt, Mapping):
            return None
        expected_sha256 = receipt.get("sha256")
        record = self._records[-1]
        candidate = record.continuation_candidate
        if (
            record.relift_round != len(self._records)
            or record.extraction is not None
            or not isinstance(candidate, Mapping)
            or not isinstance(expected_sha256, str)
        ):
            return None
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != expected_sha256:
            return None
        return dict(candidate)

    @staticmethod
    def _read_continuation_candidate(workspace: Any) -> dict[str, Any] | None:
        """Read a complete worker draft under the existing output policy."""

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                workspace.read_candidate(),
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict) or set(candidate) != {
                "schema",
                "j",
                "c",
                "pi",
            }:
                return None
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
            return None
        return candidate

    def _relift_context(self, feedback: Any) -> SCRCReliftContext | None:
        if feedback is None:
            return None
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        diagnostics = getattr(feedback, "diagnostics", None)
        previous_artifact = getattr(feedback, "previous_artifact", None)
        previous_candidate = (
            self._latest_rejected_candidate(diagnostics)
            if kind == "lift"
            else None
        )
        if previous_candidate is None and kind == "lift":
            previous_candidate = self._latest_continuation_candidate(
                diagnostics
            )
        if previous_candidate is None and hasattr(previous_artifact, "to_json"):
            previous_candidate = previous_artifact.to_json()
        return SCRCReliftContext(
            max(1, len(self._records)),
            kind,
            diagnostics if isinstance(diagnostics, Mapping) else {},
            previous_candidate,
        )

    def _continued_worker_config(self) -> dict[str, Any]:
        config = copy.deepcopy(dict(self.worker_config))
        if self.workflow != "single":
            return config
        if self.session_wall_time_limit_seconds <= 0:
            raise ValueError("Lift owner session wall-time limit must be positive")
        remaining_seconds = self.session_wall_time_limit_seconds - (
            self._session_active_elapsed_ms / 1000.0
        )
        if remaining_seconds <= 0:
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {
                    "diagnostic": "branch-local Lift owner wall-time exhausted",
                    "failure_class": "lift_owner_time_exhausted",
                    "session_id": self._session_id,
                    "active_elapsed_ms": self._session_active_elapsed_ms,
                },
                physical_calls=0,
                terminal=True,
            )
        agent = config.get("agent")
        agent_config = dict(agent) if isinstance(agent, Mapping) else {}
        # Tool/model/check counts remain telemetry. The branch owner is bounded by
        # cumulative active wall time and the outer formal Gate1 budget instead.
        agent_config["step_limit"] = 0
        agent_config["wall_time_limit_seconds"] = max(
            1,
            math.ceil(remaining_seconds),
        )
        config["agent"] = agent_config
        return config

    def _continuation_message(self, feedback: Any, round_index: int) -> str:
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        diagnostics = getattr(feedback, "diagnostics", None)
        closed_diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        diagnostic_bytes = json.dumps(
            closed_diagnostics,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        envelope = {
            "protocol": "intentir.lift-owner-continuation.v1",
            "owner_session": self._session_id,
            "lift_invocation": round_index,
            "failure_kind": kind,
            "feedback_path": "/input/feedback.json",
            "candidate_path": "/output/candidate.json",
            "exact_feedback_embedded": (
                len(diagnostic_bytes) <= MAX_CONTINUATION_FEEDBACK_BYTES
            ),
        }
        if len(diagnostic_bytes) <= MAX_CONTINUATION_FEEDBACK_BYTES:
            envelope["closed_feedback"] = closed_diagnostics
        else:
            envelope["closed_feedback_receipt"] = {
                "bytes": len(diagnostic_bytes),
                "sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
            }
        return (
            json.dumps(envelope, sort_keys=True, ensure_ascii=False)
            + "\nThe same branch-local source-aware Lift task continues. "
            "When exact_feedback_embedded is true, use it directly instead of "
            "spending a turn rediscovering feedback; otherwise read the bounded "
            "feedback path. The latest complete candidate remains "
            "in the same persistent workspace. Modify that file and resubmit one "
            "complete (J,C,pi). This is not a new "
            "writer and not a request for a patch-only artifact."
        )

    def _record_session_result(self, result: Any) -> None:
        if self.workflow != "single":
            return
        self._session_active_elapsed_ms += float(result.elapsed_ms)
        self._session_model_turns += int(result.model_calls)
        self._session_physical_provider_calls += int(
            result.physical_provider_calls
        )
        self._session_transport_retries += int(result.transport_retries)
        self._session_history = _continued_lift_history(result)

    def _session_evidence(
        self,
        evidence: Mapping[str, Any],
        *,
        fresh_session: bool,
        result: Any,
    ) -> dict[str, Any]:
        value = dict(evidence)
        value.update(
            {
                "fresh_session": fresh_session,
                "persistent_owner": self.workflow == "single",
                "owner_session_id": self._session_id,
                "continued_from_messages": result.continued_from_messages,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_model_turns": self._session_model_turns,
                "session_physical_provider_calls": (
                    self._session_physical_provider_calls
                ),
                "session_transport_retries": self._session_transport_retries,
                "session_wall_time_limit_seconds": (
                    self.session_wall_time_limit_seconds
                ),
            }
        )
        return value

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("MiniSWE Lift requires SourceProgram.text")
        if self.workflow not in {"single", "staged"}:
            raise ValueError("MiniSWE Lift workflow must be 'single' or 'staged'")
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        if self._session_source_sha256 is None:
            self._session_source_sha256 = source_sha256
        elif self._session_source_sha256 != source_sha256:
            raise ValueError("one Lift owner session cannot cross Source programs")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        if timeout is None:
            provider_timeout = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
            timeout = min(provider_timeout, MINISWE_LIFT_TURN_TIMEOUT_SECONDS)
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=self._relift_context(feedback),
            source_symbol=self.source_symbol,
        )
        round_index = len(self._records) + 1
        fresh_owner_session = (
            self.workflow != "single" or self._session_history is None
        )
        effective_worker_config = self._continued_worker_config()
        configured_owner_remaining = effective_worker_config.get("agent", {}).get(
            "wall_time_limit_seconds"
        )
        if (
            isinstance(configured_owner_remaining, (int, float))
            and not isinstance(configured_owner_remaining, bool)
            and configured_owner_remaining > 0
        ):
            timeout = min(timeout, float(configured_owner_remaining))
        continuation_history = (
            self._session_history
            if self.workflow == "single" and not fresh_owner_session
            else None
        )
        continuation_message = (
            self._continuation_message(feedback, round_index)
            if continuation_history is not None
            else None
        )
        candidate_check_context = scrc_candidate_check_context(prepared)
        try:
            from minisweagent.environments.intentir_lift import (
                IntentIRLiftWorkspace,
            )
            from minisweagent.run.intentir_lift import (
                IntentIRLiftRunError,
                create_intentir_lift_model,
                run_intentir_lift,
                run_intentir_staged_lift,
            )
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires `pip install -e '.[lift-agent,verify]'`",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc

        input_files = _input_files(prepared)
        fresh_workspace = self.workflow != "single" or self._session_workspace is None
        if self.workflow == "single":
            if self._session_workspace is None:
                self._session_workspace = IntentIRLiftWorkspace.create(input_files)
            else:
                self._session_workspace.refresh_inputs(input_files)
            workspace = self._session_workspace
        else:
            workspace = IntentIRLiftWorkspace.create(input_files)
        result = None
        try:
            model = create_intentir_lift_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
            )
            run_worker = (
                run_intentir_staged_lift
                if self.workflow == "staged"
                else run_intentir_lift
            )
            run_kwargs: dict[str, Any] = {
                "model": model,
                "workspace": workspace,
                "config_overrides": effective_worker_config,
                "candidate_checker": lambda candidate_text: (
                    check_scrc_lift_candidate(
                        candidate_text,
                        candidate_check_context,
                    )
                ),
            }
            if self.workflow == "single":
                run_kwargs.update(
                    {
                        "continuation_history": continuation_history,
                        "continuation_message": continuation_message,
                    }
                )
            result = run_worker(
                **run_kwargs,
            )
            self._record_session_result(result)
        except IntentIRLiftRunError as exc:
            result = exc.result
            self._record_session_result(result)
            safe_trajectory = _round_trajectory(
                _sanitize_trajectory(result.trajectory, provider.api_key),
                result.continued_from_messages,
            )
            evidence = self._session_evidence(
                _worker_evidence(result, safe_trajectory),
                fresh_session=fresh_owner_session,
                result=result,
            )
            continuation_candidate = self._read_continuation_candidate(
                workspace
            )
            continuation_receipt = None
            if continuation_candidate is not None:
                continuation_bytes = json.dumps(
                    continuation_candidate,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
                continuation_receipt = {
                    "sha256": hashlib.sha256(
                        continuation_bytes
                    ).hexdigest(),
                    "bytes": len(continuation_bytes),
                }
                evidence["continuation_draft"] = dict(
                    continuation_receipt
                )
            diagnostic = str(exc)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=evidence,
                    continuation_candidate=continuation_candidate,
                )
            )
            if (
                exc.cause is not None
                and result.exit_status != "InvalidSubmission"
            ) or result.model_calls < 1:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    diagnostic,
                    evidence={
                        key: value
                        for key, value in evidence.items()
                        if key != "trajectory"
                    },
                ) from exc
            from intentir.verify.scrc_runner import CandidateLiftError

            rejected_diagnostics: dict[str, Any] = {
                "diagnostic": diagnostic,
                "failure_class": "lift_worker_no_candidate",
            }
            candidate_check_feedback = _last_candidate_check_feedback(evidence)
            if candidate_check_feedback is not None:
                rejected_diagnostics["candidate_check_feedback"] = (
                    candidate_check_feedback
                )
            if continuation_receipt is not None:
                rejected_diagnostics["continuation_draft"] = dict(
                    continuation_receipt
                )
            raise CandidateLiftError(
                rejected_diagnostics,
                physical_calls=result.physical_provider_calls,
                terminal=result.exit_status == "TimeExceeded",
            ) from exc
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence={
                        "backend": MINISWE_LIFT_BACKEND,
                        "fresh_session": fresh_owner_session,
                        "persistent_owner": self.workflow == "single",
                        "owner_session_id": self._session_id,
                    },
                )
            )
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            if isinstance(exc, SCRCInfrastructureError):
                raise
            raise SCRCInfrastructureError(
                f"MiniSWE Lift infrastructure failed: {type(exc).__name__}: {exc}",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc
        finally:
            if self.workflow != "single":
                workspace.cleanup()

        assert result is not None and result.candidate_text is not None
        safe_trajectory = _round_trajectory(
            _sanitize_trajectory(result.trajectory, provider.api_key),
            result.continued_from_messages,
        )
        usage, response_id, response_model = _usage_and_response_identity(
            safe_trajectory
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        worker_evidence = self._session_evidence(
            _worker_evidence(result, safe_trajectory),
            fresh_session=fresh_owner_session,
            result=result,
        )
        try:
            artifact = decode_scrc_lift_artifact(result.candidate_text, prepared)
        except Exception as exc:
            attempt = SCRCLiftAttempt(
                model=provider.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=result.candidate_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                response_sha256=hashlib.sha256(
                    result.candidate_text.encode()
                ).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                elapsed_ms=result.elapsed_ms,
                backend=MINISWE_LIFT_BACKEND,
                termination_status=result.exit_status,
                trajectory=safe_trajectory,
            )
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=attempt.diagnostic,
                    rejected_attempts=(attempt,),
                    backend_evidence=worker_evidence,
                )
            )
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {
                    "diagnostic": attempt.diagnostic,
                    "failure_class": "lift_response",
                },
                physical_calls=result.physical_provider_calls,
            ) from exc

        provenance = SCRCProvenance(
            model=provider.model,
            prompt_sha256=prepared.prompt_sha256,
            response_id=response_id,
            response_model=response_model,
            usage=usage,
            response_sha256=hashlib.sha256(
                result.candidate_text.encode()
            ).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=result.candidate_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            elapsed_ms=result.elapsed_ms,
            backend=MINISWE_LIFT_BACKEND,
            termination_status=result.exit_status,
            trajectory=safe_trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
        )
        self._records.append(
            SCRCReliftRecord(
                round_index,
                extraction=extraction,
                backend_evidence=worker_evidence,
            )
        )
        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            artifact,
            physical_calls=result.physical_provider_calls,
            metadata={
                "backend": MINISWE_LIFT_BACKEND,
                "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
                "model": provider.model,
                "response_model": response_model,
                "response_id": response_id,
                "usage": usage,
                "elapsed_ms": result.elapsed_ms,
                "agent_steps": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "fresh_session": fresh_owner_session,
                "fresh_workspace": fresh_workspace,
                "persistent_workspace": self.workflow == "single",
                "persistent_owner": self.workflow == "single",
                "owner_session_id": self._session_id,
                "continued_from_messages": result.continued_from_messages,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_model_turns": self._session_model_turns,
                "session_physical_provider_calls": (
                    self._session_physical_provider_calls
                ),
                "workflow": self.workflow,
                "artifact_schema": artifact.wire_schema,
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "candidate_check": worker_evidence["candidate_check"],
                "agent_runtime": worker_evidence["agent_runtime"],
                "stage_metrics": worker_evidence["stage_metrics"],
            },
        )


__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
]
