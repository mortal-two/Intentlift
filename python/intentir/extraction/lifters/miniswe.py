"""mini-swe-agent Lift backend; SCRC verification remains entirely external."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.provider import (
    ProviderConfig,
    get_provider,
)
from intentir.ir.canonical_construction import CanonicalConstruction
from minisweagent.exceptions import CandidateRejected, LiftPreparationFailed
from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCLiftRequest,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    check_scrc_semantic_candidate_surface,
    decode_scrc_lift_artifact,
    check_scrc_lift_candidate,
    _construct_scrc_candidate,
    prepare_scrc_lift,
    scrc_relift_context_from_feedback,
    scrc_candidate_delta_receipt,
    scrc_candidate_check_context,
    _construction_with_source_observation,
    _source_authoring_blockers,
    _source_ttir_observation_for_request,
)
from .scrc_schema import (
    canonical_artifact_skeleton,
    canonical_source_wire_bundle,
    canonical_wire_bundle,
    render_canonical_wire_contract,
    render_canonical_operation_reference,
    render_canonical_preflight,
    scrc_lift_core_json_schema,
)


MINISWE_LIFT_BACKEND = "minisweagent.intentir-lift.v4"
MINISWE_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v70"
# This is only the safety fallback for an explicitly invalid/unlimited caller
# override.  The normal value comes from minisweagent/config/intentir_lift.yaml
# and must not be re-injected by the adapter, otherwise the two config layers
# silently diverge.
DEFAULT_MINISWE_LIFT_STEP_LIMIT = 6
MAX_RELIFT_CHECK_FEEDBACK_ITEMS = 20
MAX_RELIFT_CHECK_FEEDBACK_CHARS = 2048
_FEEDBACK_CATEGORY_PRIORITY = {
    "interface": 0,
    "reduction": 1,
    "predicate_effect": 2,
    "op_multiset": 3,
    "dataflow": 4,
}


def _safe_rule_path_rows(value: Any) -> list[dict[str, Any]]:
    """Keep only bounded rule/path/count metadata for a fresh owner."""

    if not isinstance(value, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in value[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]:
        if not isinstance(raw, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in ("source", "rule_code", "path", "status"):
            item = raw.get(key)
            if isinstance(item, str) and item:
                row[key] = item[:MAX_RELIFT_CHECK_FEEDBACK_CHARS]
        for key in ("stage", "omitted_count"):
            item = raw.get(key)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                row[key] = item
        if row:
            rows.append(row)
    return rows


def _safe_candidate_check_feedback(value: Any) -> dict[str, Any] | None:
    """Project CandidateCheck feedback to its protocol and rule/path rows."""

    if not isinstance(value, Mapping):
        return None
    protocol = value.get("protocol")
    if protocol != CANDIDATE_CHECK_PROTOCOL:
        return None
    diagnostics = _safe_rule_path_rows(value.get("diagnostics"))
    if not diagnostics:
        return None
    return {"protocol": protocol, "diagnostics": diagnostics}


def _provider_failure_class(error: BaseException | None) -> str | None:
    """Classify a provider-side failure without retaining provider text.

    The worker wrapper turns model exceptions into ``IntentIRLiftRunError``.
    Preserve the preparation-stage classification at that boundary so an
    incomplete Responses/SSE turn is not reported as a candidate failure.
    Only a closed class is returned; the exception message is never copied
    into evidence by this helper.
    """

    if error is None:
        return None
    name = type(error).__name__.casefold()
    module = type(error).__module__.casefold()
    text = str(error).casefold()
    if "auth" in name or "unauthorized" in text or "forbidden" in text:
        return "provider_authentication_error"
    if "timeout" in name or "timed out" in text:
        return "provider_timeout"
    if (
        module.startswith(("litellm", "httpx"))
        or "sse" in text
        or "responses stream" in text
        or "unexpected eof" in text
        or "service unavailable" in name.replace("error", "")
        or "connection" in name
        or "rate limit" in name.replace("error", "")
    ):
        return "provider_transport_error"
    return None


def _safe_preparation_receipt(value: Any) -> dict[str, Any] | None:
    """Keep one source-free authoring-preparation receipt for a fresh owner.

    Preparation has no candidate. Preserve only public transaction/rule
    metadata; declaration bytes, Source content,
    provider response text, and trajectories are intentionally absent.
    """

    if not isinstance(value, Mapping):
        return None
    receipt: dict[str, Any] = {}
    for key in (
        "failure_class",
        "diagnostic",
        "attempted_step",
        "next_legal_action",
    ):
        item = value.get(key)
        if isinstance(item, str) and item:
            receipt[key] = item[:256]
    completed = value.get("completed_steps")
    if isinstance(completed, list) and all(
        isinstance(item, str) and item for item in completed
    ):
        receipt["completed_steps"] = list(completed[:16])
    graph = value.get("operation_graph_failure")
    if isinstance(graph, Mapping):
        graph_receipt: dict[str, Any] = {}
        for key in (
            "failure_count",
            "attempt",
            "operation_rows",
            "row_index",
            "result_rank",
            "expected_rank",
            "output_ordinal",
            "access_ordinal",
            "function_visible_matching_value_count",
            "function_visible_compatible_boolean_result_count",
            "direct_single_result_consumer_id_count",
            "required_operation_count",
            "actual_operation_count",
        ):
            item = graph.get(key)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                graph_receipt[key] = item
        for key in (
            "last_rule_code",
            "json_container",
            "declaration_sha256",
            "operation",
            "field",
            "attribute",
            "required_attribute_value",
        ):
            item = graph.get(key)
            if isinstance(item, str) and item:
                graph_receipt[key] = item[:256]
        for key in ("allowed_fields", "required_fields"):
            item = graph.get(key)
            if isinstance(item, list) and all(
                isinstance(entry, str) and entry for entry in item
            ):
                graph_receipt[key] = list(item[:32])
        required_type = graph.get("required_output_type")
        if isinstance(required_type, Mapping):
            dtype = required_type.get("dtype")
            shape = required_type.get("shape")
            if (
                isinstance(dtype, str)
                and isinstance(shape, list)
                and all(
                    isinstance(extent, (str, int))
                    and not isinstance(extent, bool)
                    for extent in shape
                )
            ):
                graph_receipt["required_output_type"] = {
                    "dtype": dtype[:64],
                    "shape": list(shape[:16]),
                }
        if graph_receipt:
            receipt["operation_graph_failure"] = graph_receipt
    return receipt or None


def _feedback_frontier(
    *,
    failure_kind: str | None,
    relift_round: int | None,
    diagnostics: Mapping[str, Any] | None,
    open_guidance: Any = None,
    resolved_guidance: Any = None,
    candidate_delta: Any = None,
) -> dict[str, Any] | None:
    """Project closed diagnostics into a small, diagnostic-only first view.

    The exact feedback remains mounted separately.  This projection only drops
    bulk recursive witnesses and orders already-labelled categories; it never
    infers a cause, chooses a repair, or constructs an artifact field.
    """

    if diagnostics is None:
        return None
    closed = dict(diagnostics)
    observer = closed.get("observer_diff")
    if not isinstance(observer, Mapping):
        unresolved = closed.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, Mapping)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, Mapping)
            else None
        )

    categories: list[dict[str, Any]] = []
    raw_categories = observer.get("categories") if isinstance(observer, Mapping) else None
    if isinstance(raw_categories, list):
        for index, category in enumerate(raw_categories):
            if not isinstance(category, Mapping):
                continue
            name = str(category.get("category", "unknown"))
            item: dict[str, Any] = {"category": name[:64]}
            difference_count = category.get("difference_count")
            if (
                isinstance(difference_count, int)
                and not isinstance(difference_count, bool)
                and difference_count >= 0
            ):
                item["difference_count"] = difference_count
            item["source_category_index"] = index
            categories.append(item)
        categories.sort(
            key=lambda item: (
                _FEEDBACK_CATEGORY_PRIORITY.get(str(item["category"]), 10),
                str(item["category"]),
            )
        )

    frontier: dict[str, Any] = {
        "protocol": "intentir.lift-feedback-frontier.v1",
        "authority": (
            "deterministic bounded projection of closed diagnostics; "
            "diagnostic-only, no semantic inference or repair authority"
        ),
        "failure_kind": failure_kind,
        "relift_round": relift_round,
        "status": closed.get("status"),
        "diagnostic": closed.get("diagnostic"),
        "full_feedback_path": "/input/feedback.json",
        "categories": categories,
    }
    if isinstance(observer, Mapping):
        primary_category = observer.get("primary_difference_category")
        if isinstance(primary_category, str) and primary_category:
            frontier["primary_difference_category"] = primary_category[:64]
    candidate_check_feedback = _safe_candidate_check_feedback(
        closed.get("candidate_check_feedback")
    )
    if candidate_check_feedback is not None:
        frontier["candidate_check_feedback"] = candidate_check_feedback
    preparation_receipt = _safe_preparation_receipt(
        closed.get("preparation_receipt")
    )
    if preparation_receipt is not None:
        frontier["preparation_receipt"] = preparation_receipt
    for key in ("open_guidance", "resolved_guidance"):
        rows = _safe_rule_path_rows(closed.get(key))
        if rows:
            frontier[key] = rows
    projected_open = _safe_rule_path_rows(open_guidance)
    if projected_open:
        frontier["open_guidance"] = projected_open
    projected_resolved = _safe_rule_path_rows(resolved_guidance)
    if projected_resolved:
        frontier["resolved_guidance"] = projected_resolved
    if isinstance(candidate_delta, Mapping) or isinstance(
        closed.get("candidate_delta"), Mapping
    ):
        frontier["candidate_delta_receipt"] = {
            "present": True,
            "path": "/input/feedback.json",
            "authority": "diagnostic only; never a repair template",
        }
    if "unresolved_gate1" in closed:
        frontier["unresolved_gate1_query"] = (
            "resources/read intentir://lift/feedback/detail "
            "path=diagnostics.unresolved_gate1"
        )
    return frontier


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
    safe_environment_evidence = {
        key: dict(value)
        for key in (
            "authoring",
            "canonical_constructor",
            "mcp_semantic_construction",
            "mcp",
        )
        if isinstance(info, Mapping)
        and isinstance((value := info.get(key)), Mapping)
    }
    return {
        "backend": MINISWE_LIFT_BACKEND,
        "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
        "exit_status": result.exit_status,
        "model_turns": result.model_calls,
        "physical_provider_calls": result.physical_provider_calls,
        "transport_retries": result.transport_retries,
        "elapsed_ms": result.elapsed_ms,
        "fresh_session": True,
        "workflow": "single",
        "stage_metrics": dict(getattr(result, "stage_metrics", {})),
        "candidate_check": (
            dict(candidate_check)
            if isinstance(candidate_check, Mapping)
            else {}
        ),
        "agent_runtime": (
            dict(agent_runtime) if isinstance(agent_runtime, Mapping) else {}
        ),
        **safe_environment_evidence,
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
        summary = [
            item
            for item in _safe_rule_path_rows(rule_diagnostics)
            if item.get("rule_code") != "check_budget_exhausted"
            and isinstance(item.get("path"), str)
        ]
        if summary:
            return {"protocol": protocol, "diagnostics": summary}
    return None


def _input_files(
    request: SCRCLiftRequest,
    *,
    source_ttir_observation: Mapping[str, Any] | None = None,
    observed_construction: CanonicalConstruction | None = None,
) -> dict[str, str]:
    packet = json.loads(request.user_input)
    selected_frontend = str(packet.get("frontend") or "triton").casefold()
    selected_source_kind = str(packet.get("source_kind") or "python").casefold()
    if (
        source_ttir_observation is not None
        and (selected_frontend != "triton" or selected_source_kind != "python")
    ):
        raise ValueError(
            "source_ttir_observation is valid only for Triton/TTIR Python source"
        )
    anonymous_source = packet.get("anonymous_source")
    if not isinstance(anonymous_source, str) or not anonymous_source:
        raise ValueError("prepared Lift packet lacks anonymous Source")
    if request.source_facts is None:
        raise ValueError("MCP Lift workspace requires canonical SourceFacts")
    # The provider prompt may contain an equivalent compact view, but the
    # mounted MCP authority must be serialized from the request-owned facts,
    # not from a second packet copy.
    harness_view = request.source_facts.compiler_view()
    feedback = packet.get("relift")
    # Every attempt shares one bounded semantic-entry input shape. Prior
    # candidates remain host-side; the provider receives only a small receipt
    # and rule-level feedback.
    feedback_for_mount = (
        dict(copy.deepcopy(feedback)) if isinstance(feedback, Mapping) else {}
    )
    feedback_frontier = None
    if isinstance(feedback, Mapping):
        feedback_frontier = _feedback_frontier(
            failure_kind=(
                feedback.get("failure_kind")
                if isinstance(feedback.get("failure_kind"), str)
                else None
            ),
            relift_round=(
                feedback.get("relift_round")
                if isinstance(feedback.get("relift_round"), int)
                and not isinstance(feedback.get("relift_round"), bool)
                else None
            ),
            diagnostics=(
                feedback.get("diagnostics")
                if isinstance(feedback.get("diagnostics"), Mapping)
                else None
            ),
            open_guidance=feedback.get("open_guidance"),
            resolved_guidance=feedback.get("resolved_guidance"),
            candidate_delta=feedback.get("candidate_delta"),
        )
    if isinstance(feedback_for_mount, dict):
        # Every attempt starts from the Source semantic entry. Compatibility
        # lifecycle keys may remain in old feedback, but no prior candidate is
        # mounted as an editing surface.
        for key in tuple(feedback_for_mount):
            if str(key).startswith("previous_"):
                feedback_for_mount.pop(key, None)
        feedback_for_mount.pop("repair_base", None)
        diagnostics = feedback_for_mount.get("diagnostics")
        # Relift consumes the bounded typed Gate1 frontier, never the previous
        # source-blind implementation.  Strip legacy nested payloads as well so
        # old retained feedback cannot reopen a Replay -> Lift information path.
        if isinstance(diagnostics, dict):
            diagnostics.pop("replay_counterexample", None)
            unresolved = diagnostics.get("unresolved_gate1")
            nested = (
                unresolved.get("diagnostics")
                if isinstance(unresolved, dict)
                else None
            )
            if isinstance(nested, dict):
                nested.pop("replay_counterexample", None)
    # This control metadata fixes one authoring entry for every attempt.
    authoring_entry = {
        "mode": "semantic_entry",
        "resource": "intentir://lift/source/brief",
        "editable": False,
        "template_only": True,
        "submit_forbidden": False,
        "authority": (
            "Read the exact semantic authoring entry; the host retains the "
            "mechanical root/J/C/pi construction."
        ),
    }
    feedback_for_mount["authoring_entry"] = authoring_entry
    if not isinstance(feedback_frontier, dict):
        feedback_frontier = {
            "protocol": "intentir.lift-feedback-frontier.v1",
            "categories": [],
            "authoring_entry": authoring_entry,
            "authority": (
                "No diagnostic frontier is available; use the one canonical "
                "authoring entry above."
            ),
        }
    else:
        feedback_frontier["authoring_entry"] = authoring_entry
    from minisweagent import package_dir

    preflight = render_canonical_preflight()
    schema_query = (
        Path(package_dir) / "config" / "intentir_schema_query.py"
    ).read_text(encoding="utf-8")
    context_query = (
        Path(package_dir) / "config" / "intentir_context_query.py"
    ).read_text(encoding="utf-8")
    # The MCP workspace, semantic preflight, and canonical constructor must
    # consume one exact observed Source authority.  Recompute only to verify
    # the caller-provided transaction identity; do not let this serializer
    # privately select a different occurrence ledger from finalize.
    expected_construction = _construction_with_source_observation(
        request,
        source_ttir_observation,
    )
    construction = (
        expected_construction
        if observed_construction is None
        else observed_construction
    )
    if (
        construction.source_facts.digest != request.source_facts.digest
        or construction.authority_digest != expected_construction.authority_digest
    ):
        raise ValueError(
            "mounted Source construction differs from semantic finalize authority"
        )
    if construction.transfer_ledger.status != "pass":
        raise ValueError(
            "Source preparation did not produce a closed transfer ledger: "
            f"{construction.transfer_ledger.reason}"
        )
    authoring_blockers = _source_authoring_blockers(construction)
    if authoring_blockers:
        raise ValueError(
            "Source preparation has no legal predicate authoring representation: "
            + authoring_blockers[0]["rule_code"]
        )
    readme = """# IntentIR Lift

This is one fresh, isolated Lift owner. Use only `mcp_call_tool`: direct
`/input` reads are unavailable. Replay is a separate fresh process.

Every fresh Lift attempt: read `intentir://lift/source/brief`, then the required
`intentir://lift/source/transfer-ledger` before construction. It is the sole
Source-order access/fill/owner/output authority; do not substitute a source
page. At most one `schema_lookup(op:NAME)` is allowed. When a feedback frontier
exists, use only its rule/path/category guidance; rejected drafts and previous
artifacts are never repair templates.

MCP authors only semantic J: source `t*`, local `op*/r*/dim*/access*`, operation
graph, accesses, predicates/fills, structured control, output `{value}` choices,
and optional J-proven normalization classes. It never writes schema, C, ABI,
canonical handles, scopes, effects, returns, output attachments, or π bindings.
Preserve Source transfer order, every masked fill, producer-before-consumer SSA,
and complete loop carry/body/yield semantics.

Write exactly once in the schema-derived begin/declare/finalize order published
by the brief, including every required graph/access/output declaration. Free
candidate JSON and patches are invalid. Finalize invokes the host constructor and
fail-closed CandidateCheck; a passing preflight is not Replay, Gate1, or Gate2
success.
"""
    files = {
        "README.md": readme,
        "source.py": anonymous_source,
        # The alpha-renamer gives the selected device entry the fixed anonymous
        # spelling ``scrc_k0``.  context_query must select that mounted name,
        # rather than the privileged pre-anonymization SourceFacts symbol:
        # otherwise it traverses helpers in file order and marks the entry as
        # helperN, losing the reliable connection between entry-local aliases,
        # Source transfers, and provider-visible helper-call facts.  This is
        # deliberately only an anonymous entry selector, not a second Source
        # or compiler payload.
        "source_projection.json": json.dumps(
            {"function": "scrc_k0"},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        # These files are read through JSON-aware lookup tools.  Keep the
        # mounted transport compact; the semantic values are unchanged and
        # the focused query commands provide readable bounded projections.
        "harness_view.json": json.dumps(
            harness_view, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "canonical_construction.json": (
            json.dumps(
                construction.to_json(),
                indent=2,
                # The mounted context reader validates allocator declaration
                # order against the canonical namespace authority.  Keep the
                # host allocator order instead of alphabetically reserializing
                # its mapping keys.
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n"
        ),
        "feedback.json": json.dumps(
            feedback_for_mount, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "feedback_frontier.json": json.dumps(
            feedback_frontier, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "schema/artifact.schema.json": json.dumps(
            scrc_lift_core_json_schema(),
            indent=2,
            # Preserve the canonical declaration order.  The mounted MCP
            # reader compares schema/profile field order as part of the wire
            # contract; sorting this object would turn the schema into a
            # second, alphabetically ordered declaration.
            sort_keys=False,
            ensure_ascii=False,
        )
        + "\n",
        "schema/canonical_skeleton.json": json.dumps(
            canonical_artifact_skeleton(),
            indent=2,
            sort_keys=False,
            ensure_ascii=False,
        )
        + "\n",
        "schema/canonical_wire_bundle.json": (
            json.dumps(
                canonical_wire_bundle(),
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n"
        ),
        "schema/canonical_source_wire_bundle.json": (
            json.dumps(
                canonical_source_wire_bundle(construction),
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n"
        ),
        # This mounted fallback is the same schema-derived semantic projection
        # used by Responses, source-brief, ingress, and schema_lookup.  Do not
        # mount a second prose contract: it would become another authoring
        # authority and inflate a fresh Lift owner's context.
        "knowledge/lift_contract.md": render_canonical_wire_contract(),
        "knowledge/intent_ops.md": render_canonical_operation_reference(),
        "knowledge/preflight.md": preflight,
        "tools/schema_query.py": schema_query,
        "tools/context_query.py": context_query,
    }
    if source_ttir_observation is not None:
        files["source_ttir.json"] = (
            json.dumps(
                dict(source_ttir_observation),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    return files


@dataclass
class MiniSWELiftAdapter:
    """Run each Lift invocation independently with explicit relift state only."""

    provider: ProviderConfig | None = None
    compiler_view: Any = None
    effort: str | None = None
    source_symbol: str | None = None
    source_kind: str = "python"
    source_parameter_order: tuple[str, ...] = ()
    source_to_opaque: Mapping[str, str] | None = None
    workflow: str = "single"
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    # None means that the Lift owner has no implicit wall-time deadline.  A
    # caller may still provide a positive value for an experiment-local cap.
    session_wall_time_limit_seconds: float | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    _session_workspace: Any = field(default=None, init=False, repr=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def close(self) -> None:
        workspace = self._session_workspace
        self._session_workspace = None
        if workspace is not None:
            workspace.cleanup()

    def _begin_fresh_agent_invocation(self) -> None:
        """Drop all agent/workspace state before a fresh Lift invocation.

        Only bounded rule-level feedback may cross attempts. The workspace and
        Responses model are rebuilt, and no prior candidate or conversation is
        passed to ``run_intentir_lift``.
        """

        self.close()
        self._session_id = uuid.uuid4().hex

    def _create_session_workspace(
        self,
        files: Mapping[str, str | bytes],
    ) -> bool:
        """Create the isolated MCP workspace for one agent invocation."""

        from minisweagent.environments.intentir_lift import IntentIRLiftWorkspace

        incoming = dict(files)
        if self._session_workspace is not None:
            raise RuntimeError(
                "Lift invocation workspace was not cleared before agent start"
            )
        self._session_workspace = IntentIRLiftWorkspace.create(incoming)
        return True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _read_diagnostic_candidate_receipt(
        workspace: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Discard rejected bytes locally and return only shape/hash receipts."""

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                workspace.read_candidate(),
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict):
                return None, None
            candidate_bytes = json.dumps(
                candidate,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
            return None, None
        return (
            scrc_candidate_delta_receipt(candidate),
            {
                "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                "bytes": len(candidate_bytes),
            },
        )

    def _relift_context(self, feedback: Any) -> SCRCReliftContext | None:
        return scrc_relift_context_from_feedback(
            feedback,
            relift_round=max(1, len(self._records) + 1),
        )

    def _invocation_worker_config(self) -> dict[str, Any]:
        config = copy.deepcopy(dict(self.worker_config))
        wall_time_seconds: float | None = None
        if self.session_wall_time_limit_seconds is not None:
            if (
                isinstance(self.session_wall_time_limit_seconds, bool)
                or not isinstance(self.session_wall_time_limit_seconds, (int, float))
                or not math.isfinite(float(self.session_wall_time_limit_seconds))
                or self.session_wall_time_limit_seconds <= 0
            ):
                raise ValueError(
                    "Lift owner session wall-time limit must be None or a finite positive number"
                )
            wall_time_seconds = float(self.session_wall_time_limit_seconds)
        agent = config.get("agent")
        agent_config = dict(agent) if isinstance(agent, Mapping) else {}
        # Let run_intentir_lift merge the canonical YAML agent budget.  Only a
        # caller-supplied invalid/unlimited override is clamped here; an
        # omitted step_limit must remain omitted so the YAML value is the one
        # actually used by DefaultAgent.
        if "step_limit" in agent_config:
            step_limit = agent_config["step_limit"]
            if (
                isinstance(step_limit, bool)
                or not isinstance(step_limit, int)
                or step_limit <= 0
            ):
                agent_config["step_limit"] = DEFAULT_MINISWE_LIFT_STEP_LIMIT
        agent_config["wall_time_limit_seconds"] = (
            max(1, math.ceil(wall_time_seconds))
            if wall_time_seconds is not None
            else 0
        )
        config["agent"] = agent_config
        return config

    def _session_evidence(
        self,
        evidence: Mapping[str, Any],
        *,
        fresh_session: bool,
        result: Any,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        value = dict(evidence)
        value.update(
            {
                "fresh_session": fresh_session,
                "persistent_owner": False,
                "owner_session_id": self._session_id,
                "agent_memory": "fresh_invocation_explicit_mcp_only",
                "reasoning_effort": reasoning_effort,
                "continued_from_messages": result.continued_from_messages,
                "owner_active_elapsed_ms": float(result.elapsed_ms),
                "owner_model_turns": int(result.model_calls),
                "owner_physical_provider_calls": int(
                    result.physical_provider_calls
                ),
                "owner_transport_retries": int(result.transport_retries),
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
        if self.workflow != "single":
            raise ValueError(
                "MiniSWE Lift has one supported workflow: 'single'; staged "
                "planner/builder/reviewer ownership was retired by Task13"
            )
        # Only bounded rule-level feedback may cross this boundary. Do not
        # carry a prior candidate, model, response chain, or MCP workspace.
        self._begin_fresh_agent_invocation()
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        # Lift requests are unbounded by default.  A positive
        # model_timeout_seconds remains an opt-in cap owned by this caller.
        timeout = self.model_timeout_seconds
        gate_source = getattr(source_program, "gate_source", None)
        source_import_roots = tuple(
            str(root)
            for root in getattr(gate_source, "import_roots", ())
        )
        relift_context = self._relift_context(feedback)
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=relift_context,
            source_symbol=getattr(
                source_program, "lift_source_symbol", self.source_symbol
            ),
            source_kind=getattr(
                source_program, "lift_source_kind", self.source_kind
            ),
            source_parameter_order=tuple(
                getattr(
                    source_program,
                    "lift_parameter_order",
                    self.source_parameter_order,
                )
            ),
            source_to_opaque=(
                dict(
                    getattr(
                        source_program,
                        "lift_source_to_opaque",
                        self.source_to_opaque,
                    )
                )
                if getattr(
                    source_program,
                    "lift_source_to_opaque",
                    self.source_to_opaque,
                ) is not None
                else None
            ),
            source_import_roots=source_import_roots,
            frontend=getattr(request, "frontend", "triton"),
            source_facts=getattr(request, "source_facts", None),
            construction=getattr(request, "construction", None),
        )
        source_ttir_observation = (
            _source_ttir_observation_for_request(prepared)
            if getattr(request, "frontend", "triton").casefold() == "triton"
            else {
                "status": "pass",
                "applicable": False,
                "diagnostics": [],
            }
        )
        observed_construction = _construction_with_source_observation(
            prepared,
            source_ttir_observation,
        )
        round_index = len(self._records) + 1
        fresh_owner_session = True
        effective_worker_config = self._invocation_worker_config()
        configured_routine_effort = (
            effective_worker_config.get("model", {}).get("routine_reasoning_effort")
            if isinstance(effective_worker_config.get("model"), Mapping)
            else None
        )
        if configured_routine_effort is not None and configured_routine_effort != effective_effort:
            raise ValueError(
                "Lift reasoning effort is overridden by a different routine "
                "effort; make the override explicit or set it to null"
            )
        configured_owner_remaining = effective_worker_config.get("agent", {}).get(
            "wall_time_limit_seconds"
        )
        if (
            isinstance(configured_owner_remaining, (int, float))
            and not isinstance(configured_owner_remaining, bool)
            and configured_owner_remaining > 0
            and isinstance(timeout, (int, float))
            and not isinstance(timeout, bool)
        ):
            timeout = min(timeout, float(configured_owner_remaining))
        candidate_check_context = scrc_candidate_check_context(prepared)
        deterministic_repairs: list[dict[str, Any]] = []

        def construct_canonical_candidate(candidate_text: str) -> tuple[
            str, tuple[Mapping[str, Any], ...]
        ]:
            constructed_text, repairs = _construct_scrc_candidate(
                candidate_text,
                prepared,
                observation=source_ttir_observation,
                relift_context=relift_context,
            )
            deterministic_repairs.extend(
                dict(item) for item in repairs if isinstance(item, Mapping)
            )
            return constructed_text, repairs

        try:
            from minisweagent.run.intentir_lift import (
                IntentIRLiftRunError,
                create_intentir_lift_model,
                run_intentir_lift,
            )
            from minisweagent.exceptions import MiniSWERequestCancelled
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires `pip install -e '.[lift-agent,verify]'`",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc

        try:
            input_files = _input_files(
                prepared,
                source_ttir_observation=source_ttir_observation,
                observed_construction=observed_construction,
            )
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            # A closed Source/Harness construction bundle is a prerequisite
            # for an MCP owner.  Do not convert a missing ledger/seed into a
            # relift attempt: no candidate has been authored yet.
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                f"Lift Source/Harness preparation failed: {type(exc).__name__}: {exc}",
                evidence={
                    "backend": MINISWE_LIFT_BACKEND,
                    "failure_class": "lift_source_harness_preparation",
                    "candidate_present": False,
                },
            ) from exc
        try:
            fresh_workspace = self._create_session_workspace(input_files)
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires the lift-agent workspace dependency",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc
        workspace = self._session_workspace
        candidate_path = workspace.candidate_path
        if candidate_path.exists():
            candidate_path.unlink()
        result = None
        try:
            model = create_intentir_lift_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
                prompt_cache_key=f"intentir-lift-{self._session_id}",
            )
            model_config_kwargs = getattr(
                getattr(model, "config", None), "model_kwargs", None
            )
            if not isinstance(model_config_kwargs, Mapping) or model_config_kwargs.get(
                "reasoning_effort"
            ) != effective_effort:
                raise ValueError(
                    "Lift reasoning effort was not preserved in the model request"
                )
            set_cancel_requested = getattr(model, "set_cancel_requested", None)
            if callable(set_cancel_requested):
                set_cancel_requested(self.stop_requested)
            model_kwargs = getattr(
                getattr(model, "config", None), "model_kwargs", None
            )
            if isinstance(model_kwargs, dict):
                # This model belongs only to the current fresh invocation; the
                # request deadline follows its local remaining budget.
                model_kwargs["timeout"] = timeout
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
                "semantic_candidate_checker": (
                    lambda candidate_text: check_scrc_semantic_candidate_surface(
                        candidate_text,
                        construction=observed_construction,
                    )
                ),
                "candidate_constructor": construct_canonical_candidate,
            }
            result = run_intentir_lift(
                **run_kwargs,
            )
        except IntentIRLiftRunError as exc:
            result = exc.result
            safe_trajectory = _round_trajectory(
                _sanitize_trajectory(result.trajectory, provider.api_key),
                result.continued_from_messages,
            )
            evidence = self._session_evidence(
                _worker_evidence(result, safe_trajectory),
                fresh_session=fresh_owner_session,
                result=result,
                reasoning_effort=effective_effort,
            )
            evidence["deterministic_repairs"] = list(deterministic_repairs)
            candidate_delta, rejected_candidate_receipt = (
                self._read_diagnostic_candidate_receipt(
                    workspace
                )
            )
            if rejected_candidate_receipt is not None:
                evidence["rejected_candidate_receipt"] = dict(
                    rejected_candidate_receipt
                )
            # CandidateCheck and preparation receipts have already been
            # projected above. A rejected owner trajectory is neither repair
            # input nor durable evidence and must not cross this backend.
            evidence.pop("trajectory", None)
            diagnostic = str(exc)
            from intentir.verify.scrc_runner import CandidateLiftError

            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=evidence,
                )
            )
            if isinstance(exc.cause, MiniSWERequestCancelled):
                raise CandidateLiftError(
                    {
                        "diagnostic": diagnostic,
                        "failure_class": "candidate_superseded",
                        "winner_branch_id": exc.cause.winner_branch_id,
                    },
                    physical_calls=result.physical_provider_calls,
                ) from exc
            candidate_check_feedback = _last_candidate_check_feedback(evidence)
            no_candidate_preparation_exit = (
                candidate_delta is None
                and candidate_check_feedback is None
                and result.exit_status
                in {
                    "LimitsExceeded",
                    "AgentStepLimitExceeded",
                    "AgentCostLimitExceeded",
                    "TimeExceeded",
                    "RepeatedFormatError",
                    "LiftPreparationFailed",
                }
            )
            authoring = evidence.get("authoring")
            preparation_receipts = (
                authoring.get("preparation_failures")
                if isinstance(authoring, Mapping)
                else None
            )
            preparation_receipt = (
                dict(preparation_receipts[-1])
                if isinstance(preparation_receipts, list)
                and preparation_receipts
                and isinstance(preparation_receipts[-1], Mapping)
                else None
            )
            retryable_authoring_preparation = (
                candidate_delta is None
                and candidate_check_feedback is None
                and result.exit_status == "LiftPreparationFailed"
                and result.model_calls >= 1
                # The worker may wrap the environment exception before it
                # reaches this adapter.  The exact exit status plus the
                # bounded environment-owned preparation receipt is the stable
                # cross-layer identity; never depend solely on Python wrapper
                # type preservation here.
                and (
                    isinstance(exc.cause, LiftPreparationFailed)
                    or preparation_receipt is not None
                )
            )
            if retryable_authoring_preparation:
                # This owner reached the bounded local MCP state machine and
                # failed without producing candidate bytes. It is therefore
                # safe to spend the next logical Lift call on a wholly fresh
                # semantic-entry owner, but there is no draft state to carry
                # forward. Provider/transport errors and all global
                # limits remain infrastructure failures below.
                diagnostics: dict[str, Any] = {
                    "diagnostic": "fresh Lift authoring preparation failed",
                    "failure_class": "lift_authoring_preparation",
                    "candidate_present": False,
                    "fresh_retry_required": True,
                }
                if preparation_receipt is not None:
                    diagnostics["preparation_receipt"] = preparation_receipt
                raise CandidateLiftError(
                    diagnostics,
                    physical_calls=result.physical_provider_calls,
                    terminal=False,
                ) from exc
            if (
                exc.cause is not None
                and not isinstance(exc.cause, CandidateRejected)
                and result.exit_status != "InvalidSubmission"
            ) or result.model_calls < 1 or no_candidate_preparation_exit:
                # A provider/agent response that never produced a complete
                # model turn is a preparation-stage failure.  It contains no
                # candidate and no semantic evidence, so do not spend another
                # logical Lift call trying to repair it.  The same is true
                # when the bounded MCP owner exhausts its action budget
                # before CandidateCheck: that is an authoring-preparation
                # failure, not a semantic relift. Return the infrastructure
                # error directly; only a complete candidate may enter
                # CandidateCheck and only a rejected candidate may enter
                # relift feedback.
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                provider_failure = _provider_failure_class(exc.cause)
                preparation_evidence = {
                    key: value
                    for key, value in evidence.items()
                    if key != "trajectory"
                }
                preparation_evidence["failure_class"] = (
                    "lift_agent_preparation"
                    if no_candidate_preparation_exit
                    else "lift_provider_preparation"
                )
                preparation_evidence["candidate_present"] = False
                if provider_failure is not None:
                    preparation_evidence["provider_failure_class"] = (
                        provider_failure
                    )
                raise SCRCInfrastructureError(
                    diagnostic,
                    evidence=preparation_evidence,
                ) from exc
            rejected_diagnostics: dict[str, Any] = {
                "diagnostic": diagnostic,
                "failure_class": (
                    "candidate_rejected"
                    if candidate_check_feedback is not None
                    else "lift_worker_no_candidate"
                ),
                "candidate_present": candidate_delta is not None,
            }
            if candidate_check_feedback is not None:
                rejected_diagnostics["candidate_check_feedback"] = (
                    candidate_check_feedback
                )
            if rejected_candidate_receipt is not None:
                rejected_diagnostics["rejected_candidate_receipt"] = dict(
                    rejected_candidate_receipt
                )
            raise CandidateLiftError(
                rejected_diagnostics,
                physical_calls=result.physical_provider_calls,
                terminal=result.exit_status == "TimeExceeded",
                candidate_delta=candidate_delta,
            ) from exc
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence={
                        "backend": MINISWE_LIFT_BACKEND,
                        "fresh_session": fresh_owner_session,
                        "persistent_owner": False,
                        "owner_session_id": self._session_id,
                        "agent_memory": "fresh_invocation_explicit_mcp_only",
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
            reasoning_effort=effective_effort,
        )
        worker_evidence["deterministic_repairs"] = list(deterministic_repairs)
        from intentir.verify.scrc_runner import CandidateLiftError

        try:
            artifact = decode_scrc_lift_artifact(result.candidate_text, prepared)
        except CandidateLiftError:
            raise
        except Exception as exc:
            diagnostic = f"{type(exc).__name__}: {exc}"
            candidate_delta, rejected_candidate_receipt = (
                self._read_diagnostic_candidate_receipt(workspace)
            )
            if rejected_candidate_receipt is not None:
                worker_evidence["rejected_candidate_receipt"] = dict(
                    rejected_candidate_receipt
                )
            rejected_worker_evidence = dict(worker_evidence)
            rejected_worker_evidence.pop("trajectory", None)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=rejected_worker_evidence,
                )
            )
            raise CandidateLiftError(
                {
                    "diagnostic": diagnostic,
                    "failure_class": "lift_response",
                },
                physical_calls=result.physical_provider_calls,
                candidate_delta=candidate_delta,
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
            deterministic_repairs=tuple(deterministic_repairs),
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
            prepared.construction,
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
                "persistent_workspace": False,
                "persistent_owner": False,
                "agent_memory": "fresh_invocation_explicit_mcp_only",
                "owner_session_id": self._session_id,
                "continued_from_messages": result.continued_from_messages,
                "owner_active_elapsed_ms": float(result.elapsed_ms),
                "owner_model_turns": int(result.model_calls),
                "owner_physical_provider_calls": int(
                    result.physical_provider_calls
                ),
                "owner_transport_retries": int(result.transport_retries),
                "workflow": self.workflow,
                "artifact_schema": artifact.wire_schema,
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "candidate_check": worker_evidence["candidate_check"],
                "deterministic_repairs": list(deterministic_repairs),
                "agent_runtime": worker_evidence["agent_runtime"],
                "stage_metrics": worker_evidence["stage_metrics"],
            },
        )


__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
]
