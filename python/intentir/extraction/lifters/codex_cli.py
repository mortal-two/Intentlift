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
from .compiler_observation import observe_anonymous_source_ttir
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


CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v14"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v72"
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
input/wire_contract.json. It contains concise semantic ownership rules and the
exact IntentOp vocabulary. Do not search for another schema, tutorial, example,
preflight, or operation catalog: none is authoritative here.

Before authoring the signature or first complete candidate, materialize these
five exact read-only projections once as local proof-irrelevant files:

python .agents/intentir_agent_tool.py lift_harness_abi 1 64 > work/harness_abi.json
python .agents/intentir_agent_tool.py lift_contract core > work/contract_core.json
python .agents/intentir_agent_tool.py lift_contract boundary > work/contract_boundary.json
python .agents/intentir_agent_tool.py lift_contract pi > work/contract_pi.json
python .agents/intentir_agent_tool.py lift_contract dependent > work/contract_dependent.json

Do not print these payloads together with Source or each other: combined terminal
output may be clipped. Read the five small files separately. The first is the
complete anonymous ABI/Harness ledger. The next three project the exact required
keys, fields, enums, defaults, and closed JSON Schema sections for all artifact
object families. The final file projects only closed source-neutral dependent
checker predicates for those objects. They expose no Source meaning and never
generate an artifact field or choose a relation. Use them to form one coordinated
complete draft before the first CandidateCheck; do not use CandidateCheck as a
missing-field or dependent-rule tutorial.

Author the artifact incrementally under work/candidate_parts/ instead of emitting
one giant first patch or writing another Python builder. Each *.json file there is
one model-authored JSON object fragment. Split at recursively disjoint object keys,
for example envelope, J intent, J boundary groups, signature/relations, C, and pi
relation groups. Every leaf and every array element remains authored by you. Run:

python input/assemble_candidate.py

The fixed assembler only recursively joins disjoint JSON object keys and writes
output/candidate.json; it rejects duplicate leaves and never allocates a handle,
adds a field, concatenates an array, or infers J/C/pi semantics. Edit the small
owning fragment after diagnostics and reassemble. Completeness is required only
for the assembled candidate consumed by CandidateCheck and final submission.

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
exact current bytes, followed only by the adjacent anonymous-submission privacy
check. A privacy rejection names the artifact path but never reveals the private
Source spelling. Neither check edits or generates a field. Repair the owning
fragment, reassemble, and continue until the command reports PASS; then end this
Codex turn. The host independently reruns the same boundary after the turn and
may resume this same thread with closed diagnostics. Do not claim TypeCheck,
Replay, Gate1, Gate2, or certification. Do not create any other file under output/.
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
    "Every handle family is dense and declaration ordered from zero: unique tensors "
    "t*, recursive IntentOp ids and node_ids n*, SSA results v*, signature parameters "
    "p*, domains d*, accesses a*, effects e*, scopes q*, storage b*, C slots c*, pi "
    "regions r*, and pi instances i*. An inout output repeats its input t* instead "
    "of allocating another tensor handle. x* order is first appearance across J "
    "tensor shapes. node_ids map n* to exact structural IntentOp paths; pi route, "
    "order, and sharing endpoints are i*, never raw J/C handles.",
    "J.signature parameters are exactly the anonymous Harness p0,p1,... in ABI order. "
    "logical_ref is only a declared t*, declared x*, or null, never v*. A Harness-fixed "
    "constexpr used only for compile-time graph discrimination is a value carrier "
    "with logical_ref=null: author only the selected active J graph and do not create "
    "a selector constant or branch in J. A constexpr used as logical data, extent, "
    "shape, index, mask, predicate, or result must instead bind its authored t*/x*; "
    "only an implementation selection belongs to C.",
    "Harness concrete tensor shapes are valuations, not permission to literalize a "
    "parametric Source dimension. Preserve every declared h* shape contract as x*, "
    "author value_dimension_relations when one integral rank-0 logical value equals "
    "an extent, and author every Source-grounded dimension_relation explicitly.",
    "Preserve Source loops, branches, returns, carries, output multiplicity, access "
    "identity, and literal effect order in J and the corresponding realization "
    "topology in pi. Node declaration order is not effect order.",
    "The J input/output partition and signature modes are exact: input-only, "
    "output-only, and explicit inout may not be silently interchanged. Every declared "
    "output has exact return and output_mapping coverage with matching dtype, shape, "
    "and observable produced value; preserve canonical output ABI order separately "
    "from Source effect order.",
    "Scopes q* are exact lexical dominance objects whose definitions, captures, and "
    "effects contain neither missing nor extra handles. J.storage is optional semantic "
    "identity/lifetime, not an inventory of tensors, SSA values, or loop carries and "
    "not physical placement; keep storage=[] unless Source requires such identity. "
    "If authored, b* membership and lifetime events must cover the complete declared "
    "identity consistently.",
    "A C subject_ref may name only an existing J node, domain, access, scope, or "
    "storage. axis_ref requires subject_ref and must be in that subject's closed axis "
    "domain: reduce axes, loop trip axis, or canonical domain/access axes; scopes and "
    "storage are axis-free. Never infer an attachment or axis from role, shape, Source, "
    "pi, or a unique-looking position.",
    "A J structured loop owns complete trip, carry init/body argument/update/next, "
    "yield, and result semantics. A C-tiled structured loop uses pi.iterations and "
    "pi.loop_carriers in its loop region and preserves J carry declaration order. One "
    "pi region cannot carry both structured iterations/loop_carriers and physical-only "
    "iterations/physical_loop_carriers.",
    "pi.physical_iterations only realizes an existing non-loop J reduce, contraction, "
    "access, or domain with a same-subject/axis positive integer C tile. Its carriers "
    "may be heterogeneous pointer or value slots but each slot has type-consistent "
    "init/state/next/result and array order is the frontend tuple order. When physical "
    "pointer time coexists with a structured logical loop, use a distinct nested "
    "physical-only region rather than inventing extra J state.",
    "A non-null access fill requires a predicated load. Bind ambiguous access views "
    "to their actual node_ref. Predicated sum/max/min consumers require their exact "
    "consumer_extensions relation; computed access guards require an authored "
    "access_guard_relation and explicit Layer-A select/state semantics where needed.",
    "Choose each output result-relation class from the authored floating dependency "
    "graph. Concrete tolerance values are external and never appear in J.",
    "Each transfer_direct or transfer_async instance must be reachable through pi "
    "routes from the matching pointer/tensor signature carrier for its boundary access; "
    "fixed code never inserts an address stage. A sensitive numeric-cast refinement "
    "anchors the explicit cast node, cites its one J authorization, and uses the cast "
    "operand/result dtypes; one pi instance has at most one refinement.",
    "CandidateCheck is source-free parse/Decode/dependent-TypeCheck diagnostics over "
    "the exact current bytes. It neither judges Source semantic completeness nor "
    "repairs or certifies a candidate; Gate1 feedback is the semantic counterexample.",
)


def _compact_intent_op_reference() -> str:
    """Reuse the existing op vocabulary without exposing its legacy top level."""

    start_marker = "Atomic ops (each is one JSON object in a \"body\" list):"
    end_marker = "Hard rules:"
    start = _INTENTIR_OPERATION_REFERENCE.index(start_marker)
    end = _INTENTIR_OPERATION_REFERENCE.index(end_marker, start)
    return _INTENTIR_OPERATION_REFERENCE[start:end].strip()


_CANDIDATE_FRAGMENT_ASSEMBLER = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile

PARTS = Path("work/candidate_parts")
OUTPUT = Path("output/candidate.json")

def merge(left, right, path):
    if not isinstance(right, dict):
        raise SystemExit(f"fragment {path or '<root>'} must be a JSON object")
    for key, value in right.items():
        child = f"{path}.{key}" if path else key
        if key not in left:
            left[key] = value
        elif isinstance(left[key], dict) and isinstance(value, dict):
            merge(left[key], value, child)
        else:
            raise SystemExit(f"duplicate fragment leaf: {child}")

if not PARTS.is_dir():
    raise SystemExit("create work/candidate_parts/*.json first")
paths = sorted(PARTS.glob("*.json"))
if not paths:
    raise SystemExit("work/candidate_parts contains no JSON fragments")
candidate = {}
for path in paths:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"fragment must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    merge(candidate, value, "")
text = json.dumps(
    candidate,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
    allow_nan=False,
) + "\n"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(prefix=".candidate-", dir=OUTPUT.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, OUTPUT)
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps({"status":"pass","fragments":len(paths),"bytes":len(text.encode())}))
'''

_CONTRACT_SECTION_PROFILES = {
    "core": (
        "tensor",
        "parameter",
        "scope",
        "storage",
        "value_dimension_relation",
        "dimension_relation",
        "c_slot",
    ),
    "boundary": (
        "domain",
        "access",
        "broadcast",
        "alias",
        "effect",
        "output_mapping",
        "reduction",
        "index_bound",
        "update",
        "result_relation",
        "consumer_extension",
        "access_guard_relation",
        "authorization",
    ),
    "pi": (
        "pi_region",
        "pi_instance",
        "pi_route",
        "pi_route_extent_transform",
        "pi_order",
        "pi_sharing",
        "pi_iteration",
        "pi_loop_carrier",
        "pi_physical_iteration",
        "pi_physical_loop_carrier",
        "pi_terminal_control",
        "pi_refinement",
    ),
}


_DEPENDENT_CONTRACT_RULES = {
    "c_attachment": (
        "subject_ref, when present, names one declared schedule-addressable J node, domain, access, scope, or storage",
        "axis_ref requires subject_ref and belongs to AxisRefs_v1(J,subject_ref); scopes and storage have no axes",
    ),
    "reduction_boundary": (
        "node_ref names an existing semantic reduce or inclusive scan node",
        "one semantic reduce or scan has at most one reduction boundary contract",
    ),
    "consumer_extension": (
        "consumer_ref names a sum/max/min reduce or inclusive scan and operand_index selects one declared string operand",
        "tile_axis equals the consumer's closed local axis; outside dtype equals that operand dtype and outside value is the consumer's two-sided identity",
        "access_ref names one predicate-bearing load whose provenance unambiguously reaches that operand",
        "a required predicate-bearing reduce/scan operand has exactly one explicit outside-domain identity authority; masked_contribution and consumer_extension cannot duplicate it",
    ),
    "realization_authorization": (
        "subject_ref names a declared non-absent J carrier, never an authorization handle",
        "numeric_cast subject_ref names the explicit unary cast node and source_dtype/target_dtype exactly equal its operand/result dtypes",
        "for other typed authorizations, an explicit source_dtype equals the subject carrier dtype",
    ),
    "route": (
        "source and target name declared pi instances and the combined routes plus order graph is acyclic",
        "when target anchors a J node, every slot is a valid target operand ordinal and the source carrier type equals that operand type",
    ),
    "route_extent_transform": (
        "source is a realize instance anchored to one required positive-integer C slot",
        "target is a realize instance anchored to the J domain and axis attached by that same C slot",
    ),
    "physical_loop_carrier": (
        "region is one placement=loop region declared by physical_iterations",
        "init is visible from a strict ancestor; state and next are in the exact loop region; result is in its direct parent",
        "init/state/next/result each anchor one typed J signature, tensor, value, node, or access carrier; node carriers have exactly one typed result",
        "all four roles have identical dtype, rank, and pointer-or-value carrier class",
        "routes contain init->state and, unless state==next, state->next",
        "state reaches at least one transfer_direct, transfer_async, update_atomic, update_privatized, or synchronize instance through routes",
    ),
    "sharing": (
        "source is not one of its consumers and every consumer has a direct route from source",
    ),
}


def _compact_schema_fragment(value: Any) -> Any:
    """Remove prose while retaining exact structure, enums, and defaults."""

    if isinstance(value, Mapping):
        return {
            str(key): _compact_schema_fragment(item)
            for key, item in value.items()
            if key not in {"description", "title", "examples"}
        }
    if isinstance(value, list):
        return [_compact_schema_fragment(item) for item in value]
    return value


def _contract_object_shapes(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project required/allowed keys for the artifact's enclosing objects."""

    root_properties = schema["properties"]
    j = root_properties["j"]
    j_properties = j["properties"]
    intent = j_properties["intent"]
    intent_function = intent["properties"]["intent_function"]
    objects = {
        "root": schema,
        "j": j,
        "j.intent": intent,
        "j.intent.intent_function": intent_function,
        "j.boundary": j_properties["boundary"],
        "j.signature": j_properties["signature"],
        "c": root_properties["c"],
        "pi": root_properties["pi"],
    }
    result: dict[str, Any] = {}
    for name, item in objects.items():
        properties = item.get("properties", {})
        result[name] = {
            "required": list(item.get("required", [])),
            "fields": list(properties),
            "additional_properties": item.get("additionalProperties"),
        }
    return result


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
    wire_contract = {
        "protocol": "intentir.codex-lift-wire-contract.v2",
        "authority": (
            "source-neutral authoring reference only; the host independently "
            "runs the full schema, Decode, and dependent TypeCheck"
        ),
        "semantic_rules": list(_COMPACT_WIRE_RULES),
        "intent_op_authoring_reference": _compact_intent_op_reference(),
        "closed_contract_queries": [
            "python .agents/intentir_agent_tool.py lift_contract core",
            "python .agents/intentir_agent_tool.py lift_contract boundary",
            "python .agents/intentir_agent_tool.py lift_contract pi",
            "python .agents/intentir_agent_tool.py lift_contract dependent",
        ],
    }
    files = {
        "source.py": complete["source.py"],
        "harness_view.json": complete["harness_view.json"],
        "feedback.json": complete["feedback.json"],
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "wire_contract.json": json.dumps(
            wire_contract,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n",
        "candidate_check.json": "null\n",
        "assemble_candidate.py": _CANDIDATE_FRAGMENT_ASSEMBLER,
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
section. Repair the complete model-authored (J,C,pi) fragments against the closed
external feedback and Source, run `python input/assemble_candidate.py`, then run
`python .agents/intentir_agent_tool.py candidate_check`,
and end only after it
passes for the complete current output/candidate.json. This is not a new writer
and not a patch-only artifact. The host alone will rerun CandidateCheck and the
formal SCRC gates.
"""

_CANDIDATE_CHECK_CONTINUATION_PROMPT = """The host submission preflight rejected the current exact candidate bytes.
Read input/candidate_check.json, then repair the owning files under
work/candidate_parts/ in this same thread. Make coordinated J/C/pi changes when
required; do not merely silence the first path. Rerun
`python input/assemble_candidate.py` and then
`python .agents/intentir_agent_tool.py candidate_check`
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
                "assembler": "input/assemble_candidate.py",
                "candidate_parts": "work/candidate_parts/*.json",
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
        contract_schema = scrc_lift_core_json_schema()
        contract_sections = schema_sections(contract_schema)
        contract_shapes = _contract_object_shapes(contract_schema)

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

        def contract_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            try:
                if len(arguments) != 1:
                    raise ValueError(
                        "lift_contract accepts exactly one profile/section name"
                    )
                if arguments[0] == "dependent":
                    return {
                        "protocol": "intentir.lift-dependent-contract-view.v1",
                        "status": "pass",
                        "rules": _DEPENDENT_CONTRACT_RULES,
                        "authority": (
                            "exact source-neutral projection of necessary predicates "
                            "already enforced by dependent TypeCheck; no relation, "
                            "handle, value, or artifact field is selected or generated"
                        ),
                    }
                names: list[str] = []
                for argument in arguments:
                    if argument in _CONTRACT_SECTION_PROFILES:
                        requested = _CONTRACT_SECTION_PROFILES[argument]
                    elif argument in contract_sections:
                        requested = (argument,)
                    else:
                        choices = sorted(
                            set(_CONTRACT_SECTION_PROFILES)
                            | set(contract_sections)
                            | {"dependent"}
                        )
                        raise ValueError(
                            f"unknown contract section {argument!r}; choose {choices}"
                        )
                    for name in requested:
                        if name not in names:
                            names.append(name)
                selected = {
                    name: _compact_schema_fragment(contract_sections[name])
                    for name in names
                }
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "protocol": "intentir.lift-contract-view.v1",
                    "status": "fail",
                    "diagnostic": str(exc),
                }
            return {
                "protocol": "intentir.lift-contract-view.v1",
                "status": "pass",
                "object_shapes": contract_shapes,
                "sections": selected,
                "authority": (
                    "exact source-neutral projection of the authoritative JSON "
                    "Schema; no artifact field, handle, or semantic choice is "
                    "generated"
                ),
            }

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
                "lift_contract": contract_tool,
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
