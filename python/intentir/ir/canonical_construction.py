"""Shared Source/Harness construction authority.

This module contains only transport-neutral facts.  It deliberately does not
import a frontend, backend, Gate runner, or provider.  Source adapters and the
Lift ingress can therefore share the same facts, transfer occurrences, and
handle allocator without sharing their execution implementation.

The objects here are host-side authority records.  They are not additional J,
C, or pi fields and are never inserted into the proof artifact.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from intentir.ir.intent_op_contract import validate_intent_op_structure


_POINTER_RE = re.compile(r"^arg\[(\d+)\]:")
_HANDLE_PREFIXES = (
    "t", "v", "x", "d", "n", "a", "e", "q", "b", "s", "p", "c", "u",
)
# One stable declaration-table order is used by Source-owned construction and
# candidate-local extension.  Empty rows are intentional: an empty namespace
# is still owned by the same allocator and must not be silently re-created by
# a downstream repair helper.
_CANONICAL_NAMESPACE_ORDER = (
    "p", "c", "t", "v", "x", "d", "n", "a", "e", "q", "b", "s", "u",
)
# Public projections must import this order instead of spelling a namespace
# subset locally.  In particular, an empty candidate namespace is still owned
# by the same allocator and must not be re-created by an MCP or provider path.
CANONICAL_HANDLE_NAMESPACE_ORDER = _CANONICAL_NAMESPACE_ORDER
CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES = ("p", "c", "t", "x")
CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES = tuple(
    prefix
    for prefix in _CANONICAL_NAMESPACE_ORDER
    if prefix not in CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES
)
CANONICAL_HANDLE_NAMESPACE_PATTERNS = {
    prefix: rf"^{prefix}[0-9]+$"
    for prefix in _CANONICAL_NAMESPACE_ORDER
}

# Proof-artifact wire ownership is declared here, alongside the construction
# records and boundary table.  Gate1/model and the provider schema import these
# declarations; neither consumer keeps a second root/J/access field list.
CANONICAL_ARTIFACT_ROOT_FIELDS = ("schema", "j", "c", "pi")
CANONICAL_J_WIRE_FIELDS = (
    "kernel_id",
    "intent",
    "node_ids",
    "boundary",
    "signature",
    "scopes",
    "storage",
    "dimension_constraints",
    "value_dimension_relations",
    "dimension_relations",
)
CANONICAL_C_WIRE_FIELDS = ("slots", "values")
CANONICAL_PI_WIRE_FIELDS = (
    "schema",
    "normalization_classes",
    "program_axes",
    "program_bindings",
    "iteration_bindings",
    "control_bindings",
    "materialization_bindings",
    "access_representation_bindings",
)
CANONICAL_PI_REQUIRED_FIELDS = ("schema", "program_axes", "program_bindings")
CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS = (
    "access_id",
    "kind",
    "tensor",
    "predicate",
    "fill_value",
    "node_ref",
    "index_expression",
    "semantic_access_authorization",
    "semantic_owner_authorization",
    "semantic_guard_authorization",
)
CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS = (
    "access_id",
    "kind",
    "tensor",
    "index_expression",
)
# These are the namespaces authored by a semantic J candidate.  ``p*/c*/t*/x*``
# are SourceFacts-owned declarations: a candidate may reference them but may
# neither allocate nor renumber them.  In particular, x* dimensions are never
# a candidate-local first-appearance namespace.
CANONICAL_AUTHORED_HANDLE_PREFIXES = (
    "v", "d", "n", "a", "e", "q", "b", "s", "u",
)
_ABI_HANDLE_RE = re.compile(r"(?:p|c)[0-9]+$")
_SOURCE_MEMORY_CALLS = frozenset(
    {
        "load",
        "store",
        "atomic_add",
        "atomic_and",
        "atomic_cas",
        "atomic_max",
        "atomic_min",
        "atomic_or",
        "atomic_xchg",
        "atomic_xor",
    }
)


@dataclass(frozen=True)
class SourceSemanticOccurrence:
    """One opaque Source semantic anchor eligible for compiler attestation.

    ``token`` is the only provider-visible identity.  The source span exists
    solely in the in-memory lexical ledger so the exact Source compilation can
    attest a counterpart through its debug location sidecar; it is never
    serialized.  An occurrence without a compiler counterpart is still a
    useful Source fact, but it must not authorize a provider-local owner or
    computed guard.
    """

    token: str
    role: str
    operation_family: str
    source_line: int | None = field(default=None, repr=False, compare=False)
    source_column: int | None = field(default=None, repr=False, compare=False)
    binding_mode: str = field(default="direct", repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token, str)
            or re.fullmatch(r"so_[0-9a-f]{32}", self.token) is None
            or self.role not in {"access", "consumer", "predicate"}
            or self.operation_family not in {
                "map_binary", "map_unary", "select", "cast", "reduce",
                "iota", "reshape", "transpose", "concat", "flip", "sort",
                "memory_load", "memory_store", "memory_update",
            }
            or self.binding_mode not in {"direct", "exact_local_binding"}
        ):
            raise ValueError("invalid Source semantic occurrence")
        for value, name in (
            (self.source_line, "source line"),
            (self.source_column, "source column"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"invalid Source semantic occurrence {name}")
        if (self.source_line is None) != (self.source_column is None):
            raise ValueError("Source semantic occurrence span is incomplete")

    def to_json(self) -> dict[str, str]:
        return {
            "token": self.token,
            "role": self.role,
            "operation_family": self.operation_family,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceSemanticOccurrence":
        copied = _copy_mapping(value, "SourceSemanticOccurrence")
        if set(copied) != {"token", "role", "operation_family"}:
            raise ValueError("Source semantic occurrence wire has an incomplete shape")
        return cls(
            token=copied["token"],
            role=copied["role"],
            operation_family=copied["operation_family"],
        )


def _source_semantic_operation_family(node: ast.AST) -> str | None:
    """Return a J family only for one explicit Source expression node.

    This is deliberately narrower than the provider's operation vocabulary.
    A missing or unsupported family suppresses authorization issuance; it must
    never be approximated from a TTIR opcode, Source order, tensor name, or
    candidate-local label.
    """

    if isinstance(node, ast.BinOp | ast.BoolOp | ast.Compare):
        return "map_binary"
    if isinstance(node, ast.UnaryOp):
        return "map_unary"
    if isinstance(node, ast.IfExp):
        return "select"
    if not isinstance(node, ast.Call):
        return None
    leaf = (
        node.func.attr if isinstance(node.func, ast.Attribute)
        else node.func.id if isinstance(node.func, ast.Name)
        else None
    )
    return {
        "sigmoid": "map_unary", "exp": "map_unary", "log": "map_unary",
        "sqrt": "map_unary", "rsqrt": "map_unary", "abs": "map_unary",
        "sin": "map_unary", "cos": "map_unary", "tanh": "map_unary",
        "maximum": "map_binary", "minimum": "map_binary", "pow": "map_binary",
        "fma": "map_binary", "where": "select", "to": "cast",
        "cast": "cast", "astype": "cast", "float": "cast", "int": "cast",
        "sum": "reduce", "mean": "reduce", "max": "reduce", "min": "reduce",
        "prod": "reduce", "any": "reduce", "all": "reduce", "arange": "iota",
        "reshape": "reshape", "view": "reshape", "trans": "transpose",
        "transpose": "transpose", "join": "concat", "cat": "concat",
        "concat": "concat", "flip": "flip", "sort": "sort",
    }.get(leaf)


def _opaque_source_semantic_occurrence(
    node: ast.AST | None,
    *,
    role: str,
    scope: Sequence[str],
    source_identity: str,
    operation_family: str | None = None,
    binding_mode: str = "direct",
) -> SourceSemanticOccurrence | None:
    """Create one non-textual Source anchor, or none when not provable."""

    if node is None:
        return None
    family = operation_family or _source_semantic_operation_family(node)
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    if (
        family is None
        or not isinstance(line, int)
        or isinstance(line, bool)
        or line <= 0
        or not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
    ):
        return None
    # The preimage remains host-local.  A token is stable only for the mounted
    # Source/facts lineage and does not reveal source text, identifiers, spans,
    # or the lexical function name to the provider.
    material = json.dumps(
        {
            "source": source_identity,
            "role": role,
            "scope": list(scope),
            "family": family,
            "line": line,
            "column": column,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SourceSemanticOccurrence(
        token="so_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32],
        role=role,
        operation_family=family,
        source_line=line,
        source_column=column,
        binding_mode=binding_mode,
    )

# The boundary is part of the same construction transaction as operations,
# outputs, scopes, and dimensions.  Keep this declaration here (rather than
# re-declaring it in a consumer) so the raw boundary wire has one structural
# projection before any typed consumer interprets it.
CANONICAL_BOUNDARY_WIRE_FIELDS = (
    "domains",
    "accesses",
    "broadcasts",
    "aliases",
    "effects",
    "output_mappings",
    "reductions",
    "index_bounds",
    "updates",
    "result_relations",
    "realization_authorizations",
    "consumer_extensions",
    "access_guard_relations",
)


def canonical_artifact_envelope_error(value: Any) -> str | None:
    """Return the one transport-level error for a candidate artifact root.

    This deliberately establishes only the ownership boundary shared by every
    Lift transport.  It does not decode J/C/pi and it never repairs, moves, or
    supplies fields. Semantic validity remains the responsibility of the
    canonical decoder and CandidateCheck. Keeping the small guard here means
    a direct Responses reply and an MCP submission cannot disagree about what
    one candidate document is allowed to replace or enter for construction.
    """

    if not isinstance(value, Mapping):
        return "candidate root must be one JSON object"
    expected = set(CANONICAL_ARTIFACT_ROOT_FIELDS)
    actual = set(value)
    if actual != expected:
        expected_text = ", ".join(CANONICAL_ARTIFACT_ROOT_FIELDS)
        actual_text = ", ".join(sorted(str(key) for key in actual))
        return (
            "candidate root must contain exactly the canonical fields "
            f"{{{expected_text}}}; observed {{{actual_text}}}"
        )
    for field_name in ("j", "c", "pi"):
        if not isinstance(value.get(field_name), Mapping):
            return f"candidate root field {field_name!r} must be an object"
    return None


def require_canonical_artifact_envelope(value: Any) -> None:
    """Raise before a transport accepts a non-canonical candidate root."""

    error = canonical_artifact_envelope_error(value)
    if error is not None:
        raise ValueError(error)


# These are the actual fields emitted by the three shared construction
# records below. Keep this declaration next to the record builders so schema
# and MCP projections cannot substitute an unrelated IntentOp leaf (for
# example, map_binary) as the construction-record contract.
CANONICAL_CONSTRUCTION_RECORD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "operation": (
        "path", "id", "op", "parent_id", "lexical_role", "depth",
        "region_scope_id", "is_final", "operands", "results", "type",
        "attrs", "wire",
    ),
    "output": (
        "output", "abi_index", "output_type", "return_value",
        "abi_parameter_index", "abi_parameter_name", "abi_parameter_mode",
        "mapping_index", "mapping_value", "authored_node_ref", "node_ref",
        "producer_record", "return_index",
    ),
    "scope": (
        "scope_id", "kind", "parent_scope", "definitions", "captures",
        "effects", "operation_paths", "operation_ids", "operation_results",
    ),
    "dimension": (
        "result", "expr",
    ),
    "value_dimension": (
        "value_ref", "dimension_ref", "relation",
    ),
    "boundary": CANONICAL_BOUNDARY_WIRE_FIELDS,
}

# These fields are added only after the canonical J type elaborator has closed
# every SSA value.  Keeping them separate from the raw-wire fields lets the
# ingress repair phase remain structural while ensuring the accepted bundle
# has one typed operation contract for all consumers.
CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS = (
    *CANONICAL_CONSTRUCTION_RECORD_FIELDS["operation"],
    "operand_types", "result_types",
)


def _function_wire(function: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the one internal IntentFunction wire shape.

    ``IntentFunction.to_json()`` intentionally emits an ``intent_function``
    wrapper, while raw CandidateCheck payloads and a few ingress helpers carry
    the inner object directly.  Construction records must not silently read
    different levels of those two equivalent wires, so every shared builder
    normalizes the envelope before looking at inputs/body/outputs/returns.
    """

    if not isinstance(function, Mapping):
        return None
    nested = function.get("intent_function")
    if nested is not None:
        if set(function) != {"intent_function"} or not isinstance(nested, Mapping):
            return None
        return nested
    return function


def _operation_record_walk(
    body: Any,
    *,
    prefix: str = "body",
    parent_id: str | None = None,
    lexical_role: str = "function",
    depth: int = 0,
    region_scope_id: str | None = None,
) -> list[
    tuple[str, Mapping[str, Any], str | None, str, int, str | None, bool]
    ] | None:
    """Walk one authored operation wire in canonical lexical order.

    This is deliberately a structural walker.  It does not import or execute
    TypeCheck/Gate1 contracts, and it does not infer a missing field.  Keeping
    it here makes the operation *record* shared without making semantic proof
    part of the construction layer.
    """

    if not isinstance(body, list):
        return None
    rows: list[
        tuple[str, Mapping[str, Any], str | None, str, int, str | None, bool]
    ] = []
    for index, raw in enumerate(body):
        if not isinstance(raw, Mapping):
            return None
        path = f"{prefix}[{index}]"
        rows.append(
            (
                path,
                raw,
                parent_id,
                lexical_role,
                depth,
                region_scope_id,
                index == len(body) - 1,
            )
        )
        nested = raw.get("body", [])
        nested_scope = (
            raw.get("scope_id")
            if raw.get("op") == "loop" and isinstance(raw.get("scope_id"), str)
            else region_scope_id
        )
        nested_rows = _operation_record_walk(
            nested,
            prefix=f"{path}.body",
            parent_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
            lexical_role="body",
            depth=depth + 1,
            region_scope_id=nested_scope,
        )
        if nested_rows is None:
            return None
        rows.extend(nested_rows)
        if raw.get("op") in {"effect_if", "value_if"}:
            for arm_name in ("positive", "negative"):
                arm = raw.get(arm_name)
                if arm is None:
                    continue
                if not isinstance(arm, Mapping):
                    return None
                arm_rows = _operation_record_walk(
                    arm.get("body", []),
                    prefix=f"{path}.{arm_name}.body",
                    parent_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
                    lexical_role=arm_name,
                    depth=depth + 1,
                    region_scope_id=(
                        arm.get("scope_id")
                        if isinstance(arm.get("scope_id"), str)
                        else region_scope_id
                    ),
                )
                if arm_rows is None:
                    return None
                rows.extend(arm_rows)
    return rows


def _canonical_operation_wire(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize equivalent optional-field spellings into one operation wire.

    Typed ``IntentOp`` serialization omits empty optional fields.  Normalize
    those spellings once here so raw construction records and typed operation
    views cannot disagree about an otherwise identical operation.  This does
    not infer a semantic field or repair an incomplete operation.
    """

    if not isinstance(operation, Mapping):
        return None
    result = dict(operation)
    body = result.get("body")
    if isinstance(body, list):
        normalized_body: list[dict[str, Any]] = []
        for child in body:
            normalized = _canonical_operation_wire(child)
            if normalized is None:
                return None
            normalized_body.append(normalized)
        result["body"] = normalized_body
    elif body is not None:
        return None
    for arm_name in ("positive", "negative"):
        arm = result.get(arm_name)
        if arm is None:
            continue
        if not isinstance(arm, Mapping):
            return None
        normalized_arm = dict(arm)
        arm_body = normalized_arm.get("body")
        if isinstance(arm_body, list):
            normalized_body = []
            for child in arm_body:
                normalized = _canonical_operation_wire(child)
                if normalized is None:
                    return None
                normalized_body.append(normalized)
            normalized_arm["body"] = normalized_body
        elif arm_body is not None:
            return None
        result[arm_name] = normalized_arm

    op_name = result.get("op")
    if not isinstance(op_name, str):
        return None
    for field_name in ("attrs", "operands", "body", "yield"):
        if field_name in result and result[field_name] in (None, [], {}):
            # Loop body/yield are required structural fields even when empty.
            if op_name == "loop" and field_name in {"body", "yield"}:
                continue
            result.pop(field_name, None)
    for field_name in ("type", "scope_id", "predicate", "positive", "negative"):
        if field_name in result and result[field_name] is None:
            result.pop(field_name, None)
    if op_name != "loop" and result.get("scan") == []:
        result.pop("scan", None)
    return result


def canonical_operation_records(
    function: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Build the transport-neutral operation construction table.

    The returned rows are the sole structural interpretation shared by Lift
    ingress, canonical J elaboration, Replay presentation, and TypeCheck's
    explicit type surface.  ``wire`` is the complete authored operation, so
    structured fields (loop carry/body/yield and conditional arms) cannot be
    silently lost when a consumer reads the compact columns.

    This helper only checks recordability: operation contracts remain owned by
    their existing semantic validators.  ``None`` means no safe shared table
    exists; callers must stay diagnostic/fail-closed and may not repair from a
    partial table.
    """

    function = _function_wire(function)
    if function is None:
        return None
    rows = _operation_record_walk(function.get("body"))
    if rows is None:
        return None
    raw_inputs = function.get("inputs")
    if not isinstance(raw_inputs, list):
        return None
    input_names = {
        item.get("name")
        for item in raw_inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(input_names) != len(raw_inputs) or any(
        re.fullmatch(r"t[0-9]+", name) is None for name in input_names
    ):
        return None
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_results: set[str] = set()
    for (
        path,
        operation,
        parent_id,
        lexical_role,
        depth,
        region_scope_id,
        is_final,
    ) in rows:
        canonical_wire = _canonical_operation_wire(operation)
        if canonical_wire is None:
            return None
        operation_id = canonical_wire.get("id")
        operation_name = canonical_wire.get("op")
        if (
            not isinstance(operation_id, str)
            or not isinstance(operation_name, str)
            or re.fullmatch(r"n[0-9]+", operation_id) is None
            or operation_id in seen_ids
            or (
                parent_id is not None
                and parent_id not in seen_ids
            )
        ):
            return None
        raw_operands = canonical_wire.get("operands", [])
        if not isinstance(raw_operands, list) or any(
            not isinstance(item, str) for item in raw_operands
        ):
            return None
        raw_result = canonical_wire.get("result")
        if isinstance(raw_result, str):
            results = (raw_result,)
        elif isinstance(raw_result, list) and all(
            isinstance(item, str) for item in raw_result
        ):
            results = tuple(raw_result)
        elif raw_result is None:
            results = ()
        else:
            return None
        if any(re.fullmatch(r"v[0-9]+", item) is None for item in results):
            return None
        if (
            len(results) != len(set(results))
            or seen_results.intersection(results)
            or input_names.intersection(results)
        ):
            return None
        attrs = canonical_wire.get("attrs")
        if attrs is not None and not isinstance(attrs, Mapping):
            return None
        declared_type = canonical_wire.get("type")
        if declared_type is not None and not isinstance(declared_type, Mapping):
            return None
        seen_ids.add(operation_id)
        seen_results.update(results)
        records.append(
            {
                "path": path,
                "id": operation_id,
                "op": operation_name,
                "parent_id": parent_id,
                "lexical_role": lexical_role,
                "depth": depth,
                "region_scope_id": region_scope_id,
                "is_final": is_final,
                "operands": tuple(raw_operands),
                "results": results,
                "type": json.loads(_stable_json(declared_type)) if declared_type is not None else None,
                "attrs": json.loads(_stable_json(attrs)) if attrs is not None else None,
                "wire": json.loads(_stable_json(canonical_wire)),
            }
        )

    # Close the cross-field SSA relationships that the structural schema
    # cannot express by itself.  In particular, a loop's positional result
    # and yield arrays are witnesses for the carry roles, and every next_value
    # must be produced by a direct body operation.  Keeping this check on the
    # shared operation table prevents Gate1, Replay, and ingress repair from
    # each inventing a different loop interpretation.
    all_results = {
        result
        for record in records
        for result in record["results"]
    }
    occupied_local_bindings = set(input_names) | set(all_results)
    for record in records:
        if record["op"] != "loop":
            continue
        wire = record["wire"]
        carries = wire.get("carry")
        result_values = wire.get("result")
        yielded = wire.get("yield")
        if not isinstance(carries, list) or not isinstance(result_values, list):
            return None
        if not isinstance(yielded, list):
            return None
        expected_results = [
            state.get("result_value")
            for state in carries
            if isinstance(state, Mapping)
        ]
        expected_yields = [
            state.get("next_value")
            for state in carries
            if isinstance(state, Mapping)
        ]
        if (
            len(expected_results) != len(carries)
            or len(expected_yields) != len(carries)
            or result_values != expected_results
            or yielded != expected_yields
        ):
            return None
        role_names: list[str] = []
        local_bindings: list[str] = []
        extra_bindings: list[str] = []
        for state in carries:
            if not isinstance(state, Mapping):
                return None
            roles = [
                state.get("body_argument"),
                state.get("next_value"),
                state.get("result_value"),
            ]
            if any(not isinstance(name, str) for name in roles):
                return None
            role_names.extend(roles)
            local_bindings.append(roles[0])
            if roles[2] not in record["results"]:
                return None
        scan = wire.get("scan")
        if not isinstance(scan, list):
            return None
        for item in scan:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                return None
            local_bindings.append(item["name"])
            extra_bindings.append(item["name"])
        trip = wire.get("trip")
        if isinstance(trip, Mapping) and "index" in trip:
            if not isinstance(trip.get("index"), str):
                return None
            local_bindings.append(trip["index"])
            extra_bindings.append(trip["index"])
        if len(role_names) != len(set(role_names)):
            return None
        if len(local_bindings) != len(set(local_bindings)):
            return None
        if set(extra_bindings) & set(role_names):
            return None
        if set(local_bindings) & set(occupied_local_bindings):
            return None
        direct_body_path = record["path"] + ".body"
        direct_body_results = {
            result
            for child in records
            if isinstance(child.get("path"), str)
            and re.fullmatch(
                re.escape(direct_body_path) + r"\[\d+\]",
                child["path"],
            )
            for result in child["results"]
        }
        if any(
            state.get("next_value") not in direct_body_results
            for state in carries
        ):
            return None
        occupied_local_bindings.update(local_bindings)
    return tuple(records)


def canonical_operation_wire_rows(
    function: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
    """Return canonical lexical paths and the authored operation mappings.

    This is the mutation-capable companion to ``canonical_operation_records``
    for the bounded ingress allocator. It uses the same traversal and
    envelope normalization, but preserves references to the private candidate
    copy so a caller can rename already-authored handles transactionally. It
    never creates an operation or returns rows for an unrecordable body.
    """

    function = _function_wire(function)
    if function is None:
        return None
    rows = _operation_record_walk(function.get("body"))
    if rows is None:
        return None
    return tuple((row[0], row[1]) for row in rows)


def semantic_operation_wire_rows(
    function: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
    """Walk the provider semantic operation tree before canonical renaming.

    This is intentionally distinct from :func:`canonical_operation_records`:
    the host allocator needs lexical paths and authored operation mappings
    before ``n*``/``v*`` identities exist.  It validates only the region/body
    wire shape and never accepts, repairs, or interprets operation semantics.
    The strict canonical record builder remains the acceptance authority after
    the allocator has completed.
    """

    if not isinstance(function, Mapping):
        return None
    normalized = _function_wire(function)
    if normalized is None:
        return None
    rows = _operation_record_walk(normalized.get("body"))
    if rows is None:
        return None
    return tuple((row[0], row[1]) for row in rows)


def canonical_authored_handle_declarations(
    j: Mapping[str, Any],
) -> dict[str, list[str]] | None:
    """Collect candidate declarations plus the referenced Source x* view.

    This is the declaration half of :class:`CanonicalHandleAllocator`.  It is
    intentionally structural: it observes a complete authored J and never
    invents a missing producer, scope, relation, or loop field.  The returned
    rows are consumed by the allocator's dense-renaming pass; semantic
    validity remains owned by the strict J elaborator/CandidateCheck.

    ``x*`` is collected only to let callers reject a candidate that references
    a dimension outside the Source allocator; it is never candidate-owned or
    renumbered.  Keeping this walk beside the canonical operation-record walker is
    important.  A relift ingress must not maintain a second private traversal
    for ``v*/x*/d*/n*/e*/q*/b*/s*/u*`` and then silently disagree with Replay or
    Gate1 when a nested region is added.
    """

    if not isinstance(j, Mapping):
        return None
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    function = _function_wire(function)
    if function is None:
        return None

    ordered: dict[str, list[str]] = {
        prefix: [] for prefix in (*CANONICAL_AUTHORED_HANDLE_PREFIXES, "x")
    }
    seen: dict[str, set[str]] = {
        prefix: set() for prefix in CANONICAL_AUTHORED_HANDLE_PREFIXES
    }
    duplicate: set[str] = set()

    def add(prefix: str, value: Any) -> None:
        pattern = CANONICAL_HANDLE_NAMESPACE_PATTERNS[prefix]
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            return
        if value in seen[prefix]:
            duplicate.add(value)
            return
        seen[prefix].add(value)
        ordered[prefix].append(value)

    def add_declared_handle(value: Any, prefixes: Sequence[str]) -> None:
        """Add a typed handle without treating an unrelated string as one."""

        if not isinstance(value, str):
            return
        for prefix in prefixes:
            if re.fullmatch(CANONICAL_HANDLE_NAMESPACE_PATTERNS[prefix], value):
                add(prefix, value)
                return

    scopes = j.get("scopes")
    if isinstance(scopes, list):
        for scope in scopes:
            if isinstance(scope, Mapping):
                add("q", scope.get("scope_id"))

    boundary = j.get("boundary")
    if isinstance(boundary, Mapping):
        domains = boundary.get("domains")
        if isinstance(domains, Mapping):
            for domain in domains:
                add("d", domain)
        for field_name, prefix, id_field in (
            ("accesses", "a", "access_id"),
            ("effects", "e", "effect_id"),
            ("realization_authorizations", "u", "auth_id"),
        ):
            values = boundary.get(field_name)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        add(prefix, item.get(id_field))

    storage = j.get("storage")
    if isinstance(storage, list):
        for item in storage:
            if isinstance(item, Mapping):
                add_declared_handle(item.get("storage_id"), ("b", "s"))

    operation_rows = semantic_operation_wire_rows(function)
    if operation_rows is None:
        return None
    # Operation identity is owned by the lexical operation table, not by the
    # provider's spelling of ``id``.  A frequent malformed candidate uses the
    # result handle (v*) as the operation id (often with the same text in
    # ``result``).  Counting only literal n* ids made the shared allocator see
    # an empty node namespace, so the canonical n* rewrite never ran and the
    # candidate reached Decode with two incompatible operation identities.
    #
    # The row order and unique authored ids are already explicit here.  For a
    # unique authored table, assigning the dense n* declaration sequence is a
    # structural rename, not semantic inference; the authoring-layout pass
    # below maps each old id to its exact lexical path and rewrites all known
    # node references transactionally.  Duplicate/missing ids remain
    # fail-closed and are not turned into a guessed graph.
    operation_ids = [
        operation.get("id")
        for _path, operation in operation_rows
    ]
    if (
        any(
            not isinstance(operation_id, str)
            or re.fullmatch(r"n[0-9]+", operation_id) is None
            for operation_id in operation_ids
        )
        or len(operation_ids) != len(set(operation_ids))
    ):
        return None
    # Preserve the authored lexical IDs here.  ``for_candidate_j`` derives
    # the dense allocator declaration from this table, and the ingress repair
    # then maps (for example) n3/n7 to n0/n1 while rewriting every node_ref.
    # Replacing the table with n0...nN at this point loses the old names and
    # makes the supposedly shared allocator unable to perform that repair.
    ordered["n"] = list(operation_ids)
    for _path, operation in operation_rows:
        result = operation.get("result")
        if isinstance(result, list):
            for value in result:
                add("v", value)
        else:
            add("v", result)
        if operation.get("op") == "loop":
            trip = operation.get("trip")
            if isinstance(trip, Mapping):
                add("v", trip.get("index"))
            scan = operation.get("scan")
            if isinstance(scan, list):
                for item in scan:
                    if isinstance(item, Mapping):
                        add("v", item.get("name"))
            carry = operation.get("carry")
            if isinstance(carry, list):
                for item in carry:
                    if isinstance(item, Mapping):
                        add("s", item.get("state_id"))
                        add("v", item.get("body_argument"))
                        add("v", item.get("next_value"))
                        add("v", item.get("result_value"))

    # Shape and relation dimensions share one first-appearance x* namespace.
    dimensions: list[str] = []
    dimension_seen: set[str] = set()

    def add_dimension(value: Any) -> None:
        if (
            not isinstance(value, str)
            or re.fullmatch(CANONICAL_HANDLE_NAMESPACE_PATTERNS["x"], value)
            is None
        ):
            return
        if value not in dimension_seen:
            dimension_seen.add(value)
            dimensions.append(value)

    for collection in ("inputs", "outputs"):
        values = function.get(collection)
        if isinstance(values, list):
            for tensor in values:
                if not isinstance(tensor, Mapping):
                    continue
                shape = tensor.get("shape")
                if isinstance(shape, list):
                    for dimension in shape:
                        add_dimension(dimension)
    signature = j.get("signature")
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if isinstance(parameters, list):
        for parameter in parameters:
            if isinstance(parameter, Mapping):
                add_dimension(parameter.get("logical_ref"))
    relations = j.get("dimension_relations")
    if isinstance(relations, list):
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            add_dimension(relation.get("result"))
            expression = relation.get("expr")
            if isinstance(expression, Mapping):
                add_dimension(expression.get("arg"))
                add_dimension(expression.get("rhs"))
    value_relations = j.get("value_dimension_relations")
    if isinstance(value_relations, list):
        for relation in value_relations:
            if isinstance(relation, Mapping):
                add_dimension(relation.get("dimension_ref"))

    # Dimension handles can also occur in a typed operation shape or a
    # gather/iota extent.  They are still declarations of the same x*
    # namespace, not a second operation-local namespace.  Only exact x*
    # tokens are collected; no shape or operation is invented here.
    def collect_dimension_tokens(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                collect_dimension_tokens(item)
        elif isinstance(value, list):
            for item in value:
                collect_dimension_tokens(item)
        elif isinstance(value, str):
            for match in re.finditer(
                r"(?<![A-Za-z0-9_])x[0-9]+(?![A-Za-z0-9_])", value
            ):
                add_dimension(match.group(0))

    collect_dimension_tokens(function)
    collect_dimension_tokens(boundary)
    ordered["x"] = dimensions

    if duplicate:
        # Repeated declarations are not a safe permutation.  Leave the
        # candidate untouched so the strict checker reports the owning field.
        return None
    return ordered


def canonical_operation_objects(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Any], ...] | None:
    """Materialize typed operation views from the shared records.

    Consumers that need ``IntentOp`` behavior must deserialize the exact
    ``wire`` column of the canonical operation record. They must not walk a
    second copy of ``IntentFunction.body`` and independently assign paths.
    The record remains authoritative for path, id, operands, results, type,
    attrs, and regions.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return None
    from intentir.ir.model import IntentOp, intent_op_regions

    record_by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None
        path = record.get("path")
        wire = record.get("wire")
        if (
            not isinstance(path, str)
            or path in record_by_path
            or not isinstance(wire, Mapping)
        ):
            return None
        record_by_path[path] = record

    # Deserialize only top-level canonical wires.  The nested objects yielded
    # from those wires are the same objects used for their parent, so every
    # consumer receives one typed operation tree rather than unrelated
    # per-record reconstructions.  This matters for region ownership and for
    # callers that compare a child operation with its enclosing loop/control
    # operation.
    top_level: list[tuple[str, Any]] = []
    for path in record_by_path:
        if not re.fullmatch(r"body\[\d+\]", path):
            continue
        try:
            operation = IntentOp.from_json(dict(record_by_path[path]["wire"]))
        except (TypeError, ValueError, KeyError):
            return None
        top_level.append((path, operation))
    if not top_level and record_by_path:
        return None

    result: list[tuple[str, Any]] = []

    def visit(path: str, operation: Any) -> None:
        result.append((path, operation))
        for suffix, region in intent_op_regions(operation):
            for index, child in enumerate(region):
                visit(f"{path}.{suffix}[{index}]", child)

    for path, operation in top_level:
        visit(path, operation)
    if {path for path, _operation in result} != set(record_by_path):
        return None
    for path, operation in result:
        wire = record_by_path[path].get("wire")
        if not isinstance(wire, Mapping):
            return None
        if operation._structural_payload() != dict(wire):
            return None
    return tuple(result)


def _coherent_operation_records(
    function: Mapping[str, Any],
    supplied: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...] | None:
    """Return records proven to belong to this exact function wire."""

    authored = canonical_operation_records(function)
    if authored is None:
        return None
    if supplied is None:
        return authored
    records = tuple(supplied)
    if len(records) != len(authored):
        return None
    fields = (
        "path", "id", "op", "parent_id", "lexical_role", "depth",
        "region_scope_id", "is_final", "operands", "results", "type",
        "attrs", "wire",
    )
    for candidate, expected in zip(records, authored, strict=True):
        if not isinstance(candidate, Mapping):
            return None
        if any(candidate.get(field) != expected.get(field) for field in fields):
            return None
    return tuple(dict(candidate) for candidate in records)


def _logical_scope_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"(?:t|v)[0-9]+", value):
        return value
    return None


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _record_has_canonical_fields(
    record: Mapping[str, Any], record_name: str
) -> bool:
    required = CANONICAL_CONSTRUCTION_RECORD_FIELDS.get(record_name)
    return (
        required is not None
        and all(field in record for field in required)
    )


def canonical_scope_records(
    function: Mapping[str, Any],
    scopes: Sequence[Mapping[str, Any]],
    boundary: Mapping[str, Any],
    node_ids: Mapping[str, str],
    *,
    operation_records: Sequence[Mapping[str, Any]] | None = None,
    repairable_scope_fields: bool = False,
) -> tuple[dict[str, Any], ...] | None:
    """Allocate lexical definitions, captures, and effects once.

    This is the structural scope allocator shared by canonical J elaboration,
    ingress diagnostics, and Gate1's expected-scope check.  It deliberately
    does not prove dominance or evaluate predicates.  Those remain Gate1
    responsibilities; this function only establishes which authored region
    owns each operation result, local loop handle, external value use, and
    boundary effect.
    """

    function = _function_wire(function)
    if function is None or not isinstance(boundary, Mapping):
        return None
    if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    declared = [scope for scope in scopes if isinstance(scope, Mapping)]
    if len(declared) != len(scopes):
        return None
    for scope in declared:
        # PortableScope is the persisted consumer-facing declaration, while
        # this function is the construction authority.  Strict consumers
        # require all three authored lists before deriving the projection;
        # bounded ingress repair may leave a list absent, but never malformed,
        # because the derived record will replace that absent/stale field.
        for field_name in ("definitions", "captures", "effects"):
            values = scope.get(field_name)
            if (
                values is not None
                and (
                    not isinstance(values, list)
                    or any(not isinstance(value, str) for value in values)
                    or len(values) != len(set(values))
                )
            ):
                return None
            if not repairable_scope_fields and not isinstance(values, list):
                return None
    by_scope = {
        scope.get("scope_id"): scope
        for scope in declared
        if isinstance(scope.get("scope_id"), str)
    }
    if len(by_scope) != len(declared):
        return None
    # This allocator is also used by deterministic ingress repair, before the
    # full typed archive is admitted.  Reject disconnected or cyclic scope
    # graphs here so no consumer can derive a partial lexical record.
    for scope_id, scope in by_scope.items():
        parent = scope.get("parent_scope")
        if parent is not None and (
            not isinstance(parent, str) or parent not in by_scope
        ):
            return None
    roots = [
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "function" and scope.get("parent_scope") is None
    ]
    if len(roots) != 1:
        return None
    root_scope = roots[0]
    for scope_id in by_scope:
        seen: set[str] = set()
        current: str | None = scope_id
        while current is not None:
            if current in seen:
                return None
            seen.add(current)
            current = by_scope[current].get("parent_scope")
        if root_scope not in seen:
            return None
    program_scopes = [
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "program_instance"
        and scope.get("parent_scope") == root_scope
    ]
    if len(program_scopes) > 1:
        return None
    execution_scope = program_scopes[0] if program_scopes else root_scope

    def owner_for(record: Mapping[str, Any]) -> str:
        region = record.get("region_scope_id")
        return region if isinstance(region, str) else execution_scope

    records_by_scope: dict[str, list[Mapping[str, Any]]] = {
        scope_id: [] for scope_id in by_scope
    }
    for record in records:
        owner = owner_for(record)
        if owner not in records_by_scope:
            return None
        records_by_scope[owner].append(record)

    loop_seeds: dict[str, Mapping[str, Any]] = {}
    branch_seeds: dict[str, Mapping[str, Any]] = {}
    for record in records:
        wire = record.get("wire")
        if not isinstance(wire, Mapping):
            return None
        if record.get("op") == "loop":
            scope_id = wire.get("scope_id")
            if isinstance(scope_id, str):
                if scope_id in loop_seeds:
                    return None
                loop_seeds[scope_id] = wire
        if record.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = wire.get(polarity)
                if not isinstance(arm, Mapping):
                    continue
                scope_id = arm.get("scope_id")
                if isinstance(scope_id, str):
                    if scope_id in branch_seeds:
                        return None
                    branch_seeds[scope_id] = arm

    declared_loop_scope_ids = {
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "loop_region"
    }
    declared_branch_scope_ids = {
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "branch_region"
    }
    if set(loop_seeds) != declared_loop_scope_ids:
        return None
    if set(branch_seeds) != declared_branch_scope_ids:
        return None
    branch_parent_by_scope: dict[str, str] = {}
    for record in records:
        wire = record["wire"]
        owner = owner_for(record)
        if record.get("op") == "loop":
            loop_scope_id = wire.get("scope_id")
            if not isinstance(loop_scope_id, str):
                return None
            loop_scope = by_scope.get(loop_scope_id)
            if (
                not isinstance(loop_scope, Mapping)
                or loop_scope.get("kind") != "loop_region"
                or loop_scope.get("parent_scope") != owner
            ):
                return None
        if record.get("op") not in {"effect_if", "value_if"}:
            continue
        for polarity in ("positive", "negative"):
            arm = wire.get(polarity)
            if not isinstance(arm, Mapping):
                return None
            branch_scope_id = arm.get("scope_id")
            if not isinstance(branch_scope_id, str):
                return None
            branch_scope = by_scope.get(branch_scope_id)
            if (
                not isinstance(branch_scope, Mapping)
                or branch_scope.get("kind") != "branch_region"
                or branch_scope.get("parent_scope") != owner
            ):
                return None
            previous_parent = branch_parent_by_scope.get(branch_scope_id)
            if previous_parent is not None and previous_parent != owner:
                return None
            branch_parent_by_scope[branch_scope_id] = owner

    inputs = [
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]
    definitions: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}
    if root_scope in definitions:
        definitions[root_scope].extend(inputs)
    for scope_id, owned in records_by_scope.items():
        for record in owned:
            definitions[scope_id].extend(
                result for result in record.get("results", ())
                if isinstance(result, str)
            )
    for scope_id, wire in loop_seeds.items():
        if scope_id not in definitions:
            return None
        for scan in wire.get("scan", ()):
            if isinstance(scan, Mapping):
                definitions[scope_id].append(scan.get("name"))
        for carry in wire.get("carry", ()):
            if isinstance(carry, Mapping):
                definitions[scope_id].append(carry.get("body_argument"))
        trip = wire.get("trip")
        if isinstance(trip, Mapping) and isinstance(trip.get("index"), str):
            definitions[scope_id].append(trip.get("index"))
    definitions = {
        scope_id: _ordered_unique(values)
        for scope_id, values in definitions.items()
    }

    uses: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}

    def add_use(scope_id: str, value: Any) -> None:
        reference = _logical_scope_reference(value)
        if reference is not None:
            uses[scope_id].append(reference)

    for scope_id, owned in records_by_scope.items():
        for record in owned:
            wire = record["wire"]
            for value in wire.get("operands", ()):
                add_use(scope_id, value)
            if wire.get("op") in {"effect_if", "value_if"}:
                add_use(scope_id, wire.get("predicate"))
            if wire.get("op") == "loop":
                loop_scope = wire.get("scope_id")
                loop_use_scope = (
                    loop_scope
                    if isinstance(loop_scope, str) and loop_scope in uses
                    else scope_id
                )
                trip = wire.get("trip")
                if isinstance(trip, Mapping):
                    add_use(loop_use_scope, trip.get("upper_bound"))
                for scan in wire.get("scan", ()):
                    if isinstance(scan, Mapping):
                        # ``from`` is the canonical loop-scan input field.
                        # Keep scope derivation aligned with the operation
                        # contract and the structured SSA constructor; the
                        # legacy ``source`` spelling must not create a second
                        # visibility interpretation.
                        add_use(loop_use_scope, scan.get("from"))
                for carry in wire.get("carry", ()):
                    if isinstance(carry, Mapping):
                        add_use(loop_use_scope, carry.get("init_value"))

    for scope_id, arm in branch_seeds.items():
        add_use(scope_id, arm.get("yield_value"))

    captures: dict[str, list[str]] = {}
    for scope_id in by_scope:
        local = set(definitions[scope_id])
        captures[scope_id] = _ordered_unique(
            value for value in uses[scope_id] if value not in local
        )

    path_by_node = {
        node: path for node, path in node_ids.items()
        if isinstance(node, str) and isinstance(path, str)
    }
    record_by_path = {
        record["path"]: record for record in records
    }
    if len(path_by_node) != len(node_ids):
        return None
    expected_node_map = {
        record["id"]: record["path"]
        for record in records
        if isinstance(record.get("id"), str)
        and isinstance(record.get("path"), str)
    }
    # Check both directions.  Validating only authored node_ids entries would
    # allow a body operation to disappear from the node map and be interpreted
    # differently by Replay, Gate1, or an ingress repair pass.
    if expected_node_map != dict(path_by_node):
        return None
    for node, path in path_by_node.items():
        record = record_by_path.get(path)
        if record is None or record.get("id") != node:
            return None
    owner_by_path = {
        path: owner_for(record) for path, record in record_by_path.items()
    }
    declared_effect_owner: dict[str, str] = {}
    for scope_id, scope in by_scope.items():
        for effect_id in scope.get("effects", ()):
            if not isinstance(effect_id, str):
                continue
            previous = declared_effect_owner.get(effect_id)
            if previous is not None and previous != scope_id:
                return None
            declared_effect_owner[effect_id] = scope_id
    effect_ids: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}
    raw_effects = boundary.get("effects", [])
    if not isinstance(raw_effects, list):
        return None
    if repairable_scope_fields:
        # A node-less effect has no operation path from which ownership can be
        # derived.  Its authored scope membership is therefore the only
        # ownership witness (used for direct effects inside an effect_if arm).
        # Do not silently move such an effect to the execution scope when a
        # stale/partial scope wire omitted that witness.
        for effect in raw_effects:
            if not isinstance(effect, Mapping):
                return None
            effect_id = effect.get("effect_id")
            if effect.get("node_ref") is None and (
                not isinstance(effect_id, str)
                or effect_id not in declared_effect_owner
            ):
                return None
    seen_effects: set[str] = set()
    for effect in raw_effects:
        if not isinstance(effect, Mapping) or not isinstance(effect.get("effect_id"), str):
            return None
        effect_id = effect["effect_id"]
        if effect_id in seen_effects:
            return None
        seen_effects.add(effect_id)
        node_ref = effect.get("node_ref")
        path = path_by_node.get(node_ref) if isinstance(node_ref, str) else None
        if node_ref is not None and path is None:
            return None
        owner = (
            owner_by_path.get(path, execution_scope)
            if path is not None
            else declared_effect_owner.get(effect_id, execution_scope)
        )
        if owner not in effect_ids:
            return None
        effect_ids[owner].append(effect_id)
        if owner != execution_scope:
            add_use(owner, effect.get("target"))
    for scope_id in by_scope:
        local = set(definitions[scope_id])
        captures[scope_id] = _ordered_unique(
            value for value in captures[scope_id] + uses[scope_id]
            if value not in local
        )

    # A nested region may use a value outside its immediate parent. Carry that
    # capture upward until a lexical definition owns it; this keeps the parent
    # record complete without making a child definition visible in siblings or
    # the parent continuation.
    for _ in range(len(by_scope)):
        changed = False
        for scope_id, scope in by_scope.items():
            parent = scope.get("parent_scope")
            if not isinstance(parent, str) or parent not in captures:
                continue
            parent_defs = set(definitions[parent])
            before = len(captures[parent])
            captures[parent] = _ordered_unique(
                [
                    *captures[parent],
                    *(
                        value
                        for value in captures[scope_id]
                        if value not in parent_defs
                    ),
                ]
            )
            changed = changed or len(captures[parent]) != before
        if not changed:
            break

    # A construction record is not complete merely because every use was
    # copied into a captures list.  Prove that each captured t*/v* is defined
    # by an ancestor scope; otherwise a child can accidentally capture a
    # sibling-local or entirely unknown handle and Gate1/Replay will each
    # invent a different visibility interpretation.
    for scope_id, scope in by_scope.items():
        ancestor = scope.get("parent_scope")
        visible: set[str] = set()
        while isinstance(ancestor, str):
            parent = by_scope.get(ancestor)
            if parent is None:
                return None
            visible.update(definitions[ancestor])
            visible.update(captures[ancestor])
            ancestor = parent.get("parent_scope")
        if any(value not in visible for value in captures[scope_id]):
            return None

    result: list[dict[str, Any]] = []
    for scope in declared:
        scope_id = scope["scope_id"]
        owned = records_by_scope[scope_id]
        derived = {
            "scope_id": scope_id,
            "kind": scope.get("kind"),
            "parent_scope": scope.get("parent_scope"),
            "definitions": definitions[scope_id],
            "captures": captures[scope_id],
            "effects": effect_ids[scope_id],
            "operation_paths": [record["path"] for record in owned],
            "operation_ids": [record["id"] for record in owned],
            "operation_results": [
                value
                for record in owned
                for value in record.get("results", ())
            ],
        }
        # Definitions/captures are declarations of set membership; their
        # ordering is not semantic.  Effects are the Source/J transfer order,
        # so their order is part of the wire contract and must match exactly.
        if not repairable_scope_fields and (
            set(scope["definitions"]) != set(derived["definitions"])
            or set(scope["captures"]) != set(derived["captures"])
            or scope["effects"] != derived["effects"]
        ):
            return None
        result.append(derived)
    return tuple(result)


def _canonical_ssa_visibility(
    function: Mapping[str, Any],
) -> bool:
    """Prove lexical SSA visibility before a construction bundle is emitted.

    ``canonical_scope_records`` owns persisted definitions/captures/effects,
    but a set-like scope view alone cannot distinguish a use from a later
    sibling producer.  The host constructor must reject that forward use
    before CandidateCheck/Decode, and must also keep loop and conditional
    region bindings from leaking into a sibling or parent continuation.  This
    checker is deliberately structural: it neither infers an operation type
    nor repairs a missing producer; typed semantic validation remains at the
    existing CandidateCheck and elaboration boundaries.
    """

    normalized = _function_wire(function)
    if normalized is None:
        return False
    raw_inputs = normalized.get("inputs")
    body = normalized.get("body")
    if not isinstance(raw_inputs, list) or not isinstance(body, list):
        return False
    input_names = {
        item.get("name")
        for item in raw_inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(input_names) != len(raw_inputs):
        return False

    def result_names(operation: Mapping[str, Any]) -> tuple[str, ...] | None:
        raw_result = operation.get("result")
        if raw_result is None:
            return ()
        if isinstance(raw_result, str):
            return (raw_result,)
        if isinstance(raw_result, list) and all(
            isinstance(item, str) for item in raw_result
        ):
            return tuple(raw_result)
        return None

    def visible_reference(value: Any, visible: set[str]) -> bool:
        return isinstance(value, str) and value in visible

    def visit(
        operations: Any,
        inherited: set[str],
    ) -> tuple[set[str], set[str]] | None:
        if not isinstance(operations, list):
            return None
        visible = set(inherited)
        direct_definitions: set[str] = set()
        for operation in operations:
            if not isinstance(operation, Mapping):
                return None
            op_name = operation.get("op")
            if not isinstance(op_name, str):
                return None
            operands = operation.get("operands", [])
            if not isinstance(operands, list) or any(
                not visible_reference(operand, visible) for operand in operands
            ):
                return None
            results = result_names(operation)
            if results is None or len(results) != len(set(results)):
                return None

            if op_name in {"effect_if", "value_if"}:
                if not visible_reference(operation.get("predicate"), visible):
                    return None
                arm_visibilities: dict[str, set[str]] = {}
                for arm_name in ("positive", "negative"):
                    arm = operation.get(arm_name)
                    if not isinstance(arm, Mapping):
                        return None
                    arm_result = visit(arm.get("body"), visible)
                    if arm_result is None:
                        return None
                    arm_visible, _arm_definitions = arm_result
                    arm_visibilities[arm_name] = arm_visible
                if op_name == "effect_if":
                    if results:
                        return None
                    continue
                if len(results) != 1 or results[0] in visible:
                    return None
                for arm_name in ("positive", "negative"):
                    arm = operation[arm_name]
                    if not visible_reference(
                        arm.get("yield_value"), arm_visibilities[arm_name]
                    ):
                        return None
                visible.add(results[0])
                direct_definitions.add(results[0])
                continue

            if op_name == "loop":
                carry = operation.get("carry")
                scan = operation.get("scan", [])
                trip = operation.get("trip")
                yielded = operation.get("yield")
                if (
                    not isinstance(carry, list)
                    or not isinstance(scan, list)
                    or not isinstance(trip, Mapping)
                    or not isinstance(yielded, list)
                    or any(not isinstance(item, Mapping) for item in carry)
                    or any(not isinstance(item, Mapping) for item in scan)
                    or len(carry) != len(results)
                    or len(carry) != len(yielded)
                ):
                    return None
                local_names: set[str] = set()
                for item in scan:
                    name = item.get("name")
                    source = item.get("from")
                    if (
                        not isinstance(name, str)
                        or name in visible
                        or name in local_names
                        or not visible_reference(source, visible)
                    ):
                        return None
                    local_names.add(name)
                for state in carry:
                    body_argument = state.get("body_argument")
                    init_value = state.get("init_value")
                    if (
                        not isinstance(body_argument, str)
                        or body_argument in visible
                        or body_argument in local_names
                        or (
                            isinstance(init_value, str)
                            and init_value not in {"-inf", "inf", "nan"}
                            and not visible_reference(init_value, visible)
                        )
                    ):
                        return None
                    local_names.add(body_argument)
                if "upper_bound" in trip:
                    if not visible_reference(trip.get("upper_bound"), visible):
                        return None
                    index = trip.get("index")
                    if (
                        not isinstance(index, str)
                        or index in visible
                        or index in local_names
                    ):
                        return None
                    local_names.add(index)
                if any(result in visible or result in local_names for result in results):
                    return None
                body_result = visit(operation.get("body"), visible | local_names)
                if body_result is None:
                    return None
                _body_visible, direct_body_definitions = body_result
                expected_results: list[str] = []
                expected_yields: list[str] = []
                carry_roles: set[str] = set()
                for state in carry:
                    body_argument = state.get("body_argument")
                    next_value = state.get("next_value")
                    result_value = state.get("result_value")
                    if (
                        not isinstance(body_argument, str)
                        or not isinstance(next_value, str)
                        or next_value not in direct_body_definitions
                        or not isinstance(result_value, str)
                        or len({body_argument, next_value, result_value}) != 3
                        or body_argument in carry_roles
                        or result_value in carry_roles
                        or next_value in carry_roles
                    ):
                        return None
                    carry_roles.update({body_argument, next_value, result_value})
                    expected_results.append(result_value)
                    expected_yields.append(next_value)
                if (
                    tuple(results) != tuple(expected_results)
                    or tuple(yielded) != tuple(expected_yields)
                    or len(expected_yields) != len(set(expected_yields))
                ):
                    return None
                visible.update(results)
                direct_definitions.update(results)
                continue

            if any(result in visible for result in results):
                return None
            visible.update(results)
            direct_definitions.update(results)
        return visible, direct_definitions

    return visit(body, input_names) is not None


def canonical_output_records(
    function: Mapping[str, Any],
    boundary: Mapping[str, Any],
    operation_records: Sequence[Mapping[str, Any]] | None,
    *,
    signature: Mapping[str, Any] | None = None,
    repairable_missing_returns: bool = False,
) -> tuple[dict[str, Any], ...] | None:
    """Build the one output/return/mapping/ABI record table.

    This is the raw-wire counterpart of ``JElaboration.output_bindings``.
    Lift ingress uses it before strict decoding; canonical J elaboration uses
    the same authored relationships after decoding.  If no operation table is
    supplied, it rebuilds that table from this exact function wire; a supplied
    table must match it byte-for-byte through ``_coherent_operation_records``.
    It accepts no inferred producer and never repairs a conflicting row.  In
    the bounded repair mode, a missing return row is tolerated only when the
    same output/value is already proven by an authored mapping; the caller may
    then materialize the return and must still run strict mode before committing
    the candidate.
    """

    function = _function_wire(function)
    if function is None or not isinstance(boundary, Mapping):
        return None
    if not isinstance(signature, Mapping):
        return None
    signature_parameters = signature.get("parameters")
    if not isinstance(signature_parameters, list):
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    outputs = function.get("outputs")
    returns = function.get("returns")
    mappings = boundary.get("output_mappings", [])
    if not isinstance(outputs, list) or not isinstance(returns, list):
        if not (
            repairable_missing_returns
            and isinstance(outputs, list)
            and returns is None
        ):
            return None
        returns = []
    if not isinstance(mappings, list):
        return None

    output_names: list[str] = []
    output_types: dict[str, dict[str, Any]] = {}
    input_names = {
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if (
        not isinstance(function.get("inputs"), list)
        or len(input_names) != len(function["inputs"])
        or any(re.fullmatch(r"t[0-9]+", name) is None for name in input_names)
    ):
        return None
    for item in outputs:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("dtype"), str)
            or not isinstance(item.get("shape"), list)
        ):
            return None
        name = item["name"]
        if (
            re.fullmatch(r"t[0-9]+", name) is None
            or name in output_types
        ):
            return None
        output_names.append(name)
        output_types[name] = {
            "dtype": json.loads(_stable_json(item.get("dtype"))),
            "shape": json.loads(_stable_json(item.get("shape"))),
        }
    if not output_names:
        return None

    # The output record owns the physical ABI binding as well as the logical
    # output order.  Do not let a later consumer independently match an output
    # tensor to a p* row by position or by a second logical_ref scan.
    output_parameters: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for parameter_index, parameter in enumerate(signature_parameters):
        if not isinstance(parameter, Mapping):
            return None
        parameter_name = parameter.get("name")
        logical_ref = parameter.get("logical_ref")
        if not isinstance(logical_ref, str) or logical_ref not in output_types:
            continue
        if (
            not isinstance(parameter_name, str)
            or re.fullmatch(r"p[0-9]+", parameter_name) is None
            or not isinstance(logical_ref, str)
            or parameter.get("kind") not in {"tensor", "pointer"}
            or parameter.get("mode") not in {"output", "inout"}
            or logical_ref in output_parameters
        ):
            return None
        output_type = output_types[logical_ref]
        if (
            parameter.get("dtype") != output_type.get("dtype")
            or parameter.get("rank") != len(output_type.get("shape", ()))
        ):
            return None
        output_parameters[logical_ref] = (parameter_index, parameter)
    if set(output_parameters) != set(output_names):
        return None

    # The output record is the only owner of logical output order.  It must
    # agree with the output/inout subsequence of the physical signature; using
    # the declaration order from ``intent.outputs`` here would create a second
    # ABI interpretation for Gate2 and native Dataset projections.
    abi_output_names = [
        parameter.get("logical_ref")
        for parameter in signature_parameters
        if (
            isinstance(parameter, Mapping)
            and parameter.get("mode") in {"output", "inout"}
            and isinstance(parameter.get("logical_ref"), str)
            and parameter.get("logical_ref") in output_types
        )
    ]
    if tuple(abi_output_names) != tuple(output_names):
        return None

    mapping_by_output: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, item in enumerate(mappings):
        if not isinstance(item, Mapping):
            return None
        output = item.get("output")
        if not isinstance(output, str) or output not in output_types:
            return None
        if output in mapping_by_output:
            return None
        mapping_by_output[output] = (index, item)

    object_returns: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, item in enumerate(returns):
        if not isinstance(item, Mapping):
            return None
        output = item.get("output")
        value = item.get("value")
        if (
            not isinstance(output, str)
            or output not in output_types
            or not isinstance(value, str)
            or output in object_returns
        ):
            return None
        object_returns[output] = (index, item)
    authored_return_outputs = [item.get("output") for item in returns]
    if (
        authored_return_outputs != output_names
        and not repairable_missing_returns
    ):
        return None
    if not object_returns and not repairable_missing_returns:
        # Every declared output must have a function return.  A sparse
        # output_mapping is only an additional inout/store anchor; it cannot
        # replace the ABI return row in the canonical construction record.
        return None

    producers: dict[str, Mapping[str, Any]] = {}
    duplicate_producers: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            return None
        results = record.get("results")
        if not isinstance(results, (tuple, list)):
            return None
        for result in results:
            if not isinstance(result, str):
                return None
            if result in producers:
                duplicate_producers.add(result)
            else:
                producers[result] = record
    for result in duplicate_producers:
        producers.pop(result, None)

    operation_rows = records
    records: list[dict[str, Any]] = []
    for abi_index, output in enumerate(abi_output_names):
        parameter_index, parameter = output_parameters[output]
        candidates: list[str] = []
        authored_return = object_returns.get(output)
        if authored_return is not None:
            candidates.append(str(authored_return[1]["value"]))
        mapped = mapping_by_output.get(output)
        if mapped is not None:
            value = mapped[1].get("value")
            if not isinstance(value, str):
                return None
            candidates.append(value)
        if not candidates or len(set(candidates)) != 1:
            return None
        value = candidates[0]
        if re.fullmatch(r"(?:t|v)[0-9]+", value) is None:
            return None

        producer = producers.get(value)
        producer_is_proven = (
            producer is not None
            and (
                "contract_valid" not in producer
                or producer.get("contract_valid") is True
            )
        )
        authored_node_ref = (
            mapped[1].get("node_ref")
            if mapped is not None and isinstance(mapped[1].get("node_ref"), str)
            else None
        )
        producer_id = producer.get("id") if producer_is_proven else None
        if value.startswith("t"):
            # A t* return is a source input/inout binding.  A pure output
            # declaration is not a visible value producer and must not be
            # mistaken for one merely because it appears in outputs[].
            if value not in input_names:
                return None
        else:
            if not producer_is_proven or producer_id is None:
                return None
            if re.fullmatch(r"body\[\d+\]", str(producer.get("path"))) is None:
                # Values defined in a loop/conditional region do not escape
                # that region.  Only the top-level operation result can be a
                # function return value.
                return None
        if authored_node_ref is not None and authored_node_ref != producer_id:
            # ``output_mappings.node_ref`` is a sparse Layer-B store anchor,
            # not necessarily the Layer-A SSA producer.  Accept the distinct
            # node only when the shared boundary ledger proves one unique
            # output store/update access and matching write effect, and the
            # store operation consumes the returned value.  This keeps
            # output/return construction single-sourced without confusing
            # mutation with value production.
            accesses = boundary.get("accesses")
            effects = boundary.get("effects")
            if not isinstance(accesses, list) or not isinstance(effects, list):
                return None
            store_refs = {
                item.get("node_ref")
                for item in accesses
                if isinstance(item, Mapping)
                and item.get("kind") in {"store", "update"}
                and item.get("tensor") == output
                and isinstance(item.get("node_ref"), str)
                and any(
                    isinstance(effect, Mapping)
                    and effect.get("kind") == "write"
                    and effect.get("target") == output
                    and effect.get("access_ref") == item.get("access_id")
                    and effect.get("node_ref") == item.get("node_ref")
                    for effect in effects
                )
            }
            store_record = next(
                (
                    record for record in operation_rows
                    if record.get("id") == authored_node_ref
                    and record.get("op") in {"scatter", "masked_scatter", "store"}
                    and (
                        value in record.get("operands", ())
                        if isinstance(record.get("operands"), (tuple, list))
                        else False
                    )
                ),
                None,
            )
            if len(store_refs) != 1 or authored_node_ref not in store_refs:
                return None
            if store_record is None:
                return None
        records.append(
            {
                "output": output,
                "abi_index": abi_index,
                "output_type": output_types[output],
                "return_value": value,
                "abi_parameter_index": parameter_index,
                "abi_parameter_name": parameter["name"],
                "abi_parameter_mode": parameter["mode"],
                "mapping_index": mapped[0] if mapped is not None else None,
                "mapping_value": mapped[1].get("value") if mapped is not None else None,
                "authored_node_ref": authored_node_ref,
                "node_ref": producer_id,
                "producer_record": (
                    dict(producer) if producer_is_proven else None
                ),
                "return_index": (
                    authored_return[0] if authored_return is not None else None
                ),
            }
        )
    return tuple(records)


def canonical_value_type_records(
    function: Mapping[str, Any],
    operation_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Project explicit input/output/operation types from one op table."""

    function = _function_wire(function)
    if function is None:
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    values: dict[str, dict[str, Any]] = {}
    for collection in ("inputs", "outputs"):
        raw_values = function.get(collection)
        if not isinstance(raw_values, list):
            return None
        collection_names: set[str] = set()
        for item in raw_values:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("dtype"), str)
                or not isinstance(item.get("shape"), list)
            ):
                return None
            name = item["name"]
            if re.fullmatch(r"t[0-9]+", name) is None or name in collection_names:
                return None
            collection_names.add(name)
            value_type = {
                "dtype": json.loads(_stable_json(item.get("dtype"))),
                "shape": json.loads(_stable_json(item.get("shape"))),
            }
            previous = values.get(name)
            if previous is not None and previous != value_type:
                return None
            values[name] = value_type
    for record in records:
        results = record.get("results")
        if not isinstance(results, (tuple, list)):
            return None
        typed_results: list[tuple[str, Mapping[str, Any]]] = []
        declared = record.get("type")
        if len(results) == 1 and isinstance(declared, Mapping):
            result = results[0]
            if not isinstance(result, str):
                return None
            typed_results.append((result, declared))
        elif record.get("op") == "loop":
            wire = record.get("wire")
            carries = wire.get("carry") if isinstance(wire, Mapping) else None
            if not isinstance(carries, list) or len(carries) != len(results):
                return None
            for index, state in enumerate(carries):
                if not isinstance(state, Mapping):
                    return None
                result = state.get("result_value")
                carry_type = state.get("type")
                if not isinstance(result, str) or not isinstance(carry_type, Mapping):
                    return None
                if result != results[index]:
                    return None
                typed_results.append((result, carry_type))
        for result, declared_type in typed_results:
            normalized = json.loads(_stable_json(dict(declared_type)))
            previous = values.get(result)
            if previous is not None and previous != normalized:
                return None
            values[result] = normalized
    return values


def _canonical_relation_records(
    rows: Any,
    *,
    record_name: str,
) -> tuple[dict[str, Any], ...] | None:
    """Copy one already-validated J relation table into the construction bundle.

    Dimension relations are validated by ``SemanticArchive`` before the normal
    construction pass.  This helper does not re-infer or evaluate them; it
    only gives all downstream consumers the same ordered, JSON-stable rows.
    A missing table is distinct from an empty table at the caller boundary,
    so malformed/non-array input remains fail-closed.
    """

    required = CANONICAL_CONSTRUCTION_RECORD_FIELDS[record_name]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if set(row) != set(required):
            return None
        copied = json.loads(_stable_json(dict(row)))
        if not isinstance(copied, dict):
            return None
        if any(field not in copied for field in required):
            return None
        result.append(copied)
    return tuple(result)


def canonical_boundary_records(
    boundary: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Build the one structural boundary projection for construction.

    ``SemanticArchive`` remains responsible for the detailed boundary row
    contracts.  This helper does not infer or repair any row; it copies the
    already-decoded collections in their authored order and makes the exact
    collection set available to every later consumer.  A missing or unknown
    collection means that no shared construction bundle exists.
    """

    if not isinstance(boundary, Mapping):
        return None
    if set(boundary) != set(CANONICAL_BOUNDARY_WIRE_FIELDS):
        return None
    copied = json.loads(_stable_json(dict(boundary)))
    if not isinstance(copied, dict):
        return None
    if not all(field in copied for field in CANONICAL_BOUNDARY_WIRE_FIELDS):
        return None
    if not isinstance(copied.get("domains"), dict):
        return None
    if any(
        not isinstance(copied.get(field), list)
        for field in CANONICAL_BOUNDARY_WIRE_FIELDS
        if field != "domains"
    ):
        return None
    return (copied,)


@dataclass(frozen=True)
class CanonicalConstructionRecords:
    """Atomic operation/value/output/scope construction result.

    ``build`` returns only a complete bundle.  ``build_structural`` is the
    pre-output phase used by bounded ingress repair: it contains the same
    operation/value/scope records, but deliberately has no output records
    until the authored output/return/mapping relationships have been repaired
    or proven complete.  It is not a second construction authority; the
    complete ``build`` must still succeed before any repair is committed.
    """

    operation_records: tuple[Mapping[str, Any], ...]
    value_type_records: Mapping[str, Mapping[str, Any]]
    output_records: tuple[Mapping[str, Any], ...]
    scope_records: tuple[Mapping[str, Any], ...]
    dimension_records: tuple[Mapping[str, Any], ...] = ()
    value_dimension_records: tuple[Mapping[str, Any], ...] = ()
    boundary_records: tuple[Mapping[str, Any], ...] = ()

    @property
    def operation_by_path(self) -> dict[str, Mapping[str, Any]]:
        """Index the canonical operation rows without rebuilding their paths."""

        return {
            str(record["path"]): record
            for record in self.operation_records
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }

    @property
    def operation_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Index operation rows by the same canonical ``n*`` identity."""

        return {
            str(record["id"]): record
            for record in self.operation_records
            if isinstance(record, Mapping) and isinstance(record.get("id"), str)
        }

    def operation_record_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Mapping[str, Any]] | None:
        """Return one raw operation-record table for paths and node ids.

        This is the raw-wire companion to ``typed_operation_by_ref``.  It is
        intended for structural projections such as Gate2 policy and Layer-C
        attachment checks; it does not permit a consumer to reinterpret the
        operation.  Unknown or disagreeing authored aliases fail closed.
        """

        by_path = self.operation_by_path
        by_id = self.operation_by_id
        result = dict(by_path)
        if node_ids is None:
            return result
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            record = by_path.get(path)
            if record is None or node_id in result:
                return None
            if by_id.get(node_id) is not record:
                return None
            result[node_id] = record
        return result

    def authored_node_by_path(
        self,
        node_ids: Mapping[str, str],
    ) -> dict[str, str] | None:
        """Return the validated authored ``path -> n*`` projection."""

        if not isinstance(node_ids, Mapping):
            return None
        by_path = self.operation_by_path
        result: dict[str, str] = {}
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            record = by_path.get(path)
            if record is None or record.get("id") != node_id:
                return None
            if path in result:
                return None
            result[path] = node_id
        return result

    @property
    def typed_operation_by_path(self) -> dict[str, Any] | None:
        """Index the one typed operation tree by canonical lexical path.

        This is the only typed path table that consumers should use.  In
        particular, callers must not rebuild it by walking ``intent.body``
        after the construction pass, because that would make path/region
        ownership a second interface contract.
        """

        rows = self.typed_operation_rows
        if rows is None:
            return None
        return dict(rows)

    @property
    def typed_operation_by_id(self) -> dict[str, Any] | None:
        """Index the same typed operation tree by canonical ``n*`` id."""

        rows = self.typed_operation_rows
        if rows is None:
            return None
        result: dict[str, Any] = {}
        for path, operation in rows:
            record = self.operation_by_path.get(path)
            operation_id = record.get("id") if record is not None else None
            if not isinstance(operation_id, str) or operation_id in result:
                return None
            result[operation_id] = operation
        return result

    def typed_operation_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Return one reference table for both paths and authored node ids.

        Boundary rows are allowed to refer to an operation by its stable
        ``n*`` id or by its canonical body path.  The mapping is constructed
        once here and rejects an unknown path instead of silently dropping the
        alias.  ``None`` is a fail-closed result, not an invitation to fall
        back to a raw-body walk.
        """

        # Materialize the typed tree exactly once.  Rebuilding it separately
        # for ``by_path`` and ``by_id`` creates equal-but-distinct IntentOp
        # objects; an identity comparison between those two views then rejects
        # every valid node map.  Path and n* indexes must be projections of one
        # typed construction result, not two deserializations.
        rows = self.typed_operation_rows
        if rows is None:
            return None
        by_path = dict(rows)
        by_id: dict[str, Any] = {}
        for path, operation in rows:
            record = self.operation_by_path.get(path)
            operation_id = record.get("id") if record is not None else None
            if not isinstance(operation_id, str) or operation_id in by_id:
                return None
            by_id[operation_id] = operation
        result = dict(by_path)
        if node_ids is None:
            return result
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            operation = by_path.get(path)
            if operation is None or node_id in result:
                return None
            if by_id.get(node_id) is not operation:
                return None
            result[node_id] = operation
        return result

    @property
    def value_type_by_name(self) -> dict[str, Mapping[str, Any]]:
        """Return the closed SSA/tensor type table owned by this bundle."""

        return dict(self.value_type_records)

    @property
    def output_by_name(self) -> dict[str, Mapping[str, Any]]:
        """Index the output/return/ABI rows in the canonical bundle."""

        return {
            str(record["output"]): record
            for record in self.output_records
            if isinstance(record, Mapping) and isinstance(record.get("output"), str)
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        """Logical outputs in the already-proven physical ABI order."""

        return tuple(
            str(record["output"])
            for record in self.output_records
            if isinstance(record, Mapping) and isinstance(record.get("output"), str)
        )

    @property
    def output_values(self) -> dict[str, str]:
        """Output-to-return SSA values from the same output records."""

        return {
            str(record["output"]): str(record["return_value"])
            for record in self.output_records
            if (
                isinstance(record, Mapping)
                and isinstance(record.get("output"), str)
                and isinstance(record.get("return_value"), str)
            )
        }

    @property
    def scope_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Index the derived lexical ownership records."""

        return {
            str(record["scope_id"]): record
            for record in self.scope_records
            if isinstance(record, Mapping)
            and isinstance(record.get("scope_id"), str)
        }

    @property
    def dimension_by_result(self) -> dict[str, Mapping[str, Any]]:
        """Index the one ordered J dimension-relation construction table."""

        return {
            str(record["result"]): record
            for record in self.dimension_records
            if isinstance(record, Mapping)
            and isinstance(record.get("result"), str)
        }

    @property
    def boundary_record(self) -> Mapping[str, Any] | None:
        """Return the single construction-owned boundary wire projection."""

        if len(self.boundary_records) != 1:
            return None
        record = self.boundary_records[0]
        return record if isinstance(record, Mapping) else None

    def boundary_rows(self, name: str) -> tuple[Mapping[str, Any], ...] | None:
        """Return one boundary collection without rebuilding it from raw J."""

        record = self.boundary_record
        if record is None or name not in CANONICAL_BOUNDARY_WIRE_FIELDS:
            return None
        rows = record.get(name)
        if name == "domains" or not isinstance(rows, list):
            return None
        if any(not isinstance(item, Mapping) for item in rows):
            return None
        return tuple(rows)

    @property
    def boundary_domains(self) -> Mapping[str, Any] | None:
        """Return the construction-owned domain map without re-parsing J."""

        record = self.boundary_record
        domains = record.get("domains") if record is not None else None
        return domains if isinstance(domains, Mapping) else None

    @property
    def typed_operation_rows(self) -> tuple[tuple[str, Any], ...] | None:
        """Materialize typed operations from the stored canonical wire only."""

        return canonical_operation_objects(self.operation_records)

    def with_elaborated_value_types(
        self,
        value_types: Mapping[str, Any],
    ) -> "CanonicalConstructionRecords | None":
        """Close the raw construction table with one elaborated type map.

        The raw ingress pass can only project authored operation types.  A
        canonical J elaboration may prove additional SSA/loop types from a
        closed operation contract.  Merge those proofs into this same bundle,
        rejecting any disagreement instead of leaving raw and inferred type
        tables as competing authorities.
        """

        if not isinstance(value_types, Mapping):
            return None
        merged = {
            str(name): json.loads(_stable_json(dict(value_type)))
            for name, value_type in value_types.items()
            if isinstance(name, str) and isinstance(value_type, Mapping)
        }
        if len(merged) != len(value_types):
            return None
        for name, authored in self.value_type_records.items():
            inferred = merged.get(name)
            if inferred is not None and dict(authored) != inferred:
                return None
            merged[name] = json.loads(_stable_json(dict(authored)))
        typed_operations: list[dict[str, Any]] = []
        for record in self.operation_records:
            operands = record.get("operands")
            results = record.get("results")
            if not isinstance(operands, (tuple, list)) or not isinstance(
                results, (tuple, list)
            ):
                return None
            operand_types: list[dict[str, Any]] = []
            for operand in operands:
                if not isinstance(operand, str):
                    return None
                operand_type = merged.get(operand)
                if not isinstance(operand_type, Mapping):
                    return None
                # Preserve operand ordinal, including repeated operands;
                # keyed maps would erase that part of the operation contract.
                operand_types.append(
                    json.loads(_stable_json(dict(operand_type)))
                )
            result_types: list[dict[str, Any]] = []
            for result in results:
                if not isinstance(result, str):
                    return None
                result_type = merged.get(result)
                if not isinstance(result_type, Mapping):
                    return None
                result_types.append(
                    json.loads(_stable_json(dict(result_type)))
                )
            declared_type = record.get("type")
            if (
                isinstance(declared_type, Mapping)
                and len(results) == 1
                and dict(declared_type) != result_types[0]
            ):
                return None
            typed_record = dict(record)
            typed_record["operand_types"] = operand_types
            typed_record["result_types"] = result_types
            if any(
                field not in typed_record
                for field in CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS
            ):
                return None
            typed_operations.append(typed_record)
        return replace(
            self,
            value_type_records=merged,
            operation_records=tuple(typed_operations),
        )

    @classmethod
    def build_structural(
        cls,
        function: Mapping[str, Any],
        boundary: Mapping[str, Any],
        scopes: Sequence[Mapping[str, Any]],
        node_ids: Mapping[str, str],
        *,
        operation_records: Sequence[Mapping[str, Any]] | None = None,
        repairable_scope_fields: bool = False,
        dimension_relations: Sequence[Mapping[str, Any]] | None = None,
        value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CanonicalConstructionRecords | None":
        """Build the strict pre-output phase of the canonical bundle.

        This phase is strict about operation contracts, value types, node
        ownership, and lexical scopes.  It does not require the output table
        to be complete because deterministic ingress repair may need those
        already-proven structural facts to repair an output return or mapping.
        No caller may use this phase as a final artifact: ``build`` remains
        the commit gate.
        """

        records = _coherent_operation_records(function, operation_records)
        if records is None:
            return None
        if not _canonical_ssa_visibility(function):
            return None
        for record in records:
            if record.get("contract_valid") is False:
                return None
            wire = record.get("wire")
            if not isinstance(wire, Mapping):
                return None
            try:
                validate_intent_op_structure(
                    wire,
                    path=str(record.get("path", "body")),
                    recursive=True,
                )
            except (TypeError, ValueError):
                return None
        value_types = canonical_value_type_records(function, records)
        scope_records = canonical_scope_records(
            function,
            scopes,
            boundary,
            node_ids,
            operation_records=records,
            repairable_scope_fields=repairable_scope_fields,
        )
        if value_types is None or scope_records is None:
            return None
        if any(
            not _record_has_canonical_fields(record, "operation")
            for record in records
        ) or any(
            not _record_has_canonical_fields(record, "scope")
            for record in scope_records
        ):
            return None
        dimensions = _canonical_relation_records(
            () if dimension_relations is None else dimension_relations,
            record_name="dimension",
        )
        value_dimensions = _canonical_relation_records(
            ()
            if value_dimension_relations is None
            else value_dimension_relations,
            record_name="value_dimension",
        )
        boundary_records = canonical_boundary_records(boundary)
        if (
            dimensions is None
            or value_dimensions is None
            or boundary_records is None
        ):
            return None
        if any(
            not _record_has_canonical_fields(record, "boundary")
            for record in boundary_records
        ):
            return None
        return cls(
            records,
            value_types,
            (),
            scope_records,
            dimensions,
            value_dimensions,
            boundary_records,
        )

    @classmethod
    def build(
        cls,
        function: Mapping[str, Any],
        boundary: Mapping[str, Any],
        scopes: Sequence[Mapping[str, Any]],
        node_ids: Mapping[str, str],
        *,
        operation_records: Sequence[Mapping[str, Any]] | None = None,
        signature: Mapping[str, Any] | None = None,
        dimension_relations: Sequence[Mapping[str, Any]] | None = None,
        value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CanonicalConstructionRecords | None":
        structural = cls.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            operation_records=operation_records,
            dimension_relations=dimension_relations,
            value_dimension_relations=value_dimension_relations,
        )
        if structural is None:
            return None
        outputs = canonical_output_records(
            function,
            boundary,
            structural.operation_records,
            signature=signature,
        )
        if outputs is None:
            return None
        if any(
            not _record_has_canonical_fields(record, "output")
            for record in outputs
        ):
            return None
        if any(
            not _record_has_canonical_fields(record, "operation")
            for record in structural.operation_records
        ) or any(
            not _record_has_canonical_fields(record, "scope")
            for record in structural.scope_records
        ):
            return None
        for output in outputs:
            value = output.get("return_value")
            actual_type = structural.value_type_records.get(value)
            expected_type = output.get("output_type")
            if not isinstance(expected_type, Mapping):
                return None
            # A multi-result operation may have no single ``operation.type``
            # wire even though the canonical J elaborator proves each result
            # from its operands.  Do not invent that type in this raw record;
            # when an explicit/loop-boundary type exists, however, compare it
            # here so output ABI and value typing cannot diverge.
            if not isinstance(actual_type, Mapping):
                continue
            if dict(actual_type) != dict(expected_type):
                return None
        return cls(
            structural.operation_records,
            structural.value_type_records,
            outputs,
            structural.scope_records,
            structural.dimension_records,
            structural.value_dimension_records,
            structural.boundary_records,
        )


def canonical_archive_records(
    archive: Any,
    *,
    structural: bool = False,
    repairable_scope_fields: bool = False,
) -> CanonicalConstructionRecords | None:
    """Build the one construction bundle for a semantic archive wire.

    This is the adapter boundary for consumers that receive a decoded J
    object.  It intentionally accepts either a mapping or an object exposing
    ``to_json`` and then delegates to the same operation/value/output/scope
    builders used by Lift ingress and canonical J elaboration.  Consumers
    must use the returned indexes/views instead of re-walking ``intent.body``
    or independently joining returns, mappings, and scopes.

    ``structural=True`` is only for pre-output diagnostics/repair.  It never
    constitutes an accepted artifact; callers that commit a candidate must
    call this function again in strict (complete) mode.
    """

    if isinstance(archive, Mapping):
        payload: Any = archive
    else:
        to_json = getattr(archive, "to_json", None)
        if not callable(to_json):
            return None
        payload = to_json()
    if not isinstance(payload, Mapping):
        return None
    function = payload.get("intent")
    boundary = payload.get("boundary")
    scopes = payload.get("scopes")
    node_ids = payload.get("node_ids")
    signature = payload.get("signature")
    value_dimension_relations = payload.get("value_dimension_relations")
    dimension_relations = payload.get("dimension_relations")
    if not isinstance(function, Mapping) or not isinstance(boundary, Mapping):
        return None
    if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
        return None
    if not isinstance(node_ids, Mapping) or not isinstance(signature, Mapping):
        return None
    if structural:
        return CanonicalConstructionRecords.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            repairable_scope_fields=repairable_scope_fields,
            dimension_relations=dimension_relations,
            value_dimension_relations=value_dimension_relations,
        )
    return CanonicalConstructionRecords.build(
        function,
        boundary,
        scopes,
        node_ids,
        signature=signature,
        dimension_relations=dimension_relations,
        value_dimension_relations=value_dimension_relations,
    )


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _copy_mapping(value: Mapping[str, Any], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{context} must be a string-keyed object")
    copied = json.loads(_stable_json(dict(value)))
    if not isinstance(copied, dict):
        raise TypeError(f"{context} must be a JSON object")
    return copied


def _view_json(view: Any) -> Mapping[str, Any]:
    if isinstance(view, Mapping):
        return view
    to_json = getattr(view, "to_json", None)
    if callable(to_json):
        value = to_json()
        if isinstance(value, Mapping):
            return value
    raise TypeError("compiler view must be a mapping or expose to_json()")


def _canonical_abi_name(name: Any, mapping: Mapping[str, str], context: str) -> str:
    """Map a source spelling to its one canonical ABI carrier.

    Facts may be built from an original Harness view or from its anonymous
    projection.  The latter already uses ``p*/c*``; the former must name every
    ABI item through ``source_to_opaque``.  Silently retaining an unknown
    source spelling would make equivalent views hash differently, so it is a
    closed construction error rather than a best-effort conversion.
    """

    if not isinstance(name, str):
        raise TypeError(f"{context} name must be a string")
    canonical = mapping.get(name, name)
    if _ABI_HANDLE_RE.fullmatch(canonical) is None:
        raise ValueError(
            f"{context} name {name!r} has no canonical p*/c* mapping"
        )
    return canonical


def _canonical_compiler_facts(
    raw: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    arguments: Sequence[Mapping[str, Any]],
    parameter_order: Sequence[str],
    constexpr_parameters: Sequence[str],
    compiler_slots: Sequence[str],
) -> dict[str, Any]:
    """Project a compiler view once into the anonymous SourceFacts namespace.

    The source adapter is the only place that may know source spellings.  All
    consumers after Lift preparation must consume this copied, canonical view
    through ``SourceFacts`` instead of independently re-projecting the raw
    compiler view.  It is deliberately data-only: frontend compilation and
    execution remain owned by their respective adapters.
    """

    def canonical_object(field: str, *, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = raw.get(field, default if default is not None else {})
        if not isinstance(value, Mapping):
            raise TypeError(f"compiler view {field} must be an object")
        result: dict[str, Any] = {}
        for name, item in value.items():
            canonical = _canonical_abi_name(name, mapping, f"compiler view {field}")
            if canonical in result:
                raise ValueError(f"compiler view {field} aliases one canonical carrier twice")
            result[canonical] = item
        return result

    schedule_values = canonical_object("schedule_values")
    schedule_types = canonical_object("schedule_value_types")
    bound_values = canonical_object("bound_constexpr_values")
    bound_types = canonical_object("bound_constexpr_value_types")
    bound_coordinates = canonical_object("bound_constexpr_shape_coordinates")
    tile_coordinates = canonical_object("schedule_tile_coordinates")
    if set(schedule_types) - set(schedule_values):
        raise ValueError("compiler schedule types must name declared schedule values")
    if any(
        re.fullmatch(r"c[0-9]+", name) is None
        for name in schedule_values
    ):
        raise ValueError("compiler schedule values must use canonical c* handles")
    if set(bound_types) - set(bound_values) or set(bound_coordinates) - set(bound_values):
        raise ValueError("compiler bound constexpr facts must name bound values")
    if set(tile_coordinates) - set(schedule_values):
        raise ValueError("compiler tile coordinates must name C schedule values")

    raw_roles = raw.get("compiler_roles", {})
    if not isinstance(raw_roles, Mapping):
        raise TypeError("compiler view compiler_roles must be an object")
    roles: dict[str, Any] = {}
    for name, value in raw_roles.items():
        canonical = _canonical_abi_name(
            name, mapping, "compiler view compiler_roles"
        )
        if canonical in roles:
            raise ValueError(
                "compiler view compiler_roles aliases one canonical slot twice"
            )
        roles[canonical] = value
    for source_or_canonical, canonical in zip(
        raw.get("compiler_slots", ()), compiler_slots, strict=True
    ):
        if canonical not in roles:
            # Original views do not encode a separate role map.  Their
            # compiler-slot spelling is the trusted public role; an already
            # anonymous view supplies the same value through compiler_roles.
            roles[canonical] = source_or_canonical
    for canonical in schedule_values:
        roles.setdefault(canonical, "constexpr_parameter")
    if set(compiler_slots) - set(schedule_values):
        raise ValueError(
            "compiler slots must be present in the canonical schedule-value ledger"
        )
    if set(roles) != set(schedule_values) or any(
        not isinstance(role, str) or not role
        for role in roles.values()
    ):
        raise ValueError(
            "compiler roles must cover exactly the canonical C schedule slots"
        )

    raw_aliases = raw.get("aliases", ())
    if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, (str, bytes)):
        raise TypeError("compiler view aliases must be an ordered array")
    aliases: list[dict[str, Any]] = []
    for index, item in enumerate(raw_aliases):
        alias = _copy_mapping(item, f"compiler view aliases[{index}]")
        for alias_field in ("left", "right"):
            if alias_field in alias:
                alias[alias_field] = _canonical_abi_name(
                    alias[alias_field],
                    mapping,
                    f"compiler view aliases[{index}].{alias_field}",
                )
        aliases.append(alias)

    grid_contract = raw.get("grid_contract", ())
    if not isinstance(grid_contract, Sequence) or isinstance(grid_contract, (str, bytes)):
        raise TypeError("compiler view grid_contract must be an ordered array")
    replace_identifier = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    canonical_grid_contract = [
        replace_identifier.sub(
            lambda match: mapping.get(match.group(0), match.group(0)),
            item,
        )
        if isinstance(item, str) else item
        for item in grid_contract
    ]
    return {
        "schedule_values": schedule_values,
        "schedule_value_types": schedule_types,
        "bound_constexpr_values": bound_values,
        "bound_constexpr_value_types": bound_types,
        "bound_constexpr_shape_coordinates": bound_coordinates,
        "compiler_slots": list(compiler_slots),
        "compiler_roles": roles,
        "schedule_tile_coordinates": tile_coordinates,
        "aliases": aliases,
        "arguments": [dict(item) for item in arguments],
        "shape_coordinates": _copy_mapping(
            raw.get("shape_coordinates", {}), "compiler view shape_coordinates"
        ),
        "grid": list(raw.get("grid", ())),
        "grid_contract": canonical_grid_contract,
        "grid_shape_coordinates": list(raw.get("grid_shape_coordinates", ())),
        "parameter_order": list(parameter_order),
        "constexpr_parameters": list(constexpr_parameters),
    }


def _bounded_source_expression(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    try:
        rendered = ast.unparse(value)
    except Exception:
        return None
    return rendered if len(rendered) <= 512 else rendered[:512]


def _address_profile(polynomial: Any) -> tuple[int, int, tuple[int, ...]] | None:
    """Return the closed direct-address profile used by Source/J matching."""

    if not isinstance(polynomial, (list, tuple)):
        return None
    dynamic_terms = 0
    constants: list[int] = []
    for term in polynomial:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            return None
        monomial, coefficient = term
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            return None
        if not isinstance(monomial, (list, tuple)) or any(
            not isinstance(atom, str) for atom in monomial
        ):
            return None
        if not monomial:
            constants.append(coefficient)
            continue
        # Admit only atoms whose identity and integer type were closed by the
        # TTIR observer. Runtime scalar arguments are legitimate standalone
        # offsets; they need not be multiplied by a program/lane term. Unknown
        # region or SSA labels remain unavailable even when another atom in the
        # same monomial is recognized.
        if any(
            re.fullmatch(r"arg\[[0-9]+\]:.+", atom) is None
            and re.fullmatch(r"program_axis:[xyz]", atom) is None
            and not atom.startswith("lane:")
            and not atom.startswith(
                ("integer_relation:", "integer_memory_value:")
            )
            and re.fullmatch(r"region\[scf\.for\]\.arg\[0\]", atom) is None
            for atom in monomial
        ):
            return None
        dynamic_terms += 1
    return dynamic_terms, len(constants), tuple(
        sorted(item for item in constants if item != 0)
    )


AddressTerms = tuple[tuple[tuple[str, ...], int], ...]
AddressTypeLayer = tuple[AddressTerms, str, tuple[str, ...]]
AddressTypeProfile = tuple[AddressTypeLayer, ...]


def _canonical_address_terms(value: Any) -> AddressTerms | None:
    """Accept one closed integer-address polynomial, never a partial row."""

    if not isinstance(value, (list, tuple)):
        return None
    terms: list[tuple[tuple[str, ...], int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        atoms, coefficient = row
        if (
            not isinstance(atoms, (list, tuple))
            or any(not isinstance(atom, str) or not atom for atom in atoms)
            or isinstance(coefficient, bool)
            or not isinstance(coefficient, int)
            or coefficient == 0
        ):
            return None
        terms.append((tuple(atoms), coefficient))
    return tuple(sorted(terms))


def _combine_address_terms(layers: AddressTypeProfile) -> AddressTerms:
    """Return the exact sum of all typed addptr-layer offset polynomials."""

    combined: dict[tuple[str, ...], int] = {}
    for terms, _index_type, _chain in layers:
        for atoms, coefficient in terms:
            combined[atoms] = combined.get(atoms, 0) + coefficient
            if combined[atoms] == 0:
                del combined[atoms]
    return tuple(sorted(combined.items()))


def _coerce_address_type_profile(value: Any) -> AddressTypeProfile | None:
    """Parse one exact per-addptr typed-address proof profile.

    Every row is independently compiler-proven: the offset polynomial added by
    that ``tt.addptr``, its explicit integer/index type, and a bounded
    type-preserving SSA operation chain ending at the offset value. The host
    does not invent casts or merge type evidence from different rows.
    """

    if not isinstance(value, (list, tuple)):
        return None
    layers: list[AddressTypeLayer] = []
    for row in value:
        if isinstance(row, Mapping):
            if set(row) != {"address_terms", "index_type", "ssa_type_chain"}:
                return None
            raw_terms = row["address_terms"]
            if isinstance(raw_terms, list) and all(
                isinstance(term, Mapping)
                and set(term) == {"refs", "coefficient"}
                for term in raw_terms
            ):
                raw_terms = [
                    (term["refs"], term["coefficient"])
                    for term in raw_terms
                ]
            index_type = row["index_type"]
            raw_chain = row["ssa_type_chain"]
        elif isinstance(row, (list, tuple)) and len(row) == 3:
            raw_terms, index_type, raw_chain = row
        else:
            return None
        terms = _canonical_address_terms(raw_terms)
        if (
            not terms
            or not isinstance(index_type, str)
            or not re.fullmatch(r"(?:i[0-9]+|index)", index_type)
            or not isinstance(raw_chain, (list, tuple))
            or not raw_chain
            or any(not isinstance(item, str) or not item for item in raw_chain)
        ):
            return None
        layers.append((terms, index_type, tuple(raw_chain)))
    return tuple(layers)


def _address_type_profile_wire(profile: AddressTypeProfile) -> list[dict[str, Any]]:
    """Serialize host-owned typed-address proof without SSA value spellings."""

    return [
        {
            "address_terms": [
                {"refs": list(atoms), "coefficient": coefficient}
                for atoms, coefficient in terms
            ],
            "index_type": index_type,
            "ssa_type_chain": list(chain),
        }
        for terms, index_type, chain in profile
    ]


@dataclass(frozen=True)
class SourceFacts:
    """One canonical, privacy-safe Source/Harness fact set.

    ``arguments`` retain declaration order because that order is part of the
    ABI.  The actual Source text is never stored; only its digests and the
    selected symbol are retained.
    """

    frontend: str
    source_kind: str
    source_symbol: str
    parameter_order: tuple[str, ...]
    constexpr_parameters: tuple[str, ...] = ()
    source_to_opaque: tuple[tuple[str, str], ...] = ()
    arguments: tuple[Mapping[str, Any], ...] = ()
    compiler_slots: tuple[str, ...] = ()
    shape_coordinates: tuple[tuple[str, int], ...] = ()
    compiler_facts: Mapping[str, Any] = field(default_factory=dict)
    source_digest: str | None = None
    source_slice_digest: str | None = None

    def __post_init__(self) -> None:
        for name, label in (
            (self.frontend, "frontend"),
            (self.source_kind, "source_kind"),
            (self.source_symbol, "source_symbol"),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"SourceFacts.{label} must be non-empty")
        parameter_order = tuple(self.parameter_order)
        constexpr = tuple(self.constexpr_parameters)
        if (
            len(parameter_order) != len(set(parameter_order))
            or any(_ABI_HANDLE_RE.fullmatch(item) is None for item in parameter_order)
        ):
            raise ValueError(
                "SourceFacts parameter order must contain unique p*/c* handles"
            )
        if len(constexpr) != len(set(constexpr)) or not set(constexpr) <= set(parameter_order):
            raise ValueError("SourceFacts constexpr order must be a parameter subset")
        mapping = tuple(sorted(self.source_to_opaque, key=lambda item: item[0]))
        if len({source for source, _opaque in mapping}) != len(mapping):
            raise ValueError("SourceFacts source mapping has duplicate source names")
        if len({opaque for _source, opaque in mapping}) != len(mapping):
            raise ValueError("SourceFacts source mapping has duplicate opaque names")
        if any(
            not isinstance(source, str)
            or not isinstance(opaque, str)
            or not source.isidentifier()
            or _ABI_HANDLE_RE.fullmatch(opaque) is None
            for source, opaque in mapping
        ):
            raise ValueError("SourceFacts source mapping must use p*/c* ABI handles")
        slots = tuple(self.compiler_slots)
        if set(parameter_order) & set(slots):
            raise ValueError(
                "SourceFacts parameter and compiler ABI declarations must not overlap"
            )
        expected_abi = set(parameter_order) | set(slots)
        mapped_abi = {opaque for _source, opaque in mapping}
        if mapped_abi != expected_abi:
            raise ValueError(
                "SourceFacts source mapping must cover parameter and compiler ABI exactly"
            )
        parameter_p_order = tuple(
            name for name in parameter_order if name.startswith("p")
        )
        parameter_c_order = tuple(
            name for name in parameter_order if name.startswith("c")
        )
        mapped_p_order = tuple(
            sorted(
                (name for name in expected_abi if name.startswith("p")),
                key=lambda name: int(name[1:]),
            )
        )
        mapped_c_order = tuple(
            sorted(
                (name for name in expected_abi if name.startswith("c")),
                key=lambda name: int(name[1:]),
            )
        )
        if parameter_p_order != mapped_p_order:
            raise ValueError(
                "SourceFacts p* mapping must preserve runtime parameter order"
            )
        if (*parameter_c_order, *slots) != mapped_c_order:
            raise ValueError(
                "SourceFacts c* mapping must preserve Source-C then compiler-slot order"
            )
        if parameter_p_order != CanonicalHandleAllocator.canonical_names(
            "p", len(parameter_p_order)
        ):
            raise ValueError("SourceFacts p* handles must be dense and declaration-ordered")
        if mapped_c_order != CanonicalHandleAllocator.canonical_names(
            "c", len(mapped_c_order)
        ):
            raise ValueError("SourceFacts c* handles must be dense and declaration-ordered")
        runtime_parameter_order = tuple(
            name for name in parameter_order if name not in constexpr
        )
        arguments = tuple(
            _copy_mapping(item, "SourceFacts.arguments[]")
            for item in self.arguments
        )
        argument_names = tuple(
            item.get("name") for item in arguments if isinstance(item.get("name"), str)
        )
        if len(argument_names) != len(arguments) or len(argument_names) != len(set(argument_names)):
            raise ValueError("SourceFacts arguments must have unique names")
        if argument_names != runtime_parameter_order:
            raise ValueError(
                "SourceFacts runtime arguments must match parameter_order minus constexpr parameters"
            )
        if len(slots) != len(set(slots)) or any(
            not isinstance(name, str) or re.fullmatch(r"c[0-9]+", name) is None
            for name in slots
        ):
            raise ValueError("SourceFacts compiler slots must use unique c* handles")
        coordinates = tuple(self.shape_coordinates)
        if len({name for name, _value in coordinates}) != len(coordinates):
            raise ValueError("SourceFacts shape coordinates must be unique")
        if any(
            not isinstance(name, str)
            or re.fullmatch(r"h[0-9]+", name) is None
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in coordinates
        ):
            raise ValueError("SourceFacts shape coordinates are invalid")
        compiler_facts = _copy_mapping(
            self.compiler_facts, "SourceFacts.compiler_facts"
        )
        # Older closed SourceFacts predate Source-proven tile-coordinate facts.
        # Absence means no tile relation, never permission to infer one.
        compiler_facts.setdefault("schedule_tile_coordinates", {})
        expected_compiler_fact_keys = {
            "schedule_values", "schedule_value_types",
            "bound_constexpr_values", "bound_constexpr_value_types",
            "bound_constexpr_shape_coordinates", "compiler_slots",
            "compiler_roles", "aliases", "arguments", "shape_coordinates",
            "schedule_tile_coordinates", "grid", "grid_contract",
            "grid_shape_coordinates",
            "parameter_order", "constexpr_parameters",
        }
        if set(compiler_facts) != expected_compiler_fact_keys:
            raise ValueError("SourceFacts compiler facts have an incomplete wire shape")
        if tuple(compiler_facts["parameter_order"]) != parameter_order:
            raise ValueError("SourceFacts compiler facts disagree on parameter order")
        if tuple(compiler_facts["constexpr_parameters"]) != constexpr:
            raise ValueError("SourceFacts compiler facts disagree on constexpr order")
        if tuple(compiler_facts["compiler_slots"]) != slots:
            raise ValueError("SourceFacts compiler facts disagree on compiler slots")
        schedule_values = compiler_facts["schedule_values"]
        schedule_types = compiler_facts["schedule_value_types"]
        bound_values = compiler_facts["bound_constexpr_values"]
        bound_types = compiler_facts["bound_constexpr_value_types"]
        bound_coordinates = compiler_facts["bound_constexpr_shape_coordinates"]
        compiler_roles = compiler_facts["compiler_roles"]
        tile_coordinates = compiler_facts["schedule_tile_coordinates"]
        if not all(
            isinstance(item, Mapping)
            for item in (
                schedule_values,
                schedule_types,
                bound_values,
                bound_types,
                bound_coordinates,
                compiler_roles,
                tile_coordinates,
            )
        ):
            raise ValueError("SourceFacts compiler facts have malformed ledgers")
        # Value-stable JSON copying sorts object keys, while this particular
        # map is an ABI ledger whose C declaration sequence is semantic.  Its
        # one valid order comes from the dense shared allocator, never from a
        # downstream view or lexical object ordering.
        if set(schedule_values) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler facts disagree on canonical C declaration order"
            )
        if set(schedule_types) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler facts require one dtype for every C value"
            )
        if set(slots) - set(schedule_values):
            raise ValueError(
                "SourceFacts compiler slots must be present in schedule values"
            )
        if set(compiler_roles) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler roles must cover exactly C schedule slots"
            )
        if set(tile_coordinates) - set(mapped_c_order) or any(
            not isinstance(coordinate, str)
            or coordinate not in dict(coordinates)
            for coordinate in tile_coordinates.values()
        ):
            raise ValueError(
                "SourceFacts tile coordinates must bind C slots to shape coordinates"
            )
        if {
            name
            for name, role in compiler_roles.items()
            if role in {"tile_extent", "reduction_tile_extent"}
        } != set(tile_coordinates):
            raise ValueError(
                "SourceFacts tile coordinates and closed C tile roles differ"
            )
        compiler_facts["schedule_values"] = {
            name: schedule_values[name] for name in mapped_c_order
        }
        compiler_facts["schedule_value_types"] = {
            name: schedule_types[name] for name in mapped_c_order
        }
        compiler_facts["compiler_roles"] = {
            name: compiler_roles[name] for name in mapped_c_order
        }
        compiler_facts["schedule_tile_coordinates"] = {
            name: tile_coordinates[name]
            for name in mapped_c_order
            if name in tile_coordinates
        }
        if set(bound_types) - set(bound_values) or set(bound_coordinates) - set(bound_values):
            raise ValueError(
                "SourceFacts bound constexpr ledgers are not closed"
            )
        if tuple(compiler_facts["arguments"]) != arguments:
            raise ValueError("SourceFacts compiler facts disagree on arguments")
        if tuple(
            sorted(
                ((name, value) for name, value in compiler_facts["shape_coordinates"].items()),
                key=lambda item: item[0],
            )
        ) != coordinates:
            raise ValueError("SourceFacts compiler facts disagree on shape coordinates")
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(self, "constexpr_parameters", constexpr)
        object.__setattr__(self, "source_to_opaque", mapping)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "compiler_slots", slots)
        object.__setattr__(self, "shape_coordinates", coordinates)
        object.__setattr__(self, "compiler_facts", compiler_facts)

    @classmethod
    def from_compiler_view(
        cls,
        view: Any,
        *,
        frontend: str,
        source_kind: str,
        source_symbol: str,
        source_to_opaque: Mapping[str, str] | None = None,
        source_digest: str | None = None,
        source_slice_digest: str | None = None,
    ) -> "SourceFacts":
        raw = _view_json(view)
        raw_mapping = source_to_opaque or {}
        if not isinstance(raw_mapping, Mapping):
            raise TypeError("source_to_opaque must be a mapping")
        mapping: dict[str, str] = {}
        for source, opaque in raw_mapping.items():
            if (
                not isinstance(source, str)
                or not source.isidentifier()
                or not isinstance(opaque, str)
                or _ABI_HANDLE_RE.fullmatch(opaque) is None
            ):
                raise ValueError(
                    "source_to_opaque must map identifiers to p*/c* handles"
                )
            mapping[source] = opaque
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("source_to_opaque must be one-to-one")

        raw_arguments = raw.get("arguments", ())
        if not isinstance(raw_arguments, Sequence) or isinstance(raw_arguments, (str, bytes)):
            raise TypeError("compiler view arguments must be an ordered array")
        arguments: list[dict[str, Any]] = []
        for index, item in enumerate(raw_arguments):
            argument = _copy_mapping(item, f"compiler view argument {index}")
            argument["name"] = _canonical_abi_name(
                argument.get("name"), mapping, f"compiler view argument {index}"
            )
            arguments.append(argument)

        def ordered_names(field: str) -> tuple[str, ...]:
            raw_names = raw.get(field, ())
            if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
                raise TypeError(f"compiler view {field} must be an ordered array")
            return tuple(
                _canonical_abi_name(item, mapping, f"compiler view {field}")
                for item in raw_names
            )

        parameter_order = ordered_names("parameter_order")
        constexpr = ordered_names("constexpr_parameters")
        slots = ordered_names("compiler_slots")
        if any(not name.startswith("c") for name in slots):
            raise ValueError("compiler view compiler_slots must map to c* handles")
        mapped_abi = set(mapping.values())
        for name in (*parameter_order, *slots):
            if name not in mapped_abi:
                if _ABI_HANDLE_RE.fullmatch(name) is None:
                    raise ValueError(
                        "compiler view contains an unmapped ABI name"
                    )
                mapping[name] = name
                mapped_abi.add(name)
        raw_coordinates = raw.get("shape_coordinates", {})
        if not isinstance(raw_coordinates, Mapping):
            raise TypeError("compiler view shape_coordinates must be an object")
        coordinates = tuple(
            sorted(
                (
                    (name, value)
                    for name, value in raw_coordinates.items()
                    if isinstance(name, str)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ),
                key=lambda item: item[0],
            )
        )
        if len(coordinates) != len(raw_coordinates):
            raise ValueError("compiler view shape_coordinates are invalid")
        compiler_facts = _canonical_compiler_facts(
            raw,
            mapping,
            arguments=arguments,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr,
            compiler_slots=slots,
        )
        return cls(
            frontend=frontend,
            source_kind=source_kind,
            source_symbol=source_symbol,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr,
            source_to_opaque=tuple(mapping.items()),
            arguments=tuple(arguments),
            compiler_slots=slots,
            shape_coordinates=coordinates,
            compiler_facts=compiler_facts,
            source_digest=source_digest,
            source_slice_digest=source_slice_digest,
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceFacts":
        raw = _copy_mapping(value, "SourceFacts")
        return cls(
            frontend=raw["frontend"],
            source_kind=raw["source_kind"],
            source_symbol=raw["source_symbol"],
            parameter_order=tuple(raw["parameter_order"]),
            constexpr_parameters=tuple(raw.get("constexpr_parameters", ())),
            source_to_opaque=tuple(
                (row[0], row[1]) for row in raw.get("source_to_opaque", ())
            ),
            arguments=tuple(raw.get("arguments", ())),
            compiler_slots=tuple(raw.get("compiler_slots", ())),
            shape_coordinates=tuple(
                (row[0], row[1]) for row in raw.get("shape_coordinates", ())
            ),
            compiler_facts=raw["compiler_facts"],
            source_digest=raw.get("source_digest"),
            source_slice_digest=raw.get("source_slice_digest"),
        )

    @property
    def opaque_to_source(self) -> Mapping[str, str]:
        return {opaque: source for source, opaque in self.source_to_opaque}

    def argument_by_position(self, position: int | None) -> Mapping[str, Any] | None:
        if position is None or position < 0 or position >= len(self.arguments):
            return None
        return self.arguments[position]

    def logical_ref_for_position(self, position: int | None) -> str | None:
        argument = self.argument_by_position(position)
        value = argument.get("logical_ref") if argument is not None else None
        return value if isinstance(value, str) else None

    def rank_for_logical(self, logical_ref: str | None) -> int | None:
        if not isinstance(logical_ref, str):
            return None
        for argument in self.arguments:
            if argument.get("logical_ref") != logical_ref:
                continue
            shape = argument.get("shape")
            if isinstance(shape, list):
                return len(shape)
            rank = argument.get("rank")
            return rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend,
            "source_kind": self.source_kind,
            "source_symbol": self.source_symbol,
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
            "source_to_opaque": [list(row) for row in self.source_to_opaque],
            "arguments": [dict(row) for row in self.arguments],
            "compiler_slots": list(self.compiler_slots),
            "shape_coordinates": [list(row) for row in self.shape_coordinates],
            "compiler_facts": dict(self.compiler_facts),
            "source_digest": self.source_digest,
            "source_slice_digest": self.source_slice_digest,
        }

    def compiler_view(self) -> dict[str, Any]:
        """Return the sole canonical compiler-side projection for consumers."""

        return _copy_mapping(self.compiler_facts, "SourceFacts.compiler_facts")

    @property
    def digest(self) -> str:
        return _digest(self.to_json())


def source_logical_scalar_parameters(
    facts: SourceFacts,
    source_text: str | None,
) -> tuple[str, ...]:
    """Return Source-proven scalar ABI carriers used as logical values.

    The ABI manifest tells us which parameters are scalars, but not whether a
    scalar is only used to form an address or is consumed by the logical
    computation.  This small Source-side binder makes that distinction once,
    before MCP/J construction.  It is deliberately conservative: a scalar is
    promoted only when a Python AST occurrence is in a value expression; a
    scalar seen only in pointer-update statements remains an ABI carrier.

    This is occurrence evidence, not a semantic operator inference.  CUDA/C++
    text and unparsable sources return no promotions and remain provider-owned
    under the existing fail-closed contract.
    """

    if not isinstance(facts, SourceFacts) or not isinstance(source_text, str):
        return ()
    scalar_handles = {
        item.get("name")
        for item in facts.arguments
        if isinstance(item, Mapping)
        and item.get("kind") == "scalar"
        and isinstance(item.get("name"), str)
    }
    # A bound non-C constexpr can retain a physical ``p*`` ABI carrier while
    # being absent from ``facts.arguments``.  It controls source specialization
    # rather than providing a run-time J value; promoting it merely because it
    # appears in an AST condition creates a dead rank-zero logical input.  The
    # argument manifest is the sole authority for values which may enter J.
    if not scalar_handles:
        return ()
    opaque_to_source = facts.opaque_to_source
    accepted_names = {
        name for handle in scalar_handles
        for name in (handle, opaque_to_source.get(handle))
        if isinstance(name, str)
    }
    pointer_names = {
        item.get("name")
        for item in facts.arguments
        if isinstance(item, Mapping)
        and item.get("kind") in {"tensor", "pointer"}
        and isinstance(item.get("name"), str)
    }
    pointer_names |= {
        opaque_to_source.get(name, name)
        for name in pointer_names
        if isinstance(name, str)
    }
    try:
        tree = ast.parse(source_text, mode="exec")
    except (SyntaxError, TypeError, ValueError):
        return ()
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == facts.source_symbol
    ]
    if len(functions) != 1:
        return ()
    function = functions[0]
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def names_in(node: ast.AST) -> set[str]:
        return {
            item.id for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }

    def statement_root(node: ast.AST) -> ast.AST | None:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.stmt):
                return current
            current = parents.get(current)
        return None

    def is_pointer_update(node: ast.AST) -> bool:
        statement = statement_root(node)
        if isinstance(statement, ast.AugAssign):
            targets = names_in(statement.target)
            if targets & pointer_names:
                return True
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                names_in(statement.target)
                if isinstance(statement, ast.AnnAssign)
                else {
                    name for target in statement.targets for name in names_in(target)
                }
            )
            if targets & pointer_names:
                return True
        return False

    def is_memory_address_context(node: ast.AST) -> bool:
        """Return whether this occurrence contributes only to a memory address.

        A scalar inside the pointer operand of ``tl.load``/``tl.store`` (or
        block-pointer construction) is physically necessary but does not by
        itself establish a J rank-zero value.  The prior generic Call/BinOp
        walk promoted such address strides as logical scalars, leaving a dead
        ``t*`` input that strict TypeCheck correctly rejects.  Keep this
        source-side distinction occurrence-local: the same scalar is still
        promoted when it appears in a load value, predicate, result, or other
        non-address expression elsewhere.
        """

        current: ast.AST | None = node
        address_calls = {"load", "store", "make_block_ptr", "advance"}
        while current is not None:
            parent = parents.get(current)
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute):
                name = parent.func.attr
                if name in address_calls or name.startswith("atomic_"):
                    pointer_arguments: list[ast.AST] = []
                    if parent.args:
                        pointer_arguments.append(parent.args[0])
                    pointer_arguments.extend(
                        keyword.value
                        for keyword in parent.keywords
                        if keyword.arg in {"pointer", "ptr", "base"}
                    )
                    if any(
                        any(descendant is node for descendant in ast.walk(argument))
                        for argument in pointer_arguments
                    ):
                        return True
            current = parent
        return False

    def has_value_context(node: ast.AST) -> bool:
        if is_memory_address_context(node):
            return False
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.Compare | ast.BinOp | ast.BoolOp | ast.UnaryOp):
                return True
            if isinstance(current, ast.Call):
                # A scalar receiver such as ``scale.to(dtype)`` is a value;
                # arange/grid arguments are shape or realization facts only.
                func = current.func
                if isinstance(func, ast.Attribute) and func.attr == "to":
                    return True
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"arange", "program_id"}
                ):
                    return True
            if isinstance(current, ast.Subscript):
                return True
            current = parents.get(current)
        return False

    logical: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id not in accepted_names or is_pointer_update(node):
            continue
        if has_value_context(node):
            canonical = next(
                (handle for handle in scalar_handles
                 if node.id in {handle, opaque_to_source.get(handle)}),
                None,
            )
            if canonical is not None:
                logical.add(canonical)
    parameter_order = tuple(
        name for name in facts.parameter_order
        if name in logical
    )
    return parameter_order

@dataclass(frozen=True)
class TransferOccurrence:
    """One lexical Source memory occurrence, independent of J numbering."""

    ordinal: int
    operation: str
    kind: str
    masked: bool
    predicate: str | None
    fill_value: str | None
    pointer_base: str | None
    pointer_position: int | None
    origin_positions: tuple[int, ...]
    context: tuple[str, ...]
    address_terms: tuple[tuple[tuple[str, ...], int], ...]
    address_profile: tuple[int, int, tuple[int, ...]] | None
    address_type_profile: AddressTypeProfile = ()
    logical_ref: str | None = None
    tensor_rank: int | None = None
    element_type: str | None = None
    semantic_consumer: SourceSemanticOccurrence | None = None
    semantic_predicate: SourceSemanticOccurrence | None = None
    semantic_access: SourceSemanticOccurrence | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or self.kind not in {"load", "store", "update"}
        ):
            raise ValueError("invalid Source transfer occurrence")
        if not isinstance(self.masked, bool):
            raise TypeError("Source transfer masked flag must be boolean")
        if self.element_type is not None and (
            not isinstance(self.element_type, str) or not self.element_type
        ):
            raise ValueError("invalid Source transfer element type")
        if self.semantic_consumer is not None:
            if not isinstance(self.semantic_consumer, SourceSemanticOccurrence):
                raise TypeError("Source transfer consumer occurrence is invalid")
            if self.semantic_consumer.role != "consumer":
                raise ValueError("Source transfer consumer occurrence has invalid role")
        if self.semantic_predicate is not None:
            if not isinstance(self.semantic_predicate, SourceSemanticOccurrence):
                raise TypeError("Source transfer predicate occurrence is invalid")
            if self.semantic_predicate.role != "predicate":
                raise ValueError("Source transfer predicate occurrence has invalid role")
        if self.semantic_access is not None:
            if not isinstance(self.semantic_access, SourceSemanticOccurrence):
                raise TypeError("Source transfer access occurrence is invalid")
            if self.semantic_access.role != "access":
                raise ValueError("Source transfer access occurrence has invalid role")
            if self.semantic_access.operation_family != f"memory_{self.kind}":
                raise ValueError("Source transfer access occurrence has invalid family")
        address_terms = _canonical_address_terms(self.address_terms)
        address_type_profile = _coerce_address_type_profile(
            self.address_type_profile
        )
        if address_terms is None or address_type_profile is None:
            raise ValueError("invalid Source transfer typed address profile")
        if address_terms != _combine_address_terms(address_type_profile):
            raise ValueError(
                "Source transfer typed-address layers differ from address terms"
            )
        object.__setattr__(self, "address_terms", address_terms)
        object.__setattr__(self, "address_type_profile", address_type_profile)
        if self.address_profile is not None:
            dynamic, constants, offsets = self.address_profile
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (dynamic, constants)
            ) or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets):
                raise ValueError("invalid Source address profile")

    def to_json(self) -> dict[str, Any]:
        profile = self.address_profile
        return {
            "ordinal": self.ordinal,
            "operation": self.operation,
            "kind": self.kind,
            "masked": self.masked,
            "predicate": self.predicate,
            "fill_value": self.fill_value,
            "pointer_base": self.pointer_base,
            "pointer_position": self.pointer_position,
            "origin_positions": list(self.origin_positions),
            "context": list(self.context),
            "address_terms": [
                [list(atoms), coefficient]
                for atoms, coefficient in self.address_terms
            ],
            "address_profile": (
                {"dynamic_terms": profile[0], "constant_terms": profile[1], "constant_offsets": list(profile[2])}
                if profile is not None else None
            ),
            "logical_ref": self.logical_ref,
            "tensor_rank": self.tensor_rank,
            "element_type": self.element_type,
            "semantic_consumer": (
                self.semantic_consumer.to_json()
                if self.semantic_consumer is not None else None
            ),
            "semantic_predicate": (
                self.semantic_predicate.to_json()
                if self.semantic_predicate is not None else None
            ),
            "semantic_access": (
                self.semantic_access.to_json()
                if self.semantic_access is not None else None
            ),
            "address_type_profile": _address_type_profile_wire(
                self.address_type_profile
            ),
        }


@dataclass(frozen=True)
class SourceTransferLedger:
    """The sole Source occurrence ledger used by Source/J reconciliation."""

    source_facts_digest: str | None
    status: str
    reason: str
    occurrences: tuple[TransferOccurrence, ...] = ()
    bindings: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"pass", "unavailable", "ambiguous"}:
            raise ValueError("invalid SourceTransferLedger status")
        if self.source_facts_digest is not None and not isinstance(
            self.source_facts_digest, str
        ):
            raise TypeError("SourceTransferLedger source_facts_digest must be a string")
        if not isinstance(self.reason, str):
            raise TypeError("SourceTransferLedger reason must be a string")
        occurrences = tuple(self.occurrences)
        for occurrence in occurrences:
            if (
                not isinstance(occurrence.operation, str)
                or not occurrence.operation
                or occurrence.predicate is not None
                and not isinstance(occurrence.predicate, str)
                or occurrence.fill_value is not None
                and not isinstance(occurrence.fill_value, str)
                or occurrence.pointer_base is not None
                and not isinstance(occurrence.pointer_base, str)
                or any(
                    isinstance(position, bool) or not isinstance(position, int)
                    for position in occurrence.origin_positions
                )
                or any(not isinstance(item, str) for item in occurrence.context)
            ):
                raise ValueError("Source transfer occurrence has invalid metadata")
            if self.status == "pass" and (
                occurrence.pointer_base is None
                or occurrence.element_type is None
                or occurrence.address_terms
                and not occurrence.address_type_profile
            ):
                raise ValueError(
                    "pass Source transfer ledger requires a closed typed address"
                )
            # A pass ledger is the closed Source occurrence authority.  A
            # masked load without an explicit Source ``other=`` value is not
            # an empty-fill fact: it is an unproved occurrence and must be
            # rejected at ledger construction, including JSON rehydration.
            # ``from_source_text`` deliberately uses ``ambiguous`` for its
            # syntax-only navigation projection, so it is not accidentally
            # promoted by this check.
            if (
                self.status == "pass"
                and occurrence.kind == "load"
                and occurrence.masked
                and occurrence.fill_value is None
            ):
                raise ValueError(
                    "pass Source transfer ledger requires masked-load fill_value"
                )
        if tuple(item.ordinal for item in occurrences) != tuple(range(len(occurrences))):
            raise ValueError("Source transfer occurrence ordinals must be contiguous")
        if len({item.ordinal for item in occurrences}) != len(occurrences):
            raise ValueError("Source transfer occurrence ordinals must be unique")
        if self.status == "pass" and occurrences and not self.bindings:
            raise ValueError(
                "pass Source transfer ledger requires occurrence bindings"
            )
        normalized_bindings: list[dict[str, Any]] = []
        for item in self.bindings:
            copied = _copy_mapping(item, "SourceTransferLedger.binding")
            if "address_profile" in copied:
                profile = copied["address_profile"]
                if isinstance(profile, Mapping):
                    profile = (
                        profile.get("dynamic_terms"),
                        profile.get("constant_terms"),
                        tuple(profile.get("constant_offsets", ())),
                    )
                elif isinstance(profile, list):
                    profile = (
                        profile[0],
                        profile[1],
                        tuple(profile[2]),
                    ) if len(profile) == 3 else None
                elif isinstance(profile, tuple) and len(profile) == 3:
                    profile = (profile[0], profile[1], tuple(profile[2]))
                copied["address_profile"] = profile
            if "source_address_type_profile" in copied:
                typed_profile = _coerce_address_type_profile(
                    copied["source_address_type_profile"]
                )
                copied["source_address_type_profile"] = typed_profile
            normalized_bindings.append(copied)
        object.__setattr__(self, "bindings", tuple(normalized_bindings))
        if self.bindings:
            if len(self.bindings) != len(occurrences):
                raise ValueError(
                    "Source transfer bindings must cover every occurrence exactly once"
                )
            seen_ordinals: set[int] = set()
            seen_accesses: set[str] = set()
            seen_effects: set[str] = set()
            for row in self.bindings:
                ordinal = row.get("ordinal")
                access_ref = row.get("access_ref")
                effect_id = row.get("effect_id")
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal < 0
                    or ordinal >= len(occurrences)
                    or ordinal in seen_ordinals
                    or not isinstance(access_ref, str)
                    or not access_ref
                    or re.fullmatch(r"a[0-9]+", access_ref) is None
                    or access_ref in seen_accesses
                    or not isinstance(effect_id, str)
                    or not effect_id
                    or re.fullmatch(r"e[0-9]+", effect_id) is None
                    or effect_id in seen_effects
                ):
                    raise ValueError(
                        "Source transfer bindings must have unique ordinal/access_ref/effect_id"
                    )
                occurrence = occurrences[ordinal]
                occurrence_profile = occurrence.address_profile
                expected_address_profile = (
                    {
                        "dynamic_terms": occurrence_profile[0],
                        "constant_terms": occurrence_profile[1],
                        "constant_offsets": list(occurrence_profile[2]),
                    }
                    if occurrence_profile is not None
                    else None
                )
                expected_index_expression = {
                    "address_terms": [
                        {"refs": list(atoms), "coefficient": coefficient}
                        for atoms, coefficient in occurrence.address_terms
                    ],
                    "address_profile": expected_address_profile,
                }
                if (
                    any(
                        key not in row
                        for key in (
                            "kind",
                            "tensor",
                            "masked",
                            "address_profile",
                            "node_ref",
                            "predicate",
                            "fill_value",
                            "source_other",
                            "source_predicate",
                            "source_pointer_base",
                            "source_element_type",
                            "source_address_type_profile",
                            "semantic_authorizations",
                            "index_expression",
                            "effect_order",
                            "pi_binding",
                        )
                    )
                    or row.get("kind") != occurrence.kind
                    or not isinstance(row.get("tensor"), str)
                    or re.fullmatch(r"t[0-9]+", row["tensor"]) is None
                    or row.get("masked") is not occurrence.masked
                    or row.get("source_other") != occurrence.fill_value
                    or row.get("source_predicate") != occurrence.predicate
                    or row.get("source_pointer_base") != occurrence.pointer_base
                    or row.get("source_element_type") != occurrence.element_type
                    or row.get("source_address_type_profile")
                    != occurrence.address_type_profile
                    or row.get("address_profile") != occurrence.address_profile
                    or (row.get("predicate") is not None) != occurrence.masked
                    or (
                        row.get("node_ref") is not None
                        and (
                            not isinstance(row.get("node_ref"), str)
                            or re.fullmatch(r"n[0-9]+", row["node_ref"]) is None
                        )
                    )
                    or (
                        occurrence.logical_ref is not None
                        and row.get("tensor") != occurrence.logical_ref
                    )
                    or row.get("index_expression") != expected_index_expression
                    or not isinstance(row.get("semantic_authorizations"), Mapping)
                    or (
                        "effect_order" in row
                        and row.get("effect_order") != ordinal
                    )
                    or (
                        "pi_binding" in row
                        and (
                            not isinstance(row.get("pi_binding"), Mapping)
                            or row["pi_binding"].get("status")
                            != "source_address_owner_bound"
                            or row["pi_binding"].get("tensor")
                            != occurrence.logical_ref
                            or row["pi_binding"].get("access_ref")
                            != access_ref
                            or row["pi_binding"].get("effect_id")
                            != effect_id
                            or row["pi_binding"].get("pointer_position")
                            != occurrence.pointer_position
                            or row["pi_binding"].get("element_type")
                            != occurrence.element_type
                            or row["pi_binding"].get("address_type_profile")
                            != _address_type_profile_wire(
                                occurrence.address_type_profile
                            )
                            or row["pi_binding"].get("address_profile")
                            != expected_address_profile
                        )
                    )
                ):
                    raise ValueError(
                        "Source transfer binding disagrees with its occurrence"
                    )
                authorizations = row["semantic_authorizations"]
                if set(authorizations) != {"access_view", "owner", "computed_guard"}:
                    raise ValueError("Source transfer authorizations are incomplete")
                access_view = authorizations["access_view"]
                owner = authorizations["owner"]
                guard = authorizations["computed_guard"]
                if not isinstance(access_view, Mapping) or not isinstance(owner, Mapping) or not isinstance(guard, Mapping):
                    raise ValueError("Source transfer authorization is malformed")
                semantic_access = occurrence.semantic_access
                if (
                    semantic_access is None
                    or set(access_view) != {"status", "token", "operation_family", "compiler_attestation"}
                    or access_view.get("status") != "pass"
                    or access_view.get("token") != semantic_access.token
                    or access_view.get("operation_family") != semantic_access.operation_family
                    or not isinstance(access_view.get("compiler_attestation"), str)
                    or re.fullmatch(r"sc_[0-9a-f]{32}", access_view["compiler_attestation"]) is None
                ):
                    raise ValueError("Source transfer access-view authorization is invalid")
                if owner.get("status") == "pass":
                    semantic = occurrence.semantic_consumer
                    if (
                        semantic is None
                        or set(owner) != {
                            "status", "token", "operation_family",
                            "compiler_attestation",
                        }
                        or owner.get("token") != semantic.token
                        or owner.get("operation_family") != semantic.operation_family
                        or not isinstance(owner.get("compiler_attestation"), str)
                        or re.fullmatch(r"sc_[0-9a-f]{32}", owner["compiler_attestation"])
                        is None
                    ):
                        raise ValueError("Source transfer owner authorization is invalid")
                elif (
                    set(owner) != {"status", "reason"}
                    or owner.get("status") != "unavailable"
                    or not isinstance(owner.get("reason"), str)
                    or not owner["reason"]
                ):
                    raise ValueError("Source transfer owner authorization is invalid")
                if guard.get("status") == "pass":
                    semantic = occurrence.semantic_predicate
                    if (
                        semantic is None
                        or set(guard) != {
                            "status", "token", "operation_family",
                            "compiler_attestation",
                        }
                        or guard.get("token") != semantic.token
                        or guard.get("operation_family") != semantic.operation_family
                        or not isinstance(guard.get("compiler_attestation"), str)
                        or re.fullmatch(r"sc_[0-9a-f]{32}", guard["compiler_attestation"])
                        is None
                    ):
                        raise ValueError("Source transfer guard authorization is invalid")
                elif (
                    set(guard) != {"status", "reason"}
                    or guard.get("status") != "unavailable"
                    or not isinstance(guard.get("reason"), str)
                    or not guard["reason"]
                ):
                    raise ValueError("Source transfer guard authorization is invalid")
                seen_ordinals.add(ordinal)
                seen_accesses.add(access_ref)
                seen_effects.add(effect_id)
            if tuple(
                row.get("ordinal") for row in normalized_bindings
            ) != tuple(range(len(occurrences))):
                raise ValueError(
                    "Source transfer bindings must preserve Source ordinal order"
                )
            if seen_ordinals != set(range(len(occurrences))):
                raise ValueError(
                    "Source transfer bindings must use contiguous occurrence ordinals"
                )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_copy_mapping(item, "SourceTransferLedger.diagnostic") for item in self.diagnostics),
        )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        source_facts: SourceFacts | None = None,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> "SourceTransferLedger":
        return cls(
            source_facts.digest if source_facts is not None else None,
            "unavailable",
            reason,
            diagnostics=tuple(diagnostics),
        )

    @classmethod
    def from_observation(
        cls,
        observed: Any,
        *,
        source_facts: SourceFacts | None = None,
        logical_bindings: Mapping[str, Any] | None = None,
        lexical_ledger: "SourceTransferLedger | None" = None,
    ) -> "SourceTransferLedger":
        """Build the Source occurrence ledger from compiler sidecars.

        Compiler observations use frontend-local pointer spellings such as
        ``arg[0]`` and SSA predicates.  Those spellings are not a second
        semantic authority: each compiler memory event must instead carry a
        unique debug-location counterpart for the exact lexical Source memory
        call.  Source order, AST order, tensor/shape coincidence, and
        provider-local identity are deliberately not matching keys.
        """

        sidecar_names = (
            "diagnostic_memory_transfers",
            "diagnostic_memory_effect_origins",
            "diagnostic_memory_address_forms",
            "diagnostic_memory_source_locations",
            "diagnostic_memory_predicate_evidence",
            "diagnostic_memory_owner_evidence",
        )
        if not all(hasattr(observed, name) for name in sidecar_names):
            return cls.unavailable(
                "source_memory_occurrence_sidecars_incomplete",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                            "one or more required observation sidecars is absent"
                        ],
                    },
                ),
            )
        transfers = tuple(getattr(observed, sidecar_names[0]) or ())
        origins = tuple(getattr(observed, sidecar_names[1]) or ())
        forms = tuple(getattr(observed, sidecar_names[2]) or ())
        source_locations = tuple(getattr(observed, sidecar_names[3]) or ())
        predicate_evidence = tuple(getattr(observed, sidecar_names[4]) or ())
        owner_evidence = tuple(getattr(observed, sidecar_names[5]) or ())
        if (
            len(transfers) != len(origins)
            or len(origins) != len(forms)
            or len(forms) != len(source_locations)
            or len(source_locations) != len(predicate_evidence)
            or len(predicate_evidence) != len(owner_evidence)
        ):
            return cls.unavailable(
                "source_memory_occurrence_sidecars_incomplete",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                                "transfer sidecars disagree "
                                f"(transfers={len(transfers)} origins={len(origins)} "
                                f"forms={len(forms)} locations={len(source_locations)})"
                        ],
                    },
                ),
            )
        # An observed kernel with no memory operations is a valid empty ledger.
        # It must remain distinct from missing or mismatched sidecars above.
        if not transfers:
            return cls(
                source_facts.digest if source_facts is not None else None,
                "pass",
                "lexical_source_memory_occurrences",
            )

        if not isinstance(lexical_ledger, SourceTransferLedger):
            return cls.unavailable(
                "source_memory_occurrence_lexical_ledger_unavailable",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                            "expected:exact lexical Source occurrence ledger"
                        ],
                    },
                ),
            )

        def access_location_key(
            *,
            kind: Any,
            masked: Any,
            line: Any,
            column: Any,
        ) -> tuple[str, bool, int, int] | None:
            if (
                kind not in {"load", "store", "update"}
                or not isinstance(masked, bool)
                or isinstance(line, bool)
                or not isinstance(line, int)
                or line <= 0
                or isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
            ):
                return None
            return (kind, masked, line, column)

        source_to_compiler: dict[tuple[str, bool, int, int], int] = {}
        for compiler_index, location in enumerate(source_locations):
            if not isinstance(location, tuple) or len(location) != 4:
                return cls.unavailable(
                    "source_compiler_access_location_unavailable",
                    source_facts=source_facts,
                )
            key = access_location_key(
                kind=location[0],
                masked=location[1],
                line=location[2],
                column=location[3],
            )
            if key is None or key in source_to_compiler:
                return cls.unavailable(
                    "source_compiler_access_counterpart_ambiguous",
                    source_facts=source_facts,
                )
            source_to_compiler[key] = compiler_index

        def valid_predicate_evidence(
            value: Any,
        ) -> tuple[str, bool, int, int, str | None, int | None, int | None, bool] | None:
            if not isinstance(value, tuple) or len(value) != 8:
                return None
            kind, masked, line, column, operation, predicate_line, predicate_column, is_i1 = value
            key = access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
            )
            if key is None or not isinstance(is_i1, bool):
                return None
            if masked:
                if operation is None:
                    if (
                        predicate_line is not None
                        or predicate_column is not None
                        or is_i1
                    ):
                        return None
                elif not isinstance(operation, str) or not operation:
                    return None
                elif predicate_line is None and predicate_column is None:
                    # A direct predicate producer can survive compiler
                    # lowering without a standalone debug location.  Its
                    # only admissible consumer is the exact matched memory
                    # access that carried this sidecar; Source authorization
                    # must additionally prove a lexical def-use edge.
                    pass
                elif (
                    isinstance(predicate_line, bool)
                    or not isinstance(predicate_line, int)
                    or predicate_line <= 0
                    or isinstance(predicate_column, bool)
                    or not isinstance(predicate_column, int)
                    or predicate_column < 0
                ):
                    return None
            elif any(item is not None for item in (operation, predicate_line, predicate_column)):
                return None
            return (
                kind, masked, line, column, operation, predicate_line,
                predicate_column, is_i1,
            )

        normalized_predicate_evidence = tuple(
            valid_predicate_evidence(item) for item in predicate_evidence
        )
        if any(item is None for item in normalized_predicate_evidence):
            return cls.unavailable(
                "source_compiler_predicate_evidence_unavailable",
                source_facts=source_facts,
            )

        def valid_owner_evidence(
            value: Any,
        ) -> tuple[str, bool, int, int, str | None, int | None, int | None, bool] | None:
            if not isinstance(value, tuple) or len(value) != 8:
                return None
            kind, masked, line, column, operation, owner_line, owner_column, direct_edge = value
            key = access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
            )
            if key is None or not isinstance(direct_edge, bool):
                return None
            if operation is None:
                if direct_edge or owner_line is not None or owner_column is not None:
                    return None
            elif not isinstance(operation, str) or not operation:
                return None
            elif owner_line is None and owner_column is None:
                if not direct_edge:
                    return None
            elif (
                isinstance(owner_line, bool)
                or not isinstance(owner_line, int)
                or owner_line <= 0
                or isinstance(owner_column, bool)
                or not isinstance(owner_column, int)
                or owner_column < 0
            ):
                return None
            return (
                kind, masked, line, column, operation, owner_line,
                owner_column, direct_edge,
            )

        normalized_owner_evidence = tuple(
            valid_owner_evidence(item) for item in owner_evidence
        )
        if any(item is None for item in normalized_owner_evidence):
            return cls.unavailable(
                "source_compiler_owner_evidence_unavailable",
                source_facts=source_facts,
            )

        lexical_by_compiler: dict[int, TransferOccurrence] = {}
        for lexical in lexical_ledger.occurrences:
            semantic_access = lexical.semantic_access
            if (
                semantic_access is None
                or semantic_access.source_line is None
                or semantic_access.source_column is None
            ):
                return cls.unavailable(
                    "source_semantic_access_location_unavailable",
                    source_facts=source_facts,
                )
            key = access_location_key(
                kind=lexical.kind,
                masked=lexical.masked,
                line=semantic_access.source_line,
                column=semantic_access.source_column,
            )
            compiler_index = source_to_compiler.get(key) if key is not None else None
            if compiler_index is None or compiler_index in lexical_by_compiler:
                return cls.unavailable(
                    "source_compiler_access_counterpart_ambiguous",
                    source_facts=source_facts,
                )
            lexical_by_compiler[compiler_index] = lexical
        if len(lexical_by_compiler) != len(transfers):
            return cls.unavailable(
                "source_compiler_access_counterpart_unavailable",
                source_facts=source_facts,
            )
        occurrences: list[TransferOccurrence] = []
        # The target ledger's ordinal remains the lexical Source occurrence
        # identifier, but every row below was selected by its unique Source /
        # compiler coordinate evidence rather than by either list position.
        for lexical in lexical_ledger.occurrences:
            semantic_access = lexical.semantic_access
            assert semantic_access is not None  # checked while constructing the map
            key = access_location_key(
                kind=lexical.kind,
                masked=lexical.masked,
                line=semantic_access.source_line,
                column=semantic_access.source_column,
            )
            assert key is not None
            compiler_index = source_to_compiler[key]
            transfer = transfers[compiler_index]
            origin = origins[compiler_index]
            form = forms[compiler_index]
            ordinal = lexical.ordinal
            operation, origin_positions, context = origin
            if not isinstance(form, (tuple, list)) or len(form) != 6:
                return cls.unavailable(
                    "source_memory_occurrence_typed_address_incomplete",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_typed_address_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].index_expression",
                            "observed_refs_or_types": ["typed_address_fields=missing"],
                        },
                    ),
                )
            (
                form_operation,
                pointer_base,
                polynomial,
                form_context,
                element_type,
                raw_address_type_profile,
            ) = form
            if operation != form_operation or tuple(context) != tuple(form_context):
                return cls.unavailable(
                    "source_memory_occurrence_sidecars_disagree",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_ledger_disagreement",
                            "path": f"<source-transfer-ledger>[{ordinal}]",
                            "observed_refs_or_types": [str(operation), str(form_operation)],
                        },
                    ),
                )
            kind = getattr(transfer, "kind", None)
            masked = getattr(transfer, "masked", None)
            predicate = getattr(transfer, "predicate", None)
            fill = getattr(transfer, "other", None)
            if kind not in {"load", "store", "update"} or not isinstance(masked, bool):
                return cls.unavailable("source_memory_occurrence_invalid", source_facts=source_facts)
            pointer_position = None
            if isinstance(pointer_base, str):
                match = _POINTER_RE.match(pointer_base)
                if match is not None:
                    pointer_position = int(match.group(1))
            logical_ref = None
            if (
                source_facts is not None
                and pointer_position is not None
                and logical_bindings is not None
            ):
                argument = source_facts.argument_by_position(pointer_position)
                argument_name = (
                    argument.get("name")
                    if isinstance(argument, Mapping)
                    else None
                )
                binding = (
                    logical_bindings.get(argument_name)
                    if isinstance(argument_name, str)
                    else None
                )
                if isinstance(binding, (tuple, list)) and binding:
                    candidate = binding[0]
                    if isinstance(candidate, str) and re.fullmatch(
                        r"t[0-9]+", candidate
                    ):
                        logical_ref = candidate
            if logical_ref is None and source_facts is not None:
                logical_ref = source_facts.logical_ref_for_position(pointer_position)
            # The lexical Source expression is the provider-facing
            # semantic predicate/fill.  The compiler predicate is kept only
            # when no exact lexical expression exists; it may be an opaque
            # SSA name and will not be published as J semantics.  ``lexical``
            # was selected above by the exact Source/compiler access location,
            # never by ordinal or a semantic guess.
            if isinstance(lexical.predicate, str):
                predicate = lexical.predicate
            if isinstance(lexical.fill_value, str):
                fill = lexical.fill_value
            # ``other=`` is part of the Source occurrence contract for every
            # masked load. The exact lexical Source occurrence and compiler
            # observation are both legitimate evidence, so reject only after
            # the lexical value has had a chance to close a TTIR-side omission.
            # No default or inferred fill is introduced here.
            if kind == "load" and masked and not isinstance(fill, str):
                return cls.unavailable(
                    "source_masked_load_fill_missing",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_fill_presence_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].other",
                            "observed_refs_or_types": [
                                "masked Source load has no explicit other= fill"
                            ],
                        },
                    ),
                )
            if masked and not isinstance(predicate, str):
                return cls.unavailable(
                    "source_masked_access_predicate_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_predicate_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].predicate",
                            "observed_refs_or_types": [
                                "masked Source access has no closed predicate"
                            ],
                        },
                    ),
                )
            if logical_ref is None:
                return cls.unavailable(
                    "source_memory_occurrence_owner_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_tensor_owner_unavailable",
                            "path": f"<source-transfer-ledger>[{ordinal}].tensor",
                            "observed_refs_or_types": [
                                f"pointer_position={pointer_position}",
                                "expected:unique SourceFacts/allocator logical tensor owner",
                            ],
                        },
                    ),
                )
            address_profile = _address_profile(polynomial)
            address_terms = _canonical_address_terms(polynomial)
            address_type_profile = _coerce_address_type_profile(
                raw_address_type_profile
            )
            address_failure = (
                "address_relation_unavailable"
                if address_profile is None or address_terms is None
                else "element_type_unavailable"
                if not isinstance(element_type, str) or not element_type
                else "address_type_profile_unavailable"
                if address_terms and not address_type_profile
                else "address_type_profile_disagrees"
                if address_type_profile is None
                or address_terms != _combine_address_terms(address_type_profile)
                else None
            )
            if address_failure is not None:
                return cls.unavailable(
                    "source_memory_occurrence_address_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_index_expression_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].index_expression",
                            "observed_refs_or_types": [
                                f"reason={address_failure}",
                                f"offset_term_count={len(address_terms) if address_terms is not None else -1}",
                            ],
                        },
                    ),
                )
            occurrences.append(
                TransferOccurrence(
                    ordinal=ordinal,
                    operation=str(operation),
                    kind=kind,
                    masked=masked,
                    predicate=predicate if isinstance(predicate, str) else None,
                    fill_value=fill if isinstance(fill, str) else None,
                    pointer_base=pointer_base if isinstance(pointer_base, str) else None,
                    pointer_position=pointer_position,
                    origin_positions=tuple(
                        item for item in origin_positions
                        if isinstance(item, int) and not isinstance(item, bool)
                    ),
                    context=tuple(item for item in context if isinstance(item, str)),
                    address_terms=address_terms,
                    address_profile=address_profile,
                    address_type_profile=address_type_profile,
                    logical_ref=logical_ref,
                    tensor_rank=(
                        source_facts.rank_for_logical(logical_ref)
                        if source_facts is not None else None
                    ),
                    element_type=element_type,
                    semantic_consumer=(
                        lexical.semantic_consumer
                    ),
                    semantic_predicate=(
                        lexical.semantic_predicate
                    ),
                    semantic_access=lexical.semantic_access,
                )
            )
        raw_locations = tuple(
            getattr(observed, "diagnostic_source_locations", ()) or ()
        )
        locations_by_span: dict[tuple[int, int], list[str]] = {}
        for row in raw_locations:
            if (
                not isinstance(row, tuple)
                or len(row) != 3
                or not isinstance(row[0], str)
                or not isinstance(row[1], int)
                or isinstance(row[1], bool)
                or not isinstance(row[2], int)
                or isinstance(row[2], bool)
            ):
                continue
            locations_by_span.setdefault((row[1], row[2]), []).append(row[0])

        def compiler_family_matches(operation: str, family: str) -> bool:
            if family == "map_binary":
                return operation.startswith("arith.") or operation.startswith("math.")
            if family == "map_unary":
                return operation.startswith("arith.") or operation.startswith("math.")
            if family == "select":
                return operation in {"arith.select", "tt.select"}
            if family == "cast":
                return operation.startswith("arith.") or operation in {
                    "tt.fp_to_fp", "tt.bitcast"
                }
            if family == "reduce":
                return operation in {"tt.reduce", "tt.scan"}
            if family == "iota":
                return operation == "tt.make_range"
            return operation in {
                f"tt.{family}",
                f"tensor.{family}",
            }

        def owner_authorization(
            occurrence: TransferOccurrence,
            compiler_index: int,
        ) -> dict[str, Any]:
            semantic = occurrence.semantic_consumer
            if semantic is None:
                return {
                    "status": "unavailable",
                    "reason": "source_semantic_consumer_unavailable",
                }
            if semantic.source_line is None or semantic.source_column is None:
                return {
                    "status": "unavailable",
                    "reason": "source_semantic_consumer_location_unavailable",
                }
            matches = [
                operation
                for operation in locations_by_span.get(
                    (semantic.source_line, semantic.source_column), ()
                )
                if compiler_family_matches(operation, semantic.operation_family)
            ]
            if len(matches) == 1:
                compiler_operation = matches[0]
            elif len(matches) > 1:
                return {
                    "status": "unavailable",
                    "reason": "source_compiler_consumer_counterpart_ambiguous",
                }
            else:
                evidence = normalized_owner_evidence[compiler_index]
                assert evidence is not None
                (
                    evidence_kind, evidence_masked, access_line, access_column,
                    owner_operation, owner_line, owner_column, direct_owner_edge,
                ) = evidence
                access_semantic = occurrence.semantic_access
                has_owner_location = owner_line is not None or owner_column is not None
                if (
                    access_semantic is None
                    or access_semantic.source_line is None
                    or access_semantic.source_column is None
                    or evidence_kind != occurrence.kind
                    or evidence_masked is not occurrence.masked
                    or (access_line, access_column) != (
                        access_semantic.source_line,
                        access_semantic.source_column,
                    )
                    or has_owner_location
                    or not direct_owner_edge
                    or semantic.binding_mode not in {"direct", "exact_local_binding"}
                    or not isinstance(owner_operation, str)
                    or not compiler_family_matches(
                        owner_operation, semantic.operation_family
                    )
                ):
                    return {
                        "status": "unavailable",
                        "reason": "source_compiler_consumer_counterpart_unavailable",
                    }
                compiler_operation = owner_operation
            attestation = hashlib.sha256(
                (
                    f"{source_facts.digest if source_facts is not None else ''}|"
                    f"{semantic.token}|{compiler_operation}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            return {
                "status": "pass",
                "token": semantic.token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        def access_view_authorization(
            occurrence: TransferOccurrence,
            compiler_index: int,
        ) -> dict[str, Any]:
            semantic = occurrence.semantic_access
            if semantic is None:
                raise ValueError("exact Source access anchor is unavailable")
            transfer = transfers[compiler_index]
            if transfer.kind != occurrence.kind or transfer.masked is not occurrence.masked:
                raise ValueError("exact Source access/compiler binding is unavailable")
            attestation = hashlib.sha256(
                (
                    f"{source_facts.digest if source_facts is not None else ''}|"
                    f"{semantic.token}|{semantic.operation_family}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            return {
                "status": "pass",
                "token": semantic.token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        def guard_authorization(
            occurrence: TransferOccurrence,
            compiler_index: int,
        ) -> dict[str, Any]:
            semantic = occurrence.semantic_predicate
            evidence = normalized_predicate_evidence[compiler_index]
            assert evidence is not None
            (
                evidence_kind, evidence_masked, access_line, access_column,
                predicate_operation, predicate_line, predicate_column,
                predicate_is_i1,
            ) = evidence
            access_semantic = occurrence.semantic_access
            if not occurrence.masked:
                return {
                    "status": "unavailable",
                    "reason": "source_computed_guard_not_applicable",
                }
            if (
                semantic is None
                or access_semantic is None
                or semantic.source_line is None
                or semantic.source_column is None
                or access_semantic.source_line is None
                or access_semantic.source_column is None
            ):
                return {
                    "status": "unavailable",
                    "reason": "source_predicate_access_anchor_unavailable",
                }
            has_predicate_location = (
                predicate_line is not None or predicate_column is not None
            )
            predicate_location_matches = (
                has_predicate_location
                and (predicate_line, predicate_column) == (
                    semantic.source_line,
                    semantic.source_column,
                )
            )
            # Some Triton lowerings retain the exact predicate SSA edge on a
            # located memory operation but omit the producer's standalone
            # debug location.  This is eligible only for the lexical predicate
            # occurrence already bound directly (or through the one strict
            # straight-line local def-use rule) to this Source access.  The
            # compiler relation is the direct operand of the *already exact*
            # matched access, not a lookup by compiler order, op count, name,
            # or provider-local value.
            direct_access_predicate_edge = (
                not has_predicate_location
                and semantic.binding_mode in {"direct", "exact_local_binding"}
            )
            if (
                evidence_kind != occurrence.kind
                or evidence_masked is not True
                or (access_line, access_column) != (
                    access_semantic.source_line, access_semantic.source_column
                )
                or not (predicate_location_matches or direct_access_predicate_edge)
                or not isinstance(predicate_operation, str)
                or not predicate_is_i1
                or not compiler_family_matches(
                    predicate_operation, semantic.operation_family
                )
            ):
                return {
                    "status": "unavailable",
                    "reason": "source_compiler_predicate_access_projection_unavailable",
                }
            attestation = hashlib.sha256(
                (
                    f"{source_facts.digest if source_facts is not None else ''}|"
                    f"{semantic.token}|{occurrence.semantic_access.token}|"
                    f"{predicate_operation}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            return {
                "status": "pass",
                "token": semantic.token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        bindings: list[dict[str, Any]] = []
        for occurrence in occurrences:
            profile = occurrence.address_profile
            if profile is None or occurrence.logical_ref is None:
                return cls.unavailable(
                    "source_memory_occurrence_binding_incomplete",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_binding_incomplete",
                            "path": f"<source-transfer-ledger>[{occurrence.ordinal}]",
                            "observed_refs_or_types": [
                                "owner, index expression and address-owner binding are required"
                            ],
                        },
                    ),
                )
            index_expression = {
                "address_terms": [
                    {"refs": list(atoms), "coefficient": coefficient}
                    for atoms, coefficient in occurrence.address_terms
                ],
                "address_profile": {
                    "dynamic_terms": profile[0],
                    "constant_terms": profile[1],
                    "constant_offsets": list(profile[2]),
                },
            }
            access_ref = f"a{occurrence.ordinal}"
            effect_id = f"e{occurrence.ordinal}"
            bindings.append(
                {
                    "ordinal": occurrence.ordinal,
                    "access_ref": access_ref,
                    "effect_id": effect_id,
                    "effect_order": occurrence.ordinal,
                    "kind": occurrence.kind,
                    "tensor": occurrence.logical_ref,
                    "masked": occurrence.masked,
                    "node_ref": None,
                    "predicate": occurrence.predicate,
                    "fill_value": occurrence.fill_value,
                    "index_expression": index_expression,
                    "address_profile": profile,
                    "source_other": occurrence.fill_value,
                    "source_predicate": occurrence.predicate,
                    "source_pointer_base": occurrence.pointer_base,
                    "source_element_type": occurrence.element_type,
                    "source_address_type_profile": (
                        occurrence.address_type_profile
                    ),
                    "semantic_authorizations": {
                        "access_view": access_view_authorization(
                            occurrence,
                            source_to_compiler[
                                access_location_key(
                                    kind=occurrence.kind,
                                    masked=occurrence.masked,
                                    line=occurrence.semantic_access.source_line,
                                    column=occurrence.semantic_access.source_column,
                                )
                            ],
                        ),
                        "owner": owner_authorization(
                            occurrence,
                            source_to_compiler[
                                access_location_key(
                                    kind=occurrence.kind,
                                    masked=occurrence.masked,
                                    line=occurrence.semantic_access.source_line,
                                    column=occurrence.semantic_access.source_column,
                                )
                            ],
                        ),
                        "computed_guard": guard_authorization(
                            occurrence,
                            source_to_compiler[
                                access_location_key(
                                    kind=occurrence.kind,
                                    masked=occurrence.masked,
                                    line=occurrence.semantic_access.source_line,
                                    column=occurrence.semantic_access.source_column,
                                )
                            ],
                        ),
                    },
                    "pi_binding": {
                        "status": "source_address_owner_bound",
                        "access_ref": access_ref,
                        "effect_id": effect_id,
                        "tensor": occurrence.logical_ref,
                        "pointer_position": occurrence.pointer_position,
                        "element_type": occurrence.element_type,
                        "address_type_profile": _address_type_profile_wire(
                            occurrence.address_type_profile
                        ),
                        "address_profile": index_expression["address_profile"],
                    },
                }
            )
        return cls(
            source_facts.digest if source_facts is not None else None,
            "pass",
            "lexical_source_memory_occurrences",
            tuple(occurrences),
            tuple(bindings),
        )

    @classmethod
    def from_source_text(
        cls,
        source: str,
        *,
        source_kind: str,
        source_facts: SourceFacts | None = None,
    ) -> "SourceTransferLedger":
        """Create the MCP navigation projection from exact source text.

        This deliberately records only lexical memory-call order, a literal
        mask/fill spelling, and source scope.  It is not accepted by Source
        reconciliation: no pointer form, typed address, or J access binding is
        available until the selected frontend compiler produces its observation.
        Keeping it in the same ledger type means the MCP reads a projection of
        the host construction authority instead of recreating occurrence order
        from several independent buckets.
        """

        if not isinstance(source, str) or not source.strip():
            return cls.unavailable("source_text_absent", source_facts=source_facts)
        rows: list[TransferOccurrence] = []
        if source_kind == "cuda":
            for line_number, raw_line in enumerate(source.splitlines(), start=1):
                # This is only a Source-navigation projection; TTIR remains
                # the typed occurrence authority.  Still, preserve every
                # lexical call.  Keeping only the first call on a line loses
                # an occurrence before SourceReconcile can compare it.
                matches = re.finditer(
                    r"\b(load|store|atomic_[A-Za-z0-9_]+|atomic[A-Za-z0-9_]*)\b",
                    raw_line,
                )
                for match in matches:
                    token = match.group(1)
                    rows_kind = (
                        "load"
                        if token == "load"
                        else "store"
                        if token == "store"
                        else "update"
                    )
                    rows.append(
                        TransferOccurrence(
                            ordinal=len(rows),
                            operation=token,
                            kind=rows_kind,
                            masked="mask" in raw_line.casefold(),
                            predicate=None,
                            fill_value=None,
                            pointer_base=None,
                            pointer_position=None,
                            origin_positions=(line_number,),
                            context=(),
                            address_terms=(),
                            address_profile=None,
                            address_type_profile=(),
                            logical_ref=None,
                            tensor_rank=None,
                            element_type=None,
                        )
                    )
            return cls(
                source_facts.digest if source_facts is not None else None,
                # A non-empty lexical projection has occurrence order, but
                # no typed pointer/ABI/access/effect binding yet.  It is
                # navigation input for MCP only, not a closed Source ledger
                # that SourceReconcile, fill validation, or pi binding may use.
                "ambiguous" if rows else "pass",
                (
                    "source_native_memory_navigation_untyped"
                    if rows
                    else "source_native_memory_navigation"
                ),
                tuple(rows),
            )
        if source_kind != "python":
            return cls.unavailable(
                "source_text_memory_ledger_unsupported_kind",
                source_facts=source_facts,
            )
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return cls.unavailable("source_text_ast_unavailable", source_facts=source_facts)

        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        parent_fields = {
            id(child): (parent, field_name)
            for parent in ast.walk(tree)
            for field_name, value in ast.iter_fields(parent)
            for child in (
                value
                if isinstance(value, list)
                else (value,)
            )
            if isinstance(child, ast.AST)
        }
        source_identity = (
            source_facts.digest
            if isinstance(source_facts, SourceFacts)
            else hashlib.sha256(source.encode("utf-8")).hexdigest()
        )

        control_regions = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.Match,
            *(
                (ast.TryStar,)
                if hasattr(ast, "TryStar")
                else ()
            ),
        )

        def enclosing_function(node: ast.AST) -> ast.AST | None:
            """Return the lexical function scope owning ``node``."""

            current: ast.AST | None = node
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current
                current = parents.get(id(current))
            return None

        def scope_store_counts(function: ast.AST) -> dict[str, int]:
            """Count every lexical write in one function, excluding children.

            This is intentionally broader than the accepted direct-definition
            profile below.  Any alternate assignment, including one in a
            branch or loop, turns the name ambiguous rather than allowing a
            Source-order guess.
            """

            counts: dict[str, int] = {}

            class _StoreVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_AsyncFunctionDef(
                    self,
                    node: ast.AsyncFunctionDef,
                ) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_Name(self, node: ast.Name) -> None:
                    if isinstance(node.ctx, ast.Store):
                        counts[node.id] = counts.get(node.id, 0) + 1

            _StoreVisitor().visit(function)
            return counts

        # A mask spelling may be a local name even though its unique semantic
        # producer is the RHS of a preceding assignment.  Establish that
        # lexical def-use edge only in the narrow straight-line profile below.
        # It is not a Source-order correspondence between independent nodes:
        # the producer remains tied to the load solely through the Python
        # binding and later requires an exact compiler debug-location match.
        direct_definitions: dict[int, dict[str, tuple[ast.AST, int]]] = {}
        store_counts_by_scope: dict[int, dict[str, int]] = {}
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope_id = id(function)
            store_counts_by_scope[scope_id] = scope_store_counts(function)
            definitions: dict[str, tuple[ast.AST, int]] = {}
            for statement_index, statement in enumerate(function.body):
                target: ast.Name | None = None
                value: ast.AST | None = None
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    target = statement.targets[0]
                    value = statement.value
                elif isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target,
                    ast.Name,
                ) and statement.value is not None:
                    target = statement.target
                    value = statement.value
                if (
                    target is None
                    or value is None
                    or _source_semantic_operation_family(value) is None
                    or target.id in definitions
                ):
                    continue
                definitions[target.id] = (value, statement_index)
            direct_definitions[scope_id] = definitions

        def direct_statement_index(
            node: ast.AST,
            function: ast.AST,
        ) -> int | None:
            """Return a direct function-body statement index, or reject.

            Any control region between the use and the function body makes
            reaching-definition analysis non-unique for this collector.
            """

            current = node
            while current is not function:
                parent_field = parent_fields.get(id(current))
                if parent_field is None:
                    return None
                parent, field_name = parent_field
                if isinstance(parent, control_regions):
                    return None
                if parent is function:
                    if field_name != "body" or not isinstance(current, ast.stmt):
                        return None
                    try:
                        return list(function.body).index(current)
                    except ValueError:
                        return None
                current = parent
            return None

        def resolved_mask_semantic_anchor(
            mask: ast.AST | None,
        ) -> tuple[ast.AST | None, str]:
            """Resolve one unambiguous local mask binding, otherwise fail closed."""

            if not isinstance(mask, ast.Name):
                return mask, "direct"
            function = enclosing_function(mask)
            if function is None:
                return None, "direct"
            scope_id = id(function)
            if store_counts_by_scope.get(scope_id, {}).get(mask.id) != 1:
                return None, "direct"
            definition = direct_definitions.get(scope_id, {}).get(mask.id)
            use_index = direct_statement_index(mask, function)
            if definition is None or use_index is None:
                return None, "direct"
            value, definition_index = definition
            if definition_index >= use_index:
                return None, "direct"
            return value, "exact_local_binding"

        def direct_load_consumer_anchor(
            memory_call: ast.Call,
        ) -> tuple[ast.AST | None, str]:
            """Return one uniquely bound straight-line load consumer.

            A boundary load may name a Layer-A owner only when Python itself
            proves a single lexical def-use edge: a direct function-body
            assignment of the load to one local, followed by exactly one
            direct function-body semantic RHS that reads that local exactly
            once.  Any extra read, alternate store, control region, nested
            function, or non-semantic use leaves the owner unavailable.

            This establishes only the Source lexical edge.  The caller still
            requires an exact compiler debug-location/family counterpart, so
            neither Source order nor an AST/candidate label can select an
            owner.
            """

            parent = parents.get(id(memory_call))
            if not isinstance(parent, ast.Assign):
                return None, "direct"
            if (
                len(parent.targets) != 1
                or not isinstance(parent.targets[0], ast.Name)
                or parent.value is not memory_call
            ):
                return None, "direct"
            function = enclosing_function(memory_call)
            if function is None or enclosing_function(parent) is not function:
                return None, "direct"
            load_statement_index = direct_statement_index(memory_call, function)
            local = parent.targets[0].id
            if (
                load_statement_index is None
                or store_counts_by_scope.get(id(function), {}).get(local) != 1
            ):
                return None, "direct"

            uses: list[ast.Name] = []

            class _UseVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_Name(self, node: ast.Name) -> None:
                    if node.id == local and isinstance(node.ctx, ast.Load):
                        uses.append(node)

            _UseVisitor().visit(function)
            if len(uses) != 1:
                return None, "direct"
            use = uses[0]
            if enclosing_function(use) is not function:
                return None, "direct"
            use_statement_index = direct_statement_index(use, function)
            if use_statement_index is None or use_statement_index <= load_statement_index:
                return None, "direct"
            statement = function.body[use_statement_index]
            semantic_value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and _source_semantic_operation_family(statement.value) is not None
            ):
                semantic_value = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
                and _source_semantic_operation_family(statement.value) is not None
            ):
                semantic_value = statement.value
            if semantic_value is None:
                return None, "direct"
            semantic_uses = [
                node
                for node in ast.walk(semantic_value)
                if isinstance(node, ast.Name)
                and node.id == local
                and isinstance(node.ctx, ast.Load)
            ]
            if semantic_uses != [use]:
                return None, "direct"
            return semantic_value, "exact_local_binding"

        def resolved_store_value_anchor(
            memory_call: ast.Call,
        ) -> tuple[ast.AST | None, str]:
            """Return a unique direct semantic producer for one store value.

            The accepted local-name form is intentionally the same narrow,
            straight-line definition profile used for local masks.  It proves
            the value producer through a Python lexical binding; the later
            compiler-location/family check remains mandatory before a receipt
            can be issued.
            """

            if len(memory_call.args) < 2:
                return None, "direct"
            value = memory_call.args[1]
            if _source_semantic_operation_family(value) is not None:
                return value, "direct"
            if not isinstance(value, ast.Name):
                return None, "direct"
            function = enclosing_function(value)
            if function is None:
                return None, "direct"
            scope_id = id(function)
            if store_counts_by_scope.get(scope_id, {}).get(value.id) != 1:
                return None, "direct"
            definition = direct_definitions.get(scope_id, {}).get(value.id)
            use_index = direct_statement_index(value, function)
            if definition is None or use_index is None:
                return None, "direct"
            semantic_value, definition_index = definition
            if definition_index >= use_index:
                return None, "direct"
            return semantic_value, "exact_local_binding"

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            @staticmethod
            def _compiler_access_anchor(
                memory_call: ast.Call,
                kind: str,
            ) -> ast.AST | None:
                """Return the exact Source child owning Triton's memory location.

                Triton's direct ``tt.load`` location is carried by the Source
                pointer expression, while ``tt.store``/atomic update carry the
                stored value expression.  A Call's own span and its callee/mask
                spans are not compiler locations.  This is an operation-kind
                rule, not a Source-order or AST-order correspondence: a missing
                required operand leaves the occurrence unanchored and therefore
                unauthorized.
                """

                if kind == "load":
                    return memory_call.args[0] if memory_call.args else None
                if kind in {"store", "update"}:
                    return (
                        memory_call.args[1]
                        if len(memory_call.args) >= 2
                        else None
                    )
                return None

            def visit_Call(self, node: ast.Call) -> None:
                leaf = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                # Python evaluates the callable and its arguments before the
                # enclosing call takes effect.  Visit children first so a
                # nested Source load/store receives the earlier ledger
                # ordinal instead of being recorded after its enclosing call.
                self.generic_visit(node)
                if leaf in _SOURCE_MEMORY_CALLS:
                    kind = (
                        "load"
                        if leaf == "load"
                        else "store"
                        if leaf == "store"
                        else "update"
                    )
                    keywords = {
                        item.arg: item.value
                        for item in node.keywords
                        if item.arg is not None
                    }
                    mask = keywords.get("mask")
                    fill = keywords.get("other")
                    predicate_node, predicate_binding_mode = (
                        resolved_mask_semantic_anchor(mask)
                    )
                    consumer_node, consumer_binding_mode = (
                        direct_load_consumer_anchor(node)
                        if kind == "load"
                        else resolved_store_value_anchor(node)
                    )
                    rows.append(
                        TransferOccurrence(
                            ordinal=len(rows),
                            operation=str(leaf),
                            kind=kind,
                            masked=mask is not None,
                            predicate=_bounded_source_expression(mask),
                            fill_value=_bounded_source_expression(fill),
                            pointer_base=None,
                            pointer_position=None,
                            origin_positions=(
                                node.lineno
                                if isinstance(getattr(node, "lineno", None), int)
                                else 0,
                            ),
                            context=tuple(self.scope),
                            address_terms=(),
                            address_profile=None,
                            address_type_profile=(),
                            logical_ref=None,
                            tensor_rank=None,
                            element_type=None,
                            semantic_consumer=_opaque_source_semantic_occurrence(
                                consumer_node,
                                role="consumer",
                                scope=self.scope,
                                source_identity=source_identity,
                                binding_mode=consumer_binding_mode,
                            ),
                            semantic_predicate=_opaque_source_semantic_occurrence(
                                predicate_node,
                                role="predicate",
                                scope=self.scope,
                                source_identity=source_identity,
                                binding_mode=predicate_binding_mode,
                            ) if mask is not None else None,
                            semantic_access=_opaque_source_semantic_occurrence(
                                self._compiler_access_anchor(node, kind),
                                role="access",
                                scope=self.scope,
                                source_identity=source_identity,
                                operation_family=f"memory_{kind}",
                            ),
                        )
                    )

        _Visitor().visit(tree)
        return cls(
            source_facts.digest if source_facts is not None else None,
            # The AST projection preserves lexical occurrence order for the
            # mounted MCP brief, but it deliberately lacks compiled address
            # and ABI bindings.  Keep it distinct from a closed empty ledger.
            "ambiguous" if rows else "pass",
            (
                "source_syntax_memory_navigation_untyped"
                if rows
                else "source_syntax_memory_navigation"
            ),
            tuple(rows),
        )

    def with_bindings(
        self,
        bindings: Sequence[Mapping[str, Any]],
        *,
        status: str | None = None,
        reason: str | None = None,
        diagnostics: Sequence[Mapping[str, Any]] | None = None,
    ) -> "SourceTransferLedger":
        return replace(
            self,
            status=self.status if status is None else status,
            reason=self.reason if reason is None else reason,
            bindings=tuple(bindings),
            diagnostics=(
                self.diagnostics
                if diagnostics is None
                else tuple(diagnostics)
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_facts_digest": self.source_facts_digest,
            "status": self.status,
            "reason": self.reason,
            "occurrences": [item.to_json() for item in self.occurrences],
            "bindings": [dict(item) for item in self.bindings],
            "diagnostics": [dict(item) for item in self.diagnostics],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceTransferLedger":
        raw = _copy_mapping(value, "SourceTransferLedger")
        if set(raw) != {
            "source_facts_digest",
            "status",
            "reason",
            "occurrences",
            "bindings",
            "diagnostics",
        }:
            raise ValueError("SourceTransferLedger wire has an incomplete shape")
        if (
            not isinstance(raw["occurrences"], list)
            or not isinstance(raw["bindings"], list)
            or not isinstance(raw["diagnostics"], list)
        ):
            raise TypeError("SourceTransferLedger wire collections must be arrays")
        rows: list[TransferOccurrence] = []
        for item in raw["occurrences"]:
            if not isinstance(item, Mapping):
                raise TypeError("SourceTransferLedger occurrence must be an object")
            profile = item.get("address_profile")
            parsed_profile = None
            if isinstance(profile, Mapping):
                parsed_profile = (
                    profile.get("dynamic_terms"),
                    profile.get("constant_terms"),
                    tuple(profile.get("constant_offsets", ())),
                )
            rows.append(
                TransferOccurrence(
                    ordinal=item["ordinal"],
                    operation=item["operation"],
                    kind=item["kind"],
                    masked=item["masked"],
                    predicate=item.get("predicate"),
                    fill_value=item.get("fill_value"),
                    pointer_base=item.get("pointer_base"),
                    pointer_position=item.get("pointer_position"),
                    origin_positions=tuple(item.get("origin_positions", ())),
                    context=tuple(item.get("context", ())),
                    address_terms=item.get("address_terms"),
                    address_profile=parsed_profile,
                    address_type_profile=item.get("address_type_profile"),
                    logical_ref=item.get("logical_ref"),
                    tensor_rank=item.get("tensor_rank"),
                    element_type=item.get("element_type"),
                    semantic_consumer=(
                        SourceSemanticOccurrence.from_json(item["semantic_consumer"])
                        if isinstance(item.get("semantic_consumer"), Mapping)
                        else None
                    ),
                    semantic_predicate=(
                        SourceSemanticOccurrence.from_json(item["semantic_predicate"])
                        if isinstance(item.get("semantic_predicate"), Mapping)
                        else None
                    ),
                    semantic_access=(
                        SourceSemanticOccurrence.from_json(item["semantic_access"])
                        if isinstance(item.get("semantic_access"), Mapping)
                        else None
                    ),
                )
            )
        return cls(
            raw["source_facts_digest"],
            raw["status"],
            raw["reason"],
            tuple(rows),
            tuple(raw["bindings"]),
            tuple(raw["diagnostics"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_json())


def coerce_source_transfer_ledger(
    value: SourceTransferLedger | Mapping[str, Any] | None,
    *,
    source_facts: SourceFacts | None = None,
) -> SourceTransferLedger:
    """Normalize every ledger ingress through one wire conversion.

    Source-ledger consumers receive either the immutable dataclass or a
    transport projection.  Keeping this conversion here prevents the effect
    reorderer, fill checker, and pi binder from disagreeing about whether a
    nested ``source_transfer_ledger`` object is authoritative.  A malformed
    or absent projection is an unavailable ledger, never an empty one.
    """

    def validate_lineage(
        ledger: SourceTransferLedger,
    ) -> SourceTransferLedger:
        # A transport ledger is only usable for the Source facts that created
        # it.  Without this check a structurally valid {occurrences,
        # bindings} projection from another kernel could be consumed by the
        # effect, fill, or pi path and silently create cross-kernel ownership.
        if source_facts is not None:
            if ledger.source_facts_digest != source_facts.digest:
                return SourceTransferLedger.unavailable(
                    "source_transfer_ledger_source_facts_disagree",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_ledger_source_facts_disagree",
                            "path": "<source-transfer-ledger>.source_facts_digest",
                            "observed_refs_or_types": [
                                f"ledger={ledger.source_facts_digest!r}",
                                f"expected={source_facts.digest!r}",
                            ],
                        },
                    ),
                )
        return ledger

    if isinstance(value, SourceTransferLedger):
        return validate_lineage(value)
    if isinstance(value, Mapping):
        nested = value.get("source_transfer_ledger")
        # The reorderer exposes ``source_access_matches`` only as an
        # ingress-local projection.  It is never a ledger by itself.  Do not
        # let that envelope fall through to ``from_json``: because
        # ``occurrences`` and ``bindings`` have defaults there, the projection
        # could otherwise be silently accepted as an empty pass ledger.
        if "source_access_matches" in value and not isinstance(nested, Mapping):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        # ``reorder_boundary_effects_from_source_ledger`` may retain a
        # diagnostic match projection after discovering an order barrier.  Its
        # nested ledger is useful for explaining the failure, but it is not a
        # usable downstream authority while the envelope is ambiguous or
        # unavailable.  Do not let the nested ``pass`` status bypass the
        # outer transaction status.
        outer_status = value.get("status")
        if outer_status in {"ambiguous", "unavailable"}:
            return SourceTransferLedger.unavailable(
                f"source_transfer_ledger_projection_{outer_status}",
                source_facts=source_facts,
            )
        if outer_status is not None and outer_status != "pass":
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        raw = nested if isinstance(nested, Mapping) else value
        # A canonical transport must carry both sides of the occurrence
        # ledger.  Missing either side is an invalid wire shape, not a valid
        # empty ledger.  This keeps every downstream consumer on the same
        # occurrence/binding authority and preserves fail-closed behavior.
        if not all(key in raw for key in ("status", "occurrences", "bindings")):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        try:
            return validate_lineage(SourceTransferLedger.from_json(raw))
        except (TypeError, ValueError, KeyError):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
    return SourceTransferLedger.unavailable(
        "canonical_source_transfer_ledger_required",
        source_facts=source_facts,
    )


@dataclass(frozen=True)
class CanonicalHandleAllocator:
    """Single declaration-order allocator shared by ABI and J repair."""

    declarations: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # ``logical_bindings`` is the closed Source-to-J subset of the ``t*``
    # namespace.  It is intentionally separate from ``declarations`` because
    # a runtime scalar used only as a value is not a logical tensor until
    # Source semantics prove that use.  Shape-coordinate scalars, however,
    # are already proved by the Harness ABI and therefore belong here.
    # Rows are ``(p* ABI handle, t* handle, role)`` in ABI order.  Compiler
    # slots are intentionally excluded: a C-only carrier can never acquire a
    # J logical owner through this allocator.
    logical_bindings: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        rows = tuple(self.declarations)
        seen_names: set[str] = set()
        seen_prefixes: set[str] = set()
        normalized: list[tuple[str, tuple[str, ...]]] = []
        for prefix, values in rows:
            if prefix not in _HANDLE_PREFIXES:
                raise ValueError(f"unsupported canonical handle namespace {prefix!r}")
            if prefix in seen_prefixes:
                raise ValueError(f"canonical handle namespace {prefix!r} is repeated")
            seen_prefixes.add(prefix)
            ordered = tuple(values)
            if any(
                not isinstance(value, str) or re.fullmatch(rf"{prefix}[0-9]+", value) is None
                for value in ordered
            ) or len(ordered) != len(set(ordered)):
                raise ValueError(f"invalid {prefix}* declarations")
            if ordered != self.canonical_names(prefix, len(ordered)):
                raise ValueError(
                    f"{prefix}* declarations must be gap-free and declaration-ordered"
                )
            if seen_names & set(ordered):
                raise ValueError("canonical handle namespaces overlap")
            seen_names.update(ordered)
            normalized.append((prefix, ordered))
        object.__setattr__(self, "declarations", tuple(normalized))
        bindings = tuple(tuple(row) for row in self.logical_bindings)
        seen_abi: set[str] = set()
        seen_logical: set[str] = set()
        allowed_roles = {"tensor", "shape_scalar", "scalar"}
        declared_tensor_order = self.declared("t")
        declared_tensors = set(declared_tensor_order)
        for row in bindings:
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 3
                or not all(isinstance(item, str) for item in row)
            ):
                raise ValueError("invalid canonical logical binding row")
            abi_name, logical_name, role = row
            if (
                re.fullmatch(r"p[0-9]+", abi_name) is None
                or re.fullmatch(r"t[0-9]+", logical_name) is None
                or role not in allowed_roles
                or abi_name in seen_abi
                or logical_name in seen_logical
                or logical_name not in declared_tensors
            ):
                raise ValueError("canonical logical bindings are not unique or declared")
            seen_abi.add(abi_name)
            seen_logical.add(logical_name)
        if bindings and (
            set(seen_logical) != declared_tensors
            or tuple(row[1] for row in bindings) != declared_tensor_order
        ):
            raise ValueError(
                "canonical logical bindings must cover t* declarations in order"
            )
        object.__setattr__(self, "logical_bindings", bindings)

    @classmethod
    def from_source_facts(
        cls,
        facts: SourceFacts,
        *,
        logical_scalar_parameters: Sequence[str] = (),
    ) -> "CanonicalHandleAllocator":
        p_names = tuple(
            sorted(
                (
                    opaque for _source, opaque in facts.source_to_opaque
                    if opaque.startswith("p")
                ),
                key=lambda value: int(value[1:]),
            )
        )
        c_names = tuple(
            sorted(
                (
                    opaque for _source, opaque in facts.source_to_opaque
                    if opaque.startswith("c")
                ),
                key=lambda value: int(value[1:]),
            )
        )
        tensor_inputs = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") in {"tensor", "pointer"}
            and item.get("mode") in {"input", "inout"}
            and isinstance(item.get("name"), str)
        ]
        tensor_outputs = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") in {"tensor", "pointer"}
            and item.get("mode") == "output"
            and isinstance(item.get("name"), str)
            and item.get("name") not in tensor_inputs
        ]
        shape_scalar_parameters = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") == "scalar"
            and isinstance(item.get("shape_coordinate"), str)
            and isinstance(item.get("name"), str)
        ]
        logical_scalar_set = {
            name for name in logical_scalar_parameters
            if isinstance(name, str) and name.startswith("p")
        }
        # Source scalar bindings are an input-side logical declaration.  Keep
        # the declaration order in the physical Source ABI, then append
        # output-only tensors.  Shape scalars remain logical rank-0 values
        # even when they are not used by a data expression; a scalar promoted
        # by the Source binder uses the distinct ``scalar`` role.
        logical_inputs = [
            name for name in facts.parameter_order
            if name in set(tensor_inputs)
            or name in set(shape_scalar_parameters)
            or name in logical_scalar_set
        ]
        tensor_parameters = (*logical_inputs, *tensor_outputs)
        if len(tensor_parameters) != len(set(tensor_parameters)):
            raise ValueError("SourceFacts tensor ABI declarations are ambiguous")
        logical_bindings = tuple(
            (
                name,
                cls.canonical_names("t", len(tensor_parameters))[index],
                "shape_scalar"
                if name in shape_scalar_parameters and name not in logical_scalar_set
                else "scalar"
                if name in logical_scalar_set
                else "tensor",
            )
            for index, name in enumerate(tensor_parameters)
        )
        source_owned = {
            "p": p_names,
            "c": c_names,
            "t": cls.canonical_names("t", len(tensor_parameters)),
            "x": cls.canonical_names("x", len(facts.shape_coordinates)),
        }
        return cls(
            tuple(
                (prefix, source_owned.get(prefix, ()))
                for prefix in _CANONICAL_NAMESPACE_ORDER
            ),
            logical_bindings=logical_bindings,
        )

    @classmethod
    def from_declarations(
        cls,
        declarations: Mapping[str, Sequence[str]],
        *,
        logical_bindings: Sequence[Sequence[str]] = (),
    ) -> "CanonicalHandleAllocator":
        unknown = set(declarations) - set(_HANDLE_PREFIXES)
        if unknown:
            raise ValueError(
                "unsupported canonical handle namespace(s): "
                + ", ".join(sorted(unknown))
            )
        ordered_declarations = tuple(
            (prefix, tuple(declarations.get(prefix, ())))
            for prefix in _CANONICAL_NAMESPACE_ORDER
            if prefix in declarations
        )
        return cls(
            ordered_declarations,
            logical_bindings=tuple(tuple(row) for row in logical_bindings),
        )

    def logical_binding_map(self) -> Mapping[str, tuple[str, str]]:
        """Return the closed ABI-to-logical bindings without re-inference."""

        return {
            abi_name: (logical_name, role)
            for abi_name, logical_name, role in self.logical_bindings
        }

    @classmethod
    def abi_mapping(
        cls,
        runtime_parameters: Sequence[str],
        compiler_slots: Sequence[str],
    ) -> Mapping[str, str]:
        """Allocate one declaration-order p*/c* ABI map.

        Source adapters decide which source parameters are runtime carriers and
        which are compiler-only slots.  The naming and collision rules live
        here, so Dataset, Triton and native paths cannot accidentally create
        different opaque orders for the same declaration order.
        """

        runtime = tuple(runtime_parameters)
        slots = tuple(compiler_slots)
        if any(not isinstance(name, str) or not name.isidentifier() for name in (*runtime, *slots)):
            raise ValueError("ABI names must be identifiers")
        if len(runtime) != len(set(runtime)) or len(slots) != len(set(slots)):
            raise ValueError("ABI names must be unique within their declaration class")
        if set(runtime) & set(slots):
            raise ValueError("runtime ABI parameters and compiler slots overlap")
        p_names = cls.allocate("p", len(runtime))
        c_names = cls.allocate("c", len(slots), used=p_names)
        return {
            **dict(zip(runtime, p_names, strict=True)),
            **dict(zip(slots, c_names, strict=True)),
        }

    def declared(self, prefix: str) -> tuple[str, ...]:
        for current, values in self.declarations:
            if current == prefix:
                return values
        return ()

    def owns_namespace(self, prefix: str) -> bool:
        """Return whether this allocator owns ``prefix``, including empty rows."""

        return any(current == prefix for current, _values in self.declarations)

    @staticmethod
    def canonical_names(prefix: str, count: int) -> tuple[str, ...]:
        if prefix not in _HANDLE_PREFIXES or count < 0:
            raise ValueError("invalid canonical handle sequence")
        return tuple(f"{prefix}{index}" for index in range(count))

    def rename_map(self, prefix: str) -> Mapping[str, str]:
        return {
            old: f"{prefix}{index}"
            for index, old in enumerate(self.declared(prefix))
        }

    def all_rename_maps(self) -> Mapping[str, Mapping[str, str]]:
        return {prefix: self.rename_map(prefix) for prefix, _values in self.declarations}

    def with_declarations(
        self, declarations: Mapping[str, Sequence[str]]
    ) -> "CanonicalHandleAllocator":
        merged = {prefix: values for prefix, values in self.declarations}
        for prefix, values in declarations.items():
            normalized = tuple(values)
            existing = merged.get(prefix)
            # Source-owned rows (p*/c*/t*/x*) are never replaced.  Candidate
            # namespaces are present as empty rows in the Source allocator so
            # the declaration-table order stays stable; their first complete
            # candidate declaration is the one allowed extension of that same
            # allocator.  A second non-empty declaration must agree exactly.
            if existing is not None and existing and existing != normalized:
                raise ValueError(
                    f"canonical {prefix}* declarations differ from owning allocator"
                )
            merged[prefix] = normalized
        return type(self).from_declarations(
            merged,
            logical_bindings=self.logical_bindings,
        )

    def for_candidate_j(
        self,
        j: Mapping[str, Any],
    ) -> "CanonicalHandleAllocator":
        """Extend this Source-owned allocator with one candidate J declaration table.

        SourceFacts remains authoritative for ``p*/c*/t*/x*``.  The candidate
        contributes only the declaration counts/order of its semantic
        namespaces; those rows are immediately converted to dense canonical
        names in this same allocator instance.  If the candidate is too
        incomplete to expose a structural table, return the Source-owned
        allocator unchanged so strict CandidateCheck remains the authority.
        A conflicting Source-owned count is likewise left for fail-closed
        validation instead of being guessed here.
        """

        declarations = canonical_authored_handle_declarations(j)
        if declarations is None:
            return self
        dynamic: dict[str, tuple[str, ...]] = {}
        for prefix, values in declarations.items():
            if prefix in CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES:
                continue
            dynamic[prefix] = self.canonical_names(prefix, len(values))
        if not dynamic:
            return self
        try:
            return self.with_declarations(dynamic)
        except ValueError:
            return self

    @staticmethod
    def allocate(prefix: str, count: int, *, used: Sequence[str] = ()) -> tuple[str, ...]:
        if prefix not in _HANDLE_PREFIXES or count < 0:
            raise ValueError("invalid canonical allocation request")
        used_set = set(used)
        result: list[str] = []
        index = 0
        while len(result) < count:
            value = f"{prefix}{index}"
            index += 1
            if value in used_set:
                continue
            result.append(value)
            used_set.add(value)
        return tuple(result)


@dataclass(frozen=True)
class CanonicalConstruction:
    """The one host-side input bundle for canonical J/C/pi construction."""

    source_facts: SourceFacts
    transfer_ledger: SourceTransferLedger
    allocator: CanonicalHandleAllocator
    logical_scalar_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_facts, SourceFacts):
            raise TypeError("CanonicalConstruction requires SourceFacts")
        if not isinstance(self.transfer_ledger, SourceTransferLedger):
            raise TypeError("CanonicalConstruction requires SourceTransferLedger")
        if not isinstance(self.allocator, CanonicalHandleAllocator):
            raise TypeError("CanonicalConstruction requires CanonicalHandleAllocator")
        ledger_digest = self.transfer_ledger.source_facts_digest
        if ledger_digest is not None and ledger_digest != self.source_facts.digest:
            raise ValueError(
                "CanonicalConstruction transfer ledger differs from SourceFacts"
            )
        logical_scalars = tuple(self.logical_scalar_parameters)
        if len(logical_scalars) != len(set(logical_scalars)) or any(
            not isinstance(name, str) or not name.startswith("p")
            for name in logical_scalars
        ):
            raise ValueError("CanonicalConstruction logical scalar declarations are invalid")
        expected = CanonicalHandleAllocator.from_source_facts(
            self.source_facts,
            logical_scalar_parameters=logical_scalars,
        )
        if (
            self.allocator.declarations != expected.declarations
            or self.allocator.logical_bindings != expected.logical_bindings
        ):
            raise ValueError(
                "CanonicalConstruction allocator differs from SourceFacts"
            )
        source_parameter_order = tuple(
            name
            for name in self.source_facts.parameter_order
            if name.startswith("p")
        )
        if source_parameter_order != expected.declared("p"):
            raise ValueError(
                "SourceFacts parameter order differs from canonical p* allocator"
            )
        source_c_parameter_order = tuple(
            name
            for name in self.source_facts.parameter_order
            if name.startswith("c")
        )
        canonical_c_order = (
            *source_c_parameter_order,
            *self.source_facts.compiler_slots,
        )
        if canonical_c_order != expected.declared("c"):
            raise ValueError(
                "SourceFacts source-C/compiler-slot order differs from canonical c* allocator"
            )

    @classmethod
    def from_compiler_view(
        cls,
        view: Any,
        *,
        frontend: str,
        source_kind: str,
        source_symbol: str,
        source_to_opaque: Mapping[str, str] | None = None,
        source_digest: str | None = None,
        source_slice_digest: str | None = None,
    ) -> "CanonicalConstruction":
        facts = SourceFacts.from_compiler_view(
            view,
            frontend=frontend,
            source_kind=source_kind,
            source_symbol=source_symbol,
            source_to_opaque=source_to_opaque,
            source_digest=source_digest,
            source_slice_digest=source_slice_digest,
        )
        return cls(
            source_facts=facts,
            transfer_ledger=SourceTransferLedger.unavailable(
                "ttir_observation_not_run",
                source_facts=facts,
            ),
            allocator=CanonicalHandleAllocator.from_source_facts(facts),
        )

    @classmethod
    def from_source_facts(
        cls,
        facts: SourceFacts,
        *,
        source_text: str | None = None,
    ) -> "CanonicalConstruction":
        if not isinstance(facts, SourceFacts):
            raise TypeError("CanonicalConstruction requires SourceFacts")
        ledger = (
            SourceTransferLedger.from_source_text(
                source_text,
                source_kind=facts.source_kind,
                source_facts=facts,
            )
            if source_text is not None
            else SourceTransferLedger.unavailable(
                "source_text_not_mounted", source_facts=facts
            )
        )
        logical_scalars = source_logical_scalar_parameters(facts, source_text)
        return cls(
            source_facts=facts,
            transfer_ledger=ledger,
            allocator=CanonicalHandleAllocator.from_source_facts(
                facts,
                logical_scalar_parameters=logical_scalars,
            ),
            logical_scalar_parameters=logical_scalars,
        )

    def with_observation(self, observed: Any) -> "CanonicalConstruction":
        return replace(
            self,
            transfer_ledger=SourceTransferLedger.from_observation(
                observed,
                source_facts=self.source_facts,
                logical_bindings=self.allocator.logical_binding_map(),
                lexical_ledger=self.transfer_ledger,
            ),
        )

    def _json_payload(self) -> dict[str, Any]:
        """Return the construction payload before its identity receipt.

        The mounted MCP bundle and the host request must refer to one exact
        SourceFacts/ledger/allocator snapshot.  Keep the digest outside the
        hashed payload so serializing the receipt cannot recurse into itself.
        """

        return {
            "source_facts": self.source_facts.to_json(),
            "source_facts_digest": self.source_facts.digest,
            "transfer_ledger": self.transfer_ledger.to_json(),
            "transfer_ledger_digest": self.transfer_ledger.digest,
            "logical_scalar_parameters": list(self.logical_scalar_parameters),
            "allocator": {
                prefix: list(values)
                for prefix, values in self.allocator.declarations
            }
            | {
                "logical_bindings": [list(row) for row in self.allocator.logical_bindings]
            },
        }

    def to_json(self) -> dict[str, Any]:
        payload = self._json_payload()
        payload["authority_digest"] = _digest(payload)
        return payload

    @property
    def authority_digest(self) -> str:
        return _digest(self._json_payload())


__all__ = [
    "CANONICAL_ARTIFACT_ROOT_FIELDS",
    "canonical_artifact_envelope_error",
    "require_canonical_artifact_envelope",
    "CANONICAL_C_WIRE_FIELDS",
    "CANONICAL_PI_REQUIRED_FIELDS",
    "CANONICAL_PI_WIRE_FIELDS",
    "CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS",
    "CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS",
    "CANONICAL_BOUNDARY_WIRE_FIELDS",
    "CANONICAL_CONSTRUCTION_RECORD_FIELDS",
    "CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS",
    "CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES",
    "CANONICAL_HANDLE_NAMESPACE_ORDER",
    "CANONICAL_HANDLE_NAMESPACE_PATTERNS",
    "CANONICAL_J_WIRE_FIELDS",
    "CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES",
    "CanonicalConstruction",
    "CanonicalConstructionRecords",
    "CanonicalHandleAllocator",
    "coerce_source_transfer_ledger",
    "CANONICAL_AUTHORED_HANDLE_PREFIXES",
    "SourceFacts",
    "SourceSemanticOccurrence",
    "SourceTransferLedger",
    "TransferOccurrence",
    "source_logical_scalar_parameters",
    "canonical_archive_records",
    "canonical_authored_handle_declarations",
    "canonical_boundary_records",
    "canonical_operation_objects",
    "canonical_operation_records",
    "canonical_operation_wire_rows",
    "semantic_operation_wire_rows",
    "canonical_output_records",
    "canonical_value_type_records",
    "canonical_scope_records",
]
