"""Responses schema for the LLM-direct ``J + pure C + pi`` Lift result.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract given to the model so closed Gate1 objects do not drift into
invented field names.  The model, rather than a construction engine, produces
the complete realization witness.  The project parser performs final
cross-field and opaque-reference validation.
"""

from __future__ import annotations

import copy
from typing import Any

from intentir.ir.intent_op_contract import intent_op_json_schema
from intentir.verify.gate1.model import (
    BOUNDARY_ACCESS_REQUIRED_FIELDS,
    BOUNDARY_WIRE_FIELDS,
    GATE1_ARTIFACT_MODEL_FIELDS,
    SEMANTIC_ARCHIVE_WIRE_FIELDS,
    render_boundary_access_wire_grammar,
    render_gate1_artifact_envelope_grammar,
)
from intentir.verify.gate1.pi import (
    REALIZATION_MECHANISMS,
    REPRESENTATION_REFINEMENTS,
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


def _closed_object(
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
    *,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if description is not None:
        schema["description"] = description
    return schema


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_STRING = {"type": "string", "minLength": 1}
_TENSOR_HANDLE = {"type": "string", "pattern": "^t[0-9]+$"}
_NODE_HANDLE = {"type": "string", "pattern": "^n[0-9]+$"}
_C_HANDLE = {"type": "string", "pattern": "^c[0-9]+$"}
_PARAMETER_HANDLE = {"type": "string", "pattern": "^p[0-9]+$"}
_PORTABLE_PREDICATE_STRING = {
    "type": "string",
    "minLength": 1,
    "description": (
        "One boolean ContractExpr v1 expression: symbols/literals, arithmetic, "
        "comparisons, and/or/not, and only ceil_div/cdiv/min/max calls. No "
        "assignment, prose, range notation, frontend calls, attributes, or "
        "program/grid/tile/lane coordinate construction."
    ),
}
_STRING_ARRAY = _array(_STRING)
_INT_OR_STRING = {"anyOf": [{"type": "integer"}, _STRING]}


def render_semantic_archive_wire_grammar() -> str:
    """Render global archive invariants already enforced by the strict parser."""

    return (
        "J scopes are an exact derived dominance archive, not a copy of pi "
        "regions. Straight-line J uses one parentless function scope. Its "
        "definitions are exactly J input handles plus results actually produced "
        "by operations in that scope; a declared logical output is not a "
        "definition unless an operation really produces that same handle. "
        "captures and effects are likewise exact, with no convenience extras. "
        "Every semantic loop has exactly one corresponding loop_region scope. "
        "The storage array may be empty: ordinary SSA temporaries do not have "
        "semantic storage or lifetime objects. Add StorageObject only when J "
        "semantics requires storage identity, such as an alias/destination or "
        "loop-carried state. write_once requires exactly one matching boundary "
        "write event; producing an SSA value is not such an event. Every loop "
        "state requires one storage object binding its state, body, next and "
        "result values plus a non-literal init value, if any, across iterations."
    )


def _tensor_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "name": _TENSOR_HANDLE,
            "dtype": _STRING,
            "shape": _array(_INT_OR_STRING),
        },
        ("name", "dtype", "shape"),
    )


def _intent_schema() -> dict[str, Any]:
    logical_return = _closed_object(
        {"output": _STRING, "value": _STRING},
        ("output", "value"),
        description=(
            "Layer-A value-to-result binding. Destination aliasing and partial "
            "write behavior belong only to the canonical J boundary."
        ),
    )
    function = _closed_object(
        {
            "name": {"type": "string", "enum": ["scrc_j0"]},
            "inputs": _array(_tensor_schema()),
            "outputs": _array(_tensor_schema()),
            "returns": _array(logical_return),
            "body": _array({"$ref": "#/$defs/intent_op"}),
        },
        ("name", "inputs", "outputs", "body"),
    )
    return _closed_object(
        {"intent_function": function}, ("intent_function",)
    )


def _boundary_schema() -> dict[str, Any]:
    access = _closed_object(
        {
            "access_id": _STRING,
            "kind": {
                "type": "string",
                "enum": ["load", "store", "update"],
            },
            "tensor": _STRING,
            "predicate": _nullable(_PORTABLE_PREDICATE_STRING),
            "fill_value": {},
            "node_ref": _nullable(_STRING),
            "index_expression": _STRING,
        },
        BOUNDARY_ACCESS_REQUIRED_FIELDS,
    )
    broadcast = _closed_object(
        {
            "node_ref": _STRING,
            "operand": _STRING,
            "axes": _array(_INT_OR_STRING),
        },
        ("node_ref", "operand", "axes"),
    )
    alias = _closed_object(
        {
            "left": _STRING,
            "right": _STRING,
            "kind": {
                "type": "string",
                "enum": ["must_alias", "may_alias", "no_alias"],
            },
        },
        ("left", "right", "kind"),
    )
    effect = _closed_object(
        {
            "effect_id": _STRING,
            "kind": {
                "type": "string",
                "enum": [
                    "read",
                    "write",
                    "read_write",
                    "update",
                ],
            },
            "target": _STRING,
            "node_ref": _nullable(_STRING),
            "access_ref": _nullable(_STRING),
        },
        ("effect_id", "kind", "target", "access_ref"),
    )
    output_mapping = _closed_object(
        {
            "output": _STRING,
            "value": _STRING,
            "node_ref": _nullable(_STRING),
        },
        ("output", "value"),
    )
    reduction = _closed_object(
        {
            "node_ref": _STRING,
            "tile_axis": _INT_OR_STRING,
            "identity": {},
            "masked_contribution": {},
        },
        ("node_ref", "tile_axis"),
    )
    index_bounds = _closed_object(
        {
            "node_ref": _STRING,
            "data": _STRING,
            "indices": _STRING,
            "axis": {"type": "integer", "minimum": 0},
        },
        ("node_ref", "data", "indices", "axis"),
        description=(
            "Typed precondition 0 <= indices < dim(data, axis), exactly bound "
            "to one gather/scatter node. Never use a proof string or proven flag."
        ),
    )
    update = _closed_object(
        {
            "node_ref": _STRING,
            "base": _STRING,
            "indices": _STRING,
            "values": _STRING,
            "axis": {"type": "integer", "minimum": 0},
            "combiner": {
                "type": "string", "enum": ["overwrite", "add"],
            },
            "duplicate_policy": {
                "type": "string", "enum": ["forbid", "combine", "one_of"],
            },
            "order": {
                "type": "string", "enum": ["input_order", "unordered"],
            },
        },
        (
            "node_ref", "base", "indices", "values", "axis", "combiner",
            "duplicate_policy", "order",
        ),
        description=(
            "Logical update only. Atomic/CAS/thread ownership are realization "
            "choices and must not appear here."
        ),
    )
    result_relation = _closed_object(
        {
            "output": _STRING,
            "kind": {
                "type": "string",
                "enum": ["exact", "tolerance", "one_of_updates"],
            },
            "update_node_ref": _STRING,
        },
        ("output", "kind"),
        description=(
            "Comparison class owned by J. Numeric tolerance values are forbidden "
            "and come only from external Gate2 policy Q. one_of_updates requires "
            "one non-empty update_node_ref. exact/tolerance must omit the "
            "update_node_ref field entirely; empty string or null is invalid."
        ),
    )
    authorization = _closed_object(
        {
            "auth_id": {"type": "string", "pattern": "^u[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": [
                    "signed_widen",
                    "unsigned_widen",
                    "numeric_cast",
                    "accumulation_dtype",
                    "reassociation",
                    "native_atomic",
                    "privatized_update",
                ],
            },
            "subject_ref": _STRING,
            "source_dtype": _STRING,
            "target_dtype": _STRING,
        },
        ("auth_id", "kind", "subject_ref"),
        description=(
            "J/B-owned authorization for one sensitive representation choice. "
            "Pi may cite auth_id but may not redefine its semantic contract."
        ),
    )
    return _closed_object(
        {
            "domains": {
                "type": "object",
                "propertyNames": {"pattern": "^d[0-9]+$"},
                "additionalProperties": _PORTABLE_PREDICATE_STRING,
                "description": (
                    "Map each d* handle directly to one ContractExpr string. "
                    "Never emit an axes/members/domain object here."
                ),
            },
            "accesses": _array(access),
            "broadcasts": _array(broadcast),
            "aliases": _array(alias),
            "effects": _array(effect),
            "output_mappings": _array(output_mapping),
            "reductions": _array(reduction),
            "index_bounds": _array(index_bounds),
            "updates": _array(update),
            "result_relations": _array(result_relation),
            "realization_authorizations": _array(authorization),
        },
        BOUNDARY_WIRE_FIELDS,
        description=render_boundary_access_wire_grammar(),
    )


def _signature_schema() -> dict[str, Any]:
    parameter = _closed_object(
        {
            "name": _PARAMETER_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["tensor", "pointer", "scalar", "opaque", "constexpr"],
            },
            "mode": {
                "type": "string",
                "enum": ["input", "output", "inout", "value"],
            },
            "dtype": _nullable(_STRING),
            "rank": _nullable({"type": "integer", "minimum": 0}),
            "alias_of": _nullable(_STRING),
            "logical_ref": _nullable(_STRING),
        },
        ("name", "kind", "mode"),
        description=(
            "One J-owned logical callable handle. logical_ref links a "
            "pointer/tensor/scalar/semantic-constexpr handle to a J tensor, "
            "value, or dimension. Every trusted "
            "compiler-view tensor argument, including one unused by the frozen "
            "specialization, keeps a logical_ref whose J tensor dtype, rank and "
            "shape exactly match that view. The inactive tensor remains a typed "
            "ABI declaration but has no invented operation, access or effect. "
            "Implementation-selection constexprs are C slots and never appear "
            "in this array. A semantic constexpr has a J logical_ref; its concrete "
            "value stays in Harness. An opaque parameter is "
            "representation-only frontend plumbing such as cute.CopyAtom; it "
            "has dtype but no logical_ref and contributes no J semantics."
        ),
    )
    return _closed_object(
        {"parameters": _array(parameter)},
        ("parameters",),
    )


def _c_schema() -> dict[str, Any]:
    slot = _closed_object(
        {
            "name": _C_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["constexpr", "compiler"],
            },
            "role": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "value_type": _STRING,
            "required": {"type": "boolean"},
        },
        ("name", "kind", "role"),
    )
    return _closed_object(
        {
            "slots": _array(slot),
            "values": {
                "type": "object",
                "propertyNames": {"pattern": "^c[0-9]+$"},
                "additionalProperties": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "string"},
                    ]
                },
            },
        },
        ("slots", "values"),
    )


def _pi_schema() -> dict[str, Any]:
    opaque_ref = {
        "type": "string",
        "pattern": "^(t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+$",
    }
    region = _closed_object(
        {
            "id": {"type": "string", "pattern": "^r[0-9]+$"},
            "parent": _nullable({"type": "string", "pattern": "^r[0-9]+$"}),
            "placement": {
                "type": "string", "enum": ["root", "program", "loop", "branch"],
            },
        },
        ("id", "parent", "placement"),
    )
    instance = _closed_object(
        {
            "id": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Declaration-order pi graph-node ID. This is the only ID "
                    "namespace allowed at route/order/sharing endpoints."
                ),
            },
            "semantic_ref": {
                **opaque_ref,
                "description": (
                    "Anchor into a declared anonymous J/C handle. "
                    "It is not a pi graph-node ID and never appears directly "
                    "at route/order/sharing endpoints."
                ),
            },
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "mechanism": {
                "type": "string",
                "enum": list(REALIZATION_MECHANISMS),
                "description": render_realization_mechanism_grammar(),
            },
        },
        ("id", "semantic_ref", "region", "mechanism"),
    )
    route = _closed_object(
        {
            "source": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A previously declared pi instance ID, never a raw anchor handle.",
            },
            "target": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A previously declared pi instance ID, never a raw anchor handle.",
            },
            "slots": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "uniqueItems": True,
            },
        },
        ("source", "target", "slots"),
    )
    order = _closed_object(
        {
            "before": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A declared pi instance ID.",
            },
            "after": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A declared pi instance ID.",
            },
        },
        ("before", "after"),
    )
    sharing = _closed_object(
        {
            "source": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "The declared pi instance reused by all consumers.",
            },
            "consumers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^i[0-9]+$",
                    "description": "A declared pi consumer instance ID.",
                },
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        ("source", "consumers"),
    )
    refinement = _closed_object(
        {
            "instance": {"type": "string", "pattern": "^i[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": list(REPRESENTATION_REFINEMENTS),
            },
            "auth_ref": _nullable(
                {"type": "string", "pattern": "^u[0-9]+$"}
            ),
        },
        ("instance", "kind", "auth_ref"),
        description=(
            "A representation refinement. Sensitive kinds cite one J/B-owned "
            "authorization; identity materialization uses the verifier's closed "
            "global-safe rule."
        ),
    )
    return _closed_object(
        {
            "regions": _array(region),
            "instances": _array(instance),
            "routes": _array(route),
            "order": _array(order),
            "sharing": _array(sharing),
            "refinements": _array(refinement),
        },
        ("regions", "instances", "routes", "order", "sharing", "refinements"),
        description=(
            render_realization_witness_wire_grammar()
            + " "
            + render_realization_mechanism_grammar()
            + " Preserve route/order/share/region multiplicity. No operator, "
            "value, predicate, dtype, frontend API, rule identity or source text."
        ),
    )
def _portable_scope_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "scope_id": _STRING,
            "kind": {
                "type": "string",
                "enum": [
                    "function", "program_instance", "loop_region",
                    "branch_region",
                ],
            },
            "parent_scope": _nullable(_STRING),
            "captures": _STRING_ARRAY,
            "definitions": _STRING_ARRAY,
            "effects": _STRING_ARRAY,
        },
        (
            "scope_id", "kind", "parent_scope", "captures", "definitions",
            "effects",
        ),
        description=(
            "Exact portable dominance/capture/effect scope derived from J. "
            "definitions/captures/effects must contain neither missing nor extra "
            "handles. Contains no source AST, helper name or frontend spelling."
        ),
    )


def _storage_schema() -> dict[str, Any]:
    event = _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": [
                    "scope_entry", "scope_exit", "node", "access", "state",
                    "effect",
                ],
            },
            "ref": _STRING,
        },
        ("kind", "ref"),
    )
    return _closed_object(
        {
            "storage_id": _STRING,
            "members": _STRING_ARRAY,
            "scope_id": _STRING,
            "mutability": {
                "type": "string",
                "enum": ["immutable", "write_once", "mutable"],
            },
            "init_event": event,
            "last_use_event": event,
            "escapes": {"type": "boolean"},
            "crosses_iteration": {"type": "boolean"},
        },
        (
            "storage_id", "members", "scope_id", "mutability", "init_event",
            "last_use_event", "escapes", "crosses_iteration",
        ),
        description=(
            "Optional semantic storage identity/lifetime only; ordinary SSA "
            "temporaries require no entry. Physical allocation, register/shared/"
            "global placement and double buffering belong to C."
        ),
    )


def _location_map_schema() -> dict[str, Any]:
    span = _closed_object(
        {
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "start_column": _nullable({"type": "integer", "minimum": 0}),
            "end_column": _nullable({"type": "integer", "minimum": 0}),
        },
        ("start_line", "end_line"),
    )
    entry = _closed_object(
        {
            "span": span,
            "node_refs": _STRING_ARRAY,
            "access_refs": _STRING_ARRAY,
        },
        ("span", "node_refs", "access_refs"),
    )
    return _closed_object(
        {
            "source_name": {
                "type": "string", "enum": ["anonymous_source"]
            },
            "entries": _array(entry),
        },
        ("source_name", "entries"),
    )


def scrc_lift_core_json_schema() -> dict[str, Any]:
    """Return the complete source-aware LLM response schema."""

    semantic_archive = _closed_object(
        {
            "kernel_id": {"type": "string", "enum": ["scrc_k0"]},
            "intent": _intent_schema(),
            "node_ids": {
                "type": "object",
                "propertyNames": {"pattern": "^n[0-9]+$"},
                "additionalProperties": {
                    "type": "string",
                    "pattern": "^(intent_function\\.)?body\\[[0-9]+\\](\\.body\\[[0-9]+\\])*$",
                },
                "description": (
                    "Map n0,n1,... to structural body paths such as body[0], "
                    "not to SSA/result handles such as v0."
                ),
            },
            "boundary": _boundary_schema(),
            "signature": _signature_schema(),
            "scopes": {
                "type": "array",
                "minItems": 1,
                "items": _portable_scope_schema(),
            },
            "storage": _array(_storage_schema()),
        },
        SEMANTIC_ARCHIVE_WIRE_FIELDS,
        description=render_semantic_archive_wire_grammar(),
    )
    schema = _closed_object(
        {
            "schema": {
                "type": "string",
                "enum": ["intentir.scrc.artifact.v1"],
            },
            "j": semantic_archive,
            "c": _c_schema(),
            "pi": _pi_schema(),
        },
        GATE1_ARTIFACT_MODEL_FIELDS,
        description=render_gate1_artifact_envelope_grammar(),
    )
    schema["$defs"] = {"intent_op": intent_op_json_schema()}
    return copy.deepcopy(schema)


def scrc_lift_core_text_format() -> dict[str, Any]:
    """Return the Responses format for Lift's source-aware LLM substage."""

    return {
        "type": "json_schema",
        "name": "scrc_lift_core",
        # Operation-specific semantic attrs remain open during the IntentOp
        # robustness audit. B is the one closed top-level boundary; C and pi
        # are separate closed objects.
        "strict": False,
        "schema": scrc_lift_core_json_schema(),
    }


__all__ = [
    "render_boundary_access_wire_grammar",
    "render_gate1_artifact_envelope_grammar",
    "render_semantic_archive_wire_grammar",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
]
