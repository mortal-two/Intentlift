"""Responses JSON Schema for the source-visible SCRC lift artifact.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract given to the model so closed Gate1 objects do not drift into
invented field names.  IntentIR operation payloads retain extensible
``attrs``/``layer_b`` vocabularies, while semantic loop state is closed because
it crosses region boundaries.  ``J.intent.layer_c`` is forced empty because
SCRC stores concrete C only outside J.  The Responses format is non-strict and
the project parser performs the final cross-field validation.
"""

from __future__ import annotations

import copy
from typing import Any


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
_REDUCTION_ACCUMULATOR_DTYPE = {
    "type": "string",
    "enum": [
        "f16", "bf16", "f32", "f64", "i1", "i8", "i16", "i32", "i64",
    ],
}
_LOOP_INIT_VALUE = {
    "anyOf": [
        _STRING,
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
    ]
}


def _tensor_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "name": _STRING,
            "dtype": _STRING,
            "shape": _array(_INT_OR_STRING),
        },
        ("name", "dtype", "shape"),
    )


def _intent_op_schema() -> dict[str, Any]:
    loop_value_type = _closed_object(
        {
            "dtype": _STRING,
            "shape": _array(_INT_OR_STRING),
        },
        ("dtype", "shape"),
    )
    state_binding = _closed_object(
        {
            "state_id": _STRING,
            "init_value": _LOOP_INIT_VALUE,
            "body_argument": _STRING,
            "next_value": _STRING,
            "result_value": _STRING,
            "type": loop_value_type,
        },
        (
            "state_id", "init_value", "body_argument", "next_value",
            "result_value", "type",
        ),
        description=(
            "One explicit semantic loop binding. result/yield arrays must "
            "repeat result_value/next_value in exactly the same order."
        ),
    )
    trip = _closed_object(
        {"axis": _STRING},
        ("axis",),
        description=(
            "Semantic trip dimension only. Tile/block choices belong to "
            "C.schedule and must not be copied into J."
        ),
    )
    scan_input = _closed_object(
        {
            "name": _STRING,
            "from": _STRING,
            "axis": {"type": "integer", "minimum": 0},
        },
        ("name", "from", "axis"),
    )
    return _closed_object(
        {
            "op": _STRING,
            "result": {
                "anyOf": [_STRING, _STRING_ARRAY, {"type": "null"}]
            },
            "operands": _STRING_ARRAY,
            "attrs": {"type": "object"},
            "type": _nullable({"type": "object"}),
            "layer_b": {"type": "object"},
            "out": _nullable(_STRING),
            "scope_id": _nullable(_STRING),
            "carry": _array(state_binding),
            "trip": _nullable(trip),
            "scan": _array(scan_input),
            "body": _array({"$ref": "#/$defs/intent_op"}),
            "yield": _STRING_ARRAY,
        },
        ("op",),
    )


def _intent_schema() -> dict[str, Any]:
    function = _closed_object(
        {
            "name": _STRING,
            "inputs": _array(_tensor_schema()),
            "outputs": _array(_tensor_schema()),
            "layer_c": {"type": "object", "maxProperties": 0},
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
                "enum": ["load", "store", "atomic", "synchronize"],
            },
            "tensor": _STRING,
            "predicate": _nullable(_PORTABLE_PREDICATE_STRING),
            "fill_value": {},
            "node_ref": _nullable(_STRING),
            "index_expression": _nullable(_STRING),
        },
        ("access_id", "kind", "tensor", "index_expression"),
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
                    "atomic",
                    "synchronize",
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
    return _closed_object(
        {
            "domains": {
                "type": "object",
                "additionalProperties": _PORTABLE_PREDICATE_STRING,
            },
            "accesses": _array(access),
            "broadcasts": _array(broadcast),
            "aliases": _array(alias),
            "effects": _array(effect),
            "output_mappings": _array(output_mapping),
            "reductions": _array(reduction),
        },
        (
            "domains",
            "accesses",
            "broadcasts",
            "aliases",
            "effects",
            "output_mappings",
            "reductions",
        ),
    )


def _interface_schema() -> dict[str, Any]:
    parameter = _closed_object(
        {
            "name": _STRING,
            "kind": {
                "type": "string",
                "enum": ["tensor", "pointer", "scalar", "constexpr"],
            },
            "mode": {
                "type": "string",
                "enum": ["input", "output", "inout", "value"],
            },
            "dtype": _nullable(_STRING),
            "rank": _nullable({"type": "integer", "minimum": 0}),
            "alias_of": _nullable(_STRING),
            "tensor_ref": _nullable(_STRING),
        },
        ("name", "kind", "mode"),
        description=(
            "A callable parameter. tensor_ref is the only field that links a "
            "pointer/tensor parameter or a computation-valued scalar parameter "
            "to a J tensor; semantic scalars use shape=[]. Every trusted "
            "compiler-view tensor argument, including one unused by the frozen "
            "specialization, keeps a tensor_ref whose J tensor dtype, rank and "
            "shape exactly match that view. The inactive tensor remains a typed "
            "ABI declaration but has no invented operation, access or effect. "
            "Do not invent dimension reference fields."
        ),
    )
    return _closed_object(
        {
            "kernel_id": _STRING,
            "frontend": {
                "type": "string",
                "enum": ["triton", "tilelang", "cuda", "other"],
            },
            "function_name": _STRING,
            "parameters": _array(parameter),
            "grid": _STRING_ARRAY,
            "compiler_slots": _STRING_ARRAY,
            "do_not_specialize": _STRING_ARRAY,
        },
        (
            "kernel_id",
            "frontend",
            "function_name",
            "parameters",
            "grid",
            "compiler_slots",
        ),
    )


def _schedule_schema() -> dict[str, Any]:
    slot = _closed_object(
        {
            "name": _STRING,
            "kind": {
                "type": "string",
                "enum": ["constexpr", "compiler"],
            },
            "value_type": _STRING,
            "required": {"type": "boolean"},
        },
        ("name", "kind"),
    )
    return _closed_object(
        {
            "slots": _array(slot),
            "values": {"type": "object"},
        },
        ("slots", "values"),
    )


def _strategy_schema() -> dict[str, Any]:
    simple_bindings = [
        ("program_coordinate_width", "wide"),
        ("address_association", "pointer_staged"),
        ("mask_materialization", "staged_reuse"),
        ("reduction_call_shape", "flattened"),
        (
            "reduction_preheader_staging",
            "stage_invariant_access_scaffolding",
        ),
        ("reduction_tile_fusion", "shared_producer_outputs"),
    ]
    variants = [
        _closed_object(
            {
                "target_ref": _STRING,
                "family": {"type": "string", "enum": [family]},
                "choice": {"type": "string", "enum": [choice]},
            },
            ("target_ref", "family", "choice"),
        )
        for family, choice in simple_bindings
    ]
    variants.append(
        _closed_object(
            {
                "target_ref": _STRING,
                "family": {
                    "type": "string",
                    "enum": ["reduction_tiling"],
                },
                "choice": {
                    "type": "string",
                    "enum": ["multi_tile_accumulate"],
                },
                "tile_slot": _STRING,
                "accumulator_dtype": _REDUCTION_ACCUMULATOR_DTYPE,
            },
            (
                "target_ref",
                "family",
                "choice",
                "tile_slot",
                "accumulator_dtype",
            ),
        )
    )
    binding = {
        "oneOf": variants
    }
    return _closed_object(
        {"bindings": _array(binding)},
        ("bindings",),
        description=(
            "Source-specific non-canonical C strategy choices. Absence of a "
            "target/family binding selects the canonical RuleCard."
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
            "Portable dominance/capture/effect scope. Contains no source AST, "
            "line information, helper names, or frontend API spellings."
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
            "Semantic storage identity/lifetime only. Physical allocation, "
            "register/shared/global placement and double buffering belong to C."
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
        {"source_name": _STRING, "entries": _array(entry)},
        ("source_name", "entries"),
    )


def scrc_lift_core_json_schema() -> dict[str, Any]:
    """Return the source-aware LLM subresponse schema inside Lift.

    This is deliberately not the schema of the completed Lift artifact.  The
    enclosing Lift operation deterministically materializes pi and compose
    provenance before returning ``SCRCExtraction.artifact`` to Gate1.
    """

    semantic_archive = _closed_object(
        {
            "kernel_id": _STRING,
            "intent": _intent_schema(),
            "node_ids": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "boundary": _boundary_schema(),
            "scopes": {
                "type": "array",
                "minItems": 1,
                "items": _portable_scope_schema(),
            },
            "storage": _array(_storage_schema()),
        },
        ("kernel_id", "intent", "node_ids", "boundary", "scopes", "storage"),
    )
    schema = _closed_object(
        {
            "j": semantic_archive,
            "interface": _interface_schema(),
            "schedule": _schedule_schema(),
            "strategy": _strategy_schema(),
            "location_map": _location_map_schema(),
        },
        ("j", "interface", "schedule", "strategy", "location_map"),
    )
    schema["$defs"] = {"intent_op": _intent_op_schema()}
    return copy.deepcopy(schema)


def scrc_lift_core_text_format() -> dict[str, Any]:
    """Return the Responses format for Lift's source-aware LLM substage."""

    return {
        "type": "json_schema",
        "name": "scrc_lift_core",
        # IntentIR keeps op-specific attrs and Layer B JSON open. J.layer_c is
        # empty; concrete C is carried by the separate schedule object.
        "strict": False,
        "schema": scrc_lift_core_json_schema(),
    }


def scrc_artifact_json_schema() -> dict[str, Any]:
    """Compatibility alias for the historical LiftCore schema name."""

    return scrc_lift_core_json_schema()


def scrc_artifact_text_format() -> dict[str, Any]:
    """Compatibility alias for the historical LiftCore format name."""

    return scrc_lift_core_text_format()


__all__ = [
    "scrc_artifact_json_schema",
    "scrc_artifact_text_format",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
]
