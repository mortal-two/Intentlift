"""Responses JSON Schema for the source-visible SCRC lift artifact.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract given to the model so closed Gate1 objects do not drift into
invented field names.  IntentIR operation payloads intentionally retain their
existing open ``attrs``/``layer_b`` representation.  ``J.intent.layer_c`` is
forced empty because SCRC stores concrete C only in ``schedule``.  The Responses
format is non-strict and the project parser performs the final validation.
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
_STRING_ARRAY = _array(_STRING)
_INT_OR_STRING = {"anyOf": [{"type": "integer"}, _STRING]}


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
    # ``attrs``, ``type``, ``layer_b`` and loop records are open JSON objects in
    # the current IntentIR model.  Their enclosing objects are nevertheless
    # closed here, matching ``_strict_intent`` in Gate1's model.
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
            "carry": _array({"type": "object"}),
            "trip": _nullable({"type": "object"}),
            "scan": _array({"type": "object"}),
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
            "predicate": _nullable(_STRING),
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
        },
        ("kind", "target"),
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
                "additionalProperties": {"type": "string"},
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
            "to a J tensor; semantic scalars use shape=[]. Do not invent "
            "dimension reference fields."
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


def _plan_schema() -> dict[str, Any]:
    step = _closed_object(
        {
            "rule": _STRING,
            "node_ref": _nullable(_STRING),
            "refs": _STRING_ARRAY,
        },
        ("rule",),
    )
    return _closed_object({"steps": _array(step)}, ("steps",))


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


def scrc_artifact_json_schema() -> dict[str, Any]:
    """Return the model-produced portion of ``Gate1Artifact``."""

    semantic_archive = _closed_object(
        {
            "kernel_id": _STRING,
            "intent": _intent_schema(),
            "node_ids": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "boundary": _boundary_schema(),
        },
        ("kernel_id", "intent", "node_ids", "boundary"),
    )
    schema = _closed_object(
        {
            "j": semantic_archive,
            "interface": _interface_schema(),
            "schedule": _schedule_schema(),
            "plan": _plan_schema(),
            "location_map": _location_map_schema(),
        },
        ("j", "interface", "schedule", "plan", "location_map"),
    )
    schema["$defs"] = {"intent_op": _intent_op_schema()}
    return copy.deepcopy(schema)


def scrc_artifact_text_format() -> dict[str, Any]:
    """Return a Responses ``text.format`` value for the lift call."""

    return {
        "type": "json_schema",
        "name": "scrc_gate1_artifact",
        # IntentIR keeps op-specific attrs and Layer B JSON open. J.layer_c is
        # empty; concrete C is carried by the separate schedule object.
        "strict": False,
        "schema": scrc_artifact_json_schema(),
    }


__all__ = ["scrc_artifact_json_schema", "scrc_artifact_text_format"]
