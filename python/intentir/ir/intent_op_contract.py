"""Single structural contract for canonical Layer-A ``IntentOp`` JSON.

The contract deliberately recognizes three structural discriminators:
``op == "loop"``, ``op == "effect_if"``, and ``op == "value_if"``.  Every ordinary
semantic op shares the same open-opcode/non-region shape; this is not an
operation catalog.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


COMMON_FIELDS = frozenset(
    {"id", "op", "result", "operands", "attrs", "type"}
)
LOOP_ONLY_FIELDS = frozenset(
    {"scope_id", "carry", "trip", "scan", "body", "yield"}
)
EFFECT_IF_ONLY_FIELDS = frozenset(
    {"predicate", "positive", "negative"}
)
VALUE_IF_ONLY_FIELDS = frozenset(
    {"result", "type", "predicate", "positive", "negative"}
)
COMMON_REQUIRED_FIELDS = frozenset({"op"})
LOOP_REQUIRED_FIELDS = frozenset(
    {"result", "scope_id", "carry", "trip", "scan", "body", "yield"}
)
EFFECT_IF_REQUIRED_FIELDS = frozenset(
    {"predicate", "positive", "negative"}
)
VALUE_IF_REQUIRED_FIELDS = frozenset(
    {"result", "type", "predicate", "positive", "negative"}
)


class IntentOpContractError(ValueError):
    """An IntentOp violates the shared wire/parser structural shape."""


def _fail(path: str, message: str) -> None:
    raise IntentOpContractError(f"{path} {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(path, "must be an object with string keys")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _string_array(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result = [
        _nonempty_string(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    if nonempty and not result:
        _fail(path, "must be a non-empty array")
    if unique and len(result) != len(set(result)):
        _fail(path, "must contain unique values")
    return result


def _nonnegative_integer(value: Any, path: str) -> int:
    # JSON Schema defines 1 and 1.0 as the same mathematical integer.  Mirror
    # that wire semantics, then let the typed model canonicalize it back to int.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a non-negative integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        _fail(path, "must be a non-negative integer")
    result = int(value)
    if result < 0:
        _fail(path, "must be a non-negative integer")
    return result


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _json_value(item, f"{path}.{key}")
        return
    _fail(path, "must contain only finite JSON values")


def _exact_keys(
    value: Any,
    path: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> dict[str, Any]:
    item = _object(value, path)
    missing = sorted(required - set(item))
    unknown = sorted(set(item) - allowed)
    if missing:
        _fail(path, "is missing required key(s): " + ", ".join(missing))
    if unknown:
        _fail(path, "has unknown key(s): " + ", ".join(unknown))
    return item


def _shape(value: Any, path: str) -> None:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    for index, dimension in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(dimension, str):
            _nonempty_string(dimension, item_path)
        else:
            _nonnegative_integer(dimension, item_path)


def _result(value: Any, path: str, *, loop: bool) -> None:
    if loop:
        _string_array(value, path, nonempty=True, unique=True)
        return
    if value is None:
        return
    if isinstance(value, str):
        _nonempty_string(value, path)
        return
    _string_array(value, path, unique=True)


def _loop_state(value: Any, path: str) -> None:
    fields = frozenset(
        {
            "state_id",
            "init_value",
            "body_argument",
            "next_value",
            "result_value",
            "type",
        }
    )
    item = _exact_keys(value, path, allowed=fields, required=fields)
    for key in ("state_id", "body_argument", "next_value", "result_value"):
        _nonempty_string(item[key], f"{path}.{key}")
    init = item["init_value"]
    if isinstance(init, (dict, list)):
        _fail(
            f"{path}.init_value",
            "must be a scalar literal or a visible value reference",
        )
    _json_value(init, f"{path}.init_value")
    value_type = _exact_keys(
        item["type"],
        f"{path}.type",
        allowed=frozenset({"dtype", "shape"}),
        required=frozenset({"dtype", "shape"}),
    )
    _nonempty_string(value_type["dtype"], f"{path}.type.dtype")
    _shape(value_type["shape"], f"{path}.type.shape")


def _trip(value: Any, path: str) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"axis"}),
        required=frozenset({"axis"}),
    )
    _nonempty_string(item["axis"], f"{path}.axis")


def _scan(value: Any, path: str) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"name", "from", "axis"}),
        required=frozenset({"name", "from", "axis"}),
    )
    _nonempty_string(item["name"], f"{path}.name")
    _nonempty_string(item["from"], f"{path}.from")
    _nonnegative_integer(item["axis"], f"{path}.axis")


def _effect_if_arm(value: Any, path: str, *, recursive: bool) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"scope_id", "body"}),
        required=frozenset({"scope_id", "body"}),
    )
    _nonempty_string(item["scope_id"], f"{path}.scope_id")
    body = item["body"]
    if not isinstance(body, list):
        _fail(f"{path}.body", "must be an array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation,
                path=f"{path}.body[{index}]",
                recursive=True,
            )


def _value_if_arm(value: Any, path: str, *, recursive: bool) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"scope_id", "body", "yield_value"}),
        required=frozenset({"scope_id", "body", "yield_value"}),
    )
    _nonempty_string(item["scope_id"], f"{path}.scope_id")
    _nonempty_string(item["yield_value"], f"{path}.yield_value")
    body = item["body"]
    if not isinstance(body, list):
        _fail(f"{path}.body", "must be an array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation,
                path=f"{path}.body[{index}]",
                recursive=True,
            )


def validate_intent_op_structure(
    value: Any,
    *,
    path: str = "IntentOp",
    recursive: bool = True,
) -> None:
    """Validate only the shape shared by wire and parser.

    Visibility, type equality, scope ownership and storage lifetime remain the
    full semantic parser's responsibility.
    """

    item = _object(value, path)
    op = _nonempty_string(item.get("op"), f"{path}.op")
    is_loop = op == "loop"
    is_effect_if = op == "effect_if"
    is_value_if = op == "value_if"
    structural_fields = (
        LOOP_ONLY_FIELDS if is_loop
        else EFFECT_IF_ONLY_FIELDS if is_effect_if
        else VALUE_IF_ONLY_FIELDS if is_value_if
        else frozenset()
    )
    structural_required = (
        LOOP_REQUIRED_FIELDS if is_loop
        else EFFECT_IF_REQUIRED_FIELDS if is_effect_if
        else VALUE_IF_REQUIRED_FIELDS if is_value_if
        else frozenset()
    )
    # ``effect_if`` is effect-only at its interface: even common ordinary-op
    # result/operand/attr/type fields are not legal on the region operation.
    common_allowed = (
        frozenset({"id", "op"}) if is_effect_if
        else frozenset({"id", "op"}) if is_value_if
        else COMMON_FIELDS
    )
    allowed = common_allowed | structural_fields
    required = COMMON_REQUIRED_FIELDS | structural_required
    missing = sorted(required - set(item))
    unknown = sorted(set(item) - allowed)
    if missing:
        _fail(path, "is missing required key(s): " + ", ".join(missing))
    if unknown:
        region_only = LOOP_ONLY_FIELDS | EFFECT_IF_ONLY_FIELDS | VALUE_IF_ONLY_FIELDS
        if not is_loop and not is_effect_if and not is_value_if and set(unknown) <= region_only:
            _fail(
                path,
                "has region-only key(s) on an ordinary op: " + ", ".join(unknown),
            )
        _fail(path, "has unknown key(s): " + ", ".join(unknown))

    if "id" in item:
        _nonempty_string(item["id"], f"{path}.id")
    if "result" in item:
        _result(item["result"], f"{path}.result", loop=is_loop)
    if "operands" in item:
        _string_array(item["operands"], f"{path}.operands")
    if "attrs" in item:
        _object(item["attrs"], f"{path}.attrs")
        _json_value(item["attrs"], f"{path}.attrs")
    if "type" in item and item["type"] is not None:
        _object(item["type"], f"{path}.type")
        _json_value(item["type"], f"{path}.type")

    if is_effect_if:
        _nonempty_string(item["predicate"], f"{path}.predicate")
        _effect_if_arm(
            item["positive"], f"{path}.positive", recursive=recursive
        )
        _effect_if_arm(
            item["negative"], f"{path}.negative", recursive=recursive
        )
        return
    if is_value_if:
        _nonempty_string(item["result"], f"{path}.result")
        value_type = _exact_keys(
            item["type"],
            f"{path}.type",
            allowed=frozenset({"dtype", "shape"}),
            required=frozenset({"dtype", "shape"}),
        )
        _nonempty_string(value_type["dtype"], f"{path}.type.dtype")
        _shape(value_type["shape"], f"{path}.type.shape")
        _nonempty_string(item["predicate"], f"{path}.predicate")
        _value_if_arm(item["positive"], f"{path}.positive", recursive=recursive)
        _value_if_arm(item["negative"], f"{path}.negative", recursive=recursive)
        return
    if not is_loop:
        return
    _nonempty_string(item["scope_id"], f"{path}.scope_id")
    _trip(item["trip"], f"{path}.trip")
    carry = item["carry"]
    if not isinstance(carry, list) or not carry:
        _fail(f"{path}.carry", "must be a non-empty array")
    for index, state in enumerate(carry):
        _loop_state(state, f"{path}.carry[{index}]")
    scan = item["scan"]
    if not isinstance(scan, list):
        _fail(f"{path}.scan", "must be an array")
    for index, scan_item in enumerate(scan):
        _scan(scan_item, f"{path}.scan[{index}]")
    _string_array(item["yield"], f"{path}.yield", nonempty=True, unique=True)
    body = item["body"]
    if not isinstance(body, list) or not body:
        _fail(f"{path}.body", "must be a non-empty array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation, path=f"{path}.body[{index}]", recursive=True
            )


def _closed_schema(
    properties: dict[str, Any], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def intent_op_json_schema() -> dict[str, Any]:
    """Project the shared contract into the recursive Responses schema."""

    string = {"type": "string", "minLength": 1}
    string_array = {"type": "array", "items": copy.deepcopy(string)}
    dimension = {
        "anyOf": [
            {"type": "integer", "minimum": 0},
            copy.deepcopy(string),
        ]
    }
    generic_result = {
        "anyOf": [
            copy.deepcopy(string),
            {
                "type": "array",
                "items": copy.deepcopy(string),
                "uniqueItems": True,
            },
            {"type": "null"},
        ]
    }
    common = {
        "id": copy.deepcopy(string),
        "op": copy.deepcopy(string),
        "result": generic_result,
        "operands": copy.deepcopy(string_array),
        "attrs": {"type": "object"},
        "type": {"anyOf": [{"type": "object"}, {"type": "null"}]},
    }
    state_type = _closed_schema(
        {
            "dtype": copy.deepcopy(string),
            "shape": {"type": "array", "items": dimension},
        },
        ("dtype", "shape"),
    )
    state = _closed_schema(
        {
            "state_id": copy.deepcopy(string),
            "init_value": {
                "anyOf": [
                    copy.deepcopy(string),
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
            "body_argument": copy.deepcopy(string),
            "next_value": copy.deepcopy(string),
            "result_value": copy.deepcopy(string),
            "type": state_type,
        },
        (
            "state_id",
            "init_value",
            "body_argument",
            "next_value",
            "result_value",
            "type",
        ),
    )
    trip = _closed_schema({"axis": copy.deepcopy(string)}, ("axis",))
    scan = _closed_schema(
        {
            "name": copy.deepcopy(string),
            "from": copy.deepcopy(string),
            "axis": {"type": "integer", "minimum": 0},
        },
        ("name", "from", "axis"),
    )
    non_loop = _closed_schema(
        {
            **copy.deepcopy(common),
            "op": {
                "type": "string",
                "minLength": 1,
                "pattern": "^(?!(?:loop|effect_if|value_if)$).+$",
            },
        },
        ("op",),
    )
    loop = _closed_schema(
        {
            **copy.deepcopy(common),
            "op": {"type": "string", "enum": ["loop"]},
            "result": {
                "type": "array",
                "items": copy.deepcopy(string),
                "minItems": 1,
                "uniqueItems": True,
            },
            "scope_id": copy.deepcopy(string),
            "carry": {
                "type": "array",
                "items": state,
                "minItems": 1,
            },
            "trip": trip,
            "scan": {"type": "array", "items": scan},
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
                "minItems": 1,
            },
            "yield": {
                "type": "array",
                "items": copy.deepcopy(string),
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        (
            "op",
            "result",
            "scope_id",
            "carry",
            "trip",
            "scan",
            "body",
            "yield",
        ),
    )
    arm = _closed_schema(
        {
            "scope_id": copy.deepcopy(string),
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
            },
        },
        ("scope_id", "body"),
    )
    effect_if = _closed_schema(
        {
            "id": copy.deepcopy(string),
            "op": {"type": "string", "enum": ["effect_if"]},
            "predicate": copy.deepcopy(string),
            "positive": copy.deepcopy(arm),
            "negative": copy.deepcopy(arm),
        },
        ("op", "predicate", "positive", "negative"),
    )
    value_arm = _closed_schema(
        {
            "scope_id": copy.deepcopy(string),
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
            },
            "yield_value": copy.deepcopy(string),
        },
        ("scope_id", "body", "yield_value"),
    )
    value_if = _closed_schema(
        {
            "id": copy.deepcopy(string),
            "op": {"type": "string", "enum": ["value_if"]},
            "result": copy.deepcopy(string),
            "type": copy.deepcopy(state_type),
            "predicate": copy.deepcopy(string),
            "positive": copy.deepcopy(value_arm),
            "negative": copy.deepcopy(value_arm),
        },
        ("op", "result", "type", "predicate", "positive", "negative"),
    )
    return {"anyOf": [non_loop, loop, effect_if, value_if]}


def render_intent_op_wire_grammar() -> str:
    common = ", ".join(sorted(COMMON_FIELDS))
    loop_only = ", ".join(sorted(LOOP_ONLY_FIELDS))
    required = ", ".join(sorted(LOOP_REQUIRED_FIELDS))
    return (
        "IntentOp is one global discriminated shape. Common fields are: "
        f"{common}. If and only if op is exactly 'loop', it also exposes "
        f"loop-only fields: {loop_only}, and all of these loop fields are "
        f"required: {required}. Every non-loop op must omit every loop-only "
        "field; null or an empty placeholder does not count as omission. A "
        "loop has a non-empty result list, carry list, body and yield list. "
        "If and only if op is exactly 'effect_if', it has exactly id(optional), "
        "op, predicate, positive and negative. Both arms are explicit closed "
        "{scope_id,body} objects and either body may be empty. The conditional "
        "itself has no result, operands, attrs, type or yield; arm-local values "
        "may feed arm effects but never escape to the parent continuation. "
        "If and only if op is exactly 'value_if', it has exactly id(optional), "
        "op, one string result, one closed type, predicate, positive and "
        "negative. Both arms are explicit closed {scope_id,body,yield_value} "
        "objects; either body may be empty, but each yield_value is required. "
        "value_if has no operands or attrs and is distinct from select and "
        "effect_if. "
        "This is a structural rule, not an operation catalog."
    )


__all__ = [
    "COMMON_FIELDS",
    "COMMON_REQUIRED_FIELDS",
    "EFFECT_IF_ONLY_FIELDS",
    "EFFECT_IF_REQUIRED_FIELDS",
    "IntentOpContractError",
    "LOOP_ONLY_FIELDS",
    "LOOP_REQUIRED_FIELDS",
    "VALUE_IF_ONLY_FIELDS",
    "VALUE_IF_REQUIRED_FIELDS",
    "intent_op_json_schema",
    "render_intent_op_wire_grammar",
    "validate_intent_op_structure",
]
