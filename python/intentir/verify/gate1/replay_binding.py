"""Closed episode-local target bindings for a target-neutral Replay skeleton.

Bindings are authored by the fresh Replay agent.  Fixed code validates coverage,
references, arity and generic target-AST shape; it never selects a target operator,
callee, predicate, formula or control construct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .replay_skeleton import REPLAY_BINDING_PROTOCOL, REPLAY_SKELETON_PROTOCOL


_BINDING_ID = re.compile(r"b[0-9]+")
_HOLE_ID = re.compile(r"(?:h|rh)[0-9]+")
_LOCAL_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_EXPR_KINDS = frozenset(
    {
        "operand",
        "semantic_ref",
        "parameter",
        "literal",
        "call",
        "binary",
        "unary",
        "attribute",
        "subscript",
        "tuple",
        "list",
        "raw_expr",
    }
)
_STMT_KINDS = frozenset(
    {
        "assign",
        "expr",
        "return",
        "if",
        "for",
        "while",
        "break",
        "continue",
        "pass",
        "raw_stmt",
    }
)


def _closed_object(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    if missing or unknown:
        raise ValueError(f"{path} closed fields mismatch: missing={missing}, unknown={unknown}")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _LOCAL_ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be a target-local identifier")
    return value


def _expression(value: Any, *, path: str, parameters: set[str]) -> dict[str, Any]:
    node = _closed_object(
        value,
        path=path,
        allowed={
            "kind",
            "index",
            "ref",
            "name",
            "value",
            "callee",
            "args",
            "kwargs",
            "operator",
            "left",
            "right",
            "operand",
            "base",
            "attribute",
            "items",
            "text",
            "uses",
        },
        required={"kind"},
    )
    kind = node.get("kind")
    if kind not in _EXPR_KINDS:
        raise ValueError(f"{path}.kind is not a supported generic expression")

    if kind == "operand":
        if set(node) != {"kind", "index"}:
            raise ValueError(f"{path} operand expression requires exactly index")
        index = node.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{path}.index must be a non-negative operand index")
    elif kind == "semantic_ref":
        if set(node) != {"kind", "ref"} or not isinstance(node.get("ref"), str):
            raise ValueError(f"{path} semantic_ref expression requires one ref")
    elif kind == "parameter":
        if set(node) != {"kind", "name"} or node.get("name") not in parameters:
            raise ValueError(f"{path} references an undeclared binding parameter")
    elif kind == "literal":
        if set(node) != {"kind", "value"}:
            raise ValueError(f"{path} literal expression requires exactly value")
    elif kind == "call":
        if set(node) - {"kind", "callee", "args", "kwargs"} or not isinstance(
            node.get("callee"), str
        ):
            raise ValueError(f"{path} call requires an agent-authored callee")
        args = node.get("args", [])
        kwargs = node.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, Mapping):
            raise ValueError(f"{path} call args/kwargs are malformed")
        for index, item in enumerate(args):
            _expression(item, path=f"{path}.args[{index}]", parameters=parameters)
        for name, item in kwargs.items():
            _identifier(name, f"{path}.kwargs key")
            _expression(item, path=f"{path}.kwargs.{name}", parameters=parameters)
    elif kind == "binary":
        if set(node) != {"kind", "operator", "left", "right"} or not isinstance(
            node.get("operator"), str
        ):
            raise ValueError(f"{path} binary expression is malformed")
        _expression(node["left"], path=f"{path}.left", parameters=parameters)
        _expression(node["right"], path=f"{path}.right", parameters=parameters)
    elif kind == "unary":
        if set(node) != {"kind", "operator", "operand"} or not isinstance(
            node.get("operator"), str
        ):
            raise ValueError(f"{path} unary expression is malformed")
        _expression(node["operand"], path=f"{path}.operand", parameters=parameters)
    elif kind == "attribute":
        if set(node) != {"kind", "base", "attribute"}:
            raise ValueError(f"{path} attribute expression is malformed")
        _expression(node["base"], path=f"{path}.base", parameters=parameters)
        _identifier(node.get("attribute"), f"{path}.attribute")
    elif kind == "subscript":
        if set(node) != {"kind", "base", "index"}:
            raise ValueError(f"{path} subscript expression is malformed")
        _expression(node["base"], path=f"{path}.base", parameters=parameters)
        _expression(node["index"], path=f"{path}.index", parameters=parameters)
    elif kind in {"tuple", "list"}:
        if set(node) != {"kind", "items"} or not isinstance(node.get("items"), list):
            raise ValueError(f"{path} aggregate expression requires items")
        for index, item in enumerate(node["items"]):
            _expression(item, path=f"{path}.items[{index}]", parameters=parameters)
    else:
        if set(node) != {"kind", "text", "uses"}:
            raise ValueError(f"{path} raw_expr requires text and explicit uses")
        if not isinstance(node.get("text"), str) or not node["text"].strip():
            raise ValueError(f"{path}.text must be non-empty")
        uses = node.get("uses")
        if not isinstance(uses, list) or not all(
            isinstance(item, str) and item in parameters for item in uses
        ):
            raise ValueError(f"{path}.uses must name declared binding parameters")
    return dict(node)


def _statements(
    values: Any, *, path: str, parameters: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise TypeError(f"{path} must be an array")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        node = _closed_object(
            value,
            path=item_path,
            allowed={
                "kind",
                "target",
                "value",
                "test",
                "body",
                "orelse",
                "iterator",
                "text",
                "uses",
            },
            required={"kind"},
        )
        kind = node.get("kind")
        if kind not in _STMT_KINDS:
            raise ValueError(f"{item_path}.kind is not a generic statement")
        if kind == "assign":
            if set(node) != {"kind", "target", "value"}:
                raise ValueError(f"{item_path} assign is malformed")
            _identifier(node.get("target"), f"{item_path}.target")
            _expression(node["value"], path=f"{item_path}.value", parameters=parameters)
        elif kind == "expr":
            if set(node) != {"kind", "value"}:
                raise ValueError(f"{item_path} expr is malformed")
            _expression(node["value"], path=f"{item_path}.value", parameters=parameters)
        elif kind == "return":
            if set(node) - {"kind", "value"}:
                raise ValueError(f"{item_path} return is malformed")
            if "value" in node:
                _expression(node["value"], path=f"{item_path}.value", parameters=parameters)
        elif kind == "if":
            if set(node) - {"kind", "test", "body", "orelse"} or not {
                "kind",
                "test",
                "body",
            }.issubset(node):
                raise ValueError(f"{item_path} if is malformed")
            _expression(node["test"], path=f"{item_path}.test", parameters=parameters)
            _statements(node["body"], path=f"{item_path}.body", parameters=parameters)
            _statements(node.get("orelse", []), path=f"{item_path}.orelse", parameters=parameters)
        elif kind == "for":
            if set(node) != {"kind", "target", "iterator", "body"}:
                raise ValueError(f"{item_path} for is malformed")
            _identifier(node.get("target"), f"{item_path}.target")
            _expression(node["iterator"], path=f"{item_path}.iterator", parameters=parameters)
            _statements(node["body"], path=f"{item_path}.body", parameters=parameters)
        elif kind == "while":
            if set(node) != {"kind", "test", "body"}:
                raise ValueError(f"{item_path} while is malformed")
            _expression(node["test"], path=f"{item_path}.test", parameters=parameters)
            _statements(node["body"], path=f"{item_path}.body", parameters=parameters)
        elif kind in {"break", "continue", "pass"}:
            if set(node) != {"kind"}:
                raise ValueError(f"{item_path} {kind} accepts no other fields")
        else:
            if set(node) != {"kind", "text", "uses"}:
                raise ValueError(f"{item_path} raw_stmt requires text and uses")
            if not isinstance(node.get("text"), str) or not node["text"].strip():
                raise ValueError(f"{item_path}.text must be non-empty")
            uses = node.get("uses")
            if not isinstance(uses, list) or not all(
                isinstance(item, str) and item in parameters for item in uses
            ):
                raise ValueError(f"{item_path}.uses must name declared parameters")
        result.append(dict(node))
    return result


@dataclass(frozen=True)
class ReplayTargetBinding:
    templates: tuple[dict[str, Any], ...]
    applications: tuple[dict[str, Any], ...]
    by_hole: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_BINDING_PROTOCOL,
            "templates": [dict(item) for item in self.templates],
            "applications": [dict(item) for item in self.applications],
        }


def parse_replay_target_binding(
    skeleton: Mapping[str, Any], value: Mapping[str, Any]
) -> ReplayTargetBinding:
    """Validate one complete agent-authored binding against an exact skeleton."""

    if skeleton.get("protocol") != REPLAY_SKELETON_PROTOCOL:
        raise ValueError("target binding requires a supported Replay skeleton")
    root = _closed_object(
        value,
        path="binding",
        allowed={"protocol", "templates", "applications"},
        required={"protocol", "templates", "applications"},
    )
    if root.get("protocol") != REPLAY_BINDING_PROTOCOL:
        raise ValueError("unsupported Replay target-binding protocol")
    raw_templates = root.get("templates")
    raw_applications = root.get("applications")
    if not isinstance(raw_templates, list) or not isinstance(raw_applications, list):
        raise TypeError("binding templates/applications must be arrays")

    templates: list[dict[str, Any]] = []
    template_ids: set[str] = set()
    for index, raw in enumerate(raw_templates):
        template = _closed_object(
            raw,
            path=f"binding.templates[{index}]",
            allowed={"id", "parameters", "result", "body"},
            required={"id", "parameters", "result", "body"},
        )
        template_id = template.get("id")
        if not isinstance(template_id, str) or _BINDING_ID.fullmatch(template_id) is None:
            raise ValueError("binding template ids must be b0,b1,... handles")
        if template_id in template_ids:
            raise ValueError(f"duplicate binding template {template_id}")
        template_ids.add(template_id)
        parameters_value = template.get("parameters")
        if not isinstance(parameters_value, list):
            raise TypeError(f"binding template {template_id} parameters must be an array")
        parameters = [_identifier(item, f"binding template {template_id} parameter") for item in parameters_value]
        if len(set(parameters)) != len(parameters):
            raise ValueError(f"binding template {template_id} parameters must be unique")
        result = template.get("result")
        if result is not None:
            _identifier(result, f"binding template {template_id} result")
        body = _statements(
            template.get("body"),
            path=f"binding.templates[{index}].body",
            parameters=set(parameters),
        )
        templates.append(
            {
                "id": template_id,
                "parameters": parameters,
                "result": result,
                "body": body,
            }
        )

    holes = skeleton.get("holes")
    if not isinstance(holes, Mapping):
        raise TypeError("Replay skeleton holes must be an object")
    hole_records = [
        item
        for collection in ("semantic", "realization")
        for item in holes.get(collection, [])
        if isinstance(item, Mapping)
    ]
    hole_ids = {
        str(item.get("id")) for item in hole_records if isinstance(item.get("id"), str)
    }
    regions = skeleton.get("realization", {}).get("regions", [])
    region_ids = {
        str(item.get("id"))
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }

    applications: list[dict[str, Any]] = []
    by_hole: dict[str, str] = {}
    for index, raw in enumerate(raw_applications):
        application = _closed_object(
            raw,
            path=f"binding.applications[{index}]",
            allowed={"binding", "holes", "region", "order"},
            required={"binding", "holes", "region", "order"},
        )
        binding_id = application.get("binding")
        if binding_id not in template_ids:
            raise ValueError(f"binding application {index} references unknown template")
        application_holes = application.get("holes")
        if not isinstance(application_holes, list) or not application_holes:
            raise ValueError(f"binding application {index} needs at least one hole")
        for hole_id in application_holes:
            if not isinstance(hole_id, str) or _HOLE_ID.fullmatch(hole_id) is None:
                raise ValueError(f"binding application {index} has malformed hole id")
            if hole_id not in hole_ids:
                raise ValueError(f"binding application {index} references unknown hole {hole_id}")
            if hole_id in by_hole:
                raise ValueError(f"Replay binding hole {hole_id} is covered more than once")
            by_hole[hole_id] = str(binding_id)
        region = application.get("region")
        if region is not None and region not in region_ids:
            raise ValueError(f"binding application {index} references unknown region")
        order = application.get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError(f"binding application {index} order must be non-negative")
        applications.append(dict(application))

    missing = sorted(hole_ids - set(by_hole))
    if missing:
        raise ValueError(f"Replay target binding leaves required holes unbound: {missing[:32]}")
    return ReplayTargetBinding(tuple(templates), tuple(applications), by_hole)


__all__ = ["ReplayTargetBinding", "parse_replay_target_binding"]
