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
_APPLICATION_ID = re.compile(r"a[0-9]+")
_HOLE_ID = re.compile(r"(?:h|rh)[0-9]+")
_LOCAL_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_EXPR_KINDS = frozenset(
    {
        "operand",
        "semantic_ref",
        "semantic_attr",
        "hole_field",
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
            "path",
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
    elif kind == "semantic_attr":
        if set(node) != {"kind", "name"} or not isinstance(node.get("name"), str):
            raise ValueError(f"{path} semantic_attr expression requires one name")
    elif kind == "hole_field":
        field_path = node.get("path")
        if (
            set(node) != {"kind", "path"}
            or not isinstance(field_path, str)
            or not field_path
            or any(not part for part in field_path.split("."))
        ):
            raise ValueError(f"{path} hole_field expression requires one dotted path")
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
    regions: tuple[dict[str, Any], ...]
    by_hole: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_BINDING_PROTOCOL,
            "templates": [dict(item) for item in self.templates],
            "applications": [dict(item) for item in self.applications],
            "regions": [dict(item) for item in self.regions],
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
        allowed={"protocol", "templates", "applications", "regions"},
        required={"protocol", "templates", "applications", "regions"},
    )
    if root.get("protocol") != REPLAY_BINDING_PROTOCOL:
        raise ValueError("unsupported Replay target-binding protocol")
    raw_templates = root.get("templates")
    raw_applications = root.get("applications")
    raw_regions = root.get("regions")
    if not all(
        isinstance(item, list)
        for item in (raw_templates, raw_applications, raw_regions)
    ):
        raise TypeError("binding templates/applications/regions must be arrays")

    templates: list[dict[str, Any]] = []
    template_ids: set[str] = set()
    for index, raw in enumerate(raw_templates):
        template = _closed_object(
            raw,
            path=f"binding.templates[{index}]",
            allowed={"id", "kind", "parameters", "result", "body"},
            required={"id", "kind", "parameters", "result", "body"},
        )
        template_id = template.get("id")
        if not isinstance(template_id, str) or _BINDING_ID.fullmatch(template_id) is None:
            raise ValueError("binding template ids must be b0,b1,... handles")
        if template_id in template_ids:
            raise ValueError(f"duplicate binding template {template_id}")
        template_ids.add(template_id)
        template_kind = template.get("kind")
        if template_kind not in {"expression", "statement", "control", "raw"}:
            raise ValueError(f"binding template {template_id} has unknown kind")
        parameters_value = template.get("parameters")
        if not isinstance(parameters_value, list):
            raise TypeError(f"binding template {template_id} parameters must be an array")
        parameters = [_identifier(item, f"binding template {template_id} parameter") for item in parameters_value]
        if len(set(parameters)) != len(parameters):
            raise ValueError(f"binding template {template_id} parameters must be unique")
        result = template.get("result")
        result_names: list[str] = []
        if isinstance(result, str):
            result_names = [_identifier(result, f"binding template {template_id} result")]
        elif isinstance(result, list):
            result_names = [
                _identifier(item, f"binding template {template_id} result")
                for item in result
            ]
            if not result_names or len(set(result_names)) != len(result_names):
                raise ValueError(
                    f"binding template {template_id} result aliases must be non-empty and unique"
                )
        elif result is not None:
            raise TypeError(
                f"binding template {template_id} result must be null, an identifier, or an array"
            )
        body = _statements(
            template.get("body"),
            path=f"binding.templates[{index}].body",
            parameters={*parameters, *result_names},
        )
        if not body:
            raise ValueError(f"binding template {template_id} body must be non-empty")
        templates.append(
            {
                "id": template_id,
                "kind": template_kind,
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
    hole_index = {
        str(item["id"]): item
        for item in hole_records
        if isinstance(item.get("id"), str)
    }
    template_index = {str(item["id"]): item for item in templates}
    regions = skeleton.get("realization", {}).get("regions", [])
    region_ids = {
        str(item.get("id"))
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }

    applications: list[dict[str, Any]] = []
    application_ids: set[str] = set()
    by_hole: dict[str, str] = {}
    for index, raw in enumerate(raw_applications):
        application = _closed_object(
            raw,
            path=f"binding.applications[{index}]",
            allowed={"id", "binding", "holes", "region"},
            required={"id", "binding", "holes", "region"},
        )
        application_id = application.get("id")
        if (
            not isinstance(application_id, str)
            or _APPLICATION_ID.fullmatch(application_id) is None
            or application_id in application_ids
        ):
            raise ValueError("binding application ids must be unique a0,a1,... handles")
        application_ids.add(application_id)
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
            hole_kind = hole_index[hole_id].get("kind")
            template = template_index[str(binding_id)]
            template_kind = template.get("kind")
            allowed_template_kinds = {
                "semantic_expression": {"expression", "raw"},
                "domain_predicate": {"expression", "raw"},
                "semantic_control": {"control", "raw"},
                "boundary_access": {"expression", "statement", "raw"},
                "boundary_effect": {"statement", "control", "raw"},
                "realization_binding": {"control", "statement", "raw"},
            }.get(str(hole_kind), {"raw"})
            if template_kind not in allowed_template_kinds:
                raise ValueError(
                    f"binding {binding_id} kind {template_kind} cannot cover {hole_id} kind {hole_kind}"
                )
            if hole_kind in {"semantic_expression", "domain_predicate"} and not template.get(
                "result"
            ):
                raise ValueError(
                    f"binding {binding_id} must declare result aliases for {hole_id}"
                )
            by_hole[hole_id] = str(binding_id)
        region = application.get("region")
        if region not in region_ids:
            raise ValueError(f"binding application {index} references unknown region")
        applications.append(dict(application))

    missing = sorted(hole_ids - set(by_hole))
    if missing:
        raise ValueError(f"Replay target binding leaves required holes unbound: {missing[:32]}")

    skeleton_region_records = {
        str(item.get("id")): item
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    binding_regions: list[dict[str, Any]] = []
    bound_region_ids: set[str] = set()
    placed_applications: set[str] = set()
    placed_children: set[str] = set()
    application_regions = {
        str(item["id"]): str(item["region"]) for item in applications
    }
    for index, raw in enumerate(raw_regions):
        region = _closed_object(
            raw,
            path=f"binding.regions[{index}]",
            allowed={
                "region",
                "construct",
                "body",
                "orelse",
                "test",
                "target",
                "iterator",
                "raw_header",
                "raw_footer",
            },
            required={"region", "construct", "body", "orelse"},
        )
        region_id = region.get("region")
        if region_id not in skeleton_region_records or region_id in bound_region_ids:
            raise ValueError("binding regions must cover unique skeleton region ids")
        bound_region_ids.add(str(region_id))
        construct = region.get("construct")
        if construct not in {"transparent", "if", "for", "while", "raw"}:
            raise ValueError(f"binding region {region_id} has unknown construct")
        if construct == "transparent":
            if set(region) != {"region", "construct", "body", "orelse"}:
                raise ValueError(f"transparent region {region_id} has extra fields")
        elif construct in {"if", "while"}:
            if set(region) != {"region", "construct", "body", "orelse", "test"}:
                raise ValueError(f"{construct} region {region_id} requires one test")
            _expression(
                region["test"],
                path=f"binding.regions[{index}].test",
                parameters=set(),
            )
        elif construct == "for":
            if set(region) != {
                "region",
                "construct",
                "body",
                "orelse",
                "target",
                "iterator",
            }:
                raise ValueError(f"for region {region_id} is malformed")
            _identifier(region.get("target"), f"binding region {region_id} target")
            _expression(
                region["iterator"],
                path=f"binding.regions[{index}].iterator",
                parameters=set(),
            )
        else:
            if set(region) != {
                "region",
                "construct",
                "body",
                "orelse",
                "raw_header",
                "raw_footer",
            }:
                raise ValueError(f"raw region {region_id} is malformed")
            if not all(
                isinstance(region.get(name), str)
                for name in ("raw_header", "raw_footer")
            ):
                raise ValueError(f"raw region {region_id} headers must be strings")

        direct_children = {
            child_id
            for child_id, child in skeleton_region_records.items()
            if child.get("parent") == region_id
        }
        for arm_name in ("body", "orelse"):
            arm = region.get(arm_name)
            if not isinstance(arm, list) or not all(isinstance(item, str) for item in arm):
                raise TypeError(f"binding region {region_id} {arm_name} must be refs")
            if len(set(arm)) != len(arm):
                raise ValueError(f"binding region {region_id} {arm_name} repeats refs")
            for ref in arm:
                if ref in application_ids:
                    if application_regions[ref] != region_id or ref in placed_applications:
                        raise ValueError(f"application {ref} has invalid region placement")
                    placed_applications.add(ref)
                elif ref in direct_children:
                    if ref in placed_children:
                        raise ValueError(f"child region {ref} is placed more than once")
                    placed_children.add(ref)
                else:
                    raise ValueError(
                        f"binding region {region_id} references non-local item {ref}"
                    )
        binding_regions.append(dict(region))

    missing_regions = sorted(region_ids - bound_region_ids)
    missing_apps = sorted(application_ids - placed_applications)
    non_root_regions = {
        region_id
        for region_id, record in skeleton_region_records.items()
        if record.get("parent") is not None
    }
    missing_children = sorted(non_root_regions - placed_children)
    if missing_regions or missing_apps or missing_children:
        raise ValueError(
            "Replay binding structural coverage is incomplete: "
            f"regions={missing_regions[:16]}, applications={missing_apps[:16]}, "
            f"children={missing_children[:16]}"
        )
    return ReplayTargetBinding(
        tuple(templates), tuple(applications), tuple(binding_regions), by_hole
    )


__all__ = ["ReplayTargetBinding", "parse_replay_target_binding"]
