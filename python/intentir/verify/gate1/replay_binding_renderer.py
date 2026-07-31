"""Generic Triton syntax rendering for agent-authored Replay target bindings.

This renderer performs no instruction selection.  Every callee, operator, literal,
predicate and control construct comes from the fresh binding.  Fixed code only
resolves J handles, allocates stable locals, nests authored regions and emits syntax.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .replay_binding import ReplayTargetBinding


REPLAY_BINDING_RENDER_PROTOCOL = "intentir.replay-target-binding-render.v1"
_RAW_PARAMETER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BINARY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "@",
        "<<",
        ">>",
        "&",
        "|",
        "^",
        "and",
        "or",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "is",
        "is not",
        "in",
        "not in",
    }
)
_UNARY_OPERATORS = frozenset({"+", "-", "~", "not"})


def _local_name(handle: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", handle)
    return f"_intentir_{normalized}"


def _semantic_records(skeleton: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    semantic = skeleton.get("semantic")
    if not isinstance(semantic, Mapping):
        raise TypeError("Replay skeleton semantic section must be an object")
    result: dict[str, dict[str, Any]] = {}
    for node in semantic.get("nodes", []):
        if isinstance(node, Mapping) and isinstance(node.get("id"), str):
            result[str(node["id"])] = {"kind": "intent_op", **dict(node)}
    boundary = semantic.get("boundary", {})
    if not isinstance(boundary, Mapping):
        raise TypeError("Replay skeleton boundary must be an object")
    for relation_name, id_field, kind in (
        ("accesses", "access_id", "boundary_access"),
        ("effects", "effect_id", "boundary_effect"),
    ):
        rows = boundary.get(relation_name, [])
        if not isinstance(rows, list):
            raise TypeError(f"Replay skeleton boundary {relation_name} must be an array")
        for row in rows:
            if isinstance(row, Mapping) and isinstance(row.get(id_field), str):
                result[str(row[id_field])] = {"kind": kind, **dict(row)}
    domains = boundary.get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("Replay skeleton domains must be an object")
    for domain_ref, expression in domains.items():
        result[str(domain_ref)] = {
            "kind": "domain",
            "id": str(domain_ref),
            "expression": expression,
        }
    return result


def _global_names(skeleton: Mapping[str, Any]) -> dict[str, str]:
    semantic = skeleton.get("semantic")
    if not isinstance(semantic, Mapping):
        raise TypeError("Replay skeleton semantic section must be an object")
    names: dict[str, str] = {}
    signature = semantic.get("signature", {})
    if not isinstance(signature, Mapping):
        raise TypeError("Replay skeleton signature must be an object")
    parameters = signature.get("parameters", [])
    if not isinstance(parameters, list):
        raise TypeError("Replay skeleton signature parameters must be an array")
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str):
            raise TypeError("Replay skeleton signature parameter is malformed")
        name = str(parameter["name"])
        names[name] = name
        logical_ref = parameter.get("logical_ref")
        if isinstance(logical_ref, str):
            names.setdefault(logical_ref, name)
    masked_c = skeleton.get("masked_c", {})
    if not isinstance(masked_c, Mapping):
        raise TypeError("Replay skeleton MaskC must be an object")
    slots = masked_c.get("slots", [])
    if not isinstance(slots, list):
        raise TypeError("Replay skeleton MaskC slots must be an array")
    for slot in slots:
        if isinstance(slot, Mapping) and isinstance(slot.get("name"), str):
            names[str(slot["name"])] = str(slot["name"])
    for node in semantic.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        produced = node.get("result")
        for ref in (
            [produced]
            if isinstance(produced, str)
            else produced
            if isinstance(produced, list)
            else []
        ):
            if isinstance(ref, str):
                names[ref] = _local_name(ref)
    boundary = semantic.get("boundary", {})
    domains = boundary.get("domains", {}) if isinstance(boundary, Mapping) else {}
    if isinstance(domains, Mapping):
        for domain_ref in domains:
            names[str(domain_ref)] = _local_name(str(domain_ref))
    return names


def _raw_text(text: str, uses: Sequence[str], parameters: Mapping[str, str]) -> str:
    allowed = set(uses)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in allowed or name not in parameters:
            raise ValueError(f"raw target text references unbound parameter {name}")
        return parameters[name]

    rendered = _RAW_PARAMETER.sub(replace, text)
    unresolved = _RAW_PARAMETER.findall(rendered)
    if unresolved:
        raise ValueError(f"raw target text has unresolved parameters: {unresolved}")
    return rendered


def _target_literal(value: Any, globals_: Mapping[str, str]) -> str:
    if isinstance(value, str) and value in globals_:
        return globals_[value]
    if isinstance(value, list):
        return "[" + ", ".join(_target_literal(item, globals_) for item in value) + "]"
    if isinstance(value, Mapping):
        raise ValueError("target binding cannot inject an object-valued semantic field")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def _flatten_binding_fields(
    value: Any, *, prefix: str, output: dict[str, str], globals_: Mapping[str, str]
) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            child = f"{prefix}.{name}" if prefix else str(name)
            _flatten_binding_fields(item, prefix=child, output=output, globals_=globals_)
        return
    output[prefix] = _target_literal(value, globals_)


def _render_expression(
    node: Mapping[str, Any],
    *,
    operands: Sequence[str],
    parameters: Mapping[str, str],
    globals_: Mapping[str, str],
) -> str:
    kind = node["kind"]
    if kind == "operand":
        index = int(node["index"])
        if index >= len(operands):
            raise ValueError(f"target binding operand {index} exceeds semantic arity")
        return operands[index]
    if kind == "semantic_ref":
        ref = str(node["ref"])
        if ref not in globals_:
            raise ValueError(f"target binding references unavailable semantic handle {ref}")
        return globals_[ref]
    if kind == "semantic_attr":
        name = "@attr." + str(node["name"])
        if name not in globals_:
            raise ValueError(f"target binding references unavailable semantic attr {node['name']}")
        return globals_[name]
    if kind == "hole_field":
        name = "@hole." + str(node["path"])
        if name not in globals_:
            raise ValueError(f"target binding references unavailable hole field {node['path']}")
        return globals_[name]
    if kind == "parameter":
        name = str(node["name"])
        if name not in parameters:
            raise ValueError(f"target binding parameter {name} is unbound")
        return parameters[name]
    if kind == "literal":
        value = node["value"]
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return repr(value)
    if kind == "call":
        args = [
            _render_expression(
                item, operands=operands, parameters=parameters, globals_=globals_
            )
            for item in node.get("args", [])
        ]
        kwargs = [
            f"{name}="
            + _render_expression(
                item, operands=operands, parameters=parameters, globals_=globals_
            )
            for name, item in node.get("kwargs", {}).items()
        ]
        return f"{node['callee']}({', '.join([*args, *kwargs])})"
    if kind == "binary":
        operator = str(node["operator"])
        if operator not in _BINARY_OPERATORS:
            raise ValueError(f"unsupported target binary syntax {operator!r}")
        left = _render_expression(
            node["left"], operands=operands, parameters=parameters, globals_=globals_
        )
        right = _render_expression(
            node["right"], operands=operands, parameters=parameters, globals_=globals_
        )
        return f"({left} {operator} {right})"
    if kind == "unary":
        operator = str(node["operator"])
        if operator not in _UNARY_OPERATORS:
            raise ValueError(f"unsupported target unary syntax {operator!r}")
        operand = _render_expression(
            node["operand"], operands=operands, parameters=parameters, globals_=globals_
        )
        separator = " " if operator == "not" else ""
        return f"({operator}{separator}{operand})"
    if kind == "attribute":
        base = _render_expression(
            node["base"], operands=operands, parameters=parameters, globals_=globals_
        )
        return f"{base}.{node['attribute']}"
    if kind == "subscript":
        base = _render_expression(
            node["base"], operands=operands, parameters=parameters, globals_=globals_
        )
        index = _render_expression(
            node["index"], operands=operands, parameters=parameters, globals_=globals_
        )
        return f"{base}[{index}]"
    if kind in {"tuple", "list"}:
        values = [
            _render_expression(
                item, operands=operands, parameters=parameters, globals_=globals_
            )
            for item in node["items"]
        ]
        if kind == "list":
            return "[" + ", ".join(values) + "]"
        suffix = "," if len(values) == 1 else ""
        return "(" + ", ".join(values) + suffix + ")"
    if kind == "raw_expr":
        return _raw_text(str(node["text"]), node["uses"], parameters)
    raise ValueError(f"unknown target expression kind {kind!r}")


def _render_statements(
    statements: Sequence[Mapping[str, Any]],
    *,
    indent: int,
    operands: Sequence[str],
    parameters: Mapping[str, str],
    globals_: Mapping[str, str],
) -> list[str]:
    prefix = "    " * indent
    lines: list[str] = []
    for node in statements:
        kind = node["kind"]
        if kind == "assign":
            target = parameters.get(str(node["target"]), str(node["target"]))
            value = _render_expression(
                node["value"], operands=operands, parameters=parameters, globals_=globals_
            )
            lines.append(f"{prefix}{target} = {value}")
        elif kind == "expr":
            value = _render_expression(
                node["value"], operands=operands, parameters=parameters, globals_=globals_
            )
            lines.append(f"{prefix}{value}")
        elif kind == "return":
            if "value" in node:
                value = _render_expression(
                    node["value"], operands=operands, parameters=parameters, globals_=globals_
                )
                lines.append(f"{prefix}return {value}")
            else:
                lines.append(f"{prefix}return")
        elif kind in {"if", "while"}:
            test = _render_expression(
                node["test"], operands=operands, parameters=parameters, globals_=globals_
            )
            lines.append(f"{prefix}{kind} {test}:")
            body = _render_statements(
                node["body"],
                indent=indent + 1,
                operands=operands,
                parameters=parameters,
                globals_=globals_,
            )
            lines.extend(body or ["    " * (indent + 1) + "pass"])
            orelse = node.get("orelse", [])
            if orelse:
                lines.append(f"{prefix}else:")
                lines.extend(
                    _render_statements(
                        orelse,
                        indent=indent + 1,
                        operands=operands,
                        parameters=parameters,
                        globals_=globals_,
                    )
                )
        elif kind == "for":
            target = parameters.get(str(node["target"]), str(node["target"]))
            iterator = _render_expression(
                node["iterator"], operands=operands, parameters=parameters, globals_=globals_
            )
            lines.append(f"{prefix}for {target} in {iterator}:")
            body = _render_statements(
                node["body"],
                indent=indent + 1,
                operands=operands,
                parameters=parameters,
                globals_=globals_,
            )
            lines.extend(body or ["    " * (indent + 1) + "pass"])
        elif kind in {"break", "continue", "pass"}:
            lines.append(f"{prefix}{kind}")
        elif kind == "raw_stmt":
            raw = _raw_text(str(node["text"]), node["uses"], parameters)
            lines.extend(f"{prefix}{line}" if line else "" for line in raw.splitlines())
        else:
            raise ValueError(f"unknown target statement kind {kind!r}")
    return lines


@dataclass(frozen=True)
class ReplayBindingRender:
    source: str
    source_map: tuple[dict[str, Any], ...]

    def to_response(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_BINDING_RENDER_PROTOCOL,
            "status": "pass",
            "source_bytes": len(self.source.encode("utf-8")),
            "source_map": [dict(item) for item in self.source_map],
            "authority": (
                "agent-authored target binding rendered through exact J/pi skeleton; "
                "fixed code selected no operator, callee, predicate, formula or control"
            ),
        }


def render_triton_target_binding(
    skeleton: Mapping[str, Any], binding: ReplayTargetBinding
) -> ReplayBindingRender:
    """Render one complete agent-authored binding as Triton Python syntax."""

    semantic_records = _semantic_records(skeleton)
    globals_ = _global_names(skeleton)
    holes = {
        str(item["id"]): item
        for name in ("semantic", "realization")
        for item in skeleton["holes"][name]
    }
    templates = {str(item["id"]): item for item in binding.templates}
    applications = {str(item["id"]): item for item in binding.applications}
    regions = {str(item["region"]): item for item in binding.regions}

    semantic = skeleton["semantic"]
    signature = semantic["signature"]
    parameters = []
    for parameter in signature["parameters"]:
        name = str(parameter["name"])
        annotation = (
            ": tl.constexpr" if parameter.get("kind") == "constexpr" else ""
        )
        parameters.append(name + annotation)
    for slot in skeleton["masked_c"].get("slots", []):
        name = str(slot["name"])
        annotation = ": tl.constexpr" if slot.get("kind") == "constexpr" else ""
        parameters.append(name + annotation)
    public = skeleton.get("public_frontend", {})
    function_name = (
        str(public.get("replay_function"))
        if isinstance(public, Mapping) and public.get("replay_function")
        else "scrc_replay"
    )
    lines = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    if parameters:
        lines.append(f"def {function_name}(")
        lines.extend(f"    {item}," for item in parameters)
        lines.append("):")
    else:
        lines.append(f"def {function_name}():")
    source_map: list[dict[str, Any]] = []

    def render_application(application_id: str, indent: int) -> list[str]:
        application = applications[application_id]
        template = templates[str(application["binding"])]
        result: list[str] = []
        for hole_id in application["holes"]:
            hole = holes[str(hole_id)]
            semantic_ref = hole.get("semantic_ref")
            record = semantic_records.get(str(semantic_ref), {})
            operand_refs = record.get("operands", [])
            if not isinstance(operand_refs, list):
                operand_refs = []
            operand_names = []
            for ref in operand_refs:
                if not isinstance(ref, str) or ref not in globals_:
                    raise ValueError(
                        f"binding hole {hole_id} operand {ref!r} has no target local"
                    )
                operand_names.append(globals_[ref])
            template_parameters = list(template["parameters"])
            if len(template_parameters) > len(operand_names):
                raise ValueError(
                    f"binding {template['id']} declares more parameters than hole {hole_id} operands"
                )
            parameter_names = {
                name: operand_names[index]
                for index, name in enumerate(template_parameters)
            }
            local_globals = dict(globals_)
            attrs = record.get("attrs", {})
            if isinstance(attrs, Mapping):
                for name, value in attrs.items():
                    local_globals[f"@attr.{name}"] = _target_literal(value, globals_)
            flattened_hole_fields: dict[str, str] = {}
            _flatten_binding_fields(
                hole,
                prefix="",
                output=flattened_hole_fields,
                globals_=globals_,
            )
            local_globals.update(
                {f"@hole.{name}": value for name, value in flattened_hole_fields.items()}
            )
            result_value = template.get("result")
            result_aliases = (
                [result_value]
                if isinstance(result_value, str)
                else result_value
                if isinstance(result_value, list)
                else []
            )
            produced = record.get("result")
            if record.get("kind") == "domain":
                produced = semantic_ref
            produced_refs = (
                [produced]
                if isinstance(produced, str)
                else produced
                if isinstance(produced, list)
                else []
            )
            if result_aliases:
                if len(result_aliases) != len(produced_refs) or any(
                    not isinstance(item, str) or item not in globals_
                    for item in produced_refs
                ):
                    raise ValueError(
                        f"binding {template['id']} result arity does not match hole {hole_id}"
                    )
                for result_alias, produced_ref in zip(
                    result_aliases, produced_refs, strict=True
                ):
                    parameter_names[str(result_alias)] = globals_[str(produced_ref)]
            start = len(lines) + len(result) + 1
            rendered = _render_statements(
                template["body"],
                indent=indent,
                operands=operand_names,
                parameters=parameter_names,
                globals_=local_globals,
            )
            result.extend(rendered)
            source_map.append(
                {
                    "application": application_id,
                    "binding": template["id"],
                    "hole": hole_id,
                    "semantic_ref": semantic_ref,
                    "region": application["region"],
                    "start_line": start,
                    "end_line": start + max(0, len(rendered) - 1),
                }
            )
        return result

    def render_region(region_id: str, indent: int) -> list[str]:
        region = regions[region_id]

        def render_refs(refs: Sequence[str], body_indent: int) -> list[str]:
            result: list[str] = []
            for ref in refs:
                if ref in applications:
                    result.extend(render_application(ref, body_indent))
                elif ref in regions:
                    result.extend(render_region(ref, body_indent))
                else:
                    raise ValueError(f"binding region references unknown item {ref}")
            return result

        construct = region["construct"]
        prefix = "    " * indent
        if construct == "transparent":
            result = render_refs(region["body"], indent)
            result.extend(render_refs(region["orelse"], indent))
            return result
        if construct in {"if", "while"}:
            test = _render_expression(
                region["test"], operands=(), parameters={}, globals_=globals_
            )
            result = [f"{prefix}{construct} {test}:"]
        elif construct == "for":
            iterator = _render_expression(
                region["iterator"], operands=(), parameters={}, globals_=globals_
            )
            result = [f"{prefix}for {region['target']} in {iterator}:"]
        else:
            header = str(region["raw_header"])
            result = [f"{prefix}{line}" for line in header.splitlines()]
        body = render_refs(region["body"], indent + 1)
        result.extend(body or ["    " * (indent + 1) + "pass"])
        if region["orelse"]:
            result.append(f"{prefix}else:")
            result.extend(render_refs(region["orelse"], indent + 1))
        if construct == "raw" and region["raw_footer"]:
            result.extend(
                f"{prefix}{line}" for line in str(region["raw_footer"]).splitlines()
            )
        return result

    roots = [
        str(item["id"])
        for item in skeleton["realization"]["regions"]
        if item.get("parent") is None
    ]
    if len(roots) != 1:
        raise ValueError("Replay skeleton must have exactly one root region")
    body = render_region(roots[0], 1)
    lines.extend(body or ["    pass"])
    source = "\n".join(lines) + "\n"
    return ReplayBindingRender(source, tuple(source_map))


__all__ = [
    "REPLAY_BINDING_RENDER_PROTOCOL",
    "ReplayBindingRender",
    "render_triton_target_binding",
]
