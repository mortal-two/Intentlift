"""Deterministic, parameterized ``J -> S_J`` view for source-blind Replay.

``S_J`` is a read-only compiler input, not a target-source skeleton.  This module
canonicalizes normal J, retains symbolic dimensions and J anchors, and presents the
typed semantic program without consulting C, pi, Source, Harness values, a target,
or Gate evidence.  A masked episode is materialized from its masked J bytes rather
than from the normal semantic archive.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j, j_elaboration

from .model import SemanticArchive


REPLAY_SEMANTIC_PROGRAM_PROTOCOL = "intentir.replay-semantic-program.v1"
REPLAY_SEMANTIC_QUERY_PROTOCOL = "intentir.replay-semantic-query.v1"


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _contains_mask(value: Any) -> bool:
    if value == "<masked>":
        return True
    if isinstance(value, Mapping):
        return any(_contains_mask(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mask(item) for item in value)
    return False


def _intent_function(j: Mapping[str, Any]) -> Mapping[str, Any]:
    intent = _object(j.get("intent"), "J.intent")
    return _object(
        intent.get("intent_function", intent),
        "J.intent.intent_function",
    )


def _node_stream(j: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten J lexical structure while retaining every authored node field."""

    function = _intent_function(j)
    node_ids = _object(j.get("node_ids", {}), "J.node_ids")
    path_to_node = {str(path): str(node) for node, path in node_ids.items()}
    records: list[dict[str, Any]] = []

    def visit(
        body: Sequence[Any],
        *,
        prefix: str,
        parent_node: str | None,
        lexical_role: str,
        depth: int,
    ) -> None:
        for index, raw in enumerate(body):
            operation = _object(raw, f"J.intent.{prefix}[{index}]")
            path = f"{prefix}[{index}]"
            operation_id = operation.get("id")
            if not isinstance(operation_id, str):
                operation_id = path_to_node.get(path)
            if not isinstance(operation_id, str) and operation.get("op") == "<masked>":
                result = operation.get("result")
                operation_id = result if isinstance(result, str) else None
            if not isinstance(operation_id, str):
                operation_id = f"node@{path}"

            header = {
                str(key): copy.deepcopy(value)
                for key, value in operation.items()
                if key not in {"body", "positive", "negative", "id"}
            }
            arms: dict[str, Any] = {}
            for arm_name in ("positive", "negative"):
                arm = operation.get(arm_name)
                if arm is None:
                    continue
                arm_object = _object(arm, f"J.intent.{path}.{arm_name}")
                arms[arm_name] = {
                    str(key): copy.deepcopy(value)
                    for key, value in arm_object.items()
                    if key != "body"
                }
            if arms:
                header["arms"] = arms
            records.append(
                {
                    "id": operation_id,
                    "path": path,
                    "parent_node": parent_node,
                    "lexical_role": lexical_role,
                    "depth": depth,
                    **header,
                }
            )

            nested = operation.get("body", [])
            if nested:
                visit(
                    _array(nested, f"J.intent.{path}.body"),
                    prefix=f"{path}.body",
                    parent_node=operation_id,
                    lexical_role="body",
                    depth=depth + 1,
                )
            for arm_name in ("positive", "negative"):
                arm = operation.get(arm_name)
                if arm is None:
                    continue
                arm_object = _object(arm, f"J.intent.{path}.{arm_name}")
                visit(
                    _array(arm_object.get("body", []), f"J.intent.{path}.{arm_name}.body"),
                    prefix=f"{path}.{arm_name}.body",
                    parent_node=operation_id,
                    lexical_role=arm_name,
                    depth=depth + 1,
                )

    visit(
        _array(function.get("body", []), "J.intent.body"),
        prefix="body",
        parent_node=None,
        lexical_role="function",
        depth=0,
    )
    return records


def _value_types(
    j: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    *,
    canonical_types: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if canonical_types is not None:
        return [
            {"value": name, **copy.deepcopy(dict(value_type))}
            for name, value_type in canonical_types.items()
        ]

    function = _intent_function(j)
    values: dict[str, dict[str, Any]] = {}
    for role in ("inputs", "outputs"):
        for raw in _array(function.get(role, []), f"J.intent.{role}"):
            tensor = _object(raw, f"J.intent.{role}[]")
            name = tensor.get("name")
            if isinstance(name, str):
                values[name] = {
                    "dtype": tensor.get("dtype"),
                    "shape": copy.deepcopy(tensor.get("shape")),
                    "io_role": role[:-1],
                }
    for node in nodes:
        produced = node.get("result")
        names = [produced] if isinstance(produced, str) else produced if isinstance(produced, list) else []
        node_type = node.get("type", {"dtype": "<masked>", "shape": ["<masked>"]})
        for name in names:
            if isinstance(name, str):
                values[name] = copy.deepcopy(node_type) if isinstance(node_type, Mapping) else {"type": copy.deepcopy(node_type)}
    return [{"value": name, **value} for name, value in values.items()]


def _boundary_node_refs(
    boundary: Mapping[str, Any], node_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in node_ids}
    for relation, value in boundary.items():
        if not isinstance(value, list):
            continue
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                continue
            refs = {
                item
                for key, item in raw.items()
                if (key == "node_ref" or key.endswith("_node_ref"))
                and isinstance(item, str)
                and item in node_ids
            }
            for ref in sorted(refs):
                result[ref].append({"relation": relation, "index": index})
    return {name: rows for name, rows in result.items() if rows}


def _render_text(value: Mapping[str, Any]) -> str:
    lines = [
        f"semantic_program {value['protocol']}",
        f"status {value['status']}",
        "authority read_only_J_semantics; not_target_source; no_C; no_pi; no_Harness_valuation",
        "",
        "signature:",
    ]
    signature = _object(value.get("signature", {}), "S_J.signature")
    for parameter in _array(signature.get("parameters", []), "S_J.signature.parameters"):
        lines.append("  parameter " + _compact(parameter))

    function = _object(value.get("function", {}), "S_J.function")
    lines.append("")
    lines.append(f"function {function.get('name', '<masked>')}:")
    for role in ("inputs", "outputs"):
        for tensor in _array(function.get(role, []), f"S_J.function.{role}"):
            lines.append(f"  {role[:-1]} " + _compact(tensor))
    for node in _array(value.get("nodes", []), "S_J.nodes"):
        node_object = _object(node, "S_J.nodes[]")
        depth = node_object.get("depth", 0)
        indent = "  " + "  " * (depth if isinstance(depth, int) else 0)
        record = {key: item for key, item in node_object.items() if key != "depth"}
        lines.append(indent + "node " + _compact(record))
    for binding in _array(value.get("output_bindings", []), "S_J.output_bindings"):
        lines.append("  return " + _compact(binding))

    boundary = _object(value.get("boundary", {}), "S_J.boundary")
    lines.append("")
    lines.append("boundary:")
    for name, rows in boundary.items():
        if isinstance(rows, Mapping):
            for ref, item in rows.items():
                lines.append(f"  {name} {ref} = " + _compact(item))
        elif isinstance(rows, list):
            for index, item in enumerate(rows):
                lines.append(f"  {name}[{index}] " + _compact(item))
        else:
            lines.append(f"  {name} " + _compact(rows))

    for section in (
        "dimension_constraints",
        "value_dimension_relations",
        "dimension_relations",
        "scopes",
        "storage",
    ):
        lines.append("")
        lines.append(section + ":")
        rows = value.get(section, [])
        if isinstance(rows, list):
            lines.extend("  " + _compact(item) for item in rows)
        else:
            lines.append("  " + _compact(rows))
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ReplaySemanticProgram:
    value: dict[str, Any]
    text: str

    @property
    def masked(self) -> bool:
        return self.value.get("status") == "masked"

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)


def materialize_replay_semantics(j_value: Mapping[str, Any]) -> ReplaySemanticProgram:
    """Materialize one normal or MaskJ source-blind J view without valuation."""

    masked = _contains_mask(j_value)
    canonical_types: dict[str, Any] | None = None
    structured_boundary: dict[str, Any] | None = None
    if masked:
        j = copy.deepcopy(dict(j_value))
    else:
        canonical = elaborate_j(SemanticArchive.from_json(copy.deepcopy(dict(j_value))))
        j = canonical.to_json()
        elaboration = j_elaboration(canonical)
        canonical_types = {
            name: value_type.to_json()
            for name, value_type in elaboration.value_types
        }
        structured_boundary = elaboration.structured_boundary_json()

    function = _intent_function(j)
    nodes = _node_stream(j)
    boundary = copy.deepcopy(dict(_object(j.get("boundary", {}), "J.boundary")))
    node_ids = {str(node["id"]) for node in nodes}
    output_bindings = (
        copy.deepcopy(structured_boundary["output_bindings"])
        if structured_boundary is not None
        else copy.deepcopy(function.get("returns", []))
    )
    value: dict[str, Any] = {
        "protocol": REPLAY_SEMANTIC_PROGRAM_PROTOCOL,
        "status": "masked" if masked else "normal",
        "signature": copy.deepcopy(j.get("signature", {})),
        "function": {
            key: copy.deepcopy(function.get(key, [] if key != "name" else "<masked>"))
            for key in ("name", "inputs", "outputs")
        },
        "nodes": nodes,
        "value_types": _value_types(
            j,
            nodes,
            canonical_types=canonical_types,
        ),
        "boundary": boundary,
        "boundary_refs_by_node": _boundary_node_refs(boundary, node_ids),
        "output_bindings": output_bindings,
        "node_ids": copy.deepcopy(j.get("node_ids", {})),
        "scopes": copy.deepcopy(j.get("scopes", [])),
        "storage": copy.deepcopy(j.get("storage", [])),
        "dimension_constraints": copy.deepcopy(j.get("dimension_constraints", [])),
        "value_dimension_relations": copy.deepcopy(j.get("value_dimension_relations", [])),
        "dimension_relations": copy.deepcopy(j.get("dimension_relations", [])),
        "canonical_boundary": structured_boundary,
        "authority": (
            "deterministic read-only parameterized view of this episode's J only; "
            "no C, pi, Source, Harness valuation, target choice, Gate, generation, "
            "repair, or certification"
        ),
    }
    return ReplaySemanticProgram(value, _render_text(value))


def query_replay_semantics(
    semantic_program: Mapping[str, Any], arguments: Sequence[str]
) -> dict[str, Any]:
    """Return a bounded exact view of one already-materialized ``S_J``."""

    if semantic_program.get("protocol") != REPLAY_SEMANTIC_PROGRAM_PROTOCOL:
        raise ValueError("unsupported Replay semantic-program protocol")
    nodes = _array(semantic_program.get("nodes", []), "S_J.nodes")
    boundary = _object(semantic_program.get("boundary", {}), "S_J.boundary")
    value_types = _array(semantic_program.get("value_types", []), "S_J.value_types")
    if list(arguments) == ["overview"]:
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "overview",
            "semantic_status": semantic_program.get("status"),
            "node_count": len(nodes),
            "value_count": len(value_types),
            "boundary_counts": {
                name: len(value) if isinstance(value, (list, Mapping)) else 1
                for name, value in boundary.items()
            },
            "outputs": semantic_program.get("output_bindings", []),
            "next": "node REF | value REF | boundary NAME START COUNT",
        }
    if len(arguments) == 2 and arguments[0] == "node":
        ref = arguments[1]
        matches = [item for item in nodes if isinstance(item, Mapping) and item.get("id") == ref]
        if len(matches) != 1:
            raise KeyError(f"unknown S_J node {ref!r}")
        relations = semantic_program.get("boundary_refs_by_node", {}).get(ref, [])
        expanded: list[dict[str, Any]] = []
        for relation in relations if isinstance(relations, list) else []:
            if not isinstance(relation, Mapping):
                continue
            name = relation.get("relation")
            index = relation.get("index")
            rows = boundary.get(name) if isinstance(name, str) else None
            if isinstance(rows, list) and isinstance(index, int) and 0 <= index < len(rows):
                expanded.append({"relation": name, "index": index, "value": rows[index]})
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "node",
            "node": matches[0],
            "boundary_relations": expanded,
        }
    if len(arguments) == 2 and arguments[0] == "value":
        ref = arguments[1]
        matches = [item for item in value_types if isinstance(item, Mapping) and item.get("value") == ref]
        if len(matches) != 1:
            raise KeyError(f"unknown S_J value {ref!r}")
        consumers = [
            item.get("id")
            for item in nodes
            if isinstance(item, Mapping)
            and ref in (item.get("operands") if isinstance(item.get("operands"), list) else [])
        ]
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "value",
            "value": matches[0],
            "consumers": consumers,
        }
    if len(arguments) == 4 and arguments[0] == "boundary":
        name = arguments[1]
        start = int(arguments[2])
        count = int(arguments[3])
        if start < 0 or count < 1 or count > 24:
            raise ValueError("boundary START/COUNT is outside the closed bound")
        rows = boundary.get(name)
        if isinstance(rows, Mapping):
            values = [{"id": key, "value": value} for key, value in rows.items()]
        elif isinstance(rows, list):
            values = list(rows)
        else:
            raise KeyError(f"unknown or scalar S_J boundary relation {name!r}")
        items = values[start : start + count]
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "boundary",
            "relation": name,
            "total": len(values),
            "start": start,
            "items": items,
            "next": (
                f"boundary {name} {start + len(items)} {count}"
                if start + len(items) < len(values)
                else None
            ),
        }
    raise ValueError(
        "usage: overview | node REF | value REF | boundary NAME START COUNT"
    )


__all__ = [
    "REPLAY_SEMANTIC_PROGRAM_PROTOCOL",
    "REPLAY_SEMANTIC_QUERY_PROTOCOL",
    "ReplaySemanticProgram",
    "materialize_replay_semantics",
    "query_replay_semantics",
]
