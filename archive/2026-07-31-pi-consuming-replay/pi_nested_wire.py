"""Bijective nested wire for canonical pi realization witnesses.

The wire removes repeated global region/instance ids from the authoring surface.  It
does not omit or infer a realization fact: Decode reconstructs canonical flat pi and
the encoder requires an exact round trip.  Semantic minimization is a separate,
stricter step and must never be confused with this mechanical representation change.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


PI_NESTED_WIRE_PROTOCOL = "intentir.pi-min.nested-wire.v2"
_REGION_ID = re.compile(r"r([0-9]+)")
_INSTANCE_ID = re.compile(r"i([0-9]+)")

_RELATION_COLUMNS: dict[str, tuple[str, ...]] = {
    "routes": ("source", "target", "slots"),
    "order": ("before", "after"),
    "sharing": ("source", "consumers"),
    "refinements": ("instance", "kind", "auth_ref"),
    "program_axes": ("domain_ref", "axis"),
    "program_linearizations": ("domain_ref", "dimension_refs"),
}


def _ordinal(handle: Any, pattern: re.Pattern[str], path: str) -> int:
    if not isinstance(handle, str):
        raise ValueError(f"{path} must be a canonical opaque handle")
    match = pattern.fullmatch(handle)
    if match is None:
        raise ValueError(f"{path} must be a canonical opaque handle")
    return int(match.group(1))


def _object_rows(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{path} must be an object array")
    return list(value)


def encode_pi_nested_wire(pi: Mapping[str, Any]) -> dict[str, Any]:
    """Encode canonical flat pi as a nested region/occurrence authoring wire."""

    regions = _object_rows(pi.get("regions"), "pi.regions")
    instances = _object_rows(pi.get("instances"), "pi.instances")
    region_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(regions):
        region_id = row.get("id")
        if _ordinal(region_id, _REGION_ID, f"pi.regions[{index}].id") != index:
            raise ValueError("pi region ids must be contiguous list ordinals")
        if set(row) != {"id", "parent", "placement"}:
            raise ValueError("pi region row does not match the closed nested wire")
        region_rows[str(region_id)] = row
    instance_rows: dict[str, Mapping[str, Any]] = {}
    by_region: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        name: [] for name in region_rows
    }
    for index, row in enumerate(instances):
        instance_id = row.get("id")
        if _ordinal(instance_id, _INSTANCE_ID, f"pi.instances[{index}].id") != index:
            raise ValueError("pi instance ids must be contiguous list ordinals")
        if set(row) != {"id", "semantic_ref", "mechanism", "region"}:
            raise ValueError("pi instance row does not match the closed nested wire")
        region = row.get("region")
        if region not in region_rows:
            raise ValueError("pi instance references an unknown region")
        instance_rows[str(instance_id)] = row
        by_region[str(region)].append((index, row))

    children: dict[str | None, list[str]] = {}
    for region_id, row in region_rows.items():
        parent = row.get("parent")
        if parent is not None and parent not in region_rows:
            raise ValueError("pi region references an unknown parent")
        children.setdefault(parent, []).append(region_id)
    roots = children.get(None, [])
    if len(roots) != 1:
        raise ValueError("nested pi wire requires exactly one root region")

    def region_node(region_id: str) -> dict[str, Any]:
        row = region_rows[region_id]
        return {
            "ordinal": _ordinal(region_id, _REGION_ID, "pi region id"),
            "placement": row["placement"],
            "occurrences": [
                [ordinal, item["semantic_ref"], item["mechanism"]]
                for ordinal, item in by_region[region_id]
            ],
            "children": [
                region_node(child)
                for child in sorted(
                    children.get(region_id, []),
                    key=lambda name: _ordinal(name, _REGION_ID, "pi child region"),
                )
            ],
        }

    relations: dict[str, Any] = {}
    columns: dict[str, list[str]] = {}
    for name, value in pi.items():
        if name in {"regions", "instances"}:
            continue
        relation_columns = _RELATION_COLUMNS.get(name)
        if relation_columns is None:
            relations[name] = value
            continue
        rows = _object_rows(value, f"pi.{name}")
        encoded_rows: list[list[Any]] = []
        for index, row in enumerate(rows):
            if set(row) != set(relation_columns):
                raise ValueError(f"pi.{name}[{index}] has unexpected fields")
            encoded_rows.append([row[column] for column in relation_columns])
        columns[name] = list(relation_columns)
        relations[name] = encoded_rows
    wire = {
        "protocol": PI_NESTED_WIRE_PROTOCOL,
        "region_tree": region_node(roots[0]),
        "relation_columns": columns,
        "relations": relations,
    }
    if decode_pi_nested_wire(wire) != dict(pi):
        raise RuntimeError("nested pi wire failed exact canonical round trip")
    return wire


def decode_pi_nested_wire(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expand one nested wire into the exact canonical flat pi object."""

    if set(value) != {"protocol", "region_tree", "relation_columns", "relations"}:
        raise ValueError("nested pi wire has unknown or missing root fields")
    if value.get("protocol") != PI_NESTED_WIRE_PROTOCOL:
        raise ValueError("unsupported nested pi wire protocol")
    columns = value.get("relation_columns")
    relations = value.get("relations")
    if not isinstance(columns, Mapping) or not isinstance(relations, Mapping):
        raise TypeError("nested pi relation tables must be objects")
    region_rows: dict[int, dict[str, Any]] = {}
    instance_rows: dict[int, dict[str, Any]] = {}

    def visit(node: Any, parent: str | None) -> None:
        if not isinstance(node, Mapping) or set(node) != {
            "ordinal",
            "placement",
            "occurrences",
            "children",
        }:
            raise ValueError("nested pi region node has a closed-field mismatch")
        ordinal = node.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("nested pi region ordinal must be non-negative")
        if ordinal in region_rows or not isinstance(node.get("placement"), str):
            raise ValueError("nested pi region ordinal/placement is malformed")
        region_id = f"r{ordinal}"
        region_rows[ordinal] = {
            "id": region_id,
            "parent": parent,
            "placement": node["placement"],
        }
        occurrences = node.get("occurrences")
        if not isinstance(occurrences, list):
            raise TypeError("nested pi occurrences must be an array")
        for occurrence in occurrences:
            if (
                not isinstance(occurrence, list)
                or len(occurrence) != 3
                or isinstance(occurrence[0], bool)
                or not isinstance(occurrence[0], int)
                or occurrence[0] < 0
                or not isinstance(occurrence[1], str)
                or not isinstance(occurrence[2], str)
                or occurrence[0] in instance_rows
            ):
                raise ValueError("nested pi occurrence is malformed")
            instance_rows[occurrence[0]] = {
                "id": f"i{occurrence[0]}",
                "semantic_ref": occurrence[1],
                "mechanism": occurrence[2],
                "region": region_id,
            }
        child_nodes = node.get("children")
        if not isinstance(child_nodes, list):
            raise TypeError("nested pi children must be an array")
        for child in child_nodes:
            visit(child, region_id)

    visit(value["region_tree"], None)
    if sorted(region_rows) != list(range(len(region_rows))):
        raise ValueError("nested pi region ordinals must be contiguous")
    if sorted(instance_rows) != list(range(len(instance_rows))):
        raise ValueError("nested pi occurrence ordinals must be contiguous")

    result: dict[str, Any] = {
        "regions": [region_rows[index] for index in range(len(region_rows))],
        "instances": [instance_rows[index] for index in range(len(instance_rows))],
    }
    for name, relation in relations.items():
        relation_columns = columns.get(name)
        if relation_columns is None:
            result[str(name)] = relation
            continue
        if (
            not isinstance(relation_columns, list)
            or not all(isinstance(column, str) for column in relation_columns)
            or tuple(relation_columns) != _RELATION_COLUMNS.get(str(name))
            or not isinstance(relation, list)
        ):
            raise ValueError(f"nested pi relation {name} has invalid columns")
        expanded: list[dict[str, Any]] = []
        for row in relation:
            if not isinstance(row, list) or len(row) != len(relation_columns):
                raise ValueError(f"nested pi relation {name} has malformed rows")
            expanded.append(
                {column: row[index] for index, column in enumerate(relation_columns)}
            )
        result[str(name)] = expanded
    return result


__all__ = [
    "PI_NESTED_WIRE_PROTOCOL",
    "decode_pi_nested_wire",
    "encode_pi_nested_wire",
]
