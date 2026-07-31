"""Exact, read-only compiler view of pi for one source-blind Replay episode.

The view resolves opaque occurrence and region handles next to the relations that
already mention them and may colocate the occurrence's exact local S_J anchor.  This
is a mechanical cross-index over already-mounted inputs: it does not infer a semantic
edge, map a relation to a target statement, choose a spelling, or add a realization
fact.  ``pi.json`` and ``S_J`` remain the separate authorities.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping


REPLAY_REALIZATION_VIEW_PROTOCOL = "intentir.replay-realization-view.v2"


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ReplayRealizationView:
    pi: dict[str, Any]
    text: str

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.pi)


def _object_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _references(value: Any, known: set[str], *, path: str = "") -> list[tuple[str, str]]:
    """Return exact paths to already-known pi handles without interpreting them."""

    result: list[tuple[str, str]] = []
    if isinstance(value, str):
        if value in known:
            result.append((path, value))
        return result
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            result.extend(_references(item, known, path=child))
        return result
    if isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_references(item, known, path=f"{path}[{index}]"))
    return result


def _closed_denotation(row_path: str, row: Any) -> str | None:
    """Spell only protocol-fixed pi denotations, never a frontend implementation."""

    if not isinstance(row, Mapping):
        return None
    if row_path.startswith("program_axes["):
        return (
            f"domain_coordinate({row.get('domain_ref')}) reads physical_axis({row.get('axis')})"
        )
    if row_path.startswith("program_linearizations["):
        dimensions = row.get("dimension_refs")
        if isinstance(dimensions, list) and all(isinstance(item, str) for item in dimensions):
            return (
                f"row_major_decode domain_coordinate({row.get('domain_ref')}) into "
                f"logical_coordinates({_compact(dimensions)}); component[i] = "
                "(ordinal // product(extent[i+1:])) % extent[i]"
            )
    return None


def _semantic_anchors(semantic_program: Mapping[str, Any] | None) -> dict[str, Any]:
    """Index exact local S_J records; never infer a missing pi or J relation."""

    if semantic_program is None:
        return {}
    result: dict[str, Any] = {}
    signature = semantic_program.get("signature")
    if isinstance(signature, Mapping):
        parameters = signature.get("parameters")
        for item in parameters if isinstance(parameters, list) else []:
            if not isinstance(item, Mapping):
                continue
            anchor = {"kind": "signature_parameter", "exact": copy.deepcopy(dict(item))}
            for key in ("name", "logical_ref"):
                ref = item.get(key)
                if isinstance(ref, str):
                    result.setdefault(ref, anchor)

    function = semantic_program.get("function")
    tensor_types: dict[str, Any] = {}
    if isinstance(function, Mapping):
        for role in ("inputs", "outputs"):
            rows = function.get(role)
            for item in rows if isinstance(rows, list) else []:
                if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                    continue
                ref = str(item["name"])
                tensor_types[ref] = copy.deepcopy(dict(item))
                result.setdefault(
                    ref,
                    {"kind": f"function_{role[:-1]}", "exact": tensor_types[ref]},
                )

    values = semantic_program.get("value_types")
    for item in values if isinstance(values, list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("value"), str):
            result.setdefault(
                str(item["value"]),
                {"kind": "typed_value", "exact": copy.deepcopy(dict(item))},
            )

    nodes = semantic_program.get("nodes")
    for item in nodes if isinstance(nodes, list) else []:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            continue
        summary = {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if key != "depth"
        }
        result[str(item["id"])] = {"kind": "semantic_node", "exact": summary}

    boundary = semantic_program.get("boundary")
    if isinstance(boundary, Mapping):
        for relation, rows in boundary.items():
            if isinstance(rows, Mapping):
                for ref, item in rows.items():
                    if isinstance(ref, str):
                        result.setdefault(
                            ref,
                            {"kind": f"boundary.{relation}", "exact": copy.deepcopy(item)},
                        )
                continue
            for item in rows if isinstance(rows, list) else []:
                if not isinstance(item, Mapping):
                    continue
                refs = [
                    value
                    for key, value in item.items()
                    if isinstance(value, str) and (key == "id" or key.endswith("_id"))
                ]
                for ref in refs:
                    anchor: dict[str, Any] = {
                        "kind": f"boundary.{relation}",
                        "exact": copy.deepcopy(dict(item)),
                    }
                    tensor = item.get("tensor")
                    if isinstance(tensor, str) and tensor in tensor_types:
                        anchor["tensor_type"] = copy.deepcopy(tensor_types[tensor])
                    result.setdefault(ref, anchor)
    return result


def materialize_replay_realization(
    pi_value: Mapping[str, Any],
    *,
    semantic_program: Mapping[str, Any] | None = None,
) -> ReplayRealizationView:
    """Render exact pi once, with opaque-handle and local S_J-anchor indices."""

    pi = copy.deepcopy(dict(pi_value))
    regions = _object_rows(pi.get("regions"))
    instances = _object_rows(pi.get("instances"))
    region_by_id = {
        str(item["id"]): item
        for item in regions
        if isinstance(item.get("id"), str)
    }
    instance_by_id = {
        str(item["id"]): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    instances_by_semantic: dict[str, list[str]] = {}
    for instance_id, item in instance_by_id.items():
        semantic_ref = item.get("semantic_ref")
        if isinstance(semantic_ref, str):
            instances_by_semantic.setdefault(semantic_ref, []).append(instance_id)
    anchors = _semantic_anchors(semantic_program)

    known = set(region_by_id) | set(instance_by_id) | set(instances_by_semantic)
    relation_rows: list[tuple[str, Any]] = []
    for section, value in pi.items():
        if section in {"regions", "instances"}:
            continue
        if isinstance(value, list):
            rows = [(f"{section}[{index}]", item) for index, item in enumerate(value)]
        elif isinstance(value, Mapping):
            rows = [(f"{section}.{name}", item) for name, item in value.items()]
        else:
            rows = [(section, value)]
        relation_rows.extend(rows)

    def resolved(handle: str) -> str:
        instance = instance_by_id.get(handle)
        if instance is not None:
            region_id = str(instance.get("region"))
            region = region_by_id.get(region_id, {})
            return (
                f"{handle}<{instance.get('semantic_ref')}:{instance.get('mechanism')}"
                f"@{region_id}/{region.get('placement')}>"
            )
        region = region_by_id.get(handle)
        if region is not None:
            return f"{handle}<{region.get('placement')}@{region.get('parent')}>"
        anchors = instances_by_semantic.get(handle)
        if anchors is not None:
            return f"{handle}<semantic:{','.join(anchors)}>"
        raise KeyError(handle)

    lines = [
        f"realization_view {REPLAY_REALIZATION_VIEW_PROTOCOL}",
        (
            "authority exact_read_only_pi; attached_S_J_anchors_are_exact_local_index_only; "
            "no_C_value; no_Source; no_target_recipe; no_generation; no_repair; no_Gate"
        ),
        "layout regions_then_occurrences_then_exact_relations; refs are a mechanical opaque-handle index",
        "",
        "regions_and_occurrences:",
    ]
    emitted_anchors: set[str] = set()
    for region in regions:
        region_id = str(region.get("id"))
        lines.append(f"  region exact=" + _compact(region))
        for instance in instances:
            if instance.get("region") != region_id:
                continue
            semantic_ref = instance.get("semantic_ref")
            anchor = anchors.get(semantic_ref) if isinstance(semantic_ref, str) else None
            include_anchor = (
                anchor is not None
                and isinstance(semantic_ref, str)
                and semantic_ref not in emitted_anchors
            )
            lines.append(
                "    occurrence exact="
                + _compact(instance)
                + (" S_J_anchor=" + _compact(anchor) if include_anchor else "")
            )
            if include_anchor:
                emitted_anchors.add(semantic_ref)

    lines.extend(["", "exact_relations:"])
    for row_path, row in relation_rows:
        annotations = [
            f"{ref_path}={resolved(handle)}"
            for ref_path, handle in _references(row, known)
        ]
        denotation = _closed_denotation(row_path, row)
        lines.append(
            f"  {row_path} exact="
            + _compact(row)
            + (" refs=[" + ";".join(annotations) + "]" if annotations else "")
            + (" denotation=" + denotation if denotation is not None else "")
        )
    return ReplayRealizationView(pi, "\n".join(lines) + "\n")


__all__ = [
    "REPLAY_REALIZATION_VIEW_PROTOCOL",
    "ReplayRealizationView",
    "materialize_replay_realization",
]
