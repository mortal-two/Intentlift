"""Target-neutral Replay skeletons derived only from canonical J/C/pi.

The skeleton is a source-blind authoring surface.  It joins the semantic graph and
realization witness so a Replay agent can author compact target bindings instead of
rewriting one target statement per physical occurrence.  This module deliberately
contains no Intent-op-to-target mapping and emits no target source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REPLAY_SKELETON_PROTOCOL = "intentir.replay-typed-skeleton.v1"
REPLAY_BINDING_PROTOCOL = "intentir.replay-target-binding.v1"


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _flatten_operations(
    body: Sequence[Any],
    *,
    prefix: str = "body",
) -> list[dict[str, Any]]:
    """Flatten authored J operations while retaining exact structured paths."""

    result: list[dict[str, Any]] = []
    for index, raw in enumerate(body):
        operation = _object(raw, f"J.intent.{prefix}[{index}]")
        path = f"{prefix}[{index}]"
        record = {
            "path": path,
            **{
                str(key): value
                for key, value in operation.items()
                if key not in {"body", "positive", "negative"}
            },
        }
        nested = operation.get("body", [])
        if nested:
            record["body_operation_count"] = len(
                _array(nested, f"J.intent.{path}.body")
            )
        for arm in ("positive", "negative"):
            arm_value = operation.get(arm)
            if arm_value is None:
                continue
            arm_object = _object(arm_value, f"J.intent.{path}.{arm}")
            arm_body = _array(
                arm_object.get("body", []), f"J.intent.{path}.{arm}.body"
            )
            record[arm] = {
                **{str(key): value for key, value in arm_object.items() if key != "body"},
                "body_operation_count": len(arm_body),
            }
        result.append(record)
        result.extend(
            _flatten_operations(
                _array(nested, f"J.intent.{path}.body"),
                prefix=f"{path}.body",
            )
        )
        for arm in ("positive", "negative"):
            arm_value = operation.get(arm)
            if arm_value is None:
                continue
            arm_object = _object(arm_value, f"J.intent.{path}.{arm}")
            result.extend(
                _flatten_operations(
                    _array(
                        arm_object.get("body", []),
                        f"J.intent.{path}.{arm}.body",
                    ),
                    prefix=f"{path}.{arm}.body",
                )
            )
    return result


def _semantic_index(
    j: Mapping[str, Any], c: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    intent_container = _object(j.get("intent"), "J.intent")
    intent = _object(
        intent_container.get("intent_function", intent_container),
        "J.intent.intent_function",
    )
    operations = _flatten_operations(_array(intent.get("body", []), "J.intent.body"))
    node_ids = _object(j.get("node_ids", {}), "J.node_ids")
    path_to_node = {str(path): str(node) for node, path in node_ids.items()}
    index: dict[str, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    for operation in operations:
        path = str(operation["path"])
        node = path_to_node.get(path)
        if node is None:
            raise ValueError(f"J operation {path} has no canonical node id")
        record = {"id": node, **operation}
        nodes.append(record)
        index[node] = {"kind": "intent_op", **record}
        produced = operation.get("result")
        produced_values = (
            [produced]
            if isinstance(produced, str)
            else produced
            if isinstance(produced, list)
            else []
        )
        for value_ref in produced_values:
            if isinstance(value_ref, str):
                index[value_ref] = {
                    "kind": "intent_value",
                    "id": value_ref,
                    "producer": node,
                    "type": operation.get("type"),
                }

    for role in ("inputs", "outputs"):
        for raw in _array(intent.get(role, []), f"J.intent.{role}"):
            value = _object(raw, f"J.intent.{role}[]")
            name = value.get("name")
            if isinstance(name, str):
                index[name] = {
                    "kind": "intent_value",
                    "id": name,
                    "io_role": role[:-1],
                    **dict(value),
                }

    boundary = _object(j.get("boundary"), "J.boundary")
    for relation_name, id_field, kind in (
        ("accesses", "access_id", "boundary_access"),
        ("effects", "effect_id", "boundary_effect"),
    ):
        for raw in _array(boundary.get(relation_name, []), f"J.boundary.{relation_name}"):
            relation = _object(raw, f"J.boundary.{relation_name}[]")
            relation_id = relation.get(id_field)
            if isinstance(relation_id, str):
                index[relation_id] = {
                    "kind": kind,
                    "id": relation_id,
                    **dict(relation),
                }
    domains = _object(boundary.get("domains", {}), "J.boundary.domains")
    for domain_ref, expression in domains.items():
        index[str(domain_ref)] = {
            "kind": "domain",
            "id": str(domain_ref),
            "expression": expression,
        }

    signature = _object(j.get("signature"), "J.signature")
    for raw in _array(signature.get("parameters", []), "J.signature.parameters"):
        parameter = _object(raw, "J.signature.parameters[]")
        name = parameter.get("name")
        if isinstance(name, str):
            index[name] = {
                "kind": "signature_parameter",
                "id": name,
                **dict(parameter),
            }
    for raw in _array(c.get("slots", []), "C.slots"):
        slot = _object(raw, "C.slots[]")
        name = slot.get("name")
        if isinstance(name, str):
            index[name] = {"kind": "c_slot", "id": name, **dict(slot)}
    return nodes, index


def _relations(pi: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        dict(_object(item, f"pi.{name}[]"))
        for item in _array(pi.get(name, []), f"pi.{name}")
    ]


def _binding_kind(semantic: Mapping[str, Any]) -> str | None:
    kind = semantic.get("kind")
    if kind == "intent_op":
        op = semantic.get("op")
        if op in {"loop", "effect_if", "value_if"}:
            return "semantic_control"
        return "semantic_expression"
    if kind == "boundary_access":
        return "boundary_access"
    if kind == "boundary_effect":
        return "boundary_effect"
    if kind == "domain":
        return "domain_predicate"
    return None


@dataclass(frozen=True)
class ReplaySkeleton:
    value: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return self.value


def build_replay_skeleton(packet: Mapping[str, Any]) -> ReplaySkeleton:
    """Build a target-neutral skeleton from one canonical source-blind packet."""

    if set(packet) != {"j", "c", "pi", "u_f"}:
        raise ValueError("Replay packet must contain exactly j, c, pi, and u_f")
    j = _object(packet["j"], "J")
    c = _object(packet["c"], "C")
    pi = _object(packet["pi"], "pi")
    nodes, semantic_index = _semantic_index(j, c)

    regions = _relations(pi, "regions")
    region_index = {
        str(item.get("id")): item
        for item in regions
        if isinstance(item.get("id"), str)
    }
    if len(region_index) != len(regions):
        raise ValueError("pi.regions contains duplicate or malformed ids")
    instances = _relations(pi, "instances")
    instance_index = {
        str(item.get("id")): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    if len(instance_index) != len(instances):
        raise ValueError("pi.instances contains duplicate or malformed ids")

    occurrences: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {
        semantic_ref: {
            "semantic_ref": semantic_ref,
            "occurrences": [],
        }
        for semantic_ref, semantic in semantic_index.items()
        if _binding_kind(semantic) is not None
    }
    for instance in instances:
        instance_id = str(instance["id"])
        semantic_ref = str(instance.get("semantic_ref"))
        region_ref = str(instance.get("region"))
        if region_ref not in region_index:
            raise ValueError(f"pi instance {instance_id} references unknown region")
        semantic = semantic_index.get(
            semantic_ref,
            {"kind": "opaque_realization_anchor", "id": semantic_ref},
        )
        occurrence = dict(instance)
        occurrences.append(occurrence)
        group = groups.setdefault(
            semantic_ref,
            {
                "semantic_ref": semantic_ref,
                "occurrences": [],
            },
        )
        group["occurrences"].append(instance_id)

    routes = _relations(pi, "routes")
    order = _relations(pi, "order")
    sharing = _relations(pi, "sharing")
    for route_index, route in enumerate(routes):
        source = route.get("source")
        target = route.get("target")
        if source not in instance_index or target not in instance_index:
            raise ValueError(f"pi.routes[{route_index}] has an unknown endpoint")

    region_worksets: list[dict[str, Any]] = []
    for region in regions:
        region_id = str(region["id"])
        region_occurrence_ids = [
            str(item["id"])
            for item in occurrences
            if item.get("region") == region_id
        ]
        region_occurrence_set = set(region_occurrence_ids)
        region_worksets.append(
            {
                "region_ref": region_id,
                "child_region_refs": [
                    item["id"] for item in regions if item.get("parent") == region_id
                ],
                "occurrence_ids": region_occurrence_ids,
                "route_indices": [
                    index
                    for index, route in enumerate(routes)
                    if route.get("source") in region_occurrence_set
                    or route.get("target") in region_occurrence_set
                ],
                "order_indices": [
                    index
                    for index, relation in enumerate(order)
                    if relation.get("before") in region_occurrence_set
                    or relation.get("after") in region_occurrence_set
                ],
            }
        )

    binding_holes: list[dict[str, Any]] = []
    for semantic_ref, group in groups.items():
        semantic = semantic_index.get(semantic_ref)
        if semantic is None:
            continue
        binding_kind = _binding_kind(semantic)
        if binding_kind is None:
            continue
        binding_holes.append(
            {
                "id": f"h{len(binding_holes)}",
                "kind": binding_kind,
                "semantic_ref": semantic_ref,
                "occurrences": group["occurrences"],
                "required_binding": True,
            }
        )

    realization_relation_names = (
        "program_axes",
        "program_linearizations",
        "iterations",
        "loop_carriers",
        "physical_iterations",
        "physical_loop_carriers",
        "terminal_controls",
        "route_extent_transforms",
    )
    realization_relations = {
        name: _relations(pi, name) for name in realization_relation_names
    }
    realization_holes: list[dict[str, Any]] = []
    for relation_name, rows in realization_relations.items():
        for relation_index, relation in enumerate(rows):
            realization_holes.append(
                {
                    "id": f"rh{len(realization_holes)}",
                    "kind": "realization_binding",
                    "relation": relation_name,
                    "relation_index": relation_index,
                    "authored_relation": relation,
                    "required_binding": True,
                }
            )

    intent_container = _object(j["intent"], "J.intent")
    intent_function = _object(
        intent_container.get("intent_function", intent_container),
        "J.intent.intent_function",
    )

    skeleton = {
        "protocol": REPLAY_SKELETON_PROTOCOL,
        "binding_protocol": REPLAY_BINDING_PROTOCOL,
        "semantic": {
            "signature": j.get("signature"),
            "intent_io": {
                key: intent_function.get(key)
                for key in ("name", "inputs", "outputs", "returns")
                if key in intent_function
            },
            "nodes": nodes,
            "boundary": j.get("boundary"),
            "dimension_constraints": j.get("dimension_constraints", []),
            "value_dimension_relations": j.get("value_dimension_relations", []),
            "dimension_relations": j.get("dimension_relations", []),
        },
        "realization": {
            "regions": regions,
            "occurrences": occurrences,
            "routes": routes,
            "order": order,
            "sharing": sharing,
            **realization_relations,
        },
        "semantic_groups": list(groups.values()),
        "region_worksets": region_worksets,
        "holes": {
            "semantic": binding_holes,
            "realization": realization_holes,
        },
        "masked_c": c,
        "public_frontend": packet["u_f"],
        "authority": (
            "exact target-neutral J/C/pi reindex only; no Source, Harness value, "
            "Gate evidence, target API, instruction selection, generation, or repair"
        ),
    }
    return ReplaySkeleton(skeleton)


def _skeleton_semantic_record(
    skeleton: Mapping[str, Any], semantic_ref: str
) -> dict[str, Any] | None:
    semantic = _object(skeleton.get("semantic"), "skeleton.semantic")
    for node in _array(semantic.get("nodes", []), "skeleton.semantic.nodes"):
        if isinstance(node, Mapping) and node.get("id") == semantic_ref:
            return {"kind": "intent_op", **dict(node)}
    boundary = _object(semantic.get("boundary", {}), "skeleton.semantic.boundary")
    for relation_name, id_field, kind in (
        ("accesses", "access_id", "boundary_access"),
        ("effects", "effect_id", "boundary_effect"),
    ):
        for row in _array(
            boundary.get(relation_name, []),
            f"skeleton.semantic.boundary.{relation_name}",
        ):
            if isinstance(row, Mapping) and row.get(id_field) == semantic_ref:
                return {"kind": kind, **dict(row)}
    domains = _object(
        boundary.get("domains", {}), "skeleton.semantic.boundary.domains"
    )
    if semantic_ref in domains:
        return {
            "kind": "domain",
            "id": semantic_ref,
            "expression": domains[semantic_ref],
        }
    signature = _object(semantic.get("signature", {}), "skeleton.semantic.signature")
    for parameter in _array(
        signature.get("parameters", []), "skeleton.semantic.signature.parameters"
    ):
        if isinstance(parameter, Mapping) and parameter.get("name") == semantic_ref:
            return {"kind": "signature_parameter", **dict(parameter)}
    masked_c = _object(skeleton.get("masked_c", {}), "skeleton.masked_c")
    for slot in _array(masked_c.get("slots", []), "skeleton.masked_c.slots"):
        if isinstance(slot, Mapping) and slot.get("name") == semantic_ref:
            return {"kind": "c_slot", **dict(slot)}
    return None


def query_replay_skeleton(
    skeleton: Mapping[str, Any], arguments: Sequence[str]
) -> dict[str, Any]:
    """Return one bounded exact workset from a normalized Replay skeleton."""

    if skeleton.get("protocol") != REPLAY_SKELETON_PROTOCOL:
        raise ValueError("unsupported Replay skeleton protocol")
    holes_value = _object(skeleton.get("holes"), "skeleton.holes")
    holes = [
        dict(item)
        for name in ("semantic", "realization")
        for item in _array(holes_value.get(name, []), f"skeleton.holes.{name}")
        if isinstance(item, Mapping)
    ]
    realization = _object(skeleton.get("realization"), "skeleton.realization")
    regions = _array(realization.get("regions", []), "skeleton.realization.regions")
    occurrences = _array(
        realization.get("occurrences", []), "skeleton.realization.occurrences"
    )
    routes = _array(realization.get("routes", []), "skeleton.realization.routes")
    order = _array(realization.get("order", []), "skeleton.realization.order")

    if list(arguments) == ["overview"]:
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "overview",
            "counts": {
                "semantic_nodes": len(skeleton["semantic"]["nodes"]),
                "semantic_holes": len(holes_value.get("semantic", [])),
                "realization_holes": len(holes_value.get("realization", [])),
                "regions": len(regions),
                "occurrences": len(occurrences),
                "routes": len(routes),
                "order": len(order),
            },
            "section_json_chars": {
                key: len(
                    json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                for key, value in skeleton.items()
            },
            "regions": [
                {
                    "id": item.get("id"),
                    "parent": item.get("parent"),
                    "placement": item.get("placement"),
                    "query": f"region {item.get('id')}",
                }
                for item in regions
                if isinstance(item, Mapping)
            ],
            "first_holes": [
                {
                    "id": item.get("id"),
                    "kind": item.get("kind"),
                    "semantic_ref": item.get("semantic_ref"),
                    "occurrence_count": len(item.get("occurrences", []))
                    if isinstance(item.get("occurrences", []), list)
                    else 0,
                }
                for item in holes[:16]
            ],
            "next": "holes 16 16" if len(holes) > 16 else None,
            "authority": skeleton.get("authority"),
        }
    if len(arguments) == 3 and arguments[0] == "holes":
        start = int(arguments[1])
        count = int(arguments[2])
        if start < 0 or count < 1 or count > 24:
            raise ValueError("skeleton holes START/COUNT is outside the closed bound")
        selected = holes[start : start + count]
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "holes",
            "total": len(holes),
            "start": start,
            "items": selected,
            "next": (
                f"holes {start + len(selected)} {count}"
                if start + len(selected) < len(holes)
                else None
            ),
        }
    if len(arguments) == 2 and arguments[0] == "hole":
        hole_id = arguments[1]
        matches = [item for item in holes if item.get("id") == hole_id]
        if len(matches) != 1:
            raise KeyError(f"unknown skeleton hole {hole_id!r}")
        hole = matches[0]
        occurrence_ids = set(hole.get("occurrences", []))
        local_occurrences = [
            dict(item)
            for item in occurrences
            if isinstance(item, Mapping) and item.get("id") in occurrence_ids
        ]
        semantic_ref = hole.get("semantic_ref")
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "hole",
            "hole": hole,
            "semantic": (
                _skeleton_semantic_record(skeleton, str(semantic_ref))
                if isinstance(semantic_ref, str)
                else None
            ),
            "occurrences": local_occurrences,
            "routes": [
                {"index": index, **dict(route)}
                for index, route in enumerate(routes)
                if isinstance(route, Mapping)
                and (
                    route.get("source") in occurrence_ids
                    or route.get("target") in occurrence_ids
                )
            ],
        }
    if len(arguments) == 2 and arguments[0] == "semantic":
        record = _skeleton_semantic_record(skeleton, arguments[1])
        if record is None:
            raise KeyError(f"unknown skeleton semantic ref {arguments[1]!r}")
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "semantic",
            "semantic": record,
        }
    if len(arguments) == 2 and arguments[0] == "region":
        region_ref = arguments[1]
        worksets = _array(
            skeleton.get("region_worksets", []), "skeleton.region_worksets"
        )
        matches = [
            item
            for item in worksets
            if isinstance(item, Mapping) and item.get("region_ref") == region_ref
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown skeleton region {region_ref!r}")
        workset = dict(matches[0])
        occurrence_ids = set(workset["occurrence_ids"])
        route_indices = set(workset["route_indices"])
        order_indices = set(workset["order_indices"])
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "region",
            "workset": workset,
            "region": next(
                dict(item)
                for item in regions
                if isinstance(item, Mapping) and item.get("id") == region_ref
            ),
            "occurrences": [
                dict(item)
                for item in occurrences
                if isinstance(item, Mapping) and item.get("id") in occurrence_ids
            ],
            "routes": [
                {"index": index, **dict(item)}
                for index, item in enumerate(routes)
                if index in route_indices and isinstance(item, Mapping)
            ],
            "order": [
                {"index": index, **dict(item)}
                for index, item in enumerate(order)
                if index in order_indices and isinstance(item, Mapping)
            ],
            "holes": [
                item
                for item in holes
                if occurrence_ids.intersection(item.get("occurrences", []))
            ],
        }
    if list(arguments) == ["binding_scaffold"]:
        from .replay_compact_binding import build_replay_compact_plan

        plan = build_replay_compact_plan(skeleton)
        signature = _object(
            skeleton.get("semantic", {}).get("signature", {}),
            "skeleton.semantic.signature",
        )
        aliases = {
            str(item["logical_ref"]): str(item["name"])
            for item in _array(
                signature.get("parameters", []),
                "skeleton.semantic.signature.parameters",
            )
            if isinstance(item, Mapping)
            and isinstance(item.get("logical_ref"), str)
            and isinstance(item.get("name"), str)
        }
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "binding_scaffold",
            "binding": plan.scaffold(),
            "region_sequences": {
                name: list(items) for name, items in plan.region_sequences.items()
            },
            "control_regions": {
                region: {
                    "semantic_ref": value["semantic_ref"],
                    "op": value["operation"].get("op"),
                    "trip": value["operation"].get("trip"),
                    "carry": value["operation"].get("carry", []),
                    "scan": value["operation"].get("scan", []),
                }
                for region, value in plan.control_regions.items()
            },
            "stable_names": {
                "logical_aliases": aliases,
                "program_coordinates": {
                    str(item.get("domain_ref")): (
                        f"_intentir_pid_{item.get('domain_ref')}"
                    )
                    for item in _array(
                        realization.get("program_axes", []),
                        "skeleton.realization.program_axes",
                    )
                    if isinstance(item, Mapping)
                    and isinstance(item.get("domain_ref"), str)
                },
                "linearized_dimensions": {
                    str(ref): f"_intentir_index_{ref}"
                    for item in _array(
                        realization.get("program_linearizations", []),
                        "skeleton.realization.program_linearizations",
                    )
                    if isinstance(item, Mapping)
                    for ref in item.get("dimension_refs", [])
                    if isinstance(ref, str)
                },
                "semantic_values": "use exact t*/v*/x*/p*/c* handles",
            },
            "next": "block b0" if plan.blocks else None,
            "authority": (
                "mechanically generated editable block layout only; every block "
                "source remains empty and no target computation is selected"
            ),
        }
    if len(arguments) == 2 and arguments[0] == "block":
        from .replay_compact_binding import build_replay_compact_plan

        plan = build_replay_compact_plan(skeleton)
        matches = [item for item in plan.blocks if item["id"] == arguments[1]]
        if len(matches) != 1:
            raise KeyError(f"unknown compact binding block {arguments[1]!r}")
        block = matches[0]
        access_rows = _array(
            _object(skeleton["semantic"].get("boundary", {}), "skeleton.boundary").get(
                "accesses", []
            ),
            "skeleton.boundary.accesses",
        )
        realization_holes = _array(
            holes_value.get("realization", []), "skeleton.holes.realization"
        )
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "block",
            "block": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in block.items()
            },
            "semantic": [
                _skeleton_semantic_record(skeleton, ref)
                for ref in block["semantic_refs"]
            ],
            "accesses": [
                dict(row)
                for row in access_rows
                if isinstance(row, Mapping)
                and any(
                    ref.startswith(f"access:{row.get('access_id')}@")
                    for ref in block["realization_refs"]
                )
            ],
            "realization_relations": [
                dict(hole)
                for hole in realization_holes
                if isinstance(hole, Mapping) and hole.get("id") in block["realization_refs"]
            ],
            "region": next(
                dict(item)
                for item in regions
                if isinstance(item, Mapping) and item.get("id") == block["region"]
            ),
        }
    if list(arguments) == ["binding_shape"]:
        return {
            "protocol": "intentir.replay-typed-skeleton-query.v1",
            "status": "pass",
            "view": "binding_shape",
            "protocol_value": "intentir.replay-compact-binding.v2",
            "root_fields": ["protocol", "blocks"],
            "block_fields": [
                "id",
                "semantic_refs",
                "realization_refs",
                "source_path",
            ],
            "editable_fields": ["the files named by blocks[*].source_path"],
            "source_form": (
                "dedented Triton Python statements for the fixed block region; "
                "use exact stable names reported by binding_scaffold"
            ),
            "authority": (
                "the scaffold fixes provenance/placement/order only; source strings "
                "remain entirely fresh agent-authored target computations"
            ),
        }
    raise ValueError(
        "usage: overview | holes START COUNT | hole ID | semantic REF | region ID | "
        "binding_scaffold | block ID | binding_shape"
    )


__all__ = [
    "REPLAY_BINDING_PROTOCOL",
    "REPLAY_SKELETON_PROTOCOL",
    "ReplaySkeleton",
    "build_replay_skeleton",
    "query_replay_skeleton",
]
