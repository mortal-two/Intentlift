"""Target-neutral Replay skeletons derived only from canonical J/C/pi.

The skeleton is a source-blind authoring surface.  It joins the semantic graph and
realization witness so a Replay agent can author compact target bindings instead of
rewriting one target statement per physical occurrence.  This module deliberately
contains no Intent-op-to-target mapping and emits no target source.
"""

from __future__ import annotations

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


__all__ = [
    "REPLAY_BINDING_PROTOCOL",
    "REPLAY_SKELETON_PROTOCOL",
    "ReplaySkeleton",
    "build_replay_skeleton",
]
