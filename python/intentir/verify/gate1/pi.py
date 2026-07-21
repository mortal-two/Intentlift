"""Strict LLM-direct realization witness (pi).

Pi is not an executable plan.  It carries only anonymous source-realization
topology and opaque references into J/C.  Closed record schemas deliberately
reject semantic atoms, source text and frontend API spellings.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar


class RealizationWitnessError(ValueError):
    """Raised when pi contains malformed or semantic-bearing data."""


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _record(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RealizationWitnessError(f"{path} must be an object")
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown:
        raise RealizationWitnessError(f"{path} has forbidden fields: {unknown}")
    if missing:
        raise RealizationWitnessError(f"{path} is missing fields: {missing}")
    return value


def _name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealizationWitnessError(f"{path} must be a non-empty string")
    return value


def _opaque_anchor_ref(value: Any, path: str) -> str:
    result = _name(value, path)
    if re.fullmatch(r"(?:t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+", result) is None:
        raise RealizationWitnessError(
            f"{path} must be an anonymous declaration-order handle"
        )
    return result


def _declared_instance_ref(
    value: Any,
    *,
    path: str,
    known_instances: set[str],
) -> str:
    result = _name(value, path)
    if re.fullmatch(r"i[0-9]+", result) is None:
        raise RealizationWitnessError(
            f"{path} must reference a declared pi instance id i*; got {result!r}"
        )
    if result not in known_instances:
        raise RealizationWitnessError(
            f"{path} references undeclared pi instance {result!r}"
        )
    return result


def _declared_instance_refs(
    value: Any,
    *,
    path: str,
    known_instances: set[str],
) -> list[str]:
    if not isinstance(value, list):
        raise RealizationWitnessError(f"{path} must be an array")
    result = [
        _declared_instance_ref(
            item,
            path=f"{path}[{index}]",
            known_instances=known_instances,
        )
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise RealizationWitnessError(f"{path} must not contain duplicates")
    return result


def render_realization_witness_wire_grammar() -> str:
    """Return the single model-visible grammar for pi graph references."""

    return (
        "Pi has exactly two reference roles. instances[k].semantic_ref is an "
        "anchor into a declared anonymous J/C/interface handle; it is not a pi "
        "graph node. instances[k].id declares the only pi graph-node namespace "
        "i0,i1,... in declaration order. Every routes source/target, order "
        "before/after, and sharing source/consumer must equal one already "
        "declared instance id i*. Edges never use raw t*/n*/a*/d*/e*/q*/b*/"
        "s*/v*/p*/c*/x* handles and never declare instances implicitly. To "
        "represent a p0 dependency on access a0, declare i0 anchored by p0 and "
        "i1 anchored by a0, then route i0 to i1; routing p0 to i1 is invalid. "
        "Multiple instances may share one semantic_ref when the source has "
        "multiple compiler-visible stages. Route slots are distinct, "
        "non-negative, target-local opaque input ordinals."
    )


def _require_acyclic(
    nodes: set[str], edges: list[tuple[str, str]], path: str
) -> None:
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise RealizationWitnessError(f"{path} must be acyclic")


@dataclass
class RealizationWitness:
    """Minimal direct-pi schema: region/instance/route/slot/order/share."""

    regions: list[dict[str, JsonValue]] = field(default_factory=list)
    instances: list[dict[str, JsonValue]] = field(default_factory=list)
    routes: list[dict[str, JsonValue]] = field(default_factory=list)
    order: list[dict[str, JsonValue]] = field(default_factory=list)
    sharing: list[dict[str, JsonValue]] = field(default_factory=list)

    PLACEMENTS: ClassVar[frozenset[str]] = frozenset(
        {"root", "program", "loop", "branch"}
    )
    MECHANISMS: ClassVar[frozenset[str]] = frozenset(
        {
            "realize",
            "transfer_direct",
            "transfer_async",
            "update_atomic",
            "update_privatized",
            "synchronize",
        }
    )

    def __post_init__(self) -> None:
        checked_regions: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.regions):
            item = _record(
                raw,
                path=f"pi.regions[{index}]",
                required=frozenset({"id", "parent", "placement"}),
            )
            region_id = _name(item["id"], f"pi.regions[{index}].id")
            if region_id != f"r{index}":
                raise RealizationWitnessError(
                    "pi region ids must be declaration-order handles r0, r1, ..."
                )
            parent = item["parent"]
            if parent is not None:
                parent = _name(parent, f"pi.regions[{index}].parent")
            placement = _name(item["placement"], f"pi.regions[{index}].placement")
            if placement not in self.PLACEMENTS:
                raise RealizationWitnessError(f"unsupported pi placement {placement!r}")
            checked_regions.append(
                {"id": region_id, "parent": parent, "placement": placement}
            )
        region_ids = [item["id"] for item in checked_regions]
        if len(region_ids) != len(set(region_ids)):
            raise RealizationWitnessError("pi region ids must be unique")
        known_regions = set(region_ids)
        for item in checked_regions:
            if item["parent"] is not None and item["parent"] not in known_regions:
                raise RealizationWitnessError("pi region parent is unknown")
            if item["parent"] == item["id"]:
                raise RealizationWitnessError("pi region cannot parent itself")
        _require_acyclic(
            known_regions,
            [
                (item["parent"], item["id"])
                for item in checked_regions
                if item["parent"] is not None
            ],
            "pi region tree",
        )

        checked_instances: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.instances):
            item = _record(
                raw,
                path=f"pi.instances[{index}]",
                required=frozenset({"id", "semantic_ref", "region", "mechanism"}),
            )
            instance_id = _name(item["id"], f"pi.instances[{index}].id")
            if instance_id != f"i{index}":
                raise RealizationWitnessError(
                    "pi instance ids must be declaration-order handles i0, i1, ..."
                )
            semantic_ref = _opaque_anchor_ref(
                item["semantic_ref"], f"pi.instances[{index}].semantic_ref"
            )
            region = _name(item["region"], f"pi.instances[{index}].region")
            mechanism = _name(item["mechanism"], f"pi.instances[{index}].mechanism")
            if region not in known_regions:
                raise RealizationWitnessError("pi instance region is unknown")
            if mechanism not in self.MECHANISMS:
                raise RealizationWitnessError(
                    f"pi mechanism {mechanism!r} is not source-neutral"
                )
            checked_instances.append(
                {
                    "id": instance_id,
                    "semantic_ref": semantic_ref,
                    "region": region,
                    "mechanism": mechanism,
                }
            )
        instance_ids = [item["id"] for item in checked_instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise RealizationWitnessError("pi instance ids must be unique")
        known_instances = set(instance_ids)

        checked_routes: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.routes):
            item = _record(
                raw,
                path=f"pi.routes[{index}]",
                required=frozenset({"source", "target", "slots"}),
            )
            source = _declared_instance_ref(
                item["source"],
                path=f"pi.routes[{index}].source",
                known_instances=known_instances,
            )
            target = _declared_instance_ref(
                item["target"],
                path=f"pi.routes[{index}].target",
                known_instances=known_instances,
            )
            slots = item["slots"]
            if not isinstance(slots, list):
                raise RealizationWitnessError("pi route slots must be an array")
            checked_slots: list[int] = []
            for slot in slots:
                if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                    raise RealizationWitnessError("pi route slots must be non-negative integers")
                checked_slots.append(slot)
            if len(checked_slots) != len(set(checked_slots)):
                raise RealizationWitnessError("pi route slots must be unique")
            checked_routes.append(
                {"source": source, "target": target, "slots": checked_slots}
            )
        route_keys = [
            (item["source"], item["target"], tuple(item["slots"]))
            for item in checked_routes
        ]
        if len(route_keys) != len(set(route_keys)):
            raise RealizationWitnessError("pi routes must be unique")

        checked_order: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.order):
            item = _record(
                raw,
                path=f"pi.order[{index}]",
                required=frozenset({"before", "after"}),
            )
            before = _declared_instance_ref(
                item["before"],
                path=f"pi.order[{index}].before",
                known_instances=known_instances,
            )
            after = _declared_instance_ref(
                item["after"],
                path=f"pi.order[{index}].after",
                known_instances=known_instances,
            )
            if before == after:
                raise RealizationWitnessError("pi order cannot contain a self edge")
            checked_order.append({"before": before, "after": after})
        pairs = [(item["before"], item["after"]) for item in checked_order]
        if len(pairs) != len(set(pairs)):
            raise RealizationWitnessError("pi order edges must be unique")
        _require_acyclic(known_instances, pairs, "pi order")

        checked_sharing: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.sharing):
            item = _record(
                raw,
                path=f"pi.sharing[{index}]",
                required=frozenset({"source", "consumers"}),
            )
            source = _declared_instance_ref(
                item["source"],
                path=f"pi.sharing[{index}].source",
                known_instances=known_instances,
            )
            consumers = _declared_instance_refs(
                item["consumers"],
                path=f"pi.sharing[{index}].consumers",
                known_instances=known_instances,
            )
            if not consumers:
                raise RealizationWitnessError("pi sharing consumers must not be empty")
            checked_sharing.append({"source": source, "consumers": consumers})
        sharing_sources = [item["source"] for item in checked_sharing]
        if len(sharing_sources) != len(set(sharing_sources)):
            raise RealizationWitnessError("pi sharing source must be unique")

        self.regions = checked_regions
        self.instances = checked_instances
        self.routes = checked_routes
        self.order = checked_order
        self.sharing = checked_sharing

    @classmethod
    def from_json(cls, value: Any) -> "RealizationWitness":
        data = _record(
            value,
            path="pi",
            required=frozenset({"regions", "instances", "routes", "order", "sharing"}),
        )
        for key in ("regions", "instances", "routes", "order", "sharing"):
            if not isinstance(data[key], list):
                raise RealizationWitnessError(f"pi.{key} must be an array")
        return cls(**copy.deepcopy(data))

    def to_json(self) -> dict[str, Any]:
        return {
            "regions": copy.deepcopy(self.regions),
            "instances": copy.deepcopy(self.instances),
            "routes": copy.deepcopy(self.routes),
            "order": copy.deepcopy(self.order),
            "sharing": copy.deepcopy(self.sharing),
        }


__all__ = [
    "RealizationWitness",
    "RealizationWitnessError",
    "render_realization_witness_wire_grammar",
]
