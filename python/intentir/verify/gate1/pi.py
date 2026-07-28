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

REALIZATION_MECHANISMS: tuple[str, ...] = (
    "realize",
    "address_base",
    "address_lane",
    "transfer_direct",
    "transfer_async",
    "update_atomic",
    "update_privatized",
    "synchronize",
)

ITERATION_BASES: tuple[str, ...] = (
    "extent_step",
    "tile_ordinal",
    "rounded_extent_step",
)

PHYSICAL_ITERATION_DIRECTIONS: tuple[str, ...] = ("forward", "reverse")

REPRESENTATION_REFINEMENTS: tuple[str, ...] = (
    "identity_materialization",
    "reduction_linearization",
    "signed_widen",
    "unsigned_widen",
    "numeric_cast",
    "accumulation_dtype",
    "reassociation",
    "native_atomic",
    "privatized_update",
)


def _record(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
    allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RealizationWitnessError(f"{path} must be an object")
    allowed = required if allowed is None else allowed
    unknown = sorted(set(value) - allowed)
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
        "anchor into a declared anonymous J/C handle; J.signature handles are "
        "ordinary J handles and there is no interface namespace. It is not a pi "
        "graph node. instances[k].id declares the only pi graph-node namespace "
        "i0,i1,... in declaration order. Every routes source/target, order "
        "before/after, and sharing source/consumer must equal one already "
        "declared instance id i*. Edges never use raw t*/n*/a*/d*/e*/q*/b*/"
        "s*/v*/p*/c*/x* handles and never declare instances implicitly. To "
        "represent a p0 dependency on access a0, declare i0 anchored by p0 and "
        "i1 anchored by a0, then route i0 to i1; routing p0 to i1 is invalid. "
        "Multiple instances may share one semantic_ref when the source has "
        "multiple compiler-visible stages. Route slots are distinct, "
        "non-negative, target-local opaque input ordinals. Each iterations "
        "entry relates one loop region to one instance anchored by its J loop "
        "and one instance anchored by the matching C tile slot. Its closed "
        "induction_basis records only whether the physical induction variable "
        "is measured in logical extent units, rounded extent units, or tile "
        "ordinals. The global "
        "source-neutral denotation is closed: extent_step means lower zero, "
        "upper equal to the J loop trip extent, and step equal to the matching "
        "C tile; rounded_extent_step means lower zero, upper equal to "
        "ceil_div(J loop trip extent, matching C tile) multiplied by that C "
        "tile, and step equal to the matching C tile; "
        "tile_ordinal means lower zero, upper equal to ceil_div of that "
        "J trip extent by that matching C tile, and unit step. Every frontend "
        "maps this same relation to its public constructs. The wire entry never "
        "carries a bound expression, value, operator, source fragment, or "
        "frontend API. Each declaration-ordered loop_carriers entry has exactly "
        "region, loop_instance, init, state, next, and result instance refs. "
        "Its global temporal denotation is state[0]=init, "
        "state[k+1]=next[k], result=state[T], including T=0. The temporal "
        "next-to-state edge and loop-exit result edge are not ordinary DAG "
        "routes and never carry an expression or frontend spelling. Each "
        "physical_iterations entry has exactly region, body_instance, "
        "tile_instance, induction_basis, and direction. body_instance anchors "
        "an existing non-loop J reduce/contraction or boundary access/domain with "
        "a closed dependent axis; tile_instance anchors one required positive C "
        "tile attached to that same semantic subject and axis. forward uses the "
        "corresponding zero-origin basis order; reverse "
        "visits that same closed basis in descending order. It never adds a J "
        "loop or carries bounds/formulas. Each physical_loop_carriers entry has "
        "exactly region, init, state, next, and result, with the same temporal "
        "denotation as loop_carriers, but its region is indexed by one "
        "physical_iterations relation rather than a structured J loop."
        " terminal_controls has exactly terminal_instance and continuation_instance: "
        "distinct realize instances on sibling J effect_if arm scopes. It names "
        "which arm terminates and which continues; it carries no predicate, "
        "polarity, return, Source spelling, or API. Each reduction_linearization "
        "refinement has null auth_ref and refines one realize instance whose sole "
        "slot-0 route enters a J reduce realize instance. It denotes full-axis "
        "linearization before that consumer without carrying an axis, operator, "
        "value, or frontend API."
    )


def render_realization_mechanism_grammar() -> str:
    """Return the shared, non-executable meaning of pi mechanism tags."""

    return (
        "Pi mechanism is one closed source-neutral event class. realize is a "
        "generic representation-only stage when no more specific class applies. "
        "One realize instance anchored directly to a J.signature handle or C slot "
        "may be discharged by that parameter binding and its direct routed use; "
        "it does not require an identity assignment. Multiple instances anchored "
        "to the same handle still denote distinct routed use-sites/stages. "
        "address_base is an independently compiler-visible, program-uniform "
        "pointer/base or integer base-address contribution before lane merge; "
        "Lift uses it only when that separate source stage exists. address_lane "
        "is the compiler-visible stage that forms the lane-varying contribution "
        "or final address. "
        "When routes contain address_base -> address_lane, Replay must materialize "
        "two distinct stages in that order and must not algebraically fold them "
        "into one flat vector offset. Pointer-operation count is determined only "
        "by source-neutral carrier routing and the carrier kind declared in "
        "J.signature. If a pointer/tensor carrier directly enters address_base, "
        "that base is a pointer/address operation and its route to address_lane "
        "requires a second pointer/address operation for the lane contribution. "
        "If the pointer/tensor carrier bypasses address_base and enters "
        "address_lane or the final address stage, while address_base receives only "
        "integer, domain, or C contributions, Replay merges the integer base and "
        "lane contributions before exactly one pointer/address operation. The "
        "address_base -> address_lane edge alone never decides between these cases. "
        "transfer_direct and "
        "transfer_async classify "
        "data movement after address formation; update_atomic and "
        "update_privatized classify physical update realization; synchronize "
        "classifies synchronization. Region placement='program' only scopes an "
        "instance to one program; it does not assert that a program/grid "
        "coordinate is read. A compiler-visible program/grid coordinate is a "
        "separate realization stage and requires its own realize instance and "
        "routes. Its source-neutral encoding is a realize instance anchored to "
        "the corresponding J domain in a program region whose scalar dependency "
        "path enters address_base before transfer; a domain/C dependency path "
        "entering address_lane is lane-varying instead. Replay must preserve this "
        "base-versus-lane distinction and U_f maps it to public frontend APIs. "
        "Program-axis identity follows the J domain semantic_ref, not instance "
        "multiplicity: repeated program-coordinate instances on one d* remain "
        "distinct stages of the same axis, and only a distinct domain anchor may "
        "allocate another axis. "
        "Integer coordinate arithmetic is a realize stage; address_base and "
        "address_lane classify where its program-uniform and lane-varying paths "
        "enter address formation rather than relabeling every arithmetic op. If "
        "Source visibly forms a separable program-uniform integer base and a lane "
        "offset, then merges them before its first pointer application, the program "
        "path still enters address_base and routes to address_lane while the pointer "
        "carrier bypasses address_base. Replay preserves both integer stages and "
        "uses one pointer/address operation at the lane/final stage. Only "
        "a Source with no separable program-uniform contribution uses a lone "
        "address_lane stage. "
        "Replay must not invent program_id, a grid offset, or an affine "
        "address stage from region placement, grid convention, or C alone. "
        "Address classes never carry an index/offset "
        "formula, constant, integer width, cast, dtype, shape, vector length, "
        "predicate, fill, frontend API, or logical access semantics."
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
    """Minimal direct-pi schema for source-realization relations."""

    regions: list[dict[str, JsonValue]] = field(default_factory=list)
    instances: list[dict[str, JsonValue]] = field(default_factory=list)
    routes: list[dict[str, JsonValue]] = field(default_factory=list)
    order: list[dict[str, JsonValue]] = field(default_factory=list)
    sharing: list[dict[str, JsonValue]] = field(default_factory=list)
    refinements: list[dict[str, JsonValue]] = field(default_factory=list)
    iterations: list[dict[str, JsonValue]] = field(default_factory=list)
    loop_carriers: list[dict[str, JsonValue]] = field(default_factory=list)
    physical_iterations: list[dict[str, JsonValue]] = field(default_factory=list)
    physical_loop_carriers: list[dict[str, JsonValue]] = field(default_factory=list)
    terminal_controls: list[dict[str, JsonValue]] = field(default_factory=list)

    PLACEMENTS: ClassVar[frozenset[str]] = frozenset(
        {"root", "program", "loop", "branch"}
    )
    MECHANISMS: ClassVar[frozenset[str]] = frozenset(REALIZATION_MECHANISMS)
    ITERATION_BASES: ClassVar[frozenset[str]] = frozenset(ITERATION_BASES)
    PHYSICAL_ITERATION_DIRECTIONS: ClassVar[frozenset[str]] = frozenset(
        PHYSICAL_ITERATION_DIRECTIONS
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

        checked_terminal_controls: list[dict[str, JsonValue]] = []
        terminal_control_instances: set[str] = set()
        for index, raw in enumerate(self.terminal_controls):
            path = f"pi.terminal_controls[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {"terminal_instance", "continuation_instance"}
                ),
            )
            terminal_instance = _declared_instance_ref(
                item["terminal_instance"],
                path=f"{path}.terminal_instance",
                known_instances=known_instances,
            )
            continuation_instance = _declared_instance_ref(
                item["continuation_instance"],
                path=f"{path}.continuation_instance",
                known_instances=known_instances,
            )
            if terminal_instance == continuation_instance:
                raise RealizationWitnessError(
                    "pi terminal control requires distinct terminal and "
                    "continuation instances"
                )
            if (
                terminal_instance in terminal_control_instances
                or continuation_instance in terminal_control_instances
            ):
                raise RealizationWitnessError(
                    "pi terminal control arm instances must be globally unique"
                )
            terminal_control_instances.update(
                {terminal_instance, continuation_instance}
            )
            checked_terminal_controls.append(
                {
                    "terminal_instance": terminal_instance,
                    "continuation_instance": continuation_instance,
                }
            )

        checked_iterations: list[dict[str, JsonValue]] = []
        iteration_regions: set[str] = set()
        for index, raw in enumerate(self.iterations):
            item = _record(
                raw,
                path=f"pi.iterations[{index}]",
                required=frozenset(
                    {
                        "region",
                        "loop_instance",
                        "tile_instance",
                        "induction_basis",
                    }
                ),
            )
            region = _name(item["region"], f"pi.iterations[{index}].region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi iteration relation names an unknown region"
                )
            if region in iteration_regions:
                raise RealizationWitnessError(
                    "pi iteration regions may have at most one relation"
                )
            iteration_regions.add(region)
            loop_instance = _declared_instance_ref(
                item["loop_instance"],
                path=f"pi.iterations[{index}].loop_instance",
                known_instances=known_instances,
            )
            tile_instance = _declared_instance_ref(
                item["tile_instance"],
                path=f"pi.iterations[{index}].tile_instance",
                known_instances=known_instances,
            )
            if loop_instance == tile_instance:
                raise RealizationWitnessError(
                    "pi iteration loop and tile instances must be distinct"
                )
            induction_basis = _name(
                item["induction_basis"],
                f"pi.iterations[{index}].induction_basis",
            )
            if induction_basis not in self.ITERATION_BASES:
                raise RealizationWitnessError(
                    f"unsupported pi induction basis {induction_basis!r}"
                )
            checked_iterations.append(
                {
                    "region": region,
                    "loop_instance": loop_instance,
                    "tile_instance": tile_instance,
                    "induction_basis": induction_basis,
                }
            )

        checked_physical_iterations: list[dict[str, JsonValue]] = []
        physical_iteration_regions: set[str] = set()
        for index, raw in enumerate(self.physical_iterations):
            path = f"pi.physical_iterations[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {
                        "region",
                        "body_instance",
                        "tile_instance",
                        "induction_basis",
                        "direction",
                    }
                ),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi physical iteration relation names an unknown region"
                )
            if region in physical_iteration_regions:
                raise RealizationWitnessError(
                    "pi physical iteration regions may have at most one relation"
                )
            if region in iteration_regions:
                raise RealizationWitnessError(
                    "one pi region cannot carry both structured and physical-only "
                    "iteration relations"
                )
            physical_iteration_regions.add(region)
            body_instance = _declared_instance_ref(
                item["body_instance"],
                path=f"{path}.body_instance",
                known_instances=known_instances,
            )
            tile_instance = _declared_instance_ref(
                item["tile_instance"],
                path=f"{path}.tile_instance",
                known_instances=known_instances,
            )
            if body_instance == tile_instance:
                raise RealizationWitnessError(
                    "pi physical iteration body and tile instances must be distinct"
                )
            induction_basis = _name(
                item["induction_basis"], f"{path}.induction_basis"
            )
            if induction_basis not in self.ITERATION_BASES:
                raise RealizationWitnessError(
                    f"unsupported pi physical induction basis {induction_basis!r}"
                )
            direction = _name(item["direction"], f"{path}.direction")
            if direction not in self.PHYSICAL_ITERATION_DIRECTIONS:
                raise RealizationWitnessError(
                    f"unsupported pi physical iteration direction {direction!r}"
                )
            checked_physical_iterations.append(
                {
                    "region": region,
                    "body_instance": body_instance,
                    "tile_instance": tile_instance,
                    "induction_basis": induction_basis,
                    "direction": direction,
                }
            )

        checked_loop_carriers: list[dict[str, JsonValue]] = []
        carrier_states: set[str] = set()
        carrier_results: set[str] = set()
        for index, raw in enumerate(self.loop_carriers):
            path = f"pi.loop_carriers[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {
                        "region",
                        "loop_instance",
                        "init",
                        "state",
                        "next",
                        "result",
                    }
                ),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi loop carrier relation names an unknown region"
                )
            refs = {
                key: _declared_instance_ref(
                    item[key],
                    path=f"{path}.{key}",
                    known_instances=known_instances,
                )
                for key in ("loop_instance", "init", "state", "next", "result")
            }
            if refs["init"] in {refs["state"], refs["next"], refs["result"]}:
                raise RealizationWitnessError(
                    "pi loop carrier init must be a distinct instance"
                )
            if refs["result"] in {refs["state"], refs["next"]}:
                raise RealizationWitnessError(
                    "pi loop carrier result must be a distinct instance"
                )
            if refs["state"] in carrier_states:
                raise RealizationWitnessError(
                    "pi loop carrier state slots must be unique"
                )
            if refs["result"] in carrier_results:
                raise RealizationWitnessError(
                    "pi loop carrier result slots must be unique"
                )
            carrier_states.add(refs["state"])
            carrier_results.add(refs["result"])
            checked_loop_carriers.append({"region": region, **refs})

        checked_physical_loop_carriers: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.physical_loop_carriers):
            path = f"pi.physical_loop_carriers[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"region", "init", "state", "next", "result"}),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in physical_iteration_regions:
                raise RealizationWitnessError(
                    "pi physical loop carrier requires one declared physical "
                    "iteration region"
                )
            refs = {
                key: _declared_instance_ref(
                    item[key],
                    path=f"{path}.{key}",
                    known_instances=known_instances,
                )
                for key in ("init", "state", "next", "result")
            }
            if refs["init"] in {refs["state"], refs["next"], refs["result"]}:
                raise RealizationWitnessError(
                    "pi physical loop carrier init must be a distinct instance"
                )
            if refs["result"] in {refs["state"], refs["next"]}:
                raise RealizationWitnessError(
                    "pi physical loop carrier result must be a distinct instance"
                )
            if refs["state"] in carrier_states:
                raise RealizationWitnessError(
                    "pi physical loop carrier state slots must be globally unique"
                )
            if refs["result"] in carrier_results:
                raise RealizationWitnessError(
                    "pi physical loop carrier result slots must be globally unique"
                )
            carrier_states.add(refs["state"])
            carrier_results.add(refs["result"])
            checked_physical_loop_carriers.append({"region": region, **refs})

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

        checked_refinements: list[dict[str, JsonValue]] = []
        refined_instances: set[str] = set()
        for index, raw in enumerate(self.refinements):
            item = _record(
                raw,
                path=f"pi.refinements[{index}]",
                required=frozenset({"instance", "kind", "auth_ref"}),
            )
            instance = _declared_instance_ref(
                item["instance"],
                path=f"pi.refinements[{index}].instance",
                known_instances=known_instances,
            )
            if instance in refined_instances:
                raise RealizationWitnessError(
                    "pi refinements may refine each instance at most once"
                )
            refined_instances.add(instance)
            kind = _name(item["kind"], f"pi.refinements[{index}].kind")
            if kind not in REPRESENTATION_REFINEMENTS:
                raise RealizationWitnessError(
                    f"unsupported pi representation refinement {kind!r}"
                )
            auth_ref = item["auth_ref"]
            if auth_ref is not None:
                auth_ref = _name(
                    auth_ref, f"pi.refinements[{index}].auth_ref"
                )
                if re.fullmatch(r"u[0-9]+", auth_ref) is None:
                    raise RealizationWitnessError(
                        "pi refinement auth_ref must be a J authorization handle u*"
                    )
            checked_refinements.append(
                {"instance": instance, "kind": kind, "auth_ref": auth_ref}
            )

        self.regions = checked_regions
        self.instances = checked_instances
        self.routes = checked_routes
        self.order = checked_order
        self.sharing = checked_sharing
        self.refinements = checked_refinements
        self.iterations = checked_iterations
        self.loop_carriers = checked_loop_carriers
        self.physical_iterations = checked_physical_iterations
        self.physical_loop_carriers = checked_physical_loop_carriers
        self.terminal_controls = checked_terminal_controls

    @classmethod
    def from_json(cls, value: Any) -> "RealizationWitness":
        data = _record(
            value,
            path="pi",
            required=frozenset(
                {
                    "regions",
                    "instances",
                    "routes",
                    "order",
                    "sharing",
                    "refinements",
                }
            ),
            allowed=frozenset(
                {
                    "regions",
                    "instances",
                    "routes",
                    "order",
                    "sharing",
                    "refinements",
                    "iterations",
                    "loop_carriers",
                    "physical_iterations",
                    "physical_loop_carriers",
                    "terminal_controls",
                }
            ),
        )
        for key in (
            "regions",
            "instances",
            "routes",
            "order",
            "sharing",
            "refinements",
            "iterations",
            "loop_carriers",
            "physical_iterations",
            "physical_loop_carriers",
            "terminal_controls",
        ):
            if key in data and not isinstance(data[key], list):
                raise RealizationWitnessError(f"pi.{key} must be an array")
        return cls(
            regions=copy.deepcopy(data["regions"]),
            instances=copy.deepcopy(data["instances"]),
            routes=copy.deepcopy(data["routes"]),
            order=copy.deepcopy(data["order"]),
            sharing=copy.deepcopy(data["sharing"]),
            refinements=copy.deepcopy(data["refinements"]),
            iterations=copy.deepcopy(data.get("iterations", [])),
            loop_carriers=copy.deepcopy(data.get("loop_carriers", [])),
            physical_iterations=copy.deepcopy(data.get("physical_iterations", [])),
            physical_loop_carriers=copy.deepcopy(
                data.get("physical_loop_carriers", [])
            ),
            terminal_controls=copy.deepcopy(data.get("terminal_controls", [])),
        )

    def to_json(
        self,
        *,
        include_iterations: bool = True,
        include_loop_carriers: bool = True,
        include_physical_time: bool = True,
        include_terminal_controls: bool = True,
    ) -> dict[str, Any]:
        result = {
            "regions": copy.deepcopy(self.regions),
            "instances": copy.deepcopy(self.instances),
            "routes": copy.deepcopy(self.routes),
            "order": copy.deepcopy(self.order),
            "sharing": copy.deepcopy(self.sharing),
            "refinements": copy.deepcopy(self.refinements),
        }
        if include_iterations:
            result["iterations"] = copy.deepcopy(self.iterations)
        if include_loop_carriers:
            result["loop_carriers"] = copy.deepcopy(self.loop_carriers)
        if include_physical_time:
            result["physical_iterations"] = copy.deepcopy(self.physical_iterations)
            result["physical_loop_carriers"] = copy.deepcopy(
                self.physical_loop_carriers
            )
        if include_terminal_controls:
            result["terminal_controls"] = copy.deepcopy(self.terminal_controls)
        return result


__all__ = [
    "RealizationWitness",
    "RealizationWitnessError",
    "REALIZATION_MECHANISMS",
    "ITERATION_BASES",
    "PHYSICAL_ITERATION_DIRECTIONS",
    "REPRESENTATION_REFINEMENTS",
    "render_realization_mechanism_grammar",
    "render_realization_witness_wire_grammar",
]
