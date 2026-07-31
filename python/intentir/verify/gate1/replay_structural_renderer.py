"""Structural Triton renderer for compact Replay region bindings.

This renderer owns target plumbing only: signature aliases, program coordinates,
region nesting, loop bounds and loop-carrier wiring.  Every value computation,
address, mask, memory effect and target instruction inside a block is authored by
the fresh Replay agent.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .replay_compact_binding import (
    ReplayCompactBinding,
    build_replay_compact_plan,
)


REPLAY_STRUCTURAL_RENDER_PROTOCOL = "intentir.replay-structural-render.v2"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _relations(skeleton: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    value = skeleton.get("realization", {}).get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"Replay skeleton realization.{name} must be an object array")
    return list(value)


def _signature_names(skeleton: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    signature = skeleton.get("semantic", {}).get("signature", {})
    parameters = signature.get("parameters", []) if isinstance(signature, Mapping) else []
    if not isinstance(parameters, list):
        raise TypeError("Replay skeleton signature parameters must be an array")
    declarations: list[str] = []
    aliases: dict[str, str] = {}
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str):
            raise TypeError("Replay skeleton signature parameter is malformed")
        name = str(parameter["name"])
        declarations.append(
            name + (": tl.constexpr" if parameter.get("kind") == "constexpr" else "")
        )
        logical_ref = parameter.get("logical_ref")
        if isinstance(logical_ref, str):
            aliases.setdefault(logical_ref, name)
    slots = skeleton.get("masked_c", {}).get("slots", [])
    if not isinstance(slots, list):
        raise TypeError("Replay skeleton MaskC slots must be an array")
    for slot in slots:
        if not isinstance(slot, Mapping) or not isinstance(slot.get("name"), str):
            raise TypeError("Replay skeleton C slot is malformed")
        name = str(slot["name"])
        declarations.append(
            name + (": tl.constexpr" if slot.get("kind") == "constexpr" else "")
        )
        aliases.setdefault(name, name)
    return declarations, aliases


def _semantic_node_index(skeleton: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = skeleton.get("semantic", {}).get("nodes", [])
    if not isinstance(nodes, list):
        raise TypeError("Replay skeleton semantic nodes must be an array")
    result = {
        str(item["id"]): item
        for item in nodes
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(result) != len(nodes):
        raise ValueError("Replay skeleton semantic nodes have malformed ids")
    return result


def _instance_index(skeleton: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    instances = _relations(skeleton, "occurrences")
    result = {
        str(item["id"]): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    if len(result) != len(instances):
        raise ValueError("Replay skeleton occurrences have malformed ids")
    return result


def _target_name(ref: str, aliases: Mapping[str, str]) -> str:
    return aliases.get(ref, ref)


def _block_lines(source: str, indent: int) -> list[str]:
    normalized = textwrap.dedent(source).strip("\n")
    prefix = "    " * indent
    return [prefix + line if line else "" for line in normalized.splitlines()]


@dataclass(frozen=True)
class ReplayStructuralRender:
    source: str
    source_map: tuple[dict[str, Any], ...]
    fixed_structure: tuple[dict[str, Any], ...]

    def to_response(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_STRUCTURAL_RENDER_PROTOCOL,
            "status": "pass",
            "source_bytes": len(self.source.encode("utf-8")),
            "block_count": len(self.source_map),
            "fixed_structure": [dict(item) for item in self.fixed_structure],
            "source_map": [dict(item) for item in self.source_map],
            "authority": (
                "fixed code rendered only signature aliases, program coordinates, "
                "region/loop structure and exact J/pi carriers; all computation, "
                "addressing, masks, memory effects and target instructions are "
                "fresh Replay-agent-authored block bytes"
            ),
        }


def render_replay_structural_binding(
    skeleton: Mapping[str, Any], binding: ReplayCompactBinding
) -> ReplayStructuralRender:
    plan = build_replay_compact_plan(skeleton)
    declarations, aliases = _signature_names(skeleton)
    nodes = _semantic_node_index(skeleton)
    instances = _instance_index(skeleton)
    function_name = str(
        skeleton.get("public_frontend", {}).get("replay_function", "scrc_replay")
    )

    lines = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    if declarations:
        lines.append(f"def {function_name}(")
        lines.extend(f"    {item}," for item in declarations)
        lines.append("):")
    else:
        lines.append(f"def {function_name}():")
    source_map: list[dict[str, Any]] = []
    fixed_structure: list[dict[str, Any]] = []

    for logical_ref, parameter in sorted(aliases.items()):
        if logical_ref == parameter:
            continue
        lines.append(f"    {logical_ref} = {parameter}")

    program_axis_locals: dict[str, str] = {}
    for relation in _relations(skeleton, "program_axes"):
        domain_ref = relation.get("domain_ref")
        axis = relation.get("axis")
        if not isinstance(domain_ref, str) or isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError("pi program axis is malformed")
        local = f"_intentir_pid_{_safe_name(domain_ref)}"
        lines.append(f"    {local} = tl.program_id({axis})")
        program_axis_locals[domain_ref] = local
        fixed_structure.append(
            {"kind": "program_axis", "domain_ref": domain_ref, "axis": axis, "local": local}
        )

    for relation in _relations(skeleton, "program_linearizations"):
        domain_ref = relation.get("domain_ref")
        dimension_refs = relation.get("dimension_refs")
        if (
            not isinstance(domain_ref, str)
            or domain_ref not in program_axis_locals
            or not isinstance(dimension_refs, list)
            or len(dimension_refs) < 2
            or not all(isinstance(item, str) for item in dimension_refs)
        ):
            raise ValueError("pi program linearization is malformed or has no axis")
        quotient = program_axis_locals[domain_ref]
        for index, dimension_ref in enumerate(dimension_refs):
            local = f"_intentir_index_{_safe_name(dimension_ref)}"
            suffix = dimension_refs[index + 1 :]
            if suffix:
                radix_terms = [_target_name(str(ref), aliases) for ref in suffix]
                radix = " * ".join(radix_terms)
                expression = f"({quotient} // ({radix}))"
            else:
                expression = quotient
            if index > 0:
                expression = f"({expression} % {_target_name(dimension_ref, aliases)})"
            lines.append(f"    {local} = {expression}")
            fixed_structure.append(
                {
                    "kind": "program_linearization",
                    "domain_ref": domain_ref,
                    "dimension_ref": dimension_ref,
                    "local": local,
                }
            )

    if len(lines) == 6:
        lines.append("    pass")

    def append_block(block_id: str, indent: int) -> None:
        block = binding.by_id[block_id]
        rendered = _block_lines(str(block["source"]), indent)
        start = len(lines) + 1
        lines.extend(rendered)
        source_map.append(
            {
                "block": block_id,
                "region": next(item["region"] for item in plan.blocks if item["id"] == block_id),
                "semantic_refs": list(block["semantic_refs"]),
                "realization_refs": list(block["realization_refs"]),
                "start_line": start,
                "end_line": start + len(rendered) - 1,
            }
        )

    iteration_by_region: dict[str, Mapping[str, Any]] = {}
    for iteration in _relations(skeleton, "iterations"):
        region = iteration.get("region")
        if not isinstance(region, str) or region in iteration_by_region:
            raise ValueError("each structural loop region needs one exact iteration row")
        iteration_by_region[region] = iteration

    carriers_by_region: dict[str, list[Mapping[str, Any]]] = {}
    for carrier in _relations(skeleton, "loop_carriers"):
        region = carrier.get("region")
        if not isinstance(region, str):
            raise ValueError("loop carrier region is malformed")
        carriers_by_region.setdefault(region, []).append(carrier)

    def render_region(region: str, indent: int) -> None:
        for token in plan.region_sequences.get(region, ()):
            if token in binding.by_id:
                append_block(token, indent)
                continue
            control = plan.control_regions.get(token)
            if control is None:
                raise ValueError(f"structural plan references unknown child region {token}")
            operation = control["operation"]
            if operation.get("op") != "loop":
                raise ValueError(
                    "compact structural renderer currently requires agent-authored "
                    "raw control for non-loop regions"
                )
            iteration = iteration_by_region.get(token)
            if iteration is None:
                raise ValueError(f"loop region {token} lacks an exact pi iteration")
            tile_instance = iteration.get("tile_instance")
            if not isinstance(tile_instance, str) or tile_instance not in instances:
                raise ValueError(f"loop region {token} has an unknown tile occurrence")
            tile_ref = instances[tile_instance].get("semantic_ref")
            trip = operation.get("trip")
            if not isinstance(trip, Mapping) or not isinstance(trip.get("axis"), str):
                raise ValueError(f"loop region {token} has no exact J trip axis")
            extent = _target_name(str(trip["axis"]), aliases)
            tile = _target_name(str(tile_ref), aliases)
            carry_rows = operation.get("carry", [])
            if not isinstance(carry_rows, list):
                raise TypeError(f"loop region {token} J carry must be an array")
            pi_carriers = carriers_by_region.get(token, [])
            if len(carry_rows) != len(pi_carriers):
                raise ValueError(
                    f"loop region {token} J/pi carrier arity differs: "
                    f"{len(carry_rows)} != {len(pi_carriers)}"
                )
            for carry in carry_rows:
                if not isinstance(carry, Mapping):
                    raise TypeError("J loop carry row is malformed")
                lines.append(
                    "    " * indent
                    + f"{carry['body_argument']} = {carry['init_value']}"
                )
            induction = f"_intentir_iv_{_safe_name(str(control['semantic_ref']))}"
            lines.append(
                "    " * indent + f"for {induction} in range(0, {extent}, {tile}):"
            )
            fixed_structure.append(
                {
                    "kind": "loop",
                    "region": token,
                    "semantic_ref": control["semantic_ref"],
                    "induction": induction,
                    "extent": str(trip["axis"]),
                    "tile": str(tile_ref),
                    "carrier_arity": len(carry_rows),
                }
            )
            render_region(token, indent + 1)
            for carry in carry_rows:
                lines.append(
                    "    " * (indent + 1)
                    + f"{carry['body_argument']} = {carry['next_value']}"
                )
            for carry in carry_rows:
                lines.append(
                    "    " * indent
                    + f"{carry['result_value']} = {carry['body_argument']}"
                )

    render_region(plan.program_region, 1)
    source = "\n".join(lines) + "\n"
    return ReplayStructuralRender(source, tuple(source_map), tuple(fixed_structure))


__all__ = [
    "REPLAY_STRUCTURAL_RENDER_PROTOCOL",
    "ReplayStructuralRender",
    "render_replay_structural_binding",
]
