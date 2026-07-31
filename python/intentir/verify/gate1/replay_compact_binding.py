"""Compact, episode-local Replay region bindings.

The fixed side derives the complete block/region scaffold from canonical J and pi.
The fresh source-blind Replay agent authors only target source inside those blocks.
It does not repeat node ids, region placement, carrier topology or block ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .replay_skeleton import REPLAY_SKELETON_PROTOCOL


REPLAY_COMPACT_BINDING_PROTOCOL = "intentir.replay-compact-binding.v2"
_BLOCK_ID = re.compile(r"b[0-9]+")
_DIRECT_CHILD = re.compile(r"^(.*)\[([0-9]+)\]$")


def _relations(skeleton: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    realization = skeleton.get("realization")
    if not isinstance(realization, Mapping):
        raise TypeError("Replay skeleton realization must be an object")
    value = realization.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"Replay skeleton realization.{name} must be an object array")
    return list(value)


def _semantic_nodes(skeleton: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    semantic = skeleton.get("semantic")
    if not isinstance(semantic, Mapping):
        raise TypeError("Replay skeleton semantic must be an object")
    value = semantic.get("nodes", [])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError("Replay skeleton semantic.nodes must be an object array")
    return list(value)


def _direct_nodes(nodes: Sequence[Mapping[str, Any]], prefix: str) -> list[Mapping[str, Any]]:
    result: list[tuple[int, Mapping[str, Any]]] = []
    for node in nodes:
        path = node.get("path")
        if not isinstance(path, str):
            continue
        match = _DIRECT_CHILD.fullmatch(path)
        if match is None or match.group(1) != prefix:
            continue
        result.append((int(match.group(2)), node))
    return [item for _index, item in sorted(result, key=lambda row: row[0])]


@dataclass(frozen=True)
class ReplayCompactPlan:
    blocks: tuple[dict[str, Any], ...]
    region_sequences: Mapping[str, tuple[str, ...]]
    control_regions: Mapping[str, dict[str, Any]]
    program_region: str

    def scaffold(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_COMPACT_BINDING_PROTOCOL,
            "blocks": [
                {
                    "id": item["id"],
                    "semantic_refs": list(item["semantic_refs"]),
                    "realization_refs": list(item["realization_refs"]),
                    "source_path": f"binding/{item['id']}.source",
                }
                for item in self.blocks
            ],
        }


def build_replay_compact_plan(skeleton: Mapping[str, Any]) -> ReplayCompactPlan:
    """Derive the exact editable block scaffold without selecting target semantics."""

    if skeleton.get("protocol") != REPLAY_SKELETON_PROTOCOL:
        raise ValueError("compact Replay binding requires a supported skeleton")
    nodes = _semantic_nodes(skeleton)
    regions = _relations(skeleton, "regions")
    instances = _relations(skeleton, "occurrences")
    instance_index = {
        str(item["id"]): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    region_index = {
        str(item["id"]): item
        for item in regions
        if isinstance(item.get("id"), str)
    }
    if len(instance_index) != len(instances) or len(region_index) != len(regions):
        raise ValueError("Replay skeleton has malformed occurrence or region ids")
    program_regions = [
        str(item["id"]) for item in regions if item.get("placement") == "program"
    ]
    if len(program_regions) != 1:
        raise ValueError("compact Replay currently requires exactly one program region")
    program_region = program_regions[0]

    control_ops = {
        str(item["id"]): item
        for item in nodes
        if item.get("op") in {"loop", "effect_if", "value_if"}
        and isinstance(item.get("id"), str)
    }
    control_regions: dict[str, dict[str, Any]] = {}
    scope_prefixes: dict[str, str] = {program_region: "body"}
    for node_ref, node in control_ops.items():
        occurrences = [
            item
            for item in instances
            if item.get("semantic_ref") == node_ref and item.get("mechanism") == "realize"
        ]
        if len(occurrences) != 1:
            raise ValueError(
                f"control node {node_ref} needs one exact pi realization occurrence"
            )
        child_region = str(occurrences[0].get("region"))
        if child_region not in region_index:
            raise ValueError(f"control node {node_ref} has an unknown realization region")
        path = node.get("path")
        if not isinstance(path, str):
            raise ValueError(f"control node {node_ref} has no lexical J path")
        control_regions[child_region] = {
            "semantic_ref": node_ref,
            "operation": dict(node),
            "parent": region_index[child_region].get("parent"),
        }
        if node.get("op") == "loop":
            scope_prefixes[child_region] = path + ".body"
        else:
            # Branch arms remain explicit raw blocks until pi carries an exact arm-region
            # split.  Never guess which arm a single physical region denotes.
            scope_prefixes[child_region] = path + ".body"

    block_rows: list[dict[str, Any]] = []
    region_sequences: dict[str, tuple[str, ...]] = {}
    semantic_to_block: dict[str, str] = {}

    def append_block(region: str, semantic_refs: list[str]) -> str:
        block_id = f"b{len(block_rows)}"
        row = {
            "id": block_id,
            "region": region,
            "semantic_refs": tuple(semantic_refs),
            "realization_refs": [],
        }
        block_rows.append(row)
        for ref in semantic_refs:
            semantic_to_block[ref] = block_id
        return block_id

    pending = list(scope_prefixes)
    for region in pending:
        prefix = scope_prefixes[region]
        tokens: list[str] = []
        semantic_refs: list[str] = []

        def flush() -> None:
            nonlocal semantic_refs
            if semantic_refs:
                tokens.append(append_block(region, semantic_refs))
                semantic_refs = []

        for node in _direct_nodes(nodes, prefix):
            node_ref = str(node.get("id"))
            child_region = next(
                (
                    candidate
                    for candidate, control in control_regions.items()
                    if control["semantic_ref"] == node_ref
                ),
                None,
            )
            if child_region is not None:
                flush()
                tokens.append(child_region)
            else:
                semantic_refs.append(node_ref)
        flush()
        if not tokens:
            tokens.append(append_block(region, []))
        region_sequences[region] = tuple(tokens)

    block_index = {str(item["id"]): item for item in block_rows}
    blocks_by_region: dict[str, list[str]] = {}
    for item in block_rows:
        blocks_by_region.setdefault(str(item["region"]), []).append(str(item["id"]))

    accesses = skeleton.get("semantic", {}).get("boundary", {}).get("accesses", [])
    access_index = {
        str(item["access_id"]): item
        for item in accesses
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }
    for access_ref in sorted(access_index):
        occurrence_regions = sorted(
            {
                str(item.get("region"))
                for item in instances
                if item.get("semantic_ref") == access_ref
            }
        )
        for region in occurrence_regions:
            candidates = blocks_by_region.get(region, [])
            if not candidates:
                raise ValueError(f"access {access_ref} has no editable block in {region}")
            node_ref = access_index[access_ref].get("node_ref")
            target = semantic_to_block.get(str(node_ref)) if isinstance(node_ref, str) else None
            if target not in candidates:
                target = candidates[0]
            block_index[target]["realization_refs"].append(f"access:{access_ref}@{region}")

    # Some structural address occurrences are anchored to a domain/value rather than
    # the access id they feed (for example a decoded program coordinate).  They still
    # need one block-level provenance owner; assign only the otherwise-unowned,
    # non-realize occurrence to the first block in its exact pi region.
    access_occurrence_ids = {
        str(item["id"])
        for item in instances
        if item.get("semantic_ref") in access_index and isinstance(item.get("id"), str)
    }
    for instance in instances:
        instance_id = instance.get("id")
        if (
            not isinstance(instance_id, str)
            or instance_id in access_occurrence_ids
            or instance.get("mechanism") == "realize"
        ):
            continue
        region = instance.get("region")
        candidates = blocks_by_region.get(str(region), [])
        if not candidates:
            raise ValueError(f"occurrence {instance_id} has no block in region {region}")
        block_index[candidates[0]]["realization_refs"].append(
            f"occurrence:{instance_id}"
        )

    realization_holes = skeleton.get("holes", {}).get("realization", [])
    for hole in realization_holes:
        if not isinstance(hole, Mapping) or not isinstance(hole.get("id"), str):
            continue
        relation = hole.get("authored_relation")
        if not isinstance(relation, Mapping):
            raise TypeError("realization hole relation must be an object")
        region = relation.get("region")
        if not isinstance(region, str):
            endpoint = relation.get("loop_instance") or relation.get("target") or relation.get("source")
            if isinstance(endpoint, str) and endpoint in instance_index:
                region = instance_index[endpoint].get("region")
        if not isinstance(region, str):
            region = program_region
        candidates = blocks_by_region.get(region, [])
        if not candidates:
            raise ValueError(f"realization hole {hole['id']} has no block in region {region}")
        block_index[candidates[0]]["realization_refs"].append(str(hole["id"]))

    normalized_blocks = tuple(
        {
            "id": str(item["id"]),
            "region": str(item["region"]),
            "semantic_refs": tuple(str(ref) for ref in item["semantic_refs"]),
            "realization_refs": tuple(str(ref) for ref in item["realization_refs"]),
        }
        for item in block_rows
    )
    return ReplayCompactPlan(
        normalized_blocks,
        {name: tuple(items) for name, items in region_sequences.items()},
        control_regions,
        program_region,
    )


@dataclass(frozen=True)
class ReplayCompactBinding:
    blocks: tuple[dict[str, Any], ...]
    by_id: Mapping[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_COMPACT_BINDING_PROTOCOL,
            "blocks": [
                {
                    "id": item["id"],
                    "semantic_refs": list(item["semantic_refs"]),
                    "realization_refs": list(item["realization_refs"]),
                    "source_path": f"binding/{item['id']}.source",
                }
                for item in self.blocks
            ],
        }


def parse_replay_compact_binding(
    skeleton: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    sources: Mapping[str, str],
) -> ReplayCompactBinding:
    """Validate an immutable scaffold plus separately authored source-block bytes."""

    if set(value) != {"protocol", "blocks"}:
        raise ValueError("compact Replay binding root must contain protocol and blocks")
    if value.get("protocol") != REPLAY_COMPACT_BINDING_PROTOCOL:
        raise ValueError("unsupported compact Replay binding protocol")
    raw_blocks = value.get("blocks")
    if not isinstance(raw_blocks, list):
        raise TypeError("compact Replay binding blocks must be an array")
    plan = build_replay_compact_plan(skeleton)
    expected = {str(item["id"]): item for item in plan.blocks}
    if len(raw_blocks) != len(expected):
        raise ValueError("compact Replay binding cannot add or remove scaffold blocks")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "semantic_refs",
            "realization_refs",
            "source_path",
        }:
            raise ValueError(f"binding.blocks[{index}] has non-scaffold fields")
        block_id = raw.get("id")
        if (
            not isinstance(block_id, str)
            or _BLOCK_ID.fullmatch(block_id) is None
            or block_id not in expected
            or block_id in seen
        ):
            raise ValueError(f"binding.blocks[{index}] has an invalid block id")
        seen.add(block_id)
        row = expected[block_id]
        if raw.get("semantic_refs") != list(row["semantic_refs"]):
            raise ValueError(f"binding block {block_id} changed its J provenance")
        if raw.get("realization_refs") != list(row["realization_refs"]):
            raise ValueError(f"binding block {block_id} changed its pi provenance")
        expected_source_path = f"binding/{block_id}.source"
        if raw.get("source_path") != expected_source_path:
            raise ValueError(f"binding block {block_id} changed its source path")
        source = sources.get(block_id)
        if not isinstance(source, str) or not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in source.splitlines()
        ):
            raise ValueError(f"work/{expected_source_path} is still empty")
        if "\x00" in source:
            raise ValueError(f"binding block {block_id} contains a NUL byte")
        parsed.append(
            {
                "id": block_id,
                "semantic_refs": list(row["semantic_refs"]),
                "realization_refs": list(row["realization_refs"]),
                "source": source,
            }
        )
    if seen != set(expected):
        raise ValueError("compact Replay binding does not cover the exact scaffold")
    ordered = tuple(next(item for item in parsed if item["id"] == row["id"]) for row in plan.blocks)
    return ReplayCompactBinding(ordered, {str(item["id"]): item for item in ordered})


__all__ = [
    "REPLAY_COMPACT_BINDING_PROTOCOL",
    "ReplayCompactBinding",
    "ReplayCompactPlan",
    "build_replay_compact_plan",
    "parse_replay_compact_binding",
]
