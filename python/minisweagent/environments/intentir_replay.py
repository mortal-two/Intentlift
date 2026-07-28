"""Fail-closed sandbox for one source-blind IntentIR Replay episode."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import stat
import struct
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict

from minisweagent.exceptions import Submitted


INTENTIR_REPLAY_SUBMIT = "INTENTIR_REPLAY_SUBMIT"
INTENTIR_REPLAY_SUBMIT_COMMAND = "intentir_replay_submit"
INTENTIR_REPLAY_COMMIT = "intentir_replay_commit"
INTENTIR_REPLAY_ROLLBACK = "intentir_replay_rollback"
INTENTIR_REPLAY_CHECK = "intentir_replay_check"
INTENTIR_REPLAY_COMPILE_CHECK = "intentir_replay_compile_check"
INTENTIR_REPLAY_OBLIGATION_AUDIT = "intentir_replay_obligation_audit"
INTENTIR_PACKET_QUERY = "intentir_packet_query"
REPLAY_STARTUP_PACKET_EMBED_LIMIT_BYTES = 64 * 1024
INTENTIR_REPLAY_INSPECT = "intentir_replay_inspect"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v13"
REPLAY_TRANSACTION_PROTOCOL = "intentir.replay-transaction.v1"
REPLAY_APPLY_PATCH_PROTOCOL = "intentir.replay-exact-text-patch.v1"
REPLAY_DIRECT_SUBMIT_PROTOCOL = "intentir.replay-direct-submit.v1"
MULTI_ACTION_PROTOCOL = "intentir.agent-multi-action.v1"
REPLAY_INPUT_FILES = frozenset(
    {"README.md", "packet.json", "REPLAY_CONTRACT.md", "TOOL.md"}
)
REPLAY_SANDBOX_PROTOCOL = "intentir.replay-busybox-empty-root.v1"
REPLAY_SANDBOX_HOSTNAME = "intentir-replay"
REPLAY_BWRAP = Path("/usr/bin/bwrap")
REPLAY_BUSYBOX = Path("/usr/bin/busybox")
REPLAY_TOOL_APPLETS = (
    "awk",
    "basename",
    "cat",
    "cmp",
    "cp",
    "cut",
    "dd",
    "diff",
    "dirname",
    "echo",
    "env",
    "expr",
    "false",
    "find",
    "grep",
    "head",
    "hexdump",
    "hostname",
    "ln",
    "ls",
    "mkdir",
    "mktemp",
    "mv",
    "od",
    "printf",
    "pwd",
    "readlink",
    "realpath",
    "rm",
    "rmdir",
    "sed",
    "seq",
    "sh",
    "sha256sum",
    "sleep",
    "sort",
    "stat",
    "tail",
    "tee",
    "test",
    "timeout",
    "touch",
    "tr",
    "true",
    "uniq",
    "wc",
    "which",
    "xargs",
)
MAX_REPLAY_BYTES = 1024 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 2048
MAX_WORKSPACE_DEPTH = 64
MAX_CHECK_FIELD_CHARS = 4096
MAX_CHECK_HANDLES = 32
MAX_CHECK_RESULT_BYTES = 128 * 1024
MAX_PACKET_QUERY_OUTPUT_BYTES = 10_000
MAX_REPLAY_INSPECT_OUTPUT_BYTES = 10_000
_SANDBOX_START_MARKER = b"__INTENTIR_REPLAY_SANDBOX_STARTED__\n"


def _packet_json_chars(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _packet_path(value: Any, path: str) -> Any:
    current = value
    if path in {"", "."}:
        return current
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                raise KeyError(
                    f"unknown path component {component!r}; "
                    f"choose {list(current)}"
                )
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                raise IndexError(
                    f"list index {index} out of range 0..{len(current) - 1}"
                )
            current = current[index]
        else:
            raise KeyError(f"cannot descend through {component!r}")
    return current


def _packet_children(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return {}
    result = {
        str(key): {
            "type": type(item).__name__,
            "json_chars": _packet_json_chars(item),
            "query": f"{INTENTIR_PACKET_QUERY} section {path}.{key}".replace(
                "..", "."
            ),
        }
        for key, item in list(items)[:96]
    }
    total = len(value)
    if total > len(result):
        result["<omitted>"] = {"count": total - len(result)}
    return result


def _packet_ops(body: Any, prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(body, list):
        return result
    for index, operation in enumerate(body):
        if not isinstance(operation, Mapping):
            continue
        path = f"{prefix}.{index}"
        result.append(
            {
                "path": path,
                "op": operation.get("op"),
                "result": operation.get("result"),
                "operands": operation.get("operands"),
                "scope_id": operation.get("scope_id"),
            }
        )
        result.extend(_packet_ops(operation.get("body"), f"{path}.body"))
        for arm in ("positive", "negative"):
            branch = operation.get(arm)
            if isinstance(branch, Mapping):
                result.extend(
                    _packet_ops(branch.get("body"), f"{path}.{arm}.body")
                )
    return result


def _packet_graph_ops(body: Any, prefix: str) -> list[dict[str, Any]]:
    """Flatten full authored op records without duplicating nested bodies."""

    result: list[dict[str, Any]] = []
    if not isinstance(body, list):
        return result
    for index, operation in enumerate(body):
        if not isinstance(operation, Mapping):
            continue
        path = f"{prefix}.{index}"
        record = {
            "path": path,
            **{
                str(key): value
                for key, value in operation.items()
                if key not in {"body", "positive", "negative"}
            },
        }
        for arm in ("positive", "negative"):
            branch = operation.get(arm)
            if isinstance(branch, Mapping):
                record[arm] = {
                    str(key): value
                    for key, value in branch.items()
                    if key != "body"
                }
                arm_body = branch.get("body")
                record[arm]["body_operation_count"] = (
                    len(arm_body) if isinstance(arm_body, list) else 0
                )
        nested = operation.get("body")
        if isinstance(nested, list):
            record["body_operation_count"] = len(nested)
        result.append(record)
        result.extend(_packet_graph_ops(nested, f"{path}.body"))
        for arm in ("positive", "negative"):
            branch = operation.get(arm)
            if isinstance(branch, Mapping):
                result.extend(
                    _packet_graph_ops(
                        branch.get("body"), f"{path}.{arm}.body"
                    )
                )
    return result


def _packet_overview(packet: Mapping[str, Any]) -> dict[str, Any]:
    j = packet.get("j", {})
    intent = j.get("intent", {}) if isinstance(j, Mapping) else {}
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    boundary = j.get("boundary", {}) if isinstance(j, Mapping) else {}
    pi = packet.get("pi", {})
    c = packet.get("c", {})
    u_f = packet.get("u_f", {})
    signature = j.get("signature", {}) if isinstance(j, Mapping) else {}
    operations = _packet_ops(
        function.get("body", []) if isinstance(function, Mapping) else [],
        "j.intent.intent_function.body",
    )
    return {
        "root_sections": {
            key: {
                "json_chars": _packet_json_chars(value),
                "query": f"{INTENTIR_PACKET_QUERY} section {key}",
            }
            for key, value in packet.items()
        },
        "entry": u_f.get("replay_function") if isinstance(u_f, Mapping) else None,
        "frontend": u_f.get("frontend") if isinstance(u_f, Mapping) else None,
        "j_parameter_total": len(signature.get("parameters", [])),
        "j_parameters": [
            {
                key: parameter.get(key)
                for key in ("name", "kind", "mode", "dtype", "rank")
            }
            for parameter in signature.get("parameters", [])[:24]
            if isinstance(parameter, Mapping)
        ],
        "j_parameters_omitted": max(
            0, len(signature.get("parameters", [])) - 24
        ),
        "operation_count": len(operations),
        "operations": operations[:16],
        "operations_omitted": max(0, len(operations) - 16),
        "boundary_counts": {
            key: len(value)
            for key, value in boundary.items()
            if isinstance(value, list)
        }
        if isinstance(boundary, Mapping)
        else {},
        "pi_counts": {
            key: len(value)
            for key, value in pi.items()
            if isinstance(value, list)
        }
        if isinstance(pi, Mapping)
        else {},
        "c_slots": c.get("slots", []) if isinstance(c, Mapping) else [],
        "next": [
            "This overview contains the compact ABI/op inventory and exact "
            "root section sizes; do not query whole j/pi/u_f sections.",
            f"Query `{INTENTIR_PACKET_QUERY} abi 0 48` for full parameter "
            "fields, Intent I/O, U_f entry metadata, and MaskC slots.",
            f"Query `{INTENTIR_PACKET_QUERY} graph 0 12` for full op fields; "
            "page only if `next_start` is non-null.",
            f"Query `{INTENTIR_PACKET_QUERY} realization` once, then page "
            f"`{INTENTIR_PACKET_QUERY} obligations 0 12` across every instance. "
            "It is the primary per-instance join of authored routes, order, "
            "sharing, temporal roles, and refinements. Use "
            f"`{INTENTIR_PACKET_QUERY} obligation HANDLE` to inspect one i* "
            "instance or all instances anchored to one anonymous J/C handle.",
            f"Use `{INTENTIR_PACKET_QUERY} routes 0 16`, "
            f"`{INTENTIR_PACKET_QUERY} order 0 12`, or "
            f"`{INTENTIR_PACKET_QUERY} edges 0 24` only to cross-check a named "
            "relation; do not redundantly page all three after the obligation "
            "ledger.",
            "Use `section PATH` only for a named field omitted by these views.",
            f"Use `{INTENTIR_PACKET_QUERY} find HANDLE` to follow references.",
        ],
    }


def _bounded_page(
    values: list[Any],
    *,
    start: int,
    count: int,
) -> tuple[list[Any], int | None]:
    if start < 0 or count < 1 or count > 48:
        raise ValueError("page start must be >= 0 and count must be in [1, 48]")
    end = min(len(values), start + count)
    return values[start:end], end if end < len(values) else None


def _packet_abi(packet: Mapping[str, Any], start: int, count: int) -> Any:
    j = packet.get("j", {})
    if not isinstance(j, Mapping):
        raise TypeError("packet j must be an object")
    signature = j.get("signature", {})
    intent = j.get("intent", {})
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    parameters = (
        signature.get("parameters", [])
        if isinstance(signature, Mapping)
        else []
    )
    if not isinstance(parameters, list):
        raise TypeError("packet J signature parameters must be an array")
    items, next_start = _bounded_page(
        parameters,
        start=start,
        count=count,
    )
    u_f = packet.get("u_f", {})
    u_f_summary = (
        {
            key: value
            for key, value in u_f.items()
            if key != "compile_entry_contract"
        }
        if isinstance(u_f, Mapping)
        else u_f
    )
    if isinstance(u_f, Mapping) and "compile_entry_contract" in u_f:
        u_f_summary = {
            **u_f_summary,
            "compile_entry_contract": {
                "json_chars": _packet_json_chars(
                    u_f["compile_entry_contract"]
                ),
                "query": (
                    f"{INTENTIR_PACKET_QUERY} section "
                    "u_f.compile_entry_contract"
                ),
            },
        }
    return {
        "u_f": u_f_summary,
        "intent_io": {
            key: function.get(key)
            for key in ("name", "inputs", "outputs", "returns")
            if isinstance(function, Mapping) and key in function
        },
        "mask_c": packet.get("c"),
        "parameter_total": len(parameters),
        "parameter_start": start,
        "parameters": items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} abi {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_graph(packet: Mapping[str, Any], start: int, count: int) -> Any:
    j = packet.get("j", {})
    intent = j.get("intent", {}) if isinstance(j, Mapping) else {}
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    operations = _packet_graph_ops(
        function.get("body", []) if isinstance(function, Mapping) else [],
        "j.intent.intent_function.body",
    )
    items, next_start = _bounded_page(
        operations,
        start=start,
        count=count,
    )
    return {
        "total_operations": len(operations),
        "start": start,
        "items": items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} graph {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_realization(packet: Mapping[str, Any]) -> Any:
    pi = packet.get("pi", {})
    if not isinstance(pi, Mapping):
        raise TypeError("packet pi must be an object")
    edge_keys = {"routes", "order"}
    return {
        "counts": {
            key: len(value)
            for key, value in pi.items()
            if isinstance(value, list)
        },
        "topology": {
            key: value
            for key, value in pi.items()
            if key not in edge_keys
        },
        "edge_query": (
            f"{INTENTIR_PACKET_QUERY} edges 0 24"
            if any(pi.get(key) for key in edge_keys)
            else None
        ),
    }


def _packet_requirements(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded exact checklist over authored Replay obligations.

    This view deliberately does not translate an Intent op or pi relation into
    frontend source.  It only makes compiler-visible occurrences hard to lose in
    a long packet: every op remains visible in declaration order, while the
    already-authored control/time relations are colocated with their exact
    instance and region records.
    """

    j = packet.get("j", {})
    pi = packet.get("pi", {})
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        raise TypeError("packet J/pi must be objects")
    intent = j.get("intent", {})
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    boundary = j.get("boundary", {})
    if not isinstance(function, Mapping) or not isinstance(boundary, Mapping):
        raise TypeError("packet J intent/boundary must be objects")
    operations = _packet_graph_ops(
        function.get("body", []), "j.intent.intent_function.body"
    )
    operation_counts = Counter(
        str(item.get("op"))
        for item in operations
        if isinstance(item.get("op"), str)
    )
    sequence_limit = 64
    sequence = [
        [item.get("path"), item.get("id"), item.get("op")]
        for item in operations[:sequence_limit]
    ]
    topology_ops = {
        "contraction",
        "effect_if",
        "fma",
        "loop",
        "reduce",
        "value_if",
    }
    topology_rows = [
        item for item in operations if item.get("op") in topology_ops
    ]
    topology_limit = 24

    relation_names = (
        "iterations",
        "loop_carriers",
        "physical_iterations",
        "unit_iterations",
        "physical_loop_carriers",
        "terminal_controls",
    )
    relation_views: dict[str, Any] = {}
    referenced_instances: set[str] = set()
    relation_instance_fields = {
        "iterations": ("loop_instance", "tile_instance"),
        "loop_carriers": ("loop_instance", "init", "state", "next", "result"),
        "physical_iterations": ("body_instance", "tile_instance"),
        "unit_iterations": ("body_instance",),
        "physical_loop_carriers": ("init", "state", "next", "result"),
        "terminal_controls": ("terminal_instance", "continuation_instance"),
    }
    relation_limit = 12
    for name in relation_names:
        rows = pi.get(name, [])
        if not isinstance(rows, list):
            raise TypeError(f"packet pi {name} must be an array")
        relation_views[name] = {
            "total": len(rows),
            "items": rows[:relation_limit],
            "omitted": max(0, len(rows) - relation_limit),
            "omitted_query": (
                f"{INTENTIR_PACKET_QUERY} section pi.{name}"
                if len(rows) > relation_limit
                else None
            ),
        }
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for field_name in relation_instance_fields[name]:
                value = row.get(field_name)
                if isinstance(value, str):
                    referenced_instances.add(value)

    instances = pi.get("instances", [])
    regions = pi.get("regions", [])
    if not isinstance(instances, list) or not isinstance(regions, list):
        raise TypeError("packet pi instances/regions must be arrays")
    mechanism_counts = Counter(
        str(item.get("mechanism"))
        for item in instances
        if isinstance(item, Mapping) and isinstance(item.get("mechanism"), str)
    )
    instance_limit = 48
    instance_mechanisms = [
        [
            item.get("id"),
            item.get("semantic_ref"),
            item.get("mechanism"),
            item.get("region"),
        ]
        for item in instances[:instance_limit]
        if isinstance(item, Mapping)
    ]
    selected_instances = [
        dict(item)
        for item in instances
        if isinstance(item, Mapping) and item.get("id") in referenced_instances
    ]
    referenced_regions = {
        str(item.get("region"))
        for item in selected_instances
        if isinstance(item.get("region"), str)
    }
    for name in (
        "physical_iterations",
        "unit_iterations",
        "physical_loop_carriers",
    ):
        rows = pi.get(name, [])
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping) and isinstance(row.get("region"), str):
                referenced_regions.add(str(row["region"]))
    region_index = {
        str(item.get("id")): item
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    pending_regions = list(referenced_regions)
    while pending_regions:
        region_id = pending_regions.pop()
        region = region_index.get(region_id)
        parent = region.get("parent") if isinstance(region, Mapping) else None
        if isinstance(parent, str) and parent not in referenced_regions:
            referenced_regions.add(parent)
            pending_regions.append(parent)
    selected_regions = [
        dict(item)
        for item in regions
        if isinstance(item, Mapping) and item.get("id") in referenced_regions
    ]

    effects = boundary.get("effects", [])
    if not isinstance(effects, list):
        raise TypeError("packet J boundary effects must be an array")
    effect_limit = 16
    return {
        "protocol": "intentir.replay-authored-requirements.v1",
        "meaning": (
            "A bounded declaration-order reindex of exact packet facts. It does "
            "not infer an operator, edge, schedule, frontend spelling, source "
            "statement, or repair. Every omitted row remains available through "
            "the named graph/section/obligation query."
        ),
        "operations": {
            "total": len(operations),
            "counts": dict(sorted(operation_counts.items())),
            "sequence_columns": ["path", "id", "op"],
            "sequence": sequence,
            "sequence_omitted": max(0, len(operations) - sequence_limit),
            "sequence_next": (
                f"{INTENTIR_PACKET_QUERY} graph {sequence_limit} 48"
                if len(operations) > sequence_limit
                else None
            ),
            "topology_rows": topology_rows[:topology_limit],
            "topology_rows_omitted": max(0, len(topology_rows) - topology_limit),
            "topology_rows_omitted_queries": [
                f"{INTENTIR_PACKET_QUERY} section {item['path']}"
                for item in topology_rows[topology_limit : topology_limit + 8]
                if isinstance(item.get("path"), str)
            ],
        },
        "boundary_effects": {
            "total": len(effects),
            "items": effects[:effect_limit],
            "omitted": max(0, len(effects) - effect_limit),
            "omitted_query": (
                f"{INTENTIR_PACKET_QUERY} section j.boundary.effects"
                if len(effects) > effect_limit
                else None
            ),
        },
        "pi_time_and_control": relation_views,
        "pi_instance_mechanisms": {
            "columns": ["id", "semantic_ref", "mechanism", "region"],
            "total": len(instances),
            "counts": dict(sorted(mechanism_counts.items())),
            "items": instance_mechanisms,
            "omitted": max(0, len(instances) - instance_limit),
            "omitted_query": (
                f"{INTENTIR_PACKET_QUERY} obligations {instance_limit} 48"
                if len(instances) > instance_limit
                else None
            ),
        },
        "referenced_instances": selected_instances,
        "referenced_regions": selected_regions,
        "authority": (
            "exact source-isolated packet projection only; no Source, Harness "
            "value, Gate evidence, inference, generation, repair, or schedule"
        ),
    }


def _packet_critical_facts(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return a tiny exact index of easy-to-miss authored packet facts.

    The normal startup already transports the whole packet.  This index does not
    summarize its graph or recommend a lowering; it merely colocates eight
    closed fact classes whose distant cross-references are otherwise easy to
    overlook in a large ABI: semantic dimension carriers, typed IEEE-special
    constants, physical- and unit-iteration relations, transfer occurrences,
    reduction linearizations, exact lexical-region membership, and cross-region
    order.
    """

    j = packet.get("j", {})
    pi = packet.get("pi", {})
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        raise TypeError("packet J/pi must be objects")
    signature = j.get("signature", {})
    parameters = (
        signature.get("parameters", [])
        if isinstance(signature, Mapping)
        else []
    )
    if not isinstance(parameters, list):
        raise TypeError("packet J signature parameters must be an array")
    dimension_carriers = [
        {
            key: parameter.get(key)
            for key in ("name", "kind", "dtype", "logical_ref")
        }
        for parameter in parameters
        if isinstance(parameter, Mapping)
        and isinstance(parameter.get("logical_ref"), str)
        and str(parameter["logical_ref"]).startswith("x")
    ]

    intent = j.get("intent", {})
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    operations = _packet_graph_ops(
        function.get("body", []) if isinstance(function, Mapping) else [],
        "j.intent.intent_function.body",
    )
    special_constants = []
    for operation in operations:
        attrs = operation.get("attrs")
        if operation.get("op") != "const" or not isinstance(attrs, Mapping):
            continue
        if attrs.get("value") not in {"-inf", "inf", "nan"}:
            continue
        special_constants.append(
            {
                "path": operation.get("path"),
                "id": operation.get("id"),
                "value": attrs.get("value"),
                "dtype": attrs.get("dtype"),
                "shape": attrs.get("shape", []),
            }
        )

    physical_iterations = pi.get("physical_iterations", [])
    if not isinstance(physical_iterations, list):
        raise TypeError("packet pi physical_iterations must be an array")
    unit_iterations = pi.get("unit_iterations", [])
    if not isinstance(unit_iterations, list):
        raise TypeError("packet pi unit_iterations must be an array")
    instances_value = pi.get("instances", [])
    regions_value = pi.get("regions", [])
    order_value = pi.get("order", [])
    if not isinstance(instances_value, list) or not isinstance(regions_value, list):
        raise TypeError("packet pi instances/regions must be arrays")
    if not isinstance(order_value, list):
        raise TypeError("packet pi order must be an array")
    instances = {
        str(item.get("id")): item
        for item in instances_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    regions = {
        str(item.get("id")): item
        for item in regions_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }

    def instance_scope(instance_id: str) -> dict[str, Any]:
        instance = instances[instance_id]
        region_id = str(instance.get("region"))
        region = regions.get(region_id, {})
        return {
            "id": instance_id,
            "semantic_ref": instance.get("semantic_ref"),
            "mechanism": instance.get("mechanism"),
            "region": region_id,
            "placement": region.get("placement"),
        }

    lexical_regions = []
    lexical_region_ids: set[str] = set()
    for raw_region in regions_value:
        if not isinstance(raw_region, Mapping) or raw_region.get("placement") not in {
            "loop",
            "branch",
        }:
            continue
        region_id = str(raw_region.get("id"))
        lexical_region_ids.add(region_id)
        lexical_regions.append(
            {
                "id": region_id,
                "parent": raw_region.get("parent"),
                "placement": raw_region.get("placement"),
                "instances": [
                    instance_scope(str(item.get("id")))
                    for item in instances_value
                    if isinstance(item, Mapping)
                    and isinstance(item.get("id"), str)
                    and str(item.get("region")) == region_id
                ],
            }
        )

    cross_region_order = []
    for item in order_value:
        if not isinstance(item, Mapping):
            continue
        before_id = item.get("before")
        after_id = item.get("after")
        if not isinstance(before_id, str) or not isinstance(after_id, str):
            continue
        if before_id not in instances or after_id not in instances:
            continue
        before_region = str(instances[before_id].get("region"))
        after_region = str(instances[after_id].get("region"))
        if before_region == after_region or not (
            {before_region, after_region} & lexical_region_ids
        ):
            continue
        cross_region_order.append(
            {
                "before": instance_scope(before_id),
                "after": instance_scope(after_id),
            }
        )
    routes = pi.get("routes", [])
    refinements = pi.get("refinements", [])
    if not isinstance(routes, list) or not isinstance(refinements, list):
        raise TypeError("packet pi routes/refinements must be arrays")
    reduction_linearizations = []
    for refinement in refinements:
        if not isinstance(refinement, Mapping) or refinement.get(
            "kind"
        ) != "reduction_linearization":
            continue
        instance_id = str(refinement.get("instance"))
        instance = instances.get(instance_id, {})
        reduction_linearizations.append(
            {
                "instance": instance_id,
                "semantic_ref": instance.get("semantic_ref"),
                "mechanism": instance.get("mechanism"),
                "region": instance.get("region"),
                "consumer_routes": [
                    dict(route)
                    for route in routes
                    if isinstance(route, Mapping)
                    and route.get("source") == instance_id
                ],
            }
        )
    transfer_occurrences = []
    order = pi.get("order", [])
    if not isinstance(order, list):
        raise TypeError("packet pi order must be an array")
    for instance_id, instance in instances.items():
        mechanism = instance.get("mechanism")
        if mechanism not in {"transfer_direct", "transfer_async"}:
            continue
        transfer_occurrences.append(
            {
                "instance": instance_id,
                "semantic_ref": instance.get("semantic_ref"),
                "mechanism": mechanism,
                "region": instance.get("region"),
                "incoming_routes": [
                    dict(route)
                    for route in routes
                    if isinstance(route, Mapping)
                    and route.get("target") == instance_id
                ],
                "outgoing_routes": [
                    dict(route)
                    for route in routes
                    if isinstance(route, Mapping)
                    and route.get("source") == instance_id
                ],
                "predecessor_order": [
                    dict(edge)
                    for edge in order
                    if isinstance(edge, Mapping) and edge.get("after") == instance_id
                ],
                "successor_order": [
                    dict(edge)
                    for edge in order
                    if isinstance(edge, Mapping) and edge.get("before") == instance_id
                ],
            }
        )
    return {
        "protocol": "intentir.replay-critical-facts.v2",
        "semantic_dimension_carriers": dimension_carriers,
        "typed_ieee_special_constants": special_constants,
        "physical_iterations": physical_iterations,
        "unit_iterations": unit_iterations,
        "transfer_occurrences": transfer_occurrences,
        "reduction_linearizations": reduction_linearizations,
        "lexical_regions": lexical_regions,
        "cross_region_order": cross_region_order,
        "authority": (
            "exact source-isolated packet projection only; no Source, Harness "
            "value, Gate evidence, inference, generation, repair, or schedule"
        ),
    }


def _replay_obligation_audit_view(
    packet: Mapping[str, Any], source_text: str
) -> dict[str, Any]:
    """Colocate exact authored scope/order obligations with the owner's bytes.

    This is a proof-irrelevant attention aid.  It neither maps a source statement
    to a pi instance nor judges whether Replay satisfies an obligation.
    """

    facts = _packet_critical_facts(packet)
    lexical_regions = facts["lexical_regions"]
    cross_region_order = facts["cross_region_order"]
    return {
        "protocol": "intentir.replay-obligation-audit.v1",
        "status": "pass",
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "audit_required": bool(lexical_regions or cross_region_order),
        "lexical_regions": lexical_regions,
        "cross_region_order": cross_region_order,
        "source_nonempty_lines": [
            {
                "line": line_number,
                "indent": len(line) - len(line.lstrip()),
                "text": line.strip(),
            }
            for line_number, line in enumerate(source_text.splitlines(), start=1)
            if line.strip()
        ],
        "authority": (
            "exact current Replay bytes plus exact source-isolated pi projection; "
            "no Source, Gate evidence, inference, mapping, diagnosis, repair, or "
            "acceptance decision"
        ),
    }


def _packet_edges(
    packet: Mapping[str, Any], start: int, count: int
) -> Any:
    pi = packet.get("pi", {})
    if not isinstance(pi, Mapping):
        raise TypeError("packet pi must be an object")
    routes = pi.get("routes", [])
    order = pi.get("order", [])
    if not isinstance(routes, list) or not isinstance(order, list):
        raise TypeError("packet pi routes/order must be arrays")
    route_items, route_next = _bounded_page(routes, start=start, count=count)
    order_items, order_next = _bounded_page(order, start=start, count=count)
    next_values = [item for item in (route_next, order_next) if item is not None]
    next_start = min(next_values) if next_values else None
    return {
        "start": start,
        "routes_total": len(routes),
        "routes": route_items,
        "order_total": len(order),
        "order": order_items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} edges {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_semantic_index(packet: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve anonymous J handles to compact source-blind descriptions."""

    result: dict[str, dict[str, Any]] = {}
    j = packet.get("j", {})
    if not isinstance(j, Mapping):
        return result
    intent = j.get("intent", {})
    function = (
        intent.get("intent_function", {})
        if isinstance(intent, Mapping)
        else {}
    )
    for operation in _packet_graph_ops(
        function.get("body", []) if isinstance(function, Mapping) else [],
        "j.intent.intent_function.body",
    ):
        node_id = operation.get("id")
        if isinstance(node_id, str):
            result[node_id] = {
                "semantic_kind": "intent_op",
                "op": operation.get("op"),
                "scope_id": operation.get("scope_id"),
                "path": operation.get("path"),
            }
    boundary = j.get("boundary", {})
    if isinstance(boundary, Mapping):
        accesses = boundary.get("accesses", [])
        if isinstance(accesses, list):
            for access in accesses:
                if not isinstance(access, Mapping):
                    continue
                access_id = access.get("access_id")
                if isinstance(access_id, str):
                    result[access_id] = {
                        "semantic_kind": "boundary_access",
                        "access_kind": access.get("kind"),
                        "tensor": access.get("tensor"),
                        "node_ref": access.get("node_ref"),
                    }
        effects = boundary.get("effects", [])
        if isinstance(effects, list):
            for effect in effects:
                if not isinstance(effect, Mapping):
                    continue
                effect_id = effect.get("effect_id")
                if isinstance(effect_id, str):
                    result[effect_id] = {
                        "semantic_kind": "boundary_effect",
                        "effect_kind": effect.get("kind"),
                        "target": effect.get("target"),
                        "node_ref": effect.get("node_ref"),
                        "access_ref": effect.get("access_ref"),
                    }
        domains = boundary.get("domains", {})
        if isinstance(domains, Mapping):
            for domain_id in domains:
                if isinstance(domain_id, str):
                    result[domain_id] = {"semantic_kind": "domain"}
    signature = j.get("signature", {})
    parameters = (
        signature.get("parameters", [])
        if isinstance(signature, Mapping)
        else []
    )
    if isinstance(parameters, list):
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                continue
            name = parameter.get("name")
            if isinstance(name, str):
                result[name] = {
                    "semantic_kind": "signature_parameter",
                    "kind": parameter.get("kind"),
                    "mode": parameter.get("mode"),
                    "logical_ref": parameter.get("logical_ref"),
                }
    return result


def _packet_instance_endpoint(
    instance_id: Any,
    *,
    instance_index: Mapping[Any, Any],
    region_index: Mapping[Any, Any],
    semantic_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one already-authored pi instance without inventing semantics."""

    instance = instance_index.get(instance_id)
    if not isinstance(instance, Mapping):
        return {"id": instance_id, "unresolved": True}
    semantic_ref = instance.get("semantic_ref")
    region_id = instance.get("region")
    region = region_index.get(region_id)
    return {
        "id": instance_id,
        "semantic_ref": semantic_ref,
        "semantic": semantic_index.get(
            semantic_ref,
            {"semantic_kind": "unclassified_handle"},
        ),
        "mechanism": instance.get("mechanism"),
        "region": region_id,
        "region_placement": (
            region.get("placement") if isinstance(region, Mapping) else None
        ),
    }


def _packet_route_obligations(
    packet: Mapping[str, Any], start: int, count: int
) -> Any:
    """Resolve authored pi routes to source-blind endpoint descriptions."""

    pi = packet.get("pi", {})
    if not isinstance(pi, Mapping):
        raise TypeError("packet pi must be an object")
    instances = pi.get("instances", [])
    regions = pi.get("regions", [])
    routes = pi.get("routes", [])
    if (
        not isinstance(instances, list)
        or not isinstance(regions, list)
        or not isinstance(routes, list)
    ):
        raise TypeError("packet pi instances/regions/routes must be arrays")
    instance_index = {
        item.get("id"): item
        for item in instances
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    region_index = {
        item.get("id"): item
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    semantic_index = _packet_semantic_index(packet)
    resolved: list[dict[str, Any]] = []
    for index, edge in enumerate(routes):
        if not isinstance(edge, Mapping):
            continue
        source = _packet_instance_endpoint(
            edge.get("source"),
            instance_index=instance_index,
            region_index=region_index,
            semantic_index=semantic_index,
        )
        target = _packet_instance_endpoint(
            edge.get("target"),
            instance_index=instance_index,
            region_index=region_index,
            semantic_index=semantic_index,
        )
        slots = edge.get("slots", [])
        resolved.append(
            {
                "original_index": index,
                "source": source,
                "target": target,
                "target_local_slots": slots,
                "required_flow": (
                    f"{source.get('id')} MUST flow to {target.get('id')} "
                    f"at target-local slots {slots}"
                ),
            }
        )
    items, next_start = _bounded_page(resolved, start=start, count=count)
    return {
        "protocol": "intentir.replay-resolved-routes.v1",
        "meaning": (
            "Every item expands one existing pi.routes entry. Slots remain "
            "target-local exactly as authored; this view infers no edge, value, "
            "or operation."
        ),
        "total": len(resolved),
        "start": start,
        "items": items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} routes {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_order_obligations(
    packet: Mapping[str, Any], start: int, count: int
) -> Any:
    """Resolve pi order handles and rank causal/effect edges first."""

    pi = packet.get("pi", {})
    if not isinstance(pi, Mapping):
        raise TypeError("packet pi must be an object")
    instances = pi.get("instances", [])
    regions = pi.get("regions", [])
    order = pi.get("order", [])
    if (
        not isinstance(instances, list)
        or not isinstance(regions, list)
        or not isinstance(order, list)
    ):
        raise TypeError("packet pi instances/regions/order must be arrays")
    instance_index = {
        item.get("id"): item
        for item in instances
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    region_index = {
        item.get("id"): item
        for item in regions
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    semantic_index = _packet_semantic_index(packet)

    resolved: list[dict[str, Any]] = []
    for index, edge in enumerate(order):
        if not isinstance(edge, Mapping):
            continue
        before = _packet_instance_endpoint(
            edge.get("before"),
            instance_index=instance_index,
            region_index=region_index,
            semantic_index=semantic_index,
        )
        after = _packet_instance_endpoint(
            edge.get("after"),
            instance_index=instance_index,
            region_index=region_index,
            semantic_index=semantic_index,
        )
        reasons: list[str] = []
        if before.get("region") != after.get("region"):
            reasons.append("cross_region")
        for side, value in (("before", before), ("after", after)):
            semantic = value.get("semantic")
            kind = (
                semantic.get("semantic_kind")
                if isinstance(semantic, Mapping)
                else None
            )
            if kind in {"boundary_access", "boundary_effect"}:
                reasons.append(f"{side}_observable_boundary")
            if kind == "intent_op" and semantic.get("op") in {
                "loop",
                "effect_if",
                "value_if",
            }:
                reasons.append(f"{side}_control_region")
        resolved.append(
            {
                "original_index": index,
                "before": before,
                "after": after,
                "required_direction": (
                    f"{before.get('id')} MUST execute before "
                    f"{after.get('id')}"
                ),
                "priority_reasons": reasons,
            }
        )
    ranked = sorted(
        resolved,
        key=lambda item: (
            -len(item["priority_reasons"]),
            item["original_index"],
        ),
    )
    items, next_start = _bounded_page(ranked, start=start, count=count)
    critical_total = sum(bool(item["priority_reasons"]) for item in ranked)
    return {
        "protocol": "intentir.replay-resolved-order.v1",
        "meaning": (
            "For every item, the resolved before endpoint MUST execute before "
            "the resolved after endpoint. Priority only changes this diagnostic "
            "view order; it never changes pi."
        ),
        "total": len(ranked),
        "critical_total": critical_total,
        "start": start,
        "items": items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} order {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_obligation_records(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join already-authored pi facts around each declared instance.

    This is deliberately a projection, not a realization planner: it neither
    invents an edge nor interprets Replay source.  The compact per-instance
    ledger merely saves the source-blind writer from manually joining several
    long, independently paged pi arrays.
    """

    pi = packet.get("pi", {})
    if not isinstance(pi, Mapping):
        raise TypeError("packet pi must be an object")
    list_names = (
        "regions",
        "instances",
        "routes",
        "order",
        "sharing",
        "refinements",
        "iterations",
        "loop_carriers",
        "physical_iterations",
        "unit_iterations",
        "physical_loop_carriers",
    )
    values: dict[str, list[Any]] = {}
    for name in list_names:
        value = pi.get(name, [])
        if not isinstance(value, list):
            raise TypeError(f"packet pi {name} must be an array")
        values[name] = value

    instance_index = {
        item.get("id"): item
        for item in values["instances"]
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    region_index = {
        item.get("id"): item
        for item in values["regions"]
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    semantic_index = _packet_semantic_index(packet)

    def endpoint(instance_id: Any) -> dict[str, Any]:
        instance = instance_index.get(instance_id)
        if not isinstance(instance, Mapping):
            return {"id": instance_id, "unresolved": True}
        return {
            "id": instance_id,
            "semantic_ref": instance.get("semantic_ref"),
        }

    incoming: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    outgoing: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    for route in values["routes"]:
        if not isinstance(route, Mapping):
            continue
        source = route.get("source")
        target = route.get("target")
        if isinstance(target, str) and target in incoming:
            incoming[target].append(
                {
                    "source": endpoint(source),
                    "target_local_slots": route.get("slots"),
                }
            )
        if isinstance(source, str) and source in outgoing:
            outgoing[source].append(
                {
                    "target": endpoint(target),
                    "target_local_slots": route.get("slots"),
                }
            )

    must_run_after: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    must_run_before: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    for relation in values["order"]:
        if not isinstance(relation, Mapping):
            continue
        before = relation.get("before")
        after = relation.get("after")
        if isinstance(after, str) and after in must_run_after:
            must_run_after[after].append(
                {"before": endpoint(before)}
            )
        if isinstance(before, str) and before in must_run_before:
            must_run_before[before].append(
                {"after": endpoint(after)}
            )

    sharing_roles: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    for relation in values["sharing"]:
        if not isinstance(relation, Mapping):
            continue
        source = relation.get("source")
        consumers = relation.get("consumers", [])
        if isinstance(source, str) and source in sharing_roles:
            sharing_roles[source].append(
                {
                    "role": "source",
                    "consumers": [endpoint(item) for item in consumers]
                    if isinstance(consumers, list)
                    else consumers,
                }
            )
        if isinstance(consumers, list):
            for consumer in consumers:
                if isinstance(consumer, str) and consumer in sharing_roles:
                    sharing_roles[consumer].append(
                        {
                            "role": "consumer",
                            "source": endpoint(source),
                        }
                    )

    temporal_roles: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    for relation_name, role_names in (
        ("iterations", ("loop_instance", "tile_instance")),
        (
            "loop_carriers",
            ("loop_instance", "init", "state", "next", "result"),
        ),
        (
            "physical_iterations",
            ("body_instance", "tile_instance"),
        ),
        (
            "unit_iterations",
            ("body_instance",),
        ),
        (
            "physical_loop_carriers",
            ("init", "state", "next", "result"),
        ),
    ):
        for index, relation in enumerate(values[relation_name]):
            if not isinstance(relation, Mapping):
                continue
            for role in role_names:
                instance_id = relation.get(role)
                if isinstance(instance_id, str) and instance_id in temporal_roles:
                    temporal_roles[instance_id].append(
                        {
                            "relation": relation_name,
                            "original_index": index,
                            "role": role,
                            "authored_relation": dict(relation),
                        }
                    )

    refinements: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in instance_index
    }
    for relation in values["refinements"]:
        if not isinstance(relation, Mapping):
            continue
        instance_id = relation.get("instance")
        if isinstance(instance_id, str) and instance_id in refinements:
            refinements[instance_id].append(dict(relation))

    records: list[dict[str, Any]] = []
    for instance in values["instances"]:
        if not isinstance(instance, Mapping):
            continue
        instance_id = instance.get("id")
        if not isinstance(instance_id, str):
            continue
        semantic_ref = instance.get("semantic_ref")
        region_id = instance.get("region")
        region = region_index.get(region_id)
        records.append(
            {
                "id": instance_id,
                "semantic_ref": semantic_ref,
                "semantic": semantic_index.get(
                    semantic_ref,
                    {"semantic_kind": "unclassified_handle"},
                ),
                "mechanism": instance.get("mechanism"),
                "region": (
                    dict(region)
                    if isinstance(region, Mapping)
                    else {"id": region_id, "unresolved": True}
                ),
                "incoming_routes": incoming.get(instance_id, []),
                "outgoing_routes": outgoing.get(instance_id, []),
                "must_run_after": must_run_after.get(instance_id, []),
                "must_run_before": must_run_before.get(instance_id, []),
                "sharing_roles": sharing_roles.get(instance_id, []),
                "temporal_roles": temporal_roles.get(instance_id, []),
                "refinements": refinements.get(instance_id, []),
            }
        )
    return records


def _packet_obligations(
    packet: Mapping[str, Any], start: int, count: int
) -> dict[str, Any]:
    records = _packet_obligation_records(packet)
    items, next_start = _bounded_page(records, start=start, count=count)
    compact_items: list[list[Any]] = []
    for item in items:
        semantic = item["semantic"]
        semantic_kind = semantic.get("semantic_kind")
        if semantic_kind == "intent_op":
            semantic_view = [
                semantic_kind,
                semantic.get("op"),
                semantic.get("scope_id"),
            ]
        elif semantic_kind == "boundary_access":
            semantic_view = [
                semantic_kind,
                semantic.get("access_kind"),
                semantic.get("tensor"),
                semantic.get("node_ref"),
            ]
        elif semantic_kind == "boundary_effect":
            semantic_view = [
                semantic_kind,
                semantic.get("effect_kind"),
                semantic.get("target"),
                semantic.get("node_ref"),
                semantic.get("access_ref"),
            ]
        elif semantic_kind == "signature_parameter":
            semantic_view = [
                semantic_kind,
                semantic.get("kind"),
                semantic.get("mode"),
                semantic.get("logical_ref"),
            ]
        else:
            semantic_view = [semantic_kind]
        region = item["region"]
        compact_items.append(
            [
                item["id"],
                item["semantic_ref"],
                item["mechanism"],
                [
                    region.get("id"),
                    region.get("parent"),
                    region.get("placement"),
                ],
                semantic_view,
                [
                    [route["source"]["id"], route["target_local_slots"]]
                    for route in item["incoming_routes"]
                ],
                [
                    [route["target"]["id"], route["target_local_slots"]]
                    for route in item["outgoing_routes"]
                ],
                [relation["before"]["id"] for relation in item["must_run_after"]],
                [relation["after"]["id"] for relation in item["must_run_before"]],
                [
                    [
                        relation["role"],
                        (
                            [peer["id"] for peer in relation["consumers"]]
                            if relation["role"] == "source"
                            else relation["source"]["id"]
                        ),
                    ]
                    for relation in item["sharing_roles"]
                ],
                [
                    [
                        relation["relation"],
                        relation["original_index"],
                        relation["role"],
                        relation["authored_relation"],
                    ]
                    for relation in item["temporal_roles"]
                ],
                [
                    [relation.get("kind"), relation.get("auth_ref")]
                    for relation in item["refinements"]
                ],
            ]
        )
    return {
        "protocol": "intentir.replay-instance-obligations.v2",
        "meaning": (
            "Rows are a compact declaration-order index over this packet's own "
            "already-authored J/pi facts. Columns name every row position. The "
            "view infers no edge, value, operation, source construct, or "
            "realization choice; use obligation HANDLE to expand one current "
            "instance or semantic anchor."
        ),
        "total": len(records),
        "start": start,
        "columns": [
            "id",
            "semantic_ref",
            "mechanism",
            "region[id,parent,placement]",
            "semantic",
            "incoming_routes[source,slots]",
            "outgoing_routes[target,slots]",
            "order_predecessors",
            "order_successors",
            "sharing[role,peers]",
            "temporal[relation,index,role,authored_relation]",
            "refinements[kind,auth_ref]",
        ],
        "items": compact_items,
        "next_start": next_start,
        "next": (
            f"{INTENTIR_PACKET_QUERY} obligations {next_start} {count}"
            if next_start is not None
            else None
        ),
    }


def _packet_obligation(packet: Mapping[str, Any], handle: str) -> dict[str, Any]:
    records = _packet_obligation_records(packet)
    matches = [
        item
        for item in records
        if item["id"] == handle or item["semantic_ref"] == handle
    ]
    if not matches:
        raise KeyError(f"no pi instance or semantic anchor matches {handle!r}")
    shown = matches[:4]
    omitted = matches[4:]
    return {
        "protocol": "intentir.replay-instance-obligation.v1",
        "meaning": (
            "This is a bounded local join of authored facts only; it infers no "
            "edge, value, operation, source construct, or realization choice."
        ),
        "query": handle,
        "match_total": len(matches),
        "items": shown,
        "omitted_instance_ids": [item["id"] for item in omitted],
        "omitted_queries": [
            f"{INTENTIR_PACKET_QUERY} obligation {item['id']}" for item in omitted
        ],
    }


def _packet_find(packet: Mapping[str, Any], needle: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if len(matches) >= 96:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        elif needle.casefold() in str(value).casefold():
            visible = value
            if isinstance(value, str) and len(value) > 512:
                visible = value[:512] + "...[truncated]"
            matches.append({"path": path, "value": visible})

    visit(packet, "")
    return {
        "query": needle,
        "match_count_shown": len(matches),
        "matches": matches,
    }


def _query_packet(packet: Mapping[str, Any], arguments: list[str]) -> Any:
    if arguments == [INTENTIR_PACKET_QUERY, "overview"]:
        return _packet_overview(packet)
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "abi",
    ]:
        return _packet_abi(packet, int(arguments[2]), int(arguments[3]))
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "graph",
    ]:
        return _packet_graph(packet, int(arguments[2]), int(arguments[3]))
    if arguments == [INTENTIR_PACKET_QUERY, "realization"]:
        return _packet_realization(packet)
    if arguments == [INTENTIR_PACKET_QUERY, "requirements"]:
        return _packet_requirements(packet)
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "edges",
    ]:
        return _packet_edges(packet, int(arguments[2]), int(arguments[3]))
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "routes",
    ]:
        return _packet_route_obligations(
            packet, int(arguments[2]), int(arguments[3])
        )
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "order",
    ]:
        return _packet_order_obligations(
            packet, int(arguments[2]), int(arguments[3])
        )
    if len(arguments) == 4 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "obligations",
    ]:
        return _packet_obligations(
            packet, int(arguments[2]), int(arguments[3])
        )
    if len(arguments) == 3 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "obligation",
    ]:
        return _packet_obligation(packet, arguments[2])
    if len(arguments) == 3 and arguments[:2] == [
        INTENTIR_PACKET_QUERY,
        "section",
    ]:
        path = arguments[2]
        value = _packet_path(packet, path)
        rendered = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False
        )
        if len(rendered) <= MAX_PACKET_QUERY_OUTPUT_BYTES:
            return value
        return {
            "path": path,
            "json_chars": _packet_json_chars(value),
            "children": _packet_children(value, path),
        }
    if (
        len(arguments) == 3
        and arguments[:2] == [INTENTIR_PACKET_QUERY, "find"]
        and 0 < len(arguments[2]) <= 128
    ):
        return _packet_find(packet, arguments[2])
    raise ValueError(
        "usage: intentir_packet_query overview | abi START COUNT | graph START COUNT | "
        "requirements | realization | routes START COUNT | order START COUNT | obligations START COUNT | "
        "obligation HANDLE | edges START COUNT | section PATH | find HANDLE"
    )


def _ast_qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    try:
        return ast.unparse(node)
    except (TypeError, ValueError):
        return type(node).__name__


def _ast_text(nodes: list[ast.AST], *, limit: int = 320) -> str:
    parts: list[str] = []
    for node in nodes:
        try:
            parts.append(ast.unparse(node))
        except (TypeError, ValueError):
            parts.append(type(node).__name__)
    value = " | ".join(parts)
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def _statement_header_nodes(statement: ast.stmt) -> list[ast.AST]:
    if isinstance(statement, ast.Assign):
        return [*statement.targets, statement.value]
    if isinstance(statement, ast.AnnAssign):
        return [
            statement.target,
            statement.annotation,
            *([statement.value] if statement.value is not None else []),
        ]
    if isinstance(statement, ast.AugAssign):
        return [statement.target, statement.value]
    if isinstance(statement, ast.Expr):
        return [statement.value]
    if isinstance(statement, ast.Return):
        return [statement.value] if statement.value is not None else []
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return [statement.target, statement.iter]
    if isinstance(statement, (ast.If, ast.While)):
        return [statement.test]
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        result: list[ast.AST] = []
        for item in statement.items:
            result.append(item.context_expr)
            if item.optional_vars is not None:
                result.append(item.optional_vars)
        return result
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments: list[ast.AST] = [
            *statement.decorator_list,
            *statement.args.defaults,
            *statement.args.kw_defaults,
        ]
        if statement.returns is not None:
            arguments.append(statement.returns)
        return [item for item in arguments if item is not None]
    if isinstance(statement, ast.Raise):
        return [
            item
            for item in (statement.exc, statement.cause)
            if item is not None
        ]
    if isinstance(statement, ast.Assert):
        return [
            item for item in (statement.test, statement.msg) if item is not None
        ]
    return []


def _statement_defined_names(statement: ast.stmt) -> list[str]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [statement.name]
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return sorted(
            alias.asname or alias.name.split(".")[0] for alias in statement.names
        )
    names = {
        node.id
        for header in _statement_header_nodes(statement)
        for node in ast.walk(header)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return sorted(names)


def _replay_source_statement_records(source: str) -> tuple[ast.Module, list[dict[str, Any]]]:
    """Return syntax-only records over the Replay writer's own source bytes."""

    tree = ast.parse(source, filename="replay.source")
    records: list[dict[str, Any]] = []

    def visit_block(
        statements: list[ast.stmt],
        *,
        prefix: str,
        depth: int,
        region: str,
    ) -> None:
        for index, statement in enumerate(statements):
            path = f"{prefix}.{index}"
            header_nodes = _statement_header_nodes(statement)
            used_names = sorted(
                {
                    node.id
                    for header in header_nodes
                    for node in ast.walk(header)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
            )
            calls = sorted(
                {
                    _ast_qualified_name(node.func)
                    for header in header_nodes
                    for node in ast.walk(header)
                    if isinstance(node, ast.Call)
                }
            )
            records.append(
                {
                    "path": path,
                    "kind": type(statement).__name__,
                    "line": getattr(statement, "lineno", None),
                    "end_line": getattr(statement, "end_lineno", None),
                    "depth": depth,
                    "region": region,
                    "defines": _statement_defined_names(statement),
                    "uses": used_names,
                    "calls": calls,
                    "header": _ast_text(header_nodes),
                    "body_count": len(getattr(statement, "body", [])),
                    "orelse_count": len(getattr(statement, "orelse", [])),
                }
            )
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_block(
                    statement.body,
                    prefix=f"{path}.body",
                    depth=depth + 1,
                    region=f"function:{statement.name}",
                )
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While, ast.If)):
                visit_block(
                    statement.body,
                    prefix=f"{path}.body",
                    depth=depth + 1,
                    region=f"{path}:body",
                )
                visit_block(
                    statement.orelse,
                    prefix=f"{path}.orelse",
                    depth=depth + 1,
                    region=f"{path}:orelse",
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                visit_block(
                    statement.body,
                    prefix=f"{path}.body",
                    depth=depth + 1,
                    region=f"{path}:body",
                )
            elif isinstance(statement, ast.Try):
                visit_block(
                    statement.body,
                    prefix=f"{path}.body",
                    depth=depth + 1,
                    region=f"{path}:body",
                )
                for handler_index, handler in enumerate(statement.handlers):
                    visit_block(
                        handler.body,
                        prefix=f"{path}.handlers.{handler_index}.body",
                        depth=depth + 1,
                        region=f"{path}:handler:{handler_index}",
                    )
                visit_block(
                    statement.orelse,
                    prefix=f"{path}.orelse",
                    depth=depth + 1,
                    region=f"{path}:orelse",
                )
                visit_block(
                    statement.finalbody,
                    prefix=f"{path}.finalbody",
                    depth=depth + 1,
                    region=f"{path}:finalbody",
                )
            elif isinstance(statement, ast.Match):
                for case_index, case in enumerate(statement.cases):
                    visit_block(
                        case.body,
                        prefix=f"{path}.cases.{case_index}.body",
                        depth=depth + 1,
                        region=f"{path}:case:{case_index}",
                    )

    visit_block(tree.body, prefix="module", depth=0, region="module")
    return tree, records


def _inspect_replay_source(source: str, arguments: list[str]) -> dict[str, Any]:
    tree, records = _replay_source_statement_records(source)
    authority = (
        "syntax-only inventory of the Replay writer's current replay.source; "
        "reads no packet, Source, Harness, Gate, compiler result, prior Replay, "
        "or other candidate and performs no comparison, repair, or generation"
    )
    if arguments == ["overview"]:
        call_counts = Counter(
            _ast_qualified_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        control_kinds = {
            "For",
            "AsyncFor",
            "While",
            "If",
            "Return",
            "Break",
            "Continue",
        }
        controls = [item for item in records if item["kind"] in control_kinds]
        functions = [
            {
                "name": node.name,
                "line": node.lineno,
                "parameters": [
                    argument.arg
                    for argument in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                ],
                "decorators": [
                    _ast_qualified_name(decorator.func)
                    if isinstance(decorator, ast.Call)
                    else _ast_qualified_name(decorator)
                    for decorator in node.decorator_list
                ],
            }
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return {
            "protocol": "intentir.replay-own-source-inventory.v1",
            "authority": authority,
            "functions": functions,
            "statement_count": len(records),
            "call_counts": dict(sorted(call_counts.items())),
            "control_counts": dict(
                sorted(Counter(item["kind"] for item in controls).items())
            ),
            "controls": controls[:32],
            "controls_omitted": max(0, len(controls) - 32),
            "next": (
                f"{INTENTIR_REPLAY_INSPECT} /output/replay.source statements 0 16"
            ),
        }
    if len(arguments) == 3 and arguments[0] == "statements":
        start = int(arguments[1])
        count = int(arguments[2])
        items, next_start = _bounded_page(records, start=start, count=count)
        return {
            "protocol": "intentir.replay-own-source-statements.v1",
            "authority": authority,
            "total": len(records),
            "start": start,
            "items": items,
            "next_start": next_start,
            "next": (
                f"{INTENTIR_REPLAY_INSPECT} /output/replay.source statements "
                f"{next_start} {count}"
                if next_start is not None
                else None
            ),
        }
    if len(arguments) == 2 and arguments[0] == "find" and arguments[1]:
        needle = arguments[1]
        matches = [
            item
            for item in records
            if needle in item["defines"]
            or needle in item["uses"]
            or needle in item["calls"]
            or needle in item["header"]
        ]
        return {
            "protocol": "intentir.replay-own-source-find.v1",
            "authority": authority,
            "query": needle,
            "match_total": len(matches),
            "items": matches[:32],
            "omitted": max(0, len(matches) - 32),
        }
    raise ValueError(
        "usage: intentir_replay_inspect /output/replay.source overview | "
        "statements START COUNT | find NAME"
    )


def _remove_path_iterative(path: Path) -> None:
    """Remove one exact private workspace tree without Python recursion."""

    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_stat.st_mode) or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    stack: list[tuple[Path, bool]] = [(path, False)]
    while stack:
        current, visited = stack.pop()
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(current_stat.st_mode) or current.is_symlink():
            current.unlink(missing_ok=True)
            continue
        os.chmod(current, stat.S_IRWXU)
        if visited:
            current.rmdir()
            continue
        stack.append((current, True))
        with os.scandir(current) as iterator:
            for entry in iterator:
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append((child, False))
                else:
                    child.unlink(missing_ok=True)


def _closed_failure(rule_code: str, *, path: str = "<replay>") -> dict[str, Any]:
    return {
        "protocol": REPLAY_CHECK_PROTOCOL,
        "status": "fail",
        "diagnostics": [
            {
                "rule_code": rule_code,
                "path": path,
                "anonymous_handles": [],
                "line": None,
            }
        ],
    }


def _is_static_elf(path: Path) -> bool:
    """Return true only for an ELF executable without a PT_INTERP header."""

    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 52 or header[:4] != b"\x7fELF":
            return False
        elf_class = header[4]
        byte_order = header[5]
        endian = "<" if byte_order == 1 else ">" if byte_order == 2 else ""
        if not endian:
            return False
        if elf_class == 2:
            program_offset = struct.unpack_from(endian + "Q", header, 32)[0]
            entry_size = struct.unpack_from(endian + "H", header, 54)[0]
            entry_count = struct.unpack_from(endian + "H", header, 56)[0]
        elif elf_class == 1:
            program_offset = struct.unpack_from(endian + "I", header, 28)[0]
            entry_size = struct.unpack_from(endian + "H", header, 42)[0]
            entry_count = struct.unpack_from(endian + "H", header, 44)[0]
        else:
            return False
        if entry_size < 4 or entry_count > 4096:
            return False
        for index in range(entry_count):
            stream.seek(program_offset + index * entry_size)
            entry = stream.read(entry_size)
            if len(entry) != entry_size:
                return False
            if struct.unpack_from(endian + "I", entry)[0] == 3:
                return False
    return True


@dataclass
class IntentIRReplayWorkspace:
    """Fresh host directories projected into exactly one Replay sandbox."""

    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str | bytes]) -> "IntentIRReplayWorkspace":
        if set(files) != REPLAY_INPUT_FILES:
            raise ValueError(
                "Replay input must contain exactly the fixed allowlisted files"
            )
        root = Path(tempfile.mkdtemp(prefix="intentir-replay-"))
        workspace = cls(
            root=root,
            input_dir=root / "input",
            output_dir=root / "output",
            scratch_dir=root / "work",
            tmp_dir=root / "tmp",
        )
        for directory in (
            workspace.input_dir,
            workspace.output_dir,
            workspace.scratch_dir,
            workspace.tmp_dir,
        ):
            directory.mkdir()
        try:
            for relative, content in files.items():
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError(f"unsafe Replay input path: {relative!r}")
                destination = workspace.input_dir / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    destination.write_bytes(content)
                else:
                    destination.write_text(content, encoding="utf-8")
                destination.chmod(stat.S_IRUSR)
        except Exception:
            workspace.cleanup()
            raise
        return workspace

    @property
    def replay_path(self) -> Path:
        return self.output_dir / "replay.source"

    def read_replay(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.replay_path,):
            names = sorted(path.name for path in entries)
            raise ValueError(
                "Replay output must contain exactly replay.source; "
                f"found {names}"
            )
        if self.replay_path.is_symlink() or not self.replay_path.is_file():
            raise ValueError("replay.source must be a regular file")
        size = self.replay_path.stat().st_size
        if not 1 <= size <= MAX_REPLAY_BYTES:
            raise ValueError(
                f"replay.source size must be in [1, {MAX_REPLAY_BYTES}]"
            )
        raw = self.replay_path.read_bytes()
        if len(raw) != size:
            raise ValueError("replay.source changed while being read")
        return raw.decode("utf-8")

    def cleanup(self) -> None:
        _remove_path_iterative(self.root)


class IntentIRReplayEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path | None = None
    timeout: int = 60
    replay_check_limit: int = 0
    replay_submission_check_limit: int = 0
    replay_compile_check_limit: int = 0
    replay_source_inspect_limit: int = 0


class IntentIRReplayEnvironment:
    """Run shell actions without Source, host code, network, or compiler access."""

    def __init__(
        self,
        *,
        config_class: type[IntentIRReplayEnvironmentConfig] = IntentIRReplayEnvironmentConfig,
        replay_checker: Callable[[str], Mapping[str, Any]] | None = None,
        replay_compile_checker: Callable[[str], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        if self.config.replay_check_limit < 0:
            raise ValueError("replay_check_limit must be non-negative")
        if self.config.replay_submission_check_limit < 0:
            raise ValueError("replay_submission_check_limit must be non-negative")
        if self.config.replay_compile_check_limit < 0:
            raise ValueError("replay_compile_check_limit must be non-negative")
        if self.config.replay_source_inspect_limit < 0:
            raise ValueError("replay_source_inspect_limit must be non-negative")
        self._replay_checker = replay_checker
        self._replay_compile_checker = replay_compile_checker
        self._check_calls = 0
        self._check_evaluations = 0
        self._calls_by_trigger = {"explicit": 0, "submission": 0}
        self._evaluations_by_trigger = {"explicit": 0, "submission": 0}
        self._shared_evaluations = 0
        self._submission_reserved_evaluations = 0
        self._check_audit: list[dict[str, Any]] = []
        self._compile_check_calls = 0
        self._compile_check_evaluations = 0
        self._compile_check_audit: list[dict[str, Any]] = []
        self._compile_passed_bytes: bytes | None = None
        self._source_inspect_calls = 0
        self._source_inspect_audit: list[str] = []
        self._obligation_audited_source_digests: set[str] = set()
        self._obligation_audit_log: list[dict[str, Any]] = []
        self._passed_bytes: bytes | None = None
        self._committed_bytes: bytes | None = None
        self._commit_audit: list[dict[str, Any]] = []
        self._rollback_count = 0
        self._fallback_audit: list[dict[str, Any]] = []
        self._apply_patch_audit: list[dict[str, Any]] = []
        self._direct_submit_audit: list[dict[str, Any]] = []
        self._multi_action_rejections = 0
        self._validate_paths()
        self._probe()

    def validate_actions(self, actions: list[dict[str, Any]]) -> str | None:
        """Permit a batch only when every call is a source-blind packet query."""

        if len(actions) <= 1:
            return None
        if len(actions) > 4:
            self._multi_action_rejections += 1
            return (
                f"{MULTI_ACTION_PROTOCOL}: at most four independent source-blind "
                "packet queries are allowed; no action in this response was executed"
            )
        allowed = True
        for action in actions:
            if not isinstance(action, Mapping):
                allowed = False
                break
            command = action.get("command")
            if not isinstance(command, str):
                allowed = False
                break
            try:
                arguments = shlex.split(command)
            except ValueError:
                allowed = False
                break
            if not arguments or arguments[0] != INTENTIR_PACKET_QUERY:
                allowed = False
                break
        if allowed:
            return None
        self._multi_action_rejections += 1
        return (
            f"{MULTI_ACTION_PROTOCOL}: multiple Replay calls are allowed only "
            "for independent intentir_packet_query actions; no action in this "
            "response was executed"
        )

    @staticmethod
    def _transaction_output(
        *,
        action: str,
        status: str,
        source_bytes: bytes | None = None,
        diagnostic: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "protocol": REPLAY_TRANSACTION_PROTOCOL,
            "action": action,
            "status": status,
        }
        if source_bytes is not None:
            value["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
            value["source_bytes"] = len(source_bytes)
        if diagnostic is not None:
            value["diagnostic"] = diagnostic
        return {
            "output": json.dumps(value, sort_keys=True) + "\n",
            "returncode": 0 if status == "pass" else 1,
            "exception_info": "",
        }

    def _restore_committed_source(self) -> bytes:
        committed = self._committed_bytes
        if committed is None:
            raise ValueError("no Replay source has been explicitly committed")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".replay-restore-",
            dir=self.config.output_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(committed)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.config.output_dir / "replay.source")
        finally:
            temporary.unlink(missing_ok=True)
        self._passed_bytes = committed
        return committed

    def _execute_replay_commit(self, arguments: list[str]) -> dict[str, Any]:
        if arguments != [INTENTIR_REPLAY_COMMIT, "/output/replay.source"]:
            return self._transaction_output(
                action="commit",
                status="fail",
                diagnostic="usage: intentir_replay_commit /output/replay.source",
            )
        try:
            source_bytes, _source_text = self._read_check_target(
                "/output/replay.source"
            )
        except Exception as exc:
            return self._transaction_output(
                action="commit",
                status="fail",
                diagnostic=f"Replay read failed: {type(exc).__name__}: {exc}",
            )
        if source_bytes != self._passed_bytes:
            return self._transaction_output(
                action="commit",
                status="fail",
                source_bytes=source_bytes,
                diagnostic=(
                    "the exact current bytes do not have ReplayCheck PASS; "
                    "check this complete source first"
                ),
            )
        if (
            self._replay_compile_checker is not None
            and source_bytes != self._compile_passed_bytes
        ):
            return self._transaction_output(
                action="commit",
                status="fail",
                source_bytes=source_bytes,
                diagnostic=(
                    "the exact current bytes do not have construction compile "
                    "PASS; compile-check this complete source first"
                ),
            )
        self._committed_bytes = source_bytes
        self._commit_audit.append(
            {
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
            }
        )
        return self._transaction_output(
            action="commit", status="pass", source_bytes=source_bytes
        )

    def _execute_replay_rollback(self, arguments: list[str]) -> dict[str, Any]:
        if arguments != [INTENTIR_REPLAY_ROLLBACK, "/output/replay.source"]:
            return self._transaction_output(
                action="rollback",
                status="fail",
                diagnostic="usage: intentir_replay_rollback /output/replay.source",
            )
        try:
            source_bytes = self._restore_committed_source()
        except Exception as exc:
            return self._transaction_output(
                action="rollback",
                status="fail",
                diagnostic=f"rollback unavailable: {type(exc).__name__}: {exc}",
            )
        self._rollback_count += 1
        return self._transaction_output(
            action="rollback", status="pass", source_bytes=source_bytes
        )

    def _execute_replay_patch(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one exact Replay-LLM-authored text replacement atomically."""

        path = action.get("path")
        old_text = action.get("old_text")
        new_text = action.get("new_text")
        if (
            path != "/output/replay.source"
            or not isinstance(old_text, str)
            or not old_text
            or not isinstance(new_text, str)
        ):
            result = {
                "protocol": REPLAY_APPLY_PATCH_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "apply_replay_patch requires /output/replay.source, one "
                    "non-empty old_text string, and one new_text string"
                ),
            }
            return {
                "output": json.dumps(result, sort_keys=True) + "\n",
                "returncode": 1,
                "exception_info": "",
            }
        source_path = self.config.output_dir / "replay.source"
        try:
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError("replay.source must already be a regular file")
            before = source_path.read_bytes()
            if not 1 <= len(before) <= MAX_REPLAY_BYTES:
                raise ValueError("Replay source size is outside the allowed range")
            before_text = before.decode("utf-8")
            occurrences = before_text.count(old_text)
            if occurrences != 1:
                raise ValueError(
                    f"old_text must occur exactly once; observed {occurrences}"
                )
            after = before_text.replace(old_text, new_text, 1).encode("utf-8")
            if not 1 <= len(after) <= MAX_REPLAY_BYTES:
                raise ValueError("patched Replay source is outside the allowed range")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".replay-patch-", dir=self.config.output_dir
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(after)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, source_path)
            finally:
                temporary.unlink(missing_ok=True)
        except Exception as exc:
            result = {
                "protocol": REPLAY_APPLY_PATCH_PROTOCOL,
                "status": "fail",
                "diagnostic": f"{type(exc).__name__}: {exc}",
            }
            return {
                "output": json.dumps(result, sort_keys=True) + "\n",
                "returncode": 1,
                "exception_info": "",
            }
        self._passed_bytes = None
        receipt = {
            "protocol": REPLAY_APPLY_PATCH_PROTOCOL,
            "status": "pass",
            "path": path,
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest(),
            "before_bytes": len(before),
            "after_bytes": len(after),
            "submit_requested": action.get("submit") is True,
            "authority": "exact Replay-LLM-authored text replacement only",
        }
        self._apply_patch_audit.append(dict(receipt))
        if action.get("submit") is True:
            output = {
                "output": (
                    INTENTIR_REPLAY_SUBMIT
                    + "\n"
                    + json.dumps(receipt, sort_keys=True)
                    + "\n"
                ),
                "returncode": 0,
                "exception_info": "",
            }
            self._check_submission(output)
            return output
        return {
            "output": json.dumps(receipt, sort_keys=True) + "\n",
            "returncode": 0,
            "exception_info": "",
        }

    def _execute_submit_replay(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Write and submit complete exact source bytes supplied by the Replay LLM."""

        source_text = action.get("source_text")
        if not isinstance(source_text, str) or not source_text:
            result = {
                "protocol": REPLAY_DIRECT_SUBMIT_PROTOCOL,
                "status": "fail",
                "diagnostic": "submit_replay requires non-empty source_text",
            }
            return {
                "output": json.dumps(result, sort_keys=True) + "\n",
                "returncode": 1,
                "exception_info": "",
            }
        source_bytes = source_text.encode("utf-8")
        if not 1 <= len(source_bytes) <= MAX_REPLAY_BYTES:
            result = {
                "protocol": REPLAY_DIRECT_SUBMIT_PROTOCOL,
                "status": "fail",
                "diagnostic": "source_text size is outside the allowed range",
            }
            return {
                "output": json.dumps(result, sort_keys=True) + "\n",
                "returncode": 1,
                "exception_info": "",
            }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".replay-direct-submit-", dir=self.config.output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(source_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.config.output_dir / "replay.source")
        finally:
            temporary.unlink(missing_ok=True)
        self._passed_bytes = None
        receipt = {
            "protocol": REPLAY_DIRECT_SUBMIT_PROTOCOL,
            "status": "written",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_bytes": len(source_bytes),
            "authority": (
                "exact Replay-LLM-authored bytes; no source parsing, generation, "
                "or repair"
            ),
        }
        self._direct_submit_audit.append(dict(receipt))
        output = {
            "output": (
                INTENTIR_REPLAY_SUBMIT
                + "\n"
                + json.dumps(receipt, sort_keys=True)
                + "\n"
            ),
            "returncode": 0,
            "exception_info": "",
        }
        self._check_submission(output)
        return output

    def activate_committed_fallback(self, terminal_status: str) -> str | None:
        """Restore only source bytes explicitly committed by this Replay LLM."""

        if terminal_status not in {
            "LimitsExceeded",
            "TimeExceeded",
            "RepeatedFormatError",
            "OrchestrationStall",
        } or self._committed_bytes is None:
            return None
        source_bytes = self._restore_committed_source()
        self._fallback_audit.append(
            {
                "terminal_status": terminal_status,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
            }
        )
        return source_bytes.decode("utf-8")

    def stagnation_state(self) -> dict[str, Any]:
        """Return only the exact current Replay-source byte identity."""

        path = self.config.output_dir / "replay.source"
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Replay source absent")
            raw = path.read_bytes()
        except (OSError, ValueError):
            return {"protocol": "intentir.replay-stagnation-state.v1", "source": None}
        return {
            "protocol": "intentir.replay-stagnation-state.v1",
            "source": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        }

    def material_progress_state(self) -> dict[str, Any]:
        """Return exact identities of model-writable Replay and work bytes."""

        files: list[dict[str, Any]] = []
        for label, root in (
            ("output", self.config.output_dir),
            ("work", self.config.scratch_dir),
        ):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    files.append(
                        {"root": label, "path": relative, "kind": "symlink"}
                    )
                    continue
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                files.append(
                    {
                        "root": label,
                        "path": relative,
                        "kind": "file",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        return {
            "protocol": "intentir.replay-material-progress.v1",
            "files": files,
        }

    def _validate_paths(self) -> None:
        if self.config.tmp_dir is None:
            self.config.tmp_dir = self.config.scratch_dir.parent / "tmp"
        paths = (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        )
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Replay input, output, work, and tmp directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in paths):
            raise ValueError("Replay sandbox paths must be real directories")
        (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ) = resolved

    def _base_command(self) -> list[str]:
        command = [
            str(REPLAY_BWRAP),
            "--unshare-all",
            "--unshare-user-try",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--hostname",
            REPLAY_SANDBOX_HOSTNAME,
            "--tmpfs",
            "/",
            "--dir",
            "/bin",
        ]
        # Only these entrypoint names are mounted. BusyBox remains a multicall
        # binary and could be copied/renamed inside a writable bind; the hard
        # boundary is therefore the empty rootfs, absent host tools/data, clear
        # environment, and new network/PID namespaces—not an applet claim.
        for applet in REPLAY_TOOL_APPLETS:
            command.extend(
                ["--ro-bind", str(REPLAY_BUSYBOX), f"/bin/{applet}"]
            )
        command.extend(
            [
                "--dev",
                "/dev",
                "--ro-bind",
                str(self.config.input_dir),
                "/input",
                "--bind",
                str(self.config.output_dir),
                "/output",
                "--bind",
                str(self.config.scratch_dir),
                "/work",
                "--bind",
                str(self.config.tmp_dir),
                "/tmp",
                "--chdir",
                "/work",
                "--remount-ro",
                "/dev",
                "--remount-ro",
                "/",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "LANG",
                "C",
                "--setenv",
                "LC_ALL",
                "C",
                "--setenv",
                "PATH",
                "/bin",
                "--setenv",
                "TMPDIR",
                "/tmp",
            ]
        )
        return command

    def _probe(self) -> None:
        if (
            REPLAY_BWRAP.is_symlink()
            or not REPLAY_BWRAP.is_file()
            or REPLAY_BUSYBOX.is_symlink()
            or not REPLAY_BUSYBOX.is_file()
            or not _is_static_elf(REPLAY_BUSYBOX)
        ):
            raise RuntimeError("IntentIR Replay requires static /usr/bin/busybox")
        busybox_applets = set(
            subprocess.run(
                [str(REPLAY_BUSYBOX), "--list"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                timeout=10,
            ).stdout.splitlines()
        )
        missing_applets = sorted(set(REPLAY_TOOL_APPLETS) - busybox_applets)
        if missing_applets:
            raise RuntimeError(
                "IntentIR Replay BusyBox lacks required entrypoints: "
                + ", ".join(missing_applets)
            )
        command = self._base_command() + [
            "/bin/sh",
            "-c",
            (
                "test \"$(hostname)\" "
                f"= {REPLAY_SANDBOX_HOSTNAME}; "
                "test ! -e /proc; "
                "test ! -e /usr; test ! -e /etc; test ! -e /lib; "
                "test ! -e /lib64; test ! -e /usr/local; "
                "for name in python python3 gcc g++ clang ld as nvcc ptxas "
                "triton torch intentir; do ! command -v \"$name\"; done; "
                "test \"$(find /input -type f | wc -l)\" -eq 4; "
                "test \"$(find /input -type f | sed 's#^/input/##' | sort | "
                "sha256sum | cut -d' ' -f1)\" = "
                "$(printf '%s\\n' README.md REPLAY_CONTRACT.md TOOL.md "
                "packet.json | sort | sha256sum | cut -d' ' -f1)"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"bubblewrap capability probe failed: {exc}") from exc
        if result.returncode != 0:
            diagnostic = result.stdout.strip()[:500]
            raise RuntimeError(
                "bubblewrap Replay isolation probe failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        namespace_probe = subprocess.Popen(
            self._base_command()
            + [
                "/bin/sh",
                "-c",
                "printf 'ready\\n'; sleep 2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if namespace_probe.stdout is None:
                raise RuntimeError("Replay network namespace probe has no output")
            if namespace_probe.stdout.readline().strip() != "ready":
                raise RuntimeError("Replay network namespace probe did not start")
            host_namespace = os.readlink("/proc/self/ns/net")
            pending = [namespace_probe.pid]
            observed_namespaces: set[str] = set()
            while pending:
                pid = pending.pop()
                try:
                    observed_namespaces.add(
                        os.readlink(f"/proc/{pid}/ns/net")
                    )
                    children = Path(
                        f"/proc/{pid}/task/{pid}/children"
                    ).read_text(encoding="utf-8")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                pending.extend(int(item) for item in children.split())
            if not (observed_namespaces - {host_namespace}):
                raise RuntimeError("Replay sandbox did not unshare the network")
            self._network_namespace_probed = True
        finally:
            namespace_probe.terminate()
            try:
                namespace_probe.wait(timeout=5)
            except subprocess.TimeoutExpired:
                namespace_probe.kill()
                namespace_probe.wait(timeout=5)

    def _read_check_target(self, path: str) -> tuple[bytes, str]:
        target = (
            self.config.scratch_dir / "replay.source"
            if path == "/work/replay.source"
            else self.config.output_dir / "replay.source"
        )
        if target.is_symlink() or not target.is_file():
            raise ValueError("Replay check target must be a regular file")
        size = target.stat().st_size
        if not 1 <= size <= MAX_REPLAY_BYTES:
            raise ValueError("Replay check target is outside the size bound")
        raw = target.read_bytes()
        if len(raw) != size:
            raise ValueError("Replay check target changed while being read")
        return raw, raw.decode("utf-8")

    @staticmethod
    def _validate_check_result(result: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise RuntimeError("Replay checker returned a non-object result")
        value = dict(result)
        protocol = value.get("protocol")
        status = value.get("status")
        diagnostics = value.get("diagnostics")
        valid = (
            set(value) == {"protocol", "status", "diagnostics"}
            and protocol == REPLAY_CHECK_PROTOCOL
            and status in {"pass", "fail"}
            and isinstance(diagnostics, list)
            and len(diagnostics) <= 20
            and ((status == "pass" and not diagnostics) or (status == "fail" and bool(diagnostics)))
        )
        if valid:
            for item in diagnostics:
                if not isinstance(item, Mapping) or set(item) != {
                    "rule_code", "path", "anonymous_handles", "line"
                }:
                    valid = False
                    break
                handles = item.get("anonymous_handles")
                line = item.get("line")
                if (
                    not isinstance(item.get("rule_code"), str)
                    or not isinstance(item.get("path"), str)
                    or len(item["rule_code"]) > MAX_CHECK_FIELD_CHARS
                    or len(item["path"]) > MAX_CHECK_FIELD_CHARS
                    or not isinstance(handles, list)
                    or len(handles) > MAX_CHECK_HANDLES
                    or any(
                        not isinstance(handle, str)
                        or len(handle) > MAX_CHECK_FIELD_CHARS
                        for handle in handles
                    )
                    or (
                        line is not None
                        and (
                            not isinstance(line, int)
                            or isinstance(line, bool)
                            or line < 1
                        )
                    )
                ):
                    valid = False
                    break
        if valid and len(json.dumps(value, sort_keys=True).encode()) > MAX_CHECK_RESULT_BYTES:
            valid = False
        if not valid:
            raise RuntimeError("Replay checker returned an invalid closed result")
        return value

    def _execute_replay_check(
        self,
        arguments: list[str],
        *,
        trigger: str = "explicit",
    ) -> dict[str, Any]:
        if trigger not in {"explicit", "submission"}:
            raise ValueError("unsupported ReplayCheck trigger")
        self._check_calls += 1
        self._calls_by_trigger[trigger] += 1
        source_bytes: bytes | None = None
        budget_pool: str | None = None
        evaluated = False
        if len(arguments) != 2 or arguments[1] not in {
            "/work/replay.source", "/output/replay.source"
        }:
            result = _closed_failure("invalid_replay_path")
        elif self._replay_checker is None:
            result = _closed_failure("replay_check_unavailable")
        else:
            try:
                source_bytes, source_text = self._read_check_target(arguments[1])
            except Exception:
                result = _closed_failure("replay_check_read_failed")
            else:
                if (
                    self.config.replay_check_limit == 0
                    or self._shared_evaluations < self.config.replay_check_limit
                ):
                    self._shared_evaluations += 1
                    budget_pool = "shared"
                elif (
                    trigger == "submission"
                    and self._submission_reserved_evaluations
                    < self.config.replay_submission_check_limit
                ):
                    self._submission_reserved_evaluations += 1
                    budget_pool = "submission_reserve"
                if budget_pool is None:
                    result = _closed_failure("check_budget_exhausted")
                else:
                    evaluated = True
                    self._check_evaluations += 1
                    self._evaluations_by_trigger[trigger] += 1
                    try:
                        result = self._replay_checker(source_text)
                    except Exception:
                        result = _closed_failure("replay_check_internal_failure")
        result = self._validate_check_result(result)
        diagnostics = result["diagnostics"]
        self._check_audit.append(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "trigger": trigger,
                "budget_pool": budget_pool,
                "evaluated": evaluated,
                "rule_paths": [
                    {"rule_code": item["rule_code"], "path": item["path"]}
                    for item in diagnostics
                ],
            }
        )
        if result["status"] == "pass" and evaluated and source_bytes is not None:
            self._passed_bytes = source_bytes
        elif evaluated and source_bytes == self._passed_bytes:
            self._passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if result["status"] == "pass" else 1,
            "exception_info": "",
        }

    def _execute_replay_compile_check(
        self,
        arguments: list[str],
    ) -> dict[str, Any]:
        """Run the source-blind public-profile construction compile probe.

        This only establishes that the Replay LLM's exact bytes are compilable
        enough to leave its authoring episode. It has no Gate1, semantic
        membership, or certification authority.
        """

        self._compile_check_calls += 1
        evaluated = False
        source_bytes: bytes | None = None
        if len(arguments) != 2 or arguments[1] not in {
            "/work/replay.source", "/output/replay.source"
        }:
            result = _closed_failure("invalid_replay_compile_path")
        elif self._replay_compile_checker is None:
            result = _closed_failure("replay_compile_check_unavailable")
        elif (
            self.config.replay_compile_check_limit != 0
            and self._compile_check_evaluations
            >= self.config.replay_compile_check_limit
        ):
            result = _closed_failure("compile_check_budget_exhausted")
        else:
            try:
                source_bytes, source_text = self._read_check_target(arguments[1])
            except Exception:
                result = _closed_failure("replay_compile_check_read_failed")
            else:
                source_digest = hashlib.sha256(source_bytes).hexdigest()
                facts = _packet_critical_facts(self._read_closed_packet())
                audit_required = bool(
                    facts["lexical_regions"] or facts["cross_region_order"]
                )
                if (
                    audit_required
                    and source_digest not in self._obligation_audited_source_digests
                ):
                    result = _closed_failure("replay_obligation_audit_required")
                else:
                    evaluated = True
                    self._compile_check_evaluations += 1
                    try:
                        result = self._replay_compile_checker(source_text)
                    except Exception:
                        result = _closed_failure(
                            "replay_compile_check_internal_failure"
                        )
        result = self._validate_check_result(result)
        diagnostics = result["diagnostics"]
        self._compile_check_audit.append(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "evaluated": evaluated,
                "rule_paths": [
                    {"rule_code": item["rule_code"], "path": item["path"]}
                    for item in diagnostics
                ],
            }
        )
        if result["status"] == "pass" and evaluated and source_bytes is not None:
            self._compile_passed_bytes = source_bytes
        elif evaluated and source_bytes == self._compile_passed_bytes:
            self._compile_passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if result["status"] == "pass" else 1,
            "exception_info": "",
        }

    def _read_closed_packet(self) -> Mapping[str, Any]:
        packet = json.loads(
            (self.config.input_dir / "packet.json").read_text(encoding="utf-8")
        )
        if not isinstance(packet, Mapping) or set(packet) != {
            "j",
            "c",
            "pi",
            "u_f",
        }:
            raise ValueError("Replay packet has an invalid closed root")
        return packet

    def _execute_replay_obligation_audit(
        self, arguments: list[str]
    ) -> dict[str, Any]:
        try:
            if arguments != [
                INTENTIR_REPLAY_OBLIGATION_AUDIT,
                "/output/replay.source",
            ]:
                raise ValueError(
                    "usage: intentir_replay_obligation_audit "
                    "/output/replay.source"
                )
            source_bytes, source_text = self._read_check_target(arguments[1])
            value = _replay_obligation_audit_view(
                self._read_closed_packet(), source_text
            )
            rendered = json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False
            )
            if len(rendered.encode("utf-8")) > MAX_PACKET_QUERY_OUTPUT_BYTES:
                raise ValueError("obligation audit result exceeds the closed bound")
        except Exception as exc:
            return {
                "output": (
                    "Replay obligation audit rejected: "
                    f"{type(exc).__name__}: {exc}\n"
                ),
                "returncode": 1,
                "exception_info": "",
            }
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        self._obligation_audited_source_digests.add(source_digest)
        self._obligation_audit_log.append(
            {
                "source_sha256": source_digest,
                "lexical_region_count": len(value["lexical_regions"]),
                "cross_region_order_count": len(value["cross_region_order"]),
            }
        )
        return {
            "output": rendered + "\n",
            "returncode": 0,
            "exception_info": "",
        }

    def _execute_packet_query(self, arguments: list[str]) -> dict[str, Any]:
        try:
            packet = self._read_closed_packet()
            value = _query_packet(packet, arguments)
            rendered = json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False
            )
            if len(rendered.encode("utf-8")) > MAX_PACKET_QUERY_OUTPUT_BYTES:
                raise ValueError("query result is too large; use a narrower path")
        except Exception as exc:
            return {
                "output": f"packet query rejected: {type(exc).__name__}: {exc}\n",
                "returncode": 1,
                "exception_info": "",
            }
        return {
            "output": rendered + "\n",
            "returncode": 0,
            "exception_info": "",
        }

    def _execute_replay_inspect(self, arguments: list[str]) -> dict[str, Any]:
        self._source_inspect_calls += 1
        mode = arguments[2] if len(arguments) > 2 else "invalid"
        self._source_inspect_audit.append(mode)
        try:
            if (
                self.config.replay_source_inspect_limit != 0
                and self._source_inspect_calls
                > self.config.replay_source_inspect_limit
            ):
                raise ValueError("Replay source inspect budget exhausted")
            if len(arguments) < 3 or arguments[1] not in {
                "/work/replay.source",
                "/output/replay.source",
            }:
                raise ValueError("Replay source inspect requires the private source path")
            _source_bytes, source_text = self._read_check_target(arguments[1])
            value = _inspect_replay_source(source_text, arguments[2:])
            rendered = json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False
            )
            if len(rendered.encode("utf-8")) > MAX_REPLAY_INSPECT_OUTPUT_BYTES:
                raise ValueError(
                    "Replay source inspect result is too large; use statements paging or find"
                )
        except Exception as exc:
            return {
                "output": (
                    "Replay source inspect rejected: "
                    f"{type(exc).__name__}: {exc}\n"
                ),
                "returncode": 1,
                "exception_info": "",
            }
        return {
            "output": rendered + "\n",
            "returncode": 0,
            "exception_info": "",
        }

    def _writable_usage(self) -> tuple[int, int, int]:
        entries = 0
        size = 0
        max_depth = 0
        stack = [
            (self.config.output_dir, 0),
            (self.config.scratch_dir, 0),
            (self.config.tmp_dir, 0),
        ]
        while stack:
            directory, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if max_depth > MAX_WORKSPACE_DEPTH:
                return entries, size, max_depth
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        return entries, size, max_depth
                    item_stat = entry.stat(follow_symlinks=False)
                    size += max(item_stat.st_size, item_stat.st_blocks * 512)
                    if size > MAX_WORKSPACE_BYTES:
                        return entries, size, max_depth
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((Path(entry.path), depth + 1))
        return entries, size, max_depth

    def _reset_writable_dirs(self) -> None:
        for directory in (self.config.output_dir, self.config.scratch_dir, self.config.tmp_dir):
            os.chmod(directory, stat.S_IRWXU)
            for child in directory.iterdir():
                _remove_path_iterative(child)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        if cwd not in {"", "/work"}:
            raise ValueError("IntentIR Replay cwd is fixed at /work")
        command = action.get("command", "")
        if action.get("tool_name") == "apply_replay_patch":
            return self._execute_replay_patch(action)
        if action.get("tool_name") == "submit_replay":
            return self._execute_submit_replay(action)
        if not isinstance(command, str):
            raise TypeError("Replay shell command must be a string")
        try:
            check_arguments = shlex.split(command)
        except ValueError:
            check_arguments = []
        if check_arguments and check_arguments[0] == INTENTIR_PACKET_QUERY:
            return self._execute_packet_query(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_INSPECT:
            return self._execute_replay_inspect(check_arguments)
        if (
            check_arguments
            and check_arguments[0] == INTENTIR_REPLAY_OBLIGATION_AUDIT
        ):
            return self._execute_replay_obligation_audit(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_CHECK:
            return self._execute_replay_check(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_COMMIT:
            return self._execute_replay_commit(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_ROLLBACK:
            return self._execute_replay_rollback(check_arguments)
        if check_arguments and check_arguments[0] == INTENTIR_REPLAY_SUBMIT_COMMAND:
            if check_arguments == [INTENTIR_REPLAY_SUBMIT_COMMAND, "--committed"]:
                try:
                    self._restore_committed_source()
                except Exception as exc:
                    return self._transaction_output(
                        action="submit",
                        status="fail",
                        diagnostic=(
                            "committed submission unavailable: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            elif check_arguments != [
                INTENTIR_REPLAY_SUBMIT_COMMAND,
                "/output/replay.source",
            ]:
                return self._transaction_output(
                    action="submit",
                    status="fail",
                    diagnostic=(
                        "usage: intentir_replay_submit /output/replay.source | "
                        "intentir_replay_submit --committed"
                    ),
                )
            output = {
                "output": INTENTIR_REPLAY_SUBMIT + "\n",
                "returncode": 0,
                "exception_info": "",
            }
            self._check_submission(output)
            return output
        if (
            check_arguments
            and check_arguments[0] == INTENTIR_REPLAY_COMPILE_CHECK
        ):
            return self._execute_replay_compile_check(check_arguments)
        bounded_shell = (
            f"printf '{_SANDBOX_START_MARKER.decode()}'; "
            "ulimit -t 60 -f 8192 -n 64 -p 128 -v 524288; "
            "set -o pipefail; "
            f"/bin/sh -c \"$1\" 2>&1 | "
            f"/bin/head -c {MAX_ACTION_OUTPUT_BYTES}"
        )
        invocation = self._base_command() + [
            "/bin/sh",
            "-c",
            bounded_shell,
            "intentir-replay-action",
            command,
        ]
        process: subprocess.Popen[bytes] | None = None
        limit_error = ""
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(invocation, stdout=output_file, stderr=subprocess.STDOUT)
                started = time.monotonic()
                action_timeout = timeout or self.config.timeout
                while process.poll() is None:
                    if time.monotonic() - started >= action_timeout:
                        limit_error = f"action exceeded {action_timeout} second timeout"
                        process.kill()
                        break
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = f"workspace became unreadable: {type(exc).__name__}"
                        process.kill()
                        break
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = (
                            f"workspace exceeded limit: entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                            f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                            f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                        )
                        process.kill()
                        break
                    time.sleep(0.05)
                returncode = process.wait(timeout=5)
                if not limit_error:
                    entries, size, depth = self._writable_usage()
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = "workspace exceeded limit after action"
                output_file.seek(0)
                raw_output = output_file.read(
                    len(_SANDBOX_START_MARKER) + MAX_ACTION_OUTPUT_BYTES + 1
                )
            marker_at = raw_output.find(_SANDBOX_START_MARKER)
            if marker_at < 0:
                raise RuntimeError("bubblewrap action failed before Replay sandbox start")
            action_output = raw_output[marker_at + len(_SANDBOX_START_MARKER):]
            output = {
                "output": action_output[:MAX_ACTION_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                "returncode": returncode,
                "exception_info": limit_error,
            }
            if len(action_output) >= MAX_ACTION_OUTPUT_BYTES:
                output["returncode"] = returncode or 2
                output["exception_info"] = (
                    f"action output reached {MAX_ACTION_OUTPUT_BYTES} byte limit"
                )
            if limit_error:
                output["returncode"] = -1
                self._reset_writable_dirs()
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise RuntimeError(
                f"IntentIR Replay sandbox action failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._check_submission(output)
        return output

    def _check_submission(self, output: dict[str, Any]) -> None:
        lines = str(output.get("output", "")).lstrip().splitlines()
        if not lines or lines[0].strip() != INTENTIR_REPLAY_SUBMIT or output.get("returncode") != 0:
            return
        workspace = IntentIRReplayWorkspace(
            root=self.config.input_dir.parent,
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            scratch_dir=self.config.scratch_dir,
            tmp_dir=self.config.tmp_dir,
        )
        try:
            source_text = workspace.read_replay()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += f"\nSubmission rejected by output policy: {type(exc).__name__}\n"
            return
        source_bytes = source_text.encode("utf-8")
        if self._replay_checker is not None and source_bytes != self._passed_bytes:
            preflight = self._execute_replay_check(
                [INTENTIR_REPLAY_CHECK, "/output/replay.source"],
                trigger="submission",
            )
            if preflight["returncode"] != 0:
                output["returncode"] = 2
                output["output"] += (
                    "\nSubmission preflight rejected the current Replay source. "
                    "Edit it and submit again.\n" + str(preflight["output"])
                )
                return
        if (
            self._replay_compile_checker is not None
            and source_bytes != self._compile_passed_bytes
        ):
            compile_preflight = self._execute_replay_compile_check(
                [INTENTIR_REPLAY_COMPILE_CHECK, "/output/replay.source"]
            )
            if compile_preflight["returncode"] != 0:
                output["returncode"] = 2
                output["output"] += (
                    "\nSubmission construction compile rejected the current "
                    "Replay source. Edit it and submit again.\n"
                    + str(compile_preflight["output"])
                )
                return
        try:
            final_source = workspace.read_replay()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += f"\nSubmission changed after checking: {type(exc).__name__}\n"
            return
        if final_source.encode("utf-8") != source_bytes:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected because replay.source changed after "
                "the exact bytes were checked.\n"
            )
            return
        raise Submitted(
            {
                "role": "exit",
                "content": "replay.source submitted",
                "extra": {"exit_status": "Submitted", "submission": "replay.source"},
            }
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"system": "Linux", **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "sandbox": REPLAY_SANDBOX_PROTOCOL,
                        "hostname": REPLAY_SANDBOX_HOSTNAME,
                        "network": "new-unshared-namespace-host-probed",
                        "proc": "not-mounted",
                        "input": "read-only-allowlist",
                        "output": "replay-source-only",
                        "mounted_busybox_entrypoints": list(
                            REPLAY_TOOL_APPLETS
                        ),
                        "replay_check_shared_limit": self.config.replay_check_limit,
                        "replay_check_submission_reserve": (
                            self.config.replay_submission_check_limit
                        ),
                        "replay_compile_check_limit": (
                            self.config.replay_compile_check_limit
                        ),
                        "replay_source_inspect_limit": (
                            self.config.replay_source_inspect_limit
                        ),
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "replay_check": {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "calls": self._check_calls,
                    "evaluations": self._check_evaluations,
                    "limit": (
                        0
                        if self.config.replay_check_limit == 0
                        else self.config.replay_check_limit
                        + self.config.replay_submission_check_limit
                    ),
                    "time_governed": self.config.replay_check_limit == 0,
                    "explicit": {
                        "calls": self._calls_by_trigger["explicit"],
                        "evaluations": self._evaluations_by_trigger["explicit"],
                    },
                    "submission": {
                        "calls": self._calls_by_trigger["submission"],
                        "evaluations": self._evaluations_by_trigger["submission"],
                    },
                    "shared_budget": {
                        "used": self._shared_evaluations,
                        "limit": self.config.replay_check_limit,
                    },
                    "submission_reserve": {
                        "used": self._submission_reserved_evaluations,
                        "limit": self.config.replay_submission_check_limit,
                    },
                    "results": list(self._check_audit),
                },
                "replay_compile_check": {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "authority": (
                        "source_blind_construction_preflight_only; no Gate1, "
                        "membership, or certification authority"
                    ),
                    "calls": self._compile_check_calls,
                    "evaluations": self._compile_check_evaluations,
                    "limit": self.config.replay_compile_check_limit,
                    "results": list(self._compile_check_audit),
                },
                "replay_source_inspect": {
                    "protocol": "intentir.replay-own-source-inventory.v1",
                    "authority": "diagnostic_syntax_only_own_replay_source",
                    "calls": self._source_inspect_calls,
                    "limit": self.config.replay_source_inspect_limit,
                    "modes": list(self._source_inspect_audit),
                },
                "replay_transaction": {
                    "protocol": REPLAY_TRANSACTION_PROTOCOL,
                    "authority": (
                        "exact Replay-LLM-authored bytes and exact-text replacement "
                        "only; no source is generated or inferred"
                    ),
                    "commit_count": len(self._commit_audit),
                    "commits": list(self._commit_audit),
                    "rollback_count": self._rollback_count,
                    "fallback_count": len(self._fallback_audit),
                    "fallbacks": list(self._fallback_audit),
                    "apply_patch_count": len(self._apply_patch_audit),
                    "apply_patches": list(self._apply_patch_audit),
                    "direct_submit_count": len(self._direct_submit_audit),
                    "direct_submits": list(self._direct_submit_audit),
                },
                "multi_action_policy": {
                    "protocol": MULTI_ACTION_PROTOCOL,
                    "max_actions": 4,
                    "authority": "whole-response fail-before-execute",
                    "allowed": (
                        "independent source-blind intentir_packet_query actions"
                    ),
                    "rejections": self._multi_action_rejections,
                },
            }
        }


__all__ = [
    "INTENTIR_PACKET_QUERY",
    "INTENTIR_REPLAY_COMMIT",
    "INTENTIR_REPLAY_CHECK",
    "INTENTIR_REPLAY_COMPILE_CHECK",
    "INTENTIR_REPLAY_INSPECT",
    "INTENTIR_REPLAY_OBLIGATION_AUDIT",
    "INTENTIR_REPLAY_ROLLBACK",
    "INTENTIR_REPLAY_SUBMIT",
    "INTENTIR_REPLAY_SUBMIT_COMMAND",
    "MULTI_ACTION_PROTOCOL",
    "REPLAY_CHECK_PROTOCOL",
    "REPLAY_APPLY_PATCH_PROTOCOL",
    "REPLAY_DIRECT_SUBMIT_PROTOCOL",
    "REPLAY_TRANSACTION_PROTOCOL",
    "REPLAY_INPUT_FILES",
    "IntentIRReplayEnvironment",
    "IntentIRReplayEnvironmentConfig",
    "IntentIRReplayWorkspace",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_REPLAY_BYTES",
]
