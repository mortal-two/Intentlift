"""Compact, source-neutral lookup over the supplied IntentIR references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_OP_HEADING = re.compile(r"^  ([a-z][a-z0-9_]*)\s*:")


def _load_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("schema root must be an object")
    return value


def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _object_line(label: str, value: dict[str, Any]) -> None:
    required = value.get("required", [])
    properties = value.get("properties", {})
    print(f"{label}: required={required}; fields={list(properties)}")


def _top(schema: dict[str, Any]) -> None:
    _object_line("root", schema)
    root = schema["properties"]
    j = root["j"]
    _object_line("j", j)
    _object_line("j.intent", j["properties"]["intent"])
    _object_line(
        "j.intent.intent_function",
        j["properties"]["intent"]["properties"]["intent_function"],
    )
    _object_line("j.boundary", j["properties"]["boundary"])
    _object_line("j.signature", j["properties"]["signature"])
    _object_line("c", root["c"])
    _object_line("pi", root["pi"])
    print(
        "Use `section NAME` for one leaf schema, `catalog` for operation "
        "names, and `op NAME` for one operation contract."
    )


def _operation_sections(reference: str) -> dict[str, str]:
    lines = reference.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _OP_HEADING.match(line)
        if match:
            starts.append((index, match.group(1)))
    sections: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections[name] = "\n".join(lines[start:end]).rstrip()
    if not sections:
        raise ValueError("operation reference contains no operation headings")
    return sections


def _catalog(reference: str) -> None:
    sections = _operation_sections(reference)
    print("ordinary operations: " + ", ".join(sections))
    print("structured operations: loop, effect_if, value_if")
    print("Use `op NAME` to print only one operation contract.")


def _op(schema: dict[str, Any], reference: str, name: str) -> None:
    if name in {"loop", "effect_if", "value_if"}:
        variants = schema["$defs"]["intent_op"]["anyOf"]
        for variant in variants:
            op_schema = variant.get("properties", {}).get("op", {})
            if name in op_schema.get("enum", []):
                print(json.dumps(variant, indent=2, sort_keys=True))
                return
        raise KeyError(f"schema contains no structured {name} operation")
    sections = _operation_sections(reference)
    if name not in sections:
        raise KeyError(
            f"unknown operation: {name}; choose {sorted(sections)} or "
            "loop/effect_if/value_if"
        )
    print(sections[name])


def _array_item(value: dict[str, Any]) -> dict[str, Any]:
    item = value.get("items")
    if not isinstance(item, dict):
        raise TypeError("requested schema section is not an object array")
    return item


def _sections(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = schema["properties"]
    j = root["j"]["properties"]
    intent_function = j["intent"]["properties"]["intent_function"]
    boundary = j["boundary"]["properties"]
    c = root["c"]["properties"]
    pi = root["pi"]["properties"]
    return {
        "tensor": _array_item(intent_function["properties"]["inputs"]),
        "parameter": _array_item(j["signature"]["properties"]["parameters"]),
        "scope": _array_item(j["scopes"]),
        "storage": _array_item(j["storage"]),
        "domain": boundary["domains"],
        "access": _array_item(boundary["accesses"]),
        "broadcast": _array_item(boundary["broadcasts"]),
        "alias": _array_item(boundary["aliases"]),
        "effect": _array_item(boundary["effects"]),
        "output_mapping": _array_item(boundary["output_mappings"]),
        "reduction": _array_item(boundary["reductions"]),
        "index_bound": _array_item(boundary["index_bounds"]),
        "update": _array_item(boundary["updates"]),
        "result_relation": _array_item(boundary["result_relations"]),
        "authorization": _array_item(
            boundary["realization_authorizations"]
        ),
        "c_slot": _array_item(c["slots"]),
        "pi_region": _array_item(pi["regions"]),
        "pi_instance": _array_item(pi["instances"]),
        "pi_route": _array_item(pi["routes"]),
        "pi_order": _array_item(pi["order"]),
        "pi_sharing": _array_item(pi["sharing"]),
        "pi_iteration": _array_item(pi["iterations"]),
        "pi_refinement": _array_item(pi["refinements"]),
    }


def _section(schema: dict[str, Any], name: str) -> None:
    sections = _sections(schema)
    if name not in sections:
        raise KeyError(f"unknown section: {name}; choose {sorted(sections)}")
    print(json.dumps(sections[name], indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema")
    parser.add_argument(
        "--ops-reference",
        default="/input/knowledge/intent_ops.md",
        help="source-neutral operation reference mounted for this Lift round",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("top")
    subparsers.add_parser("catalog")
    op_parser = subparsers.add_parser("op")
    op_parser.add_argument("name")
    section_parser = subparsers.add_parser("section")
    section_parser.add_argument("name")
    args = parser.parse_args()
    schema = _load_json(args.schema)
    if args.command == "top":
        _top(schema)
    elif args.command == "catalog":
        _catalog(_load_text(args.ops_reference))
    elif args.command == "op":
        _op(schema, _load_text(args.ops_reference), args.name)
    else:
        _section(schema, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
