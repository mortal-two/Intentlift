"""Compact, source-neutral lookup over the supplied IntentIR references."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_OP_HEADING = re.compile(r"^  ([a-z][a-z0-9_]*)\s*:")
_BUNDLED_INPUT_ROOT = (
    Path("/input")
    if Path("/input").is_dir()
    else Path(__file__).resolve().parents[1]
)
_DEFAULT_SCHEMA = str(_BUNDLED_INPUT_ROOT / "schema" / "artifact.schema.json")
_DEFAULT_OPS_REFERENCE = str(
    _BUNDLED_INPUT_ROOT / "knowledge" / "intent_ops.md"
)
_COMMANDS = {
    "top",
    "catalog",
    "sections",
    "authoring-brief",
    "op",
    "section",
    "bundle",
}


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


def _section_catalog(schema: dict[str, Any]) -> None:
    names = list(_sections(schema))
    print("artifact sections: " + ", ".join(names))
    print("Use `section NAME` to print the exact closed schema for one item.")


def _authoring_brief(schema: dict[str, Any], reference: str) -> None:
    """Print one compact source-neutral navigation index, never a template."""

    _top(schema)
    _section_catalog(schema)
    _catalog(reference)


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
        "consumer_extension": _array_item(
            boundary["consumer_extensions"]
        ),
        "access_guard_relation": _array_item(
            boundary["access_guard_relations"]
        ),
        "value_dimension_relation": _array_item(
            j["value_dimension_relations"]
        ),
        "dimension_relation": _array_item(j["dimension_relations"]),
        "authorization": _array_item(
            boundary["realization_authorizations"]
        ),
        "c_slot": _array_item(c["slots"]),
        "pi_region": _array_item(pi["regions"]),
        "pi_instance": _array_item(pi["instances"]),
        "pi_route": _array_item(pi["routes"]),
        "pi_program_axis": _array_item(pi["program_axes"]),
        "pi_participant_mapping": _array_item(pi["participant_mappings"]),
        "pi_route_extent_transform": _array_item(
            pi["route_extent_transforms"]
        ),
        "pi_order": _array_item(pi["order"]),
        "pi_sharing": _array_item(pi["sharing"]),
        "pi_iteration": _array_item(pi["iterations"]),
        "pi_loop_carrier": _array_item(pi["loop_carriers"]),
        "pi_physical_iteration": _array_item(pi["physical_iterations"]),
        "pi_physical_loop_carrier": _array_item(pi["physical_loop_carriers"]),
        "pi_terminal_control": _array_item(pi["terminal_controls"]),
        "pi_refinement": _array_item(pi["refinements"]),
    }


def schema_sections(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the exact source-neutral leaf sections used by CLI and agents."""

    return _sections(schema)


def _section(schema: dict[str, Any], name: str) -> None:
    sections = _sections(schema)
    if name not in sections:
        raise KeyError(f"unknown section: {name}; choose {sorted(sections)}")
    print(json.dumps(sections[name], indent=2, sort_keys=True))


def _bundle(schema: dict[str, Any], reference: str, items: list[str]) -> None:
    """Render several independent source-neutral lookups in one tool action."""

    for index, item in enumerate(items):
        if index:
            print()
        print(f"=== {item} ===")
        if item == "top":
            _top(schema)
        elif item == "catalog":
            _catalog(reference)
        elif item == "sections":
            _section_catalog(schema)
        elif item == "authoring-brief":
            _authoring_brief(schema, reference)
        elif item.startswith("op:") and len(item) > 3:
            _op(schema, reference, item[3:])
        elif item.startswith("section:") and len(item) > 8:
            _section(schema, item[8:])
        else:
            raise ValueError(
                "bundle items must be top, catalog, sections, authoring-brief, "
                "op:NAME, or section:NAME"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema")
    parser.add_argument(
        "--ops-reference",
        default=_DEFAULT_OPS_REFERENCE,
        help="source-neutral operation reference mounted for this Lift round",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("top")
    subparsers.add_parser("catalog")
    subparsers.add_parser("sections")
    subparsers.add_parser("authoring-brief")
    op_parser = subparsers.add_parser("op")
    op_parser.add_argument("name")
    section_parser = subparsers.add_parser("section")
    section_parser.add_argument("name")
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("items", nargs="+")
    argv = sys.argv[1:]
    if argv and argv[0] in _COMMANDS:
        argv = [_DEFAULT_SCHEMA, *argv]
    args = parser.parse_args(argv)
    schema = _load_json(args.schema)
    if args.command == "top":
        _top(schema)
    elif args.command == "catalog":
        _catalog(_load_text(args.ops_reference))
    elif args.command == "sections":
        _section_catalog(schema)
    elif args.command == "authoring-brief":
        _authoring_brief(schema, _load_text(args.ops_reference))
    elif args.command == "op":
        _op(schema, _load_text(args.ops_reference), args.name)
    elif args.command == "section":
        _section(schema, args.name)
    else:
        _bundle(schema, _load_text(args.ops_reference), args.items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
