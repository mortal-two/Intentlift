"""Bounded, source-aware context lookup for one IntentIR Lift worker.

The tool exposes only files already mounted in the worker sandbox.  It never
constructs or repairs J/C/pi.  Its purpose is to keep Source, Harness and
external diagnostics observable even when the model transport elides long
shell output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAX_OUTPUT_CHARS = 10_000
QUERY_COMMAND = "python /input/tools/context_query.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _feedback_without_candidate(input_root: Path) -> dict[str, Any] | None:
    value = _load_json(input_root / "feedback.json")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("feedback.json must be null or an object")
    return {
        str(key): item
        for key, item in value.items()
        if key != "previous_artifact"
    }


def _previous_candidate(input_root: Path) -> dict[str, Any] | None:
    value = _load_json(input_root / "feedback.json")
    if not isinstance(value, dict):
        return None
    candidate = value.get("previous_artifact")
    if isinstance(candidate, str):
        candidate = json.loads(candidate)
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise TypeError("feedback previous_artifact must be an object")
    return candidate


def _write_previous(input_root: Path, output_root: Path) -> bool:
    candidate = _previous_candidate(input_root)
    if candidate is None:
        return False
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "candidate.json"
    target.write_text(
        json.dumps(
            candidate,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def _observer_categories(feedback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback, dict):
        return []
    diagnostics = feedback.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    observer = diagnostics.get("observer_diff")
    if not isinstance(observer, dict):
        return []
    categories = observer.get("categories")
    if not isinstance(categories, list):
        return []
    answer: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        answer.append(
            {
                "category": category.get("category"),
                "difference_count": category.get("difference_count"),
            }
        )
    return answer


def _emit_json(value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"query result is {len(text)} characters; use a narrower command"
        )
    print(text)


def _bootstrap(input_root: Path, output_root: Path) -> None:
    feedback = _feedback_without_candidate(input_root)
    copied = _write_previous(input_root, output_root)
    files = {}
    for relative in ("source.py", "harness_view.json", "feedback.json"):
        path = input_root / relative
        files[relative] = {
            "bytes": path.stat().st_size,
            "read_command": {
                "source.py": f"{QUERY_COMMAND} source --start 1 --lines 160",
                "harness_view.json": f"{QUERY_COMMAND} harness",
                "feedback.json": f"{QUERY_COMMAND} feedback",
            }[relative],
        }
    overview = {
        "candidate_copied_to_output": copied,
        "files": files,
        "feedback_kind": (
            feedback.get("kind") if isinstance(feedback, dict) else None
        ),
        "feedback_round": (
            feedback.get("round") if isinstance(feedback, dict) else None
        ),
        "observer_categories": _observer_categories(feedback),
        "next": [
            f"Run `{QUERY_COMMAND} feedback` before repairing a prior candidate.",
            "Read Source and Harness separately so neither can be transport-elided.",
            f"Use `{QUERY_COMMAND} knowledge QUERY` and schema_query for focused lookup.",
        ],
    }
    _emit_json(overview)


def _feedback(input_root: Path) -> None:
    value = _feedback_without_candidate(input_root)
    try:
        _emit_json(value)
    except ValueError:
        if not isinstance(value, dict):
            raise
        diagnostics = value.get("diagnostics")
        reduced = dict(value)
        if isinstance(diagnostics, dict):
            reduced_diagnostics = dict(diagnostics)
            reduced_diagnostics.pop("observer_diff", None)
            reduced["diagnostics"] = reduced_diagnostics
        reduced["observer_categories"] = _observer_categories(value)
        reduced["narrow_query"] = (
            f"{QUERY_COMMAND} feedback-category CATEGORY"
        )
        _emit_json(reduced)


def _feedback_category(input_root: Path, name: str) -> None:
    value = _feedback_without_candidate(input_root)
    diagnostics = value.get("diagnostics") if isinstance(value, dict) else None
    observer = (
        diagnostics.get("observer_diff")
        if isinstance(diagnostics, dict)
        else None
    )
    categories = observer.get("categories") if isinstance(observer, dict) else None
    if not isinstance(categories, list):
        raise KeyError("feedback contains no observer-diff categories")
    for category in categories:
        if isinstance(category, dict) and category.get("category") == name:
            _emit_json(category)
            return
    available = [
        item.get("category") for item in categories if isinstance(item, dict)
    ]
    raise KeyError(f"unknown feedback category {name!r}; choose {available}")


def _source(input_root: Path, start: int, line_count: int) -> None:
    if start < 1 or line_count < 1:
        raise ValueError("source start and lines must be positive")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    selected: list[str] = []
    used = 0
    for ordinal, line in enumerate(lines[start - 1 :], start=start):
        rendered = f"{ordinal:5d} | {line}"
        if selected and used + len(rendered) + 1 > MAX_OUTPUT_CHARS - 300:
            break
        selected.append(rendered)
        used += len(rendered) + 1
        if len(selected) >= line_count:
            break
    print("\n".join(selected))
    next_line = start + len(selected)
    if next_line <= len(lines):
        print(
            f"\n[source continues: run `{QUERY_COMMAND} source --start "
            f"{next_line} --lines {line_count}`]"
        )


def _resolve_json_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for component in path.split("."):
        if isinstance(current, dict):
            if component not in current:
                raise KeyError(
                    f"unknown JSON path component {component!r}; "
                    f"choose {list(current)}"
                )
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            current = current[int(component)]
        else:
            raise KeyError(f"cannot descend through {component!r}")
    return current


def _harness(input_root: Path, path: str | None) -> None:
    value = _resolve_json_path(
        _load_json(input_root / "harness_view.json"), path
    )
    try:
        _emit_json(value)
    except ValueError:
        if path is not None or not isinstance(value, dict):
            raise
        _emit_json(
            {
                "keys": list(value),
                "query": f"{QUERY_COMMAND} harness --path KEY[.SUBKEY]",
            }
        )


def _knowledge(input_root: Path, query: str, context: int) -> None:
    if not query:
        raise ValueError("knowledge query must be non-empty")
    needle = query.casefold()
    blocks: list[str] = []
    for relative in (
        "knowledge/lift_contract.md",
        "knowledge/preflight.md",
    ):
        lines = (input_root / relative).read_text(encoding="utf-8").splitlines()
        ranges: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            if needle in line.casefold():
                ranges.append(
                    (
                        max(0, index - context),
                        min(len(lines), index + context + 1),
                    )
                )
        merged: list[tuple[int, int]] = []
        for begin, end in ranges:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        for begin, end in merged:
            rendered = [f"--- {relative}:{begin + 1} ---"]
            rendered.extend(
                f"{ordinal + 1:5d} | {lines[ordinal]}"
                for ordinal in range(begin, end)
            )
            block = "\n".join(rendered)
            if sum(len(item) + 1 for item in blocks) + len(block) > MAX_OUTPUT_CHARS:
                print("\n".join(blocks))
                print("\n[more matches omitted; use a narrower query]")
                return
            blocks.append(block)
    if not blocks:
        raise KeyError(f"no contract/preflight lines match {query!r}")
    print("\n".join(blocks))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/output"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("feedback")
    category = subparsers.add_parser("feedback-category")
    category.add_argument("name")
    source = subparsers.add_parser("source")
    source.add_argument("--start", type=int, default=1)
    source.add_argument("--lines", type=int, default=160)
    harness = subparsers.add_parser("harness")
    harness.add_argument("--path")
    knowledge = subparsers.add_parser("knowledge")
    knowledge.add_argument("query")
    knowledge.add_argument("--context", type=int, default=3)
    subparsers.add_parser("copy-previous")
    args = parser.parse_args()

    if args.command == "bootstrap":
        _bootstrap(args.input_root, args.output_root)
    elif args.command == "feedback":
        _feedback(args.input_root)
    elif args.command == "feedback-category":
        _feedback_category(args.input_root, args.name)
    elif args.command == "source":
        _source(args.input_root, args.start, args.lines)
    elif args.command == "harness":
        _harness(args.input_root, args.path)
    elif args.command == "knowledge":
        _knowledge(args.input_root, args.query, args.context)
    elif not _write_previous(args.input_root, args.output_root):
        raise ValueError("feedback contains no previous artifact")
    else:
        print("copied previous artifact to /output/candidate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
