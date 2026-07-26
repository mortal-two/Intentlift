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
        unresolved = diagnostics.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, dict)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, dict)
            else None
        )
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
            feedback.get("failure_kind")
            if isinstance(feedback, dict)
            else None
        ),
        "feedback_round": (
            feedback.get("relift_round")
            if isinstance(feedback, dict)
            else None
        ),
        "observer_categories": _observer_categories(feedback),
        "feedback_diagnostic_keys": (
            list(feedback.get("diagnostics", {}))
            if isinstance(feedback, dict)
            and isinstance(feedback.get("diagnostics"), dict)
            else []
        ),
        "next": [
            f"Run `{QUERY_COMMAND} feedback` before repairing a prior candidate.",
            f"Use `{QUERY_COMMAND} feedback --path diagnostics.KEY` for one exact diagnostic section.",
            "Read Source and Harness separately so neither can be transport-elided.",
            f"Use `{QUERY_COMMAND} source-find TEXT --context 3` for literal Source lookup.",
            f"After drafting, run `{QUERY_COMMAND} candidate-abi` for a read-only ABI diff.",
            f"Inspect a draft with `{QUERY_COMMAND} candidate --path j.PATH` or `candidate-find HANDLE`.",
            f"Page an authored graph with `{QUERY_COMMAND} candidate-nodes --start 1 --count 40`.",
            "When Gate1 reports dataflow, run `python /input/tools/context_query.py feedback-category dataflow`; it preserves effect-order witnesses even when bulk node witnesses are paged out.",
            f"Use `{QUERY_COMMAND} knowledge QUERY` and schema_query for focused lookup.",
        ],
    }
    _emit_json(overview)


def _feedback(input_root: Path, path: str | None = None) -> None:
    value = _feedback_without_candidate(input_root)
    if path is not None:
        _emit_json(_resolve_json_path(value, path))
        return
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
    for category_index, category in enumerate(categories):
        if isinstance(category, dict) and category.get("category") == name:
            result = dict(category)
            rendered = json.dumps(category, sort_keys=True)
            category_path = (
                "diagnostics.observer_diff.categories."
                f"{category_index}"
            )
            if name == "dataflow" and isinstance(
                result.get("effect_order_witnesses"), list
            ):
                result["effect_order_hint"] = (
                    "Compare the ordered Source and Replay event sequences, "
                    "then inspect the LLM-authored J scopes and pi order. The "
                    "witness is diagnostic only and does not prescribe a repair."
                )
                result["effect_order_query"] = (
                    f"{QUERY_COMMAND} feedback --path {category_path}."
                    "effect_order_witnesses"
                )
            if (
                name == "dataflow"
                and "region[root]" in rendered
                and "region[scf.for:" in rendered
                and "region.member" in rendered
            ):
                result["diagnostic_hint"] = (
                    "The observer examples cross root/scf.for region membership. "
                    "Inspect exact Source definition scope and pi instance regions; "
                    "fixed code has not inferred which producer, if any, must move."
                )
                result["knowledge_query"] = (
                    f"{QUERY_COMMAND} knowledge 'lexical lifetime' --context 5"
                )
            if len(
                json.dumps(result, sort_keys=True, ensure_ascii=False)
            ) > MAX_OUTPUT_CHARS:
                node_witnesses = result.pop("node_witnesses", None)
                if isinstance(node_witnesses, list):
                    result["node_witness_count"] = len(node_witnesses)
                    result["node_witness_query"] = (
                        f"{QUERY_COMMAND} feedback --path {category_path}."
                        "node_witnesses"
                    )
                    result["bounded_view"] = (
                        "bulk node witnesses omitted so causal role/effect "
                        "witnesses remain visible"
                    )
            _emit_json(result)
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


def _source_find(input_root: Path, query: str, context: int) -> None:
    """Find literal Source text with bounded line-numbered context.

    This is deliberately a text lookup rather than a parser or semantic
    extractor.  Lift still decides what the matched Source means and authors
    every J/C/pi field.
    """

    if not query:
        raise ValueError("source-find query must be non-empty")
    if context < 0 or context > 20:
        raise ValueError("source-find context must be in [0, 20]")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    needle = query.casefold()
    matches = [
        index for index, line in enumerate(lines) if needle in line.casefold()
    ]
    if not matches:
        raise KeyError(f"no Source lines match {query!r}")

    ranges: list[tuple[int, int]] = []
    for index in matches:
        begin = max(0, index - context)
        end = min(len(lines), index + context + 1)
        if ranges and begin <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((begin, end))

    blocks: list[str] = []
    used = 0
    omitted = False
    match_set = set(matches)
    for begin, end in ranges:
        rendered = [f"--- source.py:{begin + 1} ---"]
        rendered.extend(
            f"{'>' if ordinal in match_set else ' '} {ordinal + 1:5d} | "
            f"{lines[ordinal]}"
            for ordinal in range(begin, end)
        )
        block = "\n".join(rendered)
        remaining = MAX_OUTPUT_CHARS - 300 - used
        if len(block) + 1 > remaining:
            if remaining > 80:
                blocks.append(block[: remaining - 40] + "\n[context truncated]")
            omitted = True
            break
        blocks.append(block)
        used += len(block) + 1
    print("\n".join(blocks))
    if omitted:
        print("\n[more matches omitted; use a narrower literal query]")


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


def _candidate(output_root: Path, path: str | None) -> None:
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    value = _resolve_json_path(_load_json(candidate_path), path)
    try:
        _emit_json(value)
    except ValueError:
        if path is not None or not isinstance(value, dict):
            raise
        _emit_json(
            {
                "keys": list(value),
                "query": f"{QUERY_COMMAND} candidate --path KEY[.SUBKEY]",
            }
        )


def _candidate_find(output_root: Path, needle: str) -> None:
    """List bounded exact-value occurrences in the current LLM draft."""

    if not needle:
        raise ValueError("candidate-find requires a non-empty exact value")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    value = _load_json(candidate_path)
    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(paths) >= 256:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}.{index}" if path else str(index))
        elif item == needle:
            paths.append(path)

    visit(value, "")
    _emit_json(
        {
            "exact_value": needle,
            "paths": paths,
            "match_count": len(paths),
            "truncated": len(paths) >= 256,
            "authority": "read-only; candidate.json was not edited",
        }
    )


def _iter_candidate_nodes(
    value: Any, path: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    """Return authored node objects in their JSON traversal order."""

    nodes: list[tuple[str, dict[str, Any]]] = []

    def visit(item: Any, current_path: str) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("id"), str) and isinstance(
                item.get("op"), str
            ):
                nodes.append((current_path, item))
            for key, child in item.items():
                visit(
                    child,
                    f"{current_path}.{key}" if current_path else str(key),
                )
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(
                    child,
                    f"{current_path}.{index}" if current_path else str(index),
                )

    visit(value, path)
    return nodes


def _node_summary(path: str, node: dict[str, Any]) -> dict[str, Any]:
    """Project only fields already authored on a node into a compact view."""

    summary: dict[str, Any] = {
        "path": path,
        "id": node.get("id"),
        "op": node.get("op"),
    }
    for field in ("operands", "result", "results", "type", "attrs"):
        if field in node:
            summary[field] = node[field]
    for field in (
        "body",
        "then_body",
        "else_body",
        "positive_body",
        "negative_body",
        "carry",
        "results",
    ):
        value = node.get(field)
        if isinstance(value, list):
            summary[f"{field}_item_count"] = len(value)
    summary["authored_keys"] = list(node)
    return summary


def _candidate_nodes(
    output_root: Path,
    start: int,
    count: int,
    op: str | None = None,
) -> None:
    """Page a compact read-only view of the current authored J nodes."""

    if start < 1:
        raise ValueError("candidate-nodes start must be positive")
    if count < 1 or count > 100:
        raise ValueError("candidate-nodes count must be in [1, 100]")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    candidate = _load_json(candidate_path)
    nodes = _iter_candidate_nodes(candidate)
    if op is not None:
        nodes = [item for item in nodes if item[1].get("op") == op]
    selected = nodes[start - 1 : start - 1 + count]
    summaries: list[dict[str, Any]] = []
    for path, node in selected:
        summary = _node_summary(path, node)
        trial = summaries + [summary]
        if len(
            json.dumps(trial, sort_keys=True, ensure_ascii=False)
        ) > MAX_OUTPUT_CHARS - 1_500:
            if not summaries:
                summaries.append(
                    {
                        "path": path,
                        "id": node.get("id"),
                        "op": node.get("op"),
                        "authored_keys": list(node),
                        "inspect_command": (
                            f"{QUERY_COMMAND} candidate-node {node.get('id')}"
                        ),
                        "detail_omitted": "single compact node exceeded page budget",
                    }
                )
            break
        summaries = trial
    next_start = start + len(summaries)
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-node-query.v1",
            "filter_op": op,
            "start": start,
            "returned": len(summaries),
            "total": len(nodes),
            "nodes": summaries,
            "next_command": (
                f"{QUERY_COMMAND} candidate-nodes --start {next_start} "
                f"--count {count}" + (f" --op {op}" if op is not None else "")
                if next_start <= len(nodes)
                else None
            ),
            "authority": "read-only; candidate.json was not edited or repaired",
        }
    )


def _candidate_node(output_root: Path, node_id: str) -> None:
    """Inspect one exact authored node without inferring its meaning."""

    if not node_id:
        raise ValueError("candidate-node requires a non-empty node id")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    matches = [
        item
        for item in _iter_candidate_nodes(_load_json(candidate_path))
        if item[1].get("id") == node_id
    ]
    if not matches:
        raise KeyError(f"candidate contains no authored node {node_id!r}")
    if len(matches) != 1:
        raise ValueError(
            f"candidate contains {len(matches)} authored nodes named {node_id!r}"
        )
    path, node = matches[0]
    result = {
        "protocol": "intentir.lift-candidate-node-query.v1",
        "path": path,
        "node": node,
        "authority": "read-only; candidate.json was not edited or repaired",
    }
    try:
        _emit_json(result)
    except ValueError:
        _emit_json(
            {
                "protocol": result["protocol"],
                "summary": _node_summary(path, node),
                "nested_nodes": [
                    _node_summary(child_path, child)
                    for child_path, child in _iter_candidate_nodes(node, path)
                    if child is not node
                ][:48],
                "exact_path_command": (
                    f"{QUERY_COMMAND} candidate --path {path}"
                ),
                "authority": result["authority"],
            }
        )


def _candidate_abi(input_root: Path, output_root: Path) -> None:
    """Compare one draft with the Harness-owned mechanical ABI, read-only.

    This intentionally does not propose logical refs, tensor declarations,
    symbolic dimensions, C subjects/axes, or any other semantic field.  It
    merely makes the same already-authored public plumbing visible without
    forcing an agent to manually diff two large JSON documents.
    """

    view = _load_json(input_root / "harness_view.json")
    if not isinstance(view, dict):
        raise TypeError("harness_view.json must be an object")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        _emit_json(
            {
                "protocol": "intentir.lift-candidate-abi-query.v1",
                "candidate_found": False,
                "next": (
                    "Write a complete LLM-authored /output/candidate.json, "
                    f"then rerun `{QUERY_COMMAND} candidate-abi`."
                ),
            }
        )
        return
    candidate = _load_json(candidate_path)
    if not isinstance(candidate, dict):
        raise TypeError("candidate.json must contain one root object")

    issues: list[dict[str, Any]] = []

    def issue(
        code: str,
        path: str,
        expected: Any,
        actual: Any,
    ) -> None:
        issues.append(
            {
                "code": code,
                "path": path,
                "expected": expected,
                "actual": actual,
            }
        )

    parameter_order = view.get("parameter_order", [])
    if not isinstance(parameter_order, list) or not all(
        isinstance(name, str) for name in parameter_order
    ):
        raise TypeError("Harness parameter_order must be an array of strings")
    expected_p = [name for name in parameter_order if name.startswith("p")]
    expected_c = [name for name in parameter_order if name.startswith("c")]

    j = candidate.get("j")
    signature = j.get("signature") if isinstance(j, dict) else None
    parameters = (
        signature.get("parameters") if isinstance(signature, dict) else None
    )
    if not isinstance(parameters, list):
        parameters = []
        issue(
            "signature_parameters_missing",
            "j.signature.parameters",
            expected_p,
            None,
        )
    actual_parameters = {
        item.get("name"): item
        for item in parameters
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    actual_p = [
        item.get("name") if isinstance(item, dict) else None
        for item in parameters
    ]
    if actual_p != expected_p:
        issue(
            "parameter_order_mismatch",
            "j.signature.parameters[*].name",
            expected_p,
            actual_p,
        )

    arguments = view.get("arguments", [])
    if not isinstance(arguments, list):
        raise TypeError("Harness arguments must be an array")
    arguments_by_name = {
        item.get("name"): item
        for item in arguments
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    bound_constexprs = view.get("bound_constexpr_values", {})
    if not isinstance(bound_constexprs, dict):
        raise TypeError("Harness bound_constexpr_values must be an object")

    for name in expected_p:
        actual = actual_parameters.get(name)
        if actual is None:
            issue(
                "parameter_missing",
                f"j.signature.parameters[{name}]",
                "one declaration in Harness order",
                None,
            )
            continue
        argument = arguments_by_name.get(name)
        if argument is None:
            if name not in bound_constexprs:
                issue(
                    "parameter_unclassified",
                    f"j.signature.parameters[{name}]",
                    "Harness argument or bound constexpr",
                    actual,
                )
                continue
            expected_fields = {
                "kind": ["constexpr"],
                "mode": "value",
            }
        else:
            source_kind = argument.get("kind")
            allowed_kinds = {
                "tensor": ["pointer", "tensor"],
                "scalar": ["scalar"],
                "opaque": ["opaque"],
                "none": ["opaque"],
            }.get(source_kind)
            if allowed_kinds is None:
                issue(
                    "argument_kind_unknown",
                    f"harness.arguments[{name}].kind",
                    ["tensor", "scalar", "opaque", "none"],
                    source_kind,
                )
                continue
            expected_fields = {
                "kind": allowed_kinds,
                "dtype": argument.get("dtype"),
            }
            if "mode" in argument:
                expected_fields["mode"] = argument.get("mode")
            if source_kind == "tensor":
                shape = argument.get("shape")
                expected_fields["rank"] = (
                    len(shape) if isinstance(shape, list) else "Harness shape"
                )
            elif source_kind == "scalar":
                expected_fields["rank"] = 0
            else:
                expected_fields["rank"] = None
                expected_fields["logical_ref"] = None
        for field, expected in expected_fields.items():
            observed = actual.get(field)
            matches = (
                observed in expected
                if isinstance(expected, list)
                else observed == expected
            )
            if not matches:
                issue(
                    "parameter_field_mismatch",
                    f"j.signature.parameters[{name}].{field}",
                    expected,
                    observed,
                )

    c = candidate.get("c")
    slots = c.get("slots") if isinstance(c, dict) else None
    values = c.get("values") if isinstance(c, dict) else None
    if not isinstance(slots, list):
        slots = []
        issue("c_slots_missing", "c.slots", "array", None)
    actual_slots = [item for item in slots if isinstance(item, dict)]
    actual_constexpr_c = [
        item.get("name")
        for item in actual_slots
        if item.get("kind") == "constexpr"
    ]
    if actual_constexpr_c != expected_c:
        issue(
            "constexpr_c_order_mismatch",
            "c.slots[kind=constexpr].name",
            expected_c,
            actual_constexpr_c,
        )
    expected_compiler = view.get("compiler_slots", [])
    if not isinstance(expected_compiler, list):
        raise TypeError("Harness compiler_slots must be an array")
    actual_compiler = [
        item.get("name")
        for item in actual_slots
        if item.get("kind") == "compiler"
    ]
    if actual_compiler != expected_compiler:
        issue(
            "compiler_c_order_mismatch",
            "c.slots[kind=compiler].name",
            expected_compiler,
            actual_compiler,
        )
    expected_values = view.get("schedule_values", {})
    if values != expected_values:
        issue("c_values_mismatch", "c.values", expected_values, values)
    expected_roles = view.get("compiler_roles", {})
    if not isinstance(expected_roles, dict):
        raise TypeError("Harness compiler_roles must be an object")
    slots_by_name = {
        item.get("name"): item
        for item in actual_slots
        if isinstance(item.get("name"), str)
    }
    for name, role in expected_roles.items():
        slot = slots_by_name.get(name)
        observed = (
            {"kind": slot.get("kind"), "role": slot.get("role")}
            if slot is not None
            else None
        )
        expected = {"kind": "compiler", "role": role}
        if observed != expected:
            issue(
                "compiler_c_role_mismatch",
                f"c.slots[{name}]",
                expected,
                observed,
            )

    expected_constexprs = view.get("constexpr_parameters", [])
    if not isinstance(expected_constexprs, list):
        raise TypeError("Harness constexpr_parameters must be an array")
    actual_constexprs = [
        item.get("name")
        for item in parameters
        if isinstance(item, dict) and item.get("kind") == "constexpr"
    ] + actual_constexpr_c
    if set(actual_constexprs) != set(expected_constexprs):
        issue(
            "constexpr_membership_mismatch",
            "J constexpr parameters union C constexpr slots",
            sorted(expected_constexprs),
            sorted(
                name for name in actual_constexprs if isinstance(name, str)
            ),
        )

    expected_aliases = view.get("aliases", [])
    boundary = j.get("boundary") if isinstance(j, dict) else None
    actual_aliases = (
        boundary.get("aliases") if isinstance(boundary, dict) else None
    )
    if actual_aliases != expected_aliases:
        issue(
            "alias_mismatch",
            "j.boundary.aliases",
            expected_aliases,
            actual_aliases,
        )

    issue_limit = 48
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-abi-query.v1",
            "candidate_found": True,
            "matches_public_mechanical_abi": not issues,
            "issue_count": len(issues),
            "issues": issues[:issue_limit],
            "issues_omitted": max(0, len(issues) - issue_limit),
            "counts": {
                "expected_j_parameters": len(expected_p),
                "actual_j_parameters": len(parameters),
                "expected_constexpr_c_slots": len(expected_c),
                "expected_compiler_c_slots": len(expected_compiler),
            },
            "agent_owned_not_inferred": [
                "parameter logical_ref and logical tensor/dimension ownership",
                "bound-constexpr semantic dtype/rank/logical_ref",
                "C slot subject_ref and axis_ref",
                "all J graph, boundary, scope, storage, and pi content",
            ],
            "authority": (
                "Read-only diagnostic only; validator, CandidateCheck and the "
                "external Runner remain authoritative and no field was edited."
            ),
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
    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("--path")
    category = subparsers.add_parser("feedback-category")
    category.add_argument("name")
    source = subparsers.add_parser("source")
    source.add_argument("--start", type=int, default=1)
    source.add_argument("--lines", type=int, default=160)
    source_find = subparsers.add_parser("source-find")
    source_find.add_argument("query")
    source_find.add_argument("--context", type=int, default=3)
    harness = subparsers.add_parser("harness")
    harness.add_argument("--path")
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--path")
    candidate_find = subparsers.add_parser("candidate-find")
    candidate_find.add_argument("value")
    candidate_nodes = subparsers.add_parser("candidate-nodes")
    candidate_nodes.add_argument("--start", type=int, default=1)
    candidate_nodes.add_argument("--count", type=int, default=40)
    candidate_nodes.add_argument("--op")
    candidate_node = subparsers.add_parser("candidate-node")
    candidate_node.add_argument("node_id")
    subparsers.add_parser("candidate-abi")
    knowledge = subparsers.add_parser("knowledge")
    knowledge.add_argument("query")
    knowledge.add_argument("--context", type=int, default=3)
    subparsers.add_parser("copy-previous")
    args = parser.parse_args()

    if args.command == "bootstrap":
        _bootstrap(args.input_root, args.output_root)
    elif args.command == "feedback":
        _feedback(args.input_root, args.path)
    elif args.command == "feedback-category":
        _feedback_category(args.input_root, args.name)
    elif args.command == "source":
        _source(args.input_root, args.start, args.lines)
    elif args.command == "source-find":
        _source_find(args.input_root, args.query, args.context)
    elif args.command == "harness":
        _harness(args.input_root, args.path)
    elif args.command == "candidate":
        _candidate(args.output_root, args.path)
    elif args.command == "candidate-find":
        _candidate_find(args.output_root, args.value)
    elif args.command == "candidate-nodes":
        _candidate_nodes(args.output_root, args.start, args.count, args.op)
    elif args.command == "candidate-node":
        _candidate_node(args.output_root, args.node_id)
    elif args.command == "candidate-abi":
        _candidate_abi(args.input_root, args.output_root)
    elif args.command == "knowledge":
        _knowledge(args.input_root, args.query, args.context)
    elif not _write_previous(args.input_root, args.output_root):
        raise ValueError("feedback contains no previous artifact")
    else:
        print("copied previous artifact to /output/candidate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
