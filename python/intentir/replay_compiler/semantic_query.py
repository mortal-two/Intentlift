"""Bounded exact queries over one immutable ``S_J``."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .semantic_program import SemanticProgram


class SemanticQueryError(ValueError):
    pass


def _bounded_int(value: str, *, name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise SemanticQueryError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise SemanticQueryError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def query_semantic_program(
    program: SemanticProgram,
    arguments: Sequence[str],
) -> Mapping[str, Any]:
    """Return one exact bounded projection; never infer target code or pi facts."""

    payload = program.to_json()
    mode = arguments[0] if arguments else "summary"
    if mode == "summary":
        return {
            "schema": program.schema,
            "mode": program.mode,
            "parameterized": True,
            "counts": dict(program.counts),
            "json_bytes": program.json_bytes,
            "text_bytes": program.text_bytes,
        }
    if mode == "raw":
        if len(arguments) > 3:
            raise SemanticQueryError("raw accepts START [COUNT]")
        start = _bounded_int(
            arguments[1] if len(arguments) >= 2 else "1",
            name="START",
            minimum=1,
            maximum=max(1, len(program.text.splitlines())),
        )
        count = _bounded_int(
            arguments[2] if len(arguments) >= 3 else "80",
            name="COUNT",
            minimum=1,
            maximum=160,
        )
        lines = program.text.splitlines()
        selected = lines[start - 1 : start - 1 + count]
        return {
            "start": start,
            "total": len(lines),
            "lines": [
                [ordinal, line]
                for ordinal, line in enumerate(selected, start=start)
            ],
            "next_start": (
                start + len(selected)
                if start - 1 + len(selected) < len(lines)
                else None
            ),
        }
    if mode == "find":
        if len(arguments) not in {2, 3}:
            raise SemanticQueryError("find accepts TOKEN [CONTEXT]")
        token = arguments[1]
        if not token or len(token) > 256:
            raise SemanticQueryError("TOKEN must contain 1..256 characters")
        context = _bounded_int(
            arguments[2] if len(arguments) == 3 else "2",
            name="CONTEXT",
            minimum=0,
            maximum=12,
        )
        lines = program.text.splitlines()
        matches = [index for index, line in enumerate(lines) if token in line]
        selected: list[list[Any]] = []
        seen: set[int] = set()
        for index in matches[:32]:
            for ordinal in range(
                max(0, index - context), min(len(lines), index + context + 1)
            ):
                if ordinal not in seen:
                    selected.append([ordinal + 1, lines[ordinal]])
                    seen.add(ordinal)
        return {"token": token, "match_count": len(matches), "lines": selected}
    if mode in {"op", "value", "region"}:
        if len(arguments) != 2:
            raise SemanticQueryError(f"{mode} accepts exactly one REF")
        collection = {
            "op": "operations",
            "value": "values",
            "region": "regions",
        }[mode]
        keys = {
            "op": ("node_ref", "path"),
            "value": ("ref",),
            "region": ("id",),
        }[mode]
        matches = [
            item
            for item in payload[collection]
            if any(item.get(key) == arguments[1] for key in keys)
        ]
        if not matches:
            raise SemanticQueryError(f"{mode} reference is absent")
        return {"kind": mode, "ref": arguments[1], "items": matches[:8]}
    if mode == "path":
        if len(arguments) != 2:
            raise SemanticQueryError("path accepts one dotted PATH")
        value: Any = payload
        for component in arguments[1].split("."):
            if not component:
                raise SemanticQueryError("PATH contains an empty component")
            if isinstance(value, Mapping) and component in value:
                value = value[component]
            elif isinstance(value, list) and component.isdigit():
                index = int(component)
                if index >= len(value):
                    raise SemanticQueryError("PATH list index is out of range")
                value = value[index]
            else:
                raise SemanticQueryError("PATH is absent")
        return {"path": arguments[1], "value": value}
    raise SemanticQueryError("mode must be summary/raw/find/op/value/region/path")


__all__ = ["SemanticQueryError", "query_semantic_program"]
