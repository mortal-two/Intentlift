"""Schema-tolerant queries over accepted Layer-C lowering metadata.

Layer C intentionally permits frontends and the LLM to organize schedule facts
as a launch object, a named parameter group, or a list of anchored hints.  A
backend must therefore query a parameter by meaning instead of depending on one
JSON presentation.  Conflicting values fail closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LayerCResolutionError(ValueError):
    """Raised when a Layer-C parameter is absent or has conflicting values."""


@dataclass(frozen=True)
class ResolvedLayerCScalar:
    parameter: str
    value: bool | int | float | str
    paths: tuple[str, ...]


def _is_scalar(value: object) -> bool:
    return isinstance(value, (bool, int, float, str))


def resolve_layer_c_scalar(
    layer_c: Mapping[str, Any], parameter: str
) -> ResolvedLayerCScalar:
    """Resolve one scalar schedule parameter across equivalent Layer-C layouts."""

    if not isinstance(layer_c, Mapping):
        raise LayerCResolutionError("Layer C must be a JSON object")
    if not isinstance(parameter, str) or not parameter:
        raise LayerCResolutionError("Layer-C parameter name must be non-empty")
    candidates: list[tuple[str, bool | int | float | str]] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            named_parameter = value.get("parameter")
            named_value = value.get("value")
            if named_parameter == parameter and _is_scalar(named_value):
                candidates.append((f"{path}.value", named_value))
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key == parameter:
                    if _is_scalar(child):
                        candidates.append((child_path, child))
                    elif isinstance(child, Mapping) and _is_scalar(
                        child.get("value")
                    ):
                        candidates.append((f"{child_path}.value", child["value"]))
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(layer_c, "$.declared_layer_c")
    if not candidates:
        raise LayerCResolutionError(
            f"Layer-C parameter {parameter!r} is absent"
        )
    values: dict[str, bool | int | float | str] = {}
    for _path, value in candidates:
        values.setdefault(
            json.dumps(value, sort_keys=True, ensure_ascii=False), value
        )
    if len(values) != 1:
        details = ", ".join(f"{path}={value!r}" for path, value in candidates)
        raise LayerCResolutionError(
            f"Layer-C parameter {parameter!r} conflicts: {details}"
        )
    return ResolvedLayerCScalar(
        parameter=parameter,
        value=next(iter(values.values())),
        paths=tuple(sorted({path for path, _value in candidates})),
    )


__all__ = [
    "LayerCResolutionError",
    "ResolvedLayerCScalar",
    "resolve_layer_c_scalar",
]

