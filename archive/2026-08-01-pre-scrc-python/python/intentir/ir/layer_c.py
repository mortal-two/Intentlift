"""Strict queries over the one canonical pure-C JSON presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class LayerCResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedLayerCScalar:
    role: str
    slot: str
    value: bool | int | float | str
    path: str


def resolve_layer_c_scalar(
    layer_c: Mapping[str, Any], role: str
) -> ResolvedLayerCScalar:
    """Resolve a schedule role from ``{slots, values}``; reject all aliases."""

    if not isinstance(layer_c, Mapping) or set(layer_c) != {"slots", "values"}:
        raise LayerCResolutionError(
            "Layer C must be the canonical {slots, values} object"
        )
    slots = layer_c["slots"]
    values = layer_c["values"]
    if not isinstance(slots, list) or not isinstance(values, Mapping):
        raise LayerCResolutionError("Layer C slots/values have invalid types")
    if not isinstance(role, str) or not role:
        raise LayerCResolutionError("Layer-C role must be non-empty")
    matches: list[str] = []
    declared: set[str] = set()
    for index, item in enumerate(slots):
        if not isinstance(item, Mapping):
            raise LayerCResolutionError(f"Layer C slot {index} must be an object")
        if set(item) != {"name", "kind", "role", "value_type", "required"}:
            raise LayerCResolutionError("Layer C slot uses a non-canonical schema")
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise LayerCResolutionError("Layer C slot name must be non-empty")
        if name in declared:
            raise LayerCResolutionError("Layer C slot names must be unique")
        declared.add(name)
        if item["role"] == role:
            matches.append(name)
    if set(values) - declared:
        raise LayerCResolutionError("Layer C values contain undeclared slots")
    if not matches:
        raise LayerCResolutionError(f"Layer-C role {role!r} is absent")
    if len(matches) != 1:
        raise LayerCResolutionError(f"Layer-C role {role!r} is ambiguous")
    slot = matches[0]
    if slot not in values:
        raise LayerCResolutionError(f"Layer-C role {role!r} has no value")
    value = values[slot]
    if not isinstance(value, (bool, int, float, str)):
        raise LayerCResolutionError("Layer-C scalar value is not a JSON scalar")
    return ResolvedLayerCScalar(role, slot, value, f"$.values.{slot}")


__all__ = [
    "LayerCResolutionError", "ResolvedLayerCScalar", "resolve_layer_c_scalar",
]
