"""Static, materialization, and runtime checks for J-owned scalar equalities.

The relation itself is authored in J.  The helpers in this module never infer
one from tensor shapes, Harness values, C, or pi; they only validate and
project an already declared relation for J consumers.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import numpy as np


_INTEGRAL_DTYPES = re.compile(r"^(?:(?P<sign>[iu])(?P<bits>8|16|32|64)|index)$")


class ValueDimensionRelationError(ValueError):
    """A declared relation or its public valuation is not closed."""


def _dtype_key(value: str) -> str:
    aliases = {
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    spelling = value.casefold().removeprefix("torch.")
    return aliases.get(spelling, spelling)


def validate_value_dimension_relations(j: Any) -> None:
    """Validate only model-authored relation declarations, without valuations."""

    inputs = {item.name: item for item in j.intent.inputs}
    dimensions = {
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str)
    }
    seen_relations: set[tuple[str, str, str]] = set()
    for index, relation in enumerate(j.value_dimension_relations):
        path = f"j.value_dimension_relations[{index}]"
        if relation.relation != "eq":
            raise ValueDimensionRelationError(
                f"{path}.relation must be the closed equality 'eq'"
            )
        value = inputs.get(relation.value_ref)
        if (
            value is None
            or re.fullmatch(r"t[0-9]+", relation.value_ref) is None
            or value.shape
            or _INTEGRAL_DTYPES.fullmatch(_dtype_key(value.dtype)) is None
        ):
            raise ValueDimensionRelationError(
                f"{path}.value_ref must name an integral rank-0 J input t*"
            )
        if (
            relation.dimension_ref not in dimensions
            or re.fullmatch(r"x[0-9]+", relation.dimension_ref) is None
        ):
            raise ValueDimensionRelationError(
                f"{path}.dimension_ref must name a declared symbolic x* dimension"
            )
        key = (relation.value_ref, relation.dimension_ref, relation.relation)
        if key in seen_relations:
            raise ValueDimensionRelationError(
                f"{path} duplicates an earlier value/dimension equality"
            )
        seen_relations.add(key)


def _closed_integer(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueDimensionRelationError(f"{context} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    raise ValueDimensionRelationError(f"{context} must be an integer")


def _dtype_bounds(dtype: str) -> tuple[int, int]:
    key = _dtype_key(dtype)
    if key == "index":
        return -(1 << 63), (1 << 63) - 1
    match = _INTEGRAL_DTYPES.fullmatch(key)
    if match is None:  # validate_value_dimension_relations normally catches it.
        raise ValueDimensionRelationError(
            f"logical scalar dtype {dtype!r} is not integral"
        )
    bits = int(match.group("bits"))
    if match.group("sign") == "u":
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _check_representable(value: int, dtype: str, *, context: str) -> None:
    lower, upper = _dtype_bounds(dtype)
    if not lower <= value <= upper:
        raise ValueDimensionRelationError(
            f"{context}={value} is not representable by logical scalar dtype "
            f"{dtype!r}"
        )


def resolve_value_dimension_equalities(
    j: Any,
    dimensions: Mapping[str, int],
) -> tuple[tuple[int, int], ...]:
    """Resolve authored equalities to stable ``(input arg index, value)`` pairs.

    This is the only projection needed by production materializers.  Multiple
    dimensions may equal the same scalar input; equal valuations collapse to
    one pair, while conflicting valuations fail closed.
    """

    validate_value_dimension_relations(j)
    input_indices = {item.name: index for index, item in enumerate(j.intent.inputs)}
    input_types = {item.name: item.dtype for item in j.intent.inputs}
    resolved: dict[int, int] = {}
    for relation in j.value_dimension_relations:
        if relation.dimension_ref not in dimensions:
            raise ValueDimensionRelationError(
                "value/dimension equality lacks dimension valuation "
                f"{relation.dimension_ref!r}"
            )
        value = _closed_integer(
            dimensions[relation.dimension_ref],
            context=f"dimension {relation.dimension_ref!r}",
        )
        _check_representable(
            value,
            input_types[relation.value_ref],
            context=f"dimension {relation.dimension_ref}",
        )
        arg_index = input_indices[relation.value_ref]
        previous = resolved.setdefault(arg_index, value)
        if previous != value:
            raise ValueDimensionRelationError(
                f"logical value {relation.value_ref!r} is equated to conflicting "
                f"dimension valuations {previous} and {value}"
            )
    return tuple(sorted(resolved.items()))


def _rank_zero_integral(value: Any, *, ref: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueDimensionRelationError(
            f"logical value {ref!r} must be one integral rank-0 invocation value"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind not in {"i", "u"}:
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            )
        return int(value.item())

    # Torch tensors, including CUDA tensors, cannot in general be passed to
    # np.asarray.  Use their typed scalar API without importing torch or
    # accepting arbitrary ``item()``-convertible objects.
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ == "Tensor":
        try:
            rank = int(value.dim())
            dtype = _dtype_key(str(value.dtype))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            ) from exc
        if rank != 0 or _INTEGRAL_DTYPES.fullmatch(dtype) is None:
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            )
        return int(value.item())

    raise ValueDimensionRelationError(
        f"logical value {ref!r} must be one integral rank-0 invocation value"
    )


def validate_value_dimension_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
    logical_values: Mapping[str, Any],
) -> None:
    """Check actual invocation values against all authored equality relations."""

    resolved = dict(resolve_value_dimension_equalities(j, dimensions))
    input_indices = {item.name: index for index, item in enumerate(j.intent.inputs)}
    input_types = {item.name: item.dtype for item in j.intent.inputs}
    for relation in j.value_dimension_relations:
        if relation.value_ref not in logical_values:
            raise ValueDimensionRelationError(
                f"value/dimension equality lacks logical value "
                f"{relation.value_ref!r}"
            )
        value = _rank_zero_integral(
            logical_values[relation.value_ref], ref=relation.value_ref
        )
        _check_representable(
            value,
            input_types[relation.value_ref],
            context=f"logical value {relation.value_ref}",
        )
        dimension = resolved[input_indices[relation.value_ref]]
        if value != dimension:
            raise ValueDimensionRelationError(
                "J value/dimension equality mismatch: "
                f"{relation.value_ref}={value}, "
                f"{relation.dimension_ref}={dimension}"
            )


__all__ = [
    "ValueDimensionRelationError",
    "resolve_value_dimension_equalities",
    "validate_value_dimension_instantiation",
    "validate_value_dimension_relations",
]
