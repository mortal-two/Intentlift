"""Neutral logical ABI records shared by lowering, bundles, and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LogicalABIError(ValueError):
    """Raised when a logical lowering ABI is malformed."""


_ARGUMENT_KINDS = frozenset({"tensor", "scalar"})
_ARGUMENT_ROLES = frozenset({"input", "output", "inout", "parameter"})


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LogicalABIError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LogicalABIError(f"{field_name} must be an integer >= {minimum}")
    return value


def _strings(values: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LogicalABIError(f"{field_name} must be an array")
    return tuple(_identifier(value, field_name=field_name) for value in values)


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise LogicalABIError(f"{context} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise LogicalABIError(f"{context} missing fields: {missing}")
    if unknown:
        raise LogicalABIError(f"{context} has unknown fields: {unknown}")


@dataclass(frozen=True)
class LogicalArgument:
    name: str
    kind: str
    role: str
    dtype: str
    rank: int | None = None
    symbolic_shape: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _identifier(self.name, field_name="argument name")
        )
        if self.kind not in _ARGUMENT_KINDS:
            raise LogicalABIError(f"unsupported argument kind {self.kind!r}")
        if self.role not in _ARGUMENT_ROLES:
            raise LogicalABIError(f"unsupported argument role {self.role!r}")
        object.__setattr__(
            self, "dtype", _identifier(self.dtype, field_name="argument dtype")
        )
        shape = _strings(self.symbolic_shape, field_name="symbolic_shape")
        object.__setattr__(self, "symbolic_shape", shape)
        if self.kind == "tensor":
            if self.rank is None:
                raise LogicalABIError("tensor arguments require rank")
            rank = _integer(self.rank, field_name="argument rank")
            if len(shape) != rank:
                raise LogicalABIError(
                    f"argument {self.name!r} rank does not match symbolic_shape"
                )
            object.__setattr__(self, "rank", rank)
        elif self.rank is not None or shape:
            raise LogicalABIError(
                "scalar arguments must not carry rank or symbolic_shape"
            )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "dtype": self.dtype,
        }
        if self.kind == "tensor":
            value["rank"] = self.rank
            value["symbolic_shape"] = list(self.symbolic_shape)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicalArgument":
        _strict_keys(
            value,
            required={"name", "kind", "role", "dtype"},
            optional={"rank", "symbolic_shape"},
            context="logical argument",
        )
        return cls(
            name=value["name"],
            kind=value["kind"],
            role=value["role"],
            dtype=value["dtype"],
            rank=value.get("rank"),
            symbolic_shape=tuple(value.get("symbolic_shape", ())),
        )


__all__ = ["LogicalABIError", "LogicalArgument"]
