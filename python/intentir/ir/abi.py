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
    tensor_ref: str | None = None

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
        if self.tensor_ref is not None:
            object.__setattr__(
                self,
                "tensor_ref",
                _identifier(self.tensor_ref, field_name="argument tensor_ref"),
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
        if self.tensor_ref is not None:
            value["tensor_ref"] = self.tensor_ref
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicalArgument":
        _strict_keys(
            value,
            required={"name", "kind", "role", "dtype"},
            optional={"rank", "symbolic_shape", "tensor_ref"},
            context="logical argument",
        )
        return cls(
            name=value["name"],
            kind=value["kind"],
            role=value["role"],
            dtype=value["dtype"],
            rank=value.get("rank"),
            symbolic_shape=tuple(value.get("symbolic_shape", ())),
            tensor_ref=value.get("tensor_ref"),
        )


@dataclass(frozen=True)
class LogicalInterface:
    """Target-neutral callable ABI for an :class:`IntentProgram`.

    Argument tuple order is semantic and is preserved exactly.  Launch grids,
    constexpr values and compiler options are Layer C and therefore do not
    belong to this record.
    """

    function: str
    arguments: tuple[LogicalArgument, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "function", _identifier(self.function, field_name="interface function")
        )
        arguments = tuple(
            item if isinstance(item, LogicalArgument) else LogicalArgument.from_dict(item)
            for item in self.arguments
        )
        names = [item.name for item in arguments]
        if len(names) != len(set(names)):
            raise LogicalABIError("interface argument names must be unique")
        object.__setattr__(self, "arguments", arguments)
        outputs = _strings(self.outputs, field_name="interface outputs")
        if len(outputs) != len(set(outputs)):
            raise LogicalABIError("interface outputs must be unique")
        object.__setattr__(self, "outputs", outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "arguments": [item.to_dict() for item in self.arguments],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicalInterface":
        _strict_keys(
            value,
            required={"function", "arguments", "outputs"},
            context="logical interface",
        )
        arguments = value["arguments"]
        if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
            raise LogicalABIError("interface arguments must be an array")
        return cls(
            function=value["function"],
            arguments=tuple(LogicalArgument.from_dict(item) for item in arguments),
            outputs=tuple(value["outputs"]),
        )


__all__ = ["LogicalABIError", "LogicalArgument", "LogicalInterface"]
