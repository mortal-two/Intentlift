"""Strict, JSON-serializable data model for the SCRC Gate1 pipeline.

Gate1 deliberately separates the information owned by the semantic archive
(``J``), replay plan (``pi``), schedule (``C``), and source-only diagnostics.
The classes in this module are intentionally dependency-free dataclasses.  A
strict parser is used at every persisted boundary: unknown keys, malformed
types, non-finite JSON numbers, and duplicate identifiers are rejected rather
than silently ignored.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeVar

from intentir.ir.model import IntentFunction


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class SchemaError(ValueError):
    """Raised when a Gate1 JSON object does not match its strict schema."""


T = TypeVar("T", bound="StrictJSONMixin")


class StrictJSONMixin:
    """Convenience methods shared by strict Gate1 JSON dataclasses."""

    @classmethod
    def from_json(cls: type[T], value: dict[str, Any]) -> T:  # pragma: no cover - interface
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_json_str(cls: type[T], value: str) -> T:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid JSON: {exc}") from exc
        return cls.from_json(_object(parsed, cls.__name__))

    def to_json_str(self, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_json(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{path} object keys must be strings")
    return value


def _strict_object(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    result = _object(value, path)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise SchemaError(f"{path} has unknown key(s): {', '.join(unknown)}")
    missing = sorted(required - set(result))
    if missing:
        raise SchemaError(f"{path} is missing required key(s): {', '.join(missing)}")
    return result


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{path} must be a string")
    if nonempty and not value:
        raise SchemaError(f"{path} must not be empty")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    return value


def _string_list(value: Any, path: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def _json_value(value: Any, path: str) -> JsonValue:
    """Validate and defensively copy a proper RFC-8259 JSON value."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SchemaError(f"{path} object keys must be strings")
        return {
            key: _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise SchemaError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(_string(value, path))
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise SchemaError(f"{path} must be one of: {allowed}") from exc


def _unique(items: list[Any], key: str, path: str) -> None:
    values = [getattr(item, key) for item in items]
    if len(values) != len(set(values)):
        raise SchemaError(f"{path} contains duplicate {key} values")


def _strict_intent(value: Any, path: str = "SemanticArchive.intent") -> IntentFunction:
    """Strictly validate the existing open IntentFunction representation.

    ``attrs``, ``type``, ``layer_b`` and structured-loop payloads remain open
    JSON objects because the existing IntentIR dialect defines their keys per
    operation.  The enclosing Tensor/IntentOp/IntentFunction schemas are
    nevertheless closed, which prevents a misspelled field from disappearing
    through the legacy permissive ``IntentFunction.from_json`` parser.
    """

    root = _object(value, path)
    if "intent_function" in root:
        root = _strict_object(
            root,
            path=path,
            allowed={"intent_function"},
            required={"intent_function"},
        )
        fn = _object(root["intent_function"], f"{path}.intent_function")
    else:
        fn = root
    fn = _strict_object(
        fn,
        path=f"{path}.intent_function",
        allowed={"name", "inputs", "outputs", "layer_c", "body"},
        required={"name"},
    )
    _string(fn["name"], f"{path}.intent_function.name")

    def validate_tensor(item: Any, tensor_path: str) -> None:
        tensor = _strict_object(
            item,
            path=tensor_path,
            allowed={"name", "dtype", "shape"},
            required={"name", "dtype", "shape"},
        )
        _string(tensor["name"], f"{tensor_path}.name")
        _string(tensor["dtype"], f"{tensor_path}.dtype")
        shape = tensor["shape"]
        if not isinstance(shape, list):
            raise SchemaError(f"{tensor_path}.shape must be an array")
        for index, dim in enumerate(shape):
            if isinstance(dim, bool) or not isinstance(dim, (int, str)):
                raise SchemaError(f"{tensor_path}.shape[{index}] must be an integer or string")
            if isinstance(dim, int) and dim < 0:
                raise SchemaError(f"{tensor_path}.shape[{index}] must be non-negative")

    for collection in ("inputs", "outputs"):
        tensors = fn.get(collection, [])
        if not isinstance(tensors, list):
            raise SchemaError(f"{path}.intent_function.{collection} must be an array")
        for index, tensor in enumerate(tensors):
            validate_tensor(tensor, f"{path}.intent_function.{collection}[{index}]")

    def validate_op(item: Any, op_path: str) -> None:
        op = _strict_object(
            item,
            path=op_path,
            allowed={
                "op", "result", "operands", "attrs", "type", "layer_b", "out",
                "carry", "trip", "scan", "body", "yield",
            },
            required={"op"},
        )
        _string(op["op"], f"{op_path}.op")
        if "result" in op:
            result = op["result"]
            if not isinstance(result, str):
                _string_list(result, f"{op_path}.result", unique=True)
        if "operands" in op:
            _string_list(op["operands"], f"{op_path}.operands")
        if "out" in op and op["out"] is not None:
            _string(op["out"], f"{op_path}.out")
        if "yield" in op:
            _string_list(op["yield"], f"{op_path}.yield")
        for open_key in ("attrs", "type", "layer_b", "carry", "trip", "scan"):
            if open_key in op and op[open_key] is not None:
                _json_value(op[open_key], f"{op_path}.{open_key}")
        body = op.get("body", [])
        if not isinstance(body, list):
            raise SchemaError(f"{op_path}.body must be an array")
        for index, child in enumerate(body):
            validate_op(child, f"{op_path}.body[{index}]")

    if "layer_c" in fn:
        _json_value(fn["layer_c"], f"{path}.intent_function.layer_c")
    body = fn.get("body", [])
    if not isinstance(body, list):
        raise SchemaError(f"{path}.intent_function.body must be an array")
    for index, op in enumerate(body):
        validate_op(op, f"{path}.intent_function.body[{index}]")

    normalized = {"intent_function": copy.deepcopy(fn)}
    return IntentFunction.from_json(normalized)


class Frontend(str, Enum):
    TRITON = "triton"
    TILELANG = "tilelang"
    CUDA = "cuda"
    OTHER = "other"


class ParameterKind(str, Enum):
    TENSOR = "tensor"
    POINTER = "pointer"
    SCALAR = "scalar"
    CONSTEXPR = "constexpr"


class ParameterMode(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    VALUE = "value"


class ScheduleSlotKind(str, Enum):
    CONSTEXPR = "constexpr"
    COMPILER = "compiler"


class AccessKind(str, Enum):
    LOAD = "load"
    STORE = "store"
    ATOMIC = "atomic"
    SYNCHRONIZE = "synchronize"


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ATOMIC = "atomic"
    SYNCHRONIZE = "synchronize"


class AliasKind(str, Enum):
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"
    NO_ALIAS = "no_alias"


class ProvenanceRole(str, Enum):
    LIFTER = "lifter"
    REWRITER = "rewriter"
    INJECTOR = "injector"
    COMPILER = "compiler"
    COMPARATOR = "comparator"
    DIAGNOSTICIAN = "diagnostician"


class Gate1Status(str, Enum):
    PENDING = "pending"
    LIFT_REJECTED = "lift_rejected"
    LEAKAGE_REJECTED = "leakage_rejected"
    COMPILE_FAILED = "compile_failed"
    NO_CLOSURE = "no_closure"
    REPLAY_CLOSED = "replay_closed"


@dataclass
class BoundaryAccess(StrictJSONMixin):
    access_id: str
    kind: AccessKind
    tensor: str
    predicate: str | None = None
    fill_value: JsonValue = None
    node_ref: str | None = None
    index_expression: str | None = None

    _KEYS: ClassVar[set[str]] = {
        "access_id", "kind", "tensor", "predicate", "fill_value", "node_ref",
        "index_expression",
    }

    def __post_init__(self) -> None:
        self.access_id = _string(self.access_id, "BoundaryAccess.access_id")
        self.kind = _enum(AccessKind, self.kind, "BoundaryAccess.kind")
        self.tensor = _string(self.tensor, "BoundaryAccess.tensor")
        if self.predicate is not None:
            self.predicate = _string(self.predicate, "BoundaryAccess.predicate")
        self.fill_value = _json_value(self.fill_value, "BoundaryAccess.fill_value")
        if self.node_ref is not None:
            self.node_ref = _string(self.node_ref, "BoundaryAccess.node_ref")
        if self.index_expression is not None:
            self.index_expression = _string(
                self.index_expression, "BoundaryAccess.index_expression"
            )
        if self.kind in {AccessKind.LOAD, AccessKind.STORE, AccessKind.ATOMIC}:
            if self.index_expression is None:
                raise SchemaError(
                    f"BoundaryAccess {self.access_id!r} requires index_expression"
                )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BoundaryAccess":
        data = _strict_object(
            value, path="BoundaryAccess", allowed=cls._KEYS,
            required={"access_id", "kind", "tensor"},
        )
        return cls(
            access_id=data["access_id"],
            kind=data["kind"],
            tensor=data["tensor"],
            predicate=data.get("predicate"),
            fill_value=data.get("fill_value"),
            node_ref=data.get("node_ref"),
            index_expression=data.get("index_expression"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "access_id": self.access_id,
            "kind": self.kind.value,
            "tensor": self.tensor,
        }
        if self.predicate is not None:
            result["predicate"] = self.predicate
        if self.fill_value is not None:
            result["fill_value"] = copy.deepcopy(self.fill_value)
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.index_expression is not None:
            result["index_expression"] = self.index_expression
        return result


@dataclass
class BroadcastSpec(StrictJSONMixin):
    node_ref: str
    operand: str
    axes: list[int | str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_ref = _string(self.node_ref, "BroadcastSpec.node_ref")
        self.operand = _string(self.operand, "BroadcastSpec.operand")
        checked: list[int | str] = []
        for index, axis in enumerate(self.axes):
            if isinstance(axis, bool) or not isinstance(axis, (int, str)):
                raise SchemaError(f"BroadcastSpec.axes[{index}] must be an integer or string")
            checked.append(axis)
        self.axes = checked

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BroadcastSpec":
        data = _strict_object(
            value, path="BroadcastSpec", allowed={"node_ref", "operand", "axes"},
            required={"node_ref", "operand", "axes"},
        )
        if not isinstance(data["axes"], list):
            raise SchemaError("BroadcastSpec.axes must be an array")
        return cls(data["node_ref"], data["operand"], list(data["axes"]))

    def to_json(self) -> dict[str, Any]:
        return {"node_ref": self.node_ref, "operand": self.operand, "axes": list(self.axes)}


@dataclass
class AliasSpec(StrictJSONMixin):
    left: str
    right: str
    kind: AliasKind = AliasKind.MAY_ALIAS

    def __post_init__(self) -> None:
        self.left = _string(self.left, "AliasSpec.left")
        self.right = _string(self.right, "AliasSpec.right")
        self.kind = _enum(AliasKind, self.kind, "AliasSpec.kind")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "AliasSpec":
        data = _strict_object(
            value, path="AliasSpec", allowed={"left", "right", "kind"},
            required={"left", "right"},
        )
        return cls(data["left"], data["right"], data.get("kind", AliasKind.MAY_ALIAS.value))

    def to_json(self) -> dict[str, Any]:
        return {"left": self.left, "right": self.right, "kind": self.kind.value}


@dataclass
class EffectSpec(StrictJSONMixin):
    kind: EffectKind
    target: str
    node_ref: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(EffectKind, self.kind, "EffectSpec.kind")
        self.target = _string(self.target, "EffectSpec.target")
        if self.node_ref is not None:
            self.node_ref = _string(self.node_ref, "EffectSpec.node_ref")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "EffectSpec":
        data = _strict_object(
            value, path="EffectSpec", allowed={"kind", "target", "node_ref"},
            required={"kind", "target"},
        )
        return cls(data["kind"], data["target"], data.get("node_ref"))

    def to_json(self) -> dict[str, Any]:
        result = {"kind": self.kind.value, "target": self.target}
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        return result


@dataclass
class OutputMapping(StrictJSONMixin):
    output: str
    value: str
    node_ref: str | None = None

    def __post_init__(self) -> None:
        self.output = _string(self.output, "OutputMapping.output")
        self.value = _string(self.value, "OutputMapping.value")
        if self.node_ref is not None:
            self.node_ref = _string(self.node_ref, "OutputMapping.node_ref")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "OutputMapping":
        data = _strict_object(
            value, path="OutputMapping", allowed={"output", "value", "node_ref"},
            required={"output", "value"},
        )
        return cls(data["output"], data["value"], data.get("node_ref"))

    def to_json(self) -> dict[str, Any]:
        result = {"output": self.output, "value": self.value}
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        return result


@dataclass
class ReductionBoundary(StrictJSONMixin):
    """Boundary contract for a tiled reduction.

    ``identity`` records the algebraic identity used to pad the tile;
    ``masked_contribution`` records the effective contribution when the
    frontend expresses masking after loading.  At least one must be explicit.
    """

    node_ref: str
    tile_axis: int | str
    identity: JsonValue = None
    masked_contribution: JsonValue = None

    def __post_init__(self) -> None:
        self.node_ref = _string(self.node_ref, "ReductionBoundary.node_ref")
        if isinstance(self.tile_axis, bool) or not isinstance(self.tile_axis, (int, str)):
            raise SchemaError("ReductionBoundary.tile_axis must be an integer or string")
        if isinstance(self.tile_axis, int) and self.tile_axis < 0:
            raise SchemaError("ReductionBoundary.tile_axis must be non-negative")
        if isinstance(self.tile_axis, str):
            self.tile_axis = _string(self.tile_axis, "ReductionBoundary.tile_axis")
        self.identity = _json_value(self.identity, "ReductionBoundary.identity")
        self.masked_contribution = _json_value(
            self.masked_contribution, "ReductionBoundary.masked_contribution"
        )
        if self.identity is None and self.masked_contribution is None:
            raise SchemaError(
                "ReductionBoundary requires identity or masked_contribution"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ReductionBoundary":
        data = _strict_object(
            value,
            path="ReductionBoundary",
            allowed={"node_ref", "tile_axis", "identity", "masked_contribution"},
            required={"node_ref", "tile_axis"},
        )
        return cls(
            node_ref=data["node_ref"],
            tile_axis=data["tile_axis"],
            identity=data.get("identity"),
            masked_contribution=data.get("masked_contribution"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_ref": self.node_ref,
            "tile_axis": self.tile_axis,
        }
        if self.identity is not None:
            result["identity"] = copy.deepcopy(self.identity)
        if self.masked_contribution is not None:
            result["masked_contribution"] = copy.deepcopy(self.masked_contribution)
        return result


@dataclass
class Boundary(StrictJSONMixin):
    """Closed Layer-B schema owned by ``J``."""

    domains: dict[str, str] = field(default_factory=dict)
    accesses: list[BoundaryAccess] = field(default_factory=list)
    broadcasts: list[BroadcastSpec] = field(default_factory=list)
    aliases: list[AliasSpec] = field(default_factory=list)
    effects: list[EffectSpec] = field(default_factory=list)
    output_mappings: list[OutputMapping] = field(default_factory=list)
    reductions: list[ReductionBoundary] = field(default_factory=list)

    _KEYS: ClassVar[set[str]] = {
        "domains", "accesses", "broadcasts", "aliases", "effects", "output_mappings",
        "reductions",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.domains, dict):
            raise SchemaError("Boundary.domains must be an object")
        self.domains = {
            _string(key, "Boundary.domains key"): _string(value, f"Boundary.domains.{key}")
            for key, value in self.domains.items()
        }
        self.accesses = [
            item if isinstance(item, BoundaryAccess) else BoundaryAccess.from_json(item)
            for item in self.accesses
        ]
        self.broadcasts = [
            item if isinstance(item, BroadcastSpec) else BroadcastSpec.from_json(item)
            for item in self.broadcasts
        ]
        self.aliases = [
            item if isinstance(item, AliasSpec) else AliasSpec.from_json(item)
            for item in self.aliases
        ]
        self.effects = [
            item if isinstance(item, EffectSpec) else EffectSpec.from_json(item)
            for item in self.effects
        ]
        self.output_mappings = [
            item if isinstance(item, OutputMapping) else OutputMapping.from_json(item)
            for item in self.output_mappings
        ]
        self.reductions = [
            item if isinstance(item, ReductionBoundary) else ReductionBoundary.from_json(item)
            for item in self.reductions
        ]
        _unique(self.accesses, "access_id", "Boundary.accesses")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Boundary":
        data = _strict_object(value, path="Boundary", allowed=cls._KEYS)
        for key in cls._KEYS - {"domains"}:
            if key in data and not isinstance(data[key], list):
                raise SchemaError(f"Boundary.{key} must be an array")
        domains = data.get("domains", {})
        if not isinstance(domains, dict):
            raise SchemaError("Boundary.domains must be an object")
        return cls(
            domains=dict(domains),
            accesses=[BoundaryAccess.from_json(item) for item in data.get("accesses", [])],
            broadcasts=[BroadcastSpec.from_json(item) for item in data.get("broadcasts", [])],
            aliases=[AliasSpec.from_json(item) for item in data.get("aliases", [])],
            effects=[EffectSpec.from_json(item) for item in data.get("effects", [])],
            output_mappings=[OutputMapping.from_json(item) for item in data.get("output_mappings", [])],
            reductions=[ReductionBoundary.from_json(item) for item in data.get("reductions", [])],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "domains": dict(self.domains),
            "accesses": [item.to_json() for item in self.accesses],
            "broadcasts": [item.to_json() for item in self.broadcasts],
            "aliases": [item.to_json() for item in self.aliases],
            "effects": [item.to_json() for item in self.effects],
            "output_mappings": [item.to_json() for item in self.output_mappings],
            "reductions": [item.to_json() for item in self.reductions],
        }


@dataclass
class SemanticArchive(StrictJSONMixin):
    """The semantic archive ``J``: existing IntentFunction plus strict Layer B."""

    kernel_id: str
    intent: IntentFunction
    node_ids: dict[str, str]
    boundary: Boundary

    def __post_init__(self) -> None:
        self.kernel_id = _string(self.kernel_id, "SemanticArchive.kernel_id")
        if not isinstance(self.intent, IntentFunction):
            self.intent = _strict_intent(self.intent)
        else:
            # Reparse to enforce the closed persisted representation even for a
            # directly constructed legacy IntentFunction.
            self.intent = _strict_intent(self.intent.to_json())
        if not isinstance(self.node_ids, dict):
            raise SchemaError("SemanticArchive.node_ids must be an object")
        self.node_ids = {
            _string(node_id, "SemanticArchive.node_ids key"): _string(path, f"SemanticArchive.node_ids.{node_id}")
            for node_id, path in self.node_ids.items()
        }
        if len(set(self.node_ids.values())) != len(self.node_ids):
            raise SchemaError("SemanticArchive.node_ids paths must be unique")
        if not isinstance(self.boundary, Boundary):
            self.boundary = Boundary.from_json(self.boundary)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SemanticArchive":
        data = _strict_object(
            value,
            path="SemanticArchive",
            allowed={"kernel_id", "intent", "node_ids", "boundary"},
            required={"kernel_id", "intent", "node_ids", "boundary"},
        )
        node_ids = _object(data["node_ids"], "SemanticArchive.node_ids")
        return cls(
            kernel_id=data["kernel_id"],
            intent=_strict_intent(data["intent"]),
            node_ids=dict(node_ids),
            boundary=Boundary.from_json(data["boundary"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "intent": self.intent.to_json(),
            "node_ids": dict(self.node_ids),
            "boundary": self.boundary.to_json(),
        }


@dataclass
class KernelParameter(StrictJSONMixin):
    name: str
    kind: ParameterKind
    mode: ParameterMode = ParameterMode.VALUE
    dtype: str | None = None
    rank: int | None = None
    alias_of: str | None = None
    tensor_ref: str | None = None

    def __post_init__(self) -> None:
        self.name = _string(self.name, "KernelParameter.name")
        self.kind = _enum(ParameterKind, self.kind, "KernelParameter.kind")
        self.mode = _enum(ParameterMode, self.mode, "KernelParameter.mode")
        if self.dtype is not None:
            self.dtype = _string(self.dtype, "KernelParameter.dtype")
        if self.rank is not None:
            self.rank = _integer(self.rank, "KernelParameter.rank", minimum=0)
        if self.alias_of is not None:
            self.alias_of = _string(self.alias_of, "KernelParameter.alias_of")
        if self.tensor_ref is not None:
            self.tensor_ref = _string(self.tensor_ref, "KernelParameter.tensor_ref")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "KernelParameter":
        data = _strict_object(
            value,
            path="KernelParameter",
            allowed={"name", "kind", "mode", "dtype", "rank", "alias_of", "tensor_ref"},
            required={"name", "kind"},
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            mode=data.get("mode", ParameterMode.VALUE.value),
            dtype=data.get("dtype"),
            rank=data.get("rank"),
            alias_of=data.get("alias_of"),
            tensor_ref=data.get("tensor_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "mode": self.mode.value,
        }
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.rank is not None:
            result["rank"] = self.rank
        if self.alias_of is not None:
            result["alias_of"] = self.alias_of
        if self.tensor_ref is not None:
            result["tensor_ref"] = self.tensor_ref
        return result


@dataclass
class KernelInterface(StrictJSONMixin):
    kernel_id: str
    frontend: Frontend
    parameters: list[KernelParameter]
    grid: list[str]
    compiler_slots: list[str] = field(default_factory=list)
    function_name: str | None = None
    do_not_specialize: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kernel_id = _string(self.kernel_id, "KernelInterface.kernel_id")
        self.frontend = _enum(Frontend, self.frontend, "KernelInterface.frontend")
        self.parameters = [
            item if isinstance(item, KernelParameter) else KernelParameter.from_json(item)
            for item in self.parameters
        ]
        self.grid = _string_list(self.grid, "KernelInterface.grid")
        self.compiler_slots = _string_list(
            self.compiler_slots, "KernelInterface.compiler_slots", unique=True,
        )
        if self.function_name is None:
            self.function_name = self.kernel_id
        else:
            self.function_name = _string(self.function_name, "KernelInterface.function_name")
        self.do_not_specialize = _string_list(
            self.do_not_specialize, "KernelInterface.do_not_specialize", unique=True,
        )
        _unique(self.parameters, "name", "KernelInterface.parameters")
        names = {item.name for item in self.parameters}
        unknown_aliases = sorted(
            item.alias_of for item in self.parameters
            if item.alias_of is not None and item.alias_of not in names
        )
        if unknown_aliases:
            raise SchemaError(f"KernelInterface alias_of references unknown parameters: {unknown_aliases}")
        unknown_dns = sorted(set(self.do_not_specialize) - names)
        if unknown_dns:
            raise SchemaError(
                "KernelInterface.do_not_specialize references unknown parameters: "
                + ", ".join(unknown_dns)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "KernelInterface":
        data = _strict_object(
            value,
            path="KernelInterface",
            allowed={
                "kernel_id", "frontend", "parameters", "grid", "compiler_slots",
                "function_name", "do_not_specialize",
            },
            required={"kernel_id", "frontend", "parameters", "grid"},
        )
        if not isinstance(data["parameters"], list):
            raise SchemaError("KernelInterface.parameters must be an array")
        return cls(
            kernel_id=data["kernel_id"],
            frontend=data["frontend"],
            parameters=[KernelParameter.from_json(item) for item in data["parameters"]],
            grid=_string_list(data["grid"], "KernelInterface.grid"),
            compiler_slots=_string_list(
                data.get("compiler_slots", []), "KernelInterface.compiler_slots", unique=True,
            ),
            function_name=data.get("function_name"),
            do_not_specialize=_string_list(
                data.get("do_not_specialize", []),
                "KernelInterface.do_not_specialize",
                unique=True,
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "frontend": self.frontend.value,
            "parameters": [item.to_json() for item in self.parameters],
            "grid": list(self.grid),
            "compiler_slots": list(self.compiler_slots),
            "function_name": self.function_name,
            "do_not_specialize": list(self.do_not_specialize),
        }

    def rewriter_view(self) -> dict[str, Any]:
        """Return the callable interface facts supplied to the replay writer.

        ``do_not_specialize`` remains readable in legacy artifacts, but it is a
        Triton JIT policy rather than the SCRC kernel signature.  It is not a
        replay input and is not a hard Gate1 equality condition.
        """

        return {
            "kernel_id": self.kernel_id,
            "frontend": self.frontend.value,
            "parameters": [item.to_json() for item in self.parameters],
            "grid": list(self.grid),
            "compiler_slots": list(self.compiler_slots),
            "function_name": self.function_name,
        }


@dataclass
class ScheduleSlot(StrictJSONMixin):
    name: str
    kind: ScheduleSlotKind
    value_type: str = "int"
    required: bool = True

    def __post_init__(self) -> None:
        self.name = _string(self.name, "ScheduleSlot.name")
        self.kind = _enum(ScheduleSlotKind, self.kind, "ScheduleSlot.kind")
        self.value_type = _string(self.value_type, "ScheduleSlot.value_type")
        self.required = _boolean(self.required, "ScheduleSlot.required")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ScheduleSlot":
        data = _strict_object(
            value,
            path="ScheduleSlot",
            allowed={"name", "kind", "value_type", "required"},
            required={"name", "kind"},
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            value_type=data.get("value_type", "int"),
            required=data.get("required", True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value_type": self.value_type,
            "required": self.required,
        }


@dataclass
class ScheduleSpec(StrictJSONMixin):
    """Schedule ``C`` split into visible slot declarations and secret values."""

    slots: list[ScheduleSlot] = field(default_factory=list)
    values: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slots = [
            item if isinstance(item, ScheduleSlot) else ScheduleSlot.from_json(item)
            for item in self.slots
        ]
        _unique(self.slots, "name", "ScheduleSpec.slots")
        if not isinstance(self.values, dict):
            raise SchemaError("ScheduleSpec.values must be an object")
        self.values = {
            _string(key, "ScheduleSpec.values key"): _json_value(value, f"ScheduleSpec.values.{key}")
            for key, value in self.values.items()
        }
        known = {slot.name for slot in self.slots}
        unknown = sorted(set(self.values) - known)
        if unknown:
            raise SchemaError(f"ScheduleSpec.values has no declared slot(s): {', '.join(unknown)}")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ScheduleSpec":
        data = _strict_object(
            value,
            path="ScheduleSpec",
            allowed={"slots", "values"},
            required={"slots", "values"},
        )
        if not isinstance(data["slots"], list):
            raise SchemaError("ScheduleSpec.slots must be an array")
        values = _object(data["values"], "ScheduleSpec.values")
        return cls(
            slots=[ScheduleSlot.from_json(item) for item in data["slots"]],
            values=dict(values),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "slots": [item.to_json() for item in self.slots],
            "values": copy.deepcopy(self.values),
        }

    def rewriter_view(self) -> dict[str, Any]:
        """Return slot declarations only; concrete C values are secret."""

        return {"slots": [item.to_json() for item in self.slots]}

    def missing_required_values(self) -> list[str]:
        return [slot.name for slot in self.slots if slot.required and slot.name not in self.values]


@dataclass
class ReplayStep(StrictJSONMixin):
    """One pi step.  Its persisted schema permits exactly three fields."""

    rule: str
    node_ref: str | None = None
    refs: list[str] = field(default_factory=list)

    _KEYS: ClassVar[set[str]] = {"rule", "node_ref", "refs"}

    def __post_init__(self) -> None:
        self.rule = _string(self.rule, "ReplayStep.rule")
        if self.node_ref is not None:
            self.node_ref = _string(self.node_ref, "ReplayStep.node_ref")
        self.refs = _string_list(self.refs, "ReplayStep.refs", unique=True)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ReplayStep":
        data = _strict_object(
            value,
            path="ReplayStep",
            allowed=cls._KEYS,
            required={"rule"},
        )
        return cls(
            rule=data["rule"],
            node_ref=data.get("node_ref"),
            refs=_string_list(data.get("refs", []), "ReplayStep.refs", unique=True),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"rule": self.rule}
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.refs:
            result["refs"] = list(self.refs)
        return result


@dataclass
class ReplayPlan(StrictJSONMixin):
    steps: list[ReplayStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps = [
            item if isinstance(item, ReplayStep) else ReplayStep.from_json(item)
            for item in self.steps
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ReplayPlan":
        data = _strict_object(
            value, path="ReplayPlan", allowed={"steps"}, required={"steps"},
        )
        if not isinstance(data["steps"], list):
            raise SchemaError("ReplayPlan.steps must be an array")
        return cls([ReplayStep.from_json(item) for item in data["steps"]])

    def to_json(self) -> dict[str, Any]:
        return {"steps": [item.to_json() for item in self.steps]}


@dataclass
class SourceSpan(StrictJSONMixin):
    start_line: int
    end_line: int
    start_column: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        self.start_line = _integer(self.start_line, "SourceSpan.start_line", minimum=1)
        self.end_line = _integer(self.end_line, "SourceSpan.end_line", minimum=self.start_line)
        if self.start_column is not None:
            self.start_column = _integer(self.start_column, "SourceSpan.start_column", minimum=0)
        if self.end_column is not None:
            self.end_column = _integer(self.end_column, "SourceSpan.end_column", minimum=0)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SourceSpan":
        data = _strict_object(
            value,
            path="SourceSpan",
            allowed={"start_line", "end_line", "start_column", "end_column"},
            required={"start_line", "end_line"},
        )
        return cls(
            data["start_line"], data["end_line"],
            data.get("start_column"), data.get("end_column"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"start_line": self.start_line, "end_line": self.end_line}
        if self.start_column is not None:
            result["start_column"] = self.start_column
        if self.end_column is not None:
            result["end_column"] = self.end_column
        return result


@dataclass
class LocationEntry(StrictJSONMixin):
    span: SourceSpan
    node_refs: list[str] = field(default_factory=list)
    access_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            self.span = SourceSpan.from_json(self.span)
        self.node_refs = _string_list(self.node_refs, "LocationEntry.node_refs", unique=True)
        self.access_refs = _string_list(self.access_refs, "LocationEntry.access_refs", unique=True)
        if not self.node_refs and not self.access_refs:
            raise SchemaError("LocationEntry must reference a node or access")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LocationEntry":
        data = _strict_object(
            value,
            path="LocationEntry",
            allowed={"span", "node_refs", "access_refs"},
            required={"span"},
        )
        return cls(
            SourceSpan.from_json(data["span"]),
            _string_list(data.get("node_refs", []), "LocationEntry.node_refs", unique=True),
            _string_list(data.get("access_refs", []), "LocationEntry.access_refs", unique=True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "span": self.span.to_json(),
            "node_refs": list(self.node_refs),
            "access_refs": list(self.access_refs),
        }


@dataclass
class LocationMap(StrictJSONMixin):
    source_name: str
    entries: list[LocationEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_name = _string(self.source_name, "LocationMap.source_name")
        self.entries = [
            item if isinstance(item, LocationEntry) else LocationEntry.from_json(item)
            for item in self.entries
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LocationMap":
        data = _strict_object(
            value,
            path="LocationMap",
            allowed={"source_name", "entries"},
            required={"source_name", "entries"},
        )
        if not isinstance(data["entries"], list):
            raise SchemaError("LocationMap.entries must be an array")
        return cls(data["source_name"], [LocationEntry.from_json(item) for item in data["entries"]])

    def to_json(self) -> dict[str, Any]:
        return {"source_name": self.source_name, "entries": [item.to_json() for item in self.entries]}


@dataclass
class RuntimeArgument(StrictJSONMixin):
    name: str
    value: JsonValue = None
    dtype: str | None = None
    shape: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = _string(self.name, "RuntimeArgument.name")
        self.value = _json_value(self.value, "RuntimeArgument.value")
        if self.dtype is not None:
            self.dtype = _string(self.dtype, "RuntimeArgument.dtype")
        if not isinstance(self.shape, list):
            raise SchemaError("RuntimeArgument.shape must be an array")
        self.shape = [
            _integer(dim, f"RuntimeArgument.shape[{index}]", minimum=0)
            for index, dim in enumerate(self.shape)
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuntimeArgument":
        data = _strict_object(
            value,
            path="RuntimeArgument",
            allowed={"name", "value", "dtype", "shape"},
            required={"name"},
        )
        shape = data.get("shape", [])
        if not isinstance(shape, list):
            raise SchemaError("RuntimeArgument.shape must be an array")
        return cls(data["name"], data.get("value"), data.get("dtype"), list(shape))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.value is not None:
            result["value"] = copy.deepcopy(self.value)
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.shape:
            result["shape"] = list(self.shape)
        return result


@dataclass
class CompileCase(StrictJSONMixin):
    arguments: list[RuntimeArgument]
    grid: list[int]
    case_id: str = "default"

    def __post_init__(self) -> None:
        self.case_id = _string(self.case_id, "CompileCase.case_id")
        self.arguments = [
            item if isinstance(item, RuntimeArgument) else RuntimeArgument.from_json(item)
            for item in self.arguments
        ]
        _unique(self.arguments, "name", "CompileCase.arguments")
        if not isinstance(self.grid, list):
            raise SchemaError("CompileCase.grid must be an array")
        self.grid = [
            _integer(dim, f"CompileCase.grid[{index}]", minimum=1)
            for index, dim in enumerate(self.grid)
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CompileCase":
        data = _strict_object(
            value,
            path="CompileCase",
            allowed={"case_id", "arguments", "grid"},
            required={"arguments", "grid"},
        )
        if not isinstance(data["arguments"], list):
            raise SchemaError("CompileCase.arguments must be an array")
        if not isinstance(data["grid"], list):
            raise SchemaError("CompileCase.grid must be an array")
        return cls(
            arguments=[RuntimeArgument.from_json(item) for item in data["arguments"]],
            grid=list(data["grid"]),
            case_id=data.get("case_id", "default"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arguments": [item.to_json() for item in self.arguments],
            "grid": list(self.grid),
        }


@dataclass
class ProvenanceRecord(StrictJSONMixin):
    role: ProvenanceRole
    producer: str
    model: str | None = None
    source_digest: str | None = None
    created_at: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role = _enum(ProvenanceRole, self.role, "ProvenanceRecord.role")
        self.producer = _string(self.producer, "ProvenanceRecord.producer")
        for name in ("model", "source_digest", "created_at"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _string(value, f"ProvenanceRecord.{name}"))
        if not isinstance(self.metadata, dict):
            raise SchemaError("ProvenanceRecord.metadata must be an object")
        self.metadata = _json_value(self.metadata, "ProvenanceRecord.metadata")  # type: ignore[assignment]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ProvenanceRecord":
        data = _strict_object(
            value,
            path="ProvenanceRecord",
            allowed={"role", "producer", "model", "source_digest", "created_at", "metadata"},
            required={"role", "producer"},
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SchemaError("ProvenanceRecord.metadata must be an object")
        return cls(
            role=data["role"], producer=data["producer"], model=data.get("model"),
            source_digest=data.get("source_digest"), created_at=data.get("created_at"),
            metadata=dict(metadata),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role.value, "producer": self.producer}
        for name in ("model", "source_digest", "created_at"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.metadata:
            result["metadata"] = copy.deepcopy(self.metadata)
        return result


@dataclass
class Gate1Artifact(StrictJSONMixin):
    """Complete privileged Gate1 artifact.

    ``source`` and ``location_map`` are intentionally retained for lifter-side
    diagnostics, but :meth:`rewriter_view` is the only supported hand-off to a
    replay agent and excludes both, concrete schedule values, and provenance.
    """

    j: SemanticArchive
    interface: KernelInterface
    schedule: ScheduleSpec
    plan: ReplayPlan
    location_map: LocationMap
    source: str | None = None
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            self.j = SemanticArchive.from_json(self.j)
        if not isinstance(self.interface, KernelInterface):
            self.interface = KernelInterface.from_json(self.interface)
        if not isinstance(self.schedule, ScheduleSpec):
            self.schedule = ScheduleSpec.from_json(self.schedule)
        if not isinstance(self.plan, ReplayPlan):
            self.plan = ReplayPlan.from_json(self.plan)
        if not isinstance(self.location_map, LocationMap):
            self.location_map = LocationMap.from_json(self.location_map)
        if self.source is not None:
            self.source = _string(self.source, "Gate1Artifact.source", nonempty=False)
        self.provenance = [
            item if isinstance(item, ProvenanceRecord) else ProvenanceRecord.from_json(item)
            for item in self.provenance
        ]
        if self.j.kernel_id != self.interface.kernel_id:
            raise SchemaError("Gate1Artifact J and interface kernel_id values must match")

        tensor_names = {
            tensor.name for tensor in [*self.j.intent.inputs, *self.j.intent.outputs]
        }
        bad_tensor_refs = sorted(
            parameter.tensor_ref for parameter in self.interface.parameters
            if parameter.tensor_ref is not None and parameter.tensor_ref not in tensor_names
        )
        if bad_tensor_refs:
            raise SchemaError(
                "KernelParameter.tensor_ref references unknown J tensor(s): "
                + ", ".join(bad_tensor_refs)
            )

        node_refs = set(self.j.node_ids) | set(self.j.node_ids.values())
        boundary_node_refs = [
            *(item.node_ref for item in self.j.boundary.accesses if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.broadcasts),
            *(item.node_ref for item in self.j.boundary.effects if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.output_mappings if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.reductions),
        ]
        unknown_boundary_nodes = sorted(set(boundary_node_refs) - node_refs)
        if unknown_boundary_nodes:
            raise SchemaError(
                "Boundary references unknown J node(s): " + ", ".join(unknown_boundary_nodes)
            )

        valid_plan_refs = (
            node_refs
            | tensor_names
            | set(self.j.boundary.domains)
            | {item.access_id for item in self.j.boundary.accesses}
        )
        plan_refs = [
            *(step.node_ref for step in self.plan.steps if step.node_ref is not None),
            *(ref for step in self.plan.steps for ref in step.refs),
        ]
        unknown_plan_refs = sorted(set(plan_refs) - valid_plan_refs)
        if unknown_plan_refs:
            raise SchemaError(
                "ReplayPlan references unknown J identifier(s): "
                + ", ".join(unknown_plan_refs)
            )

        access_ids = {item.access_id for item in self.j.boundary.accesses}
        location_nodes = {
            ref for entry in self.location_map.entries for ref in entry.node_refs
        }
        location_accesses = {
            ref for entry in self.location_map.entries for ref in entry.access_refs
        }
        if location_nodes - node_refs:
            raise SchemaError("LocationMap references unknown J nodes")
        if location_accesses - access_ids:
            raise SchemaError("LocationMap references unknown boundary accesses")

        schedule_by_name = {slot.name: slot for slot in self.schedule.slots}
        missing_compiler_slots = sorted(
            name for name in self.interface.compiler_slots
            if name not in schedule_by_name
            or schedule_by_name[name].kind is not ScheduleSlotKind.COMPILER
        )
        if missing_compiler_slots:
            raise SchemaError(
                "KernelInterface.compiler_slots are not declared compiler schedule slots: "
                + ", ".join(missing_compiler_slots)
            )
        missing_values = self.schedule.missing_required_values()
        if missing_values:
            raise SchemaError(
                "Gate1Artifact is missing required schedule value(s): "
                + ", ".join(missing_values)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Gate1Artifact":
        data = _strict_object(
            value,
            path="Gate1Artifact",
            allowed={"j", "interface", "schedule", "plan", "location_map", "source", "provenance"},
            required={"j", "interface", "schedule", "plan", "location_map"},
        )
        provenance = data.get("provenance", [])
        if not isinstance(provenance, list):
            raise SchemaError("Gate1Artifact.provenance must be an array")
        return cls(
            j=SemanticArchive.from_json(data["j"]),
            interface=KernelInterface.from_json(data["interface"]),
            schedule=ScheduleSpec.from_json(data["schedule"]),
            plan=ReplayPlan.from_json(data["plan"]),
            location_map=LocationMap.from_json(data["location_map"]),
            source=data.get("source"),
            provenance=[ProvenanceRecord.from_json(item) for item in provenance],
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "j": self.j.to_json(),
            "interface": self.interface.to_json(),
            "schedule": self.schedule.to_json(),
            "plan": self.plan.to_json(),
            "location_map": self.location_map.to_json(),
            "provenance": [item.to_json() for item in self.provenance],
        }
        if self.source is not None:
            result["source"] = self.source
        return result

    def rewriter_view(self, *, masked: bool = False) -> dict[str, Any]:
        """Return the source-isolated hand-off payload for the replay agent."""

        if masked:
            return _masked_rewriter_view(self)
        return {
            "j": self.j.to_json(),
            "interface": self.interface.rewriter_view(),
            "schedule": self.schedule.rewriter_view(),
            "plan": self.plan.to_json(),
        }


@dataclass
class Gate1Result(StrictJSONMixin):
    status: Gate1Status
    artifact: Gate1Artifact | None = None
    diagnostics: list[str] = field(default_factory=list)
    relift_attempts: int = 0
    repair_attempts: int = 0
    leakage_ratio: float | None = None

    def __post_init__(self) -> None:
        self.status = _enum(Gate1Status, self.status, "Gate1Result.status")
        if self.artifact is not None and not isinstance(self.artifact, Gate1Artifact):
            self.artifact = Gate1Artifact.from_json(self.artifact)
        self.diagnostics = _string_list(self.diagnostics, "Gate1Result.diagnostics")
        self.relift_attempts = _integer(self.relift_attempts, "Gate1Result.relift_attempts", minimum=0)
        self.repair_attempts = _integer(self.repair_attempts, "Gate1Result.repair_attempts", minimum=0)
        if self.leakage_ratio is not None:
            if isinstance(self.leakage_ratio, bool) or not isinstance(self.leakage_ratio, (int, float)):
                raise SchemaError("Gate1Result.leakage_ratio must be a finite number")
            self.leakage_ratio = float(self.leakage_ratio)
            if not math.isfinite(self.leakage_ratio) or self.leakage_ratio < 0:
                raise SchemaError("Gate1Result.leakage_ratio must be a finite non-negative number")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Gate1Result":
        data = _strict_object(
            value,
            path="Gate1Result",
            allowed={
                "status", "artifact", "diagnostics", "relift_attempts",
                "repair_attempts", "leakage_ratio",
            },
            required={"status"},
        )
        return cls(
            status=data["status"],
            artifact=Gate1Artifact.from_json(data["artifact"]) if data.get("artifact") is not None else None,
            diagnostics=_string_list(data.get("diagnostics", []), "Gate1Result.diagnostics"),
            relift_attempts=data.get("relift_attempts", 0),
            repair_attempts=data.get("repair_attempts", 0),
            leakage_ratio=data.get("leakage_ratio"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status.value,
            "diagnostics": list(self.diagnostics),
            "relift_attempts": self.relift_attempts,
            "repair_attempts": self.repair_attempts,
        }
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_json()
        if self.leakage_ratio is not None:
            result["leakage_ratio"] = self.leakage_ratio
        return result


def _masked_rewriter_view(artifact: Gate1Artifact) -> dict[str, Any]:
    """Create a consistently opaque J/pi/interface view for leakage ablation."""

    tensor_names = [tensor.name for tensor in artifact.j.intent.inputs]
    tensor_names.extend(tensor.name for tensor in artifact.j.intent.outputs)
    tensor_map = {name: f"t{index}" for index, name in enumerate(dict.fromkeys(tensor_names))}
    node_map = {name: f"n{index}" for index, name in enumerate(artifact.j.node_ids)}
    access_map = {
        access.access_id: f"a{index}" for index, access in enumerate(artifact.j.boundary.accesses)
    }
    domain_map = {name: f"d{index}" for index, name in enumerate(artifact.j.boundary.domains)}

    reference_map = {
        **tensor_map, **node_map, **access_map, **domain_map,
    }
    fallback_reference_map: dict[str, str] = {}

    def renamed(value: str) -> str:
        if value in reference_map:
            return reference_map[value]
        # Replace exact identifier components in qualified references while
        # leaving positional syntax (e.g. body[2]) intact.
        parts = re.split(r"([.\[\]/:])", value)
        return "".join(reference_map.get(part, part) for part in parts)

    def opaque_reference(value: str) -> str:
        mapped = renamed(value)
        if mapped != value or value in reference_map:
            return mapped
        if value not in fallback_reference_map:
            fallback_reference_map[value] = f"r{len(fallback_reference_map)}"
        return fallback_reference_map[value]

    path_to_node = {path: node_map[node_id] for node_id, path in artifact.j.node_ids.items()}
    anonymous_node_count = 0

    def mask_ops(ops: list[Any], prefix: str) -> list[dict[str, Any]]:
        nonlocal anonymous_node_count
        result: list[dict[str, Any]] = []
        for index, op in enumerate(ops):
            path = f"{prefix}[{index}]"
            node_id = path_to_node.get(path)
            if node_id is None:
                node_id = path_to_node.get(f"intent_function.{path}")
            if node_id is None:
                node_id = f"n_anon_{anonymous_node_count}"
                anonymous_node_count += 1
            masked_op: dict[str, Any] = {"op": "<masked>", "result": node_id}
            if getattr(op, "body", None):
                masked_op["body"] = mask_ops(op.body, f"{path}.body")
            result.append(masked_op)
        return result

    masked_intent = {
        "intent_function": {
            "name": "k0",
            "inputs": [
                {
                    "name": tensor_map[tensor.name],
                    "dtype": "<masked>",
                    "shape": ["<masked>"] * len(tensor.shape),
                }
                for tensor in artifact.j.intent.inputs
            ],
            "outputs": [
                {
                    "name": tensor_map[tensor.name],
                    "dtype": "<masked>",
                    "shape": ["<masked>"] * len(tensor.shape),
                }
                for tensor in artifact.j.intent.outputs
            ],
            "layer_c": {},
            "body": mask_ops(artifact.j.intent.body, "body"),
        }
    }

    masked_boundary = {
        "domains": {domain_map[name]: "<masked>" for name in artifact.j.boundary.domains},
        "accesses": [
            {
                "access_id": access_map[access.access_id],
                "kind": "<masked>",
                "tensor": opaque_reference(access.tensor),
                "index_expression": "<masked>",
                **({"predicate": "<masked>"} if access.predicate is not None else {}),
                **({"fill_value": "<masked>"} if access.fill_value is not None else {}),
                **({"node_ref": opaque_reference(access.node_ref)} if access.node_ref is not None else {}),
            }
            for access in artifact.j.boundary.accesses
        ],
        "broadcasts": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "operand": opaque_reference(item.operand),
                "axes": ["<masked>"],
            }
            for item in artifact.j.boundary.broadcasts
        ],
        "aliases": [
            {
                "left": opaque_reference(item.left),
                "right": opaque_reference(item.right),
                "kind": "<masked>",
            }
            for item in artifact.j.boundary.aliases
        ],
        "effects": [
            {
                "kind": "<masked>", "target": opaque_reference(item.target),
                **({"node_ref": opaque_reference(item.node_ref)} if item.node_ref is not None else {}),
            }
            for item in artifact.j.boundary.effects
        ],
        "output_mappings": [
            {
                "output": opaque_reference(item.output), "value": "<masked>",
                **({"node_ref": opaque_reference(item.node_ref)} if item.node_ref is not None else {}),
            }
            for item in artifact.j.boundary.output_mappings
        ],
        "reductions": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "tile_axis": "<masked>",
                **({"identity": "<masked>"} if item.identity is not None else {}),
                **(
                    {"masked_contribution": "<masked>"}
                    if item.masked_contribution is not None else {}
                ),
            }
            for item in artifact.j.boundary.reductions
        ],
    }

    masked_parameters = []
    for parameter in artifact.interface.parameters:
        item = parameter.to_json()
        if parameter.tensor_ref is not None:
            item["tensor_ref"] = opaque_reference(parameter.tensor_ref)
        masked_parameters.append(item)

    # The masked run must remain executable with the exact same CompileCase.
    # Therefore ABI names, grid expressions, compiler slots and C slot names
    # are a declared channel and remain unchanged.  Production kernel IDs and
    # parameter names are required to be semantically neutral.
    masked_schedule = artifact.schedule.rewriter_view()
    masked_plan = {
        "steps": [
            {
                "rule": step.rule,
                **({"node_ref": opaque_reference(step.node_ref)} if step.node_ref is not None else {}),
                **({"refs": [opaque_reference(ref) for ref in step.refs]} if step.refs else {}),
            }
            for step in artifact.plan.steps
        ]
    }
    return {
        "j": {
            "kernel_id": "k0",
            "intent": masked_intent,
            "node_ids": {
                node_map[node_id]: f"node[{index}]"
                for index, node_id in enumerate(artifact.j.node_ids)
            },
            "boundary": masked_boundary,
        },
        "interface": {
            "kernel_id": artifact.interface.kernel_id,
            "function_name": artifact.interface.function_name,
            "frontend": artifact.interface.frontend.value,
            "parameters": masked_parameters,
            "grid": list(artifact.interface.grid),
            "compiler_slots": list(artifact.interface.compiler_slots),
        },
        "schedule": masked_schedule,
        "plan": masked_plan,
    }


__all__ = [
    "AccessKind", "AliasKind", "AliasSpec", "Boundary", "BoundaryAccess",
    "BroadcastSpec", "CompileCase", "EffectKind", "EffectSpec", "Frontend",
    "Gate1Artifact", "Gate1Result", "Gate1Status", "JsonValue", "KernelInterface",
    "KernelParameter", "LocationEntry", "LocationMap", "OutputMapping",
    "ParameterKind", "ParameterMode", "ProvenanceRecord", "ProvenanceRole",
    "ReductionBoundary", "ReplayPlan", "ReplayStep", "RuntimeArgument", "ScheduleSlot",
    "ScheduleSlotKind", "ScheduleSpec", "SchemaError", "SemanticArchive",
    "SourceSpan", "StrictJSONMixin",
]
