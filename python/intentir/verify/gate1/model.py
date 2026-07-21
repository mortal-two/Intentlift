"""Strict, JSON-serializable data model for the SCRC Gate1 pipeline.

Gate1 deliberately separates the information owned by the semantic archive
(``J``), realization witness (``pi``), pure schedule (``C``), and harness-only
diagnostics.
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

from intentir.ir.intent_op_contract import (
    IntentOpContractError,
    validate_intent_op_structure,
)
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.pi import RealizationWitness, RealizationWitnessError


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise SchemaError(f"{path} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    return result


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    return result


def _string_list(value: Any, path: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def _dtype_key(value: str) -> str:
    key = value.removeprefix("torch.").lower()
    aliases = {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float64": "f64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "bool": "i1",
    }
    return aliases.get(key, key)


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
    """Strictly validate IntentFunction, including closed semantic loops.

    Operation-specific ``attrs`` and ``type`` remain extensible while the
    compositional IntentOp core is being audited.  Layer B is never accepted at
    op scope, and destination/writeback contracts never use a per-op ``out``
    shortcut.  Loop state carries cross-region SSA semantics, so SCRC requires
    a closed, typed ``init -> body_argument -> next_value -> result_value``
    contract and checks its positional ``result``/``yield`` witnesses.
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
        allowed={"name", "inputs", "outputs", "returns", "body"},
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
            if isinstance(dim, str):
                _string(dim, f"{tensor_path}.shape[{index}]")
            else:
                _integer(dim, f"{tensor_path}.shape[{index}]", minimum=0)

    tensor_contracts: dict[str, tuple[str, list[int | str]]] = {}
    for collection in ("inputs", "outputs"):
        tensors = fn.get(collection, [])
        if not isinstance(tensors, list):
            raise SchemaError(f"{path}.intent_function.{collection} must be an array")
        for index, tensor in enumerate(tensors):
            tensor_path = f"{path}.intent_function.{collection}[{index}]"
            validate_tensor(tensor, tensor_path)
            tensor_object = _object(tensor, tensor_path)
            name = _string(tensor_object["name"], f"{tensor_path}.name")
            if name in tensor_contracts:
                raise SchemaError(f"{path}.intent_function has duplicate tensor {name!r}")
            tensor_contracts[name] = (
                _string(tensor_object["dtype"], f"{tensor_path}.dtype"),
                list(tensor_object["shape"]),
            )

    semantic_dimensions = {
        dim
        for _dtype, shape in tensor_contracts.values()
        for dim in shape
        if isinstance(dim, str)
    }
    literal_initializers = {
        "NEG_INF", "POS_INF", "ZERO", "ONE", "zeros", "ones",
        "false", "true",
    }
    declared_node_ids: set[str] = set()

    def is_literal_reference(item: Any) -> bool:
        if item is None or isinstance(item, (bool, int, float)):
            return True
        if not isinstance(item, str):
            return False
        if item in literal_initializers:
            return True
        try:
            float(item)
        except ValueError:
            return False
        return True

    def result_names(op: dict[str, Any], op_path: str) -> list[str]:
        result = op.get("result")
        if result is None:
            return []
        if isinstance(result, str):
            return [_string(result, f"{op_path}.result")]
        return _string_list(result, f"{op_path}.result", unique=True)

    def validate_loop_type(item: Any, type_path: str) -> tuple[str, list[int | str]]:
        value_type = _strict_object(
            item,
            path=type_path,
            allowed={"dtype", "shape"},
            required={"dtype", "shape"},
        )
        dtype = _string(value_type["dtype"], f"{type_path}.dtype")
        shape = value_type["shape"]
        if not isinstance(shape, list):
            raise SchemaError(f"{type_path}.shape must be an array")
        normalized_shape: list[int | str] = []
        for index, dim in enumerate(shape):
            if isinstance(dim, str):
                normalized_shape.append(_string(dim, f"{type_path}.shape[{index}]"))
            else:
                normalized_shape.append(
                    _integer(dim, f"{type_path}.shape[{index}]", minimum=0)
                )
        return dtype, normalized_shape

    def validate_sequence(
        items: Any,
        sequence_path: str,
        visible: set[str],
        *,
        strict_operands: bool,
    ) -> tuple[set[str], set[str]]:
        if not isinstance(items, list):
            raise SchemaError(f"{sequence_path} must be an array")
        scope_visible = set(visible)
        definitions: set[str] = set()
        for index, child in enumerate(items):
            child_path = f"{sequence_path}[{index}]"
            produced = validate_op(
                child,
                child_path,
                scope_visible,
                strict_operands=strict_operands,
            )
            duplicates = sorted(set(produced) & scope_visible)
            if duplicates:
                raise SchemaError(
                    f"{child_path}.result redefines visible value(s): "
                    + ", ".join(duplicates)
                )
            scope_visible.update(produced)
            definitions.update(produced)
        return scope_visible, definitions

    def validate_op(
        item: Any,
        op_path: str,
        visible: set[str],
        *,
        strict_operands: bool,
    ) -> list[str]:
        try:
            validate_intent_op_structure(item, path=op_path, recursive=False)
        except IntentOpContractError as exc:
            raise SchemaError(str(exc)) from exc
        op = _object(item, op_path)
        op_name = _string(op["op"], f"{op_path}.op")
        if "id" in op:
            node_id = _string(op["id"], f"{op_path}.id")
            if node_id in declared_node_ids:
                raise SchemaError(f"{op_path}.id duplicates node id {node_id!r}")
            declared_node_ids.add(node_id)
        produced = result_names(op, op_path)
        if "operands" in op:
            operands = _string_list(op["operands"], f"{op_path}.operands")
            if strict_operands:
                unknown_operands = sorted(
                    operand
                    for operand in operands
                    if operand not in visible and not is_literal_reference(operand)
                )
                if unknown_operands:
                    raise SchemaError(
                        f"{op_path}.operands reference value(s) outside the current "
                        f"scope: {', '.join(unknown_operands)}"
                    )
        for open_key in ("attrs", "type"):
            if open_key in op and op[open_key] is not None:
                _json_value(op[open_key], f"{op_path}.{open_key}")

        if op_name != "loop":
            return produced

        loop_scope = _string(op["scope_id"], f"{op_path}.scope_id")

        trip = _strict_object(
            op["trip"],
            path=f"{op_path}.trip",
            allowed={"axis"},
            required={"axis"},
        )
        trip_axis = _string(trip["axis"], f"{op_path}.trip.axis")
        if semantic_dimensions and trip_axis not in semantic_dimensions:
            raise SchemaError(
                f"{op_path}.trip.axis {trip_axis!r} is not a J tensor dimension"
            )

        scans = op["scan"]
        if not isinstance(scans, list):
            raise SchemaError(f"{op_path}.scan must be an array")
        scan_names: list[str] = []
        for index, scan_item in enumerate(scans):
            scan_path = f"{op_path}.scan[{index}]"
            scan = _strict_object(
                scan_item,
                path=scan_path,
                allowed={"name", "from", "axis"},
                required={"name", "from", "axis"},
            )
            scan_name = _string(scan["name"], f"{scan_path}.name")
            source = _string(scan["from"], f"{scan_path}.from")
            axis = _integer(scan["axis"], f"{scan_path}.axis", minimum=0)
            if source not in visible:
                raise SchemaError(f"{scan_path}.from references non-visible value {source!r}")
            if scan_name in visible:
                raise SchemaError(f"{scan_path}.name shadows outer value {scan_name!r}")
            if source in tensor_contracts:
                _dtype, source_shape = tensor_contracts[source]
                if axis >= len(source_shape):
                    raise SchemaError(f"{scan_path}.axis is outside source tensor rank")
                if source_shape[axis] != trip_axis:
                    raise SchemaError(
                        f"{scan_path}.axis does not select semantic trip dimension "
                        f"{trip_axis!r}"
                    )
            scan_names.append(scan_name)
        if len(scan_names) != len(set(scan_names)):
            raise SchemaError(f"{op_path}.scan names must be unique")

        carries = op["carry"]
        if not isinstance(carries, list) or not carries:
            raise SchemaError(f"{op_path}.carry must be a non-empty array")
        states: list[dict[str, Any]] = []
        for index, carry_item in enumerate(carries):
            carry_path = f"{op_path}.carry[{index}]"
            carry = _strict_object(
                carry_item,
                path=carry_path,
                allowed={
                    "state_id", "init_value", "body_argument", "next_value",
                    "result_value", "type",
                },
                required={
                    "state_id", "init_value", "body_argument", "next_value",
                    "result_value", "type",
                },
            )
            state_id = _string(carry["state_id"], f"{carry_path}.state_id")
            body_argument = _string(carry["body_argument"], f"{carry_path}.body_argument")
            next_value = _string(carry["next_value"], f"{carry_path}.next_value")
            result_value = _string(carry["result_value"], f"{carry_path}.result_value")
            init_value = _json_value(carry["init_value"], f"{carry_path}.init_value")
            dtype, shape = validate_loop_type(carry["type"], f"{carry_path}.type")
            if not is_literal_reference(init_value):
                if not isinstance(init_value, str):
                    raise SchemaError(
                        f"{carry_path}.init_value must be a scalar literal "
                        "or a visible value reference"
                    )
                if init_value not in visible:
                    raise SchemaError(
                        f"{carry_path}.init_value references non-visible "
                        f"value {init_value!r}"
                    )
            if isinstance(init_value, str) and init_value in tensor_contracts:
                init_dtype, init_shape = tensor_contracts[init_value]
                if _dtype_key(dtype) != _dtype_key(init_dtype) or shape != init_shape:
                    raise SchemaError(
                        f"{carry_path}.type must match referenced init_value type"
                    )
            if body_argument in visible or body_argument in scan_names:
                raise SchemaError(
                    f"{carry_path}.body_argument must be local to loop scope {loop_scope!r}"
                )
            if result_value == body_argument:
                raise SchemaError(
                    f"{carry_path}.result_value must be distinct from body_argument"
                )
            states.append(
                {
                    "state_id": state_id,
                    "body_argument": body_argument,
                    "next_value": next_value,
                    "result_value": result_value,
                    "dtype": dtype,
                    "shape": shape,
                }
            )

        for key in ("state_id", "body_argument", "next_value", "result_value"):
            values = [state[key] for state in states]
            if len(values) != len(set(values)):
                raise SchemaError(f"{op_path}.carry {key} values must be unique")

        expected_results = [state["result_value"] for state in states]
        expected_yields = [state["next_value"] for state in states]
        actual_yields = _string_list(op["yield"], f"{op_path}.yield", unique=True)
        if produced != expected_results:
            raise SchemaError(
                f"{op_path}.result must exactly match carry.result_value order"
            )
        if actual_yields != expected_yields:
            raise SchemaError(
                f"{op_path}.yield must exactly match carry.next_value order"
            )

        body_visible = set(visible) | set(scan_names) | {
            state["body_argument"] for state in states
        }
        _body_visible_after, body_definitions = validate_sequence(
            op["body"], f"{op_path}.body", body_visible,
            strict_operands=True,
        )
        missing_next = sorted(set(expected_yields) - body_definitions)
        if missing_next:
            raise SchemaError(
                f"{op_path}.carry next_value must be defined in the loop body: "
                + ", ".join(missing_next)
            )

        def produced_type(
            operations: list[dict[str, Any]], value_name: str,
        ) -> tuple[str, list[int | str]] | None:
            for child_index, child in enumerate(operations):
                child_results = child.get("result")
                names = (
                    [child_results]
                    if isinstance(child_results, str)
                    else list(child_results or [])
                )
                if value_name in names:
                    if child.get("op") == "loop":
                        result_index = names.index(value_name)
                        carry_items = child.get("carry", [])
                        if result_index < len(carry_items):
                            return validate_loop_type(
                                carry_items[result_index].get("type"),
                                f"{op_path}.body[{child_index}].carry"
                                f"[{result_index}].type",
                            )
                    if "type" not in child or child["type"] is None:
                        return None
                    return validate_loop_type(
                        child["type"], f"{op_path}.body[{child_index}].type"
                    )
                nested = child.get("body", [])
                if isinstance(nested, list):
                    found = produced_type(nested, value_name)
                    if found is not None:
                        return found
            return None

        for state in states:
            next_type = produced_type(op["body"], state["next_value"])
            if next_type is None:
                raise SchemaError(
                    f"{op_path}.carry next_value {state['next_value']!r} "
                    "requires an explicit producer type"
                )
            if next_type != (state["dtype"], state["shape"]):
                raise SchemaError(
                    f"{op_path}.carry state type must exactly match next/yield type"
                )
        return produced

    body = fn.get("body", [])
    visible_after, _definitions = validate_sequence(
        body,
        f"{path}.intent_function.body",
        {
            _string(tensor["name"], f"{path}.intent_function.inputs[{index}].name")
            for index, tensor in enumerate(fn.get("inputs", []))
        },
        strict_operands=True,
    )
    if "returns" in fn:
        returns = fn["returns"]
        if not isinstance(returns, list):
            raise SchemaError(f"{path}.intent_function.returns must be an array")
        output_names = [
            _string(tensor["name"], f"{path}.intent_function.outputs[{index}].name")
            for index, tensor in enumerate(fn.get("outputs", []))
        ]
        returned_outputs: list[str] = []
        for index, item in enumerate(returns):
            return_path = f"{path}.intent_function.returns[{index}]"
            binding = _strict_object(
                item,
                path=return_path,
                allowed={"output", "value"},
                required={"output", "value"},
            )
            output = _string(binding["output"], f"{return_path}.output")
            value = _string(binding["value"], f"{return_path}.value")
            returned_outputs.append(output)
            if value not in visible_after:
                raise SchemaError(
                    f"{return_path}.value references unknown Layer-A value {value!r}"
                )
        if returned_outputs != output_names:
            raise SchemaError(
                "IntentFunction.returns must cover outputs once and in ABI order"
            )

    normalized = {"intent_function": copy.deepcopy(fn)}
    return IntentFunction.from_json(normalized)


class Frontend(str, Enum):
    TRITON = "triton"
    CUTE = "cute"
    TILELANG = "tilelang"
    CUDA = "cuda"
    OTHER = "other"


class ParameterKind(str, Enum):
    TENSOR = "tensor"
    POINTER = "pointer"
    SCALAR = "scalar"
    OPAQUE = "opaque"
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
    UPDATE = "update"


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    UPDATE = "update"


class UpdateCombiner(str, Enum):
    OVERWRITE = "overwrite"
    ADD = "add"


class DuplicatePolicy(str, Enum):
    FORBID = "forbid"
    COMBINE = "combine"
    ONE_OF = "one_of"


class UpdateOrder(str, Enum):
    INPUT_ORDER = "input_order"
    UNORDERED = "unordered"


class ResultRelationKind(str, Enum):
    EXACT = "exact"
    TOLERANCE = "tolerance"
    ONE_OF_UPDATES = "one_of_updates"


class AliasKind(str, Enum):
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"
    NO_ALIAS = "no_alias"


class ScopeKind(str, Enum):
    FUNCTION = "function"
    PROGRAM_INSTANCE = "program_instance"
    LOOP_REGION = "loop_region"
    BRANCH_REGION = "branch_region"


class StorageMutability(str, Enum):
    IMMUTABLE = "immutable"
    WRITE_ONCE = "write_once"
    MUTABLE = "mutable"


class SemanticEventKind(str, Enum):
    SCOPE_ENTRY = "scope_entry"
    SCOPE_EXIT = "scope_exit"
    NODE = "node"
    ACCESS = "access"
    STATE = "state"
    EFFECT = "effect"


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
        if self.kind in {AccessKind.LOAD, AccessKind.STORE, AccessKind.UPDATE}:
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
    effect_id: str | None = None
    access_ref: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(EffectKind, self.kind, "EffectSpec.kind")
        self.target = _string(self.target, "EffectSpec.target")
        if self.node_ref is not None:
            self.node_ref = _string(self.node_ref, "EffectSpec.node_ref")
        if self.effect_id is not None:
            self.effect_id = _string(self.effect_id, "EffectSpec.effect_id")
        if self.access_ref is not None:
            self.access_ref = _string(self.access_ref, "EffectSpec.access_ref")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "EffectSpec":
        data = _strict_object(
            value,
            path="EffectSpec",
            allowed={
                "effect_id", "kind", "target", "node_ref", "access_ref",
            },
            required={"kind", "target"},
        )
        return cls(
            data["kind"], data["target"], data.get("node_ref"),
            data.get("effect_id"), data.get("access_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result = {"kind": self.kind.value, "target": self.target}
        if self.effect_id is not None:
            result["effect_id"] = self.effect_id
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.access_ref is not None:
            result["access_ref"] = self.access_ref
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
class IndexBoundsSpec(StrictJSONMixin):
    """Typed precondition for one data-dependent index operation.

    Its fixed meaning is ``0 <= indices < dim(data, axis)`` for every active
    index.  There is deliberately no boolean ``proven`` flag or proof text.
    """

    node_ref: str
    data: str
    indices: str
    axis: int

    def __post_init__(self) -> None:
        self.node_ref = _string(self.node_ref, "IndexBoundsSpec.node_ref")
        self.data = _string(self.data, "IndexBoundsSpec.data")
        self.indices = _string(self.indices, "IndexBoundsSpec.indices")
        self.axis = _integer(self.axis, "IndexBoundsSpec.axis", minimum=0)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "IndexBoundsSpec":
        data = _strict_object(
            value,
            path="IndexBoundsSpec",
            allowed={"node_ref", "data", "indices", "axis"},
            required={"node_ref", "data", "indices", "axis"},
        )
        return cls(data["node_ref"], data["data"], data["indices"], data["axis"])

    def to_json(self) -> dict[str, Any]:
        return {
            "node_ref": self.node_ref,
            "data": self.data,
            "indices": self.indices,
            "axis": self.axis,
        }


@dataclass
class LogicalUpdateSpec(StrictJSONMixin):
    """Source-neutral logical update; atomic realization belongs to pi/backend."""

    node_ref: str
    base: str
    indices: str
    values: str
    axis: int
    combiner: UpdateCombiner
    duplicate_policy: DuplicatePolicy
    order: UpdateOrder

    def __post_init__(self) -> None:
        self.node_ref = _string(self.node_ref, "LogicalUpdateSpec.node_ref")
        self.base = _string(self.base, "LogicalUpdateSpec.base")
        self.indices = _string(self.indices, "LogicalUpdateSpec.indices")
        self.values = _string(self.values, "LogicalUpdateSpec.values")
        self.axis = _integer(self.axis, "LogicalUpdateSpec.axis", minimum=0)
        self.combiner = _enum(
            UpdateCombiner, self.combiner, "LogicalUpdateSpec.combiner"
        )
        self.duplicate_policy = _enum(
            DuplicatePolicy,
            self.duplicate_policy,
            "LogicalUpdateSpec.duplicate_policy",
        )
        self.order = _enum(UpdateOrder, self.order, "LogicalUpdateSpec.order")
        if (
            self.combiner is UpdateCombiner.OVERWRITE
            and self.duplicate_policy is DuplicatePolicy.COMBINE
        ):
            raise SchemaError(
                "overwrite update cannot use duplicate_policy='combine'"
            )
        if (
            self.combiner is UpdateCombiner.ADD
            and self.duplicate_policy is DuplicatePolicy.ONE_OF
        ):
            raise SchemaError("add update cannot use duplicate_policy='one_of'")
        if (
            self.duplicate_policy is DuplicatePolicy.ONE_OF
            and self.order is not UpdateOrder.UNORDERED
        ):
            raise SchemaError("one_of update requires order='unordered'")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LogicalUpdateSpec":
        data = _strict_object(
            value,
            path="LogicalUpdateSpec",
            allowed={
                "node_ref", "base", "indices", "values", "axis",
                "combiner", "duplicate_policy", "order",
            },
            required={
                "node_ref", "base", "indices", "values", "axis",
                "combiner", "duplicate_policy", "order",
            },
        )
        return cls(
            node_ref=data["node_ref"],
            base=data["base"],
            indices=data["indices"],
            values=data["values"],
            axis=data["axis"],
            combiner=data["combiner"],
            duplicate_policy=data["duplicate_policy"],
            order=data["order"],
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_ref": self.node_ref,
            "base": self.base,
            "indices": self.indices,
            "values": self.values,
            "axis": self.axis,
            "combiner": self.combiner.value,
            "duplicate_policy": self.duplicate_policy.value,
            "order": self.order.value,
        }
        return result


@dataclass
class ResultRelationSpec(StrictJSONMixin):
    """Allowed-result relation for one declared logical output."""

    output: str
    kind: ResultRelationKind
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool | None = None
    update_node_ref: str | None = None

    def __post_init__(self) -> None:
        self.output = _string(self.output, "ResultRelationSpec.output")
        self.kind = _enum(
            ResultRelationKind, self.kind, "ResultRelationSpec.kind"
        )
        if self.kind is ResultRelationKind.TOLERANCE:
            if self.rtol is None or self.atol is None or self.equal_nan is None:
                raise SchemaError(
                    "tolerance result relation requires rtol, atol, and equal_nan"
                )
            self.rtol = _number(
                self.rtol, "ResultRelationSpec.rtol", minimum=0.0
            )
            self.atol = _number(
                self.atol, "ResultRelationSpec.atol", minimum=0.0
            )
            self.equal_nan = _boolean(
                self.equal_nan, "ResultRelationSpec.equal_nan"
            )
            if self.update_node_ref is not None:
                raise SchemaError(
                    "tolerance result relation cannot reference an update"
                )
        elif self.kind is ResultRelationKind.ONE_OF_UPDATES:
            if self.update_node_ref is None:
                raise SchemaError(
                    "one_of_updates result relation requires update_node_ref"
                )
            self.update_node_ref = _string(
                self.update_node_ref, "ResultRelationSpec.update_node_ref"
            )
            if any(
                value is not None
                for value in (self.rtol, self.atol, self.equal_nan)
            ):
                raise SchemaError(
                    "one_of_updates result relation cannot carry tolerance fields"
                )
        else:
            if self.update_node_ref is not None or any(
                value is not None
                for value in (self.rtol, self.atol, self.equal_nan)
            ):
                raise SchemaError(
                    "exact result relation cannot carry tolerance or update fields"
                )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ResultRelationSpec":
        data = _strict_object(
            value,
            path="ResultRelationSpec",
            allowed={
                "output", "kind", "rtol", "atol", "equal_nan",
                "update_node_ref",
            },
            required={"output", "kind"},
        )
        return cls(
            output=data["output"],
            kind=data["kind"],
            rtol=data.get("rtol"),
            atol=data.get("atol"),
            equal_nan=data.get("equal_nan"),
            update_node_ref=data.get("update_node_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "output": self.output,
            "kind": self.kind.value,
        }
        if self.kind is ResultRelationKind.TOLERANCE:
            result.update(
                rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
            )
        elif self.kind is ResultRelationKind.ONE_OF_UPDATES:
            result["update_node_ref"] = self.update_node_ref
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
    index_bounds: list[IndexBoundsSpec] = field(default_factory=list)
    updates: list[LogicalUpdateSpec] = field(default_factory=list)
    result_relations: list[ResultRelationSpec] = field(default_factory=list)

    _KEYS: ClassVar[set[str]] = {
        "domains", "accesses", "broadcasts", "aliases", "effects", "output_mappings",
        "reductions", "index_bounds", "updates", "result_relations",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.domains, dict):
            raise SchemaError("Boundary.domains must be an object")
        self.domains = {
            _string(key, "Boundary.domains key"): _string(
                value, f"Boundary.domains.{key}"
            )
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
        used_effect_ids = {
            item.effect_id for item in self.effects if item.effect_id is not None
        }
        for index, item in enumerate(self.effects):
            if item.effect_id is not None:
                continue
            candidate = f"e{index}"
            suffix = 0
            while candidate in used_effect_ids:
                suffix += 1
                candidate = f"e{index}_{suffix}"
            item.effect_id = candidate
            used_effect_ids.add(candidate)
        self.output_mappings = [
            item if isinstance(item, OutputMapping) else OutputMapping.from_json(item)
            for item in self.output_mappings
        ]
        self.reductions = [
            item if isinstance(item, ReductionBoundary) else ReductionBoundary.from_json(item)
            for item in self.reductions
        ]
        self.index_bounds = [
            item if isinstance(item, IndexBoundsSpec) else IndexBoundsSpec.from_json(item)
            for item in self.index_bounds
        ]
        self.updates = [
            item if isinstance(item, LogicalUpdateSpec) else LogicalUpdateSpec.from_json(item)
            for item in self.updates
        ]
        self.result_relations = [
            item if isinstance(item, ResultRelationSpec) else ResultRelationSpec.from_json(item)
            for item in self.result_relations
        ]
        _unique(self.accesses, "access_id", "Boundary.accesses")
        _unique(self.effects, "effect_id", "Boundary.effects")
        access_ids = {item.access_id for item in self.accesses}
        unknown_effect_accesses = sorted(
            item.access_ref
            for item in self.effects
            if item.access_ref is not None and item.access_ref not in access_ids
        )
        if unknown_effect_accesses:
            raise SchemaError(
                "Boundary.effects access_ref references unknown access(es): "
                + ", ".join(unknown_effect_accesses)
            )
        effect_access_refs = [
            item.access_ref for item in self.effects if item.access_ref is not None
        ]
        if len(effect_access_refs) != len(set(effect_access_refs)):
            raise SchemaError("Boundary.effects access_ref values must be unique")
        _unique(self.index_bounds, "node_ref", "Boundary.index_bounds")
        _unique(self.updates, "node_ref", "Boundary.updates")
        _unique(self.result_relations, "output", "Boundary.result_relations")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Boundary":
        data = _strict_object(
            value, path="Boundary", allowed=cls._KEYS, required=cls._KEYS
        )
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
            index_bounds=[IndexBoundsSpec.from_json(item) for item in data.get("index_bounds", [])],
            updates=[LogicalUpdateSpec.from_json(item) for item in data.get("updates", [])],
            result_relations=[ResultRelationSpec.from_json(item) for item in data.get("result_relations", [])],
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
            "index_bounds": [item.to_json() for item in self.index_bounds],
            "updates": [item.to_json() for item in self.updates],
            "result_relations": [item.to_json() for item in self.result_relations],
        }


@dataclass
class PortableScope(StrictJSONMixin):
    """Frontend-neutral dominance/capture/effect scope in semantic J.

    The record intentionally contains no Python/Triton AST spelling, source
    line, helper name, or frontend API.  Its only job is to make visibility and
    lifetime boundaries explicit enough for deterministic composition.
    """

    scope_id: str
    kind: ScopeKind
    parent_scope: str | None = None
    captures: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scope_id = _string(self.scope_id, "PortableScope.scope_id")
        self.kind = _enum(ScopeKind, self.kind, "PortableScope.kind")
        if self.parent_scope is not None:
            self.parent_scope = _string(
                self.parent_scope, "PortableScope.parent_scope"
            )
        self.captures = _string_list(
            self.captures, "PortableScope.captures", unique=True
        )
        self.definitions = _string_list(
            self.definitions, "PortableScope.definitions", unique=True
        )
        self.effects = _string_list(
            self.effects, "PortableScope.effects", unique=True
        )
        overlap = sorted(set(self.captures) & set(self.definitions))
        if overlap:
            raise SchemaError(
                "PortableScope captures and definitions overlap: "
                + ", ".join(overlap)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "PortableScope":
        data = _strict_object(
            value,
            path="PortableScope",
            allowed={
                "scope_id", "kind", "parent_scope", "captures",
                "definitions", "effects",
            },
            required={
                "scope_id", "kind", "parent_scope", "captures",
                "definitions", "effects",
            },
        )
        return cls(
            scope_id=data["scope_id"],
            kind=data["kind"],
            parent_scope=data["parent_scope"],
            captures=_string_list(data["captures"], "PortableScope.captures", unique=True),
            definitions=_string_list(
                data["definitions"], "PortableScope.definitions", unique=True
            ),
            effects=_string_list(data["effects"], "PortableScope.effects", unique=True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "kind": self.kind.value,
            "parent_scope": self.parent_scope,
            "captures": list(self.captures),
            "definitions": list(self.definitions),
            "effects": list(self.effects),
        }


@dataclass
class SemanticEventRef(StrictJSONMixin):
    """Namespaced semantic event used by storage lifetime endpoints."""

    kind: SemanticEventKind
    ref: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(SemanticEventKind, self.kind, "SemanticEventRef.kind")
        if self.ref is None:
            raise SchemaError(
                f"SemanticEventRef kind {self.kind.value!r} requires ref"
            )
        self.ref = _string(self.ref, "SemanticEventRef.ref")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SemanticEventRef":
        data = _strict_object(
            value,
            path="SemanticEventRef",
            allowed={"kind", "ref"},
            required={"kind", "ref"},
        )
        return cls(kind=data["kind"], ref=data.get("ref"))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "ref": self.ref}


@dataclass
class StorageObject(StrictJSONMixin):
    """One semantic storage identity and all J values that denote it."""

    storage_id: str
    members: list[str]
    scope_id: str
    mutability: StorageMutability
    init_event: SemanticEventRef
    last_use_event: SemanticEventRef
    escapes: bool
    crosses_iteration: bool

    def __post_init__(self) -> None:
        self.storage_id = _string(self.storage_id, "StorageObject.storage_id")
        self.members = _string_list(
            self.members, "StorageObject.members", unique=True
        )
        if not self.members:
            raise SchemaError("StorageObject.members must not be empty")
        self.scope_id = _string(self.scope_id, "StorageObject.scope_id")
        self.mutability = _enum(
            StorageMutability, self.mutability, "StorageObject.mutability"
        )
        if not isinstance(self.init_event, SemanticEventRef):
            self.init_event = SemanticEventRef.from_json(self.init_event)
        if not isinstance(self.last_use_event, SemanticEventRef):
            self.last_use_event = SemanticEventRef.from_json(self.last_use_event)
        self.escapes = _boolean(self.escapes, "StorageObject.escapes")
        self.crosses_iteration = _boolean(
            self.crosses_iteration, "StorageObject.crosses_iteration"
        )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "StorageObject":
        data = _strict_object(
            value,
            path="StorageObject",
            allowed={
                "storage_id", "members", "scope_id", "mutability",
                "init_event", "last_use_event", "escapes",
                "crosses_iteration",
            },
            required={
                "storage_id", "members", "scope_id", "mutability",
                "init_event", "last_use_event", "escapes",
                "crosses_iteration",
            },
        )
        return cls(
            storage_id=data["storage_id"],
            members=_string_list(data["members"], "StorageObject.members", unique=True),
            scope_id=data["scope_id"],
            mutability=data["mutability"],
            init_event=SemanticEventRef.from_json(data["init_event"]),
            last_use_event=SemanticEventRef.from_json(data["last_use_event"]),
            escapes=data["escapes"],
            crosses_iteration=data["crosses_iteration"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "members": list(self.members),
            "scope_id": self.scope_id,
            "mutability": self.mutability.value,
            "init_event": self.init_event.to_json(),
            "last_use_event": self.last_use_event.to_json(),
            "escapes": self.escapes,
            "crosses_iteration": self.crosses_iteration,
        }


def _validate_typed_boundary_relations(archive: "SemanticArchive") -> None:
    """Bind every typed bounds/update/result relation to this exact J graph."""

    by_path: dict[str, Any] = {}

    def visit(ops: list[Any], prefix: str = "body") -> None:
        for index, op in enumerate(ops):
            path = f"{prefix}[{index}]"
            by_path[path] = op
            visit(op.body, f"{path}.body")

    visit(archive.intent.body)
    nodes = dict(by_path)
    for node_id, path in archive.node_ids.items():
        if path in by_path:
            nodes[node_id] = by_path[path]

    tensors = {
        item.name: item
        for item in [*archive.intent.inputs, *archive.intent.outputs]
    }

    bounds_by_node: dict[str, IndexBoundsSpec] = {}
    bounded_operations: set[int] = set()
    for item in archive.boundary.index_bounds:
        op = nodes.get(item.node_ref)
        if op is None:
            raise SchemaError(
                f"IndexBoundsSpec references unknown node {item.node_ref!r}"
            )
        if op.op not in {"gather", "scatter"}:
            raise SchemaError(
                "IndexBoundsSpec must target a gather or scatter node"
            )
        if len(op.operands) < 2:
            raise SchemaError("indexed node does not have data and indices operands")
        if item.data != op.operands[0] or item.indices != op.operands[1]:
            raise SchemaError(
                "IndexBoundsSpec data/indices do not match the target node"
            )
        op_axis = op.attrs.get("axis")
        if isinstance(op_axis, bool) or not isinstance(op_axis, int):
            raise SchemaError("indexed node axis must be a canonical integer")
        if item.axis != op_axis:
            raise SchemaError("IndexBoundsSpec axis does not match the target node")
        tensor = tensors.get(item.data)
        if tensor is not None and item.axis >= len(tensor.shape):
            raise SchemaError("IndexBoundsSpec axis is outside the data rank")
        if id(op) in bounded_operations:
            raise SchemaError(
                "Boundary.index_bounds contains multiple aliases for one node"
            )
        bounded_operations.add(id(op))
        bounds_by_node[item.node_ref] = item

    gather_operations = {
        id(op) for op in by_path.values() if op.op == "gather"
    }
    if gather_operations - bounded_operations:
        raise SchemaError(
            "canonical J gather requires a typed IndexBoundsSpec"
        )

    updates_by_node: dict[str, LogicalUpdateSpec] = {}
    updated_operations: set[int] = set()
    scatter_refs: set[str] = set()
    for node_ref, op in nodes.items():
        if op.op == "scatter":
            scatter_refs.add(node_ref)
    # A node can be named by both path and opaque id; canonicalize through the
    # relation refs rather than treating those aliases as duplicate nodes.
    scatter_ops = {id(nodes[ref]): nodes[ref] for ref in scatter_refs}

    for item in archive.boundary.updates:
        op = nodes.get(item.node_ref)
        if op is None:
            raise SchemaError(
                f"LogicalUpdateSpec references unknown node {item.node_ref!r}"
            )
        if op.op != "scatter":
            raise SchemaError("LogicalUpdateSpec must target a scatter node")
        if op.attrs.get("mode") != "along_axis":
            raise SchemaError(
                "LogicalUpdateSpec currently requires scatter mode='along_axis'"
            )
        if len(op.operands) != 3:
            raise SchemaError("scatter update requires base, indices, and values")
        if [item.base, item.indices, item.values] != list(op.operands):
            raise SchemaError(
                "LogicalUpdateSpec operands do not match the target scatter"
            )
        op_axis = op.attrs.get("axis")
        if isinstance(op_axis, bool) or not isinstance(op_axis, int):
            raise SchemaError("scatter axis must be a canonical integer")
        if item.axis != op_axis:
            raise SchemaError("LogicalUpdateSpec axis does not match scatter")
        expected_combiner = {
            "none": UpdateCombiner.OVERWRITE,
            "add": UpdateCombiner.ADD,
        }.get(op.attrs.get("reduce"))
        if expected_combiner is None or item.combiner is not expected_combiner:
            raise SchemaError(
                "LogicalUpdateSpec combiner does not match scatter reduce"
            )
        if id(op) in updated_operations:
            raise SchemaError(
                "Boundary.updates contains multiple aliases for one node"
            )
        updated_operations.add(id(op))
        bound = bounds_by_node.get(item.node_ref)
        if bound is None:
            # Accept a path/id alias for the same operation, but never infer a
            # bound from an unrelated indexed node.
            matching = [
                candidate
                for ref, candidate in bounds_by_node.items()
                if nodes.get(ref) is op
            ]
            if len(matching) != 1:
                raise SchemaError(
                    "LogicalUpdateSpec requires one matching IndexBoundsSpec"
                )
        updates_by_node[item.node_ref] = item

    covered_scatter_ops = {
        id(nodes[item.node_ref]) for item in archive.boundary.updates
    }
    missing_scatter = [
        op for key, op in scatter_ops.items() if key not in covered_scatter_ops
    ]
    if missing_scatter:
        raise SchemaError(
            "canonical J scatter requires a typed LogicalUpdateSpec"
        )

    outputs = {item.name for item in archive.intent.outputs}
    return_values = {item.output: item.value for item in archive.intent.returns}
    if len(return_values) != len(archive.intent.returns):
        raise SchemaError("IntentFunction returns contain duplicate outputs")
    unknown_return_outputs = sorted(set(return_values) - outputs)
    if unknown_return_outputs:
        raise SchemaError(
            "IntentFunction returns reference unknown outputs: "
            + ", ".join(unknown_return_outputs)
        )
    observable_values = {
        output: return_values.get(output, output) for output in outputs
    }
    relation_by_output = {
        item.output: item for item in archive.boundary.result_relations
    }
    for item in archive.boundary.result_relations:
        if item.output not in outputs:
            raise SchemaError(
                f"ResultRelationSpec references unknown output {item.output!r}"
            )
        if item.kind is not ResultRelationKind.ONE_OF_UPDATES:
            continue
        update = updates_by_node.get(item.update_node_ref or "")
        update_op = nodes.get(item.update_node_ref or "")
        if update is None:
            # Resolve the alternate path/id spelling for the same update node.
            referenced = nodes.get(item.update_node_ref or "")
            matches = [
                candidate
                for ref, candidate in updates_by_node.items()
                if nodes.get(ref) is referenced
            ]
            if len(matches) == 1:
                update = matches[0]
                update_op = referenced
        if update is None or update_op is None:
            raise SchemaError(
                "one_of_updates relation references an unknown logical update"
            )
        if not (
            update.combiner is UpdateCombiner.OVERWRITE
            and update.duplicate_policy is DuplicatePolicy.ONE_OF
        ):
            raise SchemaError(
                "one_of_updates relation requires overwrite + one_of update"
            )
        if not isinstance(update_op.result, str) or (
            observable_values[item.output] != update_op.result
        ):
            raise SchemaError(
                "one_of_updates currently requires the update result to be "
                "returned directly"
            )

    if archive.boundary.updates:
        missing_relations = sorted(outputs - set(relation_by_output))
        if missing_relations:
            raise SchemaError(
                "canonical J with logical updates requires an explicit result "
                "relation for every output: " + ", ".join(missing_relations)
            )
        for item in archive.boundary.updates:
            op = nodes[item.node_ref]
            direct_outputs = {
                output
                for output, value in observable_values.items()
                if value == op.result
            }
            if item.duplicate_policy is DuplicatePolicy.ONE_OF:
                matching = [
                    relation_by_output[output]
                    for output in direct_outputs
                    if relation_by_output[output].kind
                    is ResultRelationKind.ONE_OF_UPDATES
                    and nodes.get(relation_by_output[output].update_node_ref or "")
                    is op
                ]
                if len(matching) != 1:
                    raise SchemaError(
                        "one_of logical update requires exactly one matching "
                        "one_of_updates output relation"
                    )
            elif any(
                relation.kind is ResultRelationKind.ONE_OF_UPDATES
                and nodes.get(relation.update_node_ref or "") is op
                for relation in archive.boundary.result_relations
            ):
                raise SchemaError(
                    "deterministic/combine update cannot use one_of_updates relation"
                )


def _validate_semantic_scopes_and_storage(archive: "SemanticArchive") -> None:
    """Cross-check explicit portable scopes and semantic storage against J."""

    loop_ops = [
        op
        for top in archive.intent.body
        for op in _walk_intent_ops(top)
        if op.op == "loop"
    ]
    if not archive.scopes:
        if loop_ops:
            raise SchemaError(
                "SemanticArchive.scopes is required when J contains a semantic loop"
            )
        if archive.storage:
            raise SchemaError(
                "SemanticArchive.storage requires an explicit portable scope tree"
            )
        # Migration path for legacy straight-line artifacts.  New Lift wire
        # schema always emits scopes/storage, but old cached artifacts remain
        # readable until the protocol switch in P5.
        return

    scope_by_id = {scope.scope_id: scope for scope in archive.scopes}
    _unique(archive.scopes, "scope_id", "SemanticArchive.scopes")
    roots = [
        scope
        for scope in archive.scopes
        if scope.kind is ScopeKind.FUNCTION and scope.parent_scope is None
    ]
    if len(roots) != 1:
        raise SchemaError(
            "SemanticArchive.scopes must contain exactly one parentless function scope"
        )
    root = roots[0]
    for scope in archive.scopes:
        if scope is root:
            continue
        if scope.parent_scope is None:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} requires parent_scope"
            )
        if scope.parent_scope not in scope_by_id:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} references unknown parent_scope"
            )

    for scope in archive.scopes:
        seen: set[str] = set()
        current: PortableScope | None = scope
        while current is not None:
            if current.scope_id in seen:
                raise SchemaError("SemanticArchive.scopes contains a parent cycle")
            seen.add(current.scope_id)
            current = (
                scope_by_id[current.parent_scope]
                if current.parent_scope is not None
                else None
            )
        if root.scope_id not in seen:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} is disconnected from function root"
            )

    all_program_scopes = [
        scope
        for scope in archive.scopes
        if scope.kind is ScopeKind.PROGRAM_INSTANCE
    ]
    nested_program_scopes = [
        scope
        for scope in all_program_scopes
        if scope.parent_scope != root.scope_id
    ]
    if nested_program_scopes:
        raise SchemaError(
            "program_instance scope must be the unique direct child of the "
            "function scope"
        )
    program_scopes = [
        scope
        for scope in all_program_scopes
        if scope.parent_scope == root.scope_id
    ]
    if len(program_scopes) > 1:
        raise SchemaError(
            "SemanticArchive supports at most one top-level program_instance scope"
        )
    execution_scope = program_scopes[0] if program_scopes else root
    branch_scopes = [
        scope for scope in archive.scopes
        if scope.kind is ScopeKind.BRANCH_REGION
    ]
    if branch_scopes:
        raise SchemaError(
            "branch_region scopes require a first-class IntentIR branch-region op; "
            "the current schema fails closed instead of attaching orphan scopes"
        )

    reverse_node_ids = {path: node_id for node_id, path in archive.node_ids.items()}
    required_definitions: dict[str, set[str]] = {
        scope.scope_id: set() for scope in archive.scopes
    }
    required_captures: dict[str, set[str]] = {
        scope.scope_id: set() for scope in archive.scopes
    }
    required_effects: dict[str, set[str]] = {
        scope.scope_id: set() for scope in archive.scopes
    }
    path_scope: dict[str, str] = {}
    event_order: dict[tuple[SemanticEventKind, str | None], int] = {
        (SemanticEventKind.SCOPE_ENTRY, root.scope_id): -1,
    }
    event_scope: dict[tuple[SemanticEventKind, str | None], str] = {
        (SemanticEventKind.SCOPE_ENTRY, root.scope_id): root.scope_id,
    }
    if execution_scope is not root:
        event_order[(SemanticEventKind.SCOPE_ENTRY, execution_scope.scope_id)] = -1
        event_scope[
            (SemanticEventKind.SCOPE_ENTRY, execution_scope.scope_id)
        ] = execution_scope.scope_id
    value_events: dict[str, set[tuple[SemanticEventKind, str | None]]] = {}
    valid_values = {tensor.name for tensor in archive.intent.inputs}
    valid_values.update(tensor.name for tensor in archive.intent.outputs)
    required_definitions[root.scope_id].update(
        tensor.name for tensor in archive.intent.inputs
    )
    state_records: list[tuple[Any, str, str]] = []
    referenced_loop_scopes: set[str] = set()
    order = 0

    def operation_results(op: Any) -> list[str]:
        if op.result is None:
            return []
        if isinstance(op.result, str):
            return [op.result]
        return list(op.result)

    def local_loop_uses(loop: Any) -> set[str]:
        local = {scan.name for scan in loop.scan}
        local.update(state.body_argument for state in loop.carry)
        used: set[str] = set()
        for body_op in loop.body:
            for nested in _walk_intent_ops(body_op):
                local.update(operation_results(nested))
                used.update(nested.operands)
        outer = {
            value
            for value in used - local
            if not _looks_like_semantic_literal(value)
        }
        outer.update(scan.source for scan in loop.scan)
        outer.update(
            state.init_value
            for state in loop.carry
            if isinstance(state.init_value, str)
            and not _looks_like_semantic_literal(state.init_value)
        )
        return outer

    def visit(ops: list[Any], owner_scope: str, prefix: str = "body") -> None:
        nonlocal order
        for index, op in enumerate(ops):
            path = f"{prefix}[{index}]"
            path_scope[path] = owner_scope
            path_event = (SemanticEventKind.NODE, path)
            event_order[path_event] = order
            event_scope[path_event] = owner_scope
            node_id = reverse_node_ids.get(path)
            if node_id is not None:
                node_event = (SemanticEventKind.NODE, node_id)
                event_order[node_event] = order
                event_scope[node_event] = owner_scope
            order += 1
            produced = operation_results(op)
            valid_values.update(produced)
            required_definitions[owner_scope].update(produced)
            event_ref = (SemanticEventKind.NODE, node_id or path)
            for value in [*produced, *op.operands]:
                value_events.setdefault(value, set()).add(event_ref)
            if op.op != "loop":
                if op.scope_id is not None:
                    raise SchemaError(
                        f"non-loop operation {path!r} must not declare scope_id"
                    )
                if op.body:
                    visit(op.body, owner_scope, f"{path}.body")
                continue

            loop_scope_id = op.scope_id
            if loop_scope_id is None or loop_scope_id not in scope_by_id:
                raise SchemaError(
                    f"semantic loop {path!r} references unknown portable scope"
                )
            loop_scope = scope_by_id[loop_scope_id]
            if loop_scope_id in referenced_loop_scopes:
                raise SchemaError(
                    f"portable loop scope {loop_scope_id!r} is referenced more than once"
                )
            referenced_loop_scopes.add(loop_scope_id)
            if loop_scope.kind is not ScopeKind.LOOP_REGION:
                raise SchemaError(
                    f"semantic loop {path!r} scope must have kind loop_region"
                )
            if loop_scope.parent_scope != owner_scope:
                raise SchemaError(
                    f"semantic loop {path!r} scope parent must be {owner_scope!r}"
                )
            loop_entry = (SemanticEventKind.SCOPE_ENTRY, loop_scope_id)
            event_order[loop_entry] = event_order[path_event]
            event_scope[loop_entry] = loop_scope_id
            loop_definitions = {scan.name for scan in op.scan}
            loop_definitions.update(state.body_argument for state in op.carry)
            valid_values.update(loop_definitions)
            required_definitions[loop_scope_id].update(loop_definitions)
            captures = local_loop_uses(op)
            required_captures[loop_scope_id].update(captures)
            for scan in op.scan:
                value_events.setdefault(scan.source, set()).add(event_ref)
            for state in op.carry:
                valid_values.add(state.state_id)
                state_records.append((state, loop_scope_id, node_id or path))
                state_event = (SemanticEventKind.STATE, state.state_id)
                event_order[state_event] = event_order[path_event]
                event_scope[state_event] = loop_scope_id
                for value in (
                    state.state_id, state.body_argument, state.next_value,
                    state.result_value,
                ):
                    value_events.setdefault(value, set()).add(state_event)
                if (
                    isinstance(state.init_value, str)
                    and not _looks_like_semantic_literal(state.init_value)
                ):
                    value_events.setdefault(state.init_value, set()).add(state_event)
            visit(op.body, loop_scope_id, f"{path}.body")
            loop_exit = (SemanticEventKind.SCOPE_EXIT, loop_scope_id)
            event_order[loop_exit] = order
            event_scope[loop_exit] = loop_scope_id

    visit(archive.intent.body, execution_scope.scope_id)
    declared_loop_scopes = {
        scope.scope_id
        for scope in archive.scopes
        if scope.kind is ScopeKind.LOOP_REGION
    }
    if referenced_loop_scopes != declared_loop_scopes:
        raise SchemaError(
            "portable loop scopes must be in one-to-one correspondence with J loops"
        )
    event_order[(SemanticEventKind.SCOPE_EXIT, execution_scope.scope_id)] = order + 1
    event_scope[
        (SemanticEventKind.SCOPE_EXIT, execution_scope.scope_id)
    ] = execution_scope.scope_id
    if execution_scope is not root:
        event_order[(SemanticEventKind.SCOPE_EXIT, root.scope_id)] = order + 2
        event_scope[(SemanticEventKind.SCOPE_EXIT, root.scope_id)] = root.scope_id
    if execution_scope is not root:
        required_captures[execution_scope.scope_id].update(
            tensor.name
            for tensor in archive.intent.inputs
            if value_events.get(tensor.name)
        )

    valid_effect_refs: set[str] = set()
    for effect in archive.boundary.effects:
        assert effect.effect_id is not None
        token = effect.effect_id
        valid_effect_refs.add(token)
        if effect.node_ref is None:
            owner = execution_scope.scope_id
            effect_order = order
        else:
            effect_path = archive.node_ids.get(effect.node_ref, effect.node_ref)
            owner = path_scope.get(effect_path, execution_scope.scope_id)
            effect_order = event_order.get(
                (SemanticEventKind.NODE, effect.node_ref),
                event_order.get((SemanticEventKind.NODE, effect_path), order),
            )
        required_effects[owner].add(token)
        effect_event = (SemanticEventKind.EFFECT, token)
        event_order[effect_event] = effect_order
        event_scope[effect_event] = owner
        value_events.setdefault(effect.target, set()).add(effect_event)

    access_effect_kind = {
        AccessKind.LOAD: EffectKind.READ,
        AccessKind.STORE: EffectKind.WRITE,
        AccessKind.UPDATE: EffectKind.UPDATE,
    }
    for access in archive.boundary.accesses:
        if access.node_ref is None:
            owner = execution_scope.scope_id
            access_order = order
        else:
            access_path = archive.node_ids.get(access.node_ref, access.node_ref)
            owner = path_scope.get(access_path, execution_scope.scope_id)
            access_order = event_order.get(
                (SemanticEventKind.NODE, access.node_ref),
                event_order.get((SemanticEventKind.NODE, access_path), order),
            )
        access_event = (SemanticEventKind.ACCESS, access.access_id)
        event_order[access_event] = access_order
        event_scope[access_event] = owner
        value_events.setdefault(access.tensor, set()).add(access_event)
        matches = [
            effect
            for effect in archive.boundary.effects
            if effect.kind is access_effect_kind[access.kind]
            and effect.target == access.tensor
            and effect.node_ref == access.node_ref
            and effect.access_ref == access.access_id
        ]
        if len(matches) != 1:
            raise SchemaError(
                f"boundary access {access.access_id!r} requires exactly one "
                "matching EffectSpec"
            )

    defined_owner: dict[str, str] = {}
    all_declared_values = set(valid_values)
    for scope in archive.scopes:
        expected_defs = required_definitions[scope.scope_id]
        declared_defs = set(scope.definitions)
        missing_defs = sorted(expected_defs - declared_defs)
        extra_defs = sorted(declared_defs - expected_defs)
        if missing_defs or extra_defs:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} definitions must exactly "
                f"match derived J values; missing={missing_defs}, extra={extra_defs}"
            )
        unknown_defs = sorted(declared_defs - all_declared_values)
        if unknown_defs:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown definition(s): "
                + ", ".join(unknown_defs)
            )
        for definition in scope.definitions:
            previous = defined_owner.get(definition)
            if previous is not None and previous != scope.scope_id:
                raise SchemaError(
                    f"value {definition!r} is defined in multiple portable scopes"
                )
            defined_owner[definition] = scope.scope_id
        expected_captures = required_captures[scope.scope_id]
        declared_captures = set(scope.captures)
        missing_captures = sorted(expected_captures - declared_captures)
        extra_captures = sorted(declared_captures - expected_captures)
        if missing_captures or extra_captures:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} captures must exactly "
                f"match derived outer uses; missing={missing_captures}, "
                f"extra={extra_captures}"
            )
        unknown_captures = sorted(declared_captures - all_declared_values)
        if unknown_captures:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown capture(s): "
                + ", ".join(unknown_captures)
            )
        expected_effects = required_effects[scope.scope_id]
        declared_effects = set(scope.effects)
        missing_effects = sorted(expected_effects - declared_effects)
        extra_effects = sorted(declared_effects - expected_effects)
        if missing_effects or extra_effects:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} effects must exactly "
                f"match Boundary effects; missing={missing_effects}, "
                f"extra={extra_effects}"
            )
        unknown_effects = sorted(declared_effects - valid_effect_refs)
        if unknown_effects:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown effect(s): "
                + ", ".join(unknown_effects)
            )

    for scope in archive.scopes:
        ancestor = scope.parent_scope
        ancestor_visible: set[str] = set()
        while ancestor is not None:
            parent = scope_by_id[ancestor]
            ancestor_visible.update(parent.definitions)
            ancestor_visible.update(parent.captures)
            ancestor = parent.parent_scope
        invisible = sorted(set(scope.captures) - ancestor_visible)
        if invisible:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} captures value(s) not visible "
                f"from an ancestor: {', '.join(invisible)}"
            )

    _unique(archive.storage, "storage_id", "SemanticArchive.storage")
    member_owner: dict[str, StorageObject] = {}
    for item in archive.storage:
        for member in item.members:
            previous = member_owner.get(member)
            if previous is not None and previous is not item:
                raise SchemaError(
                    f"semantic storage member {member!r} belongs to multiple "
                    "StorageObject identities"
                )
            member_owner[member] = item

    node_refs = set(archive.node_ids) | set(archive.node_ids.values())
    access_refs = {access.access_id for access in archive.boundary.accesses}
    state_refs = {state.state_id for state, _scope, _event in state_records}
    effect_refs = {
        effect.effect_id
        for effect in archive.boundary.effects
        if effect.effect_id is not None
    }
    event_domains = {
        SemanticEventKind.SCOPE_ENTRY: set(scope_by_id),
        SemanticEventKind.SCOPE_EXIT: set(scope_by_id),
        SemanticEventKind.NODE: node_refs,
        SemanticEventKind.ACCESS: access_refs,
        SemanticEventKind.STATE: state_refs,
        SemanticEventKind.EFFECT: effect_refs,
    }

    def event_key(event: SemanticEventRef, storage_id: str) -> tuple[SemanticEventKind, str | None]:
        if event.ref not in event_domains[event.kind]:
            raise SchemaError(
                f"storage {storage_id!r} {event.kind.value} event references "
                f"unknown ref {event.ref!r}"
            )
        key = (event.kind, event.ref)
        if key not in event_order:
            raise SchemaError(
                f"storage {storage_id!r} event has no deterministic order"
            )
        return key

    def is_descendant_or_same(scope_id: str, ancestor_id: str) -> bool:
        current: str | None = scope_id
        while current is not None:
            if current == ancestor_id:
                return True
            current = scope_by_id[current].parent_scope
        return False

    for item in archive.storage:
        if item.scope_id not in scope_by_id:
            raise SchemaError(
                f"storage {item.storage_id!r} references unknown scope_id"
            )
        unknown_members = sorted(set(item.members) - all_declared_values)
        if unknown_members:
            raise SchemaError(
                f"storage {item.storage_id!r} has unknown member(s): "
                + ", ".join(unknown_members)
            )
        init_key = event_key(item.init_event, item.storage_id)
        last_key = event_key(item.last_use_event, item.storage_id)
        init_order = event_order[init_key]
        last_order = event_order[last_key]
        if last_order < init_order:
            raise SchemaError(
                f"storage {item.storage_id!r} last_use_event precedes init_event"
            )
        scope = scope_by_id[item.scope_id]
        init_owner = event_scope.get(init_key)
        last_owner = event_scope.get(last_key)
        if init_owner is not None and not (
            is_descendant_or_same(init_owner, item.scope_id)
            or is_descendant_or_same(item.scope_id, init_owner)
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} init_event is outside its scope lineage"
            )
        if last_owner is not None and not is_descendant_or_same(
            last_owner, item.scope_id
        ):
            if not (
                item.escapes
                and is_descendant_or_same(item.scope_id, last_owner)
            ):
                raise SchemaError(
                    f"storage {item.storage_id!r} last_use_event is outside its "
                    "scope lineage"
                )
        if item.crosses_iteration and scope.kind is not ScopeKind.LOOP_REGION:
            raise SchemaError(
                f"storage {item.storage_id!r} crosses_iteration outside a loop scope"
            )

        observed = set().union(
            *(value_events.get(member, set()) for member in item.members)
        )
        if (
            init_key not in observed
            and item.init_event.kind is not SemanticEventKind.SCOPE_ENTRY
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} init_event is unrelated to its members"
            )
        if (
            last_key not in observed
            and item.last_use_event.kind is not SemanticEventKind.SCOPE_EXIT
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} last_use_event is unrelated to its members"
            )
        for observed_event in observed:
            observed_order = event_order[observed_event]
            if not (init_order <= observed_order <= last_order):
                raise SchemaError(
                    f"storage {item.storage_id!r} has use outside declared lifetime"
                )
            owner_scope = event_scope.get(observed_event)
            if (
                owner_scope is not None
                and not is_descendant_or_same(owner_scope, item.scope_id)
                and not item.escapes
            ):
                raise SchemaError(
                    f"storage {item.storage_id!r} escapes its scope without "
                    "escapes=true"
                )
        if observed:
            first_observed = min(event_order[key] for key in observed)
            last_observed = max(event_order[key] for key in observed)
            if init_order > first_observed:
                raise SchemaError(
                    f"storage {item.storage_id!r} is used before init_event"
                )
            if item.last_use_event.kind is not SemanticEventKind.SCOPE_EXIT:
                if last_order != last_observed:
                    raise SchemaError(
                        f"storage {item.storage_id!r} last_use_event is not its "
                        "last semantic use"
                    )
            elif not item.escapes:
                raise SchemaError(
                    f"storage {item.storage_id!r} uses scope_exit lifetime without "
                    "escapes=true"
                )

        write_events: set[tuple[str, str]] = set()
        for access in archive.boundary.accesses:
            if access.tensor not in item.members or access.kind not in {
                AccessKind.STORE, AccessKind.UPDATE,
            }:
                continue
            write_events.add(("access", access.access_id))
        for effect in archive.boundary.effects:
            if effect.target not in item.members or effect.kind not in {
                EffectKind.WRITE, EffectKind.READ_WRITE, EffectKind.UPDATE,
            }:
                continue
            assert effect.effect_id is not None
            if effect.access_ref is not None:
                write_events.add(("access", effect.access_ref))
            else:
                write_events.add(("effect", effect.effect_id))
        if item.mutability is StorageMutability.IMMUTABLE and write_events:
            raise SchemaError(
                f"immutable storage {item.storage_id!r} has a write event"
            )
        if item.mutability is StorageMutability.WRITE_ONCE and len(write_events) != 1:
            raise SchemaError(
                f"write_once storage {item.storage_id!r} requires exactly one "
                "semantic write event"
            )

    for state, scope_id, _loop_event in state_records:
        item = member_owner.get(state.state_id)
        if item is None:
            raise SchemaError(
                f"semantic loop state {state.state_id!r} requires a StorageObject"
            )
        required_members = {
            state.state_id, state.body_argument, state.next_value,
            state.result_value,
        }
        if (
            isinstance(state.init_value, str)
            and not _looks_like_semantic_literal(state.init_value)
        ):
            required_members.add(state.init_value)
        if not required_members.issubset(item.members):
            raise SchemaError(
                f"semantic loop state {state.state_id!r} storage must bind init, "
                "body argument, next value, and result value"
            )
        if item.scope_id != scope_id or not item.crosses_iteration or not item.escapes:
            raise SchemaError(
                f"semantic loop state {state.state_id!r} storage must live in its loop "
                "scope, cross iterations, and escape as the loop result"
            )

    for alias in archive.boundary.aliases:
        left = member_owner.get(alias.left)
        right = member_owner.get(alias.right)
        if left is None or right is None:
            raise SchemaError(
                "Boundary alias endpoints require explicit semantic storage"
            )
        same_storage = left is right
        if alias.kind is AliasKind.MUST_ALIAS and not same_storage:
            raise SchemaError(
                "must_alias endpoints must be members of the same StorageObject"
            )
        if alias.kind is AliasKind.NO_ALIAS and same_storage:
            raise SchemaError(
                "no_alias endpoints must use different StorageObject identities"
            )


def _walk_intent_ops(op: Any):
    yield op
    for child in op.body:
        yield from _walk_intent_ops(child)


def _looks_like_semantic_literal(value: str) -> bool:
    if value in {
        "NEG_INF", "POS_INF", "ZERO", "ONE", "zeros", "ones",
        "false", "true",
    }:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


@dataclass
class SemanticArchive(StrictJSONMixin):
    """The semantic archive ``J``: existing IntentFunction plus strict Layer B."""

    kernel_id: str
    intent: IntentFunction
    node_ids: dict[str, str]
    boundary: Boundary
    scopes: list[PortableScope] = field(default_factory=list)
    storage: list[StorageObject] = field(default_factory=list)

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
        canonical_path = re.compile(r"^body\[\d+\](?:\.body\[\d+\])*$")
        intent_json = self.intent.to_json()["intent_function"]
        for node_id, node_path in self.node_ids.items():
            if canonical_path.fullmatch(node_path) is None:
                raise SchemaError(
                    f"SemanticArchive.node_ids.{node_id} must be a canonical "
                    "body[i](.body[j])* JSON path"
                )
            operations = intent_json.get("body", [])
            indices = [int(value) for value in re.findall(r"body\[(\d+)\]", node_path)]
            for depth, index in enumerate(indices):
                if index >= len(operations):
                    raise SchemaError(
                        f"SemanticArchive.node_ids.{node_id} path is out of range"
                    )
                operation = operations[index]
                if depth + 1 < len(indices):
                    operations = operation.get("body", [])
        if not isinstance(self.boundary, Boundary):
            self.boundary = Boundary.from_json(self.boundary)
        self.scopes = [
            item if isinstance(item, PortableScope) else PortableScope.from_json(item)
            for item in self.scopes
        ]
        self.storage = [
            item if isinstance(item, StorageObject) else StorageObject.from_json(item)
            for item in self.storage
        ]
        _validate_typed_boundary_relations(self)
        _validate_semantic_scopes_and_storage(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SemanticArchive":
        data = _strict_object(
            value,
            path="SemanticArchive",
            allowed={
                "kernel_id", "intent", "node_ids", "boundary", "scopes",
                "storage",
            },
            required={"kernel_id", "intent", "node_ids", "boundary"},
        )
        node_ids = _object(data["node_ids"], "SemanticArchive.node_ids")
        scopes = data.get("scopes", [])
        storage = data.get("storage", [])
        if not isinstance(scopes, list):
            raise SchemaError("SemanticArchive.scopes must be an array")
        if not isinstance(storage, list):
            raise SchemaError("SemanticArchive.storage must be an array")
        return cls(
            kernel_id=data["kernel_id"],
            intent=_strict_intent(data["intent"]),
            node_ids=dict(node_ids),
            boundary=Boundary.from_json(data["boundary"]),
            scopes=[PortableScope.from_json(item) for item in scopes],
            storage=[StorageObject.from_json(item) for item in storage],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "intent": self.intent.to_json(),
            "node_ids": dict(self.node_ids),
            "boundary": self.boundary.to_json(),
            "scopes": [item.to_json() for item in self.scopes],
            "storage": [item.to_json() for item in self.storage],
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
        if self.kind is ParameterKind.OPAQUE and (
            self.mode is not ParameterMode.VALUE
            or self.dtype is None
            or self.rank is not None
            or self.alias_of is not None
            or self.tensor_ref is not None
        ):
            raise SchemaError(
                "opaque parameters expose only a positional name and public dtype"
            )

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
    role: str
    value_type: str = "int"
    required: bool = True

    def __post_init__(self) -> None:
        self.name = _string(self.name, "ScheduleSlot.name")
        self.kind = _enum(ScheduleSlotKind, self.kind, "ScheduleSlot.kind")
        self.role = _string(self.role, "ScheduleSlot.role")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.role) is None:
            raise SchemaError("ScheduleSlot.role must use lower_snake_case")
        forbidden_role_terms = {
            "api",
            "combiner",
            "conflict",
            "family",
            "fill",
            "identity",
            "index_map",
            "mask",
            "operator",
            "predicate",
            "realization",
            "rule",
            "semantic",
            "source",
            "strategy",
        }
        if any(term in self.role for term in forbidden_role_terms):
            raise SchemaError("ScheduleSlot.role must remain pure scheduling metadata")
        self.value_type = _string(self.value_type, "ScheduleSlot.value_type")
        self.required = _boolean(self.required, "ScheduleSlot.required")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ScheduleSlot":
        data = _strict_object(
            value,
            path="ScheduleSlot",
            allowed={"name", "kind", "role", "value_type", "required"},
            required={"name", "kind", "role"},
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            role=data["role"],
            value_type=data.get("value_type", "int"),
            required=data.get("required", True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "role": self.role,
            "value_type": self.value_type,
            "required": self.required,
        }


@dataclass
class ScheduleSpec(StrictJSONMixin):
    """Frozen ``C.schedule`` slot declarations and trusted concrete values."""

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
    """Privileged Lift result for the LLM-direct Gate1 epoch.

    Only ``J`` and pure ``C`` form Intent.  ``pi`` is a source-realization
    witness outside Intent; interface/source/location/provenance belong to the
    harness.  Persisted documents from earlier epochs are rejected rather than
    upgraded through a compatibility path.
    """

    SCHEMA: ClassVar[str] = "intentir.gate1.llm-direct.v2"

    j: SemanticArchive
    interface: KernelInterface
    schedule: ScheduleSpec
    pi: RealizationWitness
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
        if not isinstance(self.pi, RealizationWitness):
            try:
                self.pi = RealizationWitness.from_json(self.pi)
            except RealizationWitnessError as exc:
                raise SchemaError(str(exc)) from exc
        if not isinstance(self.location_map, LocationMap):
            self.location_map = LocationMap.from_json(self.location_map)
        if self.source is not None:
            self.source = _string(self.source, "Gate1Artifact.source", nonempty=False)
        self.provenance = [
            item if isinstance(item, ProvenanceRecord)
            else ProvenanceRecord.from_json(item)
            for item in self.provenance
        ]
        if self.j.kernel_id != self.interface.kernel_id:
            raise SchemaError("Gate1Artifact J and interface kernel ids must match")

        tensors = {
            tensor.name: tensor
            for tensor in [*self.j.intent.inputs, *self.j.intent.outputs]
        }
        bad_tensor_refs = sorted(
            parameter.tensor_ref
            for parameter in self.interface.parameters
            if parameter.tensor_ref is not None and parameter.tensor_ref not in tensors
        )
        if bad_tensor_refs:
            raise SchemaError(
                "interface parameters reference unknown J tensors: "
                + ", ".join(bad_tensor_refs)
            )

        node_refs = set(self.j.node_ids) | set(self.j.node_ids.values())
        boundary_node_refs = {
            *(item.node_ref for item in self.j.boundary.accesses if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.broadcasts),
            *(item.node_ref for item in self.j.boundary.effects if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.output_mappings if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.reductions),
            *(item.node_ref for item in self.j.boundary.index_bounds),
            *(item.node_ref for item in self.j.boundary.updates),
            *(
                item.update_node_ref
                for item in self.j.boundary.result_relations
                if item.update_node_ref is not None
            ),
        }
        unknown_boundary_nodes = sorted(boundary_node_refs - node_refs)
        if unknown_boundary_nodes:
            raise SchemaError(
                "Layer B references unknown J nodes: "
                + ", ".join(unknown_boundary_nodes)
            )

        semantic_refs: set[str] = set()

        def collect_opaque_handles(value: Any) -> None:
            if isinstance(value, str):
                if re.fullmatch(r"(?:t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+", value):
                    semantic_refs.add(value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    collect_opaque_handles(key)
                    collect_opaque_handles(item)
            elif isinstance(value, list):
                for item in value:
                    collect_opaque_handles(item)

        collect_opaque_handles(self.j.to_json())
        collect_opaque_handles(self.interface.to_json())
        collect_opaque_handles(self.schedule.to_json())
        unknown_pi_refs = sorted(
            {
                str(item["semantic_ref"])
                for item in self.pi.instances
                if item["semantic_ref"] not in semantic_refs
            }
        )
        if unknown_pi_refs:
            raise SchemaError(
                "pi references unknown J/C handle(s): " + ", ".join(unknown_pi_refs)
            )

        location_nodes = {
            ref for entry in self.location_map.entries for ref in entry.node_refs
        }
        location_accesses = {
            ref for entry in self.location_map.entries for ref in entry.access_refs
        }
        if location_nodes - node_refs:
            raise SchemaError("LocationMap references unknown J nodes")
        access_ids = {item.access_id for item in self.j.boundary.accesses}
        if location_accesses - access_ids:
            raise SchemaError("LocationMap references unknown Layer-B accesses")

        schedule_by_name = {slot.name: slot for slot in self.schedule.slots}
        constexprs = {
            parameter.name
            for parameter in self.interface.parameters
            if parameter.kind is ParameterKind.CONSTEXPR
        }
        declared_constexprs = {
            name for name, slot in schedule_by_name.items()
            if slot.kind is ScheduleSlotKind.CONSTEXPR
        }
        if constexprs != declared_constexprs:
            raise SchemaError("interface constexpr parameters and C slots differ")
        compiler_slots = set(self.interface.compiler_slots)
        declared_compiler = {
            name for name, slot in schedule_by_name.items()
            if slot.kind is ScheduleSlotKind.COMPILER
        }
        if not compiler_slots <= declared_compiler:
            raise SchemaError("interface compiler slots are absent from pure C")
        missing_values = self.schedule.missing_required_values()
        if missing_values:
            raise SchemaError(
                "Gate1Artifact is missing required C value(s): "
                + ", ".join(missing_values)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Gate1Artifact":
        data = _strict_object(
            value,
            path="Gate1Artifact",
            allowed={
                "schema", "j", "interface", "schedule", "pi",
                "location_map", "source", "provenance",
            },
            required={
                "schema", "j", "interface", "schedule", "pi", "location_map",
            },
        )
        if data["schema"] != cls.SCHEMA:
            raise SchemaError(
                f"unsupported Gate1 artifact schema {data['schema']!r}; "
                f"expected {cls.SCHEMA!r}"
            )
        provenance = data.get("provenance", [])
        if not isinstance(provenance, list):
            raise SchemaError("Gate1Artifact.provenance must be an array")
        try:
            pi = RealizationWitness.from_json(data["pi"])
        except RealizationWitnessError as exc:
            raise SchemaError(str(exc)) from exc
        return cls(
            j=SemanticArchive.from_json(data["j"]),
            interface=KernelInterface.from_json(data["interface"]),
            schedule=ScheduleSpec.from_json(data["schedule"]),
            pi=pi,
            location_map=LocationMap.from_json(data["location_map"]),
            source=data.get("source"),
            provenance=[ProvenanceRecord.from_json(item) for item in provenance],
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.SCHEMA,
            "j": self.j.to_json(),
            "interface": self.interface.to_json(),
            "schedule": self.schedule.to_json(),
            "pi": self.pi.to_json(),
            "location_map": self.location_map.to_json(),
            "provenance": [item.to_json() for item in self.provenance],
        }
        if self.source is not None:
            result["source"] = self.source
        return result

    def rewriter_view(self, *, masked: bool = False) -> dict[str, Any]:
        if masked:
            return _masked_rewriter_view(self)
        return _opaque_normal_rewriter_view(self)


def opaque_replay_abi(
    artifact: Gate1Artifact,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return declaration-order aliases used by both replay packets."""

    aliases, _bindings, _order = _opaque_reference_data(artifact)
    parameter_names = {
        parameter.name: aliases[parameter.name]
        for parameter in artifact.interface.parameters
    }
    schedule_names = {
        slot.name: aliases[slot.name] for slot in artifact.schedule.slots
    }
    return "scrc_k0", parameter_names, schedule_names


_OPAQUE_EXPRESSION_RESERVED = frozenset(
    {
        "and",
        "or",
        "not",
        "True",
        "False",
        "None",
        "inf",
        "nan",
    }
)


def _intent_ops_with_paths(
    body: list[dict[str, Any]], prefix: str = "body"
) -> list[tuple[dict[str, Any], str]]:
    result: list[tuple[dict[str, Any], str]] = []
    for index, op in enumerate(body):
        path = f"{prefix}[{index}]"
        result.append((op, path))
        nested = op.get("body", [])
        if isinstance(nested, list):
            result.extend(_intent_ops_with_paths(nested, f"{path}.body"))
    return result


def _intent_result_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _expression_identifiers(value: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end())
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value)
    ]


def _is_expression_function_or_qualified(
    value: str, start: int, end: int
) -> bool:
    before = value[:start].rstrip()
    after = value[end:].lstrip()
    return before.endswith(".") or after.startswith(".") or after.startswith("(")


def _boundary_expression_fields(
    artifact: Gate1Artifact,
) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for index, expression in enumerate(artifact.j.boundary.domains.values()):
        fields.append((expression, f"j.boundary.domains[{index}]"))
    for index, access in enumerate(artifact.j.boundary.accesses):
        if access.index_expression is not None:
            fields.append(
                (
                    access.index_expression,
                    f"j.boundary.accesses[{index}].index_expression",
                )
            )
        if access.predicate is not None:
            fields.append(
                (access.predicate, f"j.boundary.accesses[{index}].predicate")
            )
    for index, mapping in enumerate(artifact.j.boundary.output_mappings):
        fields.append(
            (mapping.value, f"j.boundary.output_mappings[{index}].value")
        )
    for index, broadcast in enumerate(artifact.j.boundary.broadcasts):
        for axis_index, axis in enumerate(broadcast.axes):
            if isinstance(axis, str):
                fields.append(
                    (
                        axis,
                        f"j.boundary.broadcasts[{index}].axes[{axis_index}]",
                    )
                )
    for index, reduction in enumerate(artifact.j.boundary.reductions):
        if isinstance(reduction.tile_axis, str):
            fields.append(
                (
                    reduction.tile_axis,
                    f"j.boundary.reductions[{index}].tile_axis",
                )
            )
    return fields


def _opaque_reference_data(
    artifact: Gate1Artifact,
) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
    """Build declaration-order aliases without exposing the original spelling."""

    intent_json = artifact.j.intent.to_json()["intent_function"]
    body = intent_json.get("body", [])
    expression_fields = _boundary_expression_fields(artifact)
    occupied: set[str] = {
        artifact.j.kernel_id,
        artifact.j.intent.name,
        *(tensor.name for tensor in artifact.j.intent.inputs),
        *(tensor.name for tensor in artifact.j.intent.outputs),
        *artifact.j.node_ids,
        *artifact.j.node_ids.values(),
        *artifact.j.boundary.domains,
        *(access.access_id for access in artifact.j.boundary.accesses),
        *(
            effect.effect_id
            for effect in artifact.j.boundary.effects
            if effect.effect_id is not None
        ),
        *(scope.scope_id for scope in artifact.j.scopes),
        *(storage.storage_id for storage in artifact.j.storage),
        *(member for storage in artifact.j.storage for member in storage.members),
        *(parameter.name for parameter in artifact.interface.parameters),
        *(slot.name for slot in artifact.schedule.slots),
    }
    for op, _ in _intent_ops_with_paths(body):
        occupied.update(_intent_result_names(op.get("result")))
        if isinstance(op.get("scope_id"), str):
            occupied.add(op["scope_id"])
        for state in op.get("carry", []):
            for key in (
                "state_id", "body_argument", "next_value", "result_value",
            ):
                if isinstance(state.get(key), str):
                    occupied.add(state[key])
        for scan in op.get("scan", []):
            if isinstance(scan.get("name"), str):
                occupied.add(scan["name"])
    for expression, _ in expression_fields:
        occupied.update(token for token, _, _ in _expression_identifiers(expression))

    aliases: dict[str, str] = {}
    owners: dict[str, list[dict[str, str]]] = {}
    alias_order: list[str] = []

    def allocate(original: str, stem: str, kind: str, path: str) -> str:
        if original in aliases:
            alias = aliases[original]
        else:
            candidate = stem
            suffix = 0
            while (
                candidate in owners
                or (candidate in occupied and candidate != original)
            ):
                suffix += 1
                candidate = f"{stem}_{suffix}"
            alias = candidate
            aliases[original] = alias
            owners[alias] = []
            alias_order.append(alias)
        owner = {"kind": kind, "path": path}
        if owner not in owners[alias]:
            owners[alias].append(owner)
        return alias

    tensor_index = 0
    for collection_name, tensors in (
        ("inputs", artifact.j.intent.inputs),
        ("outputs", artifact.j.intent.outputs),
    ):
        for index, tensor in enumerate(tensors):
            allocate(
                tensor.name,
                f"t{tensor_index}",
                "j_tensor",
                f"j.intent.intent_function.{collection_name}[{index}]",
            )
            tensor_index += 1

    for index, (node_id, path) in enumerate(artifact.j.node_ids.items()):
        alias = allocate(node_id, f"n{index}", "j_node", f"j.intent.{path}")
        aliases[path] = alias

    for index, name in enumerate(artifact.j.boundary.domains):
        allocate(name, f"d{index}", "boundary_domain", f"j.boundary.domains[{index}]")
    for index, access in enumerate(artifact.j.boundary.accesses):
        allocate(
            access.access_id,
            f"a{index}",
            "boundary_access",
            f"j.boundary.accesses[{index}]",
        )
    for index, effect in enumerate(artifact.j.boundary.effects):
        assert effect.effect_id is not None
        allocate(
            effect.effect_id,
            f"e{index}",
            "boundary_effect",
            f"j.boundary.effects[{index}]",
        )

    value_index = 0
    for op, path in _intent_ops_with_paths(body):
        if isinstance(op.get("scope_id"), str):
            scope_index = next(
                (
                    index
                    for index, scope in enumerate(artifact.j.scopes)
                    if scope.scope_id == op["scope_id"]
                ),
                0,
            )
            allocate(
                op["scope_id"],
                f"q{scope_index}",
                "j_scope",
                f"j.intent.{path}.scope_id",
            )
        for scan_index, scan in enumerate(op.get("scan", [])):
            allocate(
                scan["name"],
                f"v{value_index}",
                "j_loop_scan_value",
                f"j.intent.{path}.scan[{scan_index}].name",
            )
            value_index += 1
        for state_index, state in enumerate(op.get("carry", [])):
            allocate(
                state["state_id"],
                f"s{state_index}",
                "j_loop_state",
                f"j.intent.{path}.carry[{state_index}].state_id",
            )
            allocate(
                state["body_argument"],
                f"v{value_index}",
                "j_loop_body_argument",
                f"j.intent.{path}.carry[{state_index}].body_argument",
            )
            value_index += 1
        for result_index, name in enumerate(_intent_result_names(op.get("result"))):
            allocate(
                name,
                f"v{value_index}",
                "j_value",
                f"j.intent.{path}.result[{result_index}]",
            )
            value_index += 1

    for index, scope in enumerate(artifact.j.scopes):
        allocate(scope.scope_id, f"q{index}", "j_scope", f"j.scopes[{index}]")
    for index, storage in enumerate(artifact.j.storage):
        allocate(
            storage.storage_id,
            f"b{index}",
            "j_storage",
            f"j.storage[{index}]",
        )

    runtime_index = 0
    constexpr_index = 0
    for index, parameter in enumerate(artifact.interface.parameters):
        if parameter.kind is ParameterKind.CONSTEXPR:
            stem = f"c{constexpr_index}"
            constexpr_index += 1
        else:
            stem = f"p{runtime_index}"
            runtime_index += 1
        allocate(
            parameter.name,
            stem,
            "interface_parameter",
            f"interface.parameters[{index}]",
        )

    dimension_index = 0
    for tensor_index, tensor in enumerate(
        [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    ):
        for shape_index, dim in enumerate(tensor.shape):
            if not isinstance(dim, str) or dim in aliases:
                continue
            allocate(
                dim,
                f"x{dimension_index}",
                "j_dimension",
                f"j.intent.tensor_shapes[{tensor_index}][{shape_index}]",
            )
            dimension_index += 1
    compiler_index = constexpr_index
    for index, slot in enumerate(artifact.schedule.slots):
        if slot.name in aliases:
            stem = aliases[slot.name]
        else:
            stem = f"c{compiler_index}"
            compiler_index += 1
        allocate(
            slot.name,
            stem,
            "schedule_slot",
            f"schedule.slots[{index}]",
        )

    for expression, path in expression_fields:
        for token, start, end in _expression_identifiers(expression):
            if token in aliases or token in _OPAQUE_EXPRESSION_RESERVED:
                continue
            if _is_expression_function_or_qualified(expression, start, end):
                continue
            allocate(token, f"v{value_index}", "boundary_symbol", path)
            value_index += 1

    bindings = [
        {"ref": alias, "owners": copy.deepcopy(owners[alias])}
        for alias in alias_order
    ]
    return aliases, bindings, alias_order


def _opaque_reference(value: str, aliases: dict[str, str]) -> str:
    try:
        return aliases[value]
    except KeyError as exc:
        raise SchemaError(
            f"replay reference {value!r} has no opaque alias"
        ) from exc


def _opaque_expression(value: str, aliases: dict[str, str]) -> str:
    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: aliases.get(match.group(0), match.group(0)),
        value,
    )


def _opaque_json_exact(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, list):
        return [_opaque_json_exact(item, aliases) for item in value]
    if isinstance(value, dict):
        return {
            key: _opaque_json_exact(item, aliases)
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def _opaque_intent_op(
    op: dict[str, Any], aliases: dict[str, str]
) -> dict[str, Any]:
    result = copy.deepcopy(op)
    if "result" in result:
        result["result"] = _opaque_json_exact(result["result"], aliases)
    for key in (
        "operands", "scope_id", "carry", "trip", "scan", "yield",
    ):
        if key in result:
            result[key] = _opaque_json_exact(result[key], aliases)
    for key in ("attrs", "type"):
        if key in result:
            result[key] = _opaque_json_exact(result[key], aliases)
    if "body" in result:
        result["body"] = [
            _opaque_intent_op(item, aliases) for item in result["body"]
        ]
    return result



def _opaque_scope_json(
    scope: PortableScope, aliases: dict[str, str]
) -> dict[str, Any]:
    return {
        "scope_id": _opaque_reference(scope.scope_id, aliases),
        "kind": scope.kind.value,
        "parent_scope": (
            _opaque_reference(scope.parent_scope, aliases)
            if scope.parent_scope is not None else None
        ),
        "captures": [_opaque_reference(ref, aliases) for ref in scope.captures],
        "definitions": [
            _opaque_reference(ref, aliases) for ref in scope.definitions
        ],
        "effects": [_opaque_reference(ref, aliases) for ref in scope.effects],
    }


def _opaque_storage_json(
    storage: StorageObject, aliases: dict[str, str]
) -> dict[str, Any]:
    def event(value: SemanticEventRef) -> dict[str, str]:
        assert value.ref is not None
        return {
            "kind": value.kind.value,
            "ref": _opaque_reference(value.ref, aliases),
        }

    return {
        "storage_id": _opaque_reference(storage.storage_id, aliases),
        "members": [
            _opaque_reference(member, aliases) for member in storage.members
        ],
        "scope_id": _opaque_reference(storage.scope_id, aliases),
        "mutability": storage.mutability.value,
        "init_event": event(storage.init_event),
        "last_use_event": event(storage.last_use_event),
        "escapes": storage.escapes,
        "crosses_iteration": storage.crosses_iteration,
    }


def _opaque_pi_json(
    witness: RealizationWitness, aliases: dict[str, str]
) -> dict[str, Any]:
    """Alpha-rename only pi's opaque J/C references; topology is unchanged."""

    result = witness.to_json()
    for instance in result["instances"]:
        instance["semantic_ref"] = _opaque_reference(
            instance["semantic_ref"], aliases
        )
    return result


def _opaque_harness_view(
    artifact: Gate1Artifact, aliases: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose positional invocation shape without source spellings or C values."""

    interface = artifact.interface.rewriter_view()
    interface["kernel_id"] = "scrc_k0"
    interface["function_name"] = "scrc_k0"
    for parameter in interface["parameters"]:
        original = parameter["name"]
        parameter["name"] = _opaque_reference(original, aliases)
        if parameter.get("tensor_ref") is not None:
            parameter["tensor_ref"] = _opaque_reference(
                parameter["tensor_ref"], aliases
            )
        if parameter.get("alias_of") is not None:
            parameter["alias_of"] = _opaque_reference(
                parameter["alias_of"], aliases
            )
    interface["grid"] = [
        _rename_identifier_tokens(expression, aliases)
        for expression in interface["grid"]
    ]
    interface["compiler_slots"] = [
        _opaque_reference(name, aliases)
        for name in interface["compiler_slots"]
    ]

    schedule = artifact.schedule.rewriter_view()
    for slot in schedule["slots"]:
        slot["name"] = _opaque_reference(slot["name"], aliases)
    return interface, schedule


def _opaque_normal_rewriter_view(artifact: Gate1Artifact) -> dict[str, Any]:
    """Preserve J/B semantics while removing source-derived identifier hints."""

    aliases, _bindings, _ = _opaque_reference_data(artifact)
    intent = artifact.j.intent.to_json()["intent_function"]
    opaque_intent = {
        "intent_function": {
            "name": "scrc_intent0",
            "inputs": [
                {
                    "name": _opaque_reference(tensor.name, aliases),
                    "dtype": tensor.dtype,
                    "shape": [
                        aliases.get(dim, dim) if isinstance(dim, str) else dim
                        for dim in tensor.shape
                    ],
                }
                for tensor in artifact.j.intent.inputs
            ],
            "outputs": [
                {
                    "name": _opaque_reference(tensor.name, aliases),
                    "dtype": tensor.dtype,
                    "shape": [
                        aliases.get(dim, dim) if isinstance(dim, str) else dim
                        for dim in tensor.shape
                    ],
                }
                for tensor in artifact.j.intent.outputs
            ],
            "body": [
                _opaque_intent_op(op, aliases) for op in intent.get("body", [])
            ],
        }
    }

    boundary = artifact.j.boundary
    opaque_boundary = {
        "domains": {
            _opaque_reference(name, aliases): _opaque_expression(expression, aliases)
            for name, expression in boundary.domains.items()
        },
        "accesses": [
            {
                "access_id": _opaque_reference(access.access_id, aliases),
                "kind": access.kind.value,
                "tensor": _opaque_reference(access.tensor, aliases),
                "index_expression": (
                    _opaque_expression(access.index_expression, aliases)
                    if access.index_expression is not None
                    else None
                ),
                **(
                    {"predicate": _opaque_expression(access.predicate, aliases)}
                    if access.predicate is not None
                    else {}
                ),
                **(
                    {"fill_value": copy.deepcopy(access.fill_value)}
                    if access.fill_value is not None
                    else {}
                ),
                **(
                    {"node_ref": _opaque_reference(access.node_ref, aliases)}
                    if access.node_ref is not None
                    else {}
                ),
            }
            for access in boundary.accesses
        ],
        "broadcasts": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "operand": _opaque_reference(item.operand, aliases),
                "axes": [
                    _opaque_expression(axis, aliases)
                    if isinstance(axis, str)
                    else axis
                    for axis in item.axes
                ],
            }
            for item in boundary.broadcasts
        ],
        "aliases": [
            {
                "left": _opaque_reference(item.left, aliases),
                "right": _opaque_reference(item.right, aliases),
                "kind": item.kind.value,
            }
            for item in boundary.aliases
        ],
        "effects": [
            {
                "effect_id": _opaque_reference(item.effect_id, aliases),
                "kind": item.kind.value,
                "target": _opaque_reference(item.target, aliases),
                **(
                    {"node_ref": _opaque_reference(item.node_ref, aliases)}
                    if item.node_ref is not None
                    else {}
                ),
                "access_ref": (
                    _opaque_reference(item.access_ref, aliases)
                    if item.access_ref is not None else None
                ),
            }
            for item in boundary.effects
        ],
        "output_mappings": [
            {
                "output": _opaque_reference(item.output, aliases),
                "value": _opaque_expression(item.value, aliases),
                **(
                    {"node_ref": _opaque_reference(item.node_ref, aliases)}
                    if item.node_ref is not None
                    else {}
                ),
            }
            for item in boundary.output_mappings
        ],
        "reductions": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "tile_axis": (
                    _opaque_expression(item.tile_axis, aliases)
                    if isinstance(item.tile_axis, str)
                    else item.tile_axis
                ),
                **(
                    {"identity": copy.deepcopy(item.identity)}
                    if item.identity is not None
                    else {}
                ),
                **(
                    {
                        "masked_contribution": copy.deepcopy(
                            item.masked_contribution
                        )
                    }
                    if item.masked_contribution is not None
                    else {}
                ),
            }
            for item in boundary.reductions
        ],
        "index_bounds": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "data": _opaque_reference(item.data, aliases),
                "indices": _opaque_reference(item.indices, aliases),
                "axis": item.axis,
            }
            for item in boundary.index_bounds
        ],
        "updates": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "base": _opaque_reference(item.base, aliases),
                "indices": _opaque_reference(item.indices, aliases),
                "values": _opaque_reference(item.values, aliases),
                "axis": item.axis,
                "combiner": item.combiner.value,
                "duplicate_policy": item.duplicate_policy.value,
                "order": item.order.value,
            }
            for item in boundary.updates
        ],
        "result_relations": [
            {
                "output": _opaque_reference(item.output, aliases),
                "kind": item.kind.value,
                **(
                    {
                        "rtol": item.rtol,
                        "atol": item.atol,
                        "equal_nan": item.equal_nan,
                    }
                    if item.kind is ResultRelationKind.TOLERANCE
                    else {}
                ),
                **(
                    {
                        "update_node_ref": _opaque_reference(
                            item.update_node_ref, aliases
                        )
                    }
                    if item.update_node_ref is not None
                    else {}
                ),
            }
            for item in boundary.result_relations
        ],
    }

    interface, schedule = _opaque_harness_view(artifact, aliases)

    return {
        "j": {
            "kernel_id": "scrc_j0",
            "intent": opaque_intent,
            "node_ids": {
                _opaque_reference(node_id, aliases): path
                for node_id, path in artifact.j.node_ids.items()
            },
            "boundary": opaque_boundary,
            "scopes": [
                _opaque_scope_json(scope, aliases)
                for scope in artifact.j.scopes
            ],
            "storage": [
                _opaque_storage_json(storage, aliases)
                for storage in artifact.j.storage
            ],
        },
        "interface": interface,
        "schedule": schedule,
        "pi": _opaque_pi_json(artifact.pi, aliases),
    }


def _rename_identifier_tokens(value: str, aliases: dict[str, str]) -> str:
    """Alpha-rename complete Python-like identifier tokens in an expression."""

    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: aliases.get(match.group(0), match.group(0)),
        value,
    )


def _masked_rewriter_view(artifact: Gate1Artifact) -> dict[str, Any]:
    """Create a consistently opaque J/pi/interface view for leakage ablation."""

    tensor_names = [tensor.name for tensor in artifact.j.intent.inputs]
    tensor_names.extend(tensor.name for tensor in artifact.j.intent.outputs)
    tensor_map = {
        name: f"t{index}"
        for index, name in enumerate(dict.fromkeys(tensor_names))
    }
    node_map = {name: f"n{index}" for index, name in enumerate(artifact.j.node_ids)}
    access_map = {
        access.access_id: f"a{index}"
        for index, access in enumerate(artifact.j.boundary.accesses)
    }
    domain_map = {
        name: f"d{index}"
        for index, name in enumerate(artifact.j.boundary.domains)
    }
    effect_map = {
        effect.effect_id: f"e{index}"
        for index, effect in enumerate(artifact.j.boundary.effects)
        if effect.effect_id is not None
    }
    scope_map = {
        scope.scope_id: f"q{index}"
        for index, scope in enumerate(artifact.j.scopes)
    }
    storage_map = {
        storage.storage_id: f"b{index}"
        for index, storage in enumerate(artifact.j.storage)
    }

    opaque_function, parameter_map, schedule_map = opaque_replay_abi(artifact)
    abi_map = {**parameter_map, **schedule_map}

    reference_map = {
        **tensor_map, **node_map, **access_map, **domain_map, **effect_map,
        **scope_map, **storage_map,
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
        if value in abi_map:
            return abi_map[value]
        if value not in fallback_reference_map:
            fallback_reference_map[value] = f"r{len(fallback_reference_map)}"
        return fallback_reference_map[value]

    path_to_node = {
        path: node_map[node_id] for node_id, path in artifact.j.node_ids.items()
    }
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
            "name": opaque_function,
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
                **(
                    {"node_ref": opaque_reference(access.node_ref)}
                    if access.node_ref is not None
                    else {}
                ),
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
                "effect_id": effect_map[item.effect_id],
                "kind": "<masked>",
                "target": opaque_reference(item.target),
                **(
                    {"node_ref": opaque_reference(item.node_ref)}
                    if item.node_ref is not None
                    else {}
                ),
                "access_ref": (
                    opaque_reference(item.access_ref)
                    if item.access_ref is not None else None
                ),
            }
            for item in artifact.j.boundary.effects
        ],
        "output_mappings": [
            {
                "output": opaque_reference(item.output),
                "value": "<masked>",
                **(
                    {"node_ref": opaque_reference(item.node_ref)}
                    if item.node_ref is not None
                    else {}
                ),
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
        "index_bounds": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "data": opaque_reference(item.data),
                "indices": opaque_reference(item.indices),
                "axis": "<masked>",
            }
            for item in artifact.j.boundary.index_bounds
        ],
        "updates": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "base": opaque_reference(item.base),
                "indices": opaque_reference(item.indices),
                "values": opaque_reference(item.values),
                "axis": "<masked>",
                "combiner": "<masked>",
                "duplicate_policy": "<masked>",
                "order": "<masked>",
            }
            for item in artifact.j.boundary.updates
        ],
        "result_relations": [
            {
                "output": opaque_reference(item.output),
                "kind": "<masked>",
                **(
                    {
                        "rtol": "<masked>",
                        "atol": "<masked>",
                        "equal_nan": "<masked>",
                    }
                    if item.kind is ResultRelationKind.TOLERANCE
                    else {}
                ),
                **(
                    {
                        "update_node_ref": opaque_reference(
                            item.update_node_ref
                        )
                    }
                    if item.update_node_ref is not None
                    else {}
                ),
            }
            for item in artifact.j.boundary.result_relations
        ],
    }

    # The anti-leakage arm masks J only.  Its pi, C declaration and public
    # frontend/harness context are byte-identical to the normal replay packet.
    normal_aliases, _bindings, _order = _opaque_reference_data(artifact)
    opaque_interface, opaque_schedule = _opaque_harness_view(
        artifact, normal_aliases
    )
    return {
        "j": {
            "kernel_id": opaque_function,
            "intent": masked_intent,
            "node_ids": {
                node_map[node_id]: f"node[{index}]"
                for index, node_id in enumerate(artifact.j.node_ids)
            },
            "boundary": masked_boundary,
            "scopes": [
                {
                    "scope_id": scope_map[scope.scope_id],
                    "kind": "<masked>",
                    "parent_scope": (
                        scope_map[scope.parent_scope]
                        if scope.parent_scope is not None
                        else None
                    ),
                    "captures": [opaque_reference(ref) for ref in scope.captures],
                    "definitions": [
                        opaque_reference(ref) for ref in scope.definitions
                    ],
                    "effects": [opaque_reference(ref) for ref in scope.effects],
                }
                for scope in artifact.j.scopes
            ],
            "storage": [
                {
                    "storage_id": storage_map[item.storage_id],
                    "members": [opaque_reference(ref) for ref in item.members],
                    "scope_id": scope_map[item.scope_id],
                    "mutability": "<masked>",
                    "init_event": {
                        "kind": "<masked>",
                        "ref": opaque_reference(item.init_event.ref),
                    },
                    "last_use_event": {
                        "kind": "<masked>",
                        "ref": opaque_reference(item.last_use_event.ref),
                    },
                    "escapes": "<masked>",
                    "crosses_iteration": "<masked>",
                }
                for item in artifact.j.storage
            ],
        },
        "interface": opaque_interface,
        "schedule": opaque_schedule,
        "pi": _opaque_pi_json(artifact.pi, normal_aliases),
    }


__all__ = [
    "AccessKind", "AliasKind", "AliasSpec", "Boundary", "BoundaryAccess",
    "BroadcastSpec", "CompileCase", "DuplicatePolicy", "EffectKind",
    "EffectSpec", "Frontend",
    "Gate1Artifact", "Gate1Status", "JsonValue", "KernelInterface",
    "IndexBoundsSpec", "KernelParameter", "LocationEntry", "LocationMap",
    "LogicalUpdateSpec", "OutputMapping",
    "opaque_replay_abi",
    "ParameterKind", "ParameterMode", "ProvenanceRecord", "ProvenanceRole",
    "PortableScope", "RealizationWitness", "ReductionBoundary",
    "ResultRelationKind", "ResultRelationSpec",
    "RuntimeArgument", "ScheduleSlot", "ScheduleSlotKind", "ScheduleSpec",
    "SchemaError", "ScopeKind", "SemanticArchive", "SemanticEventKind",
    "SemanticEventRef", "SourceSpan", "StorageMutability", "StorageObject",
    "StrictJSONMixin", "UpdateCombiner", "UpdateOrder",
]
