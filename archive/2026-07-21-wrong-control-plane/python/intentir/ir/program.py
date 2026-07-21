"""Canonical, verification-neutral IntentIR lowering contract.

``IntentProgram`` is the only object in the active architecture that combines
Layer A, typed Layer B, the logical ABI, and an optional replaceable Layer-C
hint.  It intentionally contains no Gate verdict, source code, replay code,
provider receipt, dataset identity, or paper metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .abi import LogicalInterface
from .contracts import BoundaryContract, ContractError, JsonScalar
from .model import IntentFunction, IntentOp, IntentReturn


PROGRAM_SCHEMA = "intentir.program.v1"


class IntentProgramError(ValueError):
    """Raised when an object is not a canonical portable IntentProgram."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntentProgramError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise IntentProgramError(f"{context} keys must be strings")
    return value


def _strict(
    value: Any,
    *,
    context: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    result = _mapping(value, context)
    missing = sorted(required - set(result))
    unknown = sorted(set(result) - required - optional)
    if missing:
        raise IntentProgramError(f"{context} missing fields: {missing}")
    if unknown:
        raise IntentProgramError(f"{context} has unknown fields: {unknown}")
    return result


def _array(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise IntentProgramError(f"{context} must be an array")
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None:
        raise IntentProgramError(f"{context} must be a portable identifier")
    return value


def _json_scalar(value: Any, context: str) -> JsonScalar:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise IntentProgramError(f"{context} must be a finite JSON scalar")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_A_ATTR_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    "arg_reduce": ({"fn", "axis"}, {"tie_policy", "result_dtype"}),
    "cast": (set(), {"to_dtype", "dtype"}),
    "concat": ({"axis"}, set()),
    "const": ({"value", "dtype", "shape"}, set()),
    "contraction": ({"subscripts"}, {"accumulator_dtype", "result_dtype"}),
    "conv": ({"stride", "padding", "dilation"}, {"groups"}),
    "flip": ({"dims"}, set()),
    "gather": ({"axis"}, {"mode"}),
    "iota": ({"n"}, {"start", "step", "dtype"}),
    "loop": (set(), set()),
    "map_binary": ({"fn"}, set()),
    "map_unary": ({"fn"}, set()),
    "masked_scatter": (set(), {"mode"}),
    "masked_select": (set(), set()),
    "nonzero": (set(), set()),
    "pad": ({"low", "high", "value"}, set()),
    "polar": (set(), set()),
    "pool": ({"fn", "kH", "kW", "sH", "sW", "padH", "padW"}, set()),
    "rand": ({"shape", "dtype", "algorithm"}, {"seed", "counter"}),
    "reduce": ({"fn", "axes"}, {"assoc", "keepdims", "accumulator_dtype", "result_dtype"}),
    "scan": ({"fn", "axis"}, {"inclusive", "reverse"}),
    "scatter": ({"axis"}, {"mode", "reduce"}),
    "select": (set(), set()),
    "sort": ({"axis"}, {"descending", "return", "stable", "tie_policy"}),
    "tile": ({"repeats"}, set()),
    "transpose": ({"perm"}, set()),
    "triu": (set(), {"diagonal"}),
    "upsample": ({"mode"}, {"size", "scale", "align_corners"}),
}

_PURE_MATH_INTRINSICS = frozenset(
    {
        "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "erf",
        "exp", "exp2", "floor", "isfinite", "isinf", "isnan", "log",
        "log2", "logical_not", "neg", "recip", "round", "rsqrt", "sigmoid",
        "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    }
)
_BINARY_FUNCTIONS = frozenset(
    {
        "add", "and", "atan2", "div", "eq", "ge", "gt", "le", "logical_and",
        "logical_or", "lt", "max", "min", "mod", "mul", "ne", "or", "pow",
        "sub", "xor",
    }
)
_REDUCTION_FUNCTIONS = frozenset({"all", "any", "max", "mean", "min", "prod", "sum"})


def _validate_json_value(value: Any, context: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntentProgramError(f"{context} contains NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{context}[{index}]")
        return
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_value(item, f"{context}.{key}")
        return
    raise IntentProgramError(f"{context} contains a non-JSON value")


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list) and all(isinstance(item, str) for item in op.result):
        return tuple(op.result)
    raise IntentProgramError(f"Layer-A op {op.node_id!r} requires named result(s)")


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def _validate_op(op: IntentOp, visible: set[str]) -> tuple[str, ...]:
    if op.node_id is None:
        raise IntentProgramError("canonical Layer-A ops require stable id")
    _identifier(op.node_id, "Layer-A op id")
    if op.op not in _A_ATTR_SCHEMA:
        raise IntentProgramError(f"unsupported canonical Layer-A op {op.op!r}")
    if op.layer_b:
        raise IntentProgramError(f"op {op.node_id!r} retains legacy layer_b")
    if op.out is not None:
        raise IntentProgramError(f"op {op.node_id!r} retains legacy destination out")
    required, optional = _A_ATTR_SCHEMA[op.op]
    unknown = sorted(set(op.attrs) - required - optional)
    missing = sorted(required - set(op.attrs))
    if missing or unknown:
        raise IntentProgramError(
            f"op {op.node_id!r} attrs mismatch: missing={missing}, unknown={unknown}"
        )
    for key, value in op.attrs.items():
        _validate_json_value(value, f"op {op.node_id}.attrs.{key}")
    if op.type is not None:
        type_value = _strict(
            op.type,
            context=f"op {op.node_id} type",
            required={"dtype", "shape"},
        )
        if not isinstance(type_value["dtype"], str) or not isinstance(type_value["shape"], list):
            raise IntentProgramError(f"op {op.node_id!r} has malformed type")
    if op.op == "map_unary" and op.attrs["fn"] not in _PURE_MATH_INTRINSICS:
        raise IntentProgramError(f"op {op.node_id!r} uses unknown pure math intrinsic")
    if op.op == "map_binary" and op.attrs["fn"] not in _BINARY_FUNCTIONS:
        raise IntentProgramError(f"op {op.node_id!r} uses unknown binary function")
    if op.op == "reduce" and op.attrs["fn"] not in _REDUCTION_FUNCTIONS:
        raise IntentProgramError(f"op {op.node_id!r} uses unknown reduction function")
    for operand in op.operands:
        if operand not in visible:
            raise IntentProgramError(f"op {op.node_id!r} references unknown operand {operand!r}")
    produced = _result_names(op)
    if len(produced) != len(set(produced)) or set(produced) & visible:
        raise IntentProgramError(f"op {op.node_id!r} produces duplicate visible values")
    if op.op == "loop":
        if op.trip is None or op.trip.tile_hint is not None:
            raise IntentProgramError("canonical semantic loop requires axis and forbids tile_hint")
        if not op.carry or not op.body or len(op.yields) != len(op.carry):
            raise IntentProgramError("canonical semantic loop requires closed carry/body/yield")
        nested_visible = set(visible)
        nested_visible.update(item.body_argument for item in op.carry)
        nested_visible.update(item.name for item in op.scan)
        for child in op.body:
            child_results = _validate_op(child, nested_visible)
            nested_visible.update(child_results)
        for value in op.yields:
            if value not in nested_visible:
                raise IntentProgramError(f"loop {op.node_id!r} yields unknown value {value!r}")
    elif op.body or op.carry or op.trip is not None or op.scan or op.yields:
        raise IntentProgramError(f"non-loop op {op.node_id!r} carries loop-only fields")
    return produced


def validate_canonical_function(function: IntentFunction) -> None:
    if not isinstance(function, IntentFunction):
        raise IntentProgramError("IntentProgram.function must be IntentFunction")
    _identifier(function.name, "function name")
    if function.layer_c:
        raise IntentProgramError("canonical function must not contain legacy layer_c")
    tensor_names = [item.name for item in [*function.inputs, *function.outputs]]
    if len(tensor_names) != len(set(tensor_names)):
        raise IntentProgramError("function tensor names must be unique")
    visible = {item.name for item in function.inputs}
    node_ids: list[str] = []
    for op in function.body:
        produced = _validate_op(op, visible)
        visible.update(produced)
        node_ids.extend(item.node_id for item in _walk_ops((op,)) if item.node_id is not None)
    if len(node_ids) != len(set(node_ids)):
        raise IntentProgramError("Layer-A node ids must be unique")
    outputs = [item.name for item in function.outputs]
    returns = list(function.returns)
    if [item.output for item in returns] != outputs:
        raise IntentProgramError("logical returns must cover outputs once and in ABI order")
    for item in returns:
        _identifier(item.output, "logical return output")
        _identifier(item.value, "logical return value")
        if item.value not in visible:
            raise IntentProgramError(f"logical return references unknown value {item.value!r}")


def _canonical_function_dict(function: IntentFunction) -> dict[str, Any]:
    return {
        "name": function.name,
        "inputs": [item.to_json() for item in function.inputs],
        "outputs": [item.to_json() for item in function.outputs],
        "body": [item.to_json() for item in function.body],
        "returns": [item.to_json() for item in function.returns],
    }


def _parse_tensor(value: Any, context: str) -> dict[str, Any]:
    data = _strict(value, context=context, required={"name", "dtype", "shape"})
    _identifier(data["name"], f"{context} name")
    if not isinstance(data["dtype"], str) or not data["dtype"]:
        raise IntentProgramError(f"{context} dtype must be non-empty")
    shape = _array(data["shape"], f"{context} shape")
    for index, dim in enumerate(shape):
        if isinstance(dim, bool) or not isinstance(dim, (int, str)):
            raise IntentProgramError(f"{context} shape[{index}] is invalid")
    return dict(data)


def _parse_op(value: Any, context: str) -> dict[str, Any]:
    data = _strict(
        value,
        context=context,
        required={"id", "op", "result"},
        optional={"operands", "attrs", "type", "scope_id", "carry", "trip", "scan", "body", "yield"},
    )
    result = dict(data)
    result["body"] = [
        _parse_op(item, f"{context}.body[{index}]")
        for index, item in enumerate(_array(data.get("body", ()), f"{context}.body"))
    ]
    return result


def _function_from_dict(value: Any) -> IntentFunction:
    data = _strict(
        value,
        context="IntentProgram.function",
        required={"name", "inputs", "outputs", "body", "returns"},
    )
    inputs = [
        _parse_tensor(item, f"function.inputs[{index}]")
        for index, item in enumerate(_array(data["inputs"], "function inputs"))
    ]
    outputs = [
        _parse_tensor(item, f"function.outputs[{index}]")
        for index, item in enumerate(_array(data["outputs"], "function outputs"))
    ]
    returns = []
    for index, item in enumerate(_array(data["returns"], "function returns")):
        returns.append(
            dict(
                _strict(
                    item,
                    context=f"function.returns[{index}]",
                    required={"output", "value"},
                )
            )
        )
    body = [
        _parse_op(item, f"function.body[{index}]")
        for index, item in enumerate(_array(data["body"], "function body"))
    ]
    function = IntentFunction.from_json(
        {
            "name": data["name"], "inputs": inputs, "outputs": outputs,
            "returns": returns, "body": body,
        }
    )
    validate_canonical_function(function)
    return function


@dataclass(frozen=True)
class StrategyHint:
    family: str
    target_ref: str
    choice: str
    parameters: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, "strategy family"))
        object.__setattr__(self, "target_ref", _identifier(self.target_ref, "strategy target_ref"))
        object.__setattr__(self, "choice", _identifier(self.choice, "strategy choice"))
        parameters = tuple(
            (_identifier(name, "strategy parameter"), _json_scalar(value, "strategy parameter value"))
            for name, value in self.parameters
        )
        if len(parameters) != len({name for name, _value in parameters}):
            raise IntentProgramError("strategy parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "target_ref": self.target_ref,
            "choice": self.choice,
            "parameters": {name: value for name, value in self.parameters},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyHint":
        data = _strict(
            value, context="strategy hint",
            required={"family", "target_ref", "choice", "parameters"},
        )
        parameters = _mapping(data["parameters"], "strategy parameters")
        return cls(
            data["family"], data["target_ref"], data["choice"],
            tuple(parameters.items()),
        )


@dataclass(frozen=True)
class ScheduleHint:
    """Replaceable Layer-C launch and source-strategy hint."""

    values: tuple[tuple[str, JsonScalar], ...] = ()
    grid: tuple[str, ...] = ()
    strategies: tuple[StrategyHint, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(
            (_identifier(name, "schedule value name"), _json_scalar(value, "schedule value"))
            for name, value in self.values
        )
        if len(values) != len({name for name, _value in values}):
            raise IntentProgramError("schedule value names must be unique")
        object.__setattr__(self, "values", values)
        grid = tuple(item if isinstance(item, str) and item else None for item in self.grid)
        if any(item is None for item in grid) or len(grid) > 3:
            raise IntentProgramError("schedule grid must contain up to three expressions")
        object.__setattr__(self, "grid", grid)
        strategies = tuple(
            item if isinstance(item, StrategyHint) else StrategyHint.from_dict(item)
            for item in self.strategies
        )
        keys = [(item.target_ref, item.family) for item in strategies]
        if len(keys) != len(set(keys)):
            raise IntentProgramError("schedule strategies duplicate target/family")
        object.__setattr__(self, "strategies", strategies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {name: value for name, value in self.values},
            "grid": list(self.grid),
            "strategies": [item.to_dict() for item in self.strategies],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScheduleHint":
        data = _strict(
            value, context="schedule hint", required={"values", "grid", "strategies"}
        )
        values = _mapping(data["values"], "schedule values")
        return cls(
            values=tuple(values.items()),
            grid=tuple(_array(data["grid"], "schedule grid")),
            strategies=tuple(
                StrategyHint.from_dict(item)
                for item in _array(data["strategies"], "schedule strategies")
            ),
        )


@dataclass(frozen=True)
class IntentProgram:
    function: IntentFunction
    boundary: BoundaryContract
    interface: LogicalInterface
    schedule_hint: ScheduleHint | None = None

    def __post_init__(self) -> None:
        validate_canonical_function(self.function)
        if not isinstance(self.boundary, BoundaryContract):
            object.__setattr__(self, "boundary", BoundaryContract.from_dict(self.boundary))
        if not isinstance(self.interface, LogicalInterface):
            object.__setattr__(self, "interface", LogicalInterface.from_dict(self.interface))
        if self.schedule_hint is not None and not isinstance(self.schedule_hint, ScheduleHint):
            object.__setattr__(self, "schedule_hint", ScheduleHint.from_dict(self.schedule_hint))
        if self.interface.function != self.function.name:
            raise IntentProgramError("logical interface function differs from Layer-A function")
        output_names = tuple(item.name for item in self.function.outputs)
        if self.interface.outputs != output_names:
            raise IntentProgramError("logical interface outputs differ from function outputs")
        tensor_names = {item.name for item in [*self.function.inputs, *self.function.outputs]}
        node_by_id = {item.node_id: item for item in _walk_ops(self.function.body)}
        output_values = {item.output: item.value for item in self.function.returns}
        for item in self.boundary.accesses:
            if item.tensor not in tensor_names:
                raise IntentProgramError(f"access {item.contract_id!r} references unknown tensor")
            if item.node_ref is not None and item.node_ref not in node_by_id:
                raise IntentProgramError(f"access {item.contract_id!r} references unknown node")
            if item.logical_index_node is not None and item.logical_index_node not in node_by_id:
                raise IntentProgramError(f"access {item.contract_id!r} references unknown logical index node")
        for collection in (
            self.boundary.broadcasts, self.boundary.effects, self.boundary.reductions,
        ):
            for item in collection:
                node_ref = getattr(item, "node_ref", None)
                if node_ref is not None and node_ref not in node_by_id:
                    raise IntentProgramError(f"contract {item.contract_id!r} references unknown node")
        for item in self.boundary.reductions:
            if node_by_id[item.node_ref].op != "reduce":
                raise IntentProgramError(f"reduction contract {item.contract_id!r} targets non-reduce op")
        for item in self.boundary.destinations:
            if output_values.get(item.output) != item.value:
                raise IntentProgramError(f"destination {item.contract_id!r} disagrees with logical return")
            if item.destination not in {arg.name for arg in self.interface.arguments}:
                raise IntentProgramError(f"destination {item.contract_id!r} references unknown ABI argument")

    @property
    def j_payload(self) -> dict[str, Any]:
        return {
            "function": _canonical_function_dict(self.function),
            "boundary": self.boundary.to_dict(),
        }

    @property
    def j_digest(self) -> str:
        """Digest of exactly J=A+B; retuning C and ABI transport do not alter it."""

        return _canonical_digest(self.j_payload)

    @property
    def contract_payload(self) -> dict[str, Any]:
        return {**self.j_payload, "interface": self.interface.to_dict()}

    @property
    def contract_digest(self) -> str:
        """Digest bound by backend conformance: J plus the logical ABI."""

        return _canonical_digest(self.contract_payload)

    @property
    def program_digest(self) -> str:
        return _canonical_digest(self.to_dict()["intent_program"])

    @property
    def obligations(self):
        return self.boundary.obligations()

    def with_schedule_hint(self, schedule_hint: ScheduleHint | None) -> "IntentProgram":
        return replace(self, schedule_hint=schedule_hint)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.contract_payload)
        if self.schedule_hint is not None:
            payload["schedule_hint"] = self.schedule_hint.to_dict()
        return {"schema": PROGRAM_SCHEMA, "intent_program": payload}

    def to_json_str(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntentProgram":
        root = _strict(
            value, context="IntentProgram document",
            required={"schema", "intent_program"},
        )
        if root["schema"] != PROGRAM_SCHEMA:
            raise IntentProgramError(f"unsupported IntentProgram schema {root['schema']!r}")
        payload = _strict(
            root["intent_program"], context="intent_program",
            required={"function", "boundary", "interface"},
            optional={"schedule_hint"},
        )
        try:
            boundary = BoundaryContract.from_dict(payload["boundary"])
        except ContractError as exc:
            raise IntentProgramError(str(exc)) from exc
        return cls(
            function=_function_from_dict(payload["function"]),
            boundary=boundary,
            interface=LogicalInterface.from_dict(payload["interface"]),
            schedule_hint=(
                ScheduleHint.from_dict(payload["schedule_hint"])
                if "schedule_hint" in payload else None
            ),
        )

    @classmethod
    def from_json_str(cls, value: str) -> "IntentProgram":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IntentProgramError(f"invalid IntentProgram JSON: {exc}") from exc
        return cls.from_dict(_mapping(parsed, "IntentProgram document"))


__all__ = [
    "IntentProgram", "IntentProgramError", "PROGRAM_SCHEMA", "ScheduleHint",
    "StrategyHint", "validate_canonical_function",
]
