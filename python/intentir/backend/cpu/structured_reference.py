"""Restricted J-only NumPy denotation for structured Layer-A value semantics.

This is a Gate2 reference executor, not the production CPU backend and not a
fallback for arbitrary IntentOps.  It accepts only a closed operation table,
executes each structured loop over its complete logical trip axis as one
conservative semantic step, and validates every declared input, carry, result,
and output type.  Its API deliberately has no C or pi parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from intentir.backend.cpu.abi import (
    CPUBackendUnsupportedError,
    numpy_dtype,
    validate_array,
)
from intentir.backend.cpu.pipeline import CPUExecutionResult
from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.model import IntentOp, LoopStateBinding
from intentir.verify.gate1.model import SemanticArchive


STRUCTURED_CPU_REFERENCE_PROTOCOL = "intentir.j-structured-cpu-reference.v1"


class StructuredReferenceUnsupportedError(CPUBackendUnsupportedError):
    """The typed J lies outside this deliberately narrow denotation."""


_ORDINARY_OPS = frozenset(
    {
        "cast",
        "const",
        "contraction",
        "map_binary",
        "map_unary",
        "pad",
        "reduce",
        "scan",
        "select",
        "transpose",
    }
)

_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
    "atan2": np.arctan2,
    "gt": np.greater,
    "ge": np.greater_equal,
    "lt": np.less,
    "le": np.less_equal,
    "eq": np.equal,
    "ne": np.not_equal,
}

_UNARY = {
    "identity": lambda value: value,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda value: np.reciprocal(np.sqrt(value)),
    "neg": np.negative,
    "recip": np.reciprocal,
    "abs": np.abs,
    "tanh": np.tanh,
    "sigmoid": lambda value: np.reciprocal(
        np.asarray(1, dtype=value.dtype) + np.exp(-value)
    ),
    "cos": np.cos,
    "sin": np.sin,
    "tan": np.tan,
    "exp2": np.exp2,
    "atan": np.arctan,
    "isnan": np.isnan,
    "logical_not": np.logical_not,
}

_CARRY_LITERALS: dict[str, Any] = {
    "NEG_INF": -np.inf,
    "POS_INF": np.inf,
    "ZERO": 0,
    "zeros": 0,
    "ONE": 1,
    "ones": 1,
    "false": False,
    "true": True,
}


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def requires_structured_reference(j: SemanticArchive) -> bool:
    """Return whether J contains a semantic loop or value scan."""

    return any(op.op in {"loop", "scan"} for op in _walk_ops(j.intent.body))


def _resolved_shape(shape: Sequence[Any], dims: Mapping[str, int]) -> tuple[int, ...]:
    result: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool):
            raise ValueError("structured reference shape contains a boolean extent")
        if isinstance(dimension, int):
            extent = dimension
        elif isinstance(dimension, str) and dimension in dims:
            extent = dims[dimension]
        else:
            raise StructuredReferenceUnsupportedError(
                f"structured reference cannot resolve shape extent {dimension!r}"
            )
        if extent < 0:
            raise ValueError("structured reference shape contains a negative extent")
        result.append(int(extent))
    return tuple(result)


class _Evaluator:
    def __init__(
        self,
        j: SemanticArchive,
        dims: Mapping[str, int],
        broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]],
    ) -> None:
        self.j = j
        self.dims = dict(dims)
        self.broadcast_axes = {
            result: {
                operand: tuple(int(axis) for axis in axes)
                for operand, axes in operands.items()
            }
            for result, operands in broadcast_axes.items()
        }
        self.value_types = j_elaboration(j).value_type_by_name
        self.operation_count = 0
        self.scan_count = 0
        self.loop_count = 0
        self.logical_loop_steps = 0
        self.zero_trip_loops = 0

    def _declared_type(
        self,
        name: str,
        op: IntentOp | None = None,
    ) -> tuple[str, tuple[int, ...]] | None:
        known = self.value_types.get(name)
        if known is not None:
            return known.dtype, _resolved_shape(known.shape, self.dims)
        if op is not None and isinstance(op.type, dict):
            dtype = op.type.get("dtype")
            shape = op.type.get("shape")
            if isinstance(dtype, str) and isinstance(shape, list):
                return dtype, _resolved_shape(shape, self.dims)
        return None

    def _validate_value(
        self,
        value: Any,
        *,
        name: str,
        declared: tuple[str, tuple[int, ...]] | None,
    ) -> np.ndarray:
        array = np.asarray(value)
        if declared is None:
            return array
        dtype, shape = declared
        expected_dtype = numpy_dtype(dtype)
        if array.shape != shape:
            raise ValueError(
                f"structured value {name!r} shape mismatch: expected {shape}, "
                f"got {array.shape}"
            )
        if array.dtype != expected_dtype:
            raise ValueError(
                f"structured value {name!r} dtype mismatch: expected "
                f"{expected_dtype}, got {array.dtype}"
            )
        return array

    def _result_name(self, op: IntentOp) -> str:
        if not isinstance(op.result, str) or not op.result:
            raise StructuredReferenceUnsupportedError(
                f"structured reference requires one result for op {op.op!r}"
            )
        return op.result

    def _operand(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
        index: int,
    ) -> np.ndarray:
        try:
            name = op.operands[index]
            value = env[name]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"structured op {op.op!r} has an unavailable operand"
            ) from exc
        result = self._result_name(op)
        axes = self.broadcast_axes.get(result, {}).get(name, ())
        projected = value
        for axis in sorted(axes):
            if axis < 0 or axis > projected.ndim:
                raise ValueError("structured broadcast axis is outside result rank")
            projected = np.expand_dims(projected, axis=axis)
        return projected

    def _reduce(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) - {"fn", "axes", "assoc", "keepdims"}:
            raise StructuredReferenceUnsupportedError(
                "structured reduce has unsupported control attributes"
            )
        value = self._operand(op, env, 0)
        raw_axes = op.attrs.get("axes")
        if not isinstance(raw_axes, list) or not raw_axes:
            raise StructuredReferenceUnsupportedError(
                "structured reduce requires explicit non-empty axes"
            )
        axes = tuple(int(axis) for axis in raw_axes)
        if any(axis < 0 or axis >= value.ndim for axis in axes):
            raise ValueError("structured reduce axis is outside operand rank")
        expected = self._declared_type(self._result_name(op), op)
        keepdims = bool(op.attrs.get("keepdims", False))
        if expected is not None:
            target_rank = len(expected[1])
            if target_rank == value.ndim:
                keepdims = True
            elif target_rank != value.ndim - len(set(axes)):
                raise ValueError("structured reduce result rank is not closed")
        fn = op.attrs.get("fn")
        if fn == "sum":
            return np.sum(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "prod":
            return np.prod(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "max":
            return np.max(value, axis=axes, keepdims=keepdims)
        if fn == "min":
            return np.min(value, axis=axes, keepdims=keepdims)
        if fn == "mean":
            if value.dtype.kind != "f":
                raise StructuredReferenceUnsupportedError(
                    "structured integer mean has no closed reference policy"
                )
            return np.mean(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "any":
            return np.any(value, axis=axes, keepdims=keepdims)
        if fn == "all":
            return np.all(value, axis=axes, keepdims=keepdims)
        raise StructuredReferenceUnsupportedError(
            f"structured reduce has unsupported fn {fn!r}"
        )

    def _scan(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"fn", "axis"} or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured scan requires exactly forward inclusive fn/axis control"
            )
        value = self._operand(op, env, 0)
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured scan axis must be a closed integer"
            )
        if axis < 0 or axis >= value.ndim:
            raise ValueError("structured scan axis is outside operand rank")
        self.scan_count += 1
        fn = op.attrs.get("fn")
        if fn == "sum":
            return np.cumsum(value, axis=axis, dtype=value.dtype)
        if fn == "max":
            return np.maximum.accumulate(value, axis=axis)
        if fn == "min":
            return np.minimum.accumulate(value, axis=axis)
        raise StructuredReferenceUnsupportedError(
            f"structured scan has unsupported fn {fn!r}"
        )

    def _ordinary(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if op.op not in _ORDINARY_OPS:
            raise StructuredReferenceUnsupportedError(
                f"structured reference has no denotation for IntentOp {op.op!r}"
            )
        if op.op == "const":
            if set(op.attrs) - {"value", "dtype", "shape"}:
                raise StructuredReferenceUnsupportedError(
                    "structured const has unsupported attributes"
                )
            result = self._result_name(op)
            declared = self._declared_type(result, op)
            if declared is None:
                raise StructuredReferenceUnsupportedError(
                    "structured const requires one closed result type"
                )
            dtype, shape = declared
            return np.full(shape, op.attrs.get("value"), dtype=numpy_dtype(dtype))
        if op.op == "map_binary":
            if set(op.attrs) != {"fn"} or len(op.operands) != 2:
                raise StructuredReferenceUnsupportedError(
                    "structured map_binary requires exactly fn and two operands"
                )
            fn = op.attrs.get("fn")
            implementation = _BINARY.get(str(fn))
            if implementation is None:
                raise StructuredReferenceUnsupportedError(
                    f"structured map_binary has unsupported fn {fn!r}"
                )
            return np.asarray(
                implementation(self._operand(op, env, 0), self._operand(op, env, 1))
            )
        if op.op == "map_unary":
            if set(op.attrs) != {"fn"} or len(op.operands) != 1:
                raise StructuredReferenceUnsupportedError(
                    "structured map_unary requires exactly fn and one operand"
                )
            fn = op.attrs.get("fn")
            implementation = _UNARY.get(str(fn))
            if implementation is None:
                raise StructuredReferenceUnsupportedError(
                    f"structured map_unary has unsupported fn {fn!r}"
                )
            return np.asarray(implementation(self._operand(op, env, 0)))
        if op.op == "reduce":
            return self._reduce(op, env)
        if op.op == "scan":
            return self._scan(op, env)
        if op.op == "contraction":
            if set(op.attrs) != {"subscripts"} or len(op.operands) != 2:
                raise StructuredReferenceUnsupportedError(
                    "structured contraction requires one subscripts relation"
                )
            return np.asarray(
                np.einsum(
                    str(op.attrs["subscripts"]),
                    self._operand(op, env, 0),
                    self._operand(op, env, 1),
                    optimize=False,
                )
            )
        if op.op == "select":
            if op.attrs or len(op.operands) != 3:
                raise StructuredReferenceUnsupportedError(
                    "structured select requires three operands and no control attrs"
                )
            condition = self._operand(op, env, 0)
            if condition.dtype != np.dtype(np.bool_):
                raise ValueError("structured select condition must have i1 dtype")
            return np.asarray(
                np.where(
                    condition,
                    self._operand(op, env, 1),
                    self._operand(op, env, 2),
                )
            )
        if op.op == "cast":
            if len(op.operands) != 1 or set(op.attrs) not in (
                {"to_dtype"},
                {"dtype"},
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured cast requires one canonical dtype target"
                )
            target = op.attrs.get("to_dtype", op.attrs.get("dtype"))
            if not isinstance(target, str):
                raise StructuredReferenceUnsupportedError(
                    "structured cast target must be a dtype string"
                )
            return self._operand(op, env, 0).astype(numpy_dtype(target))
        if op.op == "transpose":
            if len(op.operands) != 1 or set(op.attrs) != {"perm"}:
                raise StructuredReferenceUnsupportedError(
                    "structured transpose requires one closed permutation"
                )
            perm = op.attrs.get("perm")
            if not isinstance(perm, list) or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in perm
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured transpose permutation must be integers"
                )
            return np.transpose(self._operand(op, env, 0), axes=tuple(perm))
        if op.op == "pad":
            if len(op.operands) != 1 or set(op.attrs) != {"lo", "hi", "value"}:
                raise StructuredReferenceUnsupportedError(
                    "structured pad requires closed lo/hi/value attributes"
                )
            lo = op.attrs.get("lo")
            hi = op.attrs.get("hi")
            if (
                not isinstance(lo, list)
                or not isinstance(hi, list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in [*lo, *hi]
                )
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured reference currently requires literal pad extents"
                )
            return np.pad(
                self._operand(op, env, 0),
                tuple(zip(lo, hi, strict=True)),
                mode="constant",
                constant_values=op.attrs["value"],
            )
        raise AssertionError(op.op)  # pragma: no cover - closed above

    def _carry_type(self, state: LoopStateBinding) -> tuple[str, tuple[int, ...]]:
        if state.type is None:
            raise StructuredReferenceUnsupportedError(
                "structured loop carry requires one closed value type"
            )
        return state.type.dtype, _resolved_shape(state.type.shape, self.dims)

    def _carry_init(
        self,
        state: LoopStateBinding,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        declared = self._carry_type(state)
        dtype, shape = declared
        init = state.init_value
        if isinstance(init, str) and init in env:
            value = np.array(env[init], copy=True)
        else:
            literal = _CARRY_LITERALS.get(str(init), init)
            if isinstance(literal, str):
                try:
                    literal = float(literal)
                except ValueError as exc:
                    raise StructuredReferenceUnsupportedError(
                        f"structured carry has unknown initializer {init!r}"
                    ) from exc
            try:
                value = np.full(shape, literal, dtype=numpy_dtype(dtype))
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"structured carry initializer {init!r} violates {dtype}"
                ) from exc
        return self._validate_value(
            value,
            name=state.body_argument,
            declared=declared,
        )

    def _loop(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if op.operands or op.attrs or op.type is not None:
            raise StructuredReferenceUnsupportedError(
                "structured loop has unsupported ordinary-op fields"
            )
        if op.trip is None or not op.scan or not op.carry:
            raise StructuredReferenceUnsupportedError(
                "structured loop requires trip, scan inputs, and typed carries"
            )
        if any(child.op == "loop" for child in _walk_ops(op.body)):
            raise StructuredReferenceUnsupportedError(
                "structured reference does not implement nested loops"
            )
        if not isinstance(op.result, list) or len(op.result) != len(op.carry):
            raise ValueError("structured loop results differ from carry order")
        trip_axis = op.trip.axis
        if trip_axis not in self.dims:
            raise StructuredReferenceUnsupportedError(
                f"structured loop trip extent {trip_axis!r} is unresolved"
            )
        extent = self.dims[trip_axis]
        if extent < 0:
            raise ValueError("structured loop trip extent must be non-negative")

        carries = {
            state.body_argument: self._carry_init(state, env)
            for state in op.carry
        }
        scan_values: dict[str, np.ndarray] = {}
        for scan in op.scan:
            try:
                source = env[scan.source]
            except KeyError as exc:
                raise ValueError(
                    f"structured loop scan source {scan.source!r} is unavailable"
                ) from exc
            if scan.axis < 0 or scan.axis >= source.ndim:
                raise ValueError("structured loop scan axis is outside source rank")
            if source.shape[scan.axis] != extent:
                raise ValueError(
                    "structured loop scan extent differs from semantic trip axis"
                )
            # No C is consulted: the complete logical axis is the one semantic
            # slice.  This is conservative and value-deterministic.
            scan_values[scan.name] = source

        self.loop_count += 1
        if extent == 0:
            self.zero_trip_loops += 1
            final_carries = carries
        else:
            local = dict(env)
            local.update(scan_values)
            local.update(carries)
            local = self.sequence(op.body, local, allow_loop=False)
            final_carries = {}
            for state in op.carry:
                if state.next_value not in local:
                    raise ValueError(
                        f"structured loop body did not define {state.next_value!r}"
                    )
                final_carries[state.body_argument] = self._validate_value(
                    local[state.next_value],
                    name=state.next_value,
                    declared=self._carry_type(state),
                )
            self.logical_loop_steps += 1

        return {
            state.result_value: self._validate_value(
                final_carries[state.body_argument],
                name=state.result_value,
                declared=self._carry_type(state),
            )
            for state in op.carry
        }

    def sequence(
        self,
        body: Sequence[IntentOp],
        env: Mapping[str, np.ndarray],
        *,
        allow_loop: bool,
    ) -> dict[str, np.ndarray]:
        result = dict(env)
        for op in body:
            self.operation_count += 1
            if op.op == "loop":
                if not allow_loop:
                    raise StructuredReferenceUnsupportedError(
                        "structured reference does not implement nested loops"
                    )
                result.update(self._loop(op, result))
                continue
            name = self._result_name(op)
            value = self._ordinary(op, result)
            result[name] = self._validate_value(
                value,
                name=name,
                declared=self._declared_type(name, op),
            )
        return result


def execute_structured_reference(
    j: SemanticArchive,
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> CPUExecutionResult:
    """Execute the restricted typed denotation of one canonical J.

    The signature intentionally excludes C and pi.  In particular,
    ``pi.loop_carriers`` cannot affect initialization, recurrence, or results.
    """

    j = elaborate_j(j)
    if not requires_structured_reference(j):
        raise StructuredReferenceUnsupportedError(
            "structured reference requires at least one loop or scan"
        )
    if len(inputs) != len(j.intent.inputs):
        raise ValueError("structured reference input arity differs from J")
    checked_dims = {
        name: int(extent)
        for name, extent in dims.items()
        if isinstance(name, str)
        and isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
    }
    if checked_dims != dict(dims):
        raise ValueError(
            "structured reference dims must be non-negative integer bindings"
        )
    checked_inputs = {
        tensor.name: validate_array(
            np.asarray(value),
            shape=_resolved_shape(tensor.shape, checked_dims),
            dtype=tensor.dtype,
            context=f"structured input {tensor.name!r}",
        )
        for tensor, value in zip(j.intent.inputs, inputs, strict=True)
    }
    validate_value_dimension_instantiation(j, checked_dims, checked_inputs)
    evaluator = _Evaluator(j, checked_dims, broadcast_axes or {})
    env = evaluator.sequence(j.intent.body, checked_inputs, allow_loop=True)
    returns = {item.output: item.value for item in j.intent.returns}
    outputs: dict[str, np.ndarray] = {}
    for tensor in j.intent.outputs:
        value_ref = returns.get(tensor.name, tensor.name)
        if value_ref not in env:
            raise ValueError(
                f"structured reference did not produce output {tensor.name!r}"
            )
        outputs[tensor.name] = evaluator._validate_value(
            env[value_ref],
            name=tensor.name,
            declared=(
                tensor.dtype,
                _resolved_shape(tensor.shape, checked_dims),
            ),
        )
    return CPUExecutionResult(
        outputs,
        {
            "protocol": STRUCTURED_CPU_REFERENCE_PROTOCOL,
            "authority": "canonical_J_only",
            "execution_mode": "full_logical_axis_conservative",
            "operation_count": evaluator.operation_count,
            "scan_count": evaluator.scan_count,
            "loop_count": evaluator.loop_count,
            "logical_loop_step_count": evaluator.logical_loop_steps,
            "zero_trip_loop_count": evaluator.zero_trip_loops,
            "input_count": len(checked_inputs),
            "output_count": len(outputs),
        },
    )


__all__ = [
    "STRUCTURED_CPU_REFERENCE_PROTOCOL",
    "StructuredReferenceUnsupportedError",
    "execute_structured_reference",
    "requires_structured_reference",
]
