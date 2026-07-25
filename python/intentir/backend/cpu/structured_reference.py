"""Restricted J-only NumPy denotation for structured Layer-A value semantics.

This is a Gate2 reference executor, not the production CPU backend and not a
fallback for arbitrary IntentOps.  It accepts only a closed operation table,
executes each structured loop over its complete logical trip axis as one
conservative semantic step, and validates every declared input, carry, result,
and output type.  Its API deliberately has no C or pi parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from intentir.backend.cpu.abi import (
    CPUBackendUnsupportedError,
    mlir_element_type,
    numpy_dtype,
    validate_array,
)
from intentir.backend.cpu.pipeline import CPUExecutionResult
from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.model import IntentOp, LoopStateBinding

if TYPE_CHECKING:
    from intentir.verify.gate1.model import SemanticArchive


STRUCTURED_CPU_REFERENCE_PROTOCOL = "intentir.j-structured-cpu-reference.v6"


class StructuredReferenceUnsupportedError(CPUBackendUnsupportedError):
    """The typed J lies outside this deliberately narrow denotation."""


_ORDINARY_OPS = frozenset(
    {
        "cast",
        "concat",
        "const",
        "contraction",
        "gather",
        "iota",
        "map_binary",
        "map_unary",
        "pad",
        "reduce",
        "scan",
        "select",
        "tile",
        "transpose",
    }
)

_FLOAT_ELEMENT_TYPES = frozenset({"f16", "bf16", "f32", "f64"})
_INTEGER_ELEMENT_TYPES = frozenset({"i1", "i8", "i16", "i32", "i64"})
_EINSUM_TERM = re.compile(r"^[A-Za-z]*$")

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
    "and": np.logical_and,
    "or": np.logical_or,
    "xor": np.logical_xor,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
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


def _element_class(dtype: str) -> str | None:
    """Mirror the element classes consumed by ContractionLowering."""

    try:
        element_type = mlir_element_type(dtype)
    except CPUBackendUnsupportedError:
        return None
    if element_type in _FLOAT_ELEMENT_TYPES:
        return "float"
    if element_type in _INTEGER_ELEMENT_TYPES:
        return "integer"
    return None


def _scalar_cpu_lowers_contraction(
    op: IntentOp,
    value_types: Mapping[str, Any],
) -> bool:
    """Return the exact typed subset eliminated by ContractionLowering.

    This mirrors ``ConvertIntentToLinalg`` rather than predicting whether a
    target compiler might happen to accept an unlowered ``intent`` op.  The
    common lowering accepts arbitrary widths within the floating class, but it
    rejects mixed integer/float operands and mixed-width integer contractions.
    Malformed or incompletely typed contractions are conservatively outside
    the scalar capability and will fail closed in the structured evaluator.
    """

    if (
        set(op.attrs) != {"subscripts"}
        or len(op.operands) != 2
        or not isinstance(op.result, str)
    ):
        return False
    operand_types = [value_types.get(name) for name in op.operands]
    result_type = value_types.get(op.result)
    if result_type is None or any(value is None for value in operand_types):
        return False
    operand_dtypes = [str(value.dtype) for value in operand_types]
    result_dtype = str(result_type.dtype)
    try:
        operand_element_types = [
            mlir_element_type(dtype) for dtype in operand_dtypes
        ]
        result_element_type = mlir_element_type(result_dtype)
    except CPUBackendUnsupportedError:
        return False
    operand_classes = [_element_class(dtype) for dtype in operand_dtypes]
    result_class = _element_class(result_dtype)
    if result_class is None or any(kind is None for kind in operand_classes):
        return False
    if any(kind != result_class for kind in operand_classes):
        return False
    if result_class == "integer" and any(
        element_type != result_element_type
        for element_type in operand_element_types
    ):
        return False
    return True


def requires_structured_reference(j: SemanticArchive) -> bool:
    """Return whether J needs the J-only typed reference denotation.

    ``tile`` selects this path because the scalar Intent-to-Linalg executor does
    not lower it.  A typed contraction selects the same path only when the
    common scalar lowering cannot eliminate that contraction.  The structured
    evaluator consequently owns a closed ordinary op table as well as
    loops/scans; it never falls back into a second consumer halfway through a
    graph.
    """

    accepted = elaborate_j(j)
    operations = _walk_ops(accepted.intent.body)
    if any(op.op in {"loop", "scan", "tile"} for op in operations):
        return True
    value_types = j_elaboration(accepted).value_type_by_name
    return any(
        op.op == "contraction"
        and not _scalar_cpu_lowers_contraction(op, value_types)
        for op in operations
    )


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
        access_guard_relations: Sequence[Any] = (),
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
        self.access_guard_loads: dict[tuple[str, int], Any] = {}
        for relation in access_guard_relations:
            if getattr(relation, "access_kind", None) != "load":
                continue
            node_ref = getattr(relation, "node_ref", None)
            operand_index = getattr(relation, "operand_index", None)
            tensor = getattr(relation, "tensor", None)
            if (
                not isinstance(node_ref, str)
                or isinstance(operand_index, bool)
                or not isinstance(operand_index, int)
                or operand_index < 0
                or not isinstance(tensor, str)
            ):
                raise ValueError(
                    "structured guarded-load receipt is not mechanically closed"
                )
            key = (node_ref, operand_index)
            if key in self.access_guard_loads:
                raise ValueError(
                    "structured guarded-load receipts duplicate one operand edge"
                )
            self.access_guard_loads[key] = relation
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
        materialization = self.access_guard_loads.get((op.node_id, index))
        if materialization is not None:
            if name != materialization.tensor:
                raise ValueError(
                    "structured guarded-load receipt names a different operand"
                )
            try:
                condition = np.asarray(env[materialization.value_ref])
            except KeyError as exc:
                raise ValueError(
                    "structured guarded-load guard does not dominate its consumer"
                ) from exc
            if condition.dtype != np.dtype(np.bool_):
                raise ValueError("structured guarded-load guard must have dtype i1")
            for axis in sorted(materialization.axes):
                if axis < 0 or axis > condition.ndim:
                    raise ValueError(
                        "structured guarded-load broadcast axis is outside target rank"
                    )
                condition = np.expand_dims(condition, axis=axis)
            if materialization.polarity == "positive":
                active = condition
            elif materialization.polarity == "negative":
                active = np.logical_not(condition)
            else:
                raise ValueError("structured guarded-load polarity is unknown")
            try:
                fill = np.asarray(
                    materialization.fill_value,
                    dtype=value.dtype,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise StructuredReferenceUnsupportedError(
                    "structured guarded-load fill has no typed NumPy denotation"
                ) from exc
            value = np.asarray(np.where(active, value, fill))
            if value.shape != env[name].shape or value.dtype != env[name].dtype:
                raise ValueError(
                    "structured guarded-load materialization changed operand type"
                )
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

    def _concat(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"axis"} or not op.operands:
            raise StructuredReferenceUnsupportedError(
                "structured concat requires one axis and at least one operand"
            )
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured concat axis must be a closed integer"
            )
        values = [self._operand(op, env, index) for index in range(len(op.operands))]
        rank = values[0].ndim
        if axis < 0 or axis >= rank:
            raise ValueError("structured concat axis is outside operand rank")
        if any(value.ndim != rank for value in values[1:]):
            raise ValueError("structured concat operands must have one common rank")
        if any(value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError("structured concat operands must have one common dtype")
        for dimension in range(rank):
            if dimension == axis:
                continue
            if any(
                value.shape[dimension] != values[0].shape[dimension]
                for value in values[1:]
            ):
                raise ValueError(
                    "structured concat operands disagree at a non-concat dimension"
                )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured concat requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        inferred_shape = list(values[0].shape)
        inferred_shape[axis] = sum(value.shape[axis] for value in values)
        if expected_shape != tuple(inferred_shape):
            raise ValueError("structured concat result shape is not the axis sum")
        if numpy_dtype(expected_dtype) != values[0].dtype:
            raise ValueError("structured concat result dtype differs from its inputs")
        return np.concatenate(values, axis=axis)

    def _gather(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"axis", "mode"} or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured gather requires exactly axis/mode and two operands"
            )
        data = self._operand(op, env, 0)
        index = self._operand(op, env, 1)
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured gather axis must be a closed integer"
            )
        if axis < 0 or axis >= data.ndim:
            raise ValueError("structured gather axis is outside data rank")
        if index.dtype.kind not in {"i", "u"}:
            raise ValueError("structured gather index must have an integer dtype")
        if index.size and (
            bool(np.any(index < 0))
            or bool(np.any(index >= data.shape[axis]))
        ):
            raise ValueError("structured gather index is outside the data extent")
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured gather requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        if numpy_dtype(expected_dtype) != data.dtype:
            raise ValueError("structured gather result dtype differs from data")
        mode = op.attrs.get("mode")
        if mode == "along_axis":
            if index.ndim != data.ndim:
                raise ValueError(
                    "structured along_axis gather index rank must equal data rank"
                )
            if any(
                index.shape[dimension] > data.shape[dimension]
                for dimension in range(data.ndim)
                if dimension != axis
            ):
                raise ValueError(
                    "structured along_axis gather index exceeds a non-axis extent"
                )
            inferred_shape = index.shape
            if expected_shape != inferred_shape:
                raise ValueError(
                    "structured along_axis gather result must be shaped like index"
                )
            return np.take_along_axis(data, index, axis=axis)
        if mode == "take":
            inferred_shape = (
                *data.shape[:axis],
                *index.shape,
                *data.shape[axis + 1 :],
            )
            if expected_shape != inferred_shape:
                raise ValueError(
                    "structured take gather result shape differs from data/index"
                )
            return np.take(data, index, axis=axis)
        raise StructuredReferenceUnsupportedError(
            f"structured gather has unsupported mode {mode!r}"
        )

    def _tile(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"reps"} or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured tile requires exactly reps and one operand"
            )
        value = self._operand(op, env, 0)
        reps = op.attrs.get("reps")
        if not isinstance(reps, list) or any(
            isinstance(rep, bool) or not isinstance(rep, int) for rep in reps
        ):
            raise StructuredReferenceUnsupportedError(
                "structured tile reps must be closed integers"
            )
        if any(rep < 0 for rep in reps):
            raise ValueError("structured tile reps must be non-negative")
        if len(reps) != value.ndim:
            raise ValueError(
                "structured tile requires one repetition for every input dimension"
            )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured tile requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        inferred_shape = tuple(
            extent * rep for extent, rep in zip(value.shape, reps, strict=True)
        )
        if len(expected_shape) != value.ndim or expected_shape != inferred_shape:
            raise ValueError(
                "structured tile result shape must equal input shape times reps"
            )
        if numpy_dtype(expected_dtype) != value.dtype:
            raise ValueError("structured tile result dtype differs from its input")
        return np.tile(value, tuple(reps))

    def _iota(self, op: IntentOp) -> np.ndarray:
        if op.operands or set(op.attrs) not in ({"n"}, {"n", "dtype"}):
            raise StructuredReferenceUnsupportedError(
                "structured iota requires no operands and one closed extent"
            )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured iota requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        raw_extent = op.attrs.get("n")
        if isinstance(raw_extent, bool):
            raise StructuredReferenceUnsupportedError(
                "structured iota extent must be a closed integer or dimension"
            )
        if isinstance(raw_extent, int):
            extent = raw_extent
        elif isinstance(raw_extent, str) and raw_extent in self.dims:
            extent = self.dims[raw_extent]
        else:
            raise StructuredReferenceUnsupportedError(
                "structured iota extent must be a closed integer or dimension"
            )
        if extent < 0:
            raise ValueError("structured iota extent must be non-negative")
        if expected_shape != (extent,):
            raise ValueError(
                "structured iota result must be rank one with its declared extent"
            )
        attribute_dtype = op.attrs.get("dtype", expected_dtype)
        if not isinstance(attribute_dtype, str):
            raise StructuredReferenceUnsupportedError(
                "structured iota dtype must be a closed string"
            )
        expected_numpy_dtype = numpy_dtype(expected_dtype)
        if expected_numpy_dtype.kind not in {"i", "u"}:
            raise StructuredReferenceUnsupportedError(
                "structured iota result dtype must be a non-boolean integer"
            )
        if numpy_dtype(attribute_dtype) != expected_numpy_dtype:
            raise ValueError(
                "structured iota result dtype differs from its declared dtype"
            )
        return np.arange(extent, dtype=expected_numpy_dtype)

    def _contraction(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"subscripts"} or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires exactly two operands and one "
                "subscripts relation"
            )
        result = self._result_name(op)
        expected = self._declared_type(result, op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires one closed declared result type"
            )
        expected_dtype, expected_shape = expected
        subscripts = op.attrs.get("subscripts")
        if not isinstance(subscripts, str):
            raise StructuredReferenceUnsupportedError(
                "structured contraction subscripts must be a closed string"
            )
        compact = "".join(subscripts.split())
        parts = compact.split("->")
        input_terms = parts[0].split(",") if len(parts) == 2 else []
        if (
            len(parts) != 2
            or len(input_terms) != 2
            or not all(_EINSUM_TERM.fullmatch(term) for term in input_terms)
            or _EINSUM_TERM.fullmatch(parts[-1]) is None
        ):
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires the closed binary einsum grammar"
            )
        lhs_labels, rhs_labels = input_terms
        output_labels = parts[1]
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("structured contraction output labels must be unique")
        if not set(output_labels).issubset(set(lhs_labels) | set(rhs_labels)):
            raise ValueError(
                "structured contraction output labels must occur in an input"
            )

        operands = [self._operand(op, env, index) for index in range(2)]
        label_extents: dict[str, int] = {}
        for index, (labels, value) in enumerate(
            zip((lhs_labels, rhs_labels), operands, strict=True)
        ):
            if len(labels) != value.ndim:
                raise ValueError(
                    f"structured contraction operand {index} labels differ from "
                    "its declared rank"
                )
            for label, extent in zip(labels, value.shape, strict=True):
                previous = label_extents.get(label)
                if previous is not None and previous != extent:
                    raise ValueError(
                        f"structured contraction label {label!r} has "
                        "inconsistent extents"
                    )
                label_extents[label] = extent
        inferred_shape = tuple(label_extents[label] for label in output_labels)
        if inferred_shape != expected_shape:
            raise ValueError(
                "structured contraction declared result shape differs from "
                "its subscripts"
            )

        result_dtype = numpy_dtype(expected_dtype)
        operand_kinds = {value.dtype.kind for value in operands}
        if result_dtype.kind == "f":
            if not operand_kinds.issubset({"b", "i", "u", "f"}):
                raise StructuredReferenceUnsupportedError(
                    "structured floating contraction requires numeric operands"
                )
        elif result_dtype.kind in {"i", "u"}:
            if not operand_kinds.issubset({"b", "i", "u"}):
                raise StructuredReferenceUnsupportedError(
                    "structured integer contraction cannot consume floating operands"
                )
        else:
            raise StructuredReferenceUnsupportedError(
                "structured contraction result dtype must be numeric"
            )
        cast_operands = [
            value.astype(result_dtype, copy=False) for value in operands
        ]
        try:
            value = np.einsum(
                compact,
                cast_operands[0],
                cast_operands[1],
                optimize=False,
            )
        except (TypeError, ValueError) as exc:
            raise StructuredReferenceUnsupportedError(
                "structured contraction dtype is unsupported by the closed "
                "NumPy denotation"
            ) from exc
        value = np.asarray(value)
        if value.dtype != result_dtype:
            raise StructuredReferenceUnsupportedError(
                "structured contraction did not preserve its declared result dtype"
            )
        return value

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
        if op.op == "concat":
            return self._concat(op, env)
        if op.op == "gather":
            return self._gather(op, env)
        if op.op == "iota":
            return self._iota(op)
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
            return self._contraction(op, env)
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
        if op.op == "tile":
            return self._tile(op, env)
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
        if op.trip is None or not op.carry:
            raise StructuredReferenceUnsupportedError(
                "structured loop requires one trip and typed carries"
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
            if state.init_value is not None
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
            if len(carries) != len(op.carry):
                raise StructuredReferenceUnsupportedError(
                    "zero-trip structured loop cannot return a null-initialized carry"
                )
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
    access_guard_relations: Sequence[Any] = (),
) -> CPUExecutionResult:
    """Execute the restricted typed denotation of one canonical J.

    The signature intentionally excludes C and pi.  In particular,
    ``pi.loop_carriers`` cannot affect initialization, recurrence, or results.
    """

    j = elaborate_j(j)
    guarded_loads = tuple(
        relation
        for relation in access_guard_relations
        if getattr(relation, "access_kind", None) == "load"
    )
    if not requires_structured_reference(j) and not guarded_loads:
        raise StructuredReferenceUnsupportedError(
            "structured reference requires a structured op or a contraction "
            "outside the scalar CPU capability"
        )
    unsupported = sorted(
        {op.op for op in _walk_ops(j.intent.body)} - (_ORDINARY_OPS | {"loop"})
    )
    if unsupported:
        raise StructuredReferenceUnsupportedError(
            "structured reference graph contains IntentOp(s) outside its "
            "closed operation table: "
            + ", ".join(unsupported)
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
    evaluator = _Evaluator(
        j,
        checked_dims,
        broadcast_axes or {},
        guarded_loads,
    )
    env = evaluator.sequence(j.intent.body, checked_inputs, allow_loop=True)
    output_values = {
        item.output: item.value
        for item in j_elaboration(j).output_bindings
    }
    outputs: dict[str, np.ndarray] = {}
    for tensor in j.intent.outputs:
        value_ref = output_values[tensor.name]
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
            "access_guard_load_materialization_count": len(
                evaluator.access_guard_loads
            ),
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
