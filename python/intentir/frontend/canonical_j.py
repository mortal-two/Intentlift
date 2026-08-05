"""Shared deterministic ``J -> intent MLIR`` materialization entry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.model import IntentFunction, IntentOp, IntentReturn
from intentir.ir.value_dimension import resolve_value_dimension_equalities

from .emit_mlir import BroadcastAxes, IndexBoundsProjection, emit_module

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from intentir.verify.gate1.model import SemanticArchive


FLAT_MATERIALIZATION = "intentir.materialize.flat-graph.v1"
LOGICAL_FULL_AXIS_STEP_MATERIALIZATION = (
    "intentir.materialize.logical-full-axis-step.v1"
)
STRUCTURED_SCAN_LOOP_MATERIALIZATION = (
    "intentir.materialize.structured-scan-loop.v1"
)
_MAX_LOGICAL_SCAN_TEMPORARY_BYTES = 128 * 1024 * 1024
_DTYPE_BYTES = {
    "i1": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}


@dataclass(frozen=True)
class JModuleMaterialization:
    """One deterministic Intent-MLIR encoding and its registered form."""

    mlir: str
    form: str
    consumed_structures: tuple[str, ...] = ()


@dataclass(frozen=True)
class JFunctionMaterialization:
    """One registered target-neutral function view of canonical J."""

    function: IntentFunction
    form: str
    consumed_structures: tuple[str, ...] = ()
    loop_schedule_tile: int | None = None


def _logical_scan_tile(
    j: SemanticArchive,
    loop: IntentOp,
    dims: Mapping[str, int],
) -> int:
    """Mirror the certified J-only bounded scan denotation for target MLIR."""

    if loop.trip is None or loop.trip.axis not in dims:
        raise ValueError("structured scan loop requires one resolved trip axis")
    trip_axis = loop.trip.axis
    extent = dims[trip_axis]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
        return extent
    if any(
        state.type is not None and trip_axis in state.type.shape
        for state in loop.carry
    ):
        return extent

    value_types: dict[str, tuple[str, list[object]]] = {
        tensor.name: (tensor.dtype, list(tensor.shape))
        for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    for scan in loop.scan:
        source = value_types.get(scan.source)
        if source is not None:
            value_types[scan.name] = source
    for operation in _walk(loop.body):
        results = (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if operation.type is None:
            continue
        for result in results:
            if isinstance(result, str):
                value_types[result] = (
                    operation.type["dtype"],
                    list(operation.type.get("shape", [])),
                )

    candidate_names = {scan.name for scan in loop.scan}
    candidate_names.update(
        result
        for operation in _walk(loop.body)
        for result in (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if isinstance(result, str)
    )
    tile = extent
    for name in candidate_names:
        value_type = value_types.get(name)
        if value_type is None:
            continue
        dtype, shape = value_type
        if trip_axis not in shape:
            continue
        if shape.count(trip_axis) != 1 or dtype not in _DTYPE_BYTES:
            return extent
        elements_per_trip = 1
        for dimension in shape:
            if dimension == trip_axis:
                continue
            if isinstance(dimension, bool):
                raise ValueError("structured loop type contains a boolean extent")
            if isinstance(dimension, int):
                dimension_extent = dimension
            elif isinstance(dimension, str) and dimension in dims:
                dimension_extent = dims[dimension]
            else:
                raise ValueError("structured loop type contains an unresolved extent")
            elements_per_trip *= int(dimension_extent)
        bytes_per_trip = max(1, elements_per_trip * _DTYPE_BYTES[dtype])
        tile = min(
            tile,
            max(1, _MAX_LOGICAL_SCAN_TEMPORARY_BYTES // bytes_per_trip),
        )
    return max(1, tile)


def _walk(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk(op.body))
        for arm in (op.positive, op.negative):
            if arm is not None:
                result.extend(_walk(arm.body))
    return tuple(result)


def _canonical_return_function(
    j: SemanticArchive,
) -> tuple[IntentFunction, tuple[str, ...]]:
    """Project the one elaborated output binding into the MLIR printer view."""

    bindings = j_elaboration(j).output_bindings
    inputs = {item.name: item for item in j.intent.inputs}
    outputs = {item.name: item for item in j.intent.outputs}
    body = list(j.intent.body)
    returns: list[IntentReturn] = []
    consumed: list[str] = []
    used_names = {
        *inputs,
        *outputs,
        *(
            result
            for operation in _walk(body)
            for result in (
                operation.result
                if isinstance(operation.result, list)
                else [operation.result]
            )
            if isinstance(result, str)
        ),
    }
    for binding in bindings:
        value = binding.value
        if value in inputs:
            output = outputs[binding.output]
            if (
                inputs[value].dtype != output.dtype
                or inputs[value].shape != output.shape
            ):
                raise ValueError(
                    "direct input/output binding requires identical logical types"
                )
            base = f"__intentir_identity_{binding.output}"
            result = base
            suffix = 0
            while result in used_names:
                suffix += 1
                result = f"{base}_{suffix}"
            used_names.add(result)
            body.append(
                IntentOp(
                    op="map_unary",
                    result=result,
                    operands=[value],
                    attrs={"fn": "identity"},
                    type={"dtype": output.dtype, "shape": list(output.shape)},
                )
            )
            returns.append(IntentReturn(binding.output, result))
            consumed.append(
                f"output_binding:{binding.output}:{value}:identity_materialized"
            )
        else:
            returns.append(IntentReturn(binding.output, value))
    return IntentFunction(
        name=j.intent.name,
        inputs=list(j.intent.inputs),
        outputs=list(j.intent.outputs),
        body=body,
        returns=returns,
    ), tuple(consumed)


def _logical_full_axis_step_function(
    j: SemanticArchive,
    dims: Mapping[str, int],
) -> IntentFunction | None:
    """Inline the closed full-axis-step denotation of one logical loop.

    Canonical structured-reference semantics evaluates a logical loop once over
    the complete logical tensor domain when its trip extent is positive.  Scan
    names therefore denote their complete source tensors in this registered
    representation; they are not implementation tiles.  Explicit carry
    initializers become either aliases of dominating J values or typed tensor
    constants, while a null-initialized carry remains legal only when the body
    does not read it.  This deterministic projection uses J alone: it does not
    consult C, a kernel name, or source realization details.

    Other loop forms remain unsupported until their complete J denotation can be
    represented without inventing an initializer or schedule.
    """

    loop_indices = [
        index for index, operation in enumerate(j.intent.body) if operation.op == "loop"
    ]
    if len(loop_indices) != 1:
        return None
    loop_index = loop_indices[0]
    outer_prefix = j.intent.body[:loop_index]
    outer_suffix = j.intent.body[loop_index + 1 :]
    if any(
        operation.op == "loop"
        for operation in _walk([*outer_prefix, *outer_suffix])
    ):
        return None
    loop = j.intent.body[loop_index]
    if loop.trip is None or loop.trip.axis not in dims:
        raise ValueError("materialize J loop requires a resolved logical trip extent")
    extent = dims[loop.trip.axis]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
        raise ValueError(
            "null-initialized logical full-axis loop requires a positive extent"
        )
    if not loop.carry:
        return None
    if any(operation.op == "loop" for operation in _walk(loop.body)):
        return None

    inputs = {item.name: item for item in j.intent.inputs}
    aliases: dict[str, str] = {}
    for scan in loop.scan:
        source = inputs.get(scan.source)
        if source is None:
            # A top-level single-loop form has no other dominating producer.
            return None
        source_shape = [
            dims[item] if isinstance(item, str) and item in dims else item
            for item in source.shape
        ]
        if (
            scan.axis < 0
            or scan.axis >= len(source_shape)
            or source_shape[scan.axis] != extent
        ):
            raise ValueError(
                "logical loop scan extent differs from its semantic trip axis"
            )
        aliases[scan.name] = scan.source

    body_references = {
        operand
        for operation in _walk(loop.body)
        for operand in operation.operands
    }
    dominating_types: dict[str, tuple[str, list[object]]] = {
        name: (tensor.dtype, list(tensor.shape)) for name, tensor in inputs.items()
    }
    for operation in _walk(outer_prefix):
        results = (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if any(isinstance(result, str) for result in results):
            if operation.type is None or len(results) != 1:
                return None
            result = results[0]
            if not isinstance(result, str):
                return None
            dominating_types[result] = (
                operation.type["dtype"],
                list(operation.type.get("shape", [])),
            )

    carry_prefix: list[IntentOp] = []
    for state in loop.carry:
        if state.type is None:
            return None
        init = state.init_value
        if init is None:
            if state.body_argument in body_references:
                return None
            continue
        if isinstance(init, str) and init in dominating_types:
            init_dtype, init_shape = dominating_types[init]
            if (
                init_dtype != state.type.dtype
                or init_shape != state.type.shape
            ):
                raise ValueError(
                    "logical loop carry initializer type differs from its state type"
                )
            aliases[state.body_argument] = init
            continue
        if isinstance(init, bool) or isinstance(init, (int, float)):
            carry_prefix.append(
                IntentOp(
                    op="const",
                    result=state.body_argument,
                    attrs={
                        "value": init,
                        "dtype": state.type.dtype,
                        "shape": list(state.type.shape),
                    },
                    type=state.type.to_json(),
                )
            )
            continue
        return None

    def resolve(name: str) -> str:
        visited: set[str] = set()
        while name in aliases:
            if name in visited:
                raise ValueError("logical loop alias cycle")
            visited.add(name)
            name = aliases[name]
        return name

    def substitute_body(body: Sequence[IntentOp]) -> list[IntentOp]:
        result: list[IntentOp] = []
        for original in body:
            operation = copy.deepcopy(original)
            operation.operands = [resolve(name) for name in operation.operands]
            if operation.predicate is not None:
                operation.predicate = resolve(operation.predicate)
            operation.body = substitute_body(operation.body)
            for arm in (operation.positive, operation.negative):
                if arm is not None:
                    arm.body = substitute_body(arm.body)
                    if hasattr(arm, "yield_value"):
                        arm.yield_value = resolve(arm.yield_value)
            result.append(operation)
        return result

    body = [
        *substitute_body(outer_prefix),
        *carry_prefix,
        *substitute_body(loop.body),
    ]
    result_to_next = {
        item.result_value: resolve(item.next_value) for item in loop.carry
    }
    if not isinstance(loop.result, list) or set(loop.result) != set(result_to_next):
        raise ValueError("logical loop results differ from its carry bindings")
    produced = {
        name
        for operation in _walk(body)
        for name in (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if isinstance(name, str)
    }
    missing_next = sorted(set(result_to_next.values()) - produced)
    if missing_next:
        raise ValueError(
            "logical loop body does not define next carry value(s): "
            + ", ".join(missing_next)
        )
    aliases.update(result_to_next)
    body.extend(substitute_body(outer_suffix))
    produced = {
        name
        for operation in _walk(body)
        for name in (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if isinstance(name, str)
    }
    returns: list[IntentReturn] = []
    for binding in j_elaboration(j).output_bindings:
        value = resolve(binding.value)
        if value not in produced:
            raise ValueError(
                "logical loop output binding does not name a materialized value"
            )
        returns.append(IntentReturn(binding.output, value))
    return IntentFunction(
        name=j.intent.name,
        inputs=list(j.intent.inputs),
        outputs=list(j.intent.outputs),
        body=body,
        returns=returns,
    )


def materialize_j_function(
    j: SemanticArchive,
    dims: Mapping[str, int],
) -> JFunctionMaterialization:
    """Choose the unique registered logical function form using J alone."""

    j = elaborate_j(j)
    if any(op.op == "loop" for op in j.intent.body):
        loops = [op for op in j.intent.body if op.op == "loop"]
        if len(loops) != 1:
            raise NotImplementedError(
                "canonical materializer requires exactly one top-level loop"
            )
        tile = _logical_scan_tile(j, loops[0], dims)
        extent = dims[loops[0].trip.axis] if loops[0].trip is not None else 0
        if 0 < tile < extent:
            if extent % tile != 0:
                raise NotImplementedError(
                    "structured scan materialization does not yet support a tail tile"
                )
            function = IntentFunction(
                name=j.intent.name,
                inputs=list(j.intent.inputs),
                outputs=list(j.intent.outputs),
                body=copy.deepcopy(j.intent.body),
                returns=[
                    IntentReturn(binding.output, binding.value)
                    for binding in j_elaboration(j).output_bindings
                ],
            )
            form = STRUCTURED_SCAN_LOOP_MATERIALIZATION
            consumed = tuple(
                f"loop:{op.node_id or 'anonymous'}:structured_scan_tile:{tile}"
                for op in loops
            )
            return JFunctionMaterialization(
                function,
                form,
                consumed,
                loop_schedule_tile=tile,
            )
        function = _logical_full_axis_step_function(j, dims)
        if function is None:
            raise NotImplementedError(
                "canonical materializer does not yet encode this structured loop form"
            )
        form = LOGICAL_FULL_AXIS_STEP_MATERIALIZATION
        consumed = tuple(
            f"loop:{op.node_id or 'anonymous'}:logical_full_axis_step"
            for op in loops
        )
    else:
        function, consumed = _canonical_return_function(j)
        form = FLAT_MATERIALIZATION
    return JFunctionMaterialization(function, form, consumed)


def materialize_j_module(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> JModuleMaterialization:
    """Materialize canonical J without reading C or choosing target schedules."""

    j = elaborate_j(j)
    resolved_dims = resolve_dimension_relations(j, dims)
    value_dimension_equalities = resolve_value_dimension_equalities(
        j, resolved_dims
    )
    logical = materialize_j_function(j, resolved_dims)
    mlir = emit_module(
        logical.function,
        resolved_dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        value_dimension_equalities=value_dimension_equalities,
        loop_schedule_tile=logical.loop_schedule_tile,
    )
    return JModuleMaterialization(
        mlir,
        logical.form,
        logical.consumed_structures,
    )


def emit_j_module(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> str:
    """Emit MLIR only after the unique deterministic J elaboration boundary."""

    return materialize_j_module(
        j,
        dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    ).mlir


__all__ = [
    "FLAT_MATERIALIZATION",
    "JFunctionMaterialization",
    "JModuleMaterialization",
    "LOGICAL_FULL_AXIS_STEP_MATERIALIZATION",
    "STRUCTURED_SCAN_LOOP_MATERIALIZATION",
    "emit_j_module",
    "materialize_j_function",
    "materialize_j_module",
]
