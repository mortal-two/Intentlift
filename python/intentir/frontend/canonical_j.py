"""Shared deterministic canonical-J materialization entries."""

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


@dataclass(frozen=True)
class JFunctionMaterialization:
    """One registered target-neutral function view of canonical J."""

    function: IntentFunction
    form: str
    consumed_structures: tuple[str, ...] = ()


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
    """Project canonical output bindings into an explicit printer view."""

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
    """Inline the closed full-axis denotation of one canonical logical loop.

    This projection consumes J only.  Scan names denote complete logical source
    tensors, and carry initializers must be explicit aliases or typed constants.
    Unsupported loop forms fail closed instead of inventing schedule semantics.
    """

    if len(j.intent.body) != 1 or j.intent.body[0].op != "loop":
        return None
    loop = j.intent.body[0]
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
    prefix: list[IntentOp] = []
    for state in loop.carry:
        if state.type is None:
            return None
        init = state.init_value
        if init is None:
            if state.body_argument in body_references:
                return None
            continue
        if isinstance(init, str) and init in inputs:
            init_tensor = inputs[init]
            if (
                init_tensor.dtype != state.type.dtype
                or init_tensor.shape != state.type.shape
            ):
                raise ValueError(
                    "logical loop carry initializer type differs from its state type"
                )
            aliases[state.body_argument] = init
            continue
        if isinstance(init, bool) or isinstance(init, (int, float)):
            prefix.append(
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

    body = [*prefix, *substitute_body(loop.body)]
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
    returns: list[IntentReturn] = []
    for binding in j_elaboration(j).output_bindings:
        if binding.value not in result_to_next:
            raise ValueError(
                "logical loop output binding does not name a loop result"
            )
        returns.append(
            IntentReturn(binding.output, result_to_next[binding.value])
        )
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
    """Choose the unique registered target-neutral function form from J."""

    j = elaborate_j(j)
    resolved_dims = resolve_dimension_relations(j, dims)
    if any(op.op == "loop" for op in j.intent.body):
        function = _logical_full_axis_step_function(j, resolved_dims)
        if function is None:
            raise NotImplementedError(
                "canonical materializer does not yet encode this structured loop form"
            )
        form = LOGICAL_FULL_AXIS_STEP_MATERIALIZATION
        consumed = tuple(
            f"loop:{op.node_id or 'anonymous'}:logical_full_axis_step"
            for op in j.intent.body
            if op.op == "loop"
        )
    else:
        function, consumed = _canonical_return_function(j)
        form = FLAT_MATERIALIZATION
    return JFunctionMaterialization(function, form, consumed)


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

    j = elaborate_j(j)
    resolved_dims = resolve_dimension_relations(j, dims)
    value_dimension_equalities = resolve_value_dimension_equalities(j, resolved_dims)
    output_values = {
        binding.output: binding.value
        for binding in j_elaboration(j).output_bindings
    }
    return emit_module(
        j.intent,
        resolved_dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        value_dimension_equalities=value_dimension_equalities,
        output_values=output_values,
    )


__all__ = [
    "FLAT_MATERIALIZATION",
    "JFunctionMaterialization",
    "LOGICAL_FULL_AXIS_STEP_MATERIALIZATION",
    "emit_j_module",
    "materialize_j_function",
]
