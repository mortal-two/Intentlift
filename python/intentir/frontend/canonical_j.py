"""Shared deterministic ``J -> intent MLIR`` materialization entry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j, j_elaboration
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


def _walk(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk(op.body))
        for arm in (op.positive, op.negative):
            if arm is not None:
                result.extend(_walk(arm.body))
    return tuple(result)


def _canonical_return_function(j: SemanticArchive) -> IntentFunction:
    """Project the one elaborated output binding into the MLIR printer view."""

    bindings = j_elaboration(j).output_bindings
    return IntentFunction(
        name=j.intent.name,
        inputs=list(j.intent.inputs),
        outputs=list(j.intent.outputs),
        body=list(j.intent.body),
        returns=[IntentReturn(item.output, item.value) for item in bindings],
    )


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
    """Choose the unique registered logical function form using J alone."""

    j = elaborate_j(j)
    if any(op.op == "loop" for op in j.intent.body):
        function = _logical_full_axis_step_function(j, dims)
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
        function = _canonical_return_function(j)
        form = FLAT_MATERIALIZATION
        consumed = ()
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
    value_dimension_equalities = resolve_value_dimension_equalities(j, dims)
    logical = materialize_j_function(j, dims)
    mlir = emit_module(
        logical.function,
        dict(dims),
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        value_dimension_equalities=value_dimension_equalities,
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
    "emit_j_module",
    "materialize_j_function",
    "materialize_j_module",
]
