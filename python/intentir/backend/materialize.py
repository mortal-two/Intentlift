"""Target-neutral deterministic J materialization and erasable C attachment."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from intentir.backend.protocol import BackendTarget
from intentir.frontend.canonical_j import (
    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
    materialize_j_function,
)
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.canonical_j import elaborate_j
from intentir.ir.dense_predicate_materialization import (
    BoundaryMaterializationProjection,
    materialize_dense_load_predicates,
)
from intentir.ir.typed_boundary_materialization import (
    TypedBoundaryMaterializationProjection,
    materialize_typed_boundary,
)
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.model import IntentFunction, IntentOp
from intentir.ir.paired_rotation import derive_paired_rotation_view
from intentir.ir.value_dimension import resolve_value_dimension_equalities
from intentir.verify.gate1.model import (
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
)


MATERIALIZER_VERSION = "intentir.production.materializer.v2"
ATTACH_C_VERSION = "intentir.production.attach-c.v1"


class MaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class MaterializedJ:
    """Immutable M0 plus the value-free Layer-B projection that admitted it."""

    j: SemanticArchive
    intent_mlir: str
    concrete_extents: tuple[tuple[str, int], ...]
    boundary_projection: (
        ContractMaterializationProjection
        | BoundaryMaterializationProjection
        | TypedBoundaryMaterializationProjection
    )
    form: str
    consumed_structures: tuple[str, ...]
    constexpr_bindings: tuple[tuple[str, bool | int | float], ...] = ()
    materializer_version: str = MATERIALIZER_VERSION

    def __post_init__(self) -> None:
        if not self.intent_mlir.strip():
            raise MaterializationError("materialization produced empty Intent MLIR")
        if self.boundary_projection.runtime_predicate_proofs:
            raise MaterializationError(
                "production materialization cannot consume runtime proof records"
            )

    @property
    def extents(self) -> dict[str, int]:
        return dict(self.concrete_extents)

    @property
    def runtime_input_names(self) -> tuple[str, ...]:
        specialized = {name for name, _value in self.constexpr_bindings}
        return tuple(
            item.name for item in self.j.intent.inputs if item.name not in specialized
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "materializer_version": self.materializer_version,
            "materialization_form": self.form,
            "concrete_extents": self.extents,
            "constexpr_bindings": dict(self.constexpr_bindings),
            "runtime_input_names": list(self.runtime_input_names),
            "boundary_projection": self.boundary_projection.to_json(),
            "consumed_structures": list(self.consumed_structures),
        }


@dataclass(frozen=True)
class CPrior:
    slot: str
    role: str
    value: bool | int | float | str
    subject_ref: str | None
    axis_ref: int | str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "role": self.role,
            "value": self.value,
            "subject_ref": self.subject_ref,
            "axis_ref": self.axis_ref,
        }


@dataclass(frozen=True)
class BackendSearchState:
    """S_C with M0 unchanged and C represented only as discardable priors."""

    base: MaterializedJ
    target: BackendTarget
    priors: tuple[CPrior, ...]
    ignored_slots: tuple[tuple[str, str], ...]
    attach_version: str = ATTACH_C_VERSION

    @property
    def base_ir(self) -> str:
        return self.base.intent_mlir

    def erase_c(self) -> str:
        return self.base.intent_mlir

    def to_record(self) -> dict[str, Any]:
        return {
            "attach_version": self.attach_version,
            "target": self.target.to_dict(),
            "priors": [item.to_record() for item in self.priors],
            "ignored_slots": [
                {"slot": slot, "reason": reason}
                for slot, reason in self.ignored_slots
            ],
            "erase_c_matches_base": self.erase_c() == self.base_ir,
        }


def _specialize_constexpr_inputs(
    j: SemanticArchive,
    function: IntentFunction,
    bindings: Mapping[str, bool | int | float],
) -> tuple[IntentFunction, tuple[str, ...]]:
    if not bindings:
        return function, ()
    tensors = {item.name: item for item in j.intent.inputs}
    allowed = {
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind is ParameterKind.CONSTEXPR
        and parameter.logical_ref in tensors
    }
    unknown = sorted(set(bindings) - allowed)
    if unknown:
        raise MaterializationError(
            "constexpr specialization is not authorized by J.signature: "
            + ", ".join(unknown)
        )
    constants: list[IntentOp] = []
    for name, value in sorted(bindings.items()):
        tensor = tensors[name]
        if tensor.shape:
            raise MaterializationError(
                f"constexpr logical input {name!r} must have rank zero"
            )
        dtype = tensor.dtype.removeprefix("torch.").lower()
        if dtype in {"bool", "i1"}:
            valid = isinstance(value, bool)
        elif dtype in {"i8", "i16", "i32", "i64", "int8", "int16", "int32", "int64"}:
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            raise MaterializationError(
                f"constexpr logical input {name!r} value is incompatible with {tensor.dtype}"
            )
        constants.append(
            IntentOp(
                op="const",
                result=name,
                attrs={"value": value, "dtype": tensor.dtype},
                type={"dtype": tensor.dtype, "shape": []},
            )
        )

    consumed_control: list[str] = []

    def specialize_body(body: list[IntentOp]) -> list[IntentOp]:
        specialized: list[IntentOp] = []
        for original in body:
            operation = copy.deepcopy(original)
            operation.body = specialize_body(operation.body)
            for arm in (operation.positive, operation.negative):
                if arm is not None:
                    arm.body = specialize_body(arm.body)
            if (
                operation.op in {"effect_if", "value_if"}
                and operation.predicate in bindings
            ):
                condition = bindings[operation.predicate]
                if not isinstance(condition, bool):
                    raise MaterializationError(
                        "effect_if constexpr predicate must be boolean"
                    )
                arm = operation.positive if condition else operation.negative
                if arm is None:
                    raise MaterializationError(
                        "effect_if constexpr specialization requires both arms"
                    )
                consumed_control.append(
                    f"{operation.op}:"
                    f"{operation.node_id or 'anonymous'}:"
                    f"constexpr_{str(condition).lower()}"
                )
                specialized.extend(arm.body)
                if operation.op == "value_if":
                    if (
                        not isinstance(operation.result, str)
                        or not isinstance(arm.yield_value, str)
                        or not isinstance(operation.type, dict)
                        or not isinstance(operation.type.get("dtype"), str)
                    ):
                        raise MaterializationError(
                            "value_if specialization requires one typed yield"
                        )
                    if arm.yield_value != operation.result:
                        specialized.append(
                            IntentOp(
                                op="cast",
                                result=operation.result,
                                operands=[arm.yield_value],
                                attrs={
                                    "to_dtype": operation.type["dtype"],
                                },
                                type=copy.deepcopy(operation.type),
                            )
                        )
                continue
            specialized.append(operation)
        return specialized

    return IntentFunction(
        name=function.name,
        inputs=[item for item in function.inputs if item.name not in bindings],
        outputs=list(function.outputs),
        body=[*constants, *specialize_body(function.body)],
        returns=list(function.returns),
    ), tuple(consumed_control)


def _project_value_dimension_equalities(
    j: SemanticArchive,
    extents: Mapping[str, int],
    function: IntentFunction,
    bindings: Mapping[str, bool | int | float],
) -> tuple[tuple[int, int], ...]:
    """Validate folded equalities or remap them to the specialized ABI."""

    resolved = resolve_value_dimension_equalities(j, extents)
    original_inputs = list(j.intent.inputs)
    runtime_indices = {
        item.name: index for index, item in enumerate(function.inputs)
    }
    projected: list[tuple[int, int]] = []
    for original_index, expected in resolved:
        name = original_inputs[original_index].name
        if name in bindings:
            actual = bindings[name]
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
                or actual != expected
            ):
                raise MaterializationError(
                    "constexpr specialization violates J value/dimension "
                    f"equality: {name}={actual!r}, expected {expected}"
                )
            continue
        if name not in runtime_indices:
            raise MaterializationError(
                f"value/dimension equality input {name!r} vanished from ABI"
            )
        projected.append((runtime_indices[name], expected))
    return tuple(projected)


def _project_inplace_outputs(
    j: SemanticArchive, function: IntentFunction
) -> tuple[int, ...]:
    """Map J-owned inout results back to their unchanged runtime inputs."""

    inout_refs = {
        item.logical_ref
        for item in j.signature.parameters
        if item.mode is ParameterMode.INOUT and item.logical_ref is not None
    }
    positions = {item.name: index for index, item in enumerate(function.inputs)}
    projected = tuple(
        positions.get(output.name, -1) if output.name in inout_refs else -1
        for output in function.outputs
    )
    return projected if any(index >= 0 for index in projected) else ()


def materialize_j(
    j: SemanticArchive,
    concrete_extents: Mapping[str, int],
    *,
    constexpr_bindings: Mapping[str, bool | int | float] | None = None,
) -> MaterializedJ:
    """Compute M0 from J alone, failing before codegen on incomplete Layer B."""

    if not isinstance(j, SemanticArchive):
        raise MaterializationError("materialize_j requires canonical J")
    extents: dict[str, int] = {}
    for name, value in concrete_extents.items():
        if not isinstance(name, str) or not name:
            raise MaterializationError("concrete extent names must be non-empty")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaterializationError(
                f"concrete extent {name!r} must be a non-negative integer"
            )
        extents[name] = value
    canonical = elaborate_j(j)
    scalar_bindings = dict(constexpr_bindings or {})
    try:
        logical = materialize_j_function(canonical, extents)
        typed_boundary = materialize_typed_boundary(
            canonical,
            logical.function,
            extents,
            consume_loop_storage=(
                logical.form
                in {
                    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
                    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
                }
            ),
        )
        predicated = None
        if typed_boundary is not None:
            projection = typed_boundary.projection
            function = typed_boundary.function
        else:
            predicated = materialize_dense_load_predicates(
                canonical,
                logical.function,
            )
        if typed_boundary is None and predicated is not None:
            projection = predicated.projection
            function = predicated.function
        elif typed_boundary is None:
            projection = project_contract_materialization(
                canonical,
                consume_loop_storage=(
                    logical.form
                    in {
                        LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
                        STRUCTURED_SCAN_LOOP_MATERIALIZATION,
                    }
                ),
            )
            function = logical.function
        function, consumed_control = _specialize_constexpr_inputs(
            canonical,
            function,
            scalar_bindings,
        )
        value_dimension_equalities = _project_value_dimension_equalities(
            canonical,
            extents,
            function,
            scalar_bindings,
        )
        inplace_outputs = _project_inplace_outputs(canonical, function)
        paired_rotation_view = derive_paired_rotation_view(
            canonical,
            extents,
            scalar_bindings,
        )
        paired_rotation = (
            None
            if paired_rotation_view is None
            else (
                paired_rotation_view.layout,
                paired_rotation_view.input_indices(function),
                paired_rotation_view.sections,
            )
        )
        mlir = emit_module(
            function,
            extents,
            broadcast_axes=projection.emitter_broadcast_axes(),
            proven_index_bounds=projection.proven_index_bounds,
            value_dimension_equalities=value_dimension_equalities,
            inplace_outputs=inplace_outputs,
            paired_rotation=paired_rotation,
            loop_schedule_tile=logical.loop_schedule_tile,
        )
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise MaterializationError(
            f"deterministic J materialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    return MaterializedJ(
        j=canonical,
        intent_mlir=mlir,
        concrete_extents=tuple(sorted(extents.items())),
        boundary_projection=projection,
        form=logical.form,
        consumed_structures=(
            *logical.consumed_structures,
            *consumed_control,
        ),
        constexpr_bindings=tuple(sorted(scalar_bindings.items())),
    )


def attach_c(
    base: MaterializedJ,
    c: ImplementationConfig | None,
    target: BackendTarget,
) -> BackendSearchState:
    """Attach C after M0 without decorating or changing the base Intent MLIR."""

    if not isinstance(base, MaterializedJ):
        raise MaterializationError("Attach_C requires one MaterializedJ base")
    if not isinstance(target, BackendTarget):
        raise MaterializationError("Attach_C requires BackendTarget")
    if c is None:
        return BackendSearchState(base, target, (), ())
    if not isinstance(c, ImplementationConfig):
        raise MaterializationError("Attach_C requires canonical C")
    priors: list[CPrior] = []
    ignored: list[tuple[str, str]] = []
    for slot in c.slots:
        if slot.name not in c.values:
            ignored.append((slot.name, "valuation_absent"))
            continue
        value = c.values[slot.name]
        if not isinstance(value, (bool, int, float, str)):
            ignored.append((slot.name, "non_scalar_value"))
            continue
        priors.append(
            CPrior(
                slot=slot.name,
                role=slot.role,
                value=value,
                subject_ref=slot.subject_ref,
                axis_ref=slot.axis_ref,
            )
        )
    state = BackendSearchState(
        base=base,
        target=target,
        priors=tuple(priors),
        ignored_slots=tuple(ignored),
    )
    if state.erase_c() != base.intent_mlir:
        raise AssertionError("Erase_C invariant violated")
    return state


__all__ = [
    "ATTACH_C_VERSION",
    "MATERIALIZER_VERSION",
    "BackendSearchState",
    "CPrior",
    "MaterializationError",
    "MaterializedJ",
    "attach_c",
    "materialize_j",
]
