"""Fail-closed canonical-J lowering boundary for the CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.backend.protocol import (
    BackendConformanceRecord,
    BackendTarget,
    ComparisonEvidence,
    LoweringRequest,
)
from intentir.backend.materialize import MaterializedJ, materialize_j
from intentir.frontend.canonical_j import LOGICAL_FULL_AXIS_STEP_MATERIALIZATION
from intentir.ir.dense_predicate_materialization import (
    BoundaryMaterializationProjection,
)
from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.typed_boundary_materialization import (
    TypedBoundaryMaterializationProjection,
)
from intentir.ir.model import IntentOp
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.verify.gate1.model import (
    DuplicatePolicy,
    ResultRelationKind,
    SemanticArchive,
    UpdateCombiner,
    UpdateOrder,
)

from .pipeline import LoweredProgram, execute_cuda_artifact, lower_to_cuda
from .selection import LoweringTarget


class CudaJLoweringError(ValueError):
    pass


class CudaJExecutionError(ValueError):
    pass


def _child_regions(op: IntentOp) -> tuple[Sequence[IntentOp], ...]:
    regions: list[Sequence[IntentOp]] = [op.body]
    if op.op in {"effect_if", "value_if"}:
        if op.positive is not None:
            regions.append(op.positive.body)
        if op.negative is not None:
            regions.append(op.negative.body)
    return tuple(regions)


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        for region in _child_regions(op):
            result.extend(_walk_ops(region))
    return tuple(result)


def _nodes(j: SemanticArchive) -> dict[str, IntentOp]:
    by_path: dict[str, IntentOp] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = op
            visit(op.body, f"{path}.body")
            if op.op in {"effect_if", "value_if"}:
                if op.positive is not None:
                    visit(op.positive.body, f"{path}.positive.body")
                if op.negative is not None:
                    visit(op.negative.body, f"{path}.negative.body")

    visit(j.intent.body)
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _contract_projection(
    j: SemanticArchive,
) -> ContractMaterializationProjection:
    """Derive CUDA's ephemeral Layer-B view from the one accepted J."""

    try:
        return project_contract_materialization(j)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise CudaJLoweringError(
            "typed contract projection failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def analyze_j_for_cuda(
    j: SemanticArchive,
    *,
    materialization: MaterializedJ | None = None,
) -> dict[str, Any]:
    """Check or consume every active Layer-B collection before emission."""

    if not isinstance(j, SemanticArchive):
        raise CudaJLoweringError("CUDA analysis requires canonical J")
    j = elaborate_j(j)
    unsupported: list[str] = []
    consumed: list[str] = []
    projection: (
        ContractMaterializationProjection
        | BoundaryMaterializationProjection
        | TypedBoundaryMaterializationProjection
        | None
    ) = (
        materialization.boundary_projection
        if materialization is not None
        else None
    )
    if projection is None:
        try:
            projection = _contract_projection(j)
        except CudaJLoweringError as exc:
            unsupported.append(str(exc))
    if projection is not None:
        consumed.extend(projection.consumed)
        unsupported.extend(projection.unsupported_lowering)
    projected_consumed = set(projection.consumed) if projection is not None else set()
    if j.boundary.access_guard_relations:
        guards_consumed = all(
            any(
                ref.startswith(
                    f"access_guard:{item.access_ref}:{item.value_ref}:"
                )
                and ref.endswith(f":{item.polarity.value}")
                for ref in projected_consumed
            )
            for item in j.boundary.access_guard_relations
        )
        if not guards_consumed:
            # Fail closed unless the shared J-only functional projection has
            # consumed every authored access-local guard.  The projection is
            # the existing generic consumer; CUDA does not invent a second
            # access recipe or infer permission from the surrounding select.
            unsupported.append("access_guard_materialization_not_implemented")
    if j.boundary.aliases:
        unsupported.append("alias_relation")
    consumed.extend(
        f"effect:{item.effect_id or index}:{item.kind.value}"
        for index, item in enumerate(j.boundary.effects)
    )
    for item in j.storage:
        storage_ref = f"storage:{item.storage_id}"
        if item.mutability.value != "mutable":
            consumed.append(storage_ref)
        elif (
            materialization is not None
            and storage_ref in projected_consumed
        ):
            # The registered full-axis-step materialization consumes the exact
            # structured carry/storage relation before the flat CUDA path is
            # considered.  Arbitrary mutable storage remains unsupported.
            consumed.append(storage_ref)
        else:
            unsupported.append(f"general_mutable_storage:{item.storage_id}")

    operations = _walk_ops(j.intent.body)
    op_names = {op.op for op in operations}
    if "masked_scatter" in op_names:
        unsupported.append("masked_scatter_logical_index_relation")

    nodes = _nodes(j)
    tensor_by_name = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    observable_values = {
        item.output: item.value
        for item in j_elaboration(j).output_bindings
    }
    relation_by_output = {
        item.output: item for item in j.boundary.result_relations
    }
    for item in j.boundary.index_bounds:
        consumed.append(f"index_bounds:{item.node_ref}")
    for item in j.boundary.result_relations:
        consumed.append(f"result_relation:{item.output}:{item.kind.value}")

    for item in j.boundary.updates:
        op = nodes[item.node_ref]
        base = tensor_by_name.get(item.base)
        indices = tensor_by_name.get(item.indices)
        values = tensor_by_name.get(item.values)
        if base is None or indices is None or values is None:
            unsupported.append(f"update_internal_operand:{item.node_ref}")
            continue
        if not all(len(tensor.shape) == 2 for tensor in (base, indices, values)):
            unsupported.append(f"update_rank_not_2:{item.node_ref}")
            continue
        outputs = [
            output
            for output, value in observable_values.items()
            if value == op.result
        ]
        if len(outputs) != 1:
            unsupported.append(f"update_not_direct_output:{item.node_ref}")
            continue
        relation = relation_by_output[outputs[0]]
        if item.duplicate_policy is DuplicatePolicy.ONE_OF:
            unsupported.append(
                f"racing_overwrite_not_cuda_defined:{item.node_ref}"
            )
            continue
        if item.combiner is UpdateCombiner.ADD:
            if base.dtype.removeprefix("torch.").lower() not in {"f32", "float32"}:
                unsupported.append(f"atomic_add_dtype:{item.node_ref}")
                continue
            if (
                item.duplicate_policy is DuplicatePolicy.COMBINE
                and item.order is not UpdateOrder.UNORDERED
            ):
                unsupported.append(f"ordered_duplicate_add:{item.node_ref}")
                continue
            if (
                item.duplicate_policy is DuplicatePolicy.COMBINE
                and relation.kind is not ResultRelationKind.TOLERANCE
            ):
                unsupported.append(
                    f"floating_duplicate_add_requires_tolerance:{item.node_ref}"
                )
                continue
        consumed.append(
            "update:"
            f"{item.node_ref}:{item.combiner.value}:"
            f"{item.duplicate_policy.value}:{item.order.value}"
        )
    if "loop" in op_names:
        loop_nodes = {
            node_ref for node_ref, operation in nodes.items()
            if node_ref in j.node_ids and operation.op == "loop"
        }
        typed_loops_consumed = all(
            f"loop:{node_ref}" in projected_consumed for node_ref in loop_nodes
        )
        logical_loop_consumed = (
            materialization is not None
            and materialization.form == LOGICAL_FULL_AXIS_STEP_MATERIALIZATION
        )
        if not (typed_loops_consumed or logical_loop_consumed):
            unsupported.append("semantic_recurrence_not_lowered_by_flat_cuda_path")
        elif materialization is not None:
            consumed.extend(materialization.consumed_structures)
            consumed.extend(f"loop:{node_ref}" for node_ref in loop_nodes)

    for item in j.boundary.reductions:
        op = nodes.get(item.node_ref)
        if op is None or op.op not in {"reduce", "scan"}:
            unsupported.append(f"invalid_reduction:{item.node_ref}")
            continue
        reduction_ref = f"reduction:{item.node_ref}"
        if reduction_ref in projected_consumed:
            consumed.append(reduction_ref)
            continue
        fn = op.attrs.get("fn")
        declared = item.identity if item.identity is not None else item.masked_contribution
        neutral = {
            "sum": {0, "ZERO"},
            "mean": {0, "ZERO"},
            "prod": {1, "ONE"},
            "any": {False, "false"},
            "all": {True, "true"},
            "max": {"NEG_INF", "-inf", float("-inf")},
            "min": {"POS_INF", "inf", "+inf", float("inf")},
        }.get(fn)
        if neutral is None or declared not in neutral:
            unsupported.append(f"non_neutral_reduction_boundary:{item.node_ref}")
        else:
            consumed.append(f"reduction:{item.node_ref}")

    if set(observable_values) != {
        item.name for item in j.intent.outputs
    }:
        unsupported.append("output_destination_relation")
    else:
        consumed.extend(f"output:{name}" for name in observable_values)
    consumed.extend(f"scope:{item.scope_id}" for item in j.scopes)
    return {
        "supported": not unsupported,
        "consumed": sorted(set(consumed)),
        "unsupported": sorted(set(unsupported)),
        "boundary_projection": (
            projection.to_json() if projection is not None else None
        ),
    }


@dataclass(frozen=True)
class CudaJLowering:
    request: LoweringRequest
    boundary_consumption: Mapping[str, Any]
    intent_mlir: str
    artifact: LoweredProgram
    concrete_extents: tuple[tuple[str, int], ...]
    materializer_version: str
    materialization_form: str

    def __post_init__(self) -> None:
        if not self.boundary_consumption.get("supported", False):
            raise CudaJLoweringError("CUDA lowering left Layer-B relations unsupported")

    @property
    def concrete_extents_map(self) -> dict[str, int]:
        return dict(self.concrete_extents)

    def conformance_record(
        self,
        *,
        test_domain: Mapping[str, Any],
        comparisons: Sequence[ComparisonEvidence],
    ) -> BackendConformanceRecord:
        return BackendConformanceRecord(
            target=self.request.target,
            variant=self.request.variant,
            boundary_consumption=dict(self.boundary_consumption),
            test_domain=dict(test_domain),
            comparisons=tuple(comparisons),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "target": self.request.target.to_dict(),
            "variant": self.request.variant,
            "concrete_extents": self.concrete_extents_map,
            "materializer_version": self.materializer_version,
            "materialization_form": self.materialization_form,
            "boundary_consumption": dict(self.boundary_consumption),
            "cuda": self.artifact.to_record(),
        }


def lower_j_to_cuda(
    request: LoweringRequest,
    concrete_extents: Mapping[str, int],
) -> CudaJLowering:
    if not isinstance(request, LoweringRequest):
        raise CudaJLoweringError("CUDA lowering requires LoweringRequest")
    if request.target.family != "cuda":
        raise CudaJLoweringError("CUDA backend requires a CUDA target")
    if request.variant not in {"generic", "auto"}:
        raise CudaJLoweringError("CUDA lowering accepts generic or auto variant")
    try:
        materialized = materialize_j(request.j, concrete_extents)
    except (TypeError, ValueError) as exc:
        raise CudaJLoweringError(str(exc)) from exc
    consumption = analyze_j_for_cuda(
        request.j,
        materialization=materialized,
    )
    if not consumption["supported"]:
        raise CudaJLoweringError(
            "CUDA cannot consume complete Layer B: "
            + ", ".join(consumption["unsupported"])
        )
    mlir = materialized.intent_mlir
    target = LoweringTarget(
        backend="cuda",
        chip=request.target.architecture,
        features=frozenset(request.target.features),
    )
    artifact = lower_to_cuda(
        mlir,
        target=target,
        allow_specialized=request.variant == "auto",
    )
    return CudaJLowering(
        request,
        consumption,
        mlir,
        artifact,
        tuple(sorted(concrete_extents.items())),
        materialized.materializer_version,
        materialized.form,
    )


def execute_j_cuda(
    lowering: CudaJLowering,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one compiled J lowering using only J-owned logical handles.

    The MLIR-level orchestrator intentionally supports positional ``argN``
    fallbacks for legacy callers.  This J-facing boundary does not: the exact
    logical input set comes from the same accepted J that produced the cubin,
    and ordered ``outN`` runtime buffers are mapped back to J output handles.
    """

    if not isinstance(lowering, CudaJLowering):
        raise CudaJExecutionError("CUDA J execution requires CudaJLowering")
    if not isinstance(inputs, Mapping):
        raise CudaJExecutionError(
            "CUDA J execution inputs must map J logical handles to values"
        )
    if any(not isinstance(name, str) for name in inputs):
        raise CudaJExecutionError(
            "CUDA J execution input keys must be J logical handle strings"
        )
    expected_inputs = tuple(item.name for item in lowering.request.j.intent.inputs)
    provided = set(inputs)
    expected = set(expected_inputs)
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise CudaJExecutionError(
            "CUDA J execution inputs differ from J logical inputs: "
            + "; ".join(details)
        )
    try:
        validate_value_dimension_instantiation(
            lowering.request.j,
            lowering.concrete_extents_map,
            {name: inputs[name] for name in expected_inputs},
        )
    except (TypeError, ValueError) as exc:
        raise CudaJExecutionError(str(exc)) from exc

    runtime_outputs = execute_cuda_artifact(
        lowering.artifact,
        lowering.intent_mlir,
        {name: inputs[name] for name in expected_inputs},
    )
    if not isinstance(runtime_outputs, Mapping):
        raise CudaJExecutionError("CUDA runtime did not return an output mapping")
    logical_outputs = tuple(item.name for item in lowering.request.j.intent.outputs)
    runtime_keys = tuple(f"out{index}" for index in range(len(logical_outputs)))
    if set(runtime_outputs) != set(runtime_keys):
        raise CudaJExecutionError(
            "CUDA runtime outputs differ from J logical outputs: expected "
            f"{list(runtime_keys)}, got {sorted(runtime_outputs)}"
        )
    return {
        logical_name: runtime_outputs[runtime_name]
        for logical_name, runtime_name in zip(
            logical_outputs, runtime_keys, strict=True
        )
    }


def cuda_target(architecture: str, *, features: Sequence[str] = ()) -> BackendTarget:
    return BackendTarget("cuda", architecture, tuple(features))


__all__ = [
    "CudaJExecutionError", "CudaJLowering", "CudaJLoweringError",
    "analyze_j_for_cuda", "cuda_target", "execute_j_cuda",
    "lower_j_to_cuda",
]
