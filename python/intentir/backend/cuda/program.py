"""Fail-closed canonical-J lowering boundary for the CUDA backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.backend.protocol import (
    BackendConformanceRecord,
    BackendTarget,
    ComparisonEvidence,
    LoweringRequest,
)
from intentir.frontend import emit_j_module
from intentir.ir.functional_boundary import (
    FunctionalBoundaryProjection,
    project_functional_boundary,
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


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def _nodes(j: SemanticArchive) -> dict[str, IntentOp]:
    by_path: dict[str, IntentOp] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = op
            visit(op.body, f"{path}.body")

    visit(j.intent.body)
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _functional_projection(
    j: SemanticArchive,
) -> FunctionalBoundaryProjection:
    """Derive CUDA's ephemeral Layer-B view from the one accepted J."""

    try:
        return project_functional_boundary(j)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise CudaJLoweringError(
            "functional boundary projection failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def analyze_j_for_cuda(j: SemanticArchive) -> dict[str, Any]:
    """Check or consume every active Layer-B collection before emission."""

    if not isinstance(j, SemanticArchive):
        raise CudaJLoweringError("CUDA analysis requires canonical J")
    unsupported: list[str] = []
    consumed: list[str] = []
    projection: FunctionalBoundaryProjection | None = None
    try:
        projection = _functional_projection(j)
    except CudaJLoweringError as exc:
        unsupported.append(str(exc))
    else:
        consumed.extend(projection.consumed)
    if j.boundary.access_guard_relations:
        # The backend may inspect the shared J-only receipt, but its current
        # target path has no general access-local guard materializer.  An
        # explicit Layer-A select is sufficient for RefEval value semantics;
        # it is not permission to erase the authored boundary relation here.
        unsupported.append("access_guard_materialization_not_implemented")
    if j.boundary.aliases:
        unsupported.append("alias_relation")
    if any(item.mutability.value == "mutable" for item in j.storage):
        unsupported.append("general_mutable_storage")
    else:
        consumed.extend(f"storage:{item.storage_id}" for item in j.storage)

    operations = _walk_ops(j.intent.body)
    op_names = {op.op for op in operations}
    if "masked_scatter" in op_names:
        unsupported.append("masked_scatter_logical_index_relation")

    nodes = _nodes(j)
    tensor_by_name = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    return_values = {item.output: item.value for item in j.intent.returns}
    observable_values = {
        item.name: return_values.get(item.name, item.name)
        for item in j.intent.outputs
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
        unsupported.append("semantic_recurrence_not_lowered_by_flat_cuda_path")

    for item in j.boundary.reductions:
        op = nodes.get(item.node_ref)
        if op is None or op.op != "reduce":
            unsupported.append(f"invalid_reduction:{item.node_ref}")
            continue
        fn = op.attrs.get("fn")
        declared = item.identity if item.identity is not None else item.masked_contribution
        neutral = {
            "sum": {0, "ZERO"},
            "mean": {0, "ZERO"},
            "prod": {1, "ONE"},
            "any": {False, "false"},
            "all": {True, "true"},
            "max": {"NEG_INF"},
            "min": {"POS_INF"},
        }.get(fn)
        if neutral is None or declared not in neutral:
            unsupported.append(f"non_neutral_reduction_boundary:{item.node_ref}")
        else:
            consumed.append(f"reduction:{item.node_ref}")

    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item.value for item in j.boundary.output_mappings}
    if mappings and mappings != returns:
        unsupported.append("output_destination_relation")
    else:
        consumed.extend(f"output:{name}" for name in mappings)
    consumed.extend(f"scope:{item.scope_id}" for item in j.scopes)
    boundary_digest = (
        projection.boundary_digest
        if projection is not None
        else hashlib.sha256(
            json.dumps(
                j.boundary.to_json(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    return {
        "supported": not unsupported,
        "consumed": sorted(set(consumed)),
        "unsupported": sorted(set(unsupported)),
        "boundary_sha256": boundary_digest,
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

    def __post_init__(self) -> None:
        if not self.boundary_consumption.get("supported", False):
            raise CudaJLoweringError("CUDA lowering left Layer-B relations unsupported")
        digest = hashlib.sha256(self.intent_mlir.encode()).hexdigest()
        if self.artifact.source_mlir_sha256 != digest:
            raise CudaJLoweringError("compiled CUDA artifact differs from emitted MLIR")

    @property
    def mlir_sha256(self) -> str:
        return hashlib.sha256(self.intent_mlir.encode()).hexdigest()

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
            j_digest=self.request.j_digest,
            c_digest=self.request.c_digest,
            target=self.request.target,
            variant=self.request.variant,
            boundary_consumption=dict(self.boundary_consumption),
            test_domain=dict(test_domain),
            comparisons=tuple(comparisons),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "j_digest": self.request.j_digest,
            "c_digest": self.request.c_digest,
            "target": self.request.target.to_dict(),
            "variant": self.request.variant,
            "concrete_extents": self.concrete_extents_map,
            "mlir_sha256": self.mlir_sha256,
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
    consumption = analyze_j_for_cuda(request.j)
    if not consumption["supported"]:
        raise CudaJLoweringError(
            "CUDA cannot consume complete Layer B: "
            + ", ".join(consumption["unsupported"])
        )
    projection = _functional_projection(request.j)
    mlir = emit_j_module(
        request.j,
        dict(concrete_extents),
        broadcast_axes=projection.emitter_broadcast_axes(),
        proven_index_bounds=projection.proven_index_bounds,
    )
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
