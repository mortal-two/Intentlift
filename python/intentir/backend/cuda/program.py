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
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import (
    DuplicatePolicy,
    ResultRelationKind,
    SemanticArchive,
    UpdateCombiner,
    UpdateOrder,
)

from .pipeline import LoweredProgram, lower_to_cuda
from .selection import LoweringTarget


class CudaJLoweringError(ValueError):
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


def _broadcast_axes(j: SemanticArchive) -> dict[str, dict[str, tuple[int, ...]]]:
    nodes = _nodes(j)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for item in j.boundary.broadcasts:
        op = nodes.get(item.node_ref)
        if op is None or not isinstance(op.result, str):
            raise CudaJLoweringError("broadcast relation requires one result consumer")
        if item.operand not in op.operands:
            raise CudaJLoweringError("broadcast relation names a non-consumed operand")
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in item.axes):
            raise CudaJLoweringError("CUDA emitter does not support symbolic broadcast axes")
        axes = tuple(int(axis) for axis in item.axes)
        previous = result.setdefault(op.result, {}).get(item.operand)
        if previous is not None and previous != axes:
            raise CudaJLoweringError("conflicting broadcast relations")
        result[op.result][item.operand] = axes
    return result


def _index_bounds(j: SemanticArchive) -> frozenset[str]:
    return frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in j.boundary.index_bounds
    )


def analyze_j_for_cuda(j: SemanticArchive) -> dict[str, Any]:
    """Check or consume every active Layer-B collection before emission."""

    if not isinstance(j, SemanticArchive):
        raise CudaJLoweringError("CUDA analysis requires canonical J")
    unsupported: list[str] = []
    consumed: list[str] = []
    if j.boundary.domains:
        unsupported.append("explicit_domain_relation")
    if j.boundary.accesses:
        unsupported.append("explicit_access_relation")
    if j.boundary.effects:
        unsupported.append("explicit_effect_relation")
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
    try:
        _broadcast_axes(j)
    except CudaJLoweringError as exc:
        unsupported.append(f"broadcast:{exc}")
    else:
        consumed.extend(
            f"broadcast:{item.node_ref}:{item.operand}"
            for item in j.boundary.broadcasts
        )
    consumed.extend(f"scope:{item.scope_id}" for item in j.scopes)
    return {
        "supported": not unsupported,
        "consumed": sorted(consumed),
        "unsupported": sorted(set(unsupported)),
        "boundary_sha256": hashlib.sha256(
            json.dumps(
                j.boundary.to_json(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


@dataclass(frozen=True)
class CudaJLowering:
    request: LoweringRequest
    boundary_consumption: Mapping[str, Any]
    intent_mlir: str
    artifact: LoweredProgram

    def __post_init__(self) -> None:
        if not self.boundary_consumption.get("supported", False):
            raise CudaJLoweringError("CUDA lowering left Layer-B relations unsupported")
        digest = hashlib.sha256(self.intent_mlir.encode()).hexdigest()
        if self.artifact.source_mlir_sha256 != digest:
            raise CudaJLoweringError("compiled CUDA artifact differs from emitted MLIR")

    @property
    def mlir_sha256(self) -> str:
        return hashlib.sha256(self.intent_mlir.encode()).hexdigest()

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
    mlir = emit_j_module(
        request.j,
        dict(concrete_extents),
        broadcast_axes=_broadcast_axes(request.j),
        proven_index_bounds=_index_bounds(request.j),
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
    return CudaJLowering(request, consumption, mlir, artifact)


def cuda_target(architecture: str, *, features: Sequence[str] = ()) -> BackendTarget:
    return BackendTarget("cuda", architecture, tuple(features))


__all__ = [
    "CudaJLowering", "CudaJLoweringError", "analyze_j_for_cuda",
    "cuda_target", "lower_j_to_cuda",
]
