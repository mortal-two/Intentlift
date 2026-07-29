"""J-only reference execution for relation-indexed pointwise graphs.

The ordinary Intent executor evaluates Layer A over complete logical tensors.
That is insufficient when Layer B explicitly gives one consumer a different
logical access view (for example, a sub-domain or an affine slice) and later
assembles an output through several stores.  This module lowers that closed
case from :class:`CanonicalContractIR` instead of recognizing kernel shapes.

The profile is intentionally conservative.  It supports flat, acyclic,
shape-preserving pointwise graphs, arbitrary typed ContractExpr indices and
predicates, explicit masked-load fills, and exhaustive non-overlapping output
stores.  Unsupported control, reduction, alias, or update relations fail
closed.  C, pi, Source, Replay, Gate observations, and kernel names are never
inputs to either projection or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from intentir.backend.cpu.abi import (
    CPUBackendUnsupportedError,
    numpy_dtype,
    validate_array,
)
from intentir.backend.cpu.pipeline import CPUExecutionResult
from intentir.backend.cpu.structured_reference import (
    _Evaluator,
    _resolved_shape,
)
from intentir.frontend.emit_mlir import _broadcast_projection
from intentir.ir.canonical_j import (
    CanonicalValueType,
    canonical_broadcast_axes,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.contract_ir import (
    CanonicalContractIR,
    ContractAccessIR,
    ContractIRProjectionError,
    project_contract_ir,
)
from intentir.ir.contract_semantics import ContractSemanticsError
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.functional_boundary import project_predicate_text
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.model import IntentOp

if TYPE_CHECKING:
    from intentir.verify.gate1.model import SemanticArchive


CONTRACT_REFERENCE_PROTOCOL = "intentir.j-contract-indexed-cpu-reference.v2"


class ContractReferenceUnsupportedError(CPUBackendUnsupportedError):
    """The accepted J is outside the closed relation-indexed profile."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _result_names(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(item for item in operation.result if isinstance(item, str))
    return ()


def _rectangular_domains(j: "SemanticArchive") -> dict[str, int | str]:
    """Return only explicitly authored zero-based rectangular domain extents."""

    domains: dict[str, int | str] = {}
    for domain_id, expression in j_elaboration(j).domains:
        try:
            projection = project_predicate_text(expression.to_text())
        except (NotImplementedError, ValueError) as exc:
            raise ContractReferenceUnsupportedError(
                "contract reference requires zero-based rectangular domains; "
                f"unsupported domain {domain_id!r}"
            ) from exc
        if projection.guards:
            raise ContractReferenceUnsupportedError(
                "contract reference does not implement guard-bearing domains"
            )
        upper_axes = {axis for axis, _extent in projection.upper_bounds}
        if upper_axes != set(projection.lower_axes):
            raise ContractReferenceUnsupportedError(
                "contract reference requires matching zero lower bounds"
            )
        for axis, extent in projection.upper_bounds:
            previous = domains.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"contract domain axis {axis!r} has conflicting extents"
                )
            domains[axis] = extent
    if not domains:
        raise ContractReferenceUnsupportedError(
            "contract reference requires at least one authored logical domain"
        )
    return domains


def _axes_for_shape(
    value_type: CanonicalValueType,
    domains: Mapping[str, int | str],
    *,
    context: str,
) -> tuple[str, ...]:
    """Bind result dimensions to uniquely matching authored domain axes."""

    axes: list[str] = []
    used: set[str] = set()
    for dimension, extent in enumerate(value_type.shape):
        candidates = [
            axis
            for axis, domain_extent in domains.items()
            if axis not in used and domain_extent == extent
        ]
        if len(candidates) != 1:
            raise ContractReferenceUnsupportedError(
                f"{context} dimension {dimension} does not have one unique "
                "J-authored domain axis"
            )
        axis = candidates[0]
        axes.append(axis)
        used.add(axis)
    return tuple(axes)


@dataclass(frozen=True)
class IndexedLoadProjection:
    access: ContractAccessIR
    operand_index: int
    result_ref: str
    result_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_operand_view",
            "access": self.access.to_json(),
            "operand_index": self.operand_index,
            "result_ref": self.result_ref,
            "result_axes": list(self.result_axes),
        }


@dataclass(frozen=True)
class IndexedStoreProjection:
    access: ContractAccessIR
    value_ref: str
    value_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_output_store",
            "access": self.access.to_json(),
            "value_ref": self.value_ref,
            "value_axes": list(self.value_axes),
        }


@dataclass(frozen=True)
class ContractReferencePlan:
    """Ephemeral source-free lowering plan derived one-to-one from J."""

    contract_ir: CanonicalContractIR
    domains: tuple[tuple[str, int | str], ...]
    loads: tuple[IndexedLoadProjection, ...]
    stores: tuple[IndexedStoreProjection, ...]
    broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]

    @property
    def domain_map(self) -> dict[str, int | str]:
        return dict(self.domains)

    @property
    def load_map(self) -> dict[tuple[str, int], IndexedLoadProjection]:
        return {
            (item.access.node_ref or "", item.operand_index): item
            for item in self.loads
        }

    @property
    def stores_by_output(self) -> dict[str, tuple[IndexedStoreProjection, ...]]:
        grouped: dict[str, list[IndexedStoreProjection]] = {}
        for item in self.stores:
            grouped.setdefault(item.access.tensor, []).append(item)
        return {name: tuple(items) for name, items in grouped.items()}

    @property
    def broadcast_map(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            result: dict(operands) for result, operands in self.broadcast_axes
        }

    def to_json(self) -> dict[str, Any]:
        record = {
            "protocol": CONTRACT_REFERENCE_PROTOCOL,
            "profile": "relation_indexed_pointwise_v1",
            "contract_ir_digest": self.contract_ir.digest,
            "domains": [list(item) for item in self.domains],
            "loads": [item.to_json() for item in self.loads],
            "stores": [item.to_json() for item in self.stores],
            "broadcast_axes": [
                [
                    result,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for result, operands in self.broadcast_axes
            ],
        }
        return {**record, "plan_digest": _digest(record)}


_POINTWISE_OPS = frozenset({"cast", "const", "map_binary", "map_unary", "select"})


def project_contract_reference(j: "SemanticArchive") -> ContractReferencePlan:
    """Project a closed relation-indexed pointwise plan from canonical J only."""

    canonical = elaborate_j(j)
    boundary = canonical.boundary
    if (
        boundary.aliases
        or boundary.reductions
        or boundary.updates
        or boundary.consumer_extensions
        or boundary.access_guard_relations
        or boundary.index_bounds
    ):
        raise ContractReferenceUnsupportedError(
            "contract reference v1 does not implement alias, reduction, update, "
            "consumer-extension, access-guard, or index-bounds relations"
        )
    operations = tuple(canonical.intent.body)
    if any(operation.body or operation.op not in _POINTWISE_OPS for operation in operations):
        raise ContractReferenceUnsupportedError(
            "contract reference v1 requires a flat pointwise Layer-A graph"
        )
    if any(len(_result_names(operation)) != 1 for operation in operations):
        raise ContractReferenceUnsupportedError(
            "contract reference v1 requires one result per Layer-A operation"
        )

    nodes = {
        operation.node_id: operation
        for operation in operations
        if isinstance(operation.node_id, str)
    }
    if len(nodes) != len(operations):
        raise ValueError("contract reference requires unique authored node ids")
    value_types = j_elaboration(canonical).value_type_by_name
    domains = _rectangular_domains(canonical)
    result_axes: dict[str, tuple[str, ...]] = {}
    for operation in operations:
        result_ref = _result_names(operation)[0]
        value_type = value_types.get(result_ref)
        if value_type is None:
            raise ValueError("contract reference operation lacks a closed result type")
        result_axes[result_ref] = _axes_for_shape(
            value_type,
            domains,
            context=f"J node {operation.node_id!r} result {result_ref!r}",
        )

    try:
        contract_ir = project_contract_ir(canonical)
    except ContractIRProjectionError as exc:
        raise ContractReferenceUnsupportedError(
            "contract reference requires one closed typed ContractIR"
        ) from exc
    input_names = {item.name for item in canonical.intent.inputs}
    output_names = {item.name for item in canonical.intent.outputs}
    loads: list[IndexedLoadProjection] = []
    stores: list[IndexedStoreProjection] = []
    seen_load_edges: set[tuple[str, int]] = set()
    seen_accesses: set[str] = set()
    for access in contract_ir.accesses:
        node_ref = access.node_ref
        operation = nodes.get(node_ref)
        if operation is None or node_ref is None:
            raise ContractReferenceUnsupportedError(
                "contract reference v1 requires every access to name one flat J node"
            )
        result_ref = _result_names(operation)[0]
        axis_refs = set().union(
            *(set(program.expression.references) for program in access.index),
            (
                set(access.predicate.expression.references)
                if access.predicate is not None else set()
            ),
        )
        unknown_axes = (axis_refs & set(domains)) - set(result_axes[result_ref])
        if unknown_axes:
            raise ContractReferenceUnsupportedError(
                "contract access uses domain axes outside its producer result: "
                + ", ".join(sorted(unknown_axes))
            )
        if access.kind == "load":
            if access.tensor not in input_names:
                raise ContractReferenceUnsupportedError(
                    "contract reference v1 loads only logical J inputs"
                )
            tensor_type = value_types.get(access.tensor)
            if tensor_type is None or len(access.index) != len(tensor_type.shape):
                raise ContractReferenceUnsupportedError(
                    "contract indexed load rank differs from its logical J tensor"
                )
            positions = [
                index
                for index, operand in enumerate(operation.operands)
                if operand == access.tensor
            ]
            if len(positions) != 1:
                raise ContractReferenceUnsupportedError(
                    "contract indexed load does not identify one unique operand edge"
                )
            edge = (node_ref, positions[0])
            if edge in seen_load_edges:
                raise ValueError("contract indexed loads duplicate one operand edge")
            seen_load_edges.add(edge)
            if access.predicate is not None and access.fill_value is None:
                raise ValueError("contract predicated load lacks its authored fill")
            loads.append(
                IndexedLoadProjection(
                    access,
                    positions[0],
                    result_ref,
                    result_axes[result_ref],
                )
            )
        elif access.kind == "store":
            if access.tensor not in output_names:
                raise ContractReferenceUnsupportedError(
                    "contract reference v1 stores only logical J outputs"
                )
            tensor_type = value_types.get(access.tensor)
            if tensor_type is None or len(access.index) != len(tensor_type.shape):
                raise ContractReferenceUnsupportedError(
                    "contract indexed store rank differs from its logical J tensor"
                )
            if access.fill_value is not None:
                raise ValueError("contract store must not declare a fill")
            stores.append(
                IndexedStoreProjection(
                    access,
                    result_ref,
                    result_axes[result_ref],
                )
            )
        else:
            raise ContractReferenceUnsupportedError(
                f"contract reference v1 does not implement access kind {access.kind!r}"
            )
        seen_accesses.add(access.access_id)

    if seen_accesses != {item.access_id for item in canonical.boundary.accesses}:
        raise ValueError("contract reference did not consume every authored access")
    stores_by_output = {item.access.tensor for item in stores}
    if stores_by_output != output_names:
        raise ContractReferenceUnsupportedError(
            "contract reference v1 requires authored stores for every output"
        )
    output_bindings = {
        item.output: item.value for item in j_elaboration(canonical).output_bindings
    }
    if any(output_bindings.get(output) != output for output in output_names):
        raise ContractReferenceUnsupportedError(
            "contract indexed stores require identity/inout output bindings"
        )

    effects = {item.access_ref: item for item in boundary.effects}
    if set(effects) != seen_accesses:
        raise ValueError("contract reference requires one effect per access")
    for access in contract_ir.accesses:
        effect = effects[access.access_id]
        expected = "read" if access.kind == "load" else "write"
        if (
            effect.kind.value != expected
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            raise ValueError("contract access/effect relation is inconsistent")

    broadcasts = canonical_broadcast_axes(canonical)
    return ContractReferencePlan(
        contract_ir=contract_ir,
        domains=tuple(sorted(domains.items())),
        loads=tuple(loads),
        stores=tuple(stores),
        broadcast_axes=tuple(
            sorted(
                (
                    result,
                    tuple(sorted((operand, tuple(axes)) for operand, axes in operands.items())),
                )
                for result, operands in broadcasts.items()
            )
        ),
    )


def _resolved_domain_bindings(
    plan: ContractReferencePlan,
    dims: Mapping[str, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for axis, extent in plan.domains:
        if type(extent) is int:
            value = extent
        elif isinstance(extent, str) and extent in dims:
            value = dims[extent]
        else:
            raise ContractReferenceUnsupportedError(
                f"contract reference cannot resolve domain extent {extent!r}"
            )
        if value < 0:
            raise ValueError("contract reference domain has a negative extent")
        result[axis] = int(value)
    return result


def _scalar_bindings(
    inputs: Mapping[str, np.ndarray],
    dims: Mapping[str, int],
) -> dict[str, int | bool]:
    bindings: dict[str, int | bool] = dict(dims)
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in {"b", "i", "u"}:
            continue
        item = array.item()
        bindings[name] = bool(item) if array.dtype.kind == "b" else int(item)
    return bindings


def _evaluate_predicate(
    access: ContractAccessIR,
    bindings: Mapping[str, int | bool],
) -> bool:
    if access.predicate is None:
        return True
    try:
        value = access.predicate.evaluate(
            bindings,
            context=f"J access {access.access_id!r} predicate",
        )
    except ContractSemanticsError as exc:
        raise ContractReferenceUnsupportedError(
            f"contract access {access.access_id!r} predicate has no closed "
            "runtime binding"
        ) from exc
    if type(value) is not bool:
        raise ValueError("contract access predicate did not produce boolean")
    return value


def _evaluate_index(
    access: ContractAccessIR,
    bindings: Mapping[str, int | bool],
) -> tuple[int, ...]:
    result: list[int] = []
    for position, program in enumerate(access.index):
        try:
            value = program.evaluate(
                bindings,
                context=f"J access {access.access_id!r} index[{position}]",
            )
        except ContractSemanticsError as exc:
            raise ContractReferenceUnsupportedError(
                f"contract access {access.access_id!r} index[{position}] has no "
                "closed runtime binding"
            ) from exc
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("contract access index did not produce integer")
        result.append(value)
    return tuple(result)


class _ContractEvaluator(_Evaluator):
    def __init__(
        self,
        j: "SemanticArchive",
        dims: Mapping[str, int],
        plan: ContractReferencePlan,
        scalar_bindings: Mapping[str, int | bool],
    ) -> None:
        super().__init__(j, dims, plan.broadcast_map)
        self.contract_loads = plan.load_map
        self.contract_scalar_bindings = dict(scalar_bindings)
        self.indexed_load_materialization_count = 0

    def _operand(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
        index: int,
    ) -> np.ndarray:
        projection = self.contract_loads.get((op.node_id or "", index))
        if projection is None:
            value = super()._operand(op, env, index)
            try:
                operand_name = op.operands[index]
            except IndexError as exc:
                raise ValueError("contract pointwise op lacks an operand") from exc
            axes = self.broadcast_axes.get(self._result_name(op), {}).get(
                operand_name, ()
            )
            if not axes:
                return value
            # ``_Evaluator``'s legacy insertion loop cannot distinguish a
            # declared axis already occupied by a singleton dimension from a
            # genuinely missing rank dimension.  Recompute the same canonical
            # alignment used by the shared Intent MLIR emitter, then materialize
            # the exact consumer-local view.
            raw = np.asarray(env[operand_name])
            declared = self._declared_type(self._result_name(op), op)
            if declared is None:
                raise ValueError("contract broadcast lacks a closed result type")
            projected_shape, _inserted = _broadcast_projection(
                list(raw.shape), axes, list(declared[1])
            )
            if projected_shape is None:  # pragma: no cover - helper is total/raises
                raise ValueError("contract broadcast shape is not closed")
            return np.broadcast_to(
                raw.reshape(tuple(projected_shape)), declared[1]
            )
        try:
            operand_name = op.operands[index]
            source = np.asarray(env[operand_name])
        except (IndexError, KeyError) as exc:
            raise ValueError("contract indexed load has an unavailable operand") from exc
        if operand_name != projection.access.tensor:
            raise ValueError("contract indexed load names a different operand")
        declared = self._declared_type(projection.result_ref, op)
        if declared is None:
            raise ValueError("contract indexed load lacks a closed result type")
        result_shape = declared[1]
        if len(result_shape) != len(projection.result_axes):
            raise ValueError("contract indexed load result axes differ from result rank")
        fill: np.ndarray | None = None
        if projection.access.predicate is not None:
            try:
                fill = np.asarray(
                    projection.access.fill_value,
                    dtype=source.dtype,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ContractReferenceUnsupportedError(
                    "contract indexed load fill has no typed NumPy denotation"
                ) from exc
            if fill.shape != ():
                raise ContractReferenceUnsupportedError(
                    "contract indexed load v1 requires a scalar fill"
                )
        result = np.empty(result_shape, dtype=source.dtype)
        for coordinate in np.ndindex(result_shape):
            bindings = {
                **self.contract_scalar_bindings,
                **dict(zip(projection.result_axes, coordinate, strict=True)),
            }
            if not _evaluate_predicate(projection.access, bindings):
                assert fill is not None
                result[coordinate] = fill
                continue
            source_index = _evaluate_index(projection.access, bindings)
            if len(source_index) != source.ndim or any(
                item < 0 or item >= source.shape[axis]
                for axis, item in enumerate(source_index)
            ):
                raise ValueError("contract indexed load is outside its J tensor")
            result[coordinate] = source[source_index]
        self.indexed_load_materialization_count += 1
        return result


def execute_contract_reference(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    plan: ContractReferencePlan | None = None,
) -> CPUExecutionResult:
    """Execute one relation-indexed plan using only J and invocation bindings."""

    canonical = elaborate_j(j)
    selected = plan or project_contract_reference(canonical)
    if len(inputs) != len(canonical.intent.inputs):
        raise ValueError("contract reference input arity differs from J")
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
            "contract reference dims must be non-negative integer bindings"
        )
    checked_dims = resolve_dimension_relations(canonical, checked_dims)
    checked_inputs = {
        tensor.name: validate_array(
            np.asarray(value),
            shape=_resolved_shape(tensor.shape, checked_dims),
            dtype=tensor.dtype,
            context=f"contract input {tensor.name!r}",
        )
        for tensor, value in zip(canonical.intent.inputs, inputs, strict=True)
    }
    validate_value_dimension_instantiation(canonical, checked_dims, checked_inputs)
    domain_extents = _resolved_domain_bindings(selected, checked_dims)
    scalar_bindings = _scalar_bindings(checked_inputs, checked_dims)
    evaluator = _ContractEvaluator(
        canonical,
        checked_dims,
        selected,
        scalar_bindings,
    )
    env = evaluator.sequence(canonical.intent.body, checked_inputs, allow_loop=False)

    value_types = j_elaboration(canonical).value_type_by_name
    outputs: dict[str, np.ndarray] = {}
    stores_by_output = selected.stores_by_output
    executed_stores = 0
    for tensor in canonical.intent.outputs:
        shape = _resolved_shape(tensor.shape, checked_dims)
        output = np.empty(shape, dtype=numpy_dtype(tensor.dtype))
        covered = np.zeros(shape, dtype=np.bool_)
        for store in stores_by_output[tensor.name]:
            try:
                value = np.asarray(env[store.value_ref])
            except KeyError as exc:
                raise ValueError("contract output store producer is unavailable") from exc
            declared = value_types.get(store.value_ref)
            if declared is None:
                raise ValueError("contract output store producer lacks a closed type")
            producer_shape = _resolved_shape(declared.shape, checked_dims)
            if value.shape != producer_shape or len(value.shape) != len(store.value_axes):
                raise ValueError("contract output store producer type is inconsistent")
            for coordinate in np.ndindex(producer_shape):
                bindings = {
                    **scalar_bindings,
                    **dict(zip(store.value_axes, coordinate, strict=True)),
                }
                if not _evaluate_predicate(store.access, bindings):
                    continue
                target_index = _evaluate_index(store.access, bindings)
                if len(target_index) != len(shape) or any(
                    item < 0 or item >= shape[axis]
                    for axis, item in enumerate(target_index)
                ):
                    raise ValueError("contract output store is outside its J tensor")
                if covered[target_index]:
                    raise ValueError("contract output stores overlap")
                output[target_index] = value[coordinate]
                covered[target_index] = True
                executed_stores += 1
        if not bool(np.all(covered)):
            raise ContractReferenceUnsupportedError(
                "contract output stores do not exhaust the logical output"
            )
        outputs[tensor.name] = validate_array(
            output,
            shape=shape,
            dtype=tensor.dtype,
            context=f"contract output {tensor.name!r}",
        )

    plan_record = selected.to_json()
    return CPUExecutionResult(
        outputs,
        {
            "protocol": CONTRACT_REFERENCE_PROTOCOL,
            "authority": "canonical_J_contract_ir_only",
            "profile": "relation_indexed_pointwise_v1",
            "plan_digest": plan_record["plan_digest"],
            "contract_ir_digest": selected.contract_ir.digest,
            "domain_axis_count": len(domain_extents),
            "indexed_load_count": evaluator.indexed_load_materialization_count,
            "executed_store_element_count": executed_stores,
            "operation_count": evaluator.operation_count,
            "input_count": len(checked_inputs),
            "output_count": len(outputs),
        },
    )


__all__ = [
    "CONTRACT_REFERENCE_PROTOCOL",
    "ContractReferencePlan",
    "ContractReferenceUnsupportedError",
    "IndexedLoadProjection",
    "IndexedStoreProjection",
    "execute_contract_reference",
    "project_contract_reference",
]
