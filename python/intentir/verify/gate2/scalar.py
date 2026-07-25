"""Simple backend-neutral ``E_J`` that consumes the complete canonical J."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.backend.cpu.abi import CPUABIValidationError, CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.backend.cpu.structured_reference import (
    execute_structured_reference,
    requires_structured_reference,
)
from intentir.ir.canonical_j import (
    ContractExpr,
    ContractSort,
    elaborate_j,
    j_elaboration,
    parse_contract_expr,
    parse_index_expression,
)
from intentir.ir.dim_expr import validate_dimension_instantiation
from intentir.ir.functional_boundary import (
    parse_direct_axes,
    project_functional_boundary,
    project_functional_boundary_for_refeval,
    project_predicate_text,
)
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import (
    AccessKind,
    EffectKind,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
)

from .model import ExecutionResult


SCALAR_LOWERING_PROTOCOL = "j_scalar_cpu_mlir_v11"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _canonical_int_mapping(value: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for key, item in value.items()
    ):
        raise ValueError("scalar dims must map names to non-negative integers")
    return tuple(sorted(value.items()))


def _canonical_str_mapping(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in value.items()
    ):
        raise ValueError("scalar bindings must map non-empty names")
    return tuple(sorted(value.items()))


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


def _node_table(j: SemanticArchive) -> dict[str, IntentOp]:
    by_path: dict[str, IntentOp] = {}

    def visit(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            by_path[path] = op
            visit(op.body, f"{path}.body")

    visit(j.intent.body)
    result: dict[str, IntentOp] = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(op.result)
    return ()


def _parse_direct_index(expression: str) -> tuple[str, ...]:
    """Compatibility helper; the shared projection owns interpretation."""

    return parse_direct_axes(expression)


def _flatten_conjunction(expression: ContractExpr) -> tuple[ContractExpr, ...]:
    if expression.op != "and":
        return (expression,)
    result: list[ContractExpr] = []
    for argument in expression.args:
        result.extend(_flatten_conjunction(argument))
    return tuple(result)


def _reference(expression: ContractExpr) -> str | None:
    if expression.op == "ref" and isinstance(expression.value, str):
        return expression.value
    return None


def _bound_value(expression: ContractExpr) -> int | str | None:
    if expression.op == "int" and isinstance(expression.value, int):
        return expression.value
    return _reference(expression)


def _domain_predicate_contract(
    expression: ContractExpr,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> tuple[
    frozenset[str],
    frozenset[str],
    tuple[tuple[str, int | str], ...],
]:
    """Project the typed AST onto the mask-erasure profile understood by E_J."""

    lower: set[str] = set()
    guards: set[str] = set()
    upper: dict[str, int | str] = {}
    for term in _flatten_conjunction(expression):
        guard = _reference(term)
        if guard is None and term.op == "not" and len(term.args) == 1:
            guard = _reference(term.args[0])
        if guard in logical_guards:
            guards.add(guard)
            continue

        left, right = term.args if len(term.args) == 2 else (None, None)
        lower_axis: str | None = None
        if term.op == "le" and left is not None and right is not None:
            if left.op == "int" and left.value == 0:
                lower_axis = _reference(right)
        elif term.op == "ge" and left is not None and right is not None:
            if right.op == "int" and right.value == 0:
                lower_axis = _reference(left)
        if lower_axis is not None:
            lower.add(lower_axis)
            continue

        upper_axis: str | None = None
        extent: int | str | None = None
        if term.op == "lt" and left is not None and right is not None:
            upper_axis = _reference(left)
            extent = _bound_value(right)
        if upper_axis is None or extent is None:
            raise NotImplementedError(
                "E_J mask erasure supports only logical-domain bound predicates"
            )
        previous = upper.get(upper_axis)
        if previous is not None and previous != extent:
            raise ValueError(
                f"E_J predicate axis {upper_axis!r} has conflicting extents"
            )
        upper[upper_axis] = extent
    if not upper and not guards:
        raise NotImplementedError(
            "E_J mask erasure requires an upper logical-domain bound"
        )
    return (
        frozenset(guards),
        frozenset(lower),
        tuple(sorted(upper.items())),
    )


def _parse_domain_predicate(
    expression: str,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> tuple[
    frozenset[str],
    frozenset[str],
    tuple[tuple[str, int | str], ...],
]:
    projection = project_predicate_text(
        expression,
        logical_guards=logical_guards,
    )
    return (
        projection.guards,
        projection.lower_axes,
        projection.upper_bounds,
    )


def _logical_scalar_inputs(j: SemanticArchive) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind in {ParameterKind.SCALAR, ParameterKind.CONSTEXPR}
        and parameter.logical_ref in inputs
        and not inputs[parameter.logical_ref].shape
    )


def _logical_boolean_guards(j: SemanticArchive) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        name
        for name in _logical_scalar_inputs(j)
        if inputs[name].dtype == "i1"
    )


def _canonical_full_domains(j: SemanticArchive) -> dict[str, int | str]:
    """Project the one accepted J onto complete logical axis bounds."""

    elaboration = j_elaboration(j)
    result: dict[str, int | str] = {}
    logical_guards = _logical_boolean_guards(j)
    for domain_id, expression in elaboration.domains:
        try:
            guards, lower, upper = _domain_predicate_contract(
                expression, logical_guards=logical_guards
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "E_J dense domain requires complete logical axis bounds; "
                f"unsupported domain {domain_id!r}"
            ) from exc
        if guards and not upper:
            continue
        if set(axis for axis, _ in upper) - set(lower):
            raise NotImplementedError(
                f"E_J dense domain {domain_id!r} lacks a lower zero bound"
            )
        for axis, extent in upper:
            previous = result.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"E_J domain axis {axis!r} has conflicting extents"
                )
            result[axis] = extent
    return result


def _parse_full_domains(j: SemanticArchive) -> dict[str, int | str]:
    """Compatibility API; consumers elaborate once and use the typed view."""

    return _canonical_full_domains(elaborate_j(j))


def _validate_masked_sum_reduction(
    j: SemanticArchive,
    *,
    domains: Mapping[str, int | str],
    axes_by_access: Mapping[str, tuple[str, ...]],
    predicates_by_access: Mapping[
        str,
        tuple[
            frozenset[str],
            frozenset[str],
            tuple[tuple[str, int | str], ...],
        ]
        | None,
    ],
) -> None:
    """Validate the narrow padded single-axis sum projection supported by E_J."""

    boundary = j.boundary
    if (
        len(j.intent.inputs) != 1
        or len(j.intent.outputs) != 1
        or len(boundary.reductions) != 1
    ):
        raise NotImplementedError(
            "E_J padded reduction requires one input, output, and contract"
        )
    if (
        boundary.aliases
        or boundary.broadcasts
        or boundary.index_bounds
        or boundary.updates
    ):
        raise NotImplementedError(
            "E_J padded reduction does not implement auxiliary boundary relations"
        )
    loads = [item for item in boundary.accesses if item.kind is AccessKind.LOAD]
    stores = [item for item in boundary.accesses if item.kind is AccessKind.STORE]
    if len(loads) != 1 or len(stores) != 1:
        raise NotImplementedError(
            "E_J padded reduction requires one dense load and one store"
        )
    load, store = loads[0], stores[0]
    if load.tensor != j.intent.inputs[0].name or store.tensor != j.intent.outputs[0].name:
        raise ValueError("E_J padded reduction access tensors differ from J ABI")
    if (
        isinstance(load.fill_value, bool)
        or not isinstance(load.fill_value, (int, float))
        or float(load.fill_value) != 0.0
    ):
        raise NotImplementedError(
            "E_J padded sum requires an explicit zero masked-load fill"
        )
    if store.predicate is not None or store.fill_value is not None:
        raise NotImplementedError(
            "E_J padded reduction requires an unpredicated output store"
        )

    nodes = _node_table(j)
    contract = boundary.reductions[0]
    reduction = nodes.get(contract.node_ref)
    if reduction is None or reduction.op != "reduce":
        raise ValueError("E_J padded reduction contract does not target reduce")
    if reduction.attrs.get("fn") != "sum":
        raise NotImplementedError(
            "E_J padded reduction currently supports sum only"
        )
    reduction_axes = reduction.attrs.get("axes")
    if (
        not isinstance(reduction_axes, list)
        or len(reduction_axes) != 1
        or isinstance(reduction_axes[0], bool)
        or not isinstance(reduction_axes[0], int)
    ):
        raise NotImplementedError(
            "E_J padded reduction requires one canonical integer axis"
        )
    reduced_axis = reduction_axes[0]
    input_tensor = j.intent.inputs[0]
    output_tensor = j.intent.outputs[0]
    if input_tensor.dtype != output_tensor.dtype:
        raise NotImplementedError(
            "E_J padded sum currently requires equal input/output dtype"
        )
    if reduced_axis < 0 or reduced_axis >= len(input_tensor.shape):
        raise ValueError("E_J padded reduction axis is outside input rank")
    expected_output_shape = [
        extent
        for index, extent in enumerate(input_tensor.shape)
        if index != reduced_axis
    ]
    if output_tensor.shape != expected_output_shape:
        raise ValueError("E_J padded reduction output shape is not axis removal")

    load_axes = axes_by_access[load.access_id]
    store_axes = axes_by_access[store.access_id]
    if set(domains) != set(load_axes):
        raise NotImplementedError(
            "E_J padded reduction requires a full domain for every input axis"
        )
    expected_store_axes = tuple(
        axis for index, axis in enumerate(load_axes) if index != reduced_axis
    )
    if store_axes != expected_store_axes:
        raise ValueError("E_J padded reduction store indices are not axis removal")
    predicate = predicates_by_access[load.access_id]
    if predicate is None:
        raise NotImplementedError("E_J padded reduction requires a load predicate")
    guards, lower_axes, upper_bounds = predicate
    if guards:
        raise NotImplementedError(
            "E_J padded reduction does not consume logical guard predicates"
        )
    reduced_index = load_axes[reduced_axis]
    expected_bound = (reduced_index, input_tensor.shape[reduced_axis])
    if upper_bounds != (expected_bound,) or not lower_axes.issubset({reduced_index}):
        raise ValueError(
            "E_J padded reduction predicate does not bind the reduced axis"
        )
    if predicates_by_access[store.access_id] is not None:
        raise NotImplementedError(
            "E_J padded reduction output store must be unpredicated"
        )
    if store.node_ref is None or nodes.get(store.node_ref) is not reduction:
        raise ValueError("E_J padded reduction store does not target reduce result")
    if not isinstance(reduction.result, str):
        raise NotImplementedError("E_J padded reduction requires one result")
    returns = {item.output: item.value for item in j.intent.returns}
    if returns != {output_tensor.name: reduction.result}:
        raise ValueError("E_J padded reduction result is not the J output")
    if (
        len(boundary.output_mappings) != 1
        or boundary.output_mappings[0].output != output_tensor.name
        or boundary.output_mappings[0].value != reduction.result
        or boundary.output_mappings[0].node_ref != contract.node_ref
    ):
        raise ValueError(
            "E_J padded reduction requires one explicit reduction output mapping"
        )

    body = j.intent.body
    if not body or body[-1] is not reduction or any(op.body for op in body):
        raise NotImplementedError(
            "E_J padded reduction requires one flat graph ending in reduction"
        )
    if sum(op.op == "reduce" for op in body) != 1:
        raise NotImplementedError(
            "E_J padded reduction requires exactly one reduction node"
        )
    if len(reduction.operands) != 1:
        raise NotImplementedError(
            "E_J padded reduction requires one reduced value"
        )
    producers: dict[str, IntentOp] = {}
    for op in body:
        results = _result_names(op)
        if len(results) != 1:
            raise NotImplementedError(
                "E_J padded reduction requires single-result value nodes"
            )
        producers[results[0]] = op
    required_prefix: set[int] = set()
    pending = list(reduction.operands)
    while pending:
        value = pending.pop()
        producer = producers.get(value)
        if producer is None:
            if value != input_tensor.name:
                raise ValueError(
                    "E_J padded reduction dataflow has a non-input leaf"
                )
            continue
        if producer is reduction:
            raise ValueError("E_J padded reduction dataflow is cyclic")
        marker = id(producer)
        if marker in required_prefix:
            continue
        required_prefix.add(marker)
        pending.extend(producer.operands)
    if required_prefix != {id(op) for op in body[:-1]}:
        raise NotImplementedError(
            "E_J padded reduction requires every prefix node to feed reduction"
        )
    load_owner = nodes.get(load.node_ref) if load.node_ref is not None else None
    if load_owner is not reduction and id(load_owner) not in required_prefix:
        raise ValueError(
            "E_J padded reduction load does not feed the reduction slice"
        )
    for op in body[:-1]:
        if (
            op.type is None
            or op.type.get("dtype") != input_tensor.dtype
            or op.type.get("shape") != input_tensor.shape
        ):
            raise NotImplementedError(
                "E_J padded reduction prefix must preserve input dtype and shape"
            )
    padded_values: dict[str, float] = {
        input_tensor.name: float(load.fill_value),
    }
    for op in body[:-1]:
        operands = [padded_values.get(value) for value in op.operands]
        if any(value is None for value in operands):
            raise NotImplementedError(
                "E_J cannot prove the padded contribution through this dataflow"
            )
        if (
            op.op == "map_binary"
            and op.attrs.get("fn") == "mul"
            and len(operands) == 2
        ):
            padded_result = float(operands[0]) * float(operands[1])
        elif (
            op.op == "map_unary"
            and op.attrs.get("fn") == "identity"
            and len(operands) == 1
        ):
            padded_result = float(operands[0])
        else:
            raise NotImplementedError(
                "E_J padded reduction supports only a zero-stable identity/mul prefix"
            )
        padded_values[_result_names(op)[0]] = padded_result
    padded_contribution = padded_values.get(reduction.operands[0])
    if padded_contribution != 0.0:
        raise NotImplementedError(
            "E_J padded reduction prefix does not map fill to sum identity"
        )
    if (
        reduction.type is None
        or reduction.type.get("dtype") != output_tensor.dtype
        or reduction.type.get("shape") != output_tensor.shape
    ):
        raise ValueError(
            "E_J padded reduction result type differs from the output ABI"
        )

    if contract.tile_axis != 0:
        raise NotImplementedError(
            "E_J padded reduction requires the single physical tile axis 0"
        )
    for name, value in (
        ("identity", contract.identity),
        ("masked_contribution", contract.masked_contribution),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) != 0.0
        ):
            raise NotImplementedError(
                f"E_J padded sum requires explicit neutral {name}=0"
            )


def _validate_dense_functional_boundary(j: SemanticArchive) -> None:
    """Consume the dense access/effect projection represented by pure J values.

    ``IntentFunction`` evaluates complete logical tensors.  Therefore an
    explicit memory boundary can be projected to that evaluator only when it
    proves a full, direct, unpredicated tensor access and its effects agree
    exactly with the functional inputs/outputs.  Anything more expressive is
    rejected rather than silently ignored.
    """

    elaboration = j_elaboration(j)
    boundary = j.boundary
    if any(
        parameter.mode is ParameterMode.INOUT or parameter.alias_of is not None
        for parameter in j.signature.parameters
    ):
        raise NotImplementedError(
            "E_J dense functional boundary does not implement inout/alias ABI"
        )
    if not boundary.accesses:
        if boundary.domains or boundary.effects:
            raise ValueError(
                "E_J boundary domains/effects require explicit access relations"
            )
        return

    domains = _canonical_full_domains(j)
    has_predicate_or_fill = any(
        access.predicate is not None or access.fill_value is not None
        for access in boundary.accesses
    )
    masked_reduction = has_predicate_or_fill and bool(boundary.reductions)
    if has_predicate_or_fill and boundary.updates:
        raise NotImplementedError(
            "E_J mask erasure does not implement update padding"
        )
    nodes = _node_table(j)
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    input_names = {item.name for item in j.intent.inputs}
    output_names = {item.name for item in j.intent.outputs}
    returns = {item.output: item.value for item in j.intent.returns}
    graph_operands = {
        operand
        for operation in _node_table(j).values()
        for operand in operation.operands
    }
    loads_by_tensor: dict[str, list[Any]] = {}
    for access in boundary.accesses:
        if access.kind is AccessKind.LOAD:
            loads_by_tensor.setdefault(access.tensor, []).append(access)
    logical_scalar_inputs = _logical_scalar_inputs(j)
    logical_guards = _logical_boolean_guards(j)
    mappings = {item.output: item for item in boundary.output_mappings}
    for output, mapping in mappings.items():
        if output in returns and mapping.value != returns[output]:
            raise ValueError(
                "E_J output mapping differs from the J return value"
            )
    access_ids = [item.access_id for item in boundary.accesses]
    if len(access_ids) != len(set(access_ids)):
        raise ValueError("E_J dense functional boundary has duplicate access ids")
    effect_ids = [item.effect_id for item in boundary.effects]
    if len(effect_ids) != len(set(effect_ids)):
        raise ValueError("E_J dense functional boundary has duplicate effect ids")
    effects_by_access: dict[str, list[Any]] = {}
    for effect in boundary.effects:
        if effect.access_ref is None:
            raise NotImplementedError(
                "E_J dense functional boundary requires access-bound effects"
            )
        effects_by_access.setdefault(effect.access_ref, []).append(effect)

    consumed_domain_axes: set[str] = set()
    inferred_axis_extents: dict[str, int | str] = dict(domains)
    predicate_contracts: dict[
        str,
        tuple[
            frozenset[str],
            frozenset[str],
            tuple[tuple[str, int | str], ...],
        ]
        | None
    ] = {}
    axes_by_access: dict[str, tuple[str, ...]] = {}
    loaded: set[str] = set()
    store_counts = {name: 0 for name in output_names}
    expected_effect = {
        AccessKind.LOAD: EffectKind.READ,
        AccessKind.STORE: EffectKind.WRITE,
    }
    canonical_accesses = elaboration.access_by_id
    for access in boundary.accesses:
        if access.kind is AccessKind.UPDATE:
            raise NotImplementedError(
                "E_J dense functional boundary does not implement update access"
            )
        op = nodes.get(access.node_ref) if access.node_ref is not None else None
        if access.node_ref is not None and op is None:
            raise ValueError(
                f"E_J access {access.access_id!r} references an unknown J node"
            )
        tensor = tensors.get(access.tensor)
        if tensor is None:
            raise ValueError(
                f"E_J access {access.access_id!r} references an unknown tensor"
            )
        canonical_access = canonical_accesses[access.access_id]
        axes = canonical_access.index.direct_axes
        if axes is None:
            raise NotImplementedError(
                "E_J dense access requires a direct ordered axis vector"
            )
        if len(set(axes)) != len(axes):
            raise ValueError("E_J dense access axes must be unique")
        axes_by_access[access.access_id] = axes
        if len(axes) != len(tensor.shape):
            raise ValueError(
                f"E_J access {access.access_id!r} rank differs from J tensor"
            )
        for axis, declared_extent in zip(axes, tensor.shape, strict=True):
            if axis == "0" and declared_extent != 1:
                raise ValueError(
                    "E_J constant zero index is valid only for a singleton axis"
                )
            # A direct axis over a typed J tensor is already bounded by that
            # tensor's logical shape.  When B also spells out a domain, consume
            # it and require exact agreement instead of treating it as an
            # alternative source of extent information.
            actual_extent = inferred_axis_extents.get(axis, declared_extent)
            if actual_extent != declared_extent:
                raise ValueError(
                    f"E_J access axis {axis!r} extent {actual_extent!r} differs "
                    f"from J tensor extent {declared_extent!r}"
                )
            inferred_axis_extents.setdefault(axis, declared_extent)
            if axis in domains:
                consumed_domain_axes.add(axis)

        predicate_contract = (
            _domain_predicate_contract(
                canonical_access.predicate,
                logical_guards=logical_guards,
            )
            if canonical_access.predicate is not None else None
        )
        predicate_contracts[access.access_id] = predicate_contract
        if predicate_contract is not None:
            guards, lower_axes, upper_bounds = predicate_contract
            graph_operands = {
                operand for node in j.intent.body for operand in node.operands
            }
            if not guards.issubset(graph_operands):
                raise ValueError(
                    "E_J access guard is absent from J value dataflow"
                )
            if not lower_axes.issubset(set(axes)):
                raise ValueError("E_J predicate lower bound uses a non-index axis")
            for axis, extent in upper_bounds:
                if axis not in axes:
                    raise ValueError(
                        "E_J predicate upper bound uses a non-index axis"
                    )
                if domains.get(axis) != extent:
                    raise ValueError(
                        "E_J access predicate differs from the full logical domain"
                    )
            if (
                upper_bounds
                and access.kind is AccessKind.LOAD
                and access.fill_value is None
            ):
                raise NotImplementedError(
                    "E_J mask erasure requires an explicit masked-load fill"
                )
        elif access.fill_value is not None:
            raise ValueError("E_J unpredicated access must not declare fill")
        if access.kind is not AccessKind.LOAD and access.fill_value is not None:
            raise ValueError("E_J store/update access must not declare fill")

        matching_effects = effects_by_access.get(access.access_id, [])
        if len(matching_effects) != 1:
            raise ValueError(
                f"E_J access {access.access_id!r} requires exactly one effect"
            )
        effect = matching_effects[0]
        if (
            effect.kind is not expected_effect[access.kind]
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            raise ValueError(
                f"E_J access {access.access_id!r} disagrees with its effect"
            )

        if access.kind is AccessKind.LOAD:
            if access.tensor not in input_names:
                raise ValueError(
                    f"E_J load {access.access_id!r} is not a J input"
                )
            if op is not None and access.tensor not in op.operands:
                raise ValueError(
                    f"E_J load {access.access_id!r} is not a direct J operand"
                )
            if op is None:
                if len(loads_by_tensor[access.tensor]) != 1:
                    raise ValueError(
                        "E_J node-less input load must be the sole load for its "
                        "J tensor"
                    )
                if (
                    access.tensor not in graph_operands
                    and (
                        j.intent.body
                        or access.tensor not in set(returns.values())
                    )
                ):
                    raise ValueError(
                        "E_J node-less input load is absent from J value dataflow"
                    )
            loaded.add(access.tensor)
            continue

        if access.tensor not in output_names:
            raise ValueError(
                f"E_J store {access.access_id!r} does not target a J output"
            )
        observable_value = returns.get(access.tensor, access.tensor)
        mapping = mappings.get(access.tensor)
        if mapping is not None:
            observable_value = mapping.value
            if mapping.node_ref is not None and nodes.get(mapping.node_ref) is not op:
                raise ValueError(
                    f"E_J store {access.access_id!r} disagrees with output mapping"
                )
        if op is not None:
            if observable_value not in _result_names(op):
                raise ValueError(
                    f"E_J store {access.access_id!r} is not produced by its J node"
                )
            if has_predicate_or_fill and not masked_reduction and (
                op.type is None
                or op.type.get("dtype") != tensor.dtype
                or op.type.get("shape") != tensor.shape
            ):
                raise ValueError(
                    "E_J store producer type differs from the output tensor"
                )
        elif (
            j.intent.body
            or mapping is None
            or observable_value not in input_names
        ):
            raise NotImplementedError(
                "E_J node-less store is supported only for direct identity J"
            )
        store_counts[access.tensor] += 1

    if set(effects_by_access) != {item.access_id for item in boundary.accesses}:
        raise ValueError("E_J contains an effect for an unknown access")
    if loaded != input_names - set(logical_scalar_inputs):
        raise NotImplementedError(
            "E_J dense functional boundary must cover every logical input"
        )
    if any(count != 1 for count in store_counts.values()):
        raise NotImplementedError(
            "E_J dense functional boundary requires exactly one store per "
            "logical output"
        )
    if consumed_domain_axes != set(domains):
        raise NotImplementedError(
            "E_J dense functional boundary contains an unconsumed domain axis"
        )
    concrete_predicates = [item for item in predicate_contracts.values() if item]
    if masked_reduction:
        _validate_masked_sum_reduction(
            j,
            domains=domains,
            axes_by_access=axes_by_access,
            predicates_by_access=predicate_contracts,
        )
    elif concrete_predicates:
        masked_axes = {
            axis
            for _guards, _lower, upper in concrete_predicates
            for axis, _extent in upper
        }
        for access in boundary.accesses:
            axes = axes_by_access[access.access_id]
            contract = predicate_contracts[access.access_id]
            upper = dict(contract[2]) if contract is not None else {}
            for axis in masked_axes.intersection(axes):
                if upper.get(axis) != domains.get(axis):
                    raise NotImplementedError(
                        "E_J mask erasure requires the same predicate on every "
                        "access of a masked axis"
                    )


def _validate_complete_boundary(j: SemanticArchive) -> str:
    """Compatibility wrapper over the shared J boundary projection."""

    return project_functional_boundary(j).boundary_digest


def scalar_index_bounds(j: SemanticArchive) -> frozenset[str]:
    """Return the exact emitter projection authorized by canonical J."""

    return project_functional_boundary(j).proven_index_bounds


def validate_index_bounds_inputs(
    j: SemanticArchive,
    values: Mapping[str, np.ndarray],
) -> None:
    """Check J's indexed-domain preconditions before native scalar execution."""

    for item in j.boundary.index_bounds:
        if item.data not in values or item.indices not in values:
            raise NotImplementedError(
                "E_J runtime bounds checking currently requires data and indices "
                "to be function inputs"
            )
        data = np.asarray(values[item.data])
        indices = np.asarray(values[item.indices])
        if item.axis >= data.ndim:
            raise ValueError("E_J indexed data rank violates canonical J")
        if indices.dtype.kind not in {"i", "u"}:
            raise ValueError("E_J indices must have an integer dtype")
        if bool(np.any(indices < 0)) or bool(np.any(indices >= data.shape[item.axis])):
            raise ValueError(
                f"J index-bounds precondition failed at node {item.node_ref!r}"
            )


def scalar_broadcast_axes(
    j: SemanticArchive,
) -> dict[str, dict[str, tuple[int, ...]]]:
    return project_functional_boundary(j).emitter_broadcast_axes()


@dataclass(frozen=True)
class ScalarLoweringSpec:
    """Invocation request holding the one elaborated J used by all consumers."""

    j: SemanticArchive
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]
    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("scalar lowering requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        dims = dict(self.dims)
        inputs = dict(self.input_bindings)
        outputs = dict(self.output_bindings)
        object.__setattr__(self, "dims", _canonical_int_mapping(dims))
        object.__setattr__(self, "input_bindings", _canonical_str_mapping(inputs))
        object.__setattr__(self, "output_bindings", _canonical_str_mapping(outputs))
        validate_dimension_instantiation(self.j, dims)
        expected_inputs = {item.name for item in self.j.intent.inputs}
        expected_outputs = {item.name for item in self.j.intent.outputs}
        if set(inputs) != expected_inputs:
            raise ValueError("scalar input bindings differ from J inputs")
        if set(outputs) != expected_outputs:
            raise ValueError("scalar output bindings differ from J outputs")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("scalar output destinations must be unique")

    @classmethod
    def create(
        cls,
        j: SemanticArchive,
        *,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
    ) -> "ScalarLoweringSpec":
        return cls(
            j,
            tuple(dims.items()),
            tuple(input_bindings.items()),
            tuple(output_bindings.items()),
        )

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

    @property
    def dims_map(self) -> dict[str, int]:
        return dict(self.dims)

    @property
    def input_binding_map(self) -> dict[str, str]:
        return dict(self.input_bindings)

    @property
    def output_binding_map(self) -> dict[str, str]:
        return dict(self.output_bindings)

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "j_digest": self.j_digest,
                "dims": self.dims_map,
                "input_bindings": self.input_binding_map,
                "output_bindings": self.output_binding_map,
            }
        )


@dataclass(frozen=True)
class ScalarLoweringUnavailable:
    j: SemanticArchive
    error: str
    failure_class: str = "j_lowering_setup"
    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("unavailable RefEval request requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        if not self.error:
            raise ValueError("unavailable E_J request requires an error")

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "availability": "unavailable",
                "j_digest": self.j_digest,
                "failure_class": self.failure_class,
                "error": self.error,
            }
        )

    def execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            "j",
            False,
            error=self.error,
            unsupported=True,
            metadata={
                "protocol": SCALAR_LOWERING_PROTOCOL,
                "j_digest": self.j_digest,
                "scalar_spec_digest": self.spec_digest,
                "failure_class": self.failure_class,
            },
        )


ScalarLoweringRequest = ScalarLoweringSpec | ScalarLoweringUnavailable


class ScalarLoweringRunner:
    def run(self, spec: ScalarLoweringSpec, values: Mapping[str, Any]) -> ExecutionResult:
        try:
            raw_inputs = [
                values[spec.input_binding_map[tensor.name]]
                for tensor in spec.j.intent.inputs
            ]
        except KeyError as exc:
            return ExecutionResult("j", False, error=f"missing E_J input {exc.args[0]!r}")
        try:
            raw_inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(
                    spec.j.intent.inputs, raw_inputs, strict=True
                )
            }
            validate_value_dimension_instantiation(
                spec.j, spec.dims_map, raw_inputs_by_ref
            )
            inputs = [
                (
                    value.detach().cpu().numpy()
                    if type(value).__module__.startswith("torch")
                    and hasattr(value, "detach")
                    else np.asarray(value)
                )
                for value in raw_inputs
            ]
            inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(spec.j.intent.inputs, inputs, strict=True)
            }
            projection = project_functional_boundary_for_refeval(
                spec.j,
                inputs_by_ref,
                spec.dims_map,
            )
            boundary_digest = projection.boundary_digest
            validate_index_bounds_inputs(spec.j, inputs_by_ref)
            broadcast_axes = projection.emitter_broadcast_axes()
            structured = requires_structured_reference(spec.j)
            execution = (
                execute_structured_reference(
                    spec.j,
                    spec.dims_map,
                    inputs,
                    broadcast_axes=broadcast_axes,
                )
                if structured
                else execute_on_cpu_outputs(
                    spec.j,
                    spec.dims_map,
                    inputs,
                    broadcast_axes=broadcast_axes,
                    proven_index_bounds=projection.proven_index_bounds,
                )
            )
            output_bindings = spec.output_binding_map
            outputs = {
                output_bindings[name]: np.asarray(value)
                for name, value in execution.outputs.items()
            }
            if set(outputs) != set(output_bindings.values()):
                raise ValueError("E_J did not produce every observable output")
            return ExecutionResult(
                "j",
                True,
                outputs,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "path": (
                        "canonical J -> typed structured CPU reference"
                        if structured
                        else "canonical J -> intent MLIR -> scalar CPU"
                    ),
                    "function": spec.j.intent.name,
                    "j_digest": spec.j_digest,
                    "boundary_digest": boundary_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "boundary_profile": projection.profile,
                    "boundary_projection": projection.to_json(),
                    "domain_relation_count": len(spec.j.boundary.domains),
                    "access_relation_count": len(spec.j.boundary.accesses),
                    "effect_relation_count": len(spec.j.boundary.effects),
                    "predicated_access_count": projection.predicated_access_count,
                    "erased_fill_count": projection.erased_fill_count,
                    "runtime_predicate_proof_count": len(
                        projection.runtime_predicate_proofs
                    ),
                    "broadcast_consumer_count": len(broadcast_axes),
                    "index_bounds_count": len(spec.j.boundary.index_bounds),
                    "result_relation_count": len(
                        spec.j.boundary.result_relations
                    ),
                    "execution_evidence": dict(execution.metadata),
                },
            )
        except (CPUBackendUnsupportedError, NotImplementedError) as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                },
            )
        except CPUABIValidationError as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"failure_class": "physical_cpu_abi"},
            )
        except Exception as exc:
            return ExecutionResult(
                "j", False, error=f"{type(exc).__name__}: {exc}"
            )


__all__ = [
    "SCALAR_LOWERING_PROTOCOL", "ScalarLoweringRequest",
    "ScalarLoweringRunner", "ScalarLoweringSpec", "ScalarLoweringUnavailable",
    "scalar_broadcast_axes", "scalar_index_bounds",
    "validate_index_bounds_inputs",
]
