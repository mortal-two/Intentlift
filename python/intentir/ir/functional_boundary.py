"""Backend-neutral projection of J's complete functional memory boundary.

This is a deterministic compiler analysis over the one accepted J.  It is not
serialized as an artifact and has no identity outside the analysis invocation.
Both RefEval and production lowering use it to prove that explicit Layer-B
domain/access/effect relations can be represented by the complete logical
tensors of ``IntentFunction`` without silently dropping semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_j import (
    ContractExpr,
    ContractSort,
    elaborate_j,
    j_elaboration,
    parse_contract_expr,
    parse_index_expression,
)
from intentir.ir.model import IntentOp


FUNCTIONAL_BOUNDARY_PROTOCOL = "intentir.j-functional-boundary.v1"


@dataclass(frozen=True)
class PredicateProjection:
    guards: frozenset[str]
    lower_axes: frozenset[str]
    upper_bounds: tuple[tuple[str, int | str], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "guards": sorted(self.guards),
            "lower_axes": sorted(self.lower_axes),
            "upper_bounds": [list(item) for item in self.upper_bounds],
        }


@dataclass(frozen=True)
class FunctionalBoundaryProjection:
    """One ephemeral proof that Layer B projects to logical tensor execution."""

    profile: str
    domains: tuple[tuple[str, int | str], ...]
    access_axes: tuple[tuple[str, tuple[str, ...]], ...]
    predicates: tuple[tuple[str, PredicateProjection | None], ...]
    broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]
    proven_index_bounds: frozenset[str]
    consumed: tuple[str, ...]
    boundary_digest: str
    predicated_access_count: int
    erased_fill_count: int

    @property
    def access_axes_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.access_axes)

    @property
    def predicate_map(self) -> dict[str, PredicateProjection | None]:
        return dict(self.predicates)

    def emitter_broadcast_axes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            result: dict(operands) for result, operands in self.broadcast_axes
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": FUNCTIONAL_BOUNDARY_PROTOCOL,
            "profile": self.profile,
            "domains": [list(item) for item in self.domains],
            "access_axes": [
                [access_id, list(axes)] for access_id, axes in self.access_axes
            ],
            "predicates": [
                [access_id, predicate.to_json() if predicate is not None else None]
                for access_id, predicate in self.predicates
            ],
            "broadcast_axes": [
                [
                    result,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for result, operands in self.broadcast_axes
            ],
            "proven_index_bounds": sorted(self.proven_index_bounds),
            "consumed": list(self.consumed),
            "boundary_digest": self.boundary_digest,
            "predicated_access_count": self.predicated_access_count,
            "erased_fill_count": self.erased_fill_count,
        }


def parse_direct_axes(expression: str) -> tuple[str, ...]:
    """Return the unique closed direct-axis interpretation of a wire form."""

    axes = parse_index_expression(
        expression,
        context="J boundary direct index",
    ).direct_axes
    if axes is None:
        raise NotImplementedError(
            "functional dense access requires a direct ordered axis vector"
        )
    if len(set(axes)) != len(axes):
        raise ValueError("functional dense access axes must be unique")
    return axes


def project_predicate_text(
    expression: str,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> PredicateProjection:
    """Parse one predicate through the same typed projection used by consumers."""

    return _predicate_projection(
        parse_contract_expr(
            expression,
            expected=ContractSort.BOOLEAN,
            context="J boundary access predicate",
        ),
        logical_guards=logical_guards,
    )

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


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(item for item in op.result if isinstance(item, str))
    return ()


def _node_table(j: Any) -> dict[str, IntentOp]:
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


def _predicate_projection(
    expression: ContractExpr,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> PredicateProjection:
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
                "functional boundary supports only logical-domain predicates"
            )
        previous = upper.get(upper_axis)
        if previous is not None and previous != extent:
            raise ValueError(
                f"functional predicate axis {upper_axis!r} has conflicting extents"
            )
        upper[upper_axis] = extent
    if not upper and not guards:
        raise NotImplementedError(
            "functional boundary predicate requires an upper bound or J guard"
        )
    return PredicateProjection(
        frozenset(guards),
        frozenset(lower),
        tuple(sorted(upper.items())),
    )


def _logical_scalar_inputs(j: Any) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind.value in {"scalar", "constexpr"}
        and parameter.logical_ref in inputs
        and not inputs[parameter.logical_ref].shape
    )


def _logical_boolean_guards(j: Any) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        name for name in _logical_scalar_inputs(j) if inputs[name].dtype == "i1"
    )


def _full_domains(j: Any) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    logical_guards = _logical_boolean_guards(j)
    for domain_id, expression in j_elaboration(j).domains:
        try:
            contract = _predicate_projection(
                expression, logical_guards=logical_guards
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "functional dense domain requires complete logical axis bounds; "
                f"unsupported domain {domain_id!r}"
            ) from exc
        if contract.guards and not contract.upper_bounds:
            continue
        if {axis for axis, _ in contract.upper_bounds} - set(contract.lower_axes):
            raise NotImplementedError(
                f"functional dense domain {domain_id!r} lacks a lower zero bound"
            )
        for axis, extent in contract.upper_bounds:
            previous = result.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"functional domain axis {axis!r} has conflicting extents"
                )
            result[axis] = extent
    return result


def _broadcast_axes(j: Any) -> dict[str, dict[str, tuple[int, ...]]]:
    nodes = _node_table(j)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in j.boundary.broadcasts:
        op = nodes.get(projection.node_ref)
        if op is None or not isinstance(op.result, str):
            raise NotImplementedError(
                "functional broadcast requires one result consumer"
            )
        if projection.operand not in op.operands:
            raise ValueError("functional broadcast names a non-consumed operand")
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in projection.axes
        ):
            raise NotImplementedError(
                "functional projection does not implement symbolic broadcast axes"
            )
        axes = tuple(int(axis) for axis in projection.axes)
        previous = result.setdefault(op.result, {}).get(projection.operand)
        if previous is not None and previous != axes:
            raise ValueError("conflicting functional broadcast relations")
        result[op.result][projection.operand] = axes
    return result


def _validate_masked_sum_reduction(
    j: Any,
    *,
    domains: Mapping[str, int | str],
    axes_by_access: Mapping[str, tuple[str, ...]],
    predicates_by_access: Mapping[str, PredicateProjection | None],
) -> None:
    boundary = j.boundary
    if (
        len(j.intent.inputs) != 1
        or len(j.intent.outputs) != 1
        or len(boundary.reductions) != 1
    ):
        raise NotImplementedError(
            "functional padded reduction requires one input, output, and contract"
        )
    if (
        boundary.aliases
        or boundary.broadcasts
        or boundary.index_bounds
        or boundary.updates
    ):
        raise NotImplementedError(
            "functional padded reduction has unsupported auxiliary relations"
        )
    loads = [item for item in boundary.accesses if item.kind.value == "load"]
    stores = [item for item in boundary.accesses if item.kind.value == "store"]
    if len(loads) != 1 or len(stores) != 1:
        raise NotImplementedError(
            "functional padded reduction requires one load and one store"
        )
    load, store = loads[0], stores[0]
    if load.tensor != j.intent.inputs[0].name or store.tensor != j.intent.outputs[0].name:
        raise ValueError("functional padded reduction access differs from J ABI")
    if (
        isinstance(load.fill_value, bool)
        or not isinstance(load.fill_value, (int, float))
        or float(load.fill_value) != 0.0
    ):
        raise NotImplementedError(
            "functional padded sum requires an explicit zero masked-load fill"
        )
    if store.predicate is not None or store.fill_value is not None:
        raise NotImplementedError(
            "functional padded reduction requires an unpredicated output store"
        )

    nodes = _node_table(j)
    contract = boundary.reductions[0]
    reduction = nodes.get(contract.node_ref)
    if reduction is None or reduction.op != "reduce":
        raise ValueError("functional reduction contract does not target reduce")
    if reduction.attrs.get("fn") != "sum":
        raise NotImplementedError(
            "functional padded reduction currently supports sum only"
        )
    reduction_axes = reduction.attrs.get("axes")
    if (
        not isinstance(reduction_axes, list)
        or len(reduction_axes) != 1
        or isinstance(reduction_axes[0], bool)
        or not isinstance(reduction_axes[0], int)
    ):
        raise NotImplementedError(
            "functional padded reduction requires one integer axis"
        )
    reduced_axis = reduction_axes[0]
    input_tensor = j.intent.inputs[0]
    output_tensor = j.intent.outputs[0]
    if input_tensor.dtype != output_tensor.dtype:
        raise NotImplementedError(
            "functional padded sum requires equal input/output dtype"
        )
    if reduced_axis < 0 or reduced_axis >= len(input_tensor.shape):
        raise ValueError("functional reduction axis is outside input rank")
    expected_output_shape = [
        extent
        for index, extent in enumerate(input_tensor.shape)
        if index != reduced_axis
    ]
    if output_tensor.shape != expected_output_shape:
        raise ValueError("functional reduction output is not axis removal")

    load_axes = axes_by_access[load.access_id]
    store_axes = axes_by_access[store.access_id]
    if set(domains) != set(load_axes):
        raise NotImplementedError(
            "functional reduction requires a full domain for every input axis"
        )
    expected_store_axes = tuple(
        axis for index, axis in enumerate(load_axes) if index != reduced_axis
    )
    if store_axes != expected_store_axes:
        raise ValueError("functional reduction store indices are not axis removal")
    predicate = predicates_by_access[load.access_id]
    if predicate is None:
        raise NotImplementedError("functional reduction requires a load predicate")
    if predicate.guards:
        raise NotImplementedError(
            "functional reduction does not consume logical guard predicates"
        )
    reduced_index = load_axes[reduced_axis]
    expected_bound = (reduced_index, input_tensor.shape[reduced_axis])
    if (
        predicate.upper_bounds != (expected_bound,)
        or not predicate.lower_axes.issubset({reduced_index})
    ):
        raise ValueError(
            "functional reduction predicate does not bind the reduced axis"
        )
    if predicates_by_access[store.access_id] is not None:
        raise NotImplementedError("functional reduction store must be unpredicated")
    if store.node_ref is None or nodes.get(store.node_ref) is not reduction:
        raise ValueError("functional reduction store does not target result")
    if not isinstance(reduction.result, str):
        raise NotImplementedError("functional reduction requires one result")
    returns = {item.output: item.value for item in j.intent.returns}
    if returns != {output_tensor.name: reduction.result}:
        raise ValueError("functional reduction result is not the J output")
    if (
        len(boundary.output_mappings) != 1
        or boundary.output_mappings[0].output != output_tensor.name
        or boundary.output_mappings[0].value != reduction.result
        or boundary.output_mappings[0].node_ref != contract.node_ref
    ):
        raise ValueError(
            "functional reduction requires an explicit reduction output mapping"
        )

    body = j.intent.body
    if not body or body[-1] is not reduction or any(op.body for op in body):
        raise NotImplementedError(
            "functional reduction requires one flat graph ending in reduction"
        )
    if sum(op.op == "reduce" for op in body) != 1:
        raise NotImplementedError("functional reduction requires exactly one reduce")
    if len(reduction.operands) != 1:
        raise NotImplementedError("functional reduction requires one operand")
    producers: dict[str, IntentOp] = {}
    for op in body:
        results = _result_names(op)
        if len(results) != 1:
            raise NotImplementedError(
                "functional reduction requires single-result value nodes"
            )
        producers[results[0]] = op
    required_prefix: set[int] = set()
    pending = list(reduction.operands)
    while pending:
        value = pending.pop()
        producer = producers.get(value)
        if producer is None:
            if value != input_tensor.name:
                raise ValueError("functional reduction has a non-input leaf")
            continue
        if producer is reduction:
            raise ValueError("functional reduction dataflow is cyclic")
        marker = id(producer)
        if marker in required_prefix:
            continue
        required_prefix.add(marker)
        pending.extend(producer.operands)
    if required_prefix != {id(op) for op in body[:-1]}:
        raise NotImplementedError(
            "functional reduction requires every prefix node to feed reduction"
        )
    load_owner = nodes.get(load.node_ref) if load.node_ref is not None else None
    if load_owner is not reduction and id(load_owner) not in required_prefix:
        raise ValueError("functional reduction load does not feed reduction")
    for op in body[:-1]:
        if (
            op.type is None
            or op.type.get("dtype") != input_tensor.dtype
            or op.type.get("shape") != input_tensor.shape
        ):
            raise NotImplementedError(
                "functional reduction prefix must preserve input type"
            )
    padded_values: dict[str, float] = {input_tensor.name: float(load.fill_value)}
    for op in body[:-1]:
        operands = [padded_values.get(value) for value in op.operands]
        if any(value is None for value in operands):
            raise NotImplementedError(
                "functional reduction cannot prove padded contribution"
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
                "functional reduction supports zero-stable identity/mul prefix"
            )
        padded_values[_result_names(op)[0]] = padded_result
    if padded_values.get(reduction.operands[0]) != 0.0:
        raise NotImplementedError(
            "functional reduction prefix does not preserve sum identity"
        )
    if (
        reduction.type is None
        or reduction.type.get("dtype") != output_tensor.dtype
        or reduction.type.get("shape") != output_tensor.shape
    ):
        raise ValueError("functional reduction result type differs from output")
    if contract.tile_axis != 0:
        raise NotImplementedError(
            "functional reduction requires a single physical tile axis 0"
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
                f"functional padded sum requires explicit neutral {name}=0"
            )


def project_functional_boundary(j: Any) -> FunctionalBoundaryProjection:
    """Prove that complete Layer B can use J's logical tensor execution."""

    j = elaborate_j(j)
    boundary = j.boundary
    if any(
        parameter.mode.value == "inout" or parameter.alias_of is not None
        for parameter in j.signature.parameters
    ):
        raise NotImplementedError(
            "functional boundary does not implement inout/alias ABI"
        )
    if boundary.aliases:
        raise NotImplementedError("functional boundary does not implement aliases")
    if any(item.mutability.value == "mutable" for item in j.storage):
        raise NotImplementedError(
            "functional boundary does not implement general mutable storage"
        )

    domains = _full_domains(j) if boundary.accesses else {}
    if not boundary.accesses and (boundary.domains or boundary.effects):
        raise ValueError(
            "functional boundary domains/effects require explicit accesses"
        )
    has_predicate_or_fill = any(
        access.predicate is not None or access.fill_value is not None
        for access in boundary.accesses
    )
    masked_reduction = has_predicate_or_fill and bool(boundary.reductions)
    if has_predicate_or_fill and boundary.updates:
        raise NotImplementedError(
            "functional mask projection does not implement update padding"
        )

    nodes = _node_table(j)
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    input_names = {item.name for item in j.intent.inputs}
    output_names = {item.name for item in j.intent.outputs}
    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item for item in boundary.output_mappings}
    for output, mapping in mappings.items():
        if output in returns and mapping.value != returns[output]:
            raise ValueError("functional output mapping differs from J return")

    effects_by_access: dict[str, list[Any]] = {}
    for effect in boundary.effects:
        if effect.access_ref is None:
            raise NotImplementedError(
                "functional boundary requires access-bound effects"
            )
        effects_by_access.setdefault(effect.access_ref, []).append(effect)
    access_ids = [item.access_id for item in boundary.accesses]
    if len(access_ids) != len(set(access_ids)):
        raise ValueError("functional boundary has duplicate access ids")
    effect_ids = [item.effect_id for item in boundary.effects]
    if len(effect_ids) != len(set(effect_ids)):
        raise ValueError("functional boundary has duplicate effect ids")

    consumed_domain_axes: set[str] = set()
    inferred_extents: dict[str, int | str] = dict(domains)
    axes_by_access: dict[str, tuple[str, ...]] = {}
    predicates: dict[str, PredicateProjection | None] = {}
    loaded: set[str] = set()
    store_counts = {name: 0 for name in output_names}
    logical_scalar_inputs = _logical_scalar_inputs(j)
    logical_guards = _logical_boolean_guards(j)
    parsed_accesses = j_elaboration(j).access_by_id
    expected_effect = {"load": "read", "store": "write"}
    for access in boundary.accesses:
        if access.kind.value == "update":
            raise NotImplementedError(
                "functional boundary does not implement update access"
            )
        op = nodes.get(access.node_ref) if access.node_ref is not None else None
        if access.node_ref is not None and op is None:
            raise ValueError(
                f"functional access {access.access_id!r} has unknown J node"
            )
        tensor = tensors.get(access.tensor)
        if tensor is None:
            raise ValueError(
                f"functional access {access.access_id!r} has unknown tensor"
            )
        axes = parsed_accesses[access.access_id].index.direct_axes
        if axes is None:
            raise NotImplementedError(
                "functional dense access requires a direct ordered axis vector"
            )
        if len(set(axes)) != len(axes):
            raise ValueError("functional dense access axes must be unique")
        if len(axes) != len(tensor.shape):
            raise ValueError(
                f"functional access {access.access_id!r} rank differs from tensor"
            )
        axes_by_access[access.access_id] = axes
        for axis, declared_extent in zip(axes, tensor.shape, strict=True):
            if axis == "0" and declared_extent != 1:
                raise ValueError(
                    "functional zero index is valid only for singleton axis"
                )
            actual_extent = inferred_extents.get(axis, declared_extent)
            if actual_extent != declared_extent:
                raise ValueError(
                    f"functional axis {axis!r} extent {actual_extent!r} differs "
                    f"from J tensor extent {declared_extent!r}"
                )
            inferred_extents.setdefault(axis, declared_extent)
            if axis in domains:
                consumed_domain_axes.add(axis)

        parsed = parsed_accesses[access.access_id].predicate
        predicate = (
            _predicate_projection(parsed, logical_guards=logical_guards)
            if parsed is not None else None
        )
        predicates[access.access_id] = predicate
        if predicate is not None:
            graph_operands = {
                operand for node in j.intent.body for operand in node.operands
            }
            if not predicate.guards.issubset(graph_operands):
                raise ValueError("functional access guard is absent from J dataflow")
            if not predicate.lower_axes.issubset(set(axes)):
                raise ValueError("functional predicate lower bound is not an axis")
            for axis, extent in predicate.upper_bounds:
                if axis not in axes:
                    raise ValueError(
                        "functional predicate upper bound is not an access axis"
                    )
                if domains.get(axis) != extent:
                    raise ValueError(
                        "functional predicate differs from the full logical domain"
                    )
            if (
                predicate.upper_bounds
                and access.kind.value == "load"
                and access.fill_value is None
            ):
                raise NotImplementedError(
                    "functional mask projection requires an explicit masked-load fill"
                )
        elif access.fill_value is not None:
            raise ValueError("functional unpredicated access must not declare fill")
        if access.kind.value != "load" and access.fill_value is not None:
            raise ValueError("functional store/update must not declare fill")

        matching_effects = effects_by_access.get(access.access_id, [])
        if len(matching_effects) != 1:
            raise ValueError(
                f"functional access {access.access_id!r} requires one effect"
            )
        effect = matching_effects[0]
        if (
            effect.kind.value != expected_effect[access.kind.value]
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            raise ValueError(
                f"functional access {access.access_id!r} disagrees with its effect"
            )

        if access.kind.value == "load":
            if access.tensor not in input_names:
                raise ValueError("functional load is not a J input")
            if op is not None and access.tensor not in op.operands:
                raise ValueError("functional load is not a direct J operand")
            if op is None and (
                j.intent.body or access.tensor not in set(returns.values())
            ):
                raise NotImplementedError(
                    "node-less load is supported only for direct identity J"
                )
            loaded.add(access.tensor)
            continue

        if access.tensor not in output_names:
            raise ValueError("functional store does not target a J output")
        observable_value = returns.get(access.tensor, access.tensor)
        mapping = mappings.get(access.tensor)
        if mapping is not None:
            observable_value = mapping.value
            if mapping.node_ref is not None and nodes.get(mapping.node_ref) is not op:
                raise ValueError("functional store disagrees with output mapping")
        if op is not None:
            if observable_value not in _result_names(op):
                raise ValueError("functional store is not produced by its J node")
            if has_predicate_or_fill and not masked_reduction and (
                op.type is None
                or op.type.get("dtype") != tensor.dtype
                or op.type.get("shape") != tensor.shape
            ):
                raise ValueError("functional store producer type differs from output")
        elif (
            j.intent.body or mapping is None or observable_value not in input_names
        ):
            raise NotImplementedError(
                "node-less store is supported only for direct identity J"
            )
        store_counts[access.tensor] += 1

    if set(effects_by_access) != set(access_ids):
        raise ValueError("functional boundary has effect for unknown access")
    if boundary.accesses and loaded != input_names - set(logical_scalar_inputs):
        raise NotImplementedError(
            "functional boundary must cover every logical tensor input"
        )
    if boundary.accesses and any(count != 1 for count in store_counts.values()):
        raise NotImplementedError(
            "functional boundary requires exactly one store per logical output"
        )
    if consumed_domain_axes != set(domains):
        raise NotImplementedError(
            "functional boundary has an unconsumed domain axis"
        )
    concrete_predicates = [item for item in predicates.values() if item is not None]
    if masked_reduction:
        _validate_masked_sum_reduction(
            j,
            domains=domains,
            axes_by_access=axes_by_access,
            predicates_by_access=predicates,
        )
    elif concrete_predicates:
        masked_axes = {
            axis
            for predicate in concrete_predicates
            for axis, _extent in predicate.upper_bounds
        }
        for access in boundary.accesses:
            axes = axes_by_access[access.access_id]
            predicate = predicates[access.access_id]
            upper = dict(predicate.upper_bounds) if predicate is not None else {}
            for axis in masked_axes.intersection(axes):
                if upper.get(axis) != domains.get(axis):
                    raise NotImplementedError(
                        "functional mask projection requires the same predicate "
                        "on every access of a masked axis"
                    )

    for node_ref, contract in {
        item.node_ref: item for item in boundary.reductions
    }.items():
        op = nodes.get(node_ref)
        if op is None or op.op != "reduce":
            raise ValueError("Layer-B reduction does not target reduce")
        fn = op.attrs.get("fn")
        neutral = {
            "sum": 0,
            "mean": 0,
            "prod": 1,
            "any": False,
            "all": True,
        }.get(fn)
        for declared in (
            value
            for value in (contract.identity, contract.masked_contribution)
            if value is not None
        ):
            if fn in {"max", "min"}:
                expected = "NEG_INF" if fn == "max" else "POS_INF"
                if declared != expected:
                    raise NotImplementedError(
                        f"functional projection requires {expected} padding"
                    )
            elif neutral is not None and declared != neutral:
                raise NotImplementedError(
                    f"functional projection cannot erase non-neutral {fn!r} padding"
                )

    mapping_values = {
        item.output: item.value for item in boundary.output_mappings
    }
    if mapping_values and mapping_values != returns:
        raise NotImplementedError(
            "functional output destination relation is incomplete"
        )
    for item in boundary.index_bounds:
        if nodes.get(item.node_ref) is None:
            raise ValueError("functional index bounds reference unknown node")
    for item in boundary.updates:
        if nodes.get(item.node_ref) is None:
            raise ValueError("functional update references unknown node")
    outputs = {item.name for item in j.intent.outputs}
    if any(item.output not in outputs for item in boundary.result_relations):
        raise ValueError("functional result relation references unknown output")

    broadcast_axes = _broadcast_axes(j)
    proven_bounds = frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in boundary.index_bounds
    )
    profile = (
        "logical_domain_reduction_padding_v1"
        if concrete_predicates and boundary.reductions
        else (
            "logical_domain_mask_erasure_v1"
            if concrete_predicates
            else ("dense_functional_v1" if boundary.accesses else "pure_value_v1")
        )
    )
    consumed = {
        *(f"domain:{name}" for name in boundary.domains),
        *(f"access:{item.access_id}" for item in boundary.accesses),
        *(f"effect:{item.effect_id}" for item in boundary.effects),
        *(f"output:{item.output}" for item in boundary.output_mappings),
        *(f"reduction:{item.node_ref}" for item in boundary.reductions),
        *(f"broadcast:{item.node_ref}:{item.operand}" for item in boundary.broadcasts),
        *(f"index_bounds:{item.node_ref}" for item in boundary.index_bounds),
        *(f"update:{item.node_ref}" for item in boundary.updates),
        *(f"result_relation:{item.output}" for item in boundary.result_relations),
    }
    return FunctionalBoundaryProjection(
        profile,
        tuple(sorted(domains.items())),
        tuple(sorted(axes_by_access.items())),
        tuple(sorted(predicates.items())),
        tuple(
            sorted(
                (
                    result,
                    tuple(sorted(operands.items())),
                )
                for result, operands in broadcast_axes.items()
            )
        ),
        proven_bounds,
        tuple(sorted(consumed)),
        _digest(boundary.to_json()),
        sum(item.predicate is not None for item in boundary.accesses),
        sum(item.fill_value is not None for item in boundary.accesses),
    )


__all__ = [
    "FUNCTIONAL_BOUNDARY_PROTOCOL",
    "FunctionalBoundaryProjection",
    "PredicateProjection",
    "parse_direct_axes",
    "project_predicate_text",
    "project_functional_boundary",
]
