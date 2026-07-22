"""Simple backend-neutral ``E_J`` that consumes the complete canonical J."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.backend.cpu.abi import CPUABIValidationError, CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import (
    AccessKind,
    EffectKind,
    ParameterMode,
    SemanticArchive,
)

from .model import ExecutionResult


SCALAR_LOWERING_PROTOCOL = "canonical_j_scalar_cpu_mlir_v5"


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_DIRECT_INDEX = re.compile(
    rf"^\[\s*(?P<axes>(?:{_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*)?)\s*\]$"
)
_BARE_DIRECT_INDEX = re.compile(rf"^(?P<axis>{_IDENTIFIER})$")
_FULL_DOMAIN_TERM = re.compile(
    rf"\s*0\s*<=\s*(?P<axis>{_IDENTIFIER})\s+and\s+"
    rf"(?P=axis)\s*<\s*(?P<extent>[0-9]+|{_IDENTIFIER})"
)
_DOMAIN_SEPARATOR = re.compile(r"\s+and\s+")
_PREDICATE_LOWER = re.compile(rf"^0\s*<=\s*(?P<axis>{_IDENTIFIER})$")
_PREDICATE_UPPER = re.compile(
    rf"^(?P<axis>{_IDENTIFIER})\s*<\s*(?P<extent>[0-9]+|{_IDENTIFIER})$"
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
    """Parse the conservative dense tensor projection supported by E_J.

    General affine, gathered, predicated, or source-realization expressions are
    deliberately not interpreted here.  They need a typed J relation and a
    corresponding evaluator instead of an ad-hoc expression evaluator.
    """

    match = _DIRECT_INDEX.fullmatch(expression)
    if match is None:
        bare = _BARE_DIRECT_INDEX.fullmatch(expression.strip())
        if bare is not None:
            return (bare.group("axis"),)
        raise NotImplementedError(
            "E_J dense access requires one bare axis or '[axis0, ...]'"
        )
    axes = match.group("axes")
    if not axes:
        return ()
    result = tuple(item.strip() for item in axes.split(","))
    if len(set(result)) != len(result):
        raise ValueError("E_J dense access axes must be unique")
    return result


def _parse_domain_predicate(
    expression: str,
) -> tuple[frozenset[str], tuple[tuple[str, int | str], ...]]:
    """Parse the closed logical-domain predicate subset used for mask erasure."""

    lower: set[str] = set()
    upper: dict[str, int | str] = {}
    terms = re.split(r"\s+and\s+", expression.strip())
    if not terms or any(not term for term in terms):
        raise ValueError("E_J access predicate must not be empty")
    for term in terms:
        lower_match = _PREDICATE_LOWER.fullmatch(term.strip())
        if lower_match is not None:
            lower.add(lower_match.group("axis"))
            continue
        upper_match = _PREDICATE_UPPER.fullmatch(term.strip())
        if upper_match is None:
            raise NotImplementedError(
                "E_J mask erasure supports only logical-domain bound predicates"
            )
        axis = upper_match.group("axis")
        extent_text = upper_match.group("extent")
        extent: int | str = (
            int(extent_text) if extent_text.isdecimal() else extent_text
        )
        previous = upper.get(axis)
        if previous is not None and previous != extent:
            raise ValueError(
                f"E_J predicate axis {axis!r} has conflicting extents"
            )
        upper[axis] = extent
    if not upper:
        raise NotImplementedError(
            "E_J mask erasure requires an upper logical-domain bound"
        )
    return frozenset(lower), tuple(sorted(upper.items()))


def _parse_full_domains(j: SemanticArchive) -> dict[str, int | str]:
    """Parse full logical tensor domains, rejecting every wider grammar.

    The accepted form is a conjunction of ``0 <= axis and axis < extent``
    terms.  This is enough to prove that a direct access denotes the complete
    logical tensor represented by IntentFunction.  Predicated/partial domains
    remain unsupported until their denotation is implemented explicitly.
    """

    result: dict[str, int | str] = {}
    for domain_id, expression in j.boundary.domains.items():
        expression = expression.strip()
        position = 0
        matched = False
        while position < len(expression):
            term = _FULL_DOMAIN_TERM.match(expression, position)
            if term is None:
                raise NotImplementedError(
                    "E_J dense domain requires '0 <= axis and axis < extent' "
                    f"terms; unsupported domain {domain_id!r}"
                )
            matched = True
            axis = term.group("axis")
            extent_text = term.group("extent")
            extent: int | str = (
                int(extent_text) if extent_text.isdecimal() else extent_text
            )
            previous = result.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"E_J domain axis {axis!r} has conflicting extents"
                )
            result[axis] = extent
            position = term.end()
            if position == len(expression):
                break
            separator = _DOMAIN_SEPARATOR.match(expression, position)
            if separator is None:
                raise NotImplementedError(
                    f"E_J dense domain {domain_id!r} has unsupported syntax"
                )
            position = separator.end()
        if not matched:
            raise ValueError(f"E_J domain {domain_id!r} must not be empty")
    return result


def _validate_masked_sum_reduction(
    j: SemanticArchive,
    *,
    domains: Mapping[str, int | str],
    axes_by_access: Mapping[str, tuple[str, ...]],
    predicates_by_access: Mapping[
        str,
        tuple[frozenset[str], tuple[tuple[str, int | str], ...]] | None,
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
    lower_axes, upper_bounds = predicate
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

    domains = _parse_full_domains(j)
    has_predicate_or_fill = any(
        access.predicate is not None or access.fill_value is not None
        for access in boundary.accesses
    )
    masked_reduction = has_predicate_or_fill and bool(boundary.reductions)
    if has_predicate_or_fill and boundary.updates:
        raise NotImplementedError(
            "E_J mask erasure does not implement update padding"
        )
    if has_predicate_or_fill and not masked_reduction and j.intent.body:
        raise NotImplementedError(
            "E_J mask erasure currently requires a direct identity J"
        )
    nodes = _node_table(j)
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    input_names = {item.name for item in j.intent.inputs}
    output_names = {item.name for item in j.intent.outputs}
    returns = {item.output: item.value for item in j.intent.returns}
    if has_predicate_or_fill and not masked_reduction and any(
        value not in input_names for value in returns.values()
    ):
        raise NotImplementedError(
            "E_J mask erasure currently requires outputs to return J inputs"
        )
    if has_predicate_or_fill and not masked_reduction:
        if (
            len(input_names) != len(output_names)
            or set(returns) != output_names
            or set(returns.values()) != input_names
            or len(set(returns.values())) != len(returns)
        ):
            raise NotImplementedError(
                "E_J mask erasure requires a one-to-one input/output identity"
            )
        if any(len(tensor.shape) != 1 for tensor in tensors.values()):
            raise NotImplementedError(
                "E_J mask erasure currently supports rank-one tensors only"
            )
        for output_name, input_name in returns.items():
            output = tensors[output_name]
            input_tensor = tensors[input_name]
            if (
                output.dtype != input_tensor.dtype
                or output.shape != input_tensor.shape
            ):
                raise ValueError(
                    "E_J mask erasure requires identical input/output dtype and shape"
                )
    mappings = {item.output: item for item in boundary.output_mappings}
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
        tuple[frozenset[str], tuple[tuple[str, int | str], ...]] | None
    ] = {}
    axes_by_access: dict[str, tuple[str, ...]] = {}
    loaded: set[str] = set()
    store_counts = {name: 0 for name in output_names}
    expected_effect = {
        AccessKind.LOAD: EffectKind.READ,
        AccessKind.STORE: EffectKind.WRITE,
    }
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
        assert access.index_expression is not None
        axes = _parse_direct_index(access.index_expression)
        axes_by_access[access.access_id] = axes
        if len(axes) != len(tensor.shape):
            raise ValueError(
                f"E_J access {access.access_id!r} rank differs from J tensor"
            )
        for axis, declared_extent in zip(axes, tensor.shape, strict=True):
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
            _parse_domain_predicate(access.predicate)
            if access.predicate is not None
            else None
        )
        predicate_contracts[access.access_id] = predicate_contract
        if predicate_contract is not None:
            if not domains:
                raise NotImplementedError(
                    "E_J mask erasure requires an explicit full logical domain"
                )
            lower_axes, upper_bounds = predicate_contract
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
            if access.kind is AccessKind.LOAD and access.fill_value is None:
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
            if op is None and (
                j.intent.body or access.tensor not in set(returns.values())
            ):
                raise NotImplementedError(
                    "E_J node-less load is supported only for direct identity J"
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
    if loaded != input_names:
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
    concrete_predicates = [
        item for item in predicate_contracts.values() if item is not None
    ]
    if masked_reduction:
        _validate_masked_sum_reduction(
            j,
            domains=domains,
            axes_by_access=axes_by_access,
            predicates_by_access=predicate_contracts,
        )
    elif concrete_predicates:
        if len(concrete_predicates) != len(predicate_contracts):
            raise NotImplementedError(
                "E_J mask erasure requires the same predicate on every access"
            )
        if len(set(concrete_predicates)) != 1:
            raise ValueError(
                "E_J mask erasure requires identical logical-domain predicates"
            )


def _validate_complete_boundary(j: SemanticArchive) -> str:
    """Consume supported B and reject every relation E_J cannot implement."""

    boundary = j.boundary
    _validate_dense_functional_boundary(j)
    if boundary.aliases:
        raise NotImplementedError("E_J does not implement alias relations")
    if any(item.mutability.value == "mutable" for item in j.storage):
        raise NotImplementedError("E_J does not implement general mutable storage")

    nodes = _node_table(j)
    reductions_by_ref = {item.node_ref: item for item in boundary.reductions}
    for node_ref, contract in reductions_by_ref.items():
        op = nodes.get(node_ref)
        if op is None or op.op != "reduce":
            raise ValueError("Layer-B reduction does not target a reduce node")
        fn = op.attrs.get("fn")
        neutral = {
            "sum": 0,
            "mean": 0,
            "prod": 1,
            "any": False,
            "all": True,
        }.get(fn)
        declared_values = [
            value
            for value in (contract.identity, contract.masked_contribution)
            if value is not None
        ]
        for declared in declared_values:
            if fn in {"max", "min"}:
                expected = "NEG_INF" if fn == "max" else "POS_INF"
                if declared != expected:
                    raise NotImplementedError(
                        f"E_J requires {expected} padding for {fn}"
                    )
            elif neutral is not None and declared != neutral:
                raise NotImplementedError(
                    f"E_J cannot eliminate non-neutral {fn!r} boundary padding"
                )

    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item.value for item in boundary.output_mappings}
    if mappings and mappings != returns:
        raise NotImplementedError("E_J output destination relation is incomplete")

    # Typed bounds/update/result relations were already cross-validated against
    # this exact J by SemanticArchive.  Bounds are checked against concrete
    # inputs below; result relations are consumed by Gate2.  A typed update is
    # recognized here, while the CPU capability preflight remains responsible
    # for rejecting scatter until a real Intent->CPU lowering exists.
    nodes = _node_table(j)
    for item in boundary.index_bounds:
        if nodes.get(item.node_ref) is None:
            raise ValueError("E_J index bounds reference an unknown node")
    for item in boundary.updates:
        if nodes.get(item.node_ref) is None:
            raise ValueError("E_J logical update references an unknown node")
    outputs = {item.name for item in j.intent.outputs}
    if any(item.output not in outputs for item in boundary.result_relations):
        raise ValueError("E_J result relation references an unknown output")
    return _digest(boundary.to_json())


def scalar_index_bounds(j: SemanticArchive) -> frozenset[str]:
    """Return the exact emitter projection authorized by canonical J."""

    return frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in j.boundary.index_bounds
    )


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
    nodes = _node_table(j)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in j.boundary.broadcasts:
        op = nodes.get(projection.node_ref)
        if op is None or not isinstance(op.result, str):
            raise NotImplementedError("scalar broadcast requires one result consumer")
        if projection.operand not in op.operands:
            raise ValueError("scalar broadcast names a non-consumed operand")
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in projection.axes):
            raise NotImplementedError("E_J does not implement symbolic broadcast axes")
        axes = tuple(int(axis) for axis in projection.axes)
        previous = result.setdefault(op.result, {}).get(projection.operand)
        if previous is not None and previous != axes:
            raise ValueError("conflicting scalar broadcast relations")
        result[op.result][projection.operand] = axes
    return result


@dataclass(frozen=True)
class ScalarLoweringSpec:
    """Invocation request holding the same J object used by Gate1."""

    j: SemanticArchive
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("scalar lowering requires canonical J")
        dims = dict(self.dims)
        inputs = dict(self.input_bindings)
        outputs = dict(self.output_bindings)
        object.__setattr__(self, "dims", _canonical_int_mapping(dims))
        object.__setattr__(self, "input_bindings", _canonical_str_mapping(inputs))
        object.__setattr__(self, "output_bindings", _canonical_str_mapping(outputs))
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
            inputs = [
                np.asarray(values[spec.input_binding_map[tensor.name]])
                for tensor in spec.j.intent.inputs
            ]
        except KeyError as exc:
            return ExecutionResult("j", False, error=f"missing E_J input {exc.args[0]!r}")
        try:
            boundary_digest = _validate_complete_boundary(spec.j)
            inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(spec.j.intent.inputs, inputs, strict=True)
            }
            validate_index_bounds_inputs(spec.j, inputs_by_ref)
            broadcast_axes = scalar_broadcast_axes(spec.j)
            execution = execute_on_cpu_outputs(
                spec.j.intent,
                spec.dims_map,
                inputs,
                broadcast_axes=broadcast_axes,
                proven_index_bounds=scalar_index_bounds(spec.j),
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
                    "path": "canonical J -> intent MLIR -> scalar CPU",
                    "function": spec.j.intent.name,
                    "j_digest": spec.j_digest,
                    "boundary_digest": boundary_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "boundary_profile": (
                        "logical_domain_reduction_padding_v1"
                        if any(
                            item.predicate is not None
                            for item in spec.j.boundary.accesses
                        )
                        and spec.j.boundary.reductions
                        else (
                            "logical_domain_mask_erasure_v1"
                            if any(
                                item.predicate is not None
                                for item in spec.j.boundary.accesses
                            )
                            else (
                                "dense_functional_v1"
                                if spec.j.boundary.accesses
                                else "pure_value_v1"
                            )
                        )
                    ),
                    "domain_relation_count": len(spec.j.boundary.domains),
                    "access_relation_count": len(spec.j.boundary.accesses),
                    "effect_relation_count": len(spec.j.boundary.effects),
                    "predicated_access_count": sum(
                        item.predicate is not None
                        for item in spec.j.boundary.accesses
                    ),
                    "erased_fill_count": sum(
                        item.fill_value is not None
                        for item in spec.j.boundary.accesses
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
