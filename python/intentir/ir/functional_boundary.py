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


FUNCTIONAL_BOUNDARY_PROTOCOL = "intentir.j-functional-boundary.v2"


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


@dataclass(frozen=True)
class FunctionalBoundaryClosureIssue:
    """One J-only inconsistency in the projectable logical-domain profile.

    This is deliberately narrower than ``FunctionalBoundaryProjection``.  It
    records only contradictions that can be decided from J's own typed tensor
    shapes, direct access vectors, predicates, and explicit domains.  An empty
    issue list does not claim that every valid J feature is implemented by the
    current RefEval/backend projection.
    """

    path: str
    detail: str
    ref: str | None = None


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


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    result: list[IntentOp] = []
    for op in body:
        result.append(op)
        result.extend(_walk_ops(op.body))
    return tuple(result)


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


def functional_boundary_closure_issues(
    j: Any,
) -> tuple[FunctionalBoundaryClosureIssue, ...]:
    """Check the shared, source-free logical-domain closure subset.

    The checker intentionally recognizes only the same zero-based rectangular
    logical-domain profile that the functional projection can erase into
    complete logical tensors.  Unsupported affine/logical predicates remain a
    consumer capability question; they are not made semantically illegal by
    TypeCheck.  Once an access predicate is in the recognized profile, however,
    its direct index axis and extent must agree with ``boundary.domains`` and
    the typed tensor shape.  Neither Source, C, pi, nor runtime values are
    consulted to fill a missing relation.
    """

    j = elaborate_j(j)
    boundary = j.boundary
    if not boundary.accesses:
        return ()

    # A non-rectangular or otherwise unsupported domain may still be a valid J
    # contract.  RefEval can report that capability boundary later; v6 must not
    # turn it into semantic illegality merely because the current projection
    # cannot consume it.
    try:
        domains = _full_domains(j)
    except NotImplementedError:
        return ()
    except ValueError as exc:
        return (
            FunctionalBoundaryClosureIssue(
                "j.boundary.domains",
                str(exc),
            ),
        )

    dimensions = {
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str)
    }
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    logical_guards = _logical_boolean_guards(j)
    parsed_accesses = j_elaboration(j).access_by_id
    issues: list[FunctionalBoundaryClosureIssue] = []

    for index, access in enumerate(boundary.accesses):
        parsed_access = parsed_accesses.get(access.access_id)
        tensor = tensors.get(access.tensor)
        if parsed_access is None or tensor is None:
            # Canonical elaboration/ordinary reference checking owns these
            # errors.  Do not duplicate them under the closure rule.
            continue
        axes = parsed_access.index.direct_axes
        if axes is None or len(set(axes)) != len(axes):
            # General affine/diagonal accesses are outside this projection
            # profile, not automatically invalid J.
            continue

        index_path = f"j.boundary.accesses[{index}].index_expression"
        if len(axes) != len(tensor.shape):
            issues.append(
                FunctionalBoundaryClosureIssue(
                    index_path,
                    "functional direct access rank differs from its J tensor",
                    access.access_id,
                )
            )
            continue
        for axis, declared_extent in zip(axes, tensor.shape, strict=True):
            domain_extent = domains.get(axis)
            if domain_extent is not None and domain_extent != declared_extent:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        index_path,
                        f"functional axis {axis!r} extent {domain_extent!r} "
                        f"differs from J tensor extent {declared_extent!r}",
                        access.access_id,
                    )
                )

        parsed_predicate = parsed_access.predicate
        if parsed_predicate is None:
            continue
        try:
            predicate = _predicate_projection(
                parsed_predicate,
                logical_guards=logical_guards,
            )
        except NotImplementedError:
            # Logical predicates outside the rectangular mask-erasure profile
            # remain valid J and must not be blanket-rejected by TypeCheck.
            continue
        except ValueError as exc:
            issues.append(
                FunctionalBoundaryClosureIssue(
                    f"j.boundary.accesses[{index}].predicate",
                    str(exc),
                    access.access_id,
                )
            )
            continue

        # ``i < j`` is a logical cross-axis predicate, not a rectangular
        # physical padding bound.  Only integer constants and declared J shape
        # dimensions identify the shared functional-domain profile.
        if any(
            not isinstance(extent, int) and extent not in dimensions
            for _axis, extent in predicate.upper_bounds
        ):
            continue
        predicate_path = f"j.boundary.accesses[{index}].predicate"
        for axis, extent in predicate.upper_bounds:
            if axis not in axes:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        predicate_path,
                        "functional predicate upper bound is not an access axis",
                        access.access_id,
                    )
                )
                continue
            if domains.get(axis) != extent:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        predicate_path,
                        "functional predicate differs from the full logical "
                        "domain declared by boundary.domains",
                        access.access_id,
                    )
                )

    return tuple(issues)


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


def _numpy_broadcast_positions(
    source: Sequence[int | str],
    target: Sequence[int | str],
) -> tuple[int, ...] | None:
    if len(source) > len(target):
        return None
    offset = len(target) - len(source)
    if not all(
        source_extent == 1 or source_extent == target[offset + index]
        for index, source_extent in enumerate(source)
    ):
        return None
    return tuple(range(offset, len(target)))


def _shape_can_numpy_broadcast(
    source: Sequence[int | str],
    target: Sequence[int | str],
) -> bool:
    return _numpy_broadcast_positions(source, target) is not None


def _axis_broadcast_positions(
    source: Sequence[int | str],
    target: Sequence[int | str],
    axes: Sequence[int],
) -> tuple[int, ...] | None:
    """Mirror the emitter's closed rank-insertion broadcast proof."""

    target_rank = len(target)
    normalized: list[int] = []
    for raw_axis in axes:
        axis = raw_axis if raw_axis >= 0 else raw_axis + target_rank
        if axis < 0 or axis >= target_rank:
            return None
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        return None
    if not source:
        return ()
    axis_set = set(normalized)
    memo: dict[tuple[int, int], tuple[int, ...] | None] = {}

    def align(source_index: int, target_index: int) -> tuple[int, ...] | None:
        key = (source_index, target_index)
        if key in memo:
            return memo[key]
        if target_index == target_rank:
            result = () if source_index == len(source) else None
            memo[key] = result
            return result
        if target_index in axis_set:
            if (
                source_index < len(source)
                and source[source_index] == 1
            ):
                suffix = align(source_index + 1, target_index + 1)
                if suffix is not None:
                    result = (target_index, *suffix)
                    memo[key] = result
                    return result
            suffix = align(source_index, target_index + 1)
            if suffix is not None:
                memo[key] = suffix
                return suffix
        elif (
            source_index < len(source)
            and source[source_index] == target[target_index]
        ):
            suffix = align(source_index + 1, target_index + 1)
            if suffix is not None:
                result = (target_index, *suffix)
                memo[key] = result
                return result
        memo[key] = None
        return None

    return align(0, 0)


def _shape_can_broadcast_on_axes(
    source: Sequence[int | str],
    target: Sequence[int | str],
    axes: Sequence[int],
) -> bool:
    return _axis_broadcast_positions(source, target, axes) is not None


def _validate_broadcast_closure(
    j: Any,
    broadcasts: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> None:
    """Prove every pointwise operand reaches its declared result shape."""

    value_types = j_elaboration(j).value_type_by_name
    operations = _walk_ops(j.intent.body)
    consumers = {
        op.result: op
        for op in operations
        if isinstance(op.result, str)
    }
    for result, operands in broadcasts.items():
        consumer = consumers.get(result)
        if consumer is None or consumer.op not in {"map_binary", "select"}:
            raise NotImplementedError(
                "functional broadcast relation requires a pointwise consumer"
            )
        if any(operand not in consumer.operands for operand in operands):
            raise ValueError(
                "functional broadcast relation names a non-consumed operand"
            )

    for op in operations:
        if op.op not in {"map_binary", "select"} or not isinstance(op.result, str):
            continue
        result_type = value_types.get(op.result)
        if result_type is None:
            raise ValueError("functional pointwise result has no closed type")
        target_shape = result_type.shape
        declared = broadcasts.get(op.result, {})
        for operand in op.operands:
            operand_type = value_types.get(operand)
            if operand_type is None:
                raise ValueError("functional pointwise operand has no closed type")
            axes = declared.get(operand)
            if axes is None:
                closed = _shape_can_numpy_broadcast(
                    operand_type.shape,
                    target_shape,
                )
            else:
                closed = _shape_can_broadcast_on_axes(
                    operand_type.shape,
                    target_shape,
                    axes,
                )
            if not closed:
                raise NotImplementedError(
                    "functional broadcast relation is incomplete for a "
                    "pointwise operand"
                )


def _reduction_padding_analysis(
    j: Any,
) -> tuple[
    tuple[FunctionalBoundaryClosureIssue, ...],
    tuple[FunctionalBoundaryClosureIssue, ...],
]:
    j = elaborate_j(j)
    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    broadcasts = _broadcast_axes(j)
    logical_guards = _logical_boolean_guards(j)
    try:
        full_domains = _full_domains(j)
    except (NotImplementedError, ValueError):
        # The TypeCheck-facing analysis is intentionally only the closed
        # zero-based rectangular profile.  General logical predicates remain
        # valid J and are handled as a consumer capability boundary.
        full_domains = {}
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }

    padding_positions: dict[str, frozenset[int]] = {
        item.name: frozenset() for item in j.intent.inputs
    }
    for access in j.boundary.accesses:
        if access.kind.value != "load":
            continue
        parsed = elaboration.access_by_id.get(access.access_id)
        if parsed is None or parsed.index.direct_axes is None:
            continue
        axes = parsed.index.direct_axes
        if parsed.predicate is None:
            continue
        tensor = tensors.get(access.tensor)
        if tensor is None or len(axes) != len(tensor.shape):
            continue
        try:
            predicate = _predicate_projection(
                parsed.predicate,
                logical_guards=logical_guards,
            )
        except (NotImplementedError, ValueError):
            continue
        if predicate.guards or not predicate.upper_bounds:
            continue
        # Seed physical-padding provenance only when J itself proves a full
        # rectangular logical domain and the predicate bound agrees with the
        # typed tensor extent.  A guard or cross-axis/logical predicate is not
        # silently reclassified as padding by TypeCheck.
        if any(
            axis not in axes
            or full_domains.get(axis) != extent
            or tensor.shape[axes.index(axis)] != extent
            for axis, extent in predicate.upper_bounds
        ):
            continue
        masked = {
            axes.index(axis)
            for axis, _extent in predicate.upper_bounds
            if axis in axes
        }
        if masked:
            padding_positions[access.tensor] = frozenset(
                set(padding_positions.get(access.tensor, frozenset())) | masked
            )

    path_by_marker: dict[int, str] = {}

    def record_paths(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            path_by_marker[id(op)] = path
            record_paths(op.body, f"{path}.body")

    record_paths(j.intent.body)
    ref_by_path = {path: node_ref for node_ref, path in j.node_ids.items()}
    mean_issues: list[FunctionalBoundaryClosureIssue] = []
    consumer_issues: list[FunctionalBoundaryClosureIssue] = []

    for op in _walk_ops(j.intent.body):
        path = path_by_marker.get(id(op), "body")
        node_ref = ref_by_path.get(path)
        padded_operands = tuple(
            operand
            for operand in op.operands
            if padding_positions.get(operand, frozenset())
        )
        if not isinstance(op.result, str):
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.op",
                        "functional projection cannot erase masked padding "
                        f"provenance through unmodeled op {op.op!r}",
                        node_ref,
                    )
                )
            continue
        result_type = value_types.get(op.result)
        if result_type is None:
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.type",
                        "functional projection cannot propagate masked padding "
                        "provenance without a closed result type",
                        node_ref,
                    )
                )
            continue
        if op.op in {"map_unary", "cast"}:
            if len(op.operands) != 1:
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.operands",
                            "functional projection cannot propagate masked "
                            f"padding provenance through malformed {op.op!r}",
                            node_ref,
                        )
                    )
                continue
            padding_positions[op.result] = padding_positions.get(
                op.operands[0], frozenset()
            )
            continue
        if op.op in {"map_binary", "select"}:
            projected: set[int] = set()
            declared = broadcasts.get(op.result, {})
            for operand in op.operands:
                operand_type = value_types.get(operand)
                if operand_type is None:
                    continue
                axes = declared.get(operand)
                positions = (
                    _numpy_broadcast_positions(
                        operand_type.shape,
                        result_type.shape,
                    )
                    if axes is None
                    else _axis_broadcast_positions(
                        operand_type.shape,
                        result_type.shape,
                        axes,
                    )
                )
                if positions is None:
                    if padding_positions.get(operand, frozenset()):
                        consumer_issues.append(
                            FunctionalBoundaryClosureIssue(
                                f"j.intent.intent_function.{path}.op",
                                "functional projection cannot propagate masked "
                                "padding provenance through an incomplete "
                                "broadcast relation",
                                node_ref,
                            )
                        )
                    continue
                projected.update(
                    positions[index]
                    for index in padding_positions.get(operand, frozenset())
                    if index < len(positions)
                )
            padding_positions[op.result] = frozenset(projected)
            continue
        if op.op == "transpose":
            if len(op.operands) != 1:
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.operands",
                            "functional projection cannot propagate masked "
                            "padding provenance through malformed transpose",
                            node_ref,
                        )
                    )
                continue
            raw_perm = op.attrs.get("perm")
            if not isinstance(raw_perm, list) or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in raw_perm
            ):
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.attrs.perm",
                            "functional projection cannot propagate masked "
                            "padding provenance through an unclosed transpose",
                            node_ref,
                        )
                    )
                continue
            perm = tuple(raw_perm)
            source_padding = padding_positions.get(
                op.operands[0], frozenset()
            )
            if sorted(perm) == list(range(len(perm))):
                padding_positions[op.result] = frozenset(
                    perm.index(axis) for axis in source_padding
                )
            elif source_padding:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.perm",
                        "functional projection cannot propagate masked padding "
                        "provenance through an unclosed transpose",
                        node_ref,
                    )
                )
            continue
        if op.op != "reduce":
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.op",
                        "functional projection cannot erase masked padding "
                        f"provenance through unmodeled op {op.op!r}",
                        node_ref,
                    )
                )
            continue
        if len(op.operands) != 1:
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.operands",
                        "functional projection cannot propagate masked padding "
                        "provenance through malformed reduce",
                        node_ref,
                    )
                )
            continue

        operand_type = value_types.get(op.operands[0])
        raw_axes = op.attrs.get("axes")
        if operand_type is None or not isinstance(raw_axes, list):
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.axes",
                        "functional projection cannot propagate masked padding "
                        "provenance through an unclosed reduction",
                        node_ref,
                    )
                )
            continue
        rank = len(operand_type.shape)
        reduced: set[int] = set()
        for raw_axis in raw_axes:
            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                reduced.clear()
                break
            axis = raw_axis if raw_axis >= 0 else raw_axis + rank
            if axis < 0 or axis >= rank:
                reduced.clear()
                break
            reduced.add(axis)
        source_padding = padding_positions.get(op.operands[0], frozenset())
        if raw_axes and not reduced:
            if source_padding:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.axes",
                        "functional projection cannot propagate masked padding "
                        "provenance through an unclosed reduction",
                        node_ref,
                    )
                )
            continue
        if op.attrs.get("fn") == "mean" and reduced.intersection(source_padding):
            mean_issues.append(
                FunctionalBoundaryClosureIssue(
                    f"j.intent.intent_function.{path}.attrs.fn",
                    "padded mean cannot prove that the physical denominator "
                    "uses the logical valid count; ReductionBoundary has no "
                    "count/denominator contract",
                    node_ref,
                )
            )

        if len(result_type.shape) == rank:
            padding_positions[op.result] = frozenset(
                axis for axis in source_padding if axis not in reduced
            )
        else:
            padding_positions[op.result] = frozenset(
                axis - sum(item < axis for item in reduced)
                for axis in source_padding
                if axis not in reduced
            )

    return tuple(mean_issues), tuple(consumer_issues)


def functional_reduction_padding_issues(
    j: Any,
) -> tuple[FunctionalBoundaryClosureIssue, ...]:
    """Return the versioned TypeCheck's narrow padded-mean omissions.

    ``typecheck.v7`` consumes this J-only necessary condition.  It reports only
    cases where modeled direct-access and pointwise/broadcast provenance proves
    that a masked axis is eliminated by ``reduce(mean)`` without a logical
    count contract.  Unsupported graph operators deliberately stop this
    TypeCheck propagation; the functional consumer separately rejects masked
    provenance at that unsupported operation instead of silently erasing it.
    """

    mean_issues, _consumer_issues = _reduction_padding_analysis(j)
    return mean_issues


def _is_reduction_neutral(fn: str, value: Any) -> bool:
    if fn in {"sum", "mean"}:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 0.0
        ) or value == "ZERO"
    if fn == "prod":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 1.0
        ) or value == "ONE"
    if fn == "max":
        return isinstance(value, str) and value in {"NEG_INF", "-inf"}
    if fn == "min":
        return isinstance(value, str) and value in {"POS_INF", "+inf", "inf"}
    if fn == "any":
        return value is False or value == "false"
    if fn == "all":
        return value is True or value == "true"
    return False


def _validate_reduction_padding_closure(
    j: Any,
    *,
    has_physical_padding: bool,
) -> None:
    """Consume reduction B facts without constraining the Layer-A graph shape.

    A predicated physical realization may disappear from logical-tensor
    RefEval only when every logical reduction has one explicit boundary
    contract and its effective masked contribution is the reducer's neutral
    element.  The contract is the J-owned fact connecting arbitrary intervening
    pointwise/broadcast dataflow to that reduction; RefEval neither reconstructs
    it from Source nor invents a per-kernel padding formula.
    """

    if has_physical_padding:
        padding_issues, consumer_issues = _reduction_padding_analysis(j)
        if consumer_issues:
            raise NotImplementedError(consumer_issues[0].detail)
        if padding_issues:
            raise NotImplementedError(padding_issues[0].detail)

    nodes = _node_table(j)
    reductions = tuple(op for op in _walk_ops(j.intent.body) if op.op == "reduce")
    contract_by_reduction: dict[int, Any] = {}
    for contract in j.boundary.reductions:
        reduction = nodes.get(contract.node_ref)
        if reduction is None or reduction.op != "reduce":
            raise ValueError("Layer-B reduction does not target reduce")
        marker = id(reduction)
        if marker in contract_by_reduction:
            raise ValueError(
                "functional reduction has duplicate boundary contracts"
            )
        contract_by_reduction[marker] = contract

    if has_physical_padding and {
        id(op) for op in reductions
    } != set(contract_by_reduction):
        raise NotImplementedError(
            "functional padded graph requires exactly one contract for every "
            "logical reduction"
        )

    for reduction in reductions:
        contract = contract_by_reduction.get(id(reduction))
        if contract is None:
            continue
        fn = reduction.attrs.get("fn")
        if not isinstance(fn, str) or fn not in {
            "sum", "mean", "prod", "max", "min", "any", "all"
        }:
            raise NotImplementedError(
                "functional projection has no closed reduction neutral element"
            )
        if contract.identity is not None and not _is_reduction_neutral(
            fn, contract.identity
        ):
            raise NotImplementedError(
                f"functional projection requires neutral {fn!r} identity"
            )
        if has_physical_padding:
            if contract.masked_contribution is None:
                raise NotImplementedError(
                    "functional padded reduction requires an explicit neutral "
                    "masked_contribution"
                )
            if not _is_reduction_neutral(fn, contract.masked_contribution):
                raise NotImplementedError(
                    f"functional projection requires neutral {fn!r} "
                    "masked_contribution"
                )
        elif (
            contract.masked_contribution is not None
            and not _is_reduction_neutral(fn, contract.masked_contribution)
        ):
            raise NotImplementedError(
                f"functional projection requires neutral {fn!r} "
                "masked_contribution"
            )


def project_functional_boundary(j: Any) -> FunctionalBoundaryProjection:
    """Prove that complete Layer B can use J's logical tensor execution."""

    j = elaborate_j(j)
    closure_issues = functional_boundary_closure_issues(j)
    if closure_issues:
        # RefEval/backend projection and TypeCheck share this exact J-only
        # necessary condition.  The detailed path is retained by TypeCheck;
        # consumers preserve their established exception surface.
        raise ValueError(closure_issues[0].detail)
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
    graph_operands = {
        operand
        for operation in _node_table(j).values()
        for operand in operation.operands
    }
    loads_by_tensor: dict[str, list[Any]] = {}
    for access in boundary.accesses:
        if access.kind.value == "load":
            loads_by_tensor.setdefault(access.tensor, []).append(access)
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
    declared_dimensions = {
        dimension
        for tensor in tensors.values()
        for dimension in tensor.shape
        if isinstance(dimension, str)
    }
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
            if predicate.guards:
                raise NotImplementedError(
                    "functional mask projection does not implement "
                    "guard-bearing access predicates"
                )
            if any(
                not isinstance(extent, int)
                and extent not in declared_dimensions
                for _axis, extent in predicate.upper_bounds
            ):
                raise NotImplementedError(
                    "functional mask projection does not implement "
                    "cross-axis logical access predicates"
                )
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
            if predicate.upper_bounds:
                missing_domains = sorted(
                    axis for axis in axes if axis != "0" and axis not in domains
                )
                if missing_domains:
                    raise NotImplementedError(
                        "functional mask projection requires a complete logical "
                        "domain for every predicated access axis"
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
            if op is None:
                if len(loads_by_tensor[access.tensor]) != 1:
                    raise ValueError(
                        "functional node-less input load must be the sole load "
                        "for its J tensor"
                    )
                if (
                    access.tensor not in graph_operands
                    and (
                        j.intent.body
                        or access.tensor not in set(returns.values())
                    )
                ):
                    raise ValueError(
                        "functional node-less input load is absent from J "
                        "value dataflow"
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
            if has_predicate_or_fill and (
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
    if concrete_predicates:
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

    broadcast_axes = _broadcast_axes(j)
    _validate_broadcast_closure(j, broadcast_axes)
    _validate_reduction_padding_closure(
        j,
        has_physical_padding=has_predicate_or_fill,
    )

    mapping_values = {item.output: item.value for item in boundary.output_mappings}
    if (returns or mapping_values) and mapping_values != returns:
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

    proven_bounds = frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in boundary.index_bounds
    )
    profile = (
        "logical_domain_reduction_padding_v2"
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
    "FunctionalBoundaryClosureIssue",
    "FunctionalBoundaryProjection",
    "PredicateProjection",
    "functional_boundary_closure_issues",
    "functional_reduction_padding_issues",
    "parse_direct_axes",
    "project_predicate_text",
    "project_functional_boundary",
]
