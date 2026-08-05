"""Closed validation and consumption of J-owned dimension definitions.

The relations are authored by Lift.  Nothing in this module infers a relation
from concrete shapes, Harness values, grid, C, pi, Source, or compiler output.
Consumers may only validate or evaluate the exact closed expression carried by
J.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


class DimensionRelationError(ValueError):
    """A J dimension relation or its public valuation is not closed."""


def _declared_dimensions(j: Any) -> frozenset[str]:
    dimensions = {
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str)
    }
    signature = getattr(j, "signature", None)
    for parameter in getattr(signature, "parameters", ()):
        logical_ref = getattr(parameter, "logical_ref", None)
        if isinstance(logical_ref, str) and re.fullmatch(r"x[0-9]+", logical_ref):
            dimensions.add(logical_ref)
    return frozenset(dimensions)


def _relation_map(j: Any) -> dict[str, Any]:
    relations: dict[str, Any] = {}
    for index, relation in enumerate(j.dimension_relations):
        path = f"j.dimension_relations[{index}]"
        if relation.result in relations:
            raise DimensionRelationError(
                f"{path}.result duplicates an earlier dimension definition"
            )
        relations[relation.result] = relation
    return relations


def validate_dimension_relations(j: Any) -> None:
    """Validate references, closed operations, unique results, and acyclicity."""

    declared = _declared_dimensions(j)
    relations = _relation_map(j)
    for index, relation in enumerate(j.dimension_relations):
        path = f"j.dimension_relations[{index}]"
        if relation.result not in declared:
            raise DimensionRelationError(
                f"{path}.result must name a declared symbolic x* dimension"
            )
        if relation.expr.arg not in declared:
            raise DimensionRelationError(
                f"{path}.expr.arg must name a declared symbolic x* dimension"
            )
        if relation.expr.rhs is not None and relation.expr.rhs not in declared:
            raise DimensionRelationError(
                f"{path}.expr.rhs must name a declared symbolic x* dimension"
            )
        if relation.expr.op == "identity":
            if relation.expr.value is not None or relation.expr.rhs is not None:
                raise DimensionRelationError(
                    f"{path}.expr identity forbids value/rhs"
                )
        elif relation.expr.op == "mul_dims":
            if relation.expr.rhs is None or relation.expr.value is not None:
                raise DimensionRelationError(
                    f"{path}.expr mul_dims requires rhs and forbids value"
                )
        elif relation.expr.op in {
            "add_const",
            "mul_const",
            "ceil_div_const",
            "round_up_const",
        }:
            if (
                isinstance(relation.expr.value, bool)
                or not isinstance(relation.expr.value, int)
                or relation.expr.value <= 0
            ):
                raise DimensionRelationError(
                    f"{path}.expr requires a positive integer constant"
                )
        else:
            raise DimensionRelationError(
                f"{path}.expr.op is outside the closed dimension grammar"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(result: str) -> None:
        if result in visited:
            return
        if result in visiting:
            raise DimensionRelationError(
                "j.dimension_relations contains a dependency cycle at "
                f"{result!r}"
            )
        visiting.add(result)
        expr = relations[result].expr
        dependencies = [expr.arg]
        if expr.rhs is not None:
            dependencies.append(expr.rhs)
        for argument in dependencies:
            if argument in relations:
                visit(argument)
        visiting.remove(result)
        visited.add(result)

    for result in relations:
        visit(result)


def _closed_extent(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DimensionRelationError(
            f"{context} must be a non-negative integer dimension"
        )
    return value


def evaluate_dimension_relation_expr(
    expr: Any, argument: int, rhs_argument: int | None = None
) -> int:
    """Evaluate one already-validated closed expression."""

    argument = _closed_extent(argument, context=f"dimension {expr.arg!r}")
    if expr.op == "identity":
        return argument
    if expr.op == "mul_dims":
        if rhs_argument is None:
            raise DimensionRelationError("mul_dims requires a concrete rhs dimension")
        rhs_argument = _closed_extent(
            rhs_argument, context=f"dimension {expr.rhs!r}"
        )
        return argument * rhs_argument
    constant = expr.value
    if isinstance(constant, bool) or not isinstance(constant, int) or constant <= 0:
        raise DimensionRelationError(
            f"{expr.op} requires a positive integer constant"
        )
    if expr.op == "add_const":
        return argument + constant
    if expr.op == "mul_const":
        return argument * constant
    if expr.op == "ceil_div_const":
        return (argument + constant - 1) // constant
    if expr.op == "round_up_const":
        return ((argument + constant - 1) // constant) * constant
    raise DimensionRelationError(
        f"unsupported closed dimension relation operation {expr.op!r}"
    )


def resolve_dimension_relations(
    j: Any,
    dimensions: Mapping[str, int],
) -> dict[str, int]:
    """Evaluate authored relations from public base valuations, failing closed.

    A supplied result valuation is checked rather than overwritten.  Missing
    derived results are filled only from the exact J relation.  A missing base
    dimension remains an error; no Harness or tensor-shape inference occurs here.
    """

    validate_dimension_relations(j)
    declared = _declared_dimensions(j)
    resolved = {
        name: _closed_extent(value, context=f"dimension {name!r}")
        for name, value in dimensions.items()
        if name in declared
    }
    unexpected = sorted(set(dimensions) - declared)
    if unexpected:
        raise DimensionRelationError(
            "public dimension valuation contains unknown J dimension(s): "
            + ", ".join(unexpected)
        )
    relations = _relation_map(j)

    pending = set(relations)
    while pending:
        progress = False
        for result in tuple(sorted(pending)):
            relation = relations[result]
            dependencies = [relation.expr.arg]
            if relation.expr.rhs is not None:
                dependencies.append(relation.expr.rhs)
            if any(dependency not in resolved for dependency in dependencies):
                continue
            expected = evaluate_dimension_relation_expr(
                relation.expr,
                resolved[relation.expr.arg],
                (
                    resolved[relation.expr.rhs]
                    if relation.expr.rhs is not None
                    else None
                ),
            )
            if result in resolved and resolved[result] != expected:
                raise DimensionRelationError(
                    "J dimension relation mismatch: "
                    f"{result}={resolved[result]}, expected {expected} from "
                    f"{relation.expr.op}({relation.expr.arg}"
                    + (
                        f", {relation.expr.rhs})"
                        if relation.expr.rhs is not None
                        else ")"
                    )
                )
            resolved[result] = expected
            pending.remove(result)
            progress = True
        if not progress:
            missing = sorted({
                dependency
                for result in pending
                for dependency in (
                    relations[result].expr.arg,
                    *((relations[result].expr.rhs,) if relations[result].expr.rhs is not None else ()),
                )
                if dependency not in resolved
            })
            raise DimensionRelationError(
                "J dimension relations lack base valuation(s): " + ", ".join(missing)
            )

    missing_declared = sorted(declared - set(resolved))
    if missing_declared:
        raise DimensionRelationError(
            "public instantiation lacks J dimension(s): "
            + ", ".join(missing_declared)
        )
    return {name: resolved[name] for name in sorted(resolved)}


def validate_dimension_relation_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
) -> None:
    """Check one complete public valuation against every authored relation."""

    resolve_dimension_relations(j, dimensions)


__all__ = [
    "DimensionRelationError",
    "evaluate_dimension_relation_expr",
    "resolve_dimension_relations",
    "validate_dimension_relation_instantiation",
    "validate_dimension_relations",
]
