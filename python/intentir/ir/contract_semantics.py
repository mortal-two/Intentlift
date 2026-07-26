"""Typed, target-neutral execution semantics for canonical ContractExpr.

The canonical J decoder owns the closed syntax.  This module owns the one
deterministic meaning of that syntax.  Consumers must not reparse expression
text or recognize kernel-shaped formula templates in order to evaluate an
index, dimension, or predicate.

The evaluator deliberately accepts only explicit reference bindings and the
closed integer/boolean opcode set admitted by :mod:`intentir.ir.canonical_j`.
Integer expressions have target-neutral mathematical-integer semantics here;
machine width and overflow are lowering obligations, not facts this consumer
may silently add to J.  Division by zero is outside the expression domain and
therefore fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

from .canonical_j import CanonicalIndex, ContractExpr, ContractSort


class ContractSemanticsError(ValueError):
    """A canonical ContractExpr has no value in the supplied typed environment."""


def _require_integer(value: Any, *, context: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ContractSemanticsError(f"{context} requires an integer value")
    return int(value)


def _require_boolean(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise ContractSemanticsError(f"{context} requires a boolean value")
    return value


def _resolved_sort(
    expression: ContractExpr,
    reference_sorts: Mapping[str, ContractSort],
    *,
    context: str,
) -> ContractSort:
    """Resolve UNKNOWN_REF leaves and validate the complete typed expression."""

    if expression.op == "ref":
        assert isinstance(expression.value, str)
        result = reference_sorts.get(expression.value)
        if result not in {ContractSort.INTEGER, ContractSort.BOOLEAN}:
            raise ContractSemanticsError(
                f"{context} lacks a closed sort for reference {expression.value!r}"
            )
        return result
    if expression.op == "int":
        return ContractSort.INTEGER
    if expression.op == "bool":
        return ContractSort.BOOLEAN

    argument_sorts = tuple(
        _resolved_sort(
            argument,
            reference_sorts,
            context=f"{context}.{expression.op}[{index}]",
        )
        for index, argument in enumerate(expression.args)
    )
    if expression.op in {"pos", "neg"}:
        expected = (ContractSort.INTEGER,)
        result = ContractSort.INTEGER
    elif expression.op == "not":
        expected = (ContractSort.BOOLEAN,)
        result = ContractSort.BOOLEAN
    elif expression.op in {"add", "sub", "mul", "floordiv", "mod"}:
        expected = (ContractSort.INTEGER, ContractSort.INTEGER)
        result = ContractSort.INTEGER
    elif expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
        expected = (ContractSort.INTEGER, ContractSort.INTEGER)
        result = ContractSort.BOOLEAN
    elif expression.op in {"and", "or"}:
        if len(argument_sorts) < 2:
            raise ContractSemanticsError(
                f"{context} boolean {expression.op} requires at least two operands"
            )
        expected = (ContractSort.BOOLEAN,) * len(argument_sorts)
        result = ContractSort.BOOLEAN
    elif expression.op == "call":
        if expression.value not in {"ceil_div", "min", "max"}:
            raise ContractSemanticsError(
                f"{context} has unsupported ContractExpr call {expression.value!r}"
            )
        expected = (ContractSort.INTEGER, ContractSort.INTEGER)
        result = ContractSort.INTEGER
    else:
        raise ContractSemanticsError(
            f"{context} has unknown ContractExpr opcode {expression.op!r}"
        )
    if argument_sorts != expected:
        raise ContractSemanticsError(
            f"{context} {expression.op} operand sorts are not closed: "
            f"expected {[item.value for item in expected]}, "
            f"got {[item.value for item in argument_sorts]}"
        )
    if expression.sort not in {ContractSort.UNKNOWN_REF, result}:
        raise ContractSemanticsError(
            f"{context} declared sort {expression.sort.value!r} disagrees with "
            f"{result.value!r}"
        )
    return result


@dataclass(frozen=True)
class TypedContractProgram:
    """One type-checked ContractExpr ready for deterministic execution/lowering."""

    expression: ContractExpr
    reference_sorts: tuple[tuple[str, ContractSort], ...]
    result_sort: ContractSort

    @classmethod
    def compile(
        cls,
        expression: ContractExpr,
        reference_sorts: Mapping[str, ContractSort],
        *,
        expected_sort: ContractSort | None = None,
        context: str = "ContractExpr",
    ) -> "TypedContractProgram":
        if not isinstance(expression, ContractExpr):
            raise TypeError("typed ContractExpr compilation requires ContractExpr")
        closed_sorts = {
            name: sort
            for name, sort in reference_sorts.items()
            if name in expression.references
        }
        if set(closed_sorts) != set(expression.references):
            missing = sorted(set(expression.references) - set(closed_sorts))
            raise ContractSemanticsError(
                f"{context} lacks reference sorts for {missing}"
            )
        result_sort = _resolved_sort(
            expression,
            closed_sorts,
            context=context,
        )
        if expected_sort is not None and result_sort is not expected_sort:
            raise ContractSemanticsError(
                f"{context} must produce {expected_sort.value}, got "
                f"{result_sort.value}"
            )
        return cls(
            expression,
            tuple(sorted(closed_sorts.items())),
            result_sort,
        )

    @property
    def reference_sort_map(self) -> dict[str, ContractSort]:
        return dict(self.reference_sorts)

    def evaluate(
        self,
        bindings: Mapping[str, Any],
        *,
        context: str = "ContractExpr",
    ) -> int | bool:
        required = set(self.expression.references)
        if not required.issubset(bindings):
            raise ContractSemanticsError(
                f"{context} lacks bindings for {sorted(required - set(bindings))}"
            )
        sorts = self.reference_sort_map

        def visit(expression: ContractExpr, path: str) -> int | bool:
            if expression.op == "ref":
                assert isinstance(expression.value, str)
                value = bindings[expression.value]
                if sorts[expression.value] is ContractSort.INTEGER:
                    return _require_integer(value, context=path)
                return _require_boolean(value, context=path)
            if expression.op == "int":
                assert isinstance(expression.value, int) and not isinstance(
                    expression.value, bool
                )
                return expression.value
            if expression.op == "bool":
                assert isinstance(expression.value, bool)
                return expression.value

            values = tuple(
                visit(argument, f"{path}.{expression.op}[{index}]")
                for index, argument in enumerate(expression.args)
            )
            if expression.op == "pos":
                return _require_integer(values[0], context=path)
            if expression.op == "neg":
                return -_require_integer(values[0], context=path)
            if expression.op == "not":
                return not _require_boolean(values[0], context=path)
            if expression.op in {"add", "sub", "mul", "floordiv", "mod"}:
                left = _require_integer(values[0], context=path)
                right = _require_integer(values[1], context=path)
                if expression.op in {"floordiv", "mod"} and right == 0:
                    raise ContractSemanticsError(f"{path} divides by zero")
                result = {
                    "add": lambda: left + right,
                    "sub": lambda: left - right,
                    "mul": lambda: left * right,
                    "floordiv": lambda: left // right,
                    "mod": lambda: left % right,
                }[expression.op]()
                return result
            if expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
                left = _require_integer(values[0], context=path)
                right = _require_integer(values[1], context=path)
                return {
                    "lt": left < right,
                    "le": left <= right,
                    "gt": left > right,
                    "ge": left >= right,
                    "eq": left == right,
                    "ne": left != right,
                }[expression.op]
            if expression.op == "and":
                return all(_require_boolean(value, context=path) for value in values)
            if expression.op == "or":
                return any(_require_boolean(value, context=path) for value in values)
            if expression.op == "call":
                left = _require_integer(values[0], context=path)
                right = _require_integer(values[1], context=path)
                if expression.value == "ceil_div":
                    if right == 0:
                        raise ContractSemanticsError(f"{path} divides by zero")
                    result = -((-left) // right)
                elif expression.value == "min":
                    result = min(left, right)
                elif expression.value == "max":
                    result = max(left, right)
                else:  # guarded by compile(), retained fail-closed for corruption
                    raise ContractSemanticsError(
                        f"{path} has unsupported call {expression.value!r}"
                    )
                return result
            raise ContractSemanticsError(
                f"{path} has unknown ContractExpr opcode {expression.op!r}"
            )

        return visit(self.expression, context)


def evaluate_contract_expr(
    expression: ContractExpr,
    bindings: Mapping[str, Any],
    *,
    reference_sorts: Mapping[str, ContractSort] | None = None,
    expected_sort: ContractSort | None = None,
    context: str = "ContractExpr",
) -> int | bool:
    """Compile and evaluate one canonical expression in a closed environment."""

    inferred_sorts = (
        dict(reference_sorts)
        if reference_sorts is not None
        else {
            name: (
                ContractSort.BOOLEAN
                if type(bindings.get(name)) is bool
                else ContractSort.INTEGER
            )
            for name in expression.references
        }
    )
    return TypedContractProgram.compile(
        expression,
        inferred_sorts,
        expected_sort=expected_sort,
        context=context,
    ).evaluate(bindings, context=context)


def evaluate_contract_index(
    index: CanonicalIndex,
    bindings: Mapping[str, Any],
    *,
    reference_sorts: Mapping[str, ContractSort] | None = None,
    context: str = "index_expression",
) -> tuple[int, ...]:
    """Execute an arbitrary canonical index vector without formula matching."""

    result: list[int] = []
    for position, component in enumerate(index.components):
        value = evaluate_contract_expr(
            component,
            bindings,
            reference_sorts=reference_sorts,
            expected_sort=ContractSort.INTEGER,
            context=f"{context}[{position}]",
        )
        result.append(_require_integer(value, context=f"{context}[{position}]"))
    return tuple(result)


__all__ = [
    "ContractSemanticsError",
    "TypedContractProgram",
    "evaluate_contract_expr",
    "evaluate_contract_index",
]
