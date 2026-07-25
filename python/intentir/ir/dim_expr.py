"""Closed affine dimension expressions and constraints for artifact.v4.

The Lift model authors both the expression and every non-trivial legal-domain
constraint.  This module is deliberately only a parser and exact arithmetic
checker: it never receives a Harness valuation and never invents a missing
bound.  Its normalized forms are private compiler data, not a second J.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical_j import ContractExpr, ContractSort, parse_contract_expr


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class DimExprError(ValueError):
    """A dimension expression has no closed affine interpretation."""


def _checked_integer(value: int, *, context: str) -> int:
    if value < _INT64_MIN or value > _INT64_MAX:
        raise DimExprError(f"{context} exceeds the signed 64-bit dimension domain")
    return value


@dataclass(frozen=True)
class AffineDimExpr:
    """Exact affine form ``constant + sum(coeff_i * x_i)``."""

    constant: int = 0
    coefficients: tuple[tuple[str, int], ...] = ()

    @classmethod
    def create(
        cls,
        constant: int,
        coefficients: Iterable[tuple[str, int]] = (),
        *,
        context: str,
    ) -> "AffineDimExpr":
        combined: dict[str, int] = {}
        for name, coefficient in coefficients:
            combined[name] = combined.get(name, 0) + coefficient
        normalized = tuple(
            (name, _checked_integer(coefficient, context=context))
            for name, coefficient in sorted(combined.items())
            if coefficient != 0
        )
        return cls(_checked_integer(constant, context=context), normalized)

    @property
    def is_constant(self) -> bool:
        return not self.coefficients

    def scaled(self, factor: int, *, context: str) -> "AffineDimExpr":
        return AffineDimExpr.create(
            self.constant * factor,
            ((name, coefficient * factor) for name, coefficient in self.coefficients),
            context=context,
        )

    def plus(
        self,
        other: "AffineDimExpr",
        *,
        factor: int = 1,
        context: str,
    ) -> "AffineDimExpr":
        return AffineDimExpr.create(
            self.constant + factor * other.constant,
            (
                *self.coefficients,
                *(
                    (name, factor * coefficient)
                    for name, coefficient in other.coefficients
                ),
            ),
            context=context,
        )

    def to_text(self) -> str:
        terms: list[str] = []
        for name, coefficient in self.coefficients:
            if coefficient == 1:
                terms.append(name)
            elif coefficient == -1:
                terms.append(f"-{name}")
            else:
                terms.append(f"{coefficient}*{name}")
        if self.constant != 0 or not terms:
            terms.append(str(self.constant))
        return " + ".join(terms).replace("+ -", "- ")

    def evaluate(self, dimensions: dict[str, int], *, context: str) -> int:
        value = self.constant
        for name, coefficient in self.coefficients:
            if name not in dimensions:
                raise DimExprError(
                    f"{context} lacks a concrete valuation for {name!r}"
                )
            extent = dimensions[name]
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise DimExprError(
                    f"{context} valuation for {name!r} must be an integer"
                )
            value += coefficient * extent
        return _checked_integer(value, context=context)


def _affine_from_contract(
    expression: ContractExpr,
    *,
    dimensions: frozenset[str],
    context: str,
) -> AffineDimExpr:
    if expression.op == "int":
        assert isinstance(expression.value, int) and not isinstance(
            expression.value, bool
        )
        return AffineDimExpr.create(expression.value, context=context)
    if expression.op == "ref":
        assert isinstance(expression.value, str)
        if expression.value not in dimensions:
            raise DimExprError(
                f"{context} references unknown symbolic dimension "
                f"{expression.value!r}"
            )
        return AffineDimExpr.create(
            0, ((expression.value, 1),), context=context
        )
    if expression.op in {"pos", "neg"}:
        operand = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        return operand if expression.op == "pos" else operand.scaled(
            -1, context=context
        )
    if expression.op in {"add", "sub"}:
        left = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        right = _affine_from_contract(
            expression.args[1], dimensions=dimensions, context=context
        )
        return left.plus(
            right,
            factor=1 if expression.op == "add" else -1,
            context=context,
        )
    if expression.op == "mul":
        left = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        right = _affine_from_contract(
            expression.args[1], dimensions=dimensions, context=context
        )
        if left.is_constant:
            return right.scaled(left.constant, context=context)
        if right.is_constant:
            return left.scaled(right.constant, context=context)
        raise DimExprError(
            f"{context} is non-affine: multiplication requires one constant operand"
        )
    raise DimExprError(
        f"{context} uses unsupported non-affine operator {expression.op!r}"
    )


def parse_affine_dim_expr(
    value: int | str,
    *,
    dimensions: frozenset[str],
    context: str,
) -> AffineDimExpr:
    """Parse one int/string surface value into an exact affine form."""

    if isinstance(value, bool):
        raise DimExprError(f"{context} must be an integer dimension expression")
    if isinstance(value, int):
        return AffineDimExpr.create(value, context=context)
    if not isinstance(value, str):
        raise DimExprError(f"{context} must be an integer or ContractExpr string")
    try:
        parsed = parse_contract_expr(
            value,
            expected=ContractSort.INTEGER,
            context=context,
        )
    except ValueError as exc:
        raise DimExprError(str(exc)) from exc
    return _affine_from_contract(
        parsed, dimensions=dimensions, context=context
    )


def _constraint_lower_forms(
    expression: ContractExpr,
    *,
    dimensions: frozenset[str],
    context: str,
) -> tuple[AffineDimExpr, ...]:
    """Return authored atoms normalized to integer forms known ``>= 0``."""

    if expression.op == "and":
        return tuple(
            form
            for argument in expression.args
            for form in _constraint_lower_forms(
                argument, dimensions=dimensions, context=context
            )
        )
    if expression.op not in {"le", "lt", "ge", "gt", "eq"}:
        raise DimExprError(
            f"{context} permits only affine comparisons and conjunction"
        )
    left = _affine_from_contract(
        expression.args[0], dimensions=dimensions, context=context
    )
    right = _affine_from_contract(
        expression.args[1], dimensions=dimensions, context=context
    )
    if expression.op in {"ge", "gt"}:
        lower = left.plus(right, factor=-1, context=context)
    else:
        lower = right.plus(left, factor=-1, context=context)
    if expression.op in {"lt", "gt"}:
        lower = lower.plus(
            AffineDimExpr.create(1, context=context),
            factor=-1,
            context=context,
        )
    if expression.op == "eq":
        return (lower, lower.scaled(-1, context=context))
    return (lower,)


@dataclass(frozen=True)
class DimensionConstraintSet:
    """Parsed affine legal-domain facts with a small, sound bound prover."""

    lower_forms: tuple[AffineDimExpr, ...]
    lower_bounds: tuple[tuple[str, int], ...]
    upper_bounds: tuple[tuple[str, int], ...]

    @classmethod
    def parse(
        cls,
        expressions: Iterable[str],
        *,
        dimensions: frozenset[str],
        context: str = "j.dimension_constraints",
    ) -> "DimensionConstraintSet":
        forms: list[AffineDimExpr] = []
        for index, value in enumerate(expressions):
            item_context = f"{context}[{index}]"
            try:
                parsed = parse_contract_expr(
                    value,
                    expected=ContractSort.BOOLEAN,
                    context=item_context,
                )
            except ValueError as exc:
                raise DimExprError(str(exc)) from exc
            forms.extend(
                _constraint_lower_forms(
                    parsed,
                    dimensions=dimensions,
                    context=item_context,
                )
            )

        lower = {name: 0 for name in dimensions}
        upper: dict[str, int] = {}
        for form in forms:
            if not form.coefficients:
                if form.constant < 0:
                    raise DimExprError(
                        f"{context} contains an unsatisfiable constant constraint"
                    )
                continue
            if len(form.coefficients) != 1:
                continue
            name, coefficient = form.coefficients[0]
            if coefficient > 0:
                # coefficient*x + constant >= 0
                bound = (-form.constant + coefficient - 1) // coefficient
                lower[name] = max(lower[name], bound)
            else:
                divisor = -coefficient
                # -divisor*x + constant >= 0
                bound = form.constant // divisor
                upper[name] = min(upper.get(name, bound), bound)
        for name in dimensions:
            if name in upper and lower[name] > upper[name]:
                raise DimExprError(
                    f"{context} is unsatisfiable for {name!r}: "
                    f"lower bound {lower[name]} exceeds upper bound {upper[name]}"
                )
        return cls(
            tuple(forms),
            tuple(sorted(lower.items())),
            tuple(sorted(upper.items())),
        )

    def proves_nonnegative(self, expression: AffineDimExpr) -> bool:
        """Prove ``expression >= 0`` without any concrete runtime values."""

        if expression in self.lower_forms:
            return True
        lower = dict(self.lower_bounds)
        upper = dict(self.upper_bounds)
        minimum = expression.constant
        for name, coefficient in expression.coefficients:
            if coefficient >= 0:
                minimum += coefficient * lower.get(name, 0)
                continue
            if name not in upper:
                return False
            minimum += coefficient * upper[name]
        return minimum >= 0

    def validate_valuation(self, dimensions: dict[str, int]) -> None:
        """Reject one public instantiation outside the authored legal domain."""

        for index, form in enumerate(self.lower_forms):
            if form.evaluate(
                dimensions,
                context=f"j.dimension_constraints atom {index}",
            ) < 0:
                raise DimExprError(
                    "public dimension valuation violates "
                    f"J.dimension_constraints atom {form.to_text()!r} >= 0"
                )


def validate_dimension_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
) -> None:
    """Validate one public J dimension valuation and every resolved pad extent."""

    declared = frozenset(
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str)
    )
    concrete = dict(dimensions)
    missing = sorted(declared - set(concrete))
    if missing:
        raise DimExprError(
            "public instantiation lacks J dimension(s): " + ", ".join(missing)
        )
    for name in sorted(declared):
        value = concrete[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DimExprError(
                f"public dimension {name!r} must be a non-negative integer"
            )

    constraints = DimensionConstraintSet.parse(
        j.dimension_constraints,
        dimensions=declared,
    )
    constraints.validate_valuation(concrete)

    def validate_pad_extents(operations: list[Mapping[str, Any]]) -> None:
        for operation in operations:
            if operation.get("op") == "pad":
                attrs = operation.get("attrs")
                if isinstance(attrs, Mapping):
                    for field in ("lo", "hi"):
                        values = attrs.get(field)
                        if not isinstance(values, list):
                            continue
                        for index, expression in enumerate(values):
                            context = f"public pad {field}[{index}]"
                            resolved = parse_affine_dim_expr(
                                expression,
                                dimensions=declared,
                                context=context,
                            ).evaluate(concrete, context=context)
                            if resolved < 0:
                                raise DimExprError(
                                    "public dimension valuation resolves "
                                    f"pad {field}[{index}] to a negative extent"
                                )
            body = operation.get("body")
            if isinstance(body, list):
                validate_pad_extents(body)

    intent = j.intent.to_json()["intent_function"]
    validate_pad_extents(intent.get("body", []))


__all__ = [
    "AffineDimExpr",
    "DimExprError",
    "DimensionConstraintSet",
    "parse_affine_dim_expr",
    "validate_dimension_instantiation",
]
