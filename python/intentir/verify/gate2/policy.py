"""Trusted external comparison policy Q for Gate2.

J declares only a closed comparison class.  Q supplies numeric thresholds and
NaN handling, so a Lift candidate cannot choose its own passing criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from intentir.verify.gate1.model import ResultRelationKind, SemanticArchive


GATE2_POLICY_PROTOCOL = "intentir.scrc.gate2-q.v1"

_DTYPE_ALIASES = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
    "bool": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
}


def canonical_dtype(value: str) -> str:
    normalized = value.removeprefix("torch.").lower()
    return _DTYPE_ALIASES.get(normalized, normalized)


class ComparisonPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class NumericTolerance:
    rtol: float
    atol: float
    equal_nan: bool = False

    def __post_init__(self) -> None:
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"Gate2 Q {name} must be finite and non-negative")
        if not isinstance(self.equal_nan, bool):
            raise TypeError("Gate2 Q equal_nan must be boolean")

    def to_json(self) -> dict[str, float | bool]:
        return {
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": self.equal_nan,
        }


@dataclass(frozen=True)
class CloseRule:
    comparison_class: ResultRelationKind
    dtype: str
    tolerance: NumericTolerance | None

    @property
    def exact(self) -> bool:
        return self.comparison_class is ResultRelationKind.EXACT


@dataclass(frozen=True)
class ComparisonPolicy:
    policy_id: str
    tolerances: tuple[tuple[str, NumericTolerance], ...]
    protocol: str = GATE2_POLICY_PROTOCOL

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_.-]*", self.policy_id) is None:
            raise ValueError("Gate2 Q policy_id must be a stable lowercase id")
        normalized = tuple(
            (canonical_dtype(dtype), tolerance)
            for dtype, tolerance in self.tolerances
        )
        keys = [dtype for dtype, _ in normalized]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("Gate2 Q dtype rules must be unique and sorted")
        if any(
            not isinstance(tolerance, NumericTolerance)
            for _, tolerance in normalized
        ):
            raise TypeError("Gate2 Q entries must be NumericTolerance values")
        object.__setattr__(self, "tolerances", normalized)
        if self.protocol != GATE2_POLICY_PROTOCOL:
            raise ValueError("unsupported Gate2 Q protocol")

    @classmethod
    def create(
        cls,
        policy_id: str,
        tolerances: Iterable[tuple[str, NumericTolerance]],
    ) -> "ComparisonPolicy":
        return cls(policy_id, tuple(sorted(tolerances, key=lambda item: item[0])))

    def rule_for(
        self,
        comparison_class: ResultRelationKind | str,
        dtype: str,
    ) -> CloseRule:
        kind = ResultRelationKind(comparison_class)
        dtype = canonical_dtype(dtype)
        if kind is ResultRelationKind.EXACT:
            return CloseRule(kind, dtype, None)
        if kind is ResultRelationKind.ONE_OF_UPDATES:
            raise ComparisonPolicyError(
                "set-valued result relations are outside the active Gate2 main path"
            )
        if not dtype.startswith(("f", "bf")):
            raise ComparisonPolicyError(
                "numeric tolerance is allowed only for floating outputs"
            )
        tolerance = dict(self.tolerances).get(dtype)
        if tolerance is None:
            raise ComparisonPolicyError(
                f"Gate2 Q has no tolerance rule for dtype {dtype!r}"
            )
        return CloseRule(kind, dtype, tolerance)

    def validate_j(self, j: SemanticArchive) -> None:
        explicit = {item.output: item for item in j.boundary.result_relations}
        for output in j.intent.outputs:
            relation = explicit.get(output.name)
            kind = (
                relation.kind
                if relation is not None
                else ResultRelationKind.EXACT
            )
            self.rule_for(kind, output.dtype)

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "policy_id": self.policy_id,
            "tolerances": {
                dtype: tolerance.to_json()
                for dtype, tolerance in self.tolerances
            },
        }


DEFAULT_GATE2_POLICY = ComparisonPolicy.create(
    "intentir.gate2.default.v1",
    (
        ("bf16", NumericTolerance(2.0e-2, 2.0e-3)),
        ("f16", NumericTolerance(1.0e-2, 1.0e-3)),
        ("f32", NumericTolerance(1.0e-5, 1.0e-6)),
        ("f64", NumericTolerance(1.0e-12, 1.0e-12)),
    ),
)


__all__ = [
    "CloseRule",
    "ComparisonPolicy",
    "ComparisonPolicyError",
    "DEFAULT_GATE2_POLICY",
    "GATE2_POLICY_PROTOCOL",
    "NumericTolerance",
    "canonical_dtype",
]
