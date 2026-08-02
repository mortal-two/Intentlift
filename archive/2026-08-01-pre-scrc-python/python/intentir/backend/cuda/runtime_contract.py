"""Runtime predicates for contract-gated CUDA lowering variants.

These predicates are backend dispatch conditions, not certification verdicts.
They consume already-recorded shape, stride, alignment, bounds-related, and
target-capability facts.  The evaluator intentionally supports a small closed
language: no Python expressions, arbitrary callbacks, or retired Gate labels
can enter a production dispatch decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class RuntimeContractError(ValueError):
    """Raised when a performance runtime contract is malformed."""


_KINDS = frozenset(
    {
        "dimension_eq",
        "dimension_ge",
        "dimension_le",
        "dimension_divisible",
        "dimension_relation",
        "stride_eq",
        "pointer_alignment",
        "target_capability",
        "dynamic_smem_le",
    }
)


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeContractError(f"{field_name} must be an integer >= {minimum}")
    return value


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], context: str
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(f"{context} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise RuntimeContractError(f"{context} missing fields: {missing}")
    if unknown:
        raise RuntimeContractError(f"{context} has unknown fields: {unknown}")


@dataclass(frozen=True)
class RuntimeLaunchFacts:
    """Logical launch facts available before a timed device launch."""

    dimensions: Mapping[str, int]
    strides: Mapping[str, tuple[int, ...]]
    pointer_addresses: Mapping[str, int]
    target_capabilities: frozenset[str]
    dynamic_smem_bytes: int = 0

    def __post_init__(self) -> None:
        dimensions: dict[str, int] = {}
        for raw_name, raw_value in self.dimensions.items():
            name = _identifier(raw_name, field_name="dimension name")
            dimensions[name] = _integer(raw_value, field_name=f"dimension {name}")
        object.__setattr__(self, "dimensions", dimensions)

        strides: dict[str, tuple[int, ...]] = {}
        for raw_name, raw_values in self.strides.items():
            name = _identifier(raw_name, field_name="stride tensor")
            if not isinstance(raw_values, tuple):
                raw_values = tuple(raw_values)
            values = tuple(
                _integer(value, field_name=f"stride {name}", minimum=0)
                for value in raw_values
            )
            strides[name] = values
        object.__setattr__(self, "strides", strides)

        pointers: dict[str, int] = {}
        for raw_name, raw_value in self.pointer_addresses.items():
            name = _identifier(raw_name, field_name="pointer tensor")
            pointers[name] = _integer(
                raw_value, field_name=f"pointer address {name}", minimum=0
            )
        object.__setattr__(self, "pointer_addresses", pointers)
        capabilities = frozenset(
            _identifier(item, field_name="target capability")
            for item in self.target_capabilities
        )
        object.__setattr__(self, "target_capabilities", capabilities)
        object.__setattr__(
            self,
            "dynamic_smem_bytes",
            _integer(self.dynamic_smem_bytes, field_name="dynamic_smem_bytes"),
        )


def runtime_launch_facts_from_tensors(
    *,
    tensors: Mapping[str, Any],
    dimension_sources: Mapping[str, tuple[str, int]],
    target_capabilities: frozenset[str] | set[str] | tuple[str, ...],
    dynamic_smem_bytes: int = 0,
) -> RuntimeLaunchFacts:
    """Derive dispatch facts from the actual tensor objects passed by the caller.

    This helper deliberately accepts no caller-supplied stride, alignment, or
    dimension booleans.  Shapes, strides, and addresses are read from the
    runtime objects so a benchmark cannot make a specialized contract match by
    asserting unchecked metadata.
    """

    if not isinstance(tensors, Mapping) or not tensors:
        raise RuntimeContractError("runtime fact extraction requires tensors")
    normalized: dict[str, Any] = {}
    strides: dict[str, tuple[int, ...]] = {}
    pointers: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for raw_name, tensor in tensors.items():
        name = _identifier(raw_name, field_name="runtime tensor name")
        if tensor is None:
            raise RuntimeContractError(f"runtime tensor {name!r} is missing")
        raw_shape = getattr(tensor, "shape", None)
        stride_fn = getattr(tensor, "stride", None)
        pointer_fn = getattr(tensor, "data_ptr", None)
        if raw_shape is None or not callable(stride_fn) or not callable(pointer_fn):
            raise RuntimeContractError(
                f"runtime tensor {name!r} must expose shape, stride(), and data_ptr()"
            )
        shape = tuple(
            _integer(int(value), field_name=f"runtime tensor {name} shape")
            for value in tuple(raw_shape)
        )
        stride = tuple(
            _integer(int(value), field_name=f"runtime tensor {name} stride")
            for value in tuple(stride_fn())
        )
        if len(shape) != len(stride):
            raise RuntimeContractError(
                f"runtime tensor {name!r} shape/stride rank mismatch"
            )
        shapes[name] = shape
        strides[name] = stride
        pointers[name] = _integer(
            int(pointer_fn()), field_name=f"runtime tensor {name} data pointer"
        )
        normalized[name] = tensor

    dimensions: dict[str, int] = {}
    for raw_dimension, source in dimension_sources.items():
        dimension = _identifier(raw_dimension, field_name="runtime dimension")
        if (
            not isinstance(source, tuple)
            or len(source) != 2
            or not isinstance(source[0], str)
            or isinstance(source[1], bool)
            or not isinstance(source[1], int)
        ):
            raise RuntimeContractError(
                "dimension sources must be (tensor_name, non-negative_axis) tuples"
            )
        tensor_name, axis = source
        if tensor_name not in normalized:
            raise RuntimeContractError(
                f"dimension {dimension!r} references unknown tensor {tensor_name!r}"
            )
        if axis < 0 or axis >= len(shapes[tensor_name]):
            raise RuntimeContractError(
                f"dimension {dimension!r} axis {axis} is outside tensor {tensor_name!r}"
            )
        dimensions[dimension] = shapes[tensor_name][axis]

    return RuntimeLaunchFacts(
        dimensions=dimensions,
        strides=strides,
        pointer_addresses=pointers,
        target_capabilities=frozenset(target_capabilities),
        dynamic_smem_bytes=dynamic_smem_bytes,
    )


@dataclass(frozen=True)
class PredicateResult:
    predicate_id: str
    matched: bool
    observed: int | str | bool | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "matched": self.matched,
            "observed": self.observed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RuntimePredicate:
    predicate_id: str
    kind: str
    subject: str
    expected: int | str | bool
    evidence_references: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _identifier(self.predicate_id, field_name="predicate_id"))
        if self.kind not in _KINDS:
            raise RuntimeContractError(f"unsupported runtime predicate kind {self.kind!r}")
        object.__setattr__(self, "subject", _identifier(self.subject, field_name="predicate subject"))
        references = tuple(
            sorted(
                {
                    _identifier(reference, field_name="predicate evidence reference")
                    for reference in self.evidence_references
                }
            )
        )
        if not references:
            raise RuntimeContractError(
                "runtime predicates require evidence references"
            )
        object.__setattr__(self, "evidence_references", references)

        if self.kind == "dimension_relation":
            _identifier(self.expected, field_name="related dimension")
        elif self.kind == "target_capability":
            if self.expected is not True:
                raise RuntimeContractError(
                    "target_capability predicates must require true"
                )
        else:
            minimum = 1 if self.kind in {"dimension_divisible", "pointer_alignment"} else 0
            expected = _integer(
                self.expected,
                field_name=f"expected value for {self.kind}",
                minimum=minimum,
            )
            object.__setattr__(self, "expected", expected)
            if self.kind == "pointer_alignment" and expected & (expected - 1):
                raise RuntimeContractError(
                    "pointer alignment must be a positive power of two"
                )
        if self.kind == "stride_eq":
            self._parse_stride_subject()
        if self.kind == "dynamic_smem_le" and self.subject != "dynamic_smem_bytes":
            raise RuntimeContractError(
                "dynamic_smem_le subject must be 'dynamic_smem_bytes'"
            )

    def _parse_stride_subject(self) -> tuple[str, int]:
        tensor, separator, raw_axis = self.subject.rpartition(":")
        if not separator or not tensor:
            raise RuntimeContractError(
                "stride_eq subject must use the form 'tensor:axis'"
            )
        try:
            axis = int(raw_axis)
        except ValueError as error:
            raise RuntimeContractError("stride axis must be an integer") from error
        if axis < 0:
            raise RuntimeContractError("stride axis must be non-negative")
        return tensor, axis

    def evaluate(self, facts: RuntimeLaunchFacts) -> PredicateResult:
        observed: int | str | bool | None
        matched = False
        reason = ""
        if self.kind.startswith("dimension_") and self.kind != "dimension_relation":
            observed = facts.dimensions.get(self.subject)
            if observed is None:
                reason = f"dimension {self.subject!r} is absent"
            elif self.kind == "dimension_eq":
                matched = observed == self.expected
            elif self.kind == "dimension_ge":
                matched = observed >= self.expected
            elif self.kind == "dimension_le":
                matched = observed <= self.expected
            else:
                matched = observed % int(self.expected) == 0
        elif self.kind == "dimension_relation":
            lhs = facts.dimensions.get(self.subject)
            rhs = facts.dimensions.get(str(self.expected))
            observed = f"{lhs}=={rhs}" if lhs is not None and rhs is not None else None
            if lhs is None or rhs is None:
                reason = "one or both related dimensions are absent"
            else:
                matched = lhs == rhs
        elif self.kind == "stride_eq":
            tensor, axis = self._parse_stride_subject()
            tensor_strides = facts.strides.get(tensor)
            observed = (
                tensor_strides[axis]
                if tensor_strides is not None and axis < len(tensor_strides)
                else None
            )
            if observed is None:
                reason = f"stride {self.subject!r} is absent"
            else:
                matched = observed == self.expected
        elif self.kind == "pointer_alignment":
            observed = facts.pointer_addresses.get(self.subject)
            if observed is None:
                reason = f"pointer {self.subject!r} is absent"
            else:
                matched = observed % int(self.expected) == 0
        elif self.kind == "target_capability":
            observed = self.subject in facts.target_capabilities
            matched = bool(observed)
        else:
            observed = facts.dynamic_smem_bytes
            matched = observed <= int(self.expected)
        if not matched and not reason:
            reason = f"observed {observed!r} does not satisfy {self.kind} {self.expected!r}"
        if matched:
            reason = "matched"
        return PredicateResult(self.predicate_id, matched, observed, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "subject": self.subject,
            "expected": self.expected,
            "evidence_references": list(self.evidence_references),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimePredicate":
        _strict_keys(
            value,
            required={"predicate_id", "kind", "subject", "expected", "evidence_references"},
            context="runtime predicate",
        )
        return cls(
            predicate_id=value["predicate_id"],
            kind=value["kind"],
            subject=value["subject"],
            expected=value["expected"],
            evidence_references=tuple(value["evidence_references"]),
        )


@dataclass(frozen=True)
class ContractMatch:
    contract_id: str
    matched: bool
    predicates: tuple[PredicateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "matched": self.matched,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
        }


@dataclass(frozen=True)
class RuntimeContract:
    contract_id: str
    predicates: tuple[RuntimePredicate, ...]
    combine: str = "all"
    schema_version: str = "intentir.paper.runtime-contract.v1"
    _predicate_ids: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != "intentir.paper.runtime-contract.v1":
            raise RuntimeContractError(f"unsupported contract schema {self.schema_version!r}")
        object.__setattr__(self, "contract_id", _identifier(self.contract_id, field_name="contract_id"))
        if self.combine != "all":
            raise RuntimeContractError("runtime contracts currently support only 'all'")
        if not self.predicates:
            raise RuntimeContractError("specialized runtime contract must not be empty")
        identifiers = [predicate.predicate_id for predicate in self.predicates]
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeContractError("runtime predicate ids must be unique")
        object.__setattr__(self, "_predicate_ids", frozenset(identifiers))

    def evaluate(self, facts: RuntimeLaunchFacts) -> ContractMatch:
        results = tuple(predicate.evaluate(facts) for predicate in self.predicates)
        return ContractMatch(
            contract_id=self.contract_id,
            matched=all(result.matched for result in results),
            predicates=results,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "combine": self.combine,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeContract":
        _strict_keys(
            value,
            required={"schema_version", "contract_id", "combine", "predicates"},
            context="runtime contract",
        )
        return cls(
            schema_version=value["schema_version"],
            contract_id=value["contract_id"],
            combine=value["combine"],
            predicates=tuple(RuntimePredicate.from_dict(item) for item in value["predicates"]),
        )


__all__ = [
    "ContractMatch",
    "PredicateResult",
    "RuntimeContract",
    "RuntimeContractError",
    "RuntimeLaunchFacts",
    "RuntimePredicate",
    "runtime_launch_facts_from_tensors",
]

