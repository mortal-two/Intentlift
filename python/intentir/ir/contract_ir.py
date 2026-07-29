"""Total, target-neutral projection of canonical J's authored contract.

``CanonicalContractIR`` is an ephemeral compiler IR derived one-to-one from J.
It preserves every Layer-B relation while replacing domain, predicate, and
index surface strings with typed ContractExpr programs.  It performs no kernel
shape recognition and does not decide whether a particular RefEval or target
backend can execute the contract.

This distinction is deliberate: constructing M0 must not use a backend
capability matcher as a J-membership oracle.  Concrete consumers lower this IR
and may report an explicit unsupported capability without changing J.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_j import ContractSort, elaborate_j, j_elaboration
from .contract_semantics import ContractSemanticsError, TypedContractProgram
from .dimension_relation import DimensionRelationError, validate_dimension_relations


CONTRACT_IR_PROTOCOL = "intentir.canonical-contract-ir.v2"


class ContractIRProjectionError(ValueError):
    """Canonical J cannot be projected without dropping or inventing a relation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _dtype_sort(dtype: str) -> ContractSort | None:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized in {"bool", "i1"}:
        return ContractSort.BOOLEAN
    if normalized in {
        "byte", "char", "short", "int", "long",
        "uint8", "uint16", "uint32", "uint64",
        "int8", "int16", "int32", "int64",
        "i8", "i16", "i32", "i64",
    }:
        return ContractSort.INTEGER
    return None


@dataclass(frozen=True)
class ContractDomainIR:
    domain_id: str
    predicate: TypedContractProgram

    def to_json(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "predicate": self.predicate.expression.to_json(),
            "reference_sorts": {
                name: sort.value for name, sort in self.predicate.reference_sorts
            },
        }


@dataclass(frozen=True)
class ContractAccessIR:
    access_id: str
    kind: str
    tensor: str
    node_ref: str | None
    index: tuple[TypedContractProgram, ...]
    predicate: TypedContractProgram | None
    fill_json: str

    @property
    def fill_value(self) -> Any:
        return json.loads(self.fill_json)

    def to_json(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "kind": self.kind,
            "tensor": self.tensor,
            "node_ref": self.node_ref,
            "index": [program.expression.to_json() for program in self.index],
            "index_reference_sorts": [
                {name: sort.value for name, sort in program.reference_sorts}
                for program in self.index
            ],
            "predicate": (
                self.predicate.expression.to_json()
                if self.predicate is not None else None
            ),
            "predicate_reference_sorts": (
                {
                    name: sort.value
                    for name, sort in self.predicate.reference_sorts
                }
                if self.predicate is not None else None
            ),
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class CanonicalContractIR:
    """Immutable, complete Layer-B/compiler-interface view of one canonical J."""

    domains: tuple[ContractDomainIR, ...]
    accesses: tuple[ContractAccessIR, ...]
    reference_sorts: tuple[tuple[str, ContractSort], ...]
    remaining_contract_json: str

    @property
    def remaining_contract(self) -> dict[str, Any]:
        value = json.loads(self.remaining_contract_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor owns it
            raise AssertionError("canonical contract payload is not an object")
        return value

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_json()).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CONTRACT_IR_PROTOCOL,
            "domains": [domain.to_json() for domain in self.domains],
            "accesses": [access.to_json() for access in self.accesses],
            "reference_sorts": {
                name: sort.value for name, sort in self.reference_sorts
            },
            "contract": self.remaining_contract,
        }


def _logical_sort_hints(j: Any) -> dict[str, ContractSort]:
    elaboration = j_elaboration(j)
    hints: dict[str, ContractSort] = {}
    for name, value_type in elaboration.value_types:
        if value_type.shape:
            continue
        sort = _dtype_sort(value_type.dtype)
        if sort is not None:
            hints[name] = sort
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        for extent in tensor.shape:
            if isinstance(extent, str):
                previous = hints.get(extent)
                if previous not in {None, ContractSort.INTEGER}:
                    raise ContractIRProjectionError(
                        f"symbolic dimension {extent!r} conflicts with a boolean value"
                    )
                hints[extent] = ContractSort.INTEGER
    for relation in j.dimension_relations:
        for extent in (relation.result, relation.expr.arg):
            previous = hints.get(extent)
            if previous not in {None, ContractSort.INTEGER}:
                raise ContractIRProjectionError(
                    f"dimension relation reference {extent!r} conflicts with a boolean value"
                )
            hints[extent] = ContractSort.INTEGER
    return hints


def _merge_sorts(
    target: dict[str, ContractSort],
    source: Mapping[str, ContractSort],
    *,
    context: str,
) -> None:
    for name, sort in source.items():
        previous = target.get(name)
        if previous is not None and previous is not sort:
            raise ContractIRProjectionError(
                f"{context} gives reference {name!r} conflicting sorts "
                f"{previous.value!r} and {sort.value!r}"
            )
        target[name] = sort


def project_contract_ir(j: Any) -> CanonicalContractIR:
    """Project all of J's contract without testing a target capability profile."""

    canonical = elaborate_j(j)
    try:
        validate_dimension_relations(canonical)
    except DimensionRelationError as exc:
        raise ContractIRProjectionError(str(exc)) from exc
    elaboration = j_elaboration(canonical)
    hints = _logical_sort_hints(canonical)
    global_sorts = dict(hints)
    domains: list[ContractDomainIR] = []
    accesses: list[ContractAccessIR] = []
    try:
        for domain_id, expression in elaboration.domains:
            program = TypedContractProgram.infer(
                expression,
                expected_sort=ContractSort.BOOLEAN,
                known_sorts=global_sorts,
                context=f"J.boundary.domains.{domain_id}",
            )
            _merge_sorts(
                global_sorts,
                program.reference_sort_map,
                context=f"J.boundary.domains.{domain_id}",
            )
            domains.append(ContractDomainIR(domain_id, program))

        structured_accesses = elaboration.access_by_id
        for authored in canonical.boundary.accesses:
            structured = structured_accesses[authored.access_id]
            index_programs: list[TypedContractProgram] = []
            for position, component in enumerate(structured.index.components):
                program = TypedContractProgram.infer(
                    component,
                    expected_sort=ContractSort.INTEGER,
                    known_sorts=global_sorts,
                    context=(
                        f"J.boundary.accesses.{authored.access_id}"
                        f".index[{position}]"
                    ),
                )
                _merge_sorts(
                    global_sorts,
                    program.reference_sort_map,
                    context=f"J.boundary.accesses.{authored.access_id}",
                )
                index_programs.append(program)
            predicate = None
            if structured.predicate is not None:
                predicate = TypedContractProgram.infer(
                    structured.predicate,
                    expected_sort=ContractSort.BOOLEAN,
                    known_sorts=global_sorts,
                    context=f"J.boundary.accesses.{authored.access_id}.predicate",
                )
                _merge_sorts(
                    global_sorts,
                    predicate.reference_sort_map,
                    context=f"J.boundary.accesses.{authored.access_id}.predicate",
                )
            accesses.append(
                ContractAccessIR(
                    access_id=authored.access_id,
                    kind=authored.kind.value,
                    tensor=authored.tensor,
                    node_ref=authored.node_ref,
                    index=tuple(index_programs),
                    predicate=predicate,
                    fill_json=_canonical_json(authored.fill_value),
                )
            )
    except ContractSemanticsError as exc:
        raise ContractIRProjectionError(str(exc)) from exc

    boundary = canonical.boundary.to_json()
    boundary.pop("domains", None)
    boundary.pop("accesses", None)
    remaining = {
        "node_ids": dict(canonical.node_ids),
        "signature": canonical.signature.to_json(),
        "boundary_relations": boundary,
        "dimension_constraints": list(canonical.dimension_constraints),
        "value_dimension_relations": [
            item.to_json() for item in canonical.value_dimension_relations
        ],
        "dimension_relations": [
            item.to_json() for item in canonical.dimension_relations
        ],
        "scopes": [item.to_json() for item in canonical.scopes],
        "storage": [item.to_json() for item in canonical.storage],
    }
    return CanonicalContractIR(
        tuple(domains),
        tuple(accesses),
        tuple(sorted(global_sorts.items())),
        _canonical_json(remaining),
    )


__all__ = [
    "CONTRACT_IR_PROTOCOL",
    "CanonicalContractIR",
    "ContractAccessIR",
    "ContractDomainIR",
    "ContractIRProjectionError",
    "project_contract_ir",
]
