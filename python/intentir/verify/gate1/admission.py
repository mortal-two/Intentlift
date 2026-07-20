"""Semantic artifact admission owned by Gate1, never extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intentir.ir.contracts import ContractError, parse_contract_predicate

from .model import Gate1Artifact


def validate_layer_b_predicates(artifact: Gate1Artifact) -> None:
    """Require retained Layer-B predicate text to be portable and boolean."""

    predicates = [
        (f"Boundary.domains.{name}", value)
        for name, value in artifact.j.boundary.domains.items()
    ]
    predicates.extend(
        (
            f"BoundaryAccess[{access.access_id}].predicate",
            access.predicate,
        )
        for access in artifact.j.boundary.accesses
        if access.predicate is not None
    )
    for path, value in predicates:
        try:
            parse_contract_predicate(value)
        except ContractError as exc:
            raise ValueError(
                f"{path} must be a portable boolean ContractExpr v1 "
                f"predicate: {exc}"
            ) from exc


def _shape_preserving_iotas(
    value: str,
    producers: Mapping[str, Any],
    *,
    seen: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    if value in seen:
        return ()
    producer = producers.get(value)
    if producer is None:
        return ()
    if producer.op == "iota":
        return (producer,)
    if producer.op not in {"cast", "map_unary", "map_binary", "select"}:
        return ()
    return tuple(
        ancestor
        for operand in producer.operands
        for ancestor in _shape_preserving_iotas(
            operand, producers, seen=seen | {value}
        )
    )


def validate_semantic_iota_extents(artifact: Gate1Artifact) -> None:
    producers = {
        op.result: op
        for op in artifact.j.intent.body
        if isinstance(op.result, str)
    }
    output_values = {
        mapping.output: mapping.value
        for mapping in artifact.j.boundary.output_mappings
    }
    for output in artifact.j.intent.outputs:
        if len(output.shape) != 1:
            continue
        extent = output.shape[0]
        if isinstance(extent, bool) or not isinstance(extent, (int, str)):
            continue
        for iota in _shape_preserving_iotas(
            output_values.get(output.name, output.name), producers
        ):
            if iota.attrs.get("n") != extent:
                raise ValueError(
                    "J.intent iota extent must equal its rank-1 semantic "
                    f"output extent: attrs.n={iota.attrs.get('n')!r}, "
                    f"output {output.name!r} shape={output.shape!r}; keep "
                    "runtime extents symbolic instead of copying the sampled "
                    "compiler-view value"
                )


def validate_opaque_lift_references(artifact: Gate1Artifact) -> None:
    """Require official LLM Lift-owned IDs to be opaque declaration handles."""

    def ordered(actual: list[str], prefix: str, label: str) -> None:
        expected = [f"{prefix}{index}" for index in range(len(actual))]
        if actual != expected:
            raise ValueError(
                f"{label} must use declaration-order opaque ids: "
                f"{actual!r} != {expected!r}"
            )

    def walk(op: Any):
        yield op
        for child in op.body:
            yield from walk(child)

    groups = (
        (list(artifact.j.node_ids), "n", "j.node_ids"),
        (list(artifact.j.boundary.domains), "d", "j.boundary.domains"),
        ([x.access_id for x in artifact.j.boundary.accesses], "a", "j.boundary.accesses.access_id"),
        ([x.effect_id for x in artifact.j.boundary.effects], "e", "j.boundary.effects.effect_id"),
        ([x.scope_id for x in artifact.j.scopes], "q", "j.scopes.scope_id"),
        ([x.storage_id for x in artifact.j.storage], "b", "j.storage.storage_id"),
        ([state.state_id for top in artifact.j.intent.body for op in walk(top) for state in op.carry], "s", "j.intent.loop.carry.state_id"),
    )
    for actual, prefix, label in groups:
        ordered(actual, prefix, label)
    all_owned = [value for actual, _, _ in groups for value in actual]
    if len(all_owned) != len(set(all_owned)):
        raise ValueError("Lift-owned opaque ids must be globally collision-free")


def requires_opaque_lift_references(artifact: Gate1Artifact) -> bool:
    return any(
        item.role.value == "lifter"
        and item.producer == "intentir.extraction.lifters.scrc"
        for item in artifact.provenance
    )


__all__ = [
    "requires_opaque_lift_references",
    "validate_layer_b_predicates",
    "validate_opaque_lift_references",
    "validate_semantic_iota_extents",
]
