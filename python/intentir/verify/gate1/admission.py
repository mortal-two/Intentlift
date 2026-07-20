"""Semantic artifact admission owned by Gate1, never extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intentir.ir.contracts import ContractError, parse_contract_predicate

from .model import Gate1Artifact


_DTYPES = {
    "bool": "i1", "float16": "f16", "half": "f16",
    "float32": "f32", "float": "f32", "float64": "f64",
    "double": "f64", "bfloat16": "bf16", "int8": "i8",
    "int16": "i16", "int32": "i32", "int64": "i64",
}


def _dtype(value: str | None) -> str | None:
    if value is None:
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPES.get(spelling, spelling)


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
    if artifact.j.intent.layer_c:
        raise ValueError(
            "J.intent.layer_c must be empty; concrete C belongs only in "
            "schedule.values"
        )
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


def validate_trusted_compiler_view(
    artifact: Gate1Artifact,
    compiler_view: Mapping[str, Any] | Any | None,
) -> None:
    """Compare J/interface/C with trusted capture inside Gate1."""

    if compiler_view is None:
        return
    if hasattr(compiler_view, "to_json"):
        compiler_view = compiler_view.to_json()
    if not isinstance(compiler_view, Mapping):
        raise TypeError("trusted compiler view must be a mapping")
    view = dict(compiler_view)
    parameters = artifact.interface.parameters
    order = list(view.get("parameter_order", ()))
    if order and [item.name for item in parameters] != order:
        raise ValueError("interface parameter order differs from trusted compiler view")
    constexprs = list(view.get("constexpr_parameters", ()))
    actual_constexprs = [
        item.name for item in parameters if item.kind.value == "constexpr"
    ]
    if constexprs and actual_constexprs != constexprs:
        raise ValueError("interface constexpr parameters differ from trusted compiler view")
    arguments = list(view.get("arguments", ()))
    runtime = [item for item in parameters if item.kind.value != "constexpr"]
    if arguments and [item.name for item in runtime] != [x["name"] for x in arguments]:
        raise ValueError("interface runtime parameters differ from trusted compiler view")
    tensors = {
        item.name: item
        for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    scalar_extents = {
        item["name"]: item.get("value")
        for item in arguments
        if item.get("kind") == "scalar"
        and isinstance(item.get("value"), int)
        and not isinstance(item.get("value"), bool)
    }
    parameter_arguments = (
        zip(runtime, arguments, strict=True) if arguments else ()
    )
    for parameter, argument in parameter_arguments:
        pointer = parameter.kind.value in {"pointer", "tensor"}
        if pointer != (argument.get("kind") == "tensor"):
            raise ValueError(
                f"interface parameter {parameter.name!r} kind differs from trusted compiler view"
            )
        if _dtype(parameter.dtype) != _dtype(argument.get("dtype")):
            raise ValueError(
                f"interface parameter {parameter.name!r} dtype differs from trusted compiler view"
            )
        if argument.get("kind") != "tensor":
            continue
        shape = list(argument.get("shape", ()))
        if parameter.rank != len(shape):
            raise ValueError(
                f"interface parameter {parameter.name!r} rank differs from trusted compiler view"
            )
        tensor = tensors.get(parameter.tensor_ref or "")
        if tensor is None or len(tensor.shape) != len(shape):
            raise ValueError(
                f"interface parameter {parameter.name!r} tensor_ref has the wrong J rank"
            )
        if _dtype(tensor.dtype) != _dtype(argument.get("dtype")):
            raise ValueError(
                f"interface parameter {parameter.name!r} J tensor dtype differs from trusted compiler view"
            )
        for declared, concrete in zip(tensor.shape, shape, strict=True):
            if isinstance(declared, str):
                if scalar_extents.get(declared) != concrete:
                    raise ValueError(
                        f"J tensor {tensor.name!r} shape symbol {declared!r} "
                        f"does not bind to compiler extent {concrete}"
                    )
            elif int(declared) != concrete:
                raise ValueError(
                    f"J tensor {tensor.name!r} static extent {declared} != {concrete}"
                )
    grid = list(view.get("grid", ()))
    if grid and len(artifact.interface.grid) != len(grid):
        raise ValueError("interface.grid rank differs from trusted compiler view")
    grid_contract = list(view.get("grid_contract", ()))
    if grid_contract and list(artifact.interface.grid) != grid_contract:
        raise ValueError("interface.grid differs from trusted grid contract")
    if dict(artifact.schedule.values) != dict(view.get("schedule_values", {})):
        raise ValueError("schedule.values differs from trusted compiler view")
    if tuple(artifact.interface.compiler_slots) != tuple(
        view.get("compiler_slots", ())
    ):
        raise ValueError("interface.compiler_slots differs from trusted compiler view")
    if [x.to_json() for x in artifact.j.boundary.aliases] != list(
        view.get("aliases", ())
    ):
        raise ValueError("boundary.aliases differs from trusted compiler view")


__all__ = [
    "requires_opaque_lift_references",
    "validate_layer_b_predicates",
    "validate_opaque_lift_references",
    "validate_semantic_iota_extents",
    "validate_trusted_compiler_view",
]
