"""One-way Gate1 artifact -> verification-neutral IntentProgram hand-off.

The adapter intentionally lives under ``verify.gate1`` so the dependency arrow
is Gate -> IR, never IR/backend -> Gate.  Privileged source, provenance,
location maps, replay plans, verdicts and archive identifiers are discarded.
Any legacy A/B field that cannot be converted losslessly causes hand-off to
fail before lowering.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from intentir.ir.abi import LogicalArgument, LogicalInterface
from intentir.ir.contracts import (
    AccessContract,
    AccessMode,
    AliasContract,
    AliasRelation,
    BoundaryContract,
    BroadcastProjection,
    ConflictKind,
    DestinationBinding,
    DomainContract,
    EffectContract,
    EffectMode,
    EmptyDomainPolicy,
    IndexRole,
    NanPolicy,
    ReassociationPolicy,
    ReductionContract,
    ScopeContract,
    StorageContract,
    WriteCoverage,
    parse_contract_expression,
)
from intentir.ir.model import IntentFunction
from intentir.ir.program import IntentProgram, ScheduleHint, StrategyHint

from .model import (
    AccessKind,
    AliasKind,
    EffectKind,
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)


class GateProgramHandoffError(ValueError):
    """A closed Gate1 artifact cannot be represented by IntentProgram v1."""


_ACCESS_MODE = {
    AccessKind.LOAD: AccessMode.READ,
    AccessKind.STORE: AccessMode.WRITE,
    AccessKind.ATOMIC: AccessMode.ATOMIC,
    AccessKind.SYNCHRONIZE: AccessMode.SYNCHRONIZE,
}
_ALIAS_RELATION = {
    AliasKind.MUST_ALIAS: AliasRelation.MUST_ALIAS,
    AliasKind.MAY_ALIAS: AliasRelation.MAY_ALIAS,
    AliasKind.NO_ALIAS: AliasRelation.NO_ALIAS,
}
_EFFECT_MODE = {
    EffectKind.READ: EffectMode.READ,
    EffectKind.WRITE: EffectMode.WRITE,
    EffectKind.READ_WRITE: EffectMode.READ_WRITE,
    EffectKind.ATOMIC: EffectMode.ATOMIC,
    EffectKind.SYNCHRONIZE: EffectMode.SYNCHRONIZE,
}
_ROLE = {
    ParameterMode.INPUT: "input",
    ParameterMode.OUTPUT: "output",
    ParameterMode.INOUT: "inout",
    ParameterMode.VALUE: "parameter",
}


def _walk_json_ops(body: list[dict[str, Any]], prefix: str = "body"):
    for index, op in enumerate(body):
        path = f"{prefix}[{index}]"
        yield path, op
        nested = op.get("body", [])
        if isinstance(nested, list):
            yield from _walk_json_ops(nested, f"{path}.body")


def _node_ref(value: str | None, node_by_path: dict[str, str]) -> str | None:
    if value is None:
        return None
    return node_by_path.get(value, value)


def _normalize_function(artifact: Gate1Artifact) -> tuple[IntentFunction, dict[str, dict[str, Any]]]:
    payload = copy.deepcopy(artifact.j.intent.to_json()["intent_function"])
    if payload.get("layer_c"):
        raise GateProgramHandoffError("Gate1 J contains concrete legacy layer_c")
    payload.pop("layer_c", None)
    node_by_path = {path: node_id for node_id, path in artifact.j.node_ids.items()}
    legacy_by_node: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for path, op in _walk_json_ops(payload.get("body", [])):
        seen_paths.add(path)
        node_id = node_by_path.get(path)
        if node_id is None:
            raise GateProgramHandoffError(f"Layer-A op {path} has no stable Gate1 node id")
        op["id"] = node_id
        legacy = op.pop("layer_b", {})
        if not isinstance(legacy, dict):
            raise GateProgramHandoffError(f"op {node_id!r} legacy layer_b is not an object")
        legacy = dict(legacy)
        attrs = op.setdefault("attrs", {})
        for old_name, canonical_name in (
            ("accumulator_dtype", "accumulator_dtype"),
            ("accumulation_dtype", "accumulator_dtype"),
            ("result_dtype", "result_dtype"),
        ):
            if old_name not in legacy:
                continue
            value = legacy.pop(old_name)
            existing = attrs.get(canonical_name)
            if existing not in (None, value):
                raise GateProgramHandoffError(
                    f"op {node_id!r} has conflicting {canonical_name} declarations"
                )
            attrs[canonical_name] = value
        legacy_out = op.pop("out", None)
        if legacy_out is not None:
            legacy["destination_out"] = legacy_out
        if legacy:
            legacy_by_node[node_id] = legacy
    missing_paths = sorted(set(node_by_path) - seen_paths)
    if missing_paths:
        raise GateProgramHandoffError(f"Gate1 node map contains stale paths: {missing_paths}")

    visible = {
        item["name"] for item in payload.get("inputs", []) if isinstance(item, dict)
    }
    for _path, op in _walk_json_ops(payload.get("body", [])):
        result = op.get("result")
        if isinstance(result, str):
            visible.add(result)
        elif isinstance(result, list):
            visible.update(item for item in result if isinstance(item, str))
    returns = [item.to_json() for item in artifact.j.boundary.output_mappings]
    for item in returns:
        item.pop("node_ref", None)
    if not returns:
        for output in payload.get("outputs", []):
            name = output.get("name")
            if name not in visible:
                raise GateProgramHandoffError(
                    f"logical output {name!r} has no explicit output mapping"
                )
            returns.append({"output": name, "value": name})
    payload["returns"] = returns
    return IntentFunction.from_json(payload), legacy_by_node


def _value_ranks(function: IntentFunction) -> dict[str, int]:
    ranks = {item.name: len(item.shape) for item in [*function.inputs, *function.outputs]}
    for op in function.body:
        operands = [ranks[name] for name in op.operands if name in ranks]
        declared_shape = op.type.get("shape") if isinstance(op.type, dict) else None
        if declared_shape is None and op.op == "const":
            declared_shape = op.attrs.get("shape")
        if isinstance(declared_shape, list):
            rank = len(declared_shape)
        elif op.op == "reduce" and operands:
            axes = op.attrs.get("axes", [])
            rank = max(0, operands[0] - len(axes))
            if op.attrs.get("keepdims"):
                rank = operands[0]
        elif op.op == "transpose" and isinstance(op.attrs.get("perm"), list):
            rank = len(op.attrs["perm"])
        elif operands:
            rank = max(operands)
        else:
            continue
        results = [op.result] if isinstance(op.result, str) else op.result
        if isinstance(results, list):
            for result in results:
                if isinstance(result, str):
                    ranks[result] = rank
    return ranks


def _artifact_from_payload(payload: dict[str, Any]) -> Gate1Artifact:
    """Parse a raw artifact or the explicitly redacted live-report form.

    Live reports replace privileged source text with a digest and byte count.
    Those two evidence fields are not part of Gate1Artifact and must not cross
    the neutral hand-off.  Recognize only that exact pair, validate it, then
    retain Gate1Artifact's strict rejection of every other unknown field.
    """

    data = copy.deepcopy(payload)
    has_digest = "source_sha256" in data
    has_bytes = "source_bytes" in data
    if has_digest != has_bytes:
        raise GateProgramHandoffError(
            "redacted Gate1 artifact requires source_sha256 and source_bytes together"
        )
    if has_digest:
        digest = data.pop("source_sha256")
        byte_count = data.pop("source_bytes")
        if "source" in data:
            raise GateProgramHandoffError(
                "Gate1 artifact cannot contain source and redacted source evidence together"
            )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise GateProgramHandoffError("redacted Gate1 source digest is malformed")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise GateProgramHandoffError("redacted Gate1 source byte count is malformed")
    return Gate1Artifact.from_json(data)


def _value_dtypes(function: IntentFunction) -> dict[str, str]:
    dtypes = {item.name: item.dtype for item in [*function.inputs, *function.outputs]}
    for op in function.body:
        declared = op.type.get("dtype") if isinstance(op.type, dict) else None
        if not isinstance(declared, str):
            declared = op.attrs.get("result_dtype")
        if not isinstance(declared, str) and op.op == "cast":
            declared = op.attrs.get("to_dtype", op.attrs.get("dtype"))
        if not isinstance(declared, str) and op.operands:
            declared = dtypes.get(op.operands[0])
        results = [op.result] if isinstance(op.result, str) else op.result
        if isinstance(declared, str) and isinstance(results, list):
            for result in results:
                if isinstance(result, str):
                    dtypes[result] = declared
    return dtypes


def _logical_interface(artifact: Gate1Artifact, function: IntentFunction) -> LogicalInterface:
    tensors = {item.name: item for item in [*function.inputs, *function.outputs]}
    arguments: list[LogicalArgument] = []
    for parameter in artifact.interface.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            continue
        tensor = tensors.get(parameter.tensor_ref) if parameter.tensor_ref is not None else None
        dtype = parameter.dtype or (tensor.dtype if tensor is not None else None)
        if dtype is None:
            raise GateProgramHandoffError(
                f"runtime ABI parameter {parameter.name!r} has no dtype"
            )
        if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            rank = parameter.rank if parameter.rank is not None else (
                len(tensor.shape) if tensor is not None else None
            )
            if rank is None:
                raise GateProgramHandoffError(
                    f"tensor ABI parameter {parameter.name!r} has no rank"
                )
            shape = tuple(str(item) for item in tensor.shape) if tensor is not None else tuple(
                f"{parameter.name}.dim{index}" for index in range(rank)
            )
            arguments.append(
                LogicalArgument(
                    parameter.name, "tensor", _ROLE[parameter.mode], dtype,
                    rank=rank, symbolic_shape=shape, tensor_ref=parameter.tensor_ref,
                )
            )
        elif parameter.kind is ParameterKind.SCALAR:
            arguments.append(
                LogicalArgument(
                    parameter.name, "scalar", _ROLE[parameter.mode], dtype,
                    tensor_ref=parameter.tensor_ref,
                )
            )
        else:
            raise GateProgramHandoffError(
                f"unsupported runtime ABI parameter kind {parameter.kind.value!r}"
            )
    return LogicalInterface(
        function=function.name,
        arguments=tuple(arguments),
        outputs=tuple(item.name for item in function.outputs),
    )


def _schedule_hint(artifact: Gate1Artifact) -> ScheduleHint:
    strategies: list[StrategyHint] = []
    for binding in artifact.strategy.bindings:
        parameters: list[tuple[str, Any]] = []
        if binding.tile_slot is not None:
            parameters.append(("tile_slot", binding.tile_slot))
        if binding.accumulator_dtype is not None:
            parameters.append(("accumulator_dtype", binding.accumulator_dtype))
        strategies.append(
            StrategyHint(
                binding.family.value,
                binding.target_ref,
                binding.choice.value,
                tuple(parameters),
            )
        )
    return ScheduleHint(
        values=tuple(artifact.schedule.values.items()),
        grid=tuple(artifact.interface.grid),
        strategies=tuple(strategies),
    )


def intent_program_from_gate1(artifact: Gate1Artifact | dict[str, Any]) -> IntentProgram:
    """Build the exact target-neutral lowering contract or fail closed."""

    if not isinstance(artifact, Gate1Artifact):
        artifact = _artifact_from_payload(artifact)
    function, legacy_by_node = _normalize_function(artifact)
    node_by_path = {path: node_id for node_id, path in artifact.j.node_ids.items()}
    nodes = {op.node_id: op for op in function.body if op.node_id is not None}
    for op in function.body:
        for child_path, child in _walk_json_ops([op.to_json()]):
            del child_path
            if child.get("id"):
                nodes[child["id"]] = child
    ranks = _value_ranks(function)
    dtypes = _value_dtypes(function)

    domains = tuple(
        DomainContract(name, name, parse_contract_expression(predicate))
        for name, predicate in artifact.j.boundary.domains.items()
    )
    accesses: list[AccessContract] = []
    access_by_id = {item.access_id: item for item in artifact.j.boundary.accesses}
    for item in artifact.j.boundary.accesses:
        node_ref = _node_ref(item.node_ref, node_by_path)
        op_value = nodes.get(node_ref)
        op_name = op_value.op if hasattr(op_value, "op") else op_value.get("op") if isinstance(op_value, dict) else None
        logical = op_name in {"gather", "scatter", "masked_scatter", "masked_select"}
        accesses.append(
            AccessContract(
                contract_id=item.access_id,
                mode=_ACCESS_MODE[item.kind],
                tensor=item.tensor,
                node_ref=node_ref,
                predicate=(parse_contract_expression(item.predicate) if item.predicate is not None else None),
                fill_value=item.fill_value,
                index_role=IndexRole.LOGICAL_NODE if logical else IndexRole.REALIZATION_ONLY,
                logical_index_node=node_ref if logical else None,
            )
        )

    broadcasts: list[BroadcastProjection] = []
    for index, item in enumerate(artifact.j.boundary.broadcasts):
        node_ref = _node_ref(item.node_ref, node_by_path)
        source_rank = ranks.get(item.operand)
        op_value = nodes.get(node_ref)
        result = op_value.result if hasattr(op_value, "result") else op_value.get("result") if isinstance(op_value, dict) else None
        result_name = result if isinstance(result, str) else result[0] if isinstance(result, list) and result else None
        consumer_rank = ranks.get(result_name) if result_name is not None else None
        if source_rank is None or consumer_rank is None:
            raise GateProgramHandoffError(
                f"broadcast {index} lacks exact producer/consumer semantic ranks"
            )
        broadcasts.append(
            BroadcastProjection(
                f"broadcast:{index}", node_ref, item.operand,
                source_rank, consumer_rank, tuple(item.axes),
            )
        )

    aliases = [
        AliasContract(f"alias:{index}", item.left, item.right, _ALIAS_RELATION[item.kind])
        for index, item in enumerate(artifact.j.boundary.aliases)
    ]
    effects = [
        EffectContract(
            item.effect_id or f"effect:{index}", _EFFECT_MODE[item.kind], item.target,
            _node_ref(item.node_ref, node_by_path), item.access_ref,
        )
        for index, item in enumerate(artifact.j.boundary.effects)
    ]

    output_values = {item.output: item.value for item in function.returns}
    interface = _logical_interface(artifact, function)
    destinations: list[DestinationBinding] = []
    for parameter in artifact.interface.parameters:
        if parameter.mode not in {ParameterMode.OUTPUT, ParameterMode.INOUT}:
            continue
        output = parameter.tensor_ref
        if output is None or output not in output_values:
            continue
        matching_accesses = [
            item for item in artifact.j.boundary.accesses
            if item.tensor == output and item.kind in {AccessKind.STORE, AccessKind.ATOMIC}
        ]
        if len(matching_accesses) != 1:
            raise GateProgramHandoffError(
                f"destination {parameter.name!r} requires exactly one write access"
            )
        access = matching_accesses[0]
        predicate = parse_contract_expression(access.predicate) if access.predicate is not None else None
        coverage = WriteCoverage.PREDICATED if predicate is not None else WriteCoverage.FULL
        conflict = ConflictKind.ATOMIC if access.kind is AccessKind.ATOMIC else ConflictKind.UNKNOWN
        destinations.append(
            DestinationBinding(
                f"destination:{len(destinations)}", output, parameter.name,
                output_values[output], coverage, conflict, access.access_id, predicate,
            )
        )
        if parameter.alias_of is not None and not any(
            {item.left, item.right} == {parameter.name, parameter.alias_of}
            and item.relation is AliasRelation.MUST_ALIAS
            for item in aliases
        ):
            aliases.append(
                AliasContract(
                    f"interface-alias:{len(aliases)}", parameter.name,
                    parameter.alias_of, AliasRelation.MUST_ALIAS,
                )
            )

    reductions: list[ReductionContract] = []
    for index, item in enumerate(artifact.j.boundary.reductions):
        node_ref = _node_ref(item.node_ref, node_by_path)
        op_value = nodes.get(node_ref)
        attrs = op_value.attrs if hasattr(op_value, "attrs") else op_value.get("attrs", {}) if isinstance(op_value, dict) else {}
        operands = op_value.operands if hasattr(op_value, "operands") else op_value.get("operands", []) if isinstance(op_value, dict) else []
        result = op_value.result if hasattr(op_value, "result") else op_value.get("result") if isinstance(op_value, dict) else None
        if not isinstance(attrs, dict) or attrs.get("fn") is None or not isinstance(attrs.get("axes"), list):
            raise GateProgramHandoffError(f"reduction {node_ref!r} has no typed Layer-A semantics")
        operand_dtype = dtypes.get(operands[0]) if operands else None
        result_name = result if isinstance(result, str) else result[0] if isinstance(result, list) and result else None
        result_dtype = attrs.get("result_dtype") or dtypes.get(result_name)
        accumulator_dtype = attrs.get("accumulator_dtype") or operand_dtype
        for binding in artifact.strategy.bindings:
            if binding.target_ref == node_ref and binding.accumulator_dtype is not None:
                accumulator_dtype = binding.accumulator_dtype
        if not isinstance(accumulator_dtype, str) or not isinstance(result_dtype, str):
            raise GateProgramHandoffError(f"reduction {node_ref!r} lacks accumulator/result dtype")
        fn = attrs["fn"]
        nan_policy = NanPolicy.PREFER_NUMBER if fn in {"max", "min"} else NanPolicy.PROPAGATE
        assoc = attrs.get("assoc")
        reassociation = (
            ReassociationPolicy.ALLOWED
            if assoc in {"assoc", "associative", "reassociate", True}
            else ReassociationPolicy.UNSPECIFIED
        )
        reductions.append(
            ReductionContract(
                f"reduction:{index}", node_ref, tuple(attrs["axes"]),
                accumulator_dtype, result_dtype, item.identity,
                item.masked_contribution, nan_policy,
                EmptyDomainPolicy.IDENTITY if item.identity is not None else EmptyDomainPolicy.UNSPECIFIED,
                reassociation,
            )
        )

    scopes = tuple(
        ScopeContract(
            item.scope_id, item.kind.value, item.parent_scope,
            tuple(item.captures), tuple(item.definitions), tuple(item.effects),
        )
        for item in artifact.j.scopes
    )
    storage = tuple(
        StorageContract(
            item.storage_id, tuple(item.members), item.scope_id,
            item.mutability.value,
            f"{item.init_event.kind.value}:{item.init_event.ref}",
            f"{item.last_use_event.kind.value}:{item.last_use_event.ref}",
            item.escapes, item.crosses_iteration,
        )
        for item in artifact.j.storage
    )

    permitted_legacy = {
        "destination_out", "mutation", "aliasing", "conflict_contract",
        "input_relationships", "index_bounds_proof", "index_in_bounds",
    }
    for node_ref, legacy in legacy_by_node.items():
        unknown = sorted(set(legacy) - permitted_legacy)
        if unknown:
            raise GateProgramHandoffError(
                f"op {node_ref!r} has unconvertible legacy Layer-B fields: {unknown}"
            )
        # These records cannot be silently asserted.  Until Gate1 emits their
        # typed outer equivalents, hand-off refuses a non-empty legacy payload.
        if legacy:
            raise GateProgramHandoffError(
                f"op {node_ref!r} still owns Layer-B facts outside typed Boundary: "
                + ", ".join(sorted(legacy))
            )

    boundary = BoundaryContract(
        domains=domains,
        accesses=tuple(accesses),
        broadcasts=tuple(broadcasts),
        destinations=tuple(destinations),
        aliases=tuple(aliases),
        effects=tuple(effects),
        reductions=tuple(reductions),
        scopes=scopes,
        storage=storage,
    )
    return IntentProgram(function, boundary, interface, _schedule_hint(artifact))


__all__ = ["GateProgramHandoffError", "intent_program_from_gate1"]
