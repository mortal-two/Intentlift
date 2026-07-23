"""Dependent, fail-closed TypeCheck for canonical ``(J, C, pi)`` artifacts.

The checker enforces explicit ownership, reference and representation
authorization rules.  It intentionally does not prove Source/J equivalence,
run an SMT solver, or decide whether a generated Replay behaves correctly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping

from intentir.verify.gate1.model import (
    ImplementationConfig,
    RefinementKind,
    ScheduleSlotKind,
    SemanticArchive,
)
from intentir.verify.gate1.pi import RealizationWitness


TYPECHECK_PROTOCOL = "intentir.scrc.typecheck.v1"
GLOBAL_SAFE_REFINEMENTS = frozenset({"identity_materialization"})


class TypeCheckStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"


class TypeCheckErrorCode(str, Enum):
    DUPLICATE_J_REF = "duplicate_j_ref"
    INVALID_C_SLOT = "invalid_c_slot"
    C_REQUIRED_VALUE_MISSING = "c_required_value_missing"
    C_VALUE_TYPE_MISMATCH = "c_value_type_mismatch"
    INVALID_ACCESS_FILL = "invalid_access_fill"
    UNRESOLVED_ANCHOR = "unresolved_anchor"
    ROUTE_CYCLE = "route_cycle"
    ROUTE_SLOT_OUT_OF_RANGE = "route_slot_out_of_range"
    ROUTE_TYPE_MISMATCH = "route_type_mismatch"
    INVALID_SHARING = "invalid_sharing"
    OBSERVABLE_REQUIRES_J_ANCHOR = "observable_requires_j_anchor"
    AUTHORIZATION_SUBJECT_UNKNOWN = "authorization_subject_unknown"
    REFINEMENT_AUTH_MISSING = "refinement_auth_missing"
    REFINEMENT_AUTH_UNKNOWN = "refinement_auth_unknown"
    REFINEMENT_AUTH_KIND_MISMATCH = "refinement_auth_kind_mismatch"
    REFINEMENT_AUTH_SUBJECT_MISMATCH = "refinement_auth_subject_mismatch"
    REFINEMENT_AUTH_TYPE_MISMATCH = "refinement_auth_type_mismatch"
    GLOBAL_SAFE_AUTH_FORBIDDEN = "global_safe_auth_forbidden"


@dataclass(frozen=True)
class TypeCheckDiagnostic:
    code: TypeCheckErrorCode
    path: str
    detail: str
    ref: str | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code.value,
            "path": self.path,
            "detail": self.detail,
        }
        if self.ref is not None:
            result["ref"] = self.ref
        return result


@dataclass(frozen=True)
class TypeCheckReport:
    status: TypeCheckStatus
    diagnostics: tuple[TypeCheckDiagnostic, ...]
    protocol: str = TYPECHECK_PROTOCOL

    @property
    def passed(self) -> bool:
        return self.status is TypeCheckStatus.PASS

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "status": self.status.value,
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _ValueType:
    dtype: str
    rank: int


@dataclass(frozen=True)
class _Anchor:
    owner: str
    kind: str
    value_type: _ValueType | None = None
    operand_types: tuple[_ValueType | None, ...] = ()


def _operation_results(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _walk_operations(
    operations: Iterable[Mapping[str, Any]], prefix: str = "body"
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for index, operation in enumerate(operations):
        path = f"{prefix}[{index}]"
        yield path, operation
        body = operation.get("body")
        if isinstance(body, list):
            yield from _walk_operations(body, f"{path}.body")


def _explicit_type(operation: Mapping[str, Any]) -> _ValueType | None:
    value = operation.get("type")
    if not isinstance(value, Mapping):
        return None
    dtype = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        return None
    return _ValueType(dtype, len(shape))


def _infer_result_type(
    operation: Mapping[str, Any], value_types: Mapping[str, _ValueType]
) -> _ValueType | None:
    explicit = _explicit_type(operation)
    if explicit is not None:
        return explicit
    operands = operation.get("operands")
    operand_refs = operands if isinstance(operands, list) else []
    first = next(
        (
            value_types[item]
            for item in operand_refs
            if isinstance(item, str) and item in value_types
        ),
        None,
    )
    if first is None:
        return None
    op = operation.get("op")
    attrs = operation.get("attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    if op == "cast" and isinstance(attrs.get("to_dtype"), str):
        return _ValueType(str(attrs["to_dtype"]), first.rank)
    if op == "reduce":
        axes = attrs.get("axes")
        if isinstance(axes, list):
            return _ValueType(first.dtype, max(0, first.rank - len(set(axes))))
    return first


def _c_value_matches(value_type: str, value: Any) -> bool:
    normalized = value_type.casefold()
    if normalized in {"bool", "i1"}:
        return isinstance(value, bool)
    if normalized in {"int", "integer"} or normalized.startswith(("i", "u")):
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"float", "number"} or normalized.startswith("f"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized in {"str", "string", "layout", "enum"}:
        # C may select one closed implementation token, but it must not become
        # a source-code or free-text side channel.  Rich layout expressions
        # need their own closed schema rather than being smuggled through a
        # string-valued slot.
        return isinstance(value, str) and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.:+/-]*", value
        ) is not None
    return False


def _has_cycle(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    ready = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(nodes)


def _build_anchors(
    j: SemanticArchive,
    c: ImplementationConfig,
    diagnostics: list[TypeCheckDiagnostic],
) -> tuple[dict[str, _Anchor], dict[str, Mapping[str, Any]]]:
    anchors: dict[str, _Anchor] = {}

    def add(ref: str, anchor: _Anchor, path: str) -> None:
        previous = anchors.get(ref)
        if previous is not None and previous != anchor:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.DUPLICATE_J_REF,
                    path,
                    f"handle {ref!r} has more than one J declaration",
                    ref,
                )
            )
            return
        anchors[ref] = anchor

    value_types: dict[str, _ValueType] = {}
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        value_type = _ValueType(tensor.dtype, len(tensor.shape))
        value_types[tensor.name] = value_type
        add(tensor.name, _Anchor("j", "tensor", value_type), "j.signature")
        for dim in tensor.shape:
            if isinstance(dim, str):
                add(dim, _Anchor("j", "dimension", _ValueType("index", 0)), "j.signature")

    for parameter in j.signature.parameters:
        add(
            parameter.name,
            _Anchor(
                "j",
                "signature",
                (
                    _ValueType(parameter.dtype, parameter.rank or 0)
                    if parameter.dtype is not None
                    else None
                ),
            ),
            f"j.signature.parameters[{parameter.name}]",
        )

    intent = j.intent.to_json()["intent_function"]
    operations = dict(_walk_operations(intent.get("body", [])))
    changed = True
    while changed:
        changed = False
        for operation in operations.values():
            inferred = _infer_result_type(operation, value_types)
            if inferred is None:
                continue
            for result in _operation_results(operation.get("result")):
                if result not in value_types:
                    value_types[result] = inferred
                    changed = True

    node_operations: dict[str, Mapping[str, Any]] = {}
    for node_id, path in j.node_ids.items():
        operation = operations.get(path)
        if operation is None:
            continue
        node_operations[node_id] = operation
        operands = operation.get("operands")
        operand_refs = operands if isinstance(operands, list) else []
        operand_types = tuple(
            value_types.get(item) if isinstance(item, str) else None
            for item in operand_refs
        )
        result_type = next(
            (
                value_types[result]
                for result in _operation_results(operation.get("result"))
                if result in value_types
            ),
            None,
        )
        anchor = _Anchor("j", "node", result_type, operand_types)
        add(node_id, anchor, f"j.node_ids.{node_id}")
        add(path, anchor, f"j.node_ids.{node_id}")
    for value, value_type in value_types.items():
        if value not in anchors:
            add(value, _Anchor("j", "value", value_type), "j.intent")

    for item in j.boundary.domains:
        add(item, _Anchor("j", "domain"), f"j.boundary.domains.{item}")
    tensor_types = {
        item.name: _ValueType(item.dtype, len(item.shape))
        for item in [*j.intent.inputs, *j.intent.outputs]
    }
    for item in j.boundary.accesses:
        value_type = tensor_types.get(item.tensor)
        arity = (value_type,) if item.kind.value in {"store", "update"} else ()
        add(
            item.access_id,
            _Anchor("j", "access", value_type, arity),
            f"j.boundary.accesses.{item.access_id}",
        )
    for item in j.boundary.effects:
        assert item.effect_id is not None
        add(
            item.effect_id,
            _Anchor("j", "effect"),
            f"j.boundary.effects.{item.effect_id}",
        )
    for item in j.scopes:
        add(item.scope_id, _Anchor("j", "scope"), f"j.scopes.{item.scope_id}")
    for item in j.storage:
        add(
            item.storage_id,
            _Anchor("j", "storage"),
            f"j.storage.{item.storage_id}",
        )
    for item in j.boundary.realization_authorizations:
        add(
            item.auth_id,
            _Anchor("j", "authorization"),
            f"j.boundary.realization_authorizations.{item.auth_id}",
        )

    for index, slot in enumerate(c.slots):
        add(slot.name, _Anchor("c", "slot"), f"c.slots[{index}]")
    return anchors, node_operations


def type_check(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
) -> TypeCheckReport:
    """Mechanically check ``Gamma_(J,C) |- pi : RealizationWitness(J,C)``."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("type_check requires SemanticArchive J")
    if not isinstance(c, ImplementationConfig):
        raise TypeError("type_check requires canonical ImplementationConfig C")
    if not isinstance(pi, RealizationWitness):
        raise TypeError("type_check requires parsed RealizationWitness pi")

    diagnostics: list[TypeCheckDiagnostic] = []
    for index, access in enumerate(j.boundary.accesses):
        if access.fill_value is None:
            continue
        path = f"j.boundary.accesses[{index}].fill_value"
        if access.kind.value != "load":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_ACCESS_FILL,
                    path,
                    "fill_value is legal only on a load access",
                    access.access_id,
                )
            )
        if access.predicate is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_ACCESS_FILL,
                    path,
                    "fill_value requires a non-null access predicate",
                    access.access_id,
                )
            )

    expected_c = [f"c{index}" for index in range(len(c.slots))]
    if [item.name for item in c.slots] != expected_c:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.INVALID_C_SLOT,
                "c.slots",
                "C slot ids must be declaration-order handles c0, c1, ...",
            )
        )
    for index, slot in enumerate(c.slots):
        value = c.values.get(slot.name)
        if slot.required and slot.name not in c.values:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_REQUIRED_VALUE_MISSING,
                    f"c.values.{slot.name}",
                    "required C slot has no Harness valuation",
                    slot.name,
                )
            )
        if value is not None and not _c_value_matches(slot.value_type, value):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH,
                    f"c.values.{slot.name}",
                    f"value does not match closed C type {slot.value_type!r}",
                    slot.name,
                )
            )
        if slot.kind not in {ScheduleSlotKind.CONSTEXPR, ScheduleSlotKind.COMPILER}:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_C_SLOT,
                    f"c.slots[{index}]",
                    "unsupported C binding kind",
                    slot.name,
                )
            )

    anchors, _operations = _build_anchors(j, c, diagnostics)
    instances = {str(item["id"]): item for item in pi.instances}
    instance_anchors: dict[str, _Anchor] = {}
    for index, item in enumerate(pi.instances):
        ref = str(item["semantic_ref"])
        anchor = anchors.get(ref)
        if anchor is None or anchor.kind == "authorization":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.UNRESOLVED_ANCHOR,
                    f"pi.instances[{index}].semantic_ref",
                    f"pi anchor {ref!r} is not declared by J or C",
                    ref,
                )
            )
            continue
        instance_anchors[str(item["id"])] = anchor

    route_edges = {
        (str(item["source"]), str(item["target"])) for item in pi.routes
    }
    order_edges = {
        (str(item["before"]), str(item["after"])) for item in pi.order
    }
    if _has_cycle(set(instances), route_edges | order_edges):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.ROUTE_CYCLE,
                "pi.routes/order",
                "combined realization dependency graph must be acyclic",
            )
        )

    for route_index, route in enumerate(pi.routes):
        source_id = str(route["source"])
        target_id = str(route["target"])
        source = instance_anchors.get(source_id)
        target = instance_anchors.get(target_id)
        if source is None or target is None:
            continue
        # Operand ordinals are mechanically meaningful for J carrier nodes.
        # Access/address and placement instances may also expose frontend
        # representation inputs, whose arity is intentionally not invented by
        # this thin checker.
        if target.kind != "node":
            continue
        for slot in route["slots"]:
            if slot >= len(target.operand_types):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.ROUTE_SLOT_OUT_OF_RANGE,
                        f"pi.routes[{route_index}].slots",
                        f"slot {slot} exceeds target carrier arity "
                        f"{len(target.operand_types)}",
                        target_id,
                    )
                )
                continue
            expected = (
                target.operand_types[slot]
                if target.operand_types and slot < len(target.operand_types)
                else None
            )
            if (
                source.value_type is not None
                and expected is not None
                and source.value_type != expected
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.ROUTE_TYPE_MISMATCH,
                        f"pi.routes[{route_index}].slots[{slot}]",
                        f"source carrier {source.value_type} is incompatible "
                        f"with target operand {expected}",
                        target_id,
                    )
                )

    for index, sharing in enumerate(pi.sharing):
        source = str(sharing["source"])
        consumers = [str(item) for item in sharing["consumers"]]
        if source in consumers or any(
            (source, consumer) not in route_edges for consumer in consumers
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_SHARING,
                    f"pi.sharing[{index}]",
                    "sharing consumers must have a direct route from the shared source",
                    source,
                )
            )

    observable_mechanisms = {
        "transfer_direct",
        "transfer_async",
        "update_atomic",
        "update_privatized",
        "synchronize",
    }
    for index, item in enumerate(pi.instances):
        if item["mechanism"] not in observable_mechanisms:
            continue
        anchor = instance_anchors.get(str(item["id"]))
        if anchor is not None and not (
            anchor.owner == "j" and anchor.kind in {"node", "access", "effect"}
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.OBSERVABLE_REQUIRES_J_ANCHOR,
                    f"pi.instances[{index}].semantic_ref",
                    "observable transfer/update/effect realization must anchor "
                    "a J-owned node, access, or effect",
                    str(item["semantic_ref"]),
                )
            )

    authorizations = {
        item.auth_id: item for item in j.boundary.realization_authorizations
    }
    valid_subjects = {
        ref
        for ref, anchor in anchors.items()
        if anchor.owner == "j" and anchor.kind != "authorization"
    }
    for index, authorization in enumerate(
        j.boundary.realization_authorizations
    ):
        subject = anchors.get(authorization.subject_ref)
        if authorization.subject_ref not in valid_subjects:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.AUTHORIZATION_SUBJECT_UNKNOWN,
                    f"j.boundary.realization_authorizations[{index}].subject_ref",
                    "authorization subject is not a J carrier",
                    authorization.subject_ref,
                )
            )
        elif (
            authorization.source_dtype is not None
            and subject is not None
            and subject.value_type is not None
            and authorization.source_dtype != subject.value_type.dtype
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                    f"j.boundary.realization_authorizations[{index}].source_dtype",
                    "authorization source dtype differs from its J carrier",
                    authorization.auth_id,
                )
            )

    refinements = {str(item["instance"]): item for item in pi.refinements}
    mechanism_refinement = {
        "update_atomic": RefinementKind.NATIVE_ATOMIC.value,
        "update_privatized": RefinementKind.PRIVATIZED_UPDATE.value,
    }
    for instance_id, instance in instances.items():
        expected_kind = mechanism_refinement.get(str(instance["mechanism"]))
        if expected_kind is None:
            continue
        refinement = refinements.get(instance_id)
        if refinement is None or refinement["kind"] != expected_kind:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_MISSING,
                    f"pi.instances[{instance_id}].mechanism",
                    f"{instance['mechanism']} requires a {expected_kind} "
                    "refinement with explicit J/B authorization",
                    instance_id,
                )
            )

    for index, refinement in enumerate(pi.refinements):
        kind = str(refinement["kind"])
        instance_id = str(refinement["instance"])
        auth_ref = refinement["auth_ref"]
        if kind in GLOBAL_SAFE_REFINEMENTS:
            if auth_ref is not None:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.GLOBAL_SAFE_AUTH_FORBIDDEN,
                        f"pi.refinements[{index}].auth_ref",
                        "global-safe identity materialization must not cite a "
                        "candidate-defined authorization",
                        str(auth_ref),
                    )
                )
            continue
        if auth_ref is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_MISSING,
                    f"pi.refinements[{index}].auth_ref",
                    "sensitive representation refinement requires J/B authorization",
                    instance_id,
                )
            )
            continue
        authorization = authorizations.get(str(auth_ref))
        if authorization is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_UNKNOWN,
                    f"pi.refinements[{index}].auth_ref",
                    "refinement cites an unknown J/B authorization",
                    str(auth_ref),
                )
            )
            continue
        if authorization.kind.value != kind:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_KIND_MISMATCH,
                    f"pi.refinements[{index}].kind",
                    "refinement kind differs from the cited authorization",
                    str(auth_ref),
                )
            )
        instance = instances.get(instance_id)
        if (
            instance is not None
            and authorization.subject_ref != instance["semantic_ref"]
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_SUBJECT_MISMATCH,
                    f"pi.refinements[{index}].auth_ref",
                    "authorization subject differs from the refined instance anchor",
                    str(auth_ref),
                )
            )

    ordered = tuple(
        sorted(diagnostics, key=lambda item: (item.path, item.code.value, item.detail))
    )
    return TypeCheckReport(
        TypeCheckStatus.PASS if not ordered else TypeCheckStatus.FAIL,
        ordered,
    )


__all__ = [
    "GLOBAL_SAFE_REFINEMENTS",
    "TYPECHECK_PROTOCOL",
    "TypeCheckDiagnostic",
    "TypeCheckErrorCode",
    "TypeCheckReport",
    "TypeCheckStatus",
    "type_check",
]
