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

from intentir.ir.canonical_j import canonicalize_semantic_archive, j_elaboration
from intentir.ir.functional_boundary import (
    functional_boundary_closure_issues,
    functional_reduction_padding_issues,
)
from intentir.verify.gate1.model import (
    ImplementationConfig,
    ParameterKind,
    RefinementKind,
    ResultRelationKind,
    ScheduleSlotKind,
    SemanticArchive,
)
from intentir.verify.gate1.pi import RealizationWitness


TYPECHECK_PROTOCOL = "intentir.scrc.typecheck.v7"
GLOBAL_SAFE_REFINEMENTS = frozenset({"identity_materialization"})

_REDUCE_FNS = frozenset({"sum", "mean", "max", "min", "prod", "any", "all"})
_REDUCE_ASSOCIATIVITY = frozenset(
    {"associative", "float_nondeterministic"}
)
_REDUCE_ATTRS = frozenset({"fn", "axes", "assoc", "keepdims"})
_CONTRACTION_ATTRS = frozenset({"subscripts"})
_EINSUM_TERM = re.compile(r"^[A-Za-z]*$")


class TypeCheckStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"


class TypeCheckErrorCode(str, Enum):
    J_ELABORATION_FAILED = "j_elaboration_failed"
    J_VALUE_TYPE_INCOMPLETE = "j_value_type_incomplete"
    KNOWN_OP_CONTRACT_INVALID = "known_op_contract_invalid"
    DUPLICATE_J_REF = "duplicate_j_ref"
    INVALID_C_SLOT = "invalid_c_slot"
    C_REQUIRED_VALUE_MISSING = "c_required_value_missing"
    C_VALUE_TYPE_MISMATCH = "c_value_type_mismatch"
    C_SUBJECT_UNKNOWN = "c_subject_unknown"
    C_AXIS_INVALID = "c_axis_invalid"
    INVALID_ACCESS_FILL = "invalid_access_fill"
    FUNCTIONAL_BOUNDARY_INCOMPLETE = "functional_boundary_incomplete"
    PADDED_MEAN_COUNT_UNPROVEN = "padded_mean_count_unproven"
    LOGICAL_INPUT_UNUSED = "logical_input_unused"
    REPRESENTATION_CARRIER_J_USE_FORBIDDEN = (
        "representation_carrier_j_use_forbidden"
    )
    RESULT_RELATION_INCOMPATIBLE = "result_relation_incompatible"
    UNRESOLVED_ANCHOR = "unresolved_anchor"
    ROUTE_CYCLE = "route_cycle"
    ROUTE_SLOT_OUT_OF_RANGE = "route_slot_out_of_range"
    ROUTE_TYPE_MISMATCH = "route_type_mismatch"
    INVALID_SHARING = "invalid_sharing"
    ITERATION_WITNESS_INVALID = "iteration_witness_invalid"
    ITERATION_WITNESS_MISSING = "iteration_witness_missing"
    OBSERVABLE_REQUIRES_J_ANCHOR = "observable_requires_j_anchor"
    AUTHORIZATION_SUBJECT_UNKNOWN = "authorization_subject_unknown"
    ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN = (
        "absent_carrier_semantic_use_forbidden"
    )
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


def _check_known_operation_contracts(
    j: SemanticArchive,
) -> tuple[TypeCheckDiagnostic, ...]:
    """Reject malformed attrs for the small set of closed known-op contracts.

    v5 deliberately closes only ``reduce`` and ``contraction`` here.  In
    particular, this checker validates the model-authored spelling; it never
    translates aliases such as ``kind`` or ``lhs_contract_axes`` into the
    canonical fields consumed by RefEval/Emit.
    """

    diagnostics: list[TypeCheckDiagnostic] = []
    try:
        value_types = j_elaboration(j).value_type_by_name
    except ValueError:
        # Canonical elaboration below remains authoritative for graph/type
        # errors.  The known-op pass can still report closed attr diagnostics.
        value_types = {}

    intent = j.intent.to_json()["intent_function"]
    for operation_path, operation in _walk_operations(
        intent.get("body", [])
    ):
        op_name = operation.get("op")
        if op_name not in {"reduce", "contraction"}:
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}

        if op_name == "contraction":
            for key in sorted(set(attrs) - _CONTRACTION_ATTRS):
                detail = (
                    "contraction uses only canonical attr 'subscripts'; "
                    f"alias {key!r} is forbidden and is never translated by "
                    "fixed code"
                    if key in {"lhs_contract_axes", "rhs_contract_axes"}
                    else f"contraction attrs contain unsupported key {key!r}"
                )
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.{key}",
                        detail,
                        ref,
                    )
                )

            operands = operation.get("operands")
            operand_refs = operands if isinstance(operands, list) else []
            if (
                len(operand_refs) != 2
                or not all(isinstance(item, str) for item in operand_refs)
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands",
                        "contraction requires exactly two value operands",
                        ref,
                    )
                )

            result = operation.get("result")
            if not isinstance(result, str) or not result:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.result",
                        "contraction requires exactly one value result",
                        ref,
                    )
                )

            subscripts = attrs.get("subscripts")
            lhs_labels: str | None = None
            rhs_labels: str | None = None
            output_labels: str | None = None
            if not isinstance(subscripts, str):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.subscripts",
                        "contraction subscripts must be an explicit binary "
                        "einsum string such as 'mk,kn->mn'",
                        ref,
                    )
                )
            else:
                compact = "".join(subscripts.split())
                inputs_and_output = compact.split("->")
                input_terms = (
                    inputs_and_output[0].split(",")
                    if len(inputs_and_output) == 2
                    else []
                )
                if (
                    len(inputs_and_output) != 2
                    or len(input_terms) != 2
                    or not all(_EINSUM_TERM.fullmatch(term) for term in input_terms)
                    or _EINSUM_TERM.fullmatch(inputs_and_output[-1]) is None
                ):
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.subscripts",
                            "contraction subscripts must use the closed binary "
                            "einsum grammar '<labels>,<labels>-><labels>' "
                            "without ellipsis",
                            ref,
                        )
                    )
                else:
                    lhs_labels, rhs_labels = input_terms
                    output_labels = inputs_and_output[1]
                    if len(set(output_labels)) != len(output_labels):
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                "contraction output labels must be unique",
                                ref,
                            )
                        )
                    unknown_output = sorted(
                        set(output_labels) - set(lhs_labels) - set(rhs_labels)
                    )
                    if unknown_output:
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                "contraction output labels must occur in an "
                                f"input term; unknown labels {unknown_output!r}",
                                ref,
                            )
                        )

            if (
                lhs_labels is not None
                and rhs_labels is not None
                and output_labels is not None
                and len(operand_refs) == 2
                and all(isinstance(item, str) for item in operand_refs)
            ):
                operand_types = [value_types.get(item) for item in operand_refs]
                for index, (labels, value_type) in enumerate(
                    zip((lhs_labels, rhs_labels), operand_types, strict=True)
                ):
                    if value_type is not None and len(labels) != len(value_type.shape):
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                f"contraction operand {index} labels have rank "
                                f"{len(labels)}, but its typed J value has rank "
                                f"{len(value_type.shape)}",
                                ref,
                            )
                        )

                if all(
                    value_type is not None
                    and len(labels) == len(value_type.shape)
                    for labels, value_type in zip(
                        (lhs_labels, rhs_labels), operand_types, strict=True
                    )
                ):
                    label_extents: dict[str, Any] = {}
                    for labels, value_type in zip(
                        (lhs_labels, rhs_labels), operand_types, strict=True
                    ):
                        assert value_type is not None
                        for label, extent in zip(
                            labels, value_type.shape, strict=True
                        ):
                            previous = label_extents.get(label, extent)
                            if previous != extent:
                                diagnostics.append(
                                    TypeCheckDiagnostic(
                                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                        f"{base}.attrs.subscripts",
                                        f"contraction label {label!r} has "
                                        f"inconsistent typed extents {previous!r} "
                                        f"and {extent!r}",
                                        ref,
                                    )
                                )
                            else:
                                label_extents[label] = extent

                    result_type = (
                        value_types.get(result)
                        if isinstance(result, str)
                        else None
                    )
                    if (
                        result_type is not None
                        and set(output_labels).issubset(label_extents)
                    ):
                        expected_shape = tuple(
                            label_extents[label] for label in output_labels
                        )
                        if result_type.shape != expected_shape:
                            diagnostics.append(
                                TypeCheckDiagnostic(
                                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                    f"{base}.type.shape",
                                    "contraction typed result shape "
                                    f"{result_type.shape!r} does not match "
                                    f"subscripts result {expected_shape!r}",
                                    ref,
                                )
                            )
            continue

        for key in sorted(set(attrs) - _REDUCE_ATTRS):
            detail = (
                "reduce uses canonical attr 'fn'; alias 'kind' is forbidden "
                "and is never translated by fixed code"
                if key == "kind"
                else f"reduce attrs contain unsupported key {key!r}"
            )
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.{key}",
                    detail,
                    ref,
                )
            )

        fn = attrs.get("fn")
        if not isinstance(fn, str) or fn not in _REDUCE_FNS:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.fn",
                    "reduce fn must be one of "
                    f"{sorted(_REDUCE_FNS)!r}; aliases are not accepted",
                    ref,
                )
            )

        assoc = attrs.get("assoc")
        if not isinstance(assoc, str) or assoc not in _REDUCE_ASSOCIATIVITY:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.assoc",
                    "reduce assoc must be 'associative' or "
                    "'float_nondeterministic'",
                    ref,
                )
            )

        operands = operation.get("operands")
        operand_refs = operands if isinstance(operands, list) else []
        if len(operand_refs) != 1 or not isinstance(operand_refs[0], str):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.operands",
                    "reduce requires exactly one value operand",
                    ref,
                )
            )
            operand_rank = None
        else:
            operand_type = value_types.get(operand_refs[0])
            operand_rank = (
                len(operand_type.shape) if operand_type is not None else None
            )

        axes = attrs.get("axes")
        if not isinstance(axes, list) or not axes:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.axes",
                    "reduce axes must be a non-empty array of integer axes",
                    ref,
                )
            )
        else:
            normalized_axes: set[int] = set()
            for index, axis in enumerate(axes):
                axis_path = f"{base}.attrs.axes[{index}]"
                if type(axis) is not int:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            "reduce axis must be an integer, not bool or text",
                            ref,
                        )
                    )
                    continue
                normalized = (
                    axis + operand_rank
                    if operand_rank is not None and axis < 0
                    else axis
                )
                if operand_rank is not None and not 0 <= normalized < operand_rank:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            f"reduce axis {axis} is out of range for "
                            f"rank-{operand_rank} operand",
                            ref,
                        )
                    )
                    continue
                if normalized in normalized_axes:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            "reduce axes must be unique after negative-axis "
                            "normalization",
                            ref,
                        )
                    )
                    continue
                normalized_axes.add(normalized)

        if "keepdims" in attrs and type(attrs["keepdims"]) is not bool:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.keepdims",
                    "optional reduce keepdims must be a boolean",
                    ref,
                )
            )

    return tuple(
        sorted(
            diagnostics,
            key=lambda item: (item.path, item.code.value, item.detail),
        )
    )


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


def _ordered_expression_refs(expression: Any) -> tuple[str, ...]:
    result: list[str] = []

    def visit(node: Any) -> None:
        if getattr(node, "op", None) == "ref":
            value = getattr(node, "value", None)
            if isinstance(value, str) and value not in result:
                result.append(value)
        for argument in getattr(node, "args", ()):
            visit(argument)

    visit(expression)
    return tuple(result)


def _axis_refs_v1(
    j: SemanticArchive,
    subject_ref: str,
    node_operations: Mapping[str, Mapping[str, Any]],
) -> frozenset[int | str]:
    """Return the closed dependent axis domain for one C attachment."""

    dimensions = {
        dim
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dim in tensor.shape
        if isinstance(dim, str) and re.fullmatch(r"x[0-9]+", dim)
    }
    if subject_ref.startswith("n"):
        operation = node_operations.get(subject_ref)
        if operation is None:
            return frozenset()
        if operation.get("op") == "reduce":
            attrs = operation.get("attrs")
            attrs = attrs if isinstance(attrs, Mapping) else {}
            axes = attrs.get("axes")
            if not isinstance(axes, list):
                return frozenset()
            return frozenset(
                axis
                for axis in axes
                if isinstance(axis, int)
                and not isinstance(axis, bool)
                and axis >= 0
            )
        if operation.get("op") == "loop":
            trip = operation.get("trip")
            axis = trip.get("axis") if isinstance(trip, Mapping) else None
            result: set[int | str] = {0}
            if isinstance(axis, str) and axis in dimensions:
                result.add(axis)
            return frozenset(result)
        return frozenset()

    elaboration = j_elaboration(j)
    if subject_ref.startswith("d"):
        expression = elaboration.domain_by_id.get(subject_ref)
        if expression is None:
            return frozenset()
        refs = _ordered_expression_refs(expression)
        logical_axes = tuple(ref for ref in refs if ref not in dimensions)
        return frozenset(
            [*range(len(logical_axes)), *(ref for ref in refs if ref in dimensions)]
        )
    if subject_ref.startswith("a"):
        access = elaboration.access_by_id.get(subject_ref)
        if access is None:
            return frozenset()
        access_spec = next(
            (
                item
                for item in j.boundary.accesses
                if item.access_id == subject_ref
            ),
            None,
        )
        tensors = {
            item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
        }
        tensor = (
            tensors.get(access_spec.tensor) if access_spec is not None else None
        )
        symbolic = {
            dim
            for index, dim in enumerate(tensor.shape if tensor is not None else [])
            if index < len(access.index.components) and dim in dimensions
        }
        return frozenset([*range(len(access.index.components)), *symbolic])
    # Scope/storage attachments are intentionally axis-free in v1.
    return frozenset()


_TOLERANCE_UNARY_FNS = frozenset(
    {
        "exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid", "cos",
        "sin", "tan", "exp2", "atan", "erf",
    }
)


def _floating_dtype(dtype: str | None) -> bool:
    if dtype is None:
        return False
    normalized = dtype.casefold()
    return normalized in {"float", "half", "bfloat"} or re.fullmatch(
        r"(?:(?:b?float)|bf|f|fp)[0-9]+", normalized
    ) is not None


def _operation_dependencies(operation: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in result:
            result.append(value)

    for value in operation.get("operands", []):
        add(value)
    if operation.get("op") == "loop":
        for item in operation.get("scan", []):
            if isinstance(item, Mapping):
                add(item.get("from"))
        for item in operation.get("carry", []):
            if isinstance(item, Mapping):
                add(item.get("init_value"))
                add(item.get("next_value"))
        for value in operation.get("yield", []):
            add(value)
    return tuple(result)


def _output_dependency(
    j: SemanticArchive, output: str
) -> tuple[set[str], list[Mapping[str, Any]]]:
    intent = j.intent.to_json()["intent_function"]
    operations = dict(_walk_operations(intent.get("body", [])))
    path_to_node = {path: node for node, path in j.node_ids.items()}
    producers: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
    for path, operation in operations.items():
        node_ref = path_to_node.get(path, path)
        for result in _operation_results(operation.get("result")):
            producers[result] = (node_ref, path, operation)

    bindings = {
        item.output: item for item in j_elaboration(j).output_bindings
    }
    binding = bindings.get(output)
    if binding is None:
        return set(), []

    refs: set[str] = set()
    dependency_ops: list[Mapping[str, Any]] = []
    pending = [binding.value]
    visited_values: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        producer = producers.get(value)
        if producer is None:
            continue
        node_ref, path, operation = producer
        refs.update({node_ref, path})
        dependency_ops.append(operation)
        pending.extend(_operation_dependencies(operation))
    return refs, dependency_ops


def _semantic_value_dependency_closure(
    j: SemanticArchive,
    seeds: Iterable[str],
) -> set[str]:
    """Return values contributing to explicit J semantic roots.

    This is ordinary finite graph reachability over the already typed J.  It
    does not inspect Source, C, pi, parameter names, or runtime values.
    """

    intent = j.intent.to_json()["intent_function"]
    producers: dict[str, Mapping[str, Any]] = {}
    for _path, operation in _walk_operations(intent.get("body", [])):
        for result in _operation_results(operation.get("result")):
            producers[result] = operation

    live: set[str] = set()
    pending = list(seeds)
    while pending:
        value = pending.pop()
        if value in live:
            continue
        live.add(value)
        producer = producers.get(value)
        if producer is not None:
            pending.extend(_operation_dependencies(producer))
    return live


def _rank_zero_semantic_live_values(j: SemanticArchive) -> set[str]:
    """Collect J-owned roots that can make a rank-0 logical input meaningful."""

    elaboration = j_elaboration(j)
    semantic_refs: set[str] = {
        binding.value for binding in elaboration.output_bindings
    }
    node_seed_refs: set[str] = set()

    intent = j.intent.to_json()["intent_function"]
    operations = dict(_walk_operations(intent.get("body", [])))
    path_to_node = {path: node for node, path in j.node_ids.items()}
    nodes: dict[str, Mapping[str, Any]] = dict(operations)
    for path, operation in operations.items():
        node = path_to_node.get(path)
        if node is not None:
            nodes[node] = operation

    def add_expression_refs(expression: Any) -> None:
        semantic_refs.update(_ordered_expression_refs(expression))

    for _domain_id, expression in elaboration.domains:
        add_expression_refs(expression)
    accesses = elaboration.access_by_id
    for access in j.boundary.accesses:
        semantic_refs.add(access.tensor)
        parsed = accesses.get(access.access_id)
        if parsed is not None:
            for component in parsed.index.components:
                add_expression_refs(component)
            if parsed.predicate is not None:
                add_expression_refs(parsed.predicate)
        if access.node_ref is not None:
            node_seed_refs.add(access.node_ref)
    for effect in j.boundary.effects:
        semantic_refs.add(effect.target)
        if effect.node_ref is not None:
            node_seed_refs.add(effect.node_ref)
    for item in j.boundary.broadcasts:
        semantic_refs.add(item.operand)
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.aliases:
        semantic_refs.update({item.left, item.right})
    for item in j.boundary.output_mappings:
        semantic_refs.add(item.value)
        if item.node_ref is not None:
            node_seed_refs.add(item.node_ref)
    for item in j.boundary.reductions:
        if isinstance(item.tile_axis, str):
            semantic_refs.add(item.tile_axis)
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.index_bounds:
        semantic_refs.update({item.data, item.indices})
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.updates:
        semantic_refs.update({item.base, item.indices, item.values})
        node_seed_refs.add(item.node_ref)
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        semantic_refs.update(
            dimension
            for dimension in tensor.shape
            if isinstance(dimension, str)
        )

    for node_ref in node_seed_refs:
        operation = nodes.get(node_ref)
        if operation is not None:
            semantic_refs.update(_operation_results(operation.get("result")))
    return _semantic_value_dependency_closure(j, semantic_refs)


def _check_rank_zero_logical_input_liveness(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Reject only mechanically dead rank-0 signature logical inputs.

    General tensor liveness is deliberately out of scope for v7.  A rank-0 J
    input is accepted when it reaches an observable output dependency or is
    explicitly used by a J-owned boundary/shape relation.  Merely routing its
    ABI parameter through pi is representation use and does not make it a
    logical input.
    """

    rank_zero_inputs = {
        tensor.name
        for tensor in j.intent.inputs
        if not tensor.shape
    }
    has_representation_scalar = any(
        parameter.kind is ParameterKind.SCALAR
        and parameter.logical_ref is None
        for parameter in j.signature.parameters
    )
    if not rank_zero_inputs and not has_representation_scalar:
        return
    live = _rank_zero_semantic_live_values(j)
    representation_j_uses = set(live)
    intent = j.intent.to_json()["intent_function"]
    for _path, operation in _walk_operations(intent.get("body", [])):
        representation_j_uses.update(_operation_dependencies(operation))
    for index, parameter in enumerate(j.signature.parameters):
        logical_ref = parameter.logical_ref
        if (
            parameter.kind is ParameterKind.SCALAR
            and logical_ref is None
            and parameter.name in representation_j_uses
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REPRESENTATION_CARRIER_J_USE_FORBIDDEN,
                    f"j.signature.parameters[{index}].logical_ref",
                    "representation-only scalar without logical_ref cannot "
                    "participate in J output, boundary, or shape semantics; "
                    "it may be referenced only by pi realization topology",
                    parameter.name,
                )
            )
            continue
        if logical_ref not in rank_zero_inputs or logical_ref in live:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.LOGICAL_INPUT_UNUSED,
                f"j.signature.parameters[{index}].logical_ref",
                "rank-0 logical input does not contribute to any J output "
                "dependency or explicit boundary/shape semantic relation; "
                "a representation-only ABI parameter must use "
                "logical_ref=null and may be realized only through pi",
                logical_ref,
            )
        )


def _check_result_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    output_types = {item.name: item.dtype for item in j.intent.outputs}
    relations = {
        item.output: (index, item)
        for index, item in enumerate(j.boundary.result_relations)
    }
    reassociation_subjects = {
        item.subject_ref
        for item in j.boundary.realization_authorizations
        if item.kind is RefinementKind.REASSOCIATION
    }
    for output, dtype in output_types.items():
        if not _floating_dtype(dtype):
            continue
        dependency_refs, dependency_ops = _output_dependency(j, output)
        requires_tolerance = bool(dependency_refs & reassociation_subjects)
        for operation in dependency_ops:
            result_type = _explicit_type(operation)
            if (
                operation.get("op") == "reduce"
                and result_type is not None
                and _floating_dtype(result_type.dtype)
            ):
                requires_tolerance = True
            attrs = operation.get("attrs")
            attrs = attrs if isinstance(attrs, Mapping) else {}
            if (
                operation.get("op") == "map_unary"
                and attrs.get("fn") in _TOLERANCE_UNARY_FNS
            ):
                requires_tolerance = True
        if not requires_tolerance:
            continue
        indexed = relations.get(output)
        relation = indexed[1] if indexed is not None else None
        if relation is not None and relation.kind is ResultRelationKind.TOLERANCE:
            continue
        path = (
            f"j.boundary.result_relations[{indexed[0]}].kind"
            if indexed is not None else "j.boundary.result_relations"
        )
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE,
                path,
                "floating transcendental/reduction/reassociation output "
                "requires explicit tolerance relation",
                output,
            )
        )


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

    value_types: dict[str, _ValueType] = {
        name: _ValueType(value.dtype, len(value.shape))
        for name, value in j_elaboration(j).value_type_by_name.items()
    }
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        value_type = _ValueType(tensor.dtype, len(tensor.shape))
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
        node_operations[path] = operation
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


def _check_total_j_value_typing(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every Layer-A result to have one closed canonical value type."""

    typed_values = j_elaboration(j).value_type_by_name
    body = j.intent.to_json()["intent_function"].get("body", [])
    for path, operation in _walk_operations(body):
        for result in _operation_results(operation.get("result")):
            if result in typed_values:
                continue
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_VALUE_TYPE_INCOMPLETE,
                    f"j.intent.intent_function.{path}.result",
                    "Layer-A result has no uniquely determined closed value type",
                    result,
                )
            )


_ITERATION_TILE_ROLES = frozenset(
    {"tile_extent", "reduction_tile_extent"}
)


def _iteration_axis_matches(axis_ref: int | str | None, trip_axis: str) -> bool:
    """Recognize the two AxisRefs_v1 spellings of one loop trip axis."""

    return axis_ref == 0 or axis_ref == trip_axis


def _check_iteration_witness(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Close the physical induction basis of every C-tiled J loop."""

    loop_axes: dict[str, str] = {}
    for node_ref, operation in node_operations.items():
        if not re.fullmatch(r"n[0-9]+", node_ref):
            continue
        if operation.get("op") != "loop":
            continue
        trip = operation.get("trip")
        axis = trip.get("axis") if isinstance(trip, Mapping) else None
        if isinstance(axis, str):
            loop_axes[node_ref] = axis

    all_slots = {slot.name: slot for slot in c.slots}
    tile_slots = {
        slot.name: slot
        for slot in c.slots
        if slot.role in _ITERATION_TILE_ROLES
        and slot.subject_ref in loop_axes
        and _iteration_axis_matches(
            slot.axis_ref, loop_axes[str(slot.subject_ref)]
        )
    }
    required_pairs = {
        (str(slot.subject_ref), slot.name) for slot in tile_slots.values()
    }
    regions = {str(item["id"]): item for item in pi.regions}
    instances = {str(item["id"]): item for item in pi.instances}

    def visible_from(instance_region: str, relation_region: str) -> bool:
        current: str | None = relation_region
        while current is not None:
            if current == instance_region:
                return True
            parent = regions.get(current, {}).get("parent")
            current = str(parent) if parent is not None else None
        return False

    witnessed_pairs: set[tuple[str, str]] = set()

    for index, relation in enumerate(pi.iterations):
        path = f"pi.iterations[{index}]"
        region_ref = str(relation["region"])
        region = regions.get(region_ref)
        if region is None or region["placement"] != "loop":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
                    f"{path}.region",
                    "iteration relation requires a placement=loop region",
                    region_ref,
                )
            )

        loop_instance_ref = str(relation["loop_instance"])
        loop_instance = instances.get(loop_instance_ref)
        loop_ref = (
            str(loop_instance["semantic_ref"])
            if loop_instance is not None
            else ""
        )
        valid_loop = (
            loop_ref in loop_axes
            and loop_instance is not None
            and visible_from(str(loop_instance["region"]), region_ref)
        )
        if not valid_loop:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
                    f"{path}.loop_instance",
                    "loop_instance must anchor a structured J loop in the "
                    "relation region or one of its lexical ancestors",
                    loop_instance_ref,
                )
            )

        tile_instance_ref = str(relation["tile_instance"])
        tile_instance = instances.get(tile_instance_ref)
        tile_ref = (
            str(tile_instance["semantic_ref"])
            if tile_instance is not None
            else ""
        )
        # An explicit iteration relation itself identifies the referenced C
        # value as the physical tile.  The reserved tile roles are used only
        # to make omission mechanically detectable; they are not a hidden
        # validity condition on an otherwise closed relation.
        slot = all_slots.get(tile_ref)
        value = c.values.get(tile_ref)
        valid_tile = (
            slot is not None
            and slot.subject_ref == loop_ref
            and loop_ref in loop_axes
            and _iteration_axis_matches(
                slot.axis_ref, loop_axes[loop_ref]
            )
            and tile_instance is not None
            and visible_from(str(tile_instance["region"]), region_ref)
            and slot.required
            and slot.value_type.casefold() in {"int", "integer", "i32", "i64", "u32", "u64"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )
        if not valid_tile:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
                    f"{path}.tile_instance",
                    "tile_instance must anchor one required positive-integer C "
                    "tile slot attached to the same J loop and trip axis, with "
                    "its instance visible from the relation region",
                    tile_instance_ref,
                )
            )

        pair = (loop_ref, tile_ref)
        if valid_loop and valid_tile and region is not None:
            witnessed_pairs.add(pair)

    for loop_ref, tile_ref in sorted(required_pairs - witnessed_pairs):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
                "pi.iterations",
                "C-tiled structured J loop requires at least one physical "
                "induction-basis relation",
                f"{loop_ref}:{tile_ref}",
            )
        )


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

    known_op_diagnostics = _check_known_operation_contracts(j)
    if known_op_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, known_op_diagnostics)

    try:
        j = canonicalize_semantic_archive(j)
    except ValueError as exc:
        return TypeCheckReport(
            TypeCheckStatus.FAIL,
            (
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_ELABORATION_FAILED,
                    "j",
                    str(exc),
                ),
            ),
        )

    # The first pass above rejects malformed surface attrs without asking the
    # elaborator to interpret them.  Re-run on accepted canonical J so axis
    # bounds can use its closed operand ranks.
    known_op_diagnostics = _check_known_operation_contracts(j)
    if known_op_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, known_op_diagnostics)

    diagnostics: list[TypeCheckDiagnostic] = []
    _check_total_j_value_typing(j, diagnostics)
    _check_rank_zero_logical_input_liveness(j, diagnostics)
    diagnostics.extend(
        TypeCheckDiagnostic(
            TypeCheckErrorCode.FUNCTIONAL_BOUNDARY_INCOMPLETE,
            issue.path,
            issue.detail,
            issue.ref,
        )
        for issue in functional_boundary_closure_issues(j)
    )
    diagnostics.extend(
        TypeCheckDiagnostic(
            TypeCheckErrorCode.PADDED_MEAN_COUNT_UNPROVEN,
            issue.path,
            issue.detail,
            issue.ref,
        )
        for issue in functional_reduction_padding_issues(j)
    )
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

    anchors, node_operations = _build_anchors(j, c, diagnostics)
    absent_signature_refs = {
        parameter.name
        for parameter in j.signature.parameters
        if parameter.kind is ParameterKind.OPAQUE and parameter.dtype == "none"
    }
    _check_result_relations(j, diagnostics)
    c_subject_kinds = {"node", "domain", "access", "scope", "storage"}
    for index, slot in enumerate(c.slots):
        if slot.subject_ref is None:
            continue
        subject = anchors.get(slot.subject_ref)
        if (
            subject is None
            or subject.owner != "j"
            or subject.kind not in c_subject_kinds
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_SUBJECT_UNKNOWN,
                    f"c.slots[{index}].subject_ref",
                    "C subject_ref is not a declared schedule-addressable J subject",
                    slot.subject_ref,
                )
            )
            continue
        if slot.axis_ref is None:
            continue
        allowed_axes = _axis_refs_v1(j, slot.subject_ref, node_operations)
        if slot.axis_ref not in allowed_axes:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_AXIS_INVALID,
                    f"c.slots[{index}].axis_ref",
                    "C axis_ref is outside AxisRefs_v1(J,subject_ref)",
                    str(slot.axis_ref),
                )
            )
    instances = {str(item["id"]): item for item in pi.instances}
    instance_anchors: dict[str, _Anchor] = {}
    for index, item in enumerate(pi.instances):
        ref = str(item["semantic_ref"])
        if ref in absent_signature_refs:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
                    f"pi.instances[{index}].semantic_ref",
                    "opaque(dtype=none) is an absent ABI carrier and cannot "
                    "own a realization instance",
                    ref,
                )
            )
            continue
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

    _check_iteration_witness(j, c, pi, node_operations, diagnostics)

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
        if authorization.subject_ref in absent_signature_refs:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
                    f"j.boundary.realization_authorizations[{index}].subject_ref",
                    "opaque(dtype=none) is an absent ABI carrier and cannot "
                    "authorize a semantic realization refinement",
                    authorization.subject_ref,
                )
            )
            continue
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
            continue
        if authorization.kind is RefinementKind.NUMERIC_CAST:
            operation = node_operations.get(authorization.subject_ref)
            valid_cast = (
                subject is not None
                and subject.kind == "node"
                and operation is not None
                and operation.get("op") == "cast"
                and len(subject.operand_types) == 1
                and subject.operand_types[0] is not None
                and subject.value_type is not None
                and authorization.source_dtype is not None
                and authorization.target_dtype is not None
            )
            if not valid_cast:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                        f"j.boundary.realization_authorizations[{index}]",
                        "numeric_cast authorization requires one typed cast "
                        "operand/result and explicit source/target dtype",
                        authorization.auth_id,
                    )
                )
                continue
            assert subject is not None
            assert subject.operand_types[0] is not None
            assert subject.value_type is not None
            if (
                authorization.source_dtype != subject.operand_types[0].dtype
                or authorization.target_dtype != subject.value_type.dtype
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                        f"j.boundary.realization_authorizations[{index}]",
                        "numeric_cast dtype must match cast operand/result types",
                        authorization.auth_id,
                    )
                )
            continue
        if (
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
        refined_instance = instances.get(instance_id)
        if (
            refined_instance is not None
            and refined_instance["semantic_ref"] in absent_signature_refs
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
                    f"pi.refinements[{index}].instance",
                    "opaque(dtype=none) is an absent ABI carrier and cannot "
                    "serve as a refinement carrier",
                    instance_id,
                )
            )
            continue
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
