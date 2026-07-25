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

from intentir.ir.canonical_j import (
    CanonicalValueType,
    canonicalize_semantic_archive,
    j_elaboration,
)
from intentir.ir.dim_expr import (
    DimExprError,
    DimensionConstraintSet,
    parse_affine_dim_expr,
)
from intentir.ir.functional_boundary import (
    functional_boundary_closure_issues,
    functional_consumer_extension_requirements,
    functional_reduction_padding_issues,
    resolve_consumer_local_tile_axis,
)
from intentir.ir.value_dimension import (
    ValueDimensionRelationError,
    validate_value_dimension_relations,
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


TYPECHECK_PROTOCOL = "intentir.scrc.typecheck.v13"
GLOBAL_SAFE_REFINEMENTS = frozenset({"identity_materialization"})

_REDUCE_FNS = frozenset({"sum", "mean", "max", "min", "prod", "any", "all"})
_REDUCE_ASSOCIATIVITY = frozenset(
    {"associative", "float_nondeterministic"}
)
_REDUCE_ATTRS = frozenset({"fn", "axes", "assoc", "keepdims"})
_CONTRACTION_ATTRS = frozenset({"subscripts"})
_TILE_ATTRS = frozenset({"reps"})
_EINSUM_TERM = re.compile(r"^[A-Za-z]*$")
_INTEGER_DTYPE = re.compile(r"^(?P<signedness>[iu])(?P<bits>[1-9][0-9]*)$")


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
    BOUNDARY_VALUE_NOT_DOMINATING = "boundary_value_not_dominating"
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
    CONSUMER_EXTENSION_INVALID = "consumer_extension_invalid"
    CONSUMER_EXTENSION_MISSING = "consumer_extension_missing"
    DIMENSION_CONSTRAINT_INVALID = "dimension_constraint_invalid"
    VALUE_DIMENSION_RELATION_INVALID = "value_dimension_relation_invalid"
    PAD_SHAPE_INVALID = "pad_shape_invalid"
    PAD_SHAPE_CONSTRAINT_MISSING = "pad_shape_constraint_missing"
    TILE_SHAPE_INVALID = "tile_shape_invalid"
    LOOP_CARRIER_INVALID = "loop_carrier_invalid"
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
    REFINEMENT_AUTH_INVALID_WIDEN = "refinement_auth_invalid_widen"
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
    carrier_class: str | None = None


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


def _surface_value_types(j: SemanticArchive) -> dict[str, CanonicalValueType]:
    """Read only explicitly authored value types without elaborating J."""

    result = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    intent = j.intent.to_json()["intent_function"]
    for _path, operation in _walk_operations(intent.get("body", [])):
        names = _operation_results(operation.get("result"))
        declared = operation.get("type")
        if (
            len(names) != 1
            or not isinstance(declared, Mapping)
            or not isinstance(declared.get("dtype"), str)
            or not isinstance(declared.get("shape"), list)
        ):
            continue
        result[names[0]] = CanonicalValueType(
            declared["dtype"], tuple(declared["shape"])
        )
    return result


def _declared_dimensions(j: SemanticArchive) -> frozenset[str]:
    return frozenset(
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str) and re.fullmatch(r"x[0-9]+", dimension)
    )


def _check_dimension_and_pad_contracts(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Prove model-authored affine pad shapes without Harness valuations."""

    dimensions = _declared_dimensions(j)
    try:
        constraints = DimensionConstraintSet.parse(
            j.dimension_constraints,
            dimensions=dimensions,
        )
    except DimExprError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.DIMENSION_CONSTRAINT_INVALID,
                "j.dimension_constraints",
                str(exc),
            )
        )
        return

    value_types = j_elaboration(j).value_type_by_name
    intent = j.intent.to_json()["intent_function"]
    for operation_path, operation in _walk_operations(intent.get("body", [])):
        if operation.get("op") != "pad":
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}
        operands = operation.get("operands")
        result = operation.get("result")
        if set(attrs) != {"lo", "hi", "value"}:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.attrs",
                    "pad requires exactly the closed attrs lo, hi, and value",
                    ref,
                )
            )
            continue
        if (
            not isinstance(operands, list)
            or len(operands) != 1
            or not isinstance(operands[0], str)
            or not isinstance(result, str)
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.operands",
                    "pad requires exactly one typed operand and one result",
                    ref,
                )
            )
            continue
        operand_type = value_types.get(operands[0])
        result_type = value_types.get(result)
        if operand_type is None or result_type is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.type",
                    "pad operand and result require closed J value types",
                    ref,
                )
            )
            continue
        if operand_type.dtype != result_type.dtype:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.type.dtype",
                    "pad preserves the operand dtype",
                    ref,
                )
            )
        lo = attrs.get("lo")
        hi = attrs.get("hi")
        rank = len(operand_type.shape)
        if (
            not isinstance(lo, list)
            or not isinstance(hi, list)
            or len(result_type.shape) != rank
            or len(lo) != rank
            or len(hi) != rank
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.attrs",
                    "pad lo/hi lengths and result rank must equal the operand rank",
                    ref,
                )
            )
            continue

        for axis, (input_dim, low, high, output_dim) in enumerate(
            zip(operand_type.shape, lo, hi, result_type.shape, strict=True)
        ):
            axis_path = f"{base}.attrs"
            try:
                input_form = parse_affine_dim_expr(
                    input_dim,
                    dimensions=dimensions,
                    context=f"{base}.operand_shape[{axis}]",
                )
                low_form = parse_affine_dim_expr(
                    low,
                    dimensions=dimensions,
                    context=f"{base}.attrs.lo[{axis}]",
                )
                high_form = parse_affine_dim_expr(
                    high,
                    dimensions=dimensions,
                    context=f"{base}.attrs.hi[{axis}]",
                )
                output_form = parse_affine_dim_expr(
                    output_dim,
                    dimensions=dimensions,
                    context=f"{base}.result_shape[{axis}]",
                )
            except DimExprError as exc:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.PAD_SHAPE_INVALID,
                        axis_path,
                        str(exc),
                        ref,
                    )
                )
                continue

            expected = input_form.plus(
                low_form, context=f"{base}.shape[{axis}]"
            ).plus(high_form, context=f"{base}.shape[{axis}]")
            if expected != output_form:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.PAD_SHAPE_INVALID,
                        axis_path,
                        "pad shape equation is not an affine identity: "
                        f"input + lo + hi normalizes to {expected.to_text()!r}, "
                        f"but result is {output_form.to_text()!r}",
                        ref,
                    )
                )
            for field, form in (("lo", low_form), ("hi", high_form)):
                if constraints.proves_nonnegative(form):
                    continue
                code = (
                    TypeCheckErrorCode.PAD_SHAPE_INVALID
                    if form.is_constant
                    else TypeCheckErrorCode.PAD_SHAPE_CONSTRAINT_MISSING
                )
                diagnostics.append(
                    TypeCheckDiagnostic(
                        code,
                        f"{base}.attrs.{field}[{axis}]",
                        "pad extent is not mechanically proven non-negative "
                        "by J.dimension_constraints",
                        ref,
                    )
                )


def _check_value_dimension_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    try:
        validate_value_dimension_relations(j)
    except ValueDimensionRelationError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.VALUE_DIMENSION_RELATION_INVALID,
                "j.value_dimension_relations",
                str(exc),
            )
        )


def _check_known_operation_contracts(
    j: SemanticArchive,
) -> tuple[TypeCheckDiagnostic, ...]:
    """Reject malformed attrs for the small set of closed known-op contracts.

    This checker validates the model-authored spelling; it never translates
    aliases or repairs dependent result types into the canonical fields
    consumed by RefEval/Emit.
    """

    diagnostics: list[TypeCheckDiagnostic] = []
    try:
        value_types = j_elaboration(j).value_type_by_name
    except (IndexError, KeyError, TypeError, ValueError):
        # Canonical elaboration below remains authoritative for graph/type
        # errors.  Explicitly authored types are still sufficient for local
        # dependent-contract diagnostics; they are never completed here.
        value_types = _surface_value_types(j)

    intent = j.intent.to_json()["intent_function"]
    for operation_path, operation in _walk_operations(
        intent.get("body", [])
    ):
        op_name = operation.get("op")
        if op_name not in {"reduce", "contraction", "tile"}:
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}

        if op_name == "tile":
            for key in sorted(set(attrs) - _TILE_ATTRS):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.{key}",
                        f"tile attrs contain unsupported key {key!r}",
                        ref,
                    )
                )
            if set(attrs) != _TILE_ATTRS:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs",
                        "tile requires exactly the closed attr reps",
                        ref,
                    )
                )

            operands = operation.get("operands")
            operand_refs = operands if isinstance(operands, list) else []
            result = operation.get("result")
            valid_arity = (
                len(operand_refs) == 1
                and isinstance(operand_refs[0], str)
                and isinstance(result, str)
                and bool(result)
            )
            if not valid_arity:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.operands",
                        "tile requires exactly one value operand and one result",
                        ref,
                    )
                )

            reps = attrs.get("reps")
            valid_reps = isinstance(reps, list) and all(
                type(repetition) is int and repetition >= 0
                for repetition in reps
            )
            if not valid_reps:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.reps",
                        "tile reps must be a closed array of non-negative "
                        "integers",
                        ref,
                    )
                )

            if not valid_arity or not valid_reps:
                continue
            assert isinstance(result, str)
            operand_type = value_types.get(operand_refs[0])
            result_type = value_types.get(result)
            if operand_type is None or result_type is None:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.type",
                        "tile operand and result require closed J value types",
                        ref,
                    )
                )
                continue
            if operand_type.dtype != result_type.dtype:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.type.dtype",
                        "tile preserves the operand dtype",
                        ref,
                    )
                )
            rank = len(operand_type.shape)
            if len(result_type.shape) != rank or len(reps) != rank:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.reps",
                        "tile reps length and result rank must equal the "
                        "operand rank",
                        ref,
                    )
                )
                continue

            dimensions = _declared_dimensions(j)
            try:
                constraints = DimensionConstraintSet.parse(
                    j.dimension_constraints,
                    dimensions=dimensions,
                )
            except DimExprError:
                # The dedicated dimension pass owns malformed constraints.
                constraints = None
            for axis, (input_dim, repetition, output_dim) in enumerate(
                zip(
                    operand_type.shape,
                    reps,
                    result_type.shape,
                    strict=True,
                )
            ):
                try:
                    input_form = parse_affine_dim_expr(
                        input_dim,
                        dimensions=dimensions,
                        context=f"{base}.operand_shape[{axis}]",
                    )
                    output_form = parse_affine_dim_expr(
                        output_dim,
                        dimensions=dimensions,
                        context=f"{base}.result_shape[{axis}]",
                    )
                    expected = input_form.scaled(
                        repetition,
                        context=f"{base}.shape[{axis}]",
                    )
                except DimExprError as exc:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.TILE_SHAPE_INVALID,
                            f"{base}.type.shape[{axis}]",
                            str(exc),
                            ref,
                        )
                    )
                    continue
                if expected == output_form:
                    continue
                difference = output_form.plus(
                    expected,
                    factor=-1,
                    context=f"{base}.shape[{axis}]",
                )
                proven_equal = (
                    constraints is not None
                    and constraints.proves_nonnegative(difference)
                    and constraints.proves_nonnegative(
                        difference.scaled(
                            -1,
                            context=f"{base}.shape[{axis}]",
                        )
                    )
                )
                if not proven_equal:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.TILE_SHAPE_INVALID,
                            f"{base}.type.shape[{axis}]",
                            "tile shape equation is not proven: input * reps "
                            f"normalizes to {expected.to_text()!r}, but result "
                            f"is {output_form.to_text()!r}",
                            ref,
                        )
                    )
            continue

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


def _check_boundary_value_dominance(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require Layer-B access expressions to use visible Layer-A values.

    ``SemanticArchive`` already checks ordinary Layer-A operands by walking
    each operation sequence with an incrementally extended visible-value set.
    Boundary index/predicate expressions are stored separately from that
    sequence, so their Layer-A value references need the same source-free
    use-order check at the access event selected by ``access.node_ref``.  Axis
    and dimension references are excluded simply because they are not members
    of the Layer-A value set; the checker does not infer an alternate meaning
    from their position in an index expression.
    """

    function = j.intent.to_json()["intent_function"]
    operations = function.get("body", [])
    inputs = {
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    all_values = set(inputs)
    visible_before: dict[str, frozenset[str]] = {}

    def operation_results(operation: Mapping[str, Any]) -> tuple[str, ...]:
        return _operation_results(operation.get("result"))

    def visit_sequence(
        body: Any,
        visible: set[str],
        prefix: str,
    ) -> set[str]:
        if not isinstance(body, list):
            return set(visible)
        current = set(visible)
        for index, raw_operation in enumerate(body):
            if not isinstance(raw_operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            visible_before[path] = frozenset(current)
            results = operation_results(raw_operation)
            all_values.update(results)

            nested = raw_operation.get("body")
            if isinstance(nested, list):
                nested_visible = set(current)
                scans = raw_operation.get("scan")
                if isinstance(scans, list):
                    nested_visible.update(
                        item.get("name")
                        for item in scans
                        if isinstance(item, Mapping)
                        and isinstance(item.get("name"), str)
                    )
                carries = raw_operation.get("carry")
                if isinstance(carries, list):
                    nested_visible.update(
                        item.get("body_argument")
                        for item in carries
                        if isinstance(item, Mapping)
                        and isinstance(item.get("body_argument"), str)
                    )
                all_values.update(nested_visible - current)
                visit_sequence(nested, nested_visible, f"{path}.body")
            current.update(results)
        return current

    visible_at_function_exit = visit_sequence(operations, inputs, "body")
    elaboration = j_elaboration(j)
    for index, access in enumerate(j.boundary.accesses):
        parsed = elaboration.access_by_id.get(access.access_id)
        if parsed is None:
            continue
        if access.node_ref is None:
            visible = visible_at_function_exit
            event_ref = "function exit"
        else:
            path = j.node_ids.get(access.node_ref, access.node_ref)
            visible = visible_before.get(path)
            if visible is None:
                # Ordinary reference checking owns unknown node handles.
                continue
            event_ref = f"access node {access.node_ref!r}"

        expressions: list[tuple[str, Any]] = [
            (
                f"j.boundary.accesses[{index}].index_expression",
                component,
            )
            for component in parsed.index.components
        ]
        if parsed.predicate is not None:
            expressions.append(
                (
                    f"j.boundary.accesses[{index}].predicate",
                    parsed.predicate,
                )
            )
        for expression_path, expression in expressions:
            for value_ref in _ordered_expression_refs(expression):
                if value_ref not in all_values or value_ref in visible:
                    continue
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.BOUNDARY_VALUE_NOT_DOMINATING,
                        expression_path,
                        f"Layer-A value {value_ref!r} does not dominate "
                        f"the {event_ref}",
                        access.access_id,
                    )
                )


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
    for relation in j.value_dimension_relations:
        semantic_refs.add(relation.value_ref)
        semantic_refs.add(relation.dimension_ref)

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
        add(
            tensor.name,
            _Anchor("j", "tensor", value_type, carrier_class="value"),
            "j.signature",
        )
        for dim in tensor.shape:
            if isinstance(dim, str):
                add(
                    dim,
                    _Anchor("j", "dimension", _ValueType("index", 0)),
                    "j.signature",
                )

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
                carrier_class=(
                    "pointer"
                    if parameter.kind is ParameterKind.POINTER
                    else "value"
                    if parameter.kind is not ParameterKind.OPAQUE
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
        anchor = _Anchor(
            "j", "node", result_type, operand_types, carrier_class="value"
        )
        add(node_id, anchor, f"j.node_ids.{node_id}")
        add(path, anchor, f"j.node_ids.{node_id}")
        node_operations[path] = operation
    for value, value_type in value_types.items():
        if value not in anchors:
            add(
                value,
                _Anchor("j", "value", value_type, carrier_class="value"),
                "j.intent",
            )

    for node_ref, operation in node_operations.items():
        if not re.fullmatch(r"n[0-9]+", node_ref) or operation.get("op") != "loop":
            continue
        carries = operation.get("carry")
        if not isinstance(carries, list):
            continue
        for ordinal, carry in enumerate(carries):
            if not isinstance(carry, Mapping):
                continue
            state_ref = carry.get("state_id")
            type_value = carry.get("type")
            if not isinstance(state_ref, str) or not isinstance(type_value, Mapping):
                continue
            dtype = type_value.get("dtype")
            shape = type_value.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                continue
            add(
                state_ref,
                _Anchor(
                    "j",
                    "loop_state",
                    _ValueType(dtype, len(shape)),
                    carrier_class="value",
                ),
                f"j.node_ids.{node_ref}.carry[{ordinal}]",
            )
            carry_type = _ValueType(dtype, len(shape))
            for role_name in ("body_argument", "next_value", "result_value"):
                role_ref = carry.get(role_name)
                if not isinstance(role_ref, str) or role_ref in anchors:
                    continue
                add(
                    role_ref,
                    _Anchor(
                        "j",
                        "value",
                        carry_type,
                        carrier_class="value",
                    ),
                    f"j.node_ids.{node_ref}.carry[{ordinal}].{role_name}",
                )

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
            _Anchor(
                "j", "access", value_type, arity, carrier_class="value"
            ),
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


_CONSUMER_EXTENSION_OPS = frozenset(
    {"map_unary", "map_binary", "select", "cast", "transpose", "pad"}
)
_CONSUMER_EXTENSION_FNS = frozenset({"sum", "max", "min"})


def _same_closed_scalar(left: Any, right: Any) -> bool:
    aliases = {
        "ZERO": "zero",
        "NEG_INF": "neg_inf",
        "-inf": "neg_inf",
        "POS_INF": "pos_inf",
        "+inf": "pos_inf",
        "inf": "pos_inf",
    }
    if isinstance(left, str) or isinstance(right, str):
        return aliases.get(str(left), str(left)) == aliases.get(str(right), str(right))
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and float(left) == float(right)
    )


def _is_consumer_identity(fn: str, dtype: str, value: Any) -> bool:
    if fn == "sum":
        return value == "ZERO" or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 0.0
        )
    if not _floating_dtype(dtype):
        return False
    if fn == "max":
        return isinstance(value, str) and value in {"NEG_INF", "-inf"}
    if fn == "min":
        return isinstance(value, str) and value in {"POS_INF", "+inf", "inf"}
    return False


def _consumer_dependency_path(
    j: SemanticArchive,
    *,
    operand: str,
    access_node_ref: str | None,
    access_tensor: str,
) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
    """Find only the J dependency path from one access to one operand.

    A node-bearing access is tied to that exact node.  A node-less ABI load is
    tied to its tensor leaf.  No tensor-name fallback is used for a node-bearing
    access, so a same-tensor sibling access cannot silently satisfy it.
    """

    intent = j.intent.to_json()["intent_function"]
    path_to_ref = {path: ref for ref, path in j.node_ids.items()}
    producers: dict[str, tuple[str | None, Mapping[str, Any]]] = {}
    for path, operation in _walk_operations(intent.get("body", [])):
        for result in _operation_results(operation.get("result")):
            producers[result] = (path_to_ref.get(path), operation)

    memo: dict[str, tuple[bool, tuple[Mapping[str, Any], ...]]] = {}
    active: set[str] = set()

    def visit(value: str) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
        if access_node_ref is None and value == access_tensor:
            return True, ()
        if value in memo:
            return memo[value]
        if value in active:
            return False, ()
        active.add(value)
        producer = producers.get(value)
        if producer is None:
            result = (False, ())
        else:
            node_ref, operation = producer
            if access_node_ref is not None and node_ref == access_node_ref:
                result = (True, (operation,))
            else:
                selected: list[Mapping[str, Any]] = []
                found = False
                for dependency in _operation_dependencies(operation):
                    child_found, child_path = visit(dependency)
                    if child_found:
                        found = True
                        selected.extend(child_path)
                if found:
                    selected.append(operation)
                result = (found, tuple(selected))
        active.remove(value)
        memo[value] = result
        return result

    return visit(operand)


def _consumer_extension_candidates(
    j: SemanticArchive,
    *,
    operand: str,
) -> tuple[
    dict[str, tuple[Any, tuple[Mapping[str, Any], ...]]],
    tuple[str, ...],
]:
    """Resolve access provenance, preferring exact access.node_ref evidence."""

    by_tensor: dict[
        str,
        list[tuple[Any, tuple[Mapping[str, Any], ...]]],
    ] = {}
    for access in j.boundary.accesses:
        if access.kind.value != "load" or access.predicate is None:
            continue
        found, path = _consumer_dependency_path(
            j,
            operand=operand,
            access_node_ref=access.node_ref,
            access_tensor=access.tensor,
        )
        if found:
            by_tensor.setdefault(access.tensor, []).append((access, path))

    selected: dict[str, tuple[Any, tuple[Mapping[str, Any], ...]]] = {}
    ambiguous: list[str] = []
    for tensor, candidates in by_tensor.items():
        node_bound = [item for item in candidates if item[0].node_ref is not None]
        if node_bound:
            for access, path in node_bound:
                selected[access.access_id] = (access, path)
            continue
        if len(candidates) != 1:
            ambiguous.append(tensor)
            continue
        access, path = candidates[0]
        selected[access.access_id] = (access, path)
    return selected, tuple(sorted(ambiguous))


def _validate_consumer_path(
    path_operations: tuple[Mapping[str, Any], ...],
    *,
    outside_value: Any,
    outside_dtype: str,
) -> str | None:
    for operation in path_operations:
        op_name = operation.get("op")
        if op_name not in _CONSUMER_EXTENSION_OPS:
            return (
                "consumer-domain provenance crosses unsupported J op "
                f"{op_name!r}"
            )
        if op_name != "pad":
            continue
        attrs = operation.get("attrs")
        operands = operation.get("operands")
        result_type = _explicit_type(operation)
        if (
            not isinstance(attrs, Mapping)
            or set(attrs) != {"lo", "hi", "value"}
            or not isinstance(operands, list)
            or len(operands) != 1
            or result_type is None
        ):
            return "pad provenance requires closed lo/hi/value and one typed operand"
        lo = attrs.get("lo")
        hi = attrs.get("hi")
        if (
            not isinstance(lo, list)
            or not isinstance(hi, list)
            or len(lo) != result_type.rank
            or len(hi) != result_type.rank
            or any(
                isinstance(item, bool) or not isinstance(item, (int, str))
                for item in [*lo, *hi]
            )
        ):
            return (
                "pad provenance requires closed affine lo/hi for every axis; "
                "their shape and non-negativity are checked by Layer-A TypeCheck"
            )
        if result_type.dtype != outside_dtype:
            return "pad result dtype differs from consumer outside dtype"
        if not _same_closed_scalar(attrs.get("value"), outside_value):
            return "pad value differs from the declared consumer outside value"
    return None


def _check_consumer_extensions(
    j: SemanticArchive,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check v4's unique per-consumer outside-domain authority."""

    value_types = j_elaboration(j).value_type_by_name
    accesses = {item.access_id: item for item in j.boundary.accesses}
    legacy_masked_reductions = {
        item.node_ref
        for item in j.boundary.reductions
        if item.masked_contribution is not None
    }
    declared = {
        (item.consumer_ref, item.operand_index, item.access_ref): item
        for item in j.boundary.consumer_extensions
    }
    required = {
        (item.consumer_ref, item.operand_index, item.access_ref)
        for item in functional_consumer_extension_requirements(j)
    }

    for index, extension in enumerate(j.boundary.consumer_extensions):
        path = f"j.boundary.consumer_extensions[{index}]"
        operation = node_operations.get(extension.consumer_ref)
        attrs = operation.get("attrs") if operation is not None else None
        fn = attrs.get("fn") if isinstance(attrs, Mapping) else None
        operands = operation.get("operands") if operation is not None else None
        if (
            operation is None
            or operation.get("op") not in {"reduce", "scan"}
            or fn not in _CONSUMER_EXTENSION_FNS
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.consumer_ref",
                    "consumer extension requires a sum/max/min reduce or scan",
                    extension.consumer_ref,
                )
            )
            continue
        if (
            not isinstance(operands, list)
            or extension.operand_index >= len(operands)
            or not isinstance(operands[extension.operand_index], str)
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.operand_index",
                    "consumer operand ordinal is outside the closed operand list",
                    str(extension.operand_index),
                )
            )
            continue
        operand = str(operands[extension.operand_index])
        axis_issue: str | None = None
        try:
            local_axis = resolve_consumer_local_tile_axis(
                j, extension.consumer_ref
            )
        except (NotImplementedError, ValueError) as exc:
            local_axis = None
            axis_issue = str(exc)
        if axis_issue is not None or extension.tile_axis != local_axis:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.tile_axis",
                    axis_issue
                    or "consumer extension tile axis differs from the consumer's "
                    "closed local axis",
                    str(extension.tile_axis),
                )
            )
        operand_type = value_types.get(operand)
        if (
            operand_type is None
            or extension.outside.dtype.casefold() != operand_type.dtype.casefold()
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.outside.dtype",
                    "outside dtype must equal the selected consumer operand dtype",
                    extension.outside.dtype,
                )
            )
        if not _is_consumer_identity(
            str(fn), extension.outside.dtype, extension.outside.value
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.outside.value",
                    "outside value is not the declared consumer's two-sided identity",
                    extension.consumer_ref,
                )
            )
        access = accesses.get(extension.access_ref)
        if access is None or access.kind.value != "load" or access.predicate is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    "consumer extension requires one predicate-bearing load access",
                    extension.access_ref,
                )
            )
            continue
        candidates, ambiguous = _consumer_extension_candidates(j, operand=operand)
        selected = candidates.get(extension.access_ref)
        if selected is None or access.tensor in ambiguous:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    "access provenance does not unambiguously reach this "
                    "consumer operand",
                    extension.access_ref,
                )
            )
            continue
        path_issue = _validate_consumer_path(
            selected[1],
            outside_value=extension.outside.value,
            outside_dtype=extension.outside.dtype,
        )
        if path_issue is not None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    path_issue,
                    extension.access_ref,
                )
            )
        if extension.consumer_ref in legacy_masked_reductions:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    path,
                    "masked_contribution and consumer extension are duplicate "
                    "authorities",
                    extension.consumer_ref,
                )
            )

    for consumer_ref, operand_index, access_ref in sorted(required - set(declared)):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.CONSUMER_EXTENSION_MISSING,
                "j.boundary.consumer_extensions",
                "predicate-bearing reduce/scan operand requires an explicit "
                "outside-domain identity relation",
                f"{consumer_ref}:{operand_index}:{access_ref}",
            )
        )


def _check_loop_carriers(
    pi: RealizationWitness,
    node_operations: Mapping[str, Mapping[str, Any]],
    anchors: Mapping[str, _Anchor],
    instance_anchors: Mapping[str, _Anchor],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check explicit physical loop-carrier slots without temporal DAG edges."""

    regions = {str(item["id"]): item for item in pi.regions}
    instances = {str(item["id"]): item for item in pi.instances}
    route_edges = {
        (str(item["source"]), str(item["target"])) for item in pi.routes
    }
    iteration_by_region = {
        str(item["region"]): str(item["loop_instance"])
        for item in pi.iterations
    }
    observable_mechanisms = {
        "transfer_direct",
        "transfer_async",
        "update_atomic",
        "update_privatized",
        "synchronize",
    }
    outgoing: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for source, target in route_edges:
        outgoing.setdefault(source, set()).add(target)

    def ancestor_regions(region_ref: str) -> tuple[str, ...]:
        result: list[str] = []
        parent = regions.get(region_ref, {}).get("parent")
        while parent is not None:
            parent_ref = str(parent)
            result.append(parent_ref)
            parent = regions.get(parent_ref, {}).get("parent")
        return tuple(result)

    def reaches_observable_consumer(state_instance: str) -> bool:
        pending = list(outgoing.get(state_instance, ()))
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            instance = instances.get(current)
            if (
                instance is not None
                and instance.get("mechanism") in observable_mechanisms
            ):
                return True
            pending.extend(outgoing.get(current, ()))
        return False

    carry_owner: dict[str, tuple[str, int, Mapping[str, Any]]] = {}
    for loop_ref, operation in node_operations.items():
        if not re.fullmatch(r"n[0-9]+", loop_ref) or operation.get("op") != "loop":
            continue
        raw_carries = operation.get("carry")
        if not isinstance(raw_carries, list):
            continue
        carries = [item for item in raw_carries if isinstance(item, Mapping)]
        for ordinal, carry in enumerate(carries):
            for key in ("state_id", "body_argument", "next_value", "result_value"):
                ref = carry.get(key)
                if isinstance(ref, str):
                    carry_owner[ref] = (loop_ref, ordinal, carry)

    state_slots: set[str] = set()
    result_slots: set[str] = set()
    claimed_j_carries: set[tuple[str, int]] = set()
    last_j_carry_ordinal: dict[tuple[str, str], int] = {}
    allowed_kinds = {
        "signature", "tensor", "loop_state", "value", "node", "access",
    }

    for ordinal, relation in enumerate(pi.loop_carriers):
        base = f"pi.loop_carriers[{ordinal}]"
        region_ref = str(relation["region"])
        region = regions.get(region_ref)
        ancestors = ancestor_regions(region_ref)
        if region is None or region.get("placement") != "loop":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.region",
                    "loop carrier requires one placement=loop region",
                    region_ref,
                )
            )

        loop_instance_ref = str(relation["loop_instance"])
        loop_instance = instances.get(loop_instance_ref)
        loop_semantic_ref = (
            str(loop_instance["semantic_ref"])
            if loop_instance is not None
            else ""
        )
        loop_operation = node_operations.get(loop_semantic_ref)
        loop_region = str(loop_instance.get("region", "")) if loop_instance else ""
        if (
            loop_operation is None
            or loop_operation.get("op") != "loop"
            or loop_region not in {region_ref, *ancestors}
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.loop_instance",
                    "loop_instance must visibly anchor one structured J loop",
                    loop_instance_ref,
                )
            )
        expected_iteration_loop = iteration_by_region.get(region_ref)
        if (
            expected_iteration_loop is not None
            and expected_iteration_loop != loop_instance_ref
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.loop_instance",
                    "loop carrier and iteration relation disagree for this region",
                    loop_instance_ref,
                )
            )

        role_ids = {
            role: str(relation[role])
            for role in ("init", "state", "next", "result")
        }
        state_id = role_ids["state"]
        result_id = role_ids["result"]
        if state_id in state_slots:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.state",
                    "loop carrier state slots must be unique",
                    state_id,
                )
            )
        if result_id in result_slots:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.result",
                    "loop carrier result slots must be unique",
                    result_id,
                )
            )
        state_slots.add(state_id)
        result_slots.add(result_id)

        role_instances = {role: instances.get(ref) for role, ref in role_ids.items()}
        init_region = (
            str(role_instances["init"].get("region"))
            if role_instances["init"] is not None else ""
        )
        state_region = (
            str(role_instances["state"].get("region"))
            if role_instances["state"] is not None else ""
        )
        next_region = (
            str(role_instances["next"].get("region"))
            if role_instances["next"] is not None else ""
        )
        result_region = (
            str(role_instances["result"].get("region"))
            if role_instances["result"] is not None else ""
        )
        direct_parent = (
            str(region.get("parent"))
            if region and region.get("parent") is not None
            else ""
        )
        if init_region not in ancestors:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.init",
                    "init must be visible from a strict ancestor region",
                    role_ids["init"],
                )
            )
        if state_region != region_ref or next_region != region_ref:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.state",
                    "state and next must be instances in the exact loop region",
                    f"{role_ids['state']}:{role_ids['next']}",
                )
            )
        if not direct_parent or result_region != direct_parent:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.result",
                    "result must be an instance in the loop region's direct parent",
                    role_ids["result"],
                )
            )

        role_anchors = {
            role: instance_anchors.get(ref) for role, ref in role_ids.items()
        }
        typed_keys: list[tuple[str, int, str]] = []
        for role, anchor in role_anchors.items():
            if (
                anchor is None
                or anchor.owner != "j"
                or anchor.kind not in allowed_kinds
                or anchor.value_type is None
                or anchor.carrier_class not in {"pointer", "value"}
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        f"{base}.{role}",
                        "carrier role must anchor a typed J carrier, not C/domain/"
                        "scope/storage/effect/authorization/absent opaque",
                        role_ids[role],
                    )
                )
                continue
            instance = role_instances[role]
            semantic_ref = (
                str(instance.get("semantic_ref")) if instance is not None else ""
            )
            if anchor.kind == "node":
                operation = node_operations.get(semantic_ref)
                if operation is None or len(
                    _operation_results(operation.get("result"))
                ) != 1:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                            f"{base}.{role}",
                            "node carrier roles require one unambiguous typed result",
                            role_ids[role],
                        )
                    )
                    continue
            typed_keys.append(
                (
                    anchor.value_type.dtype.casefold(),
                    anchor.value_type.rank,
                    anchor.carrier_class,
                )
            )
        if typed_keys and len(set(typed_keys)) != 1:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    base,
                    "init/state/next/result dtype, rank, and carrier class must match",
                    state_id,
                )
            )

        if (role_ids["init"], state_id) not in route_edges:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.init",
                    "ordinary pi routes must contain init -> state",
                    role_ids["init"],
                )
            )
        if state_id != role_ids["next"] and (
            state_id, role_ids["next"]
        ) not in route_edges:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.next",
                    "ordinary pi routes must contain state -> next; identity "
                    "state==next is the only route-free update",
                    role_ids["next"],
                )
            )
        if not reaches_observable_consumer(state_id):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    f"{base}.state",
                    "loop state must reach an observable realization consumer",
                    state_id,
                )
            )

        state_instance = role_instances["state"]
        state_semantic_ref = (
            str(state_instance.get("semantic_ref"))
            if state_instance is not None else ""
        )
        carry_match = carry_owner.get(state_semantic_ref)
        touched_carries = {
            carry_owner[str(instance.get("semantic_ref"))][:2]
            for instance in role_instances.values()
            if instance is not None
            and str(instance.get("semantic_ref")) in carry_owner
        }
        if carry_match is None and touched_carries:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                    base,
                    "a relation touching J carry members must use that carry's "
                    "state slot as its state role",
                    state_id,
                )
            )
        if carry_match is not None:
            carry_loop_ref, carry_ordinal, carry = carry_match
            claim = (carry_loop_ref, carry_ordinal)
            if state_semantic_ref not in {
                str(carry.get("state_id", "")),
                str(carry.get("body_argument", "")),
            }:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        f"{base}.state",
                        "state role may reuse only the J carry state slot or "
                        "current-iteration body argument",
                        state_semantic_ref,
                    )
                )
            if carry_loop_ref != loop_semantic_ref:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        f"{base}.state",
                        "J carry state belongs to a different structured loop",
                        state_semantic_ref,
                    )
                )
            if claim in claimed_j_carries:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        f"{base}.state",
                        "one J carry ordinal cannot be duplicated as a second "
                        "accumulator",
                        state_semantic_ref,
                    )
                )
            claimed_j_carries.add(claim)
            ordinal_key = (region_ref, carry_loop_ref)
            previous_ordinal = last_j_carry_ordinal.get(ordinal_key)
            if previous_ordinal is not None and carry_ordinal <= previous_ordinal:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        base,
                        "declaration-ordered physical carrier ordinals must "
                        "preserve J carry ordinal order",
                        state_semantic_ref,
                    )
                )
            last_j_carry_ordinal[ordinal_key] = carry_ordinal
            role_semantics = {
                role: (
                    str(instance.get("semantic_ref"))
                    if instance is not None else ""
                )
                for role, instance in role_instances.items()
            }
            init_value = carry.get("init_value")
            expected_init = (
                str(init_value)
                if isinstance(init_value, str) and init_value in anchors
                else str(carry.get("state_id", ""))
            )
            expected = {
                "init": expected_init,
                "next": str(carry.get("next_value", "")),
                "result": str(carry.get("result_value", "")),
            }
            for role, expected_ref in expected.items():
                if role_semantics[role] == expected_ref:
                    continue
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.LOOP_CARRIER_INVALID,
                        f"{base}.{role}",
                        "physical slot must reuse the corresponding J carry "
                        "init/next/result instead of encoding a second accumulator",
                        role_ids[role],
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
    _check_dimension_and_pad_contracts(j, diagnostics)
    _check_value_dimension_relations(j, diagnostics)
    _check_total_j_value_typing(j, diagnostics)
    _check_rank_zero_logical_input_liveness(j, diagnostics)
    _check_boundary_value_dominance(j, diagnostics)
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
    _check_consumer_extensions(j, node_operations, diagnostics)
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
    _check_loop_carriers(
        pi,
        node_operations,
        anchors,
        instance_anchors,
        diagnostics,
    )

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
        if authorization.kind in {
            RefinementKind.SIGNED_WIDEN,
            RefinementKind.UNSIGNED_WIDEN,
        }:
            source_match = _INTEGER_DTYPE.fullmatch(
                authorization.source_dtype or ""
            )
            target_match = _INTEGER_DTYPE.fullmatch(
                authorization.target_dtype or ""
            )
            expected_signedness = (
                "i"
                if authorization.kind is RefinementKind.SIGNED_WIDEN
                else "u"
            )
            valid_widen = (
                source_match is not None
                and target_match is not None
                and source_match.group("signedness") == expected_signedness
                and target_match.group("signedness") == expected_signedness
                and int(target_match.group("bits"))
                > int(source_match.group("bits"))
            )
            if not valid_widen:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_INVALID_WIDEN,
                        f"j.boundary.realization_authorizations[{index}]",
                        f"{authorization.kind.value} requires explicit "
                        "same-signedness integer source/target dtype with "
                        "strictly increasing width",
                        authorization.auth_id,
                    )
                )
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
