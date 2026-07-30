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
    direct_unit_dimension_refs,
    j_elaboration,
)
from intentir.ir.contract_ir import ContractIRProjectionError, project_contract_ir
from intentir.ir.dim_expr import (
    DimExprError,
    DimensionConstraintSet,
    parse_affine_dim_expr,
)
from intentir.ir.dimension_relation import (
    DimensionRelationError,
    validate_dimension_relations,
)
from intentir.ir.functional_boundary import (
    complete_conditional_effect_outputs,
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
    AccessGuardPolarity,
    AccessKind,
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    RefinementKind,
    ResultRelationKind,
    ScheduleSlotKind,
    SemanticArchive,
)
from intentir.verify.gate1.pi import RealizationWitness


TYPECHECK_PROTOCOL = "intentir.scrc.typecheck.v30"
GLOBAL_SAFE_REFINEMENTS = frozenset(
    {"identity_materialization", "reduction_linearization"}
)

_MAP_UNARY_FNS = frozenset(
    {
        "identity", "exp", "log", "log2", "neg", "abs", "sqrt", "rsqrt", "recip",
        "tanh", "sigmoid", "cos", "sin", "tan", "exp2", "erf", "atan",
        "logical_not", "isnan", "isinf", "isfinite",
    }
)
_MAP_BINARY_FNS = frozenset(
    {
        "add", "sub", "mul", "div", "max", "min", "pow", "cmp", "atan2",
        "gt", "ge", "lt", "le", "eq", "ne", "logical_and", "logical_or",
        "logical_xor", "isclose",
    }
)
_MAP_ATTRS = frozenset({"fn"})
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
    OUTPUT_BINDING_INCOMPLETE = "output_binding_incomplete"
    OUTPUT_BINDING_TYPE_MISMATCH = "output_binding_type_mismatch"
    KNOWN_OP_CONTRACT_INVALID = "known_op_contract_invalid"
    DUPLICATE_J_REF = "duplicate_j_ref"
    INVALID_C_SLOT = "invalid_c_slot"
    C_REQUIRED_VALUE_MISSING = "c_required_value_missing"
    C_VALUE_TYPE_MISMATCH = "c_value_type_mismatch"
    C_SUBJECT_UNKNOWN = "c_subject_unknown"
    C_AXIS_INVALID = "c_axis_invalid"
    INVALID_ACCESS_FILL = "invalid_access_fill"
    REDUCTION_BOUNDARY_INVALID = "reduction_boundary_invalid"
    FUNCTIONAL_BOUNDARY_INCOMPLETE = "functional_boundary_incomplete"
    PADDED_MEAN_COUNT_UNPROVEN = "padded_mean_count_unproven"
    LOGICAL_INPUT_UNUSED = "logical_input_unused"
    BOUNDARY_VALUE_NOT_DOMINATING = "boundary_value_not_dominating"
    ACCESS_GUARD_RELATION_INVALID = "access_guard_relation_invalid"
    EFFECT_CONDITIONAL_INVALID = "effect_conditional_invalid"
    VALUE_CONDITIONAL_INVALID = "value_conditional_invalid"
    REPRESENTATION_CARRIER_J_USE_FORBIDDEN = (
        "representation_carrier_j_use_forbidden"
    )
    RESULT_RELATION_INCOMPATIBLE = "result_relation_incompatible"
    UNRESOLVED_ANCHOR = "unresolved_anchor"
    ROUTE_CYCLE = "route_cycle"
    ROUTE_SLOT_OUT_OF_RANGE = "route_slot_out_of_range"
    ROUTE_TYPE_MISMATCH = "route_type_mismatch"
    ROUTE_EXTENT_TRANSFORM_INVALID = "route_extent_transform_invalid"
    PROGRAM_AXIS_BINDING_INVALID = "program_axis_binding_invalid"
    TRANSFER_ADDRESS_ROUTE_MISSING = "transfer_address_route_missing"
    INVALID_SHARING = "invalid_sharing"
    ITERATION_WITNESS_INVALID = "iteration_witness_invalid"
    ITERATION_WITNESS_MISSING = "iteration_witness_missing"
    CONSUMER_EXTENSION_INVALID = "consumer_extension_invalid"
    CONSUMER_EXTENSION_MISSING = "consumer_extension_missing"
    DIMENSION_CONSTRAINT_INVALID = "dimension_constraint_invalid"
    VALUE_DIMENSION_RELATION_INVALID = "value_dimension_relation_invalid"
    DIMENSION_RELATION_INVALID = "dimension_relation_invalid"
    PAD_SHAPE_INVALID = "pad_shape_invalid"
    PAD_SHAPE_CONSTRAINT_MISSING = "pad_shape_constraint_missing"
    TILE_SHAPE_INVALID = "tile_shape_invalid"
    LOOP_CARRIER_INVALID = "loop_carrier_invalid"
    PHYSICAL_ITERATION_INVALID = "physical_iteration_invalid"
    PHYSICAL_LOOP_CARRIER_INVALID = "physical_loop_carrier_invalid"
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
    REFINEMENT_TOPOLOGY_INVALID = "refinement_topology_invalid"
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
        if operation.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                arm_body = arm.get("body") if isinstance(arm, Mapping) else None
                if isinstance(arm_body, list):
                    yield from _walk_operations(
                        arm_body,
                        f"{path}.{polarity}.body",
                    )


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


def _check_dimension_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    try:
        validate_dimension_relations(j)
    except DimensionRelationError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.DIMENSION_RELATION_INVALID,
                "j.dimension_relations",
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
        if op_name not in {
            "const", "map_unary", "map_binary", "reduce", "contraction", "tile"
        }:
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}

        if op_name == "const":
            value = attrs.get("value")
            dtype_value = attrs.get("dtype")
            operation_type = operation.get("type")
            result_dtype = (
                operation_type.get("dtype")
                if isinstance(operation_type, Mapping)
                else None
            )
            dtype = dtype_value if isinstance(dtype_value, str) else result_dtype
            if isinstance(value, str) and (
                value not in {"-inf", "inf", "nan"}
                or not isinstance(dtype, str)
                or not _floating_dtype(dtype)
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.value",
                        "const string values are closed floating literals only: "
                        "'-inf', 'inf', or 'nan' with a floating dtype",
                        ref,
                    )
                )
            continue

        if op_name in {"map_unary", "map_binary"}:
            expected_fns = (
                _MAP_UNARY_FNS if op_name == "map_unary" else _MAP_BINARY_FNS
            )
            for key in sorted(set(attrs) - _MAP_ATTRS):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.{key}",
                        f"{op_name} attrs contain unsupported key {key!r}",
                        ref,
                    )
                )
            fn = attrs.get("fn")
            if (
                set(attrs) != _MAP_ATTRS
                or not isinstance(fn, str)
                or fn not in expected_fns
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.fn",
                        f"{op_name} requires exactly one closed fn; got {fn!r}",
                        ref,
                    )
                )
            continue

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
            if raw_operation.get("op") in {"effect_if", "value_if"}:
                predicate = raw_operation.get("predicate")
                if isinstance(predicate, str):
                    all_values.add(predicate)
                for polarity in ("positive", "negative"):
                    arm = raw_operation.get(polarity)
                    arm_body = (
                        arm.get("body") if isinstance(arm, Mapping) else None
                    )
                    if isinstance(arm_body, list):
                        # Branch-local definitions are collected for reference
                        # diagnostics but deliberately do not enter the parent
                        # continuation's visible set.
                        visit_sequence(
                            arm_body,
                            set(current),
                            f"{path}.{polarity}.body",
                        )
                    if (
                        raw_operation.get("op") == "value_if"
                        and isinstance(arm, Mapping)
                        and isinstance(arm.get("yield_value"), str)
                    ):
                        all_values.add(str(arm["yield_value"]))
            current.update(results)
        return current

    visible_at_function_exit = visit_sequence(operations, inputs, "body")
    elaboration = j_elaboration(j)
    visible_by_access: dict[str, tuple[frozenset[str], str]] = {}
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
        visible_by_access[access.access_id] = (visible, event_ref)

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

    for index, relation in enumerate(j.boundary.access_guard_relations):
        visibility = visible_by_access.get(relation.access_ref)
        if visibility is None:
            # Boundary/model reference closure owns an unknown access.
            continue
        visible, event_ref = visibility
        if relation.value_ref not in all_values or relation.value_ref in visible:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.BOUNDARY_VALUE_NOT_DOMINATING,
                f"j.boundary.access_guard_relations[{index}].value_ref",
                f"Layer-A value {relation.value_ref!r} does not dominate "
                f"the {event_ref}",
                relation.access_ref,
            )
        )


def _check_access_guard_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check the closed v5 access-local computed-guard contract.

    The relation is entirely J-owned.  This routine validates authored facts;
    it never copies a predicate, invents broadcast axes, inserts a ``select``,
    changes an output parameter into ``inout``, or consults pi.
    """

    if not j.boundary.access_guard_relations:
        return

    from intentir.frontend.emit_mlir import _broadcast_projection

    function = j.intent.to_json()["intent_function"]
    by_path: dict[str, Mapping[str, Any]] = {}
    loop_stack_by_path: dict[str, tuple[tuple[str, Mapping[str, Any]], ...]] = {}
    declared_values = {
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }

    def visit(
        operations: Any,
        prefix: str,
        loop_stack: tuple[tuple[str, Mapping[str, Any]], ...],
    ) -> None:
        if not isinstance(operations, list):
            return
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            by_path[path] = operation
            loop_stack_by_path[path] = loop_stack
            declared_values.update(_operation_results(operation.get("result")))
            if operation.get("op") != "loop":
                visit(operation.get("body"), f"{path}.body", loop_stack)
                continue
            scans = operation.get("scan")
            if isinstance(scans, list):
                declared_values.update(
                    item.get("name")
                    for item in scans
                    if isinstance(item, Mapping)
                    and isinstance(item.get("name"), str)
                )
            carries = operation.get("carry")
            if isinstance(carries, list):
                declared_values.update(
                    item.get("body_argument")
                    for item in carries
                    if isinstance(item, Mapping)
                    and isinstance(item.get("body_argument"), str)
                )
            visit(
                operation.get("body"),
                f"{path}.body",
                (*loop_stack, (path, operation)),
            )

    visit(function.get("body", []), "body", ())
    nodes = dict(by_path)
    for node_ref, path in j.node_ids.items():
        operation = by_path.get(path)
        if operation is not None:
            nodes[node_ref] = operation

    accesses = {item.access_id: item for item in j.boundary.accesses}
    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    input_names = {
        item.name for item in j.intent.inputs
    }
    output_names = {
        item.name for item in j.intent.outputs
    }
    output_values = {
        item.output: item.value for item in elaboration.output_bindings
    }
    parameters_by_logical: dict[str, list[Any]] = {}
    for parameter in j.signature.parameters:
        if parameter.logical_ref is not None:
            parameters_by_logical.setdefault(
                parameter.logical_ref, []
            ).append(parameter)

    seen_pairs: set[tuple[str, str]] = set()
    relations_by_access: dict[str, list[int]] = {}
    carry_relations: dict[tuple[str, int], list[int]] = {}

    def reject(
        index: int,
        field: str,
        detail: str,
        ref: str | None = None,
    ) -> None:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID,
                f"j.boundary.access_guard_relations[{index}].{field}",
                detail,
                ref,
            )
        )

    for index, relation in enumerate(j.boundary.access_guard_relations):
        pair = (relation.access_ref, relation.value_ref)
        if pair in seen_pairs:
            reject(
                index,
                "access_ref",
                "one (access_ref, value_ref) pair may be declared only once",
                relation.access_ref,
            )
        seen_pairs.add(pair)
        relations_by_access.setdefault(relation.access_ref, []).append(index)

        access = accesses.get(relation.access_ref)
        if access is None:
            # The strict Boundary model owns unknown access references.
            continue
        guard_type = value_types.get(relation.value_ref)
        if relation.value_ref not in declared_values or guard_type is None:
            reject(
                index,
                "value_ref",
                "value_ref must name a real typed Layer-A declaration",
                relation.value_ref,
            )
            continue
        if guard_type.dtype.casefold() != "i1":
            reject(
                index,
                "value_ref",
                "access guard value_ref must have canonical dtype i1",
                relation.value_ref,
            )

        axes = tuple(relation.axes)
        if axes != tuple(sorted(set(axes))):
            reject(
                index,
                "axes",
                "access guard axes must be strictly increasing and unique",
                relation.access_ref,
            )

        target_type = value_types.get(access.tensor)
        if target_type is None:
            reject(
                index,
                "access_ref",
                "access target must have one closed Layer-A tensor type",
                access.tensor,
            )
        else:
            try:
                _broadcast_projection(
                    list(guard_type.shape),
                    axes,
                    list(target_type.shape),
                    unit_dimensions=direct_unit_dimension_refs(
                        j.dimension_constraints
                    ),
                )
            except (NotImplementedError, ValueError) as exc:
                reject(
                    index,
                    "axes",
                    "guard shape cannot project to the access target using "
                    f"the authored broadcast axes: {exc}",
                    relation.access_ref,
                )

        parsed_access = elaboration.access_by_id.get(access.access_id)
        if (
            parsed_access is not None
            and parsed_access.predicate is not None
            and relation.value_ref
            in _ordered_expression_refs(parsed_access.predicate)
        ):
            reject(
                index,
                "value_ref",
                "the relation-bound computed guard must not also appear in "
                "the access predicate",
                relation.access_ref,
            )

        if access.kind is AccessKind.LOAD:
            if access.fill_value is None:
                reject(
                    index,
                    "access_ref",
                    "a relation-guarded load requires an explicit fill_value",
                    relation.access_ref,
                )
            continue
        if access.kind is AccessKind.UPDATE:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 has no closed update merge contract",
                relation.access_ref,
            )
            continue

        assert access.kind is AccessKind.STORE
        access_path = (
            j.node_ids.get(access.node_ref, access.node_ref)
            if access.node_ref is not None
            else None
        )
        loop_stack = (
            loop_stack_by_path.get(access_path, ())
            if access_path is not None
            else ()
        )
        observable_value = output_values.get(access.tensor, access.tensor)
        matches: list[tuple[str, int, Mapping[str, Any]]] = []
        for loop_path, loop in loop_stack:
            raw_carries = loop.get("carry")
            carries = raw_carries if isinstance(raw_carries, list) else []
            for ordinal, carry in enumerate(carries):
                if (
                    isinstance(carry, Mapping)
                    and carry.get("result_value") == observable_value
                ):
                    matches.append((loop_path, ordinal, carry))

        target_parameters = parameters_by_logical.get(access.tensor, [])
        has_real_inout = (
            access.tensor in input_names
            and access.tensor in output_names
            and len(target_parameters) == 1
            and target_parameters[0].mode is ParameterMode.INOUT
        )
        if not has_real_inout:
            reject(
                index,
                "access_ref",
                "a relation-guarded store must target one tensor declared as "
                "both J input and output and bound by exactly one mode=inout "
                "parameter; TypeCheck never upgrades output to inout",
                relation.access_ref,
            )

        operation = nodes.get(access.node_ref or "")
        operands = operation.get("operands") if operation is not None else None
        result = operation.get("result") if operation is not None else None
        if (
            operation is None
            or operation.get("op") != "select"
            or not isinstance(operands, list)
            or len(operands) != 3
            or not all(isinstance(item, str) for item in operands)
            or not isinstance(result, str)
        ):
            reject(
                index,
                "access_ref",
                "a relation-guarded store must anchor one explicit "
                "three-operand Layer-A select merge",
                relation.access_ref,
            )
            continue
        if operands[0] != relation.value_ref:
            reject(
                index,
                "value_ref",
                "the explicit select condition must equal relation.value_ref",
                relation.access_ref,
            )

        if loop_stack:
            if len(matches) != 1:
                reject(
                    index,
                    "access_ref",
                    "a relation-guarded store inside a structured loop must "
                    "map to exactly one enclosing carry result",
                    relation.access_ref,
                )
                continue
            loop_path, carry_ordinal, carry = matches[0]
            carry_relations.setdefault(
                (loop_path, carry_ordinal), []
            ).append(index)
            if carry.get("init_value") != access.tensor:
                reject(
                    index,
                    "access_ref",
                    "the guarded carry init_value must equal the exact inout "
                    "access tensor pre-state",
                    relation.access_ref,
                )
            if carry.get("next_value") != result:
                reject(
                    index,
                    "access_ref",
                    "carry.next_value must be the explicit select merge result",
                    relation.access_ref,
                )
            current = carry.get("body_argument")
        else:
            current = access.tensor
            matching_mappings = [
                item
                for item in j.boundary.output_mappings
                if item.output == access.tensor
                and item.value == result
                and item.node_ref is not None
                and nodes.get(item.node_ref) is operation
            ]
            if (
                output_values.get(access.tensor) != result
                or len(matching_mappings) > 1
            ):
                reject(
                    index,
                    "access_ref",
                    "a non-loop guarded store canonical output binding must "
                    "name the explicit select merge result exactly once",
                    relation.access_ref,
                )

        inactive_index = (
            2
            if relation.polarity is AccessGuardPolarity.POSITIVE
            else 1
        )
        active_index = 1 if inactive_index == 2 else 2
        if operands[inactive_index] != current:
            reject(
                index,
                "polarity",
                "the select branch inactive under the authored polarity must "
                "be the current carry body_argument",
                relation.access_ref,
            )
        if operands[active_index] == current:
            reject(
                index,
                "polarity",
                "the select branch active under the authored polarity must "
                "be the candidate update, not the current carry",
                relation.access_ref,
            )

        duplicate_guard_broadcasts = [
            item
            for item in j.boundary.broadcasts
            if item.operand == relation.value_ref
            and nodes.get(item.node_ref) is operation
        ]
        if duplicate_guard_broadcasts:
            reject(
                index,
                "axes",
                "access_guard_relations is the unique select-guard axes "
                "authority; boundary.broadcasts must not duplicate it",
                relation.access_ref,
            )

    for access_ref, indexes in sorted(relations_by_access.items()):
        if len(indexes) <= 1:
            continue
        for index in indexes:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 permits exactly one guard per access",
                access_ref,
            )
    for (_loop_path, _carry_ordinal), indexes in sorted(carry_relations.items()):
        if len(indexes) <= 1:
            continue
        for index in indexes:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 fails closed on multiple guarded "
                "writes to one loop carry",
                j.boundary.access_guard_relations[index].access_ref,
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
        if operation.get("op") == "contraction":
            attrs = operation.get("attrs")
            subscripts = attrs.get("subscripts") if isinstance(attrs, Mapping) else None
            operands = operation.get("operands")
            if not isinstance(subscripts, str) or not isinstance(operands, list):
                return frozenset()
            try:
                input_text, output_text = subscripts.split("->", 1)
                input_terms = input_text.split(",")
            except ValueError:
                return frozenset()
            if len(input_terms) != len(operands):
                return frozenset()
            contracted: list[str] = []
            for term in input_terms:
                for label in term:
                    if label not in output_text and label not in contracted:
                        contracted.append(label)
            result: set[int | str] = set(range(len(contracted)))
            value_types = j_elaboration(j).value_type_by_name
            for term, operand in zip(input_terms, operands, strict=True):
                value_type = value_types.get(operand) if isinstance(operand, str) else None
                if value_type is None or len(term) != len(value_type.shape):
                    continue
                for label, extent in zip(term, value_type.shape, strict=True):
                    if label in contracted and isinstance(extent, str) and extent in dimensions:
                        result.add(extent)
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
        "exp", "log", "log2", "sqrt", "rsqrt", "tanh", "sigmoid", "cos",
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
    if operation.get("op") in {"effect_if", "value_if"}:
        add(operation.get("predicate"))
    if operation.get("op") == "value_if":
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if isinstance(arm, Mapping):
                add(arm.get("yield_value"))
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
    for relation in j.dimension_relations:
        semantic_refs.add(relation.result)
        semantic_refs.add(relation.expr.arg)

    for node_ref in node_seed_refs:
        operation = nodes.get(node_ref)
        if operation is not None:
            semantic_refs.update(_operation_results(operation.get("result")))
    for operation in operations.values():
        if operation.get("op") in {"effect_if", "value_if"} and isinstance(
            operation.get("predicate"), str
        ):
            # A typed conditional predicate controls J-owned effects, so it is
            # semantic liveness rather than a representation-only pi use.
            semantic_refs.add(str(operation["predicate"]))
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


def _check_output_binding_types(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every canonical output destination to preserve full J type."""

    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    mapping_positions = {
        item.output: index
        for index, item in enumerate(j.boundary.output_mappings)
    }
    return_positions = {
        item.output: index for index, item in enumerate(j.intent.returns)
    }
    output_positions = {
        item.name: index for index, item in enumerate(j.intent.outputs)
    }
    outputs = {
        item.name: CanonicalValueType(item.dtype, tuple(item.shape))
        for item in j.intent.outputs
    }
    producible_values = {item.name for item in j.intent.inputs}
    intent = j.intent.to_json()["intent_function"]
    for _path, operation in _walk_operations(intent.get("body", [])):
        producible_values.update(_operation_results(operation.get("result")))
    try:
        effect_defined_outputs = complete_conditional_effect_outputs(j)
    except (NotImplementedError, ValueError):
        # Full functional projection owns this narrow proof.  If another boundary
        # capability is unsupported, retain the ordinary fail-closed output rule.
        effect_defined_outputs = frozenset()
    for binding in elaboration.output_bindings:
        if binding.value not in producible_values:
            if binding.output in effect_defined_outputs:
                continue
            if binding.output in mapping_positions:
                path = (
                    "j.boundary.output_mappings"
                    f"[{mapping_positions[binding.output]}].value"
                )
            elif binding.output in return_positions:
                path = (
                    "j.intent.intent_function.returns"
                    f"[{return_positions[binding.output]}].value"
                )
            else:
                path = (
                    "j.intent.intent_function.outputs"
                    f"[{output_positions[binding.output]}]"
                )
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.OUTPUT_BINDING_INCOMPLETE,
                    path,
                    "canonical output binding value is neither a J input nor "
                    "an authored operation result",
                    binding.output,
                )
            )
            continue
        expected = outputs[binding.output]
        actual = value_types.get(binding.value)
        if actual is None or actual == expected:
            continue
        if binding.output in mapping_positions:
            path = (
                "j.boundary.output_mappings"
                f"[{mapping_positions[binding.output]}].value"
            )
        elif binding.output in return_positions:
            path = (
                "j.intent.intent_function.returns"
                f"[{return_positions[binding.output]}].value"
            )
        else:
            path = (
                "j.intent.intent_function.outputs"
                f"[{output_positions[binding.output]}]"
            )
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.OUTPUT_BINDING_TYPE_MISMATCH,
                path,
                "canonical output binding type differs from its declared "
                f"output: expected={expected.to_json()}, actual={actual.to_json()}",
                binding.output,
            )
        )


_ITERATION_TILE_ROLES = frozenset(
    {"tile_extent", "reduction_tile_extent"}
)


def _iteration_axis_matches(axis_ref: int | str | None, trip_axis: str) -> bool:
    """Recognize the two AxisRefs_v1 spellings of one loop trip axis."""

    return axis_ref == 0 or axis_ref == trip_axis


def _check_effect_conditionals(
    j: SemanticArchive,
    pi: RealizationWitness,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check authored binary effect regions without constructing control.

    ``effect_if`` owns predicate polarity and lexical arms in Layer A.  The
    bound ``branch_region`` scopes own effect membership, while pi anchors each
    scope in an existing ``placement=branch`` region and may relate the two arm
    instances as terminal versus continuation topology.  This routine never
    inserts an arm, copies a predicate/polarity into pi, or rewrites the
    conditional to an SSA ``select``.
    """

    intent = j.intent.to_json()["intent_function"]
    conditionals: list[tuple[str, Mapping[str, Any], bool]] = []

    def collect_conditionals(body: Any, prefix: str) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            if operation.get("op") == "effect_if":
                conditionals.append((path, operation, index == len(body) - 1))
            collect_conditionals(operation.get("body"), f"{path}.body")
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if isinstance(arm, Mapping):
                    collect_conditionals(
                        arm.get("body"), f"{path}.{polarity}.body"
                    )

    collect_conditionals(intent.get("body", []), "body")
    if not conditionals:
        if pi.terminal_controls:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    "pi.terminal_controls",
                    "terminal control requires an authored J effect_if",
                )
            )
        return

    value_types = j_elaboration(j).value_type_by_name
    scopes = {scope.scope_id: scope for scope in j.scopes}
    scope_children: dict[str, set[str]] = {scope_id: set() for scope_id in scopes}
    for scope in j.scopes:
        if scope.parent_scope is not None:
            scope_children.setdefault(scope.parent_scope, set()).add(scope.scope_id)
    effects = {
        effect.effect_id: effect
        for effect in j.boundary.effects
        if effect.effect_id is not None
    }

    def scope_descendants(scope_id: str) -> set[str]:
        result: set[str] = set()
        pending = [scope_id]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(scope_children.get(current, ()))
        return result

    def nearest_branch_scope(scope_id: str | None) -> str | None:
        current = scope_id
        while current is not None:
            scope = scopes.get(current)
            if scope is None:
                return None
            if scope.kind.value == "branch_region":
                return current
            current = scope.parent_scope
        return None

    # Record the nearest authored branch owner for every semantic handle that
    # can later anchor pi.  This is a scope lookup, not inferred control: all
    # scope/polarity choices have already been authored by effect_if and J.scopes.
    path_to_node = {path: node for node, path in j.node_ids.items()}
    semantic_branch_owner: dict[str, str] = {}

    def collect_branch_handles(
        body: Any,
        *,
        owner_scope: str,
        prefix: str,
    ) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            branch_owner = nearest_branch_scope(owner_scope)
            if branch_owner is not None:
                node_ref = path_to_node.get(path)
                if node_ref is not None:
                    semantic_branch_owner[node_ref] = branch_owner
                for result in _operation_results(operation.get("result")):
                    semantic_branch_owner[result] = branch_owner
            if operation.get("op") == "loop":
                loop_scope = operation.get("scope_id")
                if isinstance(loop_scope, str):
                    loop_branch = nearest_branch_scope(loop_scope)
                    if loop_branch is not None:
                        for scan in operation.get("scan", []):
                            if isinstance(scan, Mapping) and isinstance(
                                scan.get("name"), str
                            ):
                                semantic_branch_owner[str(scan["name"])] = loop_branch
                        for carry in operation.get("carry", []):
                            if not isinstance(carry, Mapping):
                                continue
                            for key in (
                                "state_id",
                                "body_argument",
                                "next_value",
                                "result_value",
                            ):
                                if isinstance(carry.get(key), str):
                                    semantic_branch_owner[str(carry[key])] = loop_branch
                    collect_branch_handles(
                        operation.get("body"),
                        owner_scope=loop_scope,
                        prefix=f"{path}.body",
                    )
            elif operation.get("op") in {"effect_if", "value_if"}:
                for polarity in ("positive", "negative"):
                    arm = operation.get(polarity)
                    if not isinstance(arm, Mapping) or not isinstance(
                        arm.get("scope_id"), str
                    ):
                        continue
                    collect_branch_handles(
                        arm.get("body"),
                        owner_scope=str(arm["scope_id"]),
                        prefix=f"{path}.{polarity}.body",
                    )

    root_scope = next(
        (
            scope.scope_id
            for scope in j.scopes
            if scope.kind.value == "function" and scope.parent_scope is None
        ),
        "",
    )
    program_scope = next(
        (
            scope.scope_id
            for scope in j.scopes
            if scope.kind.value == "program_instance"
            and scope.parent_scope == root_scope
        ),
        root_scope,
    )
    collect_branch_handles(
        intent.get("body", []),
        owner_scope=program_scope,
        prefix="body",
    )
    for scope in j.scopes:
        branch_owner = nearest_branch_scope(scope.scope_id)
        if branch_owner is None:
            continue
        if scope.kind.value == "branch_region":
            semantic_branch_owner[scope.scope_id] = scope.scope_id
        for effect_ref in scope.effects:
            semantic_branch_owner[effect_ref] = branch_owner
            effect = effects.get(effect_ref)
            if effect is not None and effect.access_ref is not None:
                semantic_branch_owner[effect.access_ref] = branch_owner

    arm_records: list[tuple[str, str, str]] = []
    controlled_accesses: set[str] = set()
    for path, operation, _is_final in conditionals:
        base = f"j.intent.intent_function.{path}"
        predicate = operation.get("predicate")
        predicate_type = (
            value_types.get(predicate) if isinstance(predicate, str) else None
        )
        if (
            predicate_type is None
            or predicate_type.dtype.casefold() != "i1"
            or predicate_type.shape
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    f"{base}.predicate",
                    "effect_if predicate must be one dominating scalar i1 "
                    "Layer-A value",
                    predicate if isinstance(predicate, str) else None,
                )
            )

        pair_effect_owners: dict[str, str] = {}
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            scope_id = (
                arm.get("scope_id") if isinstance(arm, Mapping) else None
            )
            if not isinstance(scope_id, str):
                # The strict decoder owns this malformed wire shape.
                continue
            arm_records.append((path, polarity, scope_id))
            descendants = scope_descendants(scope_id)
            for descendant in descendants:
                scope = scopes.get(descendant)
                if scope is None:
                    continue
                for effect_ref in scope.effects:
                    previous = pair_effect_owners.get(effect_ref)
                    if previous is not None and previous != polarity:
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                                f"{base}.{polarity}.scope_id",
                                "one conditional effect cannot belong to both arms",
                                effect_ref,
                            )
                        )
                    pair_effect_owners[effect_ref] = polarity
                    effect = effects.get(effect_ref)
                    if effect is not None and effect.access_ref is not None:
                        controlled_accesses.add(effect.access_ref)
        if not pair_effect_owners:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "effect_if requires at least one J-owned effect across its "
                    "two explicit arms",
                )
            )

    guarded_accesses = {
        relation.access_ref for relation in j.boundary.access_guard_relations
    }
    for access_ref in sorted(controlled_accesses & guarded_accesses):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                "j.boundary.access_guard_relations",
                "effect_if branch control and access_guard_relation cannot "
                "both own one access predicate",
                access_ref,
            )
        )

    regions = {str(region["id"]): region for region in pi.regions}
    branch_regions = {
        region_id
        for region_id, region in regions.items()
        if region.get("placement") == "branch"
    }
    scope_anchor_regions: dict[str, str] = {}
    scope_anchor_instances: dict[str, str] = {}
    for path, polarity, scope_id in arm_records:
        matches = [
            instance
            for instance in pi.instances
            if instance.get("semantic_ref") == scope_id
            and instance.get("mechanism") == "realize"
        ]
        base = f"j.intent.intent_function.{path}.{polarity}.scope_id"
        if len(matches) != 1:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "each effect_if arm scope requires exactly one pi realize "
                    "instance anchor",
                    scope_id,
                )
            )
            continue
        region_ref = str(matches[0]["region"])
        region = regions.get(region_ref)
        if region is None or region.get("placement") != "branch":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "effect_if arm scope instance must live in one "
                    "placement=branch pi region",
                    scope_id,
                )
            )
            continue
        if region_ref in scope_anchor_regions.values():
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "one pi branch region cannot anchor multiple effect_if arms",
                    region_ref,
                )
            )
        scope_anchor_regions[scope_id] = region_ref
        scope_anchor_instances[scope_id] = str(matches[0]["id"])

    def region_is_within(region_ref: str, ancestor_ref: str) -> bool:
        current: str | None = region_ref
        seen: set[str] = set()
        while current is not None and current not in seen:
            if current == ancestor_ref:
                return True
            seen.add(current)
            parent = regions.get(current, {}).get("parent")
            current = str(parent) if parent is not None else None
        return False

    for scope_id, region_ref in scope_anchor_regions.items():
        parent_scope = scopes.get(scope_id).parent_scope if scope_id in scopes else None
        ancestor_scope = nearest_branch_scope(parent_scope)
        ancestor_region = (
            scope_anchor_regions.get(ancestor_scope)
            if ancestor_scope is not None else None
        )
        if ancestor_region is not None and not region_is_within(
            region_ref, ancestor_region
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    "pi.regions",
                    "nested effect_if branch region must remain inside its "
                    "authored parent branch",
                    region_ref,
                )
            )

    for index, instance in enumerate(pi.instances):
        semantic_ref = str(instance.get("semantic_ref"))
        owner_scope = semantic_branch_owner.get(semantic_ref)
        if owner_scope is None:
            continue
        owner_region = scope_anchor_regions.get(owner_scope)
        instance_region = str(instance.get("region"))
        if owner_region is not None and not region_is_within(
            instance_region, owner_region
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    f"pi.instances[{index}].region",
                    "branch-local semantic handle must be realized inside its "
                    "authored effect_if arm region",
                    semantic_ref,
                )
            )

    terminal_pairs: dict[
        frozenset[str], tuple[str, bool, dict[str, str]]
    ] = {}
    for path, operation, is_final in conditionals:
        arm_values = {
            polarity: operation.get(polarity)
            for polarity in ("positive", "negative")
        }
        scope_ids = {
            polarity: (
                arm.get("scope_id") if isinstance(arm, Mapping) else None
            )
            for polarity, arm in arm_values.items()
        }
        positive_region = scope_anchor_regions.get(str(scope_ids["positive"]))
        negative_region = scope_anchor_regions.get(str(scope_ids["negative"]))
        if positive_region is None or negative_region is None:
            continue
        positive_parent = regions[positive_region].get("parent")
        negative_parent = regions[negative_region].get("parent")
        if (
            positive_region == negative_region
            or positive_parent != negative_parent
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    f"j.intent.intent_function.{path}",
                    "positive/negative pi branch regions must be distinct "
                    "siblings with one common parent",
                )
            )

        arm_instances = {
            polarity: scope_anchor_instances.get(str(scope_ids[polarity]))
            for polarity in ("positive", "negative")
        }
        if all(isinstance(value, str) for value in arm_instances.values()):
            pair = frozenset(str(value) for value in arm_instances.values())
            terminal_pairs[pair] = (
                path,
                is_final,
                {key: str(value) for key, value in arm_instances.items()},
            )

    terminal_paths: set[str] = set()
    for index, relation in enumerate(pi.terminal_controls):
        base = f"pi.terminal_controls[{index}]"
        terminal_instance = str(relation["terminal_instance"])
        continuation_instance = str(relation["continuation_instance"])
        record = terminal_pairs.get(
            frozenset({terminal_instance, continuation_instance})
        )
        if record is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "terminal and continuation instances must anchor the two "
                    "sibling arms of one J effect_if",
                    terminal_instance,
                )
            )
            continue
        path, is_final, arm_instances = record
        if path in terminal_paths:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "one J effect_if may have at most one terminal-control relation",
                    terminal_instance,
                )
            )
        terminal_paths.add(path)
        if not is_final:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "terminal effect_if must be the final operation in its lexical "
                    "parent; the continuation belongs only to its sibling arm",
                    terminal_instance,
                )
            )
        if terminal_instance not in set(arm_instances.values()):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    f"{base}.terminal_instance",
                    "terminal instance is not an effect_if arm anchor",
                    terminal_instance,
                )
            )

    value_arm_scopes = {
        str(arm.get("scope_id"))
        for _path, operation in _walk_operations(intent.get("body", []))
        if operation.get("op") == "value_if"
        for arm in (operation.get("positive"), operation.get("negative"))
        if isinstance(arm, Mapping) and isinstance(arm.get("scope_id"), str)
    }
    value_anchor_regions = {
        str(instance.get("region"))
        for instance in pi.instances
        if instance.get("mechanism") == "realize"
        and str(instance.get("semantic_ref")) in value_arm_scopes
        and str(instance.get("region")) in branch_regions
    }
    anchored_regions = set(scope_anchor_regions.values()) | value_anchor_regions
    for region_ref in sorted(branch_regions - anchored_regions):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                "pi.regions",
                "placement=branch region is not anchored by an authored "
                "effect_if arm scope",
                region_ref,
            )
        )


_VALUE_IF_PURE_OPS = frozenset(
    {
        "map_binary", "map_unary", "reduce", "iota", "const",
        "contraction", "select", "cast", "transpose", "gather", "scatter",
        "concat", "pad", "tile", "flip", "triu", "rand", "scan",
        "arg_reduce", "sort", "pool", "upsample", "polar", "conv",
        "nonzero", "masked_select", "masked_scatter",
    }
)


def _check_value_conditionals(
    j: SemanticArchive,
    pi: RealizationWitness,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check authored single-result pure conditionals without constructing one."""

    intent = j.intent.to_json()["intent_function"]
    operations = dict(_walk_operations(intent.get("body", [])))
    conditionals = [
        (path, operation)
        for path, operation in operations.items()
        if operation.get("op") == "value_if"
    ]
    if not conditionals:
        return

    value_types = j_elaboration(j).value_type_by_name
    path_to_node = {path: node for node, path in j.node_ids.items()}
    node_to_path = {node: path for node, path in j.node_ids.items()}
    scopes = {scope.scope_id: scope for scope in j.scopes}
    scope_children: dict[str, set[str]] = {scope_id: set() for scope_id in scopes}
    for scope in j.scopes:
        if scope.parent_scope is not None:
            scope_children.setdefault(scope.parent_scope, set()).add(scope.scope_id)

    def descendants(scope_id: str) -> set[str]:
        result: set[str] = set()
        pending = [scope_id]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(scope_children.get(current, ()))
        return result

    regions = {str(region["id"]): region for region in pi.regions}
    branch_regions = {
        region_id
        for region_id, region in regions.items()
        if region.get("placement") == "branch"
    }

    def region_is_within(region_ref: str, ancestor_ref: str) -> bool:
        current: str | None = region_ref
        seen: set[str] = set()
        while current is not None and current not in seen:
            if current == ancestor_ref:
                return True
            seen.add(current)
            parent = regions.get(current, {}).get("parent")
            current = str(parent) if parent is not None else None
        return False

    # All authored conditional arms share one physical branch-region namespace.
    all_arm_scopes: set[str] = set()
    for _path, operation in operations.items():
        if operation.get("op") not in {"effect_if", "value_if"}:
            continue
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if isinstance(arm, Mapping) and isinstance(arm.get("scope_id"), str):
                all_arm_scopes.add(str(arm["scope_id"]))

    global_scope_regions: dict[str, str] = {}
    region_owners: dict[str, list[str]] = {}
    for scope_id in all_arm_scopes:
        matches = [
            instance
            for instance in pi.instances
            if instance.get("mechanism") == "realize"
            and instance.get("semantic_ref") == scope_id
        ]
        if len(matches) != 1:
            continue
        region_ref = str(matches[0].get("region"))
        if region_ref not in branch_regions:
            continue
        global_scope_regions[scope_id] = region_ref
        region_owners.setdefault(region_ref, []).append(scope_id)
    for region_ref, owners in sorted(region_owners.items()):
        if len(owners) > 1:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                    "pi.instances",
                    "one pi branch region cannot anchor multiple authored "
                    "conditional arms",
                    region_ref,
                )
            )

    local_owner: dict[str, str] = {}
    value_arm_scopes: set[str] = set()
    for path, operation in conditionals:
        base = f"j.intent.intent_function.{path}"
        predicate = operation.get("predicate")
        predicate_type = value_types.get(predicate) if isinstance(predicate, str) else None
        if (
            predicate_type is None
            or predicate_type.dtype.casefold() != "i1"
            or predicate_type.shape
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                    f"{base}.predicate",
                    "value_if predicate must be one dominating scalar i1 "
                    "Layer-A value",
                    predicate if isinstance(predicate, str) else None,
                )
            )

        result = operation.get("result")
        result_type = value_types.get(result) if isinstance(result, str) else None
        pair_regions: dict[str, str] = {}
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if not isinstance(arm, Mapping):
                continue
            arm_base = f"{base}.{polarity}"
            scope_id = arm.get("scope_id")
            if not isinstance(scope_id, str):
                continue
            value_arm_scopes.add(scope_id)
            scope = scopes.get(scope_id)
            for descendant in descendants(scope_id):
                descendant_scope = scopes.get(descendant)
                if descendant_scope is not None and descendant_scope.effects:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            f"{arm_base}.scope_id",
                            "value_if arms and descendant scopes must be pure "
                            "and contain no effects",
                            descendant,
                        )
                    )

            arm_prefix = f"{path}.{polarity}.body"
            arm_paths = {
                candidate
                for candidate in operations
                if candidate.startswith(f"{arm_prefix}[")
            }
            local_handles: set[str] = {scope_id}
            for arm_path in arm_paths:
                arm_operation = operations[arm_path]
                op_name = arm_operation.get("op")
                if op_name not in _VALUE_IF_PURE_OPS:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            f"j.intent.intent_function.{arm_path}.op",
                            "value_if arms permit only known pure ordinary "
                            "IntentIR operations and no nested regions",
                            str(op_name) if op_name is not None else None,
                        )
                    )
                local_handles.update(_operation_results(arm_operation.get("result")))
                node_ref = path_to_node.get(arm_path)
                if node_ref is not None:
                    local_handles.add(node_ref)
                local_owner.update({handle: scope_id for handle in local_handles})

            for access in j.boundary.accesses:
                access_path = (
                    node_to_path.get(access.node_ref, access.node_ref)
                    if access.node_ref is not None else None
                )
                if isinstance(access_path, str) and access_path in arm_paths:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            "j.boundary.accesses",
                            "value_if arms must not own logical accesses",
                            access.access_id,
                        )
                    )
            for effect in j.boundary.effects:
                effect_path = (
                    node_to_path.get(effect.node_ref, effect.node_ref)
                    if effect.node_ref is not None else None
                )
                if isinstance(effect_path, str) and effect_path in arm_paths:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            "j.boundary.effects",
                            "value_if arms must not own effects",
                            effect.effect_id,
                        )
                    )

            yield_value = arm.get("yield_value")
            yield_type = (
                value_types.get(yield_value)
                if isinstance(yield_value, str) else None
            )
            if result_type is None or yield_type is None or yield_type != result_type:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                        f"{arm_base}.yield_value",
                        "each value_if arm yield must have exactly the authored "
                        "single result dtype and shape",
                        yield_value if isinstance(yield_value, str) else None,
                    )
                )

            matches = [
                instance
                for instance in pi.instances
                if instance.get("mechanism") == "realize"
                and instance.get("semantic_ref") == scope_id
            ]
            if len(matches) != 1:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                        f"{arm_base}.scope_id",
                        "each value_if arm scope requires exactly one pi realize "
                        "instance anchor",
                        scope_id,
                    )
                )
                continue
            region_ref = str(matches[0].get("region"))
            if region_ref not in branch_regions:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                        f"{arm_base}.scope_id",
                        "value_if arm scope instance must live in one "
                        "placement=branch pi region",
                        scope_id,
                    )
                )
                continue
            pair_regions[polarity] = region_ref

        positive_region = pair_regions.get("positive")
        negative_region = pair_regions.get("negative")
        if positive_region is not None and negative_region is not None:
            if (
                positive_region == negative_region
                or regions[positive_region].get("parent")
                != regions[negative_region].get("parent")
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                        base,
                        "value_if positive/negative pi branch regions must be "
                        "distinct siblings with one common parent",
                    )
                )

    for index, instance in enumerate(pi.instances):
        semantic_ref = str(instance.get("semantic_ref"))
        owner_scope = local_owner.get(semantic_ref)
        owner_region = global_scope_regions.get(owner_scope or "")
        if owner_region is None:
            continue
        if not region_is_within(str(instance.get("region")), owner_region):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                    f"pi.instances[{index}].region",
                    "value_if arm-local semantic handle must be realized "
                    "inside its authored branch region",
                    semantic_ref,
                )
            )

    anchored_regions = set(global_scope_regions.values())
    for region_ref in sorted(branch_regions - anchored_regions):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                "pi.regions",
                "placement=branch region is not anchored by an authored "
                "effect_if/value_if arm scope",
                region_ref,
            )
        )


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


def _check_physical_iteration_witness(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check source-neutral physical time around an existing J subject/axis."""

    regions = {str(item["id"]): item for item in pi.regions}
    instances = {str(item["id"]): item for item in pi.instances}
    slots = {slot.name: slot for slot in c.slots}

    def visible_from(instance_region: str, relation_region: str) -> bool:
        current: str | None = relation_region
        while current is not None:
            if current == instance_region:
                return True
            parent = regions.get(current, {}).get("parent")
            current = str(parent) if parent is not None else None
        return False

    for index, relation in enumerate(pi.physical_iterations):
        base = f"pi.physical_iterations[{index}]"
        region_ref = str(relation["region"])
        region = regions.get(region_ref)
        if region is None or region.get("placement") != "loop":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_ITERATION_INVALID,
                    f"{base}.region",
                    "physical iteration requires one placement=loop region",
                    region_ref,
                )
            )

        body_instance_ref = str(relation["body_instance"])
        body_instance = instances.get(body_instance_ref)
        body_ref = (
            str(body_instance.get("semantic_ref"))
            if body_instance is not None
            else ""
        )
        body_operation = node_operations.get(body_ref)
        allowed_axes = _axis_refs_v1(j, body_ref, node_operations)
        body_valid = (
            body_instance is not None
            and bool(allowed_axes)
            and not (
                body_operation is not None and body_operation.get("op") == "loop"
            )
            and visible_from(str(body_instance.get("region")), region_ref)
        )
        if not body_valid:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_ITERATION_INVALID,
                    f"{base}.body_instance",
                    "body_instance must visibly anchor one existing non-loop J "
                    "reduce/contraction, boundary access, or domain with a closed "
                    "dependent axis; the relation never creates a semantic J loop",
                    body_instance_ref,
                )
            )

        tile_instance_ref = str(relation["tile_instance"])
        tile_instance = instances.get(tile_instance_ref)
        tile_ref = (
            str(tile_instance.get("semantic_ref"))
            if tile_instance is not None
            else ""
        )
        slot = slots.get(tile_ref)
        value = c.values.get(tile_ref)
        allowed_axes = allowed_axes if body_valid else frozenset()
        tile_valid = (
            tile_instance is not None
            and visible_from(str(tile_instance.get("region")), region_ref)
            and slot is not None
            and slot.subject_ref == body_ref
            and slot.axis_ref in allowed_axes
            and slot.required
            and slot.value_type.casefold()
            in {"int", "integer", "i32", "i64", "u32", "u64"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )
        if not tile_valid:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_ITERATION_INVALID,
                    f"{base}.tile_instance",
                    "tile_instance must visibly anchor one required positive-integer "
                    "C slot explicitly attached to the same J node and one of its "
                    "closed reduced/contracted axes",
                    tile_instance_ref,
                )
            )


def _check_route_extent_transforms(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    node_operations: Mapping[str, Mapping[str, Any]],
    instance_anchors: Mapping[str, _Anchor],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check closed C-extent partitions on existing realization route slots."""

    instances = {str(item["id"]): item for item in pi.instances}
    slots = {slot.name: slot for slot in c.slots}
    integer_types = {"int", "integer", "i32", "i64", "u32", "u64"}

    for index, relation in enumerate(pi.route_extent_transforms):
        base = f"pi.route_extent_transforms[{index}]"
        source_id = str(relation["source"])
        target_id = str(relation["target"])
        source_instance = instances.get(source_id)
        target_instance = instances.get(target_id)
        source_anchor = instance_anchors.get(source_id)
        target_anchor = instance_anchors.get(target_id)
        source_ref = (
            str(source_instance.get("semantic_ref"))
            if source_instance is not None
            else ""
        )
        target_ref = (
            str(target_instance.get("semantic_ref"))
            if target_instance is not None
            else ""
        )
        c_slot = slots.get(source_ref)
        c_value = c.values.get(source_ref)
        source_valid = (
            source_instance is not None
            and source_instance.get("mechanism") == "realize"
            and source_anchor is not None
            and source_anchor.owner == "c"
            and source_anchor.kind == "slot"
            and c_slot is not None
            and c_slot.required
            and c_slot.value_type.casefold() in integer_types
            and isinstance(c_value, int)
            and not isinstance(c_value, bool)
            and c_value > 0
        )
        if not source_valid:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ROUTE_EXTENT_TRANSFORM_INVALID,
                    f"{base}.source",
                    "route extent transform source must be one realize instance "
                    "anchored to a required positive-integer C slot",
                    source_id,
                )
            )

        allowed_axes = _axis_refs_v1(j, target_ref, node_operations)
        target_valid = (
            target_instance is not None
            and target_instance.get("mechanism") == "realize"
            and target_anchor is not None
            and target_anchor.owner == "j"
            and target_anchor.kind == "domain"
            and c_slot is not None
            and c_slot.subject_ref == target_ref
            and c_slot.axis_ref is not None
            and c_slot.axis_ref in allowed_axes
        )
        if not target_valid:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ROUTE_EXTENT_TRANSFORM_INVALID,
                    f"{base}.target",
                    "route extent transform target must be one realize instance "
                    "anchored to the J domain and axis attached by its source C slot",
                    target_id,
                )
            )


def _check_physical_loop_carriers(
    pi: RealizationWitness,
    node_operations: Mapping[str, Mapping[str, Any]],
    instance_anchors: Mapping[str, _Anchor],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check physical-only temporal carriers without inventing update semantics."""

    regions = {str(item["id"]): item for item in pi.regions}
    instances = {str(item["id"]): item for item in pi.instances}
    physical_bodies = {
        str(item["region"]): str(item["body_instance"])
        for item in pi.physical_iterations
    }
    route_edges = {
        (str(item["source"]), str(item["target"])) for item in pi.routes
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

    allowed_kinds = {"signature", "tensor", "value", "node", "access"}
    for index, relation in enumerate(pi.physical_loop_carriers):
        base = f"pi.physical_loop_carriers[{index}]"
        region_ref = str(relation["region"])
        region = regions.get(region_ref)
        ancestors = ancestor_regions(region_ref)
        if (
            region is None
            or region.get("placement") != "loop"
            or region_ref not in physical_bodies
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.region",
                    "physical carrier requires one declared physical_iterations "
                    "placement=loop region",
                    region_ref,
                )
            )

        role_ids = {
            role: str(relation[role])
            for role in ("init", "state", "next", "result")
        }
        role_instances = {role: instances.get(ref) for role, ref in role_ids.items()}
        role_regions = {
            role: (
                str(instance.get("region")) if instance is not None else ""
            )
            for role, instance in role_instances.items()
        }
        direct_parent = (
            str(region.get("parent"))
            if region is not None and region.get("parent") is not None
            else ""
        )
        if role_regions["init"] not in ancestors:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.init",
                    "physical carrier init must be visible from a strict ancestor",
                    role_ids["init"],
                )
            )
        if (
            role_regions["state"] != region_ref
            or role_regions["next"] != region_ref
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.state",
                    "physical carrier state and next must be in the exact loop region",
                    f"{role_ids['state']}:{role_ids['next']}",
                )
            )
        if not direct_parent or role_regions["result"] != direct_parent:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.result",
                    "physical carrier result must be in the loop region's direct parent",
                    role_ids["result"],
                )
            )

        typed_keys: list[tuple[str, int, str]] = []
        for role, instance_id in role_ids.items():
            anchor = instance_anchors.get(instance_id)
            if (
                anchor is None
                or anchor.owner != "j"
                or anchor.kind not in allowed_kinds
                or anchor.value_type is None
                or anchor.carrier_class not in {"pointer", "value"}
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                        f"{base}.{role}",
                        "physical carrier roles must anchor typed J carriers; C, "
                        "domain, scope, effect and absent carriers are forbidden",
                        instance_id,
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
                            TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                            f"{base}.{role}",
                            "physical node-carrier roles require one unambiguous "
                            "typed result",
                            instance_id,
                        )
                    )
                    continue
            typed_keys.append(
                (
                    anchor.value_type.dtype.casefold(),
                    anchor.value_type.rank,
                    str(anchor.carrier_class),
                )
            )
        if typed_keys and len(set(typed_keys)) != 1:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    base,
                    "physical init/state/next/result dtype, rank, and carrier "
                    "class must match",
                    role_ids["state"],
                )
            )
        if (role_ids["init"], role_ids["state"]) not in route_edges:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.init",
                    "ordinary pi routes must contain physical init -> state",
                    role_ids["init"],
                )
            )
        if role_ids["state"] != role_ids["next"] and (
            role_ids["state"], role_ids["next"]
        ) not in route_edges:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.next",
                    "ordinary pi routes must contain physical state -> next; "
                    "state==next is the only route-free update",
                    role_ids["next"],
                )
            )
        if not reaches_observable_consumer(role_ids["state"]):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PHYSICAL_LOOP_CARRIER_INVALID,
                    f"{base}.state",
                    "physical state must reach an observable realization consumer",
                    role_ids["state"],
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

    def path_score(
        path: tuple[Mapping[str, Any], ...],
    ) -> tuple[int, int, tuple[str, ...]]:
        names = tuple(str(operation.get("op")) for operation in path)
        return (
            sum(name not in _CONSUMER_EXTENSION_OPS for name in names),
            len(path),
            names,
        )

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
                candidates: list[tuple[Mapping[str, Any], ...]] = []
                for dependency in _operation_dependencies(operation):
                    child_found, child_path = visit(dependency)
                    if child_found:
                        candidates.append((*child_path, operation))
                result = (
                    (True, min(candidates, key=path_score))
                    if candidates
                    else (False, ())
                )
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


def _check_reduction_boundaries(
    j: SemanticArchive,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Keep Layer-B algebra contracts attached to semantic consumers.

    ``ReductionBoundary`` describes padding/algebra for an existing Layer-A
    reduce or inclusive scan.  It cannot manufacture a reduction for a loop or
    any other node.  Enforcing that ownership here prevents a physically
    matching Replay from hiding a missing semantic contraction in J.
    """

    seen: set[str] = set()
    for index, reduction in enumerate(j.boundary.reductions):
        path = f"j.boundary.reductions[{index}]"
        operation = node_operations.get(reduction.node_ref)
        if operation is None or operation.get("op") not in {"reduce", "scan"}:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REDUCTION_BOUNDARY_INVALID,
                    f"{path}.node_ref",
                    "reduction boundary must reference an existing semantic "
                    "reduce or inclusive scan node",
                    reduction.node_ref,
                )
            )
        if reduction.node_ref in seen:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REDUCTION_BOUNDARY_INVALID,
                    path,
                    "one semantic reduce/scan may have at most one reduction "
                    "boundary contract",
                    reduction.node_ref,
                )
            )
        seen.add(reduction.node_ref)


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


def _check_signature_mode_partition(
    j: SemanticArchive,
) -> tuple[TypeCheckDiagnostic, ...]:
    """Require the authored ABI mode to match J's logical boundary partition."""

    input_refs = {tensor.name for tensor in j.intent.inputs}
    output_refs = {tensor.name for tensor in j.intent.outputs}
    diagnostics: list[TypeCheckDiagnostic] = []
    for index, parameter in enumerate(j.signature.parameters):
        ref = parameter.logical_ref
        if ref is None:
            continue
        membership = (ref in input_refs, ref in output_refs)
        expected = {
            ParameterMode.INPUT: (True, False),
            ParameterMode.OUTPUT: (False, True),
            ParameterMode.INOUT: (True, True),
        }.get(parameter.mode)
        if expected is None or membership == expected:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.J_ELABORATION_FAILED,
                f"j.signature.parameters[{index}].mode",
                "signature mode must exactly match the logical tensor boundary "
                "partition: input binds input-only, output binds output-only, "
                "and an input/output overlap must be authored as inout",
                ref,
            )
        )
    return tuple(diagnostics)


def _check_transfer_address_routes(
    j: SemanticArchive,
    pi: RealizationWitness,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every direct/async access transfer to consume its tensor carrier.

    This is a dependent reference check, not address synthesis.  It does not
    choose address stages or their arithmetic; it only rejects a transfer whose
    authored route graph is disconnected from every signature carrier bound to
    that exact boundary tensor.
    """

    access_by_id = {item.access_id: item for item in j.boundary.accesses}
    pointer_parameters_by_tensor: dict[str, set[str]] = {}
    for parameter in j.signature.parameters:
        if (
            parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}
            and isinstance(parameter.logical_ref, str)
        ):
            pointer_parameters_by_tensor.setdefault(
                parameter.logical_ref, set()
            ).add(parameter.name)

    instances = {str(item["id"]): item for item in pi.instances}
    adjacency: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for route in pi.routes:
        adjacency.setdefault(str(route["source"]), set()).add(str(route["target"]))

    def reaches(source: str, target: str) -> bool:
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency.get(current, ()))
        return False

    for index, item in enumerate(pi.instances):
        if item["mechanism"] not in {"transfer_direct", "transfer_async"}:
            continue
        access = access_by_id.get(str(item["semantic_ref"]))
        if access is None:
            continue
        carrier_names = pointer_parameters_by_tensor.get(access.tensor, set())
        carrier_instances = {
            instance_id
            for instance_id, instance in instances.items()
            if str(instance["semantic_ref"]) in carrier_names
        }
        target = str(item["id"])
        if carrier_instances and any(
            reaches(source, target) for source in carrier_instances
        ):
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.TRANSFER_ADDRESS_ROUTE_MISSING,
                f"pi.instances[{index}]",
                "direct/async boundary transfer requires an authored route path "
                "from a pointer/tensor signature carrier bound to the same J tensor",
                target,
            )
        )


def _check_program_axes(
    j: SemanticArchive,
    pi: RealizationWitness,
    diagnostics: list[TypeCheckDiagnostic],
    *,
    required: bool,
) -> None:
    """Check the closed pi domain-to-launch-axis realization identity."""

    domains = set(j.boundary.domains)
    regions = {
        str(item["id"]): item for item in pi.regions
    }
    program_regions = {
        region_id
        for region_id, item in regions.items()
        if item.get("placement") == "program"
    }
    instances = {
        str(item["id"]): item for item in pi.instances
    }
    address_bases = {
        instance_id
        for instance_id, item in instances.items()
        if item.get("mechanism") == "address_base"
    }
    outgoing: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for route in pi.routes:
        outgoing.setdefault(str(route["source"]), set()).add(str(route["target"]))

    def reaches_address_base(source: str) -> bool:
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in address_bases:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(outgoing.get(current, ()))
        return False

    authorized = {
        str(item["semantic_ref"])
        for instance_id, item in instances.items()
        if item.get("mechanism") == "realize"
        and item.get("region") in program_regions
        and item.get("semantic_ref") in domains
        and reaches_address_base(instance_id)
    }
    bound = {str(item["domain_ref"]) for item in pi.program_axes}
    for index, item in enumerate(pi.program_axes):
        domain_ref = str(item["domain_ref"])
        if domain_ref not in domains:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                    f"pi.program_axes[{index}].domain_ref",
                    "program-axis binding must name a declared J boundary domain",
                    domain_ref,
                )
            )
        elif domain_ref not in authorized:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                    f"pi.program_axes[{index}].domain_ref",
                    "program-axis binding requires a program-region domain realize "
                    "path that reaches address_base",
                    domain_ref,
                )
            )
    if required:
        for domain_ref in sorted(authorized - bound):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                    "pi.program_axes",
                    "every authored program-coordinate domain requires one explicit "
                    "physical launch-axis binding",
                    domain_ref,
                )
            )
        for domain_ref in sorted(bound - authorized):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                    "pi.program_axes",
                    "program-axis bindings must equal the authored program-coordinate "
                    "domain set",
                    domain_ref,
                )
            )


def type_check(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    *,
    require_program_axes: bool = False,
) -> TypeCheckReport:
    """Mechanically check ``Gamma_(J,C) |- pi : RealizationWitness(J,C)``."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("type_check requires SemanticArchive J")
    if not isinstance(c, ImplementationConfig):
        raise TypeError("type_check requires canonical ImplementationConfig C")
    if not isinstance(pi, RealizationWitness):
        raise TypeError("type_check requires parsed RealizationWitness pi")

    signature_diagnostics = _check_signature_mode_partition(j)
    if signature_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, signature_diagnostics)

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

    # ContractIR is the target-neutral, one-to-one typed projection of every
    # authored Layer-B domain, index, and predicate.  Failure here is a J type
    # error, not a RefEval/backend capability decision.  In particular, an x*
    # dimension already has integer sort and cannot also stand alone as a boolean
    # domain predicate.  No Source, C value, pi topology, or target is consulted.
    try:
        project_contract_ir(j)
    except ContractIRProjectionError as exc:
        return TypeCheckReport(
            TypeCheckStatus.FAIL,
            (
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_ELABORATION_FAILED,
                    "j.boundary",
                    str(exc),
                ),
            ),
        )

    diagnostics: list[TypeCheckDiagnostic] = []
    _check_dimension_and_pad_contracts(j, diagnostics)
    _check_value_dimension_relations(j, diagnostics)
    _check_dimension_relations(j, diagnostics)
    _check_total_j_value_typing(j, diagnostics)
    _check_output_binding_types(j, diagnostics)
    _check_rank_zero_logical_input_liveness(j, diagnostics)
    _check_boundary_value_dominance(j, diagnostics)
    _check_access_guard_relations(j, diagnostics)
    _check_effect_conditionals(j, pi, diagnostics)
    _check_value_conditionals(j, pi, diagnostics)
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
    relation_guarded_accesses = {
        item.access_ref for item in j.boundary.access_guard_relations
    }
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
        if (
            access.predicate is None
            and access.access_id not in relation_guarded_accesses
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_ACCESS_FILL,
                    path,
                    "fill_value requires an authored access predicate or "
                    "access_guard_relation",
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
    _check_reduction_boundaries(j, node_operations, diagnostics)
    _check_consumer_extensions(j, node_operations, diagnostics)
    absent_signature_refs = {
        parameter.name
        for parameter in j.signature.parameters
        if parameter.kind is ParameterKind.OPAQUE and parameter.dtype == "none"
    }
    _check_result_relations(j, diagnostics)
    c_subject_kinds = {"node", "domain", "access", "scope", "storage"}
    for index, slot in enumerate(c.slots):
        if slot.kind is ScheduleSlotKind.COMPILER:
            if slot.subject_ref is not None or slot.axis_ref is not None:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.INVALID_C_SLOT,
                        f"c.slots[{index}]",
                        "compiler C slot is an external compiler-invocation "
                        "option and cannot attach to a J subject or axis",
                        slot.name,
                    )
                )
            continue
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
    compiler_slot_refs = {
        slot.name for slot in c.slots
        if slot.kind is ScheduleSlotKind.COMPILER
    }
    for index, item in enumerate(pi.instances):
        ref = str(item["semantic_ref"])
        if ref in compiler_slot_refs:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_C_SLOT,
                    f"pi.instances[{index}].semantic_ref",
                    "compiler C slot is consumed only by the external compiler "
                    "invocation and cannot own a pi realization instance",
                    ref,
                )
            )
            continue
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
    _check_physical_iteration_witness(j, c, pi, node_operations, diagnostics)
    _check_route_extent_transforms(
        j, c, pi, node_operations, instance_anchors, diagnostics
    )
    _check_loop_carriers(
        pi,
        node_operations,
        anchors,
        instance_anchors,
        diagnostics,
    )
    _check_physical_loop_carriers(
        pi, node_operations, instance_anchors, diagnostics
    )
    _check_program_axes(
        j,
        pi,
        diagnostics,
        required=require_program_axes,
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

    _check_transfer_address_routes(j, pi, diagnostics)

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
                        "global-safe representation materialization must not cite a "
                        "candidate-defined authorization",
                        str(auth_ref),
                    )
                )
            if kind == "reduction_linearization":
                outgoing = [
                    route
                    for route in pi.routes
                    if str(route["source"]) == instance_id
                ]
                valid_route = False
                if (
                    refined_instance is not None
                    and refined_instance.get("mechanism") == "realize"
                    and len(outgoing) == 1
                    and list(outgoing[0]["slots"]) == [0]
                ):
                    target = instances.get(str(outgoing[0]["target"]))
                    target_ref = (
                        str(target["semantic_ref"])
                        if target is not None
                        else ""
                    )
                    target_operation = node_operations.get(target_ref)
                    valid_route = (
                        target is not None
                        and target.get("mechanism") == "realize"
                        and target_operation is not None
                        and target_operation.get("op") == "reduce"
                    )
                if not valid_route:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.REFINEMENT_TOPOLOGY_INVALID,
                            f"pi.refinements[{index}]",
                            "reduction_linearization requires one refined realize "
                            "stage with exactly one slot-0 route into one realize "
                            "instance anchored to a J reduce",
                            instance_id,
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
