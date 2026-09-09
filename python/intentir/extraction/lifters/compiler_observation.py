"""Exact compiler observation exposed only to the source-aware Lift owner.

This module deliberately stops at compilation and the existing Gate1 observer.
It never interprets the observation as J/C/pi and never constructs artifact
fields.  The same frozen anonymous Source specialization is compiled once and
stored in the branch-local read-only Lift input tree for focused queries.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping

from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.ttir import TTIRFingerprint, fingerprint_ttir
from intentir.verify.gate1.replay import compiled_packet_obligation_diagnostics
from intentir.verify.gate1.pi import NORMALIZATION_WITNESS_SCHEMA


SOURCE_TTIR_OBSERVATION_PROTOCOL = "intentir.lift-source-ttir.v1"
SOURCE_TTIR_RECONCILE_PROTOCOL = "intentir.lift-source-reconcile.v1"

_DTYPE_ALIASES = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def _runtime_argument(value: Mapping[str, Any]) -> RuntimeArgument:
    kind = value.get("kind")
    dtype_value = value.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value:
        raise ValueError("anonymous compiler argument requires a dtype")
    dtype = _DTYPE_ALIASES.get(dtype_value, dtype_value)
    if kind == "tensor":
        shape = value.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ):
            raise ValueError("anonymous tensor argument requires an integer shape")
        return RuntimeArgument.tensor(tuple(shape), dtype)
    if kind == "scalar":
        scalar = value.get("value")
        if isinstance(scalar, bool) or isinstance(scalar, (int, float)):
            return RuntimeArgument.scalar(scalar, dtype)
        raise ValueError("anonymous scalar argument requires a numeric value")
    if kind == "none":
        return RuntimeArgument.none(dtype)
    raise ValueError(
        "Source TTIR observation supports tensor/scalar/none compiler arguments"
    )


def _compile_case(view: Mapping[str, Any]) -> CompileCase:
    raw_arguments = view.get("arguments")
    if not isinstance(raw_arguments, list) or not raw_arguments:
        raise ValueError("anonymous compiler view requires runtime arguments")
    arguments: dict[str, RuntimeArgument] = {}
    for raw in raw_arguments:
        if not isinstance(raw, Mapping):
            raise TypeError("anonymous compiler arguments must be objects")
        name = raw.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("anonymous compiler argument requires an identifier")
        if name in arguments:
            raise ValueError("anonymous compiler argument names must be unique")
        arguments[name] = _runtime_argument(raw)

    raw_grid = view.get("grid")
    if not isinstance(raw_grid, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_grid
    ):
        raise ValueError("anonymous compiler view requires an integer grid")

    raw_compiler_slots = view.get("compiler_slots", [])
    raw_compiler_roles = view.get("compiler_roles", {})
    if not isinstance(raw_compiler_slots, list) or not all(
        isinstance(item, str) and item.isidentifier()
        for item in raw_compiler_slots
    ):
        raise TypeError("anonymous compiler_slots must be identifier names")
    if not isinstance(raw_compiler_roles, Mapping):
        raise TypeError("anonymous compiler_roles must be an object")
    compiler_slots = set(raw_compiler_slots)
    compiler_roles = {
        str(name): str(role) for name, role in raw_compiler_roles.items()
    }
    if set(compiler_roles) != compiler_slots or any(
        not role.isidentifier() for role in compiler_roles.values()
    ):
        raise ValueError("anonymous compiler slots require exact identifier roles")

    schedule: dict[str, bool | int | float | str] = {}
    for field in ("bound_constexpr_values", "schedule_values"):
        values = view.get(field, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"anonymous compiler view {field} must be an object")
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("anonymous constexpr names must be identifiers")
            compile_name = compiler_roles.get(name, name)
            if compile_name in schedule:
                raise ValueError("anonymous constexpr bindings must be disjoint")
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError("anonymous constexpr values must be JSON scalars")
            schedule[compile_name] = value
    return CompileCase(arguments, tuple(raw_grid), schedule)


def _source_runtime_parameter_uses(
    source: str,
    *,
    function_name: str,
    constexpr_values: Mapping[str, bool | int | float | str] | None = None,
) -> tuple[str, ...]:
    """Return exact callable-parameter reads before compiler specialization.

    This is syntax-only occurrence evidence.  It does not classify the use,
    assign J/C/pi ownership, or infer a route.
    """

    tree = ast.parse(source, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    parameters = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    static_env = dict(constexpr_values or {})
    unknown = object()

    def static_value(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return static_env.get(node.id, unknown)
        if isinstance(node, ast.UnaryOp):
            operand = static_value(node.operand)
            if operand is unknown:
                return unknown
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            if isinstance(node.op, ast.USub) and isinstance(operand, (int, float)):
                return -operand
            if isinstance(node.op, ast.UAdd) and isinstance(operand, (int, float)):
                return +operand
            if isinstance(node.op, ast.Invert) and isinstance(operand, int):
                return ~operand
            return unknown
        if isinstance(node, ast.BoolOp):
            values = [static_value(item) for item in node.values]
            if any(item is unknown for item in values):
                return unknown
            if isinstance(node.op, ast.And):
                return all(bool(item) for item in values)
            if isinstance(node.op, ast.Or):
                return any(bool(item) for item in values)
            return unknown
        if isinstance(node, ast.BinOp):
            left = static_value(node.left)
            right = static_value(node.right)
            if left is unknown or right is unknown:
                return unknown
            try:
                if isinstance(node.op, ast.BitAnd):
                    return left & right
                if isinstance(node.op, ast.BitOr):
                    return left | right
                if isinstance(node.op, ast.BitXor):
                    return left ^ right
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
            except (TypeError, ValueError, ZeroDivisionError):
                return unknown
            return unknown
        if isinstance(node, ast.Compare):
            left = static_value(node.left)
            comparators = [static_value(item) for item in node.comparators]
            if left is unknown or any(item is unknown for item in comparators):
                return unknown
            values = [left, *comparators]
            for index, operation in enumerate(node.ops):
                lhs, rhs = values[index], values[index + 1]
                try:
                    if isinstance(operation, ast.Eq):
                        passed = lhs == rhs
                    elif isinstance(operation, ast.NotEq):
                        passed = lhs != rhs
                    elif isinstance(operation, ast.Lt):
                        passed = lhs < rhs
                    elif isinstance(operation, ast.LtE):
                        passed = lhs <= rhs
                    elif isinstance(operation, ast.Gt):
                        passed = lhs > rhs
                    elif isinstance(operation, ast.GtE):
                        passed = lhs >= rhs
                    elif isinstance(operation, ast.Is):
                        passed = lhs is rhs
                    elif isinstance(operation, ast.IsNot):
                        passed = lhs is not rhs
                    else:
                        return unknown
                except TypeError:
                    return unknown
                if not passed:
                    return False
            return True
        return unknown

    class SelectedVariantUseVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.used: set[str] = set()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id in parameters:
                self.used.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            selected = static_value(node.test)
            if selected is unknown:
                bodies = (node.body, node.orelse)
            else:
                bodies = (node.body if bool(selected) else node.orelse,)
            for body in bodies:
                for statement in body:
                    self.visit(statement)

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.visit(node.test)
            selected = static_value(node.test)
            if selected is unknown:
                self.visit(node.body)
                self.visit(node.orelse)
            else:
                self.visit(node.body if bool(selected) else node.orelse)

    visitor = SelectedVariantUseVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return tuple(sorted(visitor.used))


def observe_anonymous_source_ttir(
    anonymous_source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    compiler: TritonCompiler | None = None,
) -> dict[str, Any]:
    """Compile one exact anonymous Source specialization and expose facts only.

    Failure is returned as a closed observation rather than being reclassified as
    a model failure.  The caller may require a passing observation for a formal
    capability run, while still retaining the exact compiler diagnostic.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_OBSERVATION_PROTOCOL,
        "authority": (
            "exact anonymous Source compilation and existing Gate1 observer only; "
            "no J/C/pi field, ownership decision, or repair is generated"
        ),
        "function": "scrc_k0",
    }
    if not isinstance(anonymous_source, str) or not anonymous_source:
        return {**base, "status": "fail", "diagnostic": "anonymous Source is absent"}
    if not isinstance(compiler_view, Mapping):
        return {
            **base,
            "status": "unavailable",
            "diagnostic": "anonymous compiler view is absent",
        }
    try:
        case = _compile_case(compiler_view)
        result = (compiler or TritonCompiler()).compile(
            KernelSource.from_code(anonymous_source, "scrc_k0", trusted=True),
            case,
        )
        if not result.ok or not isinstance(result.ttir, str) or not result.ttir:
            return {
                **base,
                "status": "fail",
                "diagnostic": result.diagnostic or "compiler returned no TTIR",
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }
        fingerprint = fingerprint_ttir(
            result.ttir,
            python_interface=result.python_interface,
            schedule=case.schedule,
        )
        return {
            **base,
            "status": "pass",
            "ttir": result.ttir,
            "fingerprint": fingerprint.to_dict(),
            "fingerprint_digest": fingerprint.digest,
            "source_runtime_parameter_uses": list(
                _source_runtime_parameter_uses(
                    anonymous_source,
                    function_name="scrc_k0",
                    constexpr_values=case.schedule,
                )
            ),
            "python_interface": dict(result.python_interface),
            "compiler": dict(result.compiler),
            "target": dict(result.target),
            "elapsed_ms": result.elapsed_ms,
            "cache_hit": result.cache_hit,
        }
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }


def reconcile_candidate_source_ttir(
    candidate_text: str,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Diagnose candidate-authored obligations against Source syntax and TTIR.

    This runs only inside source-aware Lift.  It neither constructs artifact
    fields nor contributes to TypeCheck, Gate1, or certification.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
        "authority": (
            "source-aware diagnostic comparison of model-authored closed J/compact-pi "
            "obligations with the existing anonymous Source TTIR observation; reject-only, "
            "no artifact construction and no Gate or certification authority"
        ),
    }
    if not isinstance(observation, Mapping) or observation.get("status") != "pass":
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    ttir = observation.get("ttir")
    if not isinstance(ttir, str) or not ttir:
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    try:
        packet = json.loads(candidate_text)
        if not isinstance(packet, Mapping):
            raise TypeError("candidate root is not an object")
        fingerprint = fingerprint_ttir(ttir)
        diagnostics = compiled_packet_obligation_diagnostics(
            packet,
            ttir,
            fingerprint=fingerprint,
            path_prefix="<source-ttir-obligation>",
        )
        diagnostics = (
            *diagnostics,
            *_compact_j_mask_presence_diagnostics(packet, fingerprint),
            *_compact_j_recurrence_presence_diagnostics(packet, fingerprint),
            *_compact_j_terminal_control_presence_diagnostics(packet, fingerprint),
            *_source_stride_axis_diagnostics(packet, fingerprint),
            *_unowned_source_runtime_carrier_diagnostics(
                packet,
                ttir,
                source_runtime_parameter_uses=observation.get(
                    "source_runtime_parameter_uses"
                ),
            ),
        )
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "applicable": True,
            "diagnostics": [
                {
                    "rule_code": "source_reconcile_observer_failed",
                    "path": "<source-ttir-obligation>",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [type(exc).__name__],
                    "line": None,
                }
            ],
        }
    return {
        **base,
        "status": "fail" if diagnostics else "pass",
        "applicable": True,
        "diagnostics": [dict(item) for item in diagnostics],
    }


def _compact_j_mask_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject total omission of Source-visible masks from an active J draft.

    This is deliberately a one-bit necessary condition, not an occurrence
    projection: frontend optimization may merge or eliminate transfers, so exact
    TTIR counts are not compared.  The Lift owner still chooses every predicate,
    fill, guard, consumer and formula; formal Gate1 checks the complete result.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    guards = (
        boundary.get("access_guard_relations")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(accesses, list) or not isinstance(guards, list):
        return ()

    masked_loads = [
        item
        for item in accesses
        if isinstance(item, Mapping)
        and item.get("kind") == "load"
        and item.get("predicate") is not None
    ]
    masked_store_ids = {
        str(item.get("access_id"))
        for item in accesses
        if isinstance(item, Mapping)
        and item.get("kind") == "store"
        and item.get("predicate") is not None
        and isinstance(item.get("access_id"), str)
    }
    masked_store_ids.update(
        str(item.get("access_ref"))
        for item in guards
        if isinstance(item, Mapping) and isinstance(item.get("access_ref"), str)
    )
    op_counts = dict(observed.op_multiset)
    diagnostics: list[Mapping[str, Any]] = []
    if op_counts.get("tt.load.masked", 0) > 0 and not masked_loads:
        diagnostics.append(
            {
                "rule_code": "j_masked_load_presence",
                "path": (
                    "<source-ttir-obligation> Source has compiler-visible masked "
                    "loads but J.boundary has no predicate-bearing load"
                ),
                "anonymous_handles": [],
                "line": None,
            }
        )
    if op_counts.get("tt.store.masked", 0) > 0 and not masked_store_ids:
        diagnostics.append(
            {
                "rule_code": "j_masked_store_presence",
                "path": (
                    "<source-ttir-obligation> Source has compiler-visible masked "
                    "stores but J.boundary has no predicate/guard-bearing store"
                ),
                "anonymous_handles": [],
                "line": None,
            }
        )
    return tuple(diagnostics)


def _j_operation_names(packet: Mapping[str, Any]) -> frozenset[str]:
    """Return explicitly authored J operation names without interpreting them."""

    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            operation = value.get("op")
            if isinstance(operation, str):
                names.add(operation)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    j = packet.get("j")
    if isinstance(j, Mapping):
        visit(j)
    return frozenset(names)


_NONLINEAR_RECURRENCE_OPS = frozenset(
    {
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.tanh",
    }
)


def _compact_j_recurrence_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject omission of an unmistakable nonlinear multi-state recurrence.

    The condition is intentionally weaker than recurrence reconstruction.  It says
    only that a Source loop has at least two carried values, at least two yielded
    values that are not direct pointer increments, and a nonlinear operation in its
    body.  Fixed code neither classifies the carries nor supplies trip/update/yield
    formulas; the Lift owner must author the complete structured loop and formal
    Gate1 remains authoritative.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    if "loop" in _j_operation_names(packet):
        return ()

    witnesses: list[tuple[int, tuple[str, ...]]] = []
    for ordinal, loop in enumerate(observed.loops):
        body_ops = dict(loop.body_op_multiset)
        non_pointer_yields = tuple(
            value
            for value in loop.yielded_values
            if not value.startswith("tt.addptr[")
        )
        nonlinear = tuple(
            sorted(
                operation
                for operation in _NONLINEAR_RECURRENCE_OPS
                if body_ops.get(operation, 0) > 0
            )
        )
        if (
            len(loop.carried_argument_use_counts) >= 2
            and len(non_pointer_yields) >= 2
            and nonlinear
        ):
            witnesses.append((ordinal, nonlinear))
    if not witnesses:
        return ()
    return (
        {
            "rule_code": "j_nonlinear_multistate_recurrence_presence",
            "path": (
                "<source-ttir-obligation> Source has a compiler-visible nonlinear "
                "multi-state loop recurrence but J contains no structured loop"
            ),
            "anonymous_handles": [],
            "observed_refs_or_types": [
                f"loop[{ordinal}]:{','.join(operations)}"
                for ordinal, operations in witnesses[:4]
            ],
            "line": None,
        },
    )


def _source_stride_axis_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject a stride binding whose axis contradicts Source address evidence.

    ``axis_ref`` is the logical tensor axis whose coordinate is multiplied by
    a representation-only stride carrier.  The existing schema checks only
    that the axis is in range, which is insufficient: a row-stride carrier can
    be validly attached to either axis of a rank-2 tensor while only one choice
    matches the compiled Source address polynomial.  This diagnostic is
    intentionally source-aware and necessary-only.  It uses the TTIR parser's
    excluded diagnostic address forms plus the candidate's already-authored
    program binding; it never constructs or repairs an artifact.

    The closed check currently applies when a carrier is visibly multiplied by
    one direct physical ``program_id`` axis.  More complex address forms are
    left to the formal Gate1/Gate2 path rather than guessed here.
    """

    pi = packet.get("pi")
    j = packet.get("j")
    if not isinstance(pi, Mapping) or not isinstance(j, Mapping):
        return ()
    raw_bindings = pi.get("access_representation_bindings")
    raw_program_bindings = pi.get("program_bindings")
    if not isinstance(raw_bindings, list) or not isinstance(
        raw_program_bindings, list
    ):
        return ()

    def axis_number(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str):
            match = re.fullmatch(r"x([0-9]+)", value)
            if match is not None:
                return int(match.group(1))
        return None

    # Map a physical launch axis to the unique logical tensor axis explicitly
    # named by the candidate's program binding.  Do not infer this from p*
    # order, shape, or the frontend's contiguous-layout convention.  Both the
    # integer and symbolic ``x*`` spellings are legal axis_ref encodings.
    program_axis_to_logical_axes: dict[int, set[int]] = {}
    for row in raw_program_bindings:
        if not isinstance(row, Mapping):
            continue
        physical_axis = row.get("axis")
        logical_axes = row.get("logical_axes")
        if (
            isinstance(physical_axis, bool)
            or not isinstance(physical_axis, int)
            or not isinstance(logical_axes, list)
        ):
            continue
        for logical in logical_axes:
            if not isinstance(logical, Mapping):
                continue
            logical_axis = axis_number(logical.get("axis_ref"))
            if logical_axis is not None:
                program_axis_to_logical_axes.setdefault(
                    physical_axis, set()
                ).add(logical_axis)
    if not program_axis_to_logical_axes:
        return ()

    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    tensor_to_parameter_position: dict[str, int] = {}
    if isinstance(raw_parameters, list):
        for parameter in raw_parameters:
            if not isinstance(parameter, Mapping):
                continue
            logical_ref = parameter.get("logical_ref")
            name = parameter.get("name")
            if (
                parameter.get("kind") == "tensor"
                and isinstance(logical_ref, str)
                and isinstance(name, str)
                and re.fullmatch(r"p[0-9]+", name)
            ):
                tensor_to_parameter_position[logical_ref] = int(name[1:])

    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses")
        if isinstance(boundary, Mapping)
        else None
    )
    accesses = {
        str(item.get("access_id")): item
        for item in raw_accesses or []
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }

    def argument_positions(polynomial: object) -> set[int]:
        positions: set[int] = set()
        if not isinstance(polynomial, (list, tuple)):
            return positions
        for term in polynomial:
            if not isinstance(term, (list, tuple)) or len(term) != 2:
                continue
            monomial = term[0]
            if not isinstance(monomial, (list, tuple)):
                continue
            for atom in monomial:
                if not isinstance(atom, str):
                    continue
                match = re.fullmatch(r"arg\[(\d+)\]:.*", atom)
                if match is not None:
                    positions.add(int(match.group(1)))
        return positions

    def program_axes(polynomial: object) -> set[int]:
        axes: set[int] = set()
        if not isinstance(polynomial, (list, tuple)):
            return axes
        for term in polynomial:
            if not isinstance(term, (list, tuple)) or len(term) != 2:
                continue
            monomial = term[0]
            if not isinstance(monomial, (list, tuple)):
                continue
            for atom in monomial:
                if not isinstance(atom, str):
                    continue
                match = re.fullmatch(r"program_axis:([xyz])", atom)
                if match is not None:
                    axes.add("xyz".index(match.group(1)))
        return axes

    def pointer_base_position(pointer_base: object) -> int | None:
        if not isinstance(pointer_base, str):
            return None
        match = re.fullmatch(r"arg\[(\d+)\]:.*", pointer_base)
        return int(match.group(1)) if match is not None else None

    source_forms = observed.diagnostic_memory_address_forms
    diagnostics: list[Mapping[str, Any]] = []
    for binding_index, binding in enumerate(raw_bindings):
        if not isinstance(binding, Mapping):
            continue
        carrier = binding.get("carrier_ref")
        access_ref = binding.get("access_ref")
        authored_axis = axis_number(binding.get("axis_ref"))
        if (
            not isinstance(carrier, str)
            or not isinstance(access_ref, str)
            or not isinstance(authored_axis, int)
        ):
            continue
        if not re.fullmatch(r"p[0-9]+", carrier):
            continue
        access = accesses.get(access_ref)
        if not isinstance(access, Mapping):
            continue
        tensor = access.get("tensor")
        base_position = tensor_to_parameter_position.get(tensor)
        if base_position is None:
            continue
        carrier_position = int(carrier[1:])
        access_kind = access.get("kind")
        expected_axes: set[int] = set()
        for operation, pointer_base, polynomial, _context in source_forms:
            if access_kind == "load" and operation != "tt.load.masked" and operation != "tt.load":
                continue
            if access_kind == "store" and operation != "tt.store.masked" and operation != "tt.store":
                continue
            if pointer_base_position(pointer_base) != base_position:
                continue
            positions = argument_positions(polynomial)
            if carrier_position not in positions:
                continue
            for physical_axis in program_axes(polynomial):
                expected_axes.update(
                    program_axis_to_logical_axes.get(physical_axis, ())
                )
        if len(expected_axes) != 1:
            continue
        expected_axis = next(iter(expected_axes))
        if authored_axis == expected_axis:
            continue
        diagnostics.append(
            {
                "rule_code": "source_stride_axis_mismatch",
                "path": (
                    "<source-ttir-obligation> "
                    f"pi.access_representation_bindings[{binding_index}].axis_ref"
                ),
                "anonymous_handles": [carrier, access_ref],
                "observed_refs_or_types": [
                    f"source-bound stride carrier {carrier} uses the direct launch axis",
                    f"candidate program binding resolves that launch axis to tensor axis {expected_axis}",
                    f"authored axis_ref={authored_axis}",
                ],
                "line": None,
            }
        )
    return tuple(diagnostics[:64])


def _compact_j_terminal_control_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject total omission of Source-visible terminal effect control."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    effectful_forks = tuple(
        fork
        for fork in observed.terminal_forks
        if fork.true_arm.effect_order and fork.false_arm.effect_order
    )
    if not effectful_forks or "effect_if" in _j_operation_names(packet):
        return ()
    return (
        {
            "rule_code": "j_terminal_effect_control_presence",
            "path": (
                "<source-ttir-obligation> Source has compiler-visible terminal "
                "effect control but J contains no effect_if"
            ),
            "anonymous_handles": [],
            "observed_refs_or_types": [
                f"two_sided_effectful_terminal_fork_count:{len(effectful_forks)}"
            ],
            "line": None,
        },
    )


def _unowned_source_runtime_carrier_diagnostics(
    packet: Mapping[str, Any],
    ttir: str,
    *,
    source_runtime_parameter_uses: object = None,
) -> tuple[Mapping[str, Any], ...]:
    """Check legacy graph-owned ABI carriers against exact Source use.

    A null ``logical_ref`` says that a callable carrier is not a J logical value;
    it does not authorize Replay to guess whether or where that carrier participates
    in Source realization.  If the exact Source specialization still uses such an
    argument, the Lift owner must expose its role through authored pi occurrences
    (or revise the J ownership).  This check observes only occurrence/non-occurrence:
    it does not choose either ownership or construct a route.  Active compact pi
    has no occurrence graph, so this entire legacy diagnostic is inapplicable.
    """

    pi = packet.get("pi")
    j = packet.get("j")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return ()
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    instances = pi.get("instances") or []
    if not isinstance(parameters, list) or not isinstance(instances, list):
        if pi.get("schema") != "intentir.pi.normalization-witness.v1":
            return ()

    pi_refs = {
        item.get("semantic_ref")
        for item in instances
        if isinstance(item, Mapping) and isinstance(item.get("semantic_ref"), str)
    }
    instance_refs = {
        str(item.get("id")): item.get("semantic_ref")
        for item in instances
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("semantic_ref"), str)
    }
    routes = pi.get("routes") or []
    routed_source_refs = {
        instance_refs.get(str(item.get("source")))
        for item in routes
        if isinstance(routes, list) and isinstance(item, Mapping)
    } if isinstance(routes, list) else set()
    compact_access_carriers = {
        item.get("carrier_ref")
        for item in pi.get("access_representation_bindings", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("carrier_ref"), str)
    }
    diagnostics: list[Mapping[str, Any]] = []
    source_syntax_uses = {
        item
        for item in source_runtime_parameter_uses
        if isinstance(item, str)
    } if isinstance(source_runtime_parameter_uses, list) else set()
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        kind = parameter.get("kind")
        logical_ref = parameter.get("logical_ref")
        if (
            not isinstance(name, str)
            or kind not in {"pointer", "scalar"}
            or logical_ref is not None
        ):
            continue
        # The function signature contributes exactly one occurrence.  A second
        # exact SSA-token occurrence means the compiled specialization consumes
        # this runtime argument in its body.  The token boundary excludes derived
        # names such as %p11_3.
        token = re.compile(rf"%{re.escape(name)}(?![A-Za-z0-9_.$])")
        source_used_in_ttir = len(token.findall(ttir)) > 1
        source_used_in_syntax = name in source_syntax_uses
        source_used = source_used_in_ttir or source_used_in_syntax
        observed_use = (
            "source-ttir-runtime-use"
            if source_used_in_ttir
            else "source-syntax-runtime-use"
        )
        pi_owned = name in pi_refs
        compact_owned = name in compact_access_carriers
        if pi.get("schema") == "intentir.pi.normalization-witness.v1":
            if source_used and not compact_owned:
                diagnostics.append(
                    {
                        "rule_code": "source_runtime_carrier_without_access_owner",
                        "path": (
                            "<source-ttir-obligation> "
                            "pi.access_representation_bindings: "
                            f"Source-used runtime carrier {name} has logical_ref=null "
                            "but no authored address ownership row"
                        ),
                        "anonymous_handles": [name],
                        "observed_refs_or_types": [observed_use],
                        "line": None,
                    }
                )
            elif compact_owned and not source_used:
                diagnostics.append(
                    {
                        "rule_code": "source_unused_runtime_carrier_has_access_owner",
                        "path": (
                            "<source-ttir-obligation> "
                            "pi.access_representation_bindings: "
                            f"runtime carrier {name} has an address ownership row, "
                            "but the selected Source has no body use"
                        ),
                        "anonymous_handles": [name],
                        "observed_refs_or_types": ["source-ttir-runtime-unused"],
                        "line": None,
                    }
                )
            continue
        if source_used and not pi_owned:
            diagnostics.append(
                {
                    "rule_code": "source_runtime_carrier_without_owner",
                    "path": (
                        "<source-ttir-obligation> j.signature.parameters: "
                        f"Source-used runtime carrier {name} has logical_ref=null "
                        "and no pi instance occurrence"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": [observed_use],
                    "line": None,
                }
            )
        elif source_used and name not in routed_source_refs:
            diagnostics.append(
                {
                    "rule_code": "source_runtime_carrier_unwired",
                    "path": (
                        "<source-ttir-obligation> pi.routes: "
                        f"Source-used runtime carrier {name} has logical_ref=null "
                        "and pi occurrence(s), but no authored outgoing route"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": [observed_use],
                    "line": None,
                }
            )
        elif pi_owned and not source_used:
            diagnostics.append(
                {
                    "rule_code": "source_unused_runtime_carrier_has_occurrence",
                    "path": (
                        "<source-ttir-obligation> pi.instances: "
                        f"runtime carrier {name} has logical_ref=null and a pi "
                        "occurrence, but the selected Source TTIR has no body use"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": ["source-ttir-runtime-unused"],
                    "line": None,
                }
            )
    return tuple(diagnostics[:64])


__all__ = [
    "SOURCE_TTIR_RECONCILE_PROTOCOL",
    "SOURCE_TTIR_OBSERVATION_PROTOCOL",
    "observe_anonymous_source_ttir",
    "reconcile_candidate_source_ttir",
]
