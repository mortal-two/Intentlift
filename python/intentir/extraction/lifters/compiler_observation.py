"""Exact compiler observation exposed only to the source-aware Lift owner.

This module deliberately stops at compilation and the existing Gate1 observer.
It never interprets the observation as J/C/pi and never constructs artifact
fields.  The same frozen anonymous Source specialization is compiled once and
stored in the branch-local read-only Lift input tree for focused queries.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import traceback
from typing import Any, Mapping

from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.ttir import TTIRFingerprint, fingerprint_ttir
from intentir.verify.gate1.replay import compiled_packet_obligation_diagnostics
from intentir.verify.gate1.pi import (
    NORMALIZATION_WITNESS_SCHEMA,
    ProgramCoordinateProjectionError,
    canonical_program_coordinate_projection,
)
from intentir.ir.canonical_construction import (
    CanonicalConstruction,
    SourceFacts,
    SourceTransferLedger,
    TransferOccurrence,
    coerce_source_transfer_ledger,
)
from intentir.ir.canonical_j import ContractSort, parse_contract_expr


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
        str(name): str(role)
        for name, role in raw_compiler_roles.items()
        if str(name) in compiler_slots
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


def _ttir_argument_categories(
    text: str,
    *,
    function_name: str,
) -> tuple[tuple[int, str, str], ...] | None:
    """Extract the closed category/order view of the observed TTIR ABI.

    Triton may remove specialized scalar carriers from a compiled function.
    The remaining ``argN`` names therefore cannot be treated as positions in
    the pre-specialization Source ABI.  This deliberately exposes no TTIR
    program text: it returns only the argument number and its coarse ABI
    category required to prove an order-preserving projection.
    """

    if not isinstance(function_name, str) or not function_name.isidentifier():
        return None
    match = re.search(
        rf"(?:^|\n)\s*(?:tt\.func|func\.func)\s+(?:public\s+)?@{re.escape(function_name)}\s*\(",
        text,
    )
    if match is None:
        return None
    start = match.end()
    depth = 1
    end = start
    while end < len(text) and depth:
        char = text[end]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        end += 1
    if depth or end <= start:
        return None
    signature = text[start : end - 1]
    markers = list(re.finditer(r"%arg([0-9]+)\s*:\s*", signature))
    if not markers:
        return None
    result: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for index, marker in enumerate(markers):
        position = int(marker.group(1))
        if position in seen:
            return None
        seen.add(position)
        type_text = signature[
            marker.end() : markers[index + 1].start() if index + 1 < len(markers) else len(signature)
        ].lstrip()
        if type_text.startswith(("!tt.ptr<", "!tt.tensor<")):
            category = "pointer"
            element = re.match(r"!tt\.(?:ptr|tensor)<([^,>]+)", type_text)
            if element is None:
                return None
            dtype = _DTYPE_ALIASES.get(element.group(1), element.group(1)).casefold()
        elif (integer := re.match(r"(?:i|ui)[0-9]+(?:\b|$)", type_text)) is not None:
            category = "integer"
            dtype = _DTYPE_ALIASES.get(integer.group(0), integer.group(0)).casefold()
        elif (floating := re.match(r"(?:f|bf)[0-9]+(?:\b|$)", type_text)) is not None:
            category = "float"
            dtype = _DTYPE_ALIASES.get(floating.group(0), floating.group(0)).casefold()
        else:
            return None
        result.append((position, category, dtype))
    return tuple(result)


def _source_argument_category(
    argument: Mapping[str, Any],
) -> tuple[str, str] | None:
    kind = argument.get("kind")
    dtype = argument.get("dtype")
    if not isinstance(dtype, str):
        return None
    normalized = _DTYPE_ALIASES.get(dtype, dtype).casefold()
    if kind in {"tensor", "pointer"}:
        return "pointer", normalized
    if kind != "scalar":
        return None
    if normalized.startswith(("float", "bfloat")):
        return "float", normalized
    if normalized.startswith(("int", "uint", "bool")):
        return "integer", normalized
    return None


def _ttir_source_abi_aliases(
    ttir: str,
    source_facts: SourceFacts,
    *,
    function_name: str,
) -> dict[str, str] | None:
    """Prove the observed TTIR argument-to-physical-ABI projection.

    A valid full projection is the *unique* order-preserving type-category
    match from the compiler's surviving arguments to SourceFacts ``p*`` slots.
    Source slots may disappear through compiler specialization.  If scalar
    specialization leaves more than one possible scalar alignment, a unique
    pointer-only projection remains sufficient for transfer ownership: every
    ambiguous scalar is deliberately labelled ``unmappedN`` rather than being
    guessed as a physical ABI slot.
    """

    observed = _ttir_argument_categories(ttir, function_name=function_name)
    if observed is None:
        return None
    expected: list[tuple[int, str, str]] = []
    for argument in source_facts.arguments:
        name = argument.get("name") if isinstance(argument, Mapping) else None
        match = re.fullmatch(r"p([0-9]+)", name or "")
        category_and_dtype = (
            _source_argument_category(argument)
            if isinstance(argument, Mapping)
            else None
        )
        if match is None or category_and_dtype is None:
            return None
        category, dtype = category_and_dtype
        expected.append((int(match.group(1)), category, dtype))
    if not expected:
        return None
    expected.sort(key=lambda item: item[0])
    if [position for position, _category, _dtype in expected] != list(range(len(expected))):
        return None

    # Each entry is (cost, capped count, one mapping).  Capping at two
    # preserves the only distinction needed here: unique versus ambiguous.
    # Compiler index lowering can turn a Source i64 into TTIR i32, but an exact
    # dtype match is strictly preferred wherever available.
    memo: dict[
        tuple[int, int],
        tuple[int | None, int, tuple[tuple[int, int], ...] | None],
    ] = {}

    def match_cost(
        source_category: str,
        source_dtype: str,
        observed_category: str,
        observed_dtype: str,
    ) -> int | None:
        if source_category != observed_category:
            return None
        if source_category in {"pointer", "float"}:
            return 0 if source_dtype == observed_dtype else None
        return 0 if source_dtype == observed_dtype else 1

    def align(
        source_index: int,
        observed_index: int,
    ) -> tuple[int | None, int, tuple[tuple[int, int], ...] | None]:
        key = (source_index, observed_index)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if observed_index == len(observed):
            result = (0, 1, ())
        elif source_index == len(expected):
            result = (None, 0, None)
        else:
            cost, count, mapping = align(source_index + 1, observed_index)
            source_position, source_category, source_dtype = expected[source_index]
            observed_position, observed_category, observed_dtype = observed[observed_index]
            conversion_cost = match_cost(
                source_category,
                source_dtype,
                observed_category,
                observed_dtype,
            )
            if conversion_cost is not None:
                matched_cost, matched_count, matched_mapping = align(
                    source_index + 1,
                    observed_index + 1,
                )
                if matched_cost is not None and matched_count:
                    matched = (
                        ((observed_position, source_position), *matched_mapping)
                        if matched_mapping is not None
                        else None
                    )
                    matched_cost += conversion_cost
                    if cost is None or matched_cost < cost:
                        cost, count, mapping = matched_cost, matched_count, matched
                    elif matched_cost == cost:
                        count = min(2, count + matched_count)
                        mapping = None
            result = (cost, count, mapping)
        memo[key] = result
        return result

    _cost, count, mapping = align(0, 0)
    if _cost is not None and count == 1 and mapping is not None and len(mapping) == len(observed):
        return {
            f"arg{observed_position}": str(source_position)
            for observed_position, source_position in mapping
        }

    expected_pointers = [
        (position, dtype)
        for position, category, dtype in expected
        if category == "pointer"
    ]
    observed_pointers = [
        (position, dtype)
        for position, category, dtype in observed
        if category == "pointer"
    ]
    if (
        len(expected_pointers) != len(observed_pointers)
        or [dtype for _position, dtype in expected_pointers]
        != [dtype for _position, dtype in observed_pointers]
    ):
        return None
    pointer_aliases = {
        observed_position: source_position
        for (source_position, _source_dtype), (observed_position, _observed_dtype) in zip(
            expected_pointers,
            observed_pointers,
            strict=True,
        )
    }
    return {
        f"arg{observed_position}": str(pointer_aliases[observed_position])
        if observed_position in pointer_aliases
        else f"unmapped{observed_position}"
        for observed_position, _category, _dtype in observed
    }


def _source_observation_python_interface(
    ttir: str,
    python_interface: Mapping[str, Any],
    source_facts: SourceFacts,
    *,
    function_name: str,
) -> dict[str, Any] | None:
    """Attach the one proven TTIR-to-Source ABI view to an observation."""

    aliases = _ttir_source_abi_aliases(
        ttir,
        source_facts,
        function_name=function_name,
    )
    if aliases is None:
        return None
    result = dict(python_interface)
    result["ttir_argument_aliases"] = aliases
    return result


def observe_source_ttir(
    source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    source_function: str,
    compiler: TritonCompiler | None = None,
    import_roots: tuple[str, ...] = (),
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Compile one host-only exact Source specialization and expose facts only.

    Failure is returned as a closed observation rather than being reclassified as
    a model failure.  The caller may require a passing observation for a formal
    capability run, while still retaining the exact compiler diagnostic.

    ``source`` never crosses the provider boundary.  The Source/compiler
    provenance ledger deliberately uses its exact debug coordinates, while the
    independently anonymized text remains the only Source representation sent
    to Lift.  This avoids treating coordinates from two different renderings as
    if they named the same occurrence.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_OBSERVATION_PROTOCOL,
        "authority": (
            "exact host-only Source compilation and existing Gate1 observer only; "
            "no J/C/pi field, ownership decision, or repair is generated"
        ),
        "function": "source_entry",
    }
    if not isinstance(source_facts, SourceFacts):
        return {
            **base,
            "status": "unavailable",
            "diagnostic": "canonical SourceFacts are required for Source observation",
        }
    base["source_facts"] = source_facts.to_json()
    if not isinstance(source, str) or not source:
        return {**base, "status": "fail", "diagnostic": "Source is absent"}
    if not isinstance(source_function, str) or not source_function.isidentifier():
        return {**base, "status": "unavailable", "diagnostic": "Source entry is absent"}
    if not isinstance(compiler_view, Mapping):
        return {
            **base,
            "status": "unavailable",
            "diagnostic": "anonymous compiler view is absent",
        }
    try:
        case = _compile_case(compiler_view)
        result = (compiler or TritonCompiler()).compile(
            KernelSource.from_code(
                source,
                source_function,
                trusted=True,
                import_roots=import_roots,
            ),
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
        python_interface = _source_observation_python_interface(
            result.ttir,
            result.python_interface,
            source_facts,
            function_name=source_function,
        )
        if python_interface is None:
            return {
                **base,
                "status": "unavailable",
                "diagnostic": "observed TTIR ABI has no unique SourceFacts projection",
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }
        fingerprint = fingerprint_ttir(
            result.ttir,
            python_interface=python_interface,
            schedule=case.schedule,
        )
        return {
            **base,
            "status": "pass",
            "ttir": result.ttir,
            "fingerprint": fingerprint.to_dict(),
            "fingerprint_digest": fingerprint.digest,
            "source_runtime_parameter_uses": [
                opaque
                for name in _source_runtime_parameter_uses(
                    source,
                    function_name=source_function,
                    constexpr_values=case.schedule,
                )
                if (opaque := dict(source_facts.source_to_opaque).get(name))
                is not None
                and re.fullmatch(r"p[0-9]+", opaque) is not None
            ],
            "python_interface": python_interface,
            "compiler": dict(result.compiler),
            "target": dict(result.target),
            "elapsed_ms": result.elapsed_ms,
            "cache_hit": result.cache_hit,
        }
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "diagnostic": _closed_observer_exception(exc),
        }


def observe_anonymous_source_ttir(
    anonymous_source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    compiler: TritonCompiler | None = None,
    import_roots: tuple[str, ...] = (),
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for an already-anonymous Source compilation."""

    return observe_source_ttir(
        anonymous_source,
        compiler_view,
        source_function="scrc_k0",
        compiler=compiler,
        import_roots=import_roots,
        source_facts=source_facts,
    )


def _source_memory_kind(operation: str) -> str | None:
    if operation.startswith("tt.load"):
        return "load"
    if operation.startswith("tt.store"):
        return "store"
    if operation.startswith("tt.atomic"):
        return "update"
    return None


def _polynomial_atoms(
    polynomial: object,
) -> tuple[str, ...]:
    atoms: list[str] = []
    if not isinstance(polynomial, (list, tuple)):
        return ()
    for term in polynomial:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            continue
        monomial = term[0]
        if not isinstance(monomial, (list, tuple)):
            continue
        atoms.extend(item for item in monomial if isinstance(item, str))
    return tuple(atoms)


def _source_program_axes(
    polynomial: object,
) -> tuple[int, ...]:
    axes: set[int] = set()
    for atom in _polynomial_atoms(polynomial):
        match = re.fullmatch(r"program_axis:([xyz])", atom)
        if match is not None:
            axes.add("xyz".index(match.group(1)))
    return tuple(sorted(axes))


def _source_argument_positions(
    polynomial: object,
) -> tuple[int, ...]:
    positions: set[int] = set()
    for atom in _polynomial_atoms(polynomial):
        match = re.fullmatch(r"arg\[(\d+)\]:.*", atom)
        if match is not None:
            positions.add(int(match.group(1)))
    return tuple(sorted(positions))


def _source_pointer_base_position(pointer_base: object) -> int | None:
    if not isinstance(pointer_base, str):
        return None
    match = re.fullmatch(r"arg\[(\d+)\]:.*", pointer_base)
    return int(match.group(1)) if match is not None else None


def _candidate_parameter_projections(
    raw_parameters: object,
    *,
    source_facts: SourceFacts | None,
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, Mapping[str, Any]],
    dict[str, int],
] | None:
    """Project candidate ABI rows into Source-runtime and physical views.

    Source transfer occurrences use ``arg[i]`` where ``i`` is the runtime
    argument ordinal.  A candidate's ``pN`` suffix is a physical ABI handle,
    not that ordinal: constexpr parameters may occupy a physical slot without
    appearing in the runtime call.  Keep both views explicit so a consumer
    cannot accidentally use one namespace as the other.

    ``source_facts`` is required for the authoritative projection.  A missing
    or incomplete Source ABI is therefore reject-only rather than an
    invitation to infer runtime order from candidate-authored ``kind`` rows.
    """

    if not isinstance(raw_parameters, list) or not isinstance(source_facts, SourceFacts):
        return None
    physical_by_position: dict[int, Mapping[str, Any]] = {}
    physical_by_name: dict[str, Mapping[str, Any]] = {}
    for parameter in raw_parameters:
        if not isinstance(parameter, Mapping):
            return None
        name = parameter.get("name")
        match = re.fullmatch(r"p([0-9]+)", name or "")
        if match is None:
            continue
        position = int(match.group(1))
        if position in physical_by_position or name in physical_by_name:
            return None
        physical_by_position[position] = parameter
        physical_by_name[name] = parameter

    runtime_names = tuple(
        name
        for name in source_facts.parameter_order
        if name not in source_facts.constexpr_parameters
    )
    # SourceFacts permits the ABI wire to describe source-side declarations,
    # but the active canonical contract reserves c* for compiler-only slots.
    # Do not silently drop an unexpected c* runtime declaration: doing so
    # would renumber every later Source arg[i] and recreate the original
    # runtime/physical namespace collision.
    if any(
        not isinstance(name, str) or not name.startswith("p")
        for name in runtime_names
    ):
        return None
    runtime_by_position: dict[int, Mapping[str, Any]] = {}
    runtime_position_by_name: dict[str, int] = {}
    for runtime_position, name in enumerate(runtime_names):
        parameter = physical_by_name.get(name)
        if not isinstance(parameter, Mapping):
            return None
        runtime_by_position[runtime_position] = parameter
        runtime_position_by_name[name] = runtime_position
    return runtime_by_position, physical_by_position, runtime_position_by_name


def _source_domain_coordinates(
    boundary: Mapping[str, Any],
) -> dict[str, int]:
    domains = boundary.get("domains")
    if not isinstance(domains, Mapping):
        return {}
    result: dict[str, int] = {}
    for domain_ref, expression in domains.items():
        if not isinstance(domain_ref, str) or not isinstance(expression, str):
            continue
        # Canonical J accepts a direct logical coordinate independently of its
        # spelling.  Earlier Source->pi binding recognized only the historical
        # ``k*`` spelling, while the canonical constructor emits ``i*``.  Both
        # name the same closed ordinal coordinate; accepting the two explicit
        # spellings here keeps a Source transfer occurrence bound to its J
        # access instead of silently dropping its unique address carrier.
        coordinates = {
            int(match)
            for match in re.findall(r"\b(?:i|k)([0-9]+)\b", expression)
        }
        if len(coordinates) == 1:
            result[domain_ref] = next(iter(coordinates))
    return result


def _source_domain_extent_symbols(
    boundary: Mapping[str, Any],
) -> dict[str, str]:
    """Return the unique x* upper extent of a zero-based direct domain.

    The semantic-transfer wire spells each direct access axis with its
    canonical tensor-dimension handle (``x*``), while the host-derived domain
    uses a local coordinate (``i*``) bounded by that same handle.  Preserve
    the existing coordinate spelling, but expose this exact bridge to the
    Source->pi binder.  General affine, non-zero-based, inclusive, guarded,
    or otherwise ambiguous domains deliberately produce no bridge.
    """

    domains = boundary.get("domains")
    if not isinstance(domains, Mapping):
        return {}
    result: dict[str, str] = {}
    for domain_ref, expression in domains.items():
        if not isinstance(domain_ref, str) or not isinstance(expression, str):
            continue
        try:
            parsed = parse_contract_expr(
                expression,
                expected=ContractSort.BOOLEAN,
                context=f"J boundary domain {domain_ref}",
            )
        except Exception:
            continue
        atoms = parsed.args if parsed.op == "and" else (parsed,)
        lower_coordinates: set[str] = set()
        upper_pairs: set[tuple[str, str]] = set()
        for atom in atoms:
            if len(atom.args) != 2:
                continue
            left, right = atom.args
            if (
                atom.op == "le"
                and left.op == "int"
                and left.value == 0
                and right.op == "ref"
                and isinstance(right.value, str)
            ):
                lower_coordinates.add(right.value)
            elif (
                atom.op == "ge"
                and left.op == "ref"
                and isinstance(left.value, str)
                and right.op == "int"
                and right.value == 0
            ):
                lower_coordinates.add(left.value)
            if (
                atom.op == "lt"
                and left.op == "ref"
                and isinstance(left.value, str)
                and right.op == "ref"
                and isinstance(right.value, str)
                and re.fullmatch(r"x[0-9]+", right.value) is not None
            ):
                upper_pairs.add((left.value, right.value))
            elif (
                atom.op == "gt"
                and left.op == "ref"
                and isinstance(left.value, str)
                and re.fullmatch(r"x[0-9]+", left.value) is not None
                and right.op == "ref"
                and isinstance(right.value, str)
            ):
                upper_pairs.add((right.value, left.value))
        closed = {
            (coordinate, extent)
            for coordinate, extent in upper_pairs
            if coordinate in lower_coordinates
            and re.fullmatch(r"(?:i|k)[0-9]+", coordinate) is not None
        }
        if len(closed) == 1:
            _coordinate, extent = next(iter(closed))
            result[domain_ref] = extent
    return result


def _index_axis_for_coordinate(
    index_expression: object,
    coordinate: int,
    *,
    extent_symbol: str | None = None,
) -> int | None:
    if not isinstance(index_expression, str):
        return None
    expression = index_expression.strip()
    if expression.startswith("[") and expression.endswith("]"):
        expression = expression[1:-1]
    terms = [item.strip() for item in expression.split(",")]
    coordinate_matches = [
        index
        for index, term in enumerate(terms)
        if re.search(rf"\b(?:i|k){coordinate}\b", term)
    ]
    extent_matches = [
        index
        for index, term in enumerate(terms)
        if extent_symbol is not None and term == extent_symbol
    ]
    matches = set((*coordinate_matches, *extent_matches))
    return next(iter(matches)) if len(matches) == 1 else None


def _axis_ref_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"x([0-9]+)", value)
        if match is not None:
            return int(match.group(1))
    return None


def _candidate_program_axis_map(
    pi: Mapping[str, Any],
) -> dict[int, set[tuple[str, int | None]]]:
    """Read already-authored program mappings without choosing a mapping."""

    try:
        projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError:
        return {}
    result: dict[int, set[tuple[str, int | None]]] = {}
    for row in projection.rows:
        for domain_ref, axis_ref in row.logical_axes:
            numeric_axis = _axis_ref_number(axis_ref)
            # ``program_axes`` intentionally has no logical tensor-axis
            # selector and is represented by None.  Relation rows must carry
            # a concrete axis_ref; the shared projection has already checked
            # that distinction, so a non-numeric symbolic x* is simply not a
            # candidate for this numeric Source-observation index.
            if axis_ref is not None and numeric_axis is None:
                continue
            result.setdefault(row.physical_axis, set()).add(
                (domain_ref, numeric_axis)
            )
    return result


def materialize_source_program_domains(
    packet: Mapping[str, Any],
    *,
    source_facts: SourceFacts,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a uniquely Source-bound launch coordinate into J and pi.

    Fresh MCP semantic entry deliberately has no provider write for host-owned
    program mapping. A direct launch coordinate can nevertheless be closed
    without consulting a provider graph when the mounted compiler grid binds a
    physical launch axis to one ``h*`` coordinate, the mounted ABI maps that
    coordinate to exactly one canonical ``x*`` dimension, and the same axis
    occurs in the closed Source transfer ledger. This is a construction
    projection, never an inference from rank, parameter position, a concrete
    extent, or a candidate-authored pi row.

    Any absent, mixed, or ambiguous fact leaves the packet untouched. An
    existing program-axis mapping is never supplemented; a pre-existing tile
    iteration row alone is preserved while the independently proved launch
    mapping is projected. Exact provider domains may be reused, but only when
    one unique zero-based domain has the required Source dimension extent;
    otherwise projection fails closed without adding or changing a domain.
    """

    input_artifact = copy.deepcopy(dict(packet))
    original = copy.deepcopy(input_artifact)
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-program-domain.v1",
        "status": "pass",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }
    unavailable_reasons = {
        "source_facts_missing_or_malformed",
        "transfer_ledger_not_closed",
        "candidate_wire_malformed",
        "grid_facts_missing_or_malformed",
        "candidate_parameter_projection_ambiguous",
        "candidate_tensor_projection_malformed",
        "source_tensor_shape_projection_malformed",
        "shape_coordinate_binding_malformed",
        "no_used_physical_program_axis",
        "used_physical_program_axis_out_of_range",
        "direct_axis_coordinate_unavailable",
        "compound_grid_expression_unsupported",
        "compound_grid_factor_coordinate_unavailable",
        "shape_coordinate_extent_missing",
        "logical_dimension_binding_unavailable",
        "grid_extent_mismatch",
        "duplicate_logical_dimension",
        "tile_coordinate_binding_malformed",
        "tile_coordinate_logical_dimension_unavailable",
        "tile_slot_value_unavailable",
        "tile_program_dimension_overlap",
        "duplicate_tile_logical_dimension",
        "provider_domain_dimension_unavailable",
        "provider_domain_dimension_ambiguous",
        "no_closed_program_domain_projection",
    }

    def unavailable(
        reason: str,
        *observed: str,
        path: str = "j.boundary.domains",
    ) -> dict[str, Any]:
        """Return one bounded, Source-free projection failure receipt."""

        if reason not in unavailable_reasons:
            raise ValueError("unknown Source program-domain failure reason")
        safe_observed = [f"reason={reason}"]
        for value in observed:
            if (
                isinstance(value, str)
                and value
                and len(value) <= 128
                and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
            ):
                safe_observed.append(value)
        result["status"] = "unavailable"
        result["reason"] = reason
        # Projection is one atomic transaction.  Some failures are discovered
        # only after candidate-local domains or pi rows have been staged; never
        # expose that partial state to a caller or diagnostic consumer.
        result["artifact"] = copy.deepcopy(input_artifact)
        result["diagnostics"] = [
            {
                "rule_code": f"source_program_domain_projection_{reason}",
                "path": path,
                "observed_refs_or_types": safe_observed,
            }
        ]
        return result

    if not isinstance(source_facts, SourceFacts):
        return unavailable("source_facts_missing_or_malformed")
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass":
        return unavailable(
            "transfer_ledger_not_closed",
            f"ledger_status={shared_ledger.status}",
            f"occurrence_count={len(shared_ledger.occurrences)}",
            f"binding_count={len(shared_ledger.bindings)}",
            path="j.boundary.accesses",
        )

    j = original.get("j")
    pi = original.get("pi")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if not all(isinstance(value, dict) for value in (j, pi, boundary, function, signature)):
        return unavailable("candidate_wire_malformed")
    domains = boundary.get("domains")
    program_axes = pi.get("program_axes")
    program_bindings = pi.get("program_bindings")
    iteration_bindings = pi.get("iteration_bindings")
    parameters = signature.get("parameters")
    inputs = function.get("inputs")
    outputs = function.get("outputs")
    if (
        not isinstance(domains, dict)
        or not isinstance(program_axes, list)
        or not isinstance(program_bindings, list)
        or not isinstance(iteration_bindings, list)
        or not isinstance(parameters, list)
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
    ):
        return unavailable("candidate_wire_malformed")
    # A launch-axis mapping is a complete, independent π authority.  Preserve
    # it byte-for-byte and let the ordinary canonical/Source validators prove
    # or reject it rather than supplementing it here.  An iteration row alone
    # is not a launch-axis mapping, however: upstream host construction can
    # already have a Source-proven tile relation before this pass observes the
    # grid.  Treating that row as if it proved a program mapping made a valid
    # domain projection silently return without creating its required
    # program_axes/program_bindings.  Keep the row untouched and derive only
    # the uniquely Source-bound launch part below; malformed or conflicting
    # iteration rows remain fail-closed in the strict canonical validation.
    if program_axes or program_bindings:
        return result
    provider_domains_present = bool(domains)

    compiler_facts = source_facts.compiler_facts
    grid = compiler_facts.get("grid")
    grid_contract = compiler_facts.get("grid_contract")
    grid_coordinates = compiler_facts.get("grid_shape_coordinates")
    shape_coordinates = dict(source_facts.shape_coordinates)
    schedule_values = compiler_facts.get("schedule_values")
    schedule_roles = compiler_facts.get("compiler_roles")
    tile_coordinates = compiler_facts.get("schedule_tile_coordinates")
    if not all(
        isinstance(value, Mapping)
        for value in (schedule_values, schedule_roles, tile_coordinates)
    ):
        return unavailable("tile_coordinate_binding_malformed")
    if (
        not isinstance(grid, list)
        or not grid
        or len(grid) > 3
        or not isinstance(grid_contract, list)
        or not isinstance(grid_coordinates, list)
        or len(grid) != len(grid_contract)
        or (grid_coordinates and len(grid) != len(grid_coordinates))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in grid
        )
        or any(not isinstance(value, str) or not value for value in grid_contract)
        or any(
            value is not None and not isinstance(value, str)
            for value in grid_coordinates
        )
    ):
        return unavailable(
            "grid_facts_missing_or_malformed",
            f"grid_rank={len(grid) if isinstance(grid, list) else -1}",
            f"grid_contract_rank={len(grid_contract) if isinstance(grid_contract, list) else -1}",
            f"grid_coordinate_rank={len(grid_coordinates) if isinstance(grid_coordinates, list) else -1}",
            path="pi.program_bindings",
        )
    # A compound grid expression intentionally has no single h* coordinate.
    # Normalize the optional direct-coordinate sidecar to one slot per axis;
    # the closed grid contract below remains the authority for compound axes.
    if not grid_coordinates:
        grid_coordinates = [None] * len(grid)

    parameters_by_name: dict[str, Mapping[str, Any]] = {}
    for parameter in parameters:
        name = parameter.get("name") if isinstance(parameter, Mapping) else None
        if not isinstance(name, str) or name in parameters_by_name:
            return unavailable(
                "candidate_parameter_projection_ambiguous",
                f"parameter_count={len(parameters)}",
            )
        parameters_by_name[name] = parameter
    # Inputs and outputs are separate ABI collections.  A legal inout repeats
    # the same t* declaration once in each collection; only repetition within
    # one collection or a cross-collection type disagreement is ambiguous.
    # Keep this identical to the canonical value-type authority instead of
    # rejecting rope-style inouts as duplicate logical tensors.
    tensors: dict[str, Mapping[str, Any]] = {}
    tensor_types: dict[str, tuple[str, tuple[str | int, ...]]] = {}
    for collection_name, collection in (("inputs", inputs), ("outputs", outputs)):
        collection_names: set[str] = set()
        for row_index, tensor in enumerate(collection):
            name = tensor.get("name") if isinstance(tensor, Mapping) else None
            dtype = tensor.get("dtype") if isinstance(tensor, Mapping) else None
            shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
            if (
                not isinstance(name, str)
                or re.fullmatch(r"t[0-9]+", name) is None
                or name in collection_names
                or not isinstance(dtype, str)
                or not dtype
                or not isinstance(shape, list)
                or any(
                    not isinstance(extent, (str, int))
                    or isinstance(extent, bool)
                    for extent in shape
                )
            ):
                return unavailable(
                    "candidate_tensor_projection_malformed",
                    f"collection={collection_name}",
                    f"row_index={row_index}",
                )
            collection_names.add(name)
            value_type = (dtype, tuple(shape))
            previous_type = tensor_types.get(name)
            if previous_type is not None and previous_type != value_type:
                return unavailable(
                    "candidate_tensor_projection_malformed",
                    f"collection={collection_name}",
                    f"row_index={row_index}",
                    "inout_type_mismatch=1",
                )
            tensor_types[name] = value_type
            tensors.setdefault(name, tensor)

    coordinate_dimensions: dict[str, set[str]] = {}
    for argument_index, argument in enumerate(source_facts.arguments):
        if argument.get("kind") != "tensor":
            continue
        name = argument.get("name")
        # A Source tensor can have a wholly static shape.  SourceFacts records
        # that closed case by omitting ``shape_contract`` and retaining the
        # literal ``shape``; canonical_source_wire_seed deliberately preserves
        # the same literals in J instead of inventing x* dimensions.  Treat
        # that shape as the contract here too.  Requiring a list unconditionally
        # made a valid static tensor prevent an otherwise unrelated, explicitly
        # shape-coordinate-bound launch axis from being projected.
        #
        # This is not a concrete-extent fallback for h* coordinates: a dynamic
        # coordinate still has to map positionally to an explicit x* dimension.
        # The literal branch below only accepts the exact Source static extent
        # at the same tensor axis.
        source_shape = argument.get("shape")
        raw_contracts = argument.get("shape_contract")
        contracts = source_shape if raw_contracts is None else raw_contracts
        parameter = parameters_by_name.get(name) if isinstance(name, str) else None
        logical_ref = parameter.get("logical_ref") if isinstance(parameter, Mapping) else None
        tensor = tensors.get(logical_ref) if isinstance(logical_ref, str) else None
        shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
        if (
            not isinstance(contracts, list)
            or not isinstance(source_shape, list)
            or not isinstance(shape, list)
            or len(contracts) != len(shape)
            or len(source_shape) != len(contracts)
        ):
            return unavailable(
                "source_tensor_shape_projection_malformed",
                f"argument_index={argument_index}",
                f"shape_contract_rank={len(contracts) if isinstance(contracts, list) else -1}",
                f"logical_shape_rank={len(shape) if isinstance(shape, list) else -1}",
            )
        for axis, (coordinate, dimension) in enumerate(
            zip(contracts, shape, strict=True)
        ):
            if isinstance(coordinate, int) and not isinstance(coordinate, bool):
                if dimension != coordinate:
                    return unavailable(
                        "source_tensor_shape_projection_malformed",
                        f"argument_index={argument_index}",
                        f"axis={axis}",
                        "static_extent_mismatch=1",
                    )
                continue
            if (
                not isinstance(coordinate, str)
                or re.fullmatch(r"h[0-9]+", coordinate) is None
                or not isinstance(dimension, str)
                or re.fullmatch(r"x[0-9]+", dimension) is None
            ):
                return unavailable(
                    "source_tensor_shape_projection_malformed",
                    f"argument_index={argument_index}",
                    f"axis={axis}",
                    "shape_axis_contract_invalid=1",
                )
            coordinate_dimensions.setdefault(coordinate, set()).add(dimension)

    # Grid-contract identifiers are already canonical ABI/C names.  Resolve
    # only explicit shape-coordinate bindings from the same SourceFacts: a
    # runtime scalar's declared coordinate or a bound constexpr coordinate.
    # Concrete values alone never select a coordinate because distinct h*
    # dimensions may happen to have the same value.
    factor_coordinates: dict[str, str] = {}
    bound_coordinates = compiler_facts.get("bound_constexpr_shape_coordinates")
    if not isinstance(bound_coordinates, Mapping):
        return unavailable("shape_coordinate_binding_malformed")
    for name, coordinate in bound_coordinates.items():
        if isinstance(name, str) and isinstance(coordinate, str):
            factor_coordinates[name] = coordinate
    for argument_index, argument in enumerate(source_facts.arguments):
        name = argument.get("name") if isinstance(argument, Mapping) else None
        coordinate = (
            argument.get("shape_coordinate")
            if isinstance(argument, Mapping)
            else None
        )
        if isinstance(name, str) and isinstance(coordinate, str):
            previous = factor_coordinates.get(name)
            if previous is not None and previous != coordinate:
                return unavailable(
                    "shape_coordinate_binding_malformed",
                    f"argument_index={argument_index}",
                )
            factor_coordinates[name] = coordinate

    used_launch_axes = {
        axis
        for occurrence in shared_ledger.occurrences
        for axis in _source_program_axes(occurrence.address_terms)
    }
    if not used_launch_axes:
        return unavailable(
            "no_used_physical_program_axis",
            f"occurrence_count={len(shared_ledger.occurrences)}",
            f"grid_rank={len(grid)}",
            path="pi.program_axes",
        )
    out_of_range_axes = sorted(
        axis for axis in used_launch_axes if axis >= len(grid)
    )
    if out_of_range_axes:
        return unavailable(
            "used_physical_program_axis_out_of_range",
            f"physical_axis={out_of_range_axes[0]}",
            f"used_axis_count={len(used_launch_axes)}",
            f"grid_rank={len(grid)}",
            path="pi.program_axes",
        )
    projections: list[tuple[int, str, tuple[str, ...]]] = []
    for physical_axis, coordinate in enumerate(grid_coordinates):
        if physical_axis not in used_launch_axes:
            continue
        relation = "direct"
        coordinates: tuple[str, ...]
        if isinstance(coordinate, str):
            coordinates = (coordinate,)
        else:
            expression = grid_contract[physical_axis].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression) is not None:
                return unavailable(
                    "direct_axis_coordinate_unavailable",
                    f"physical_axis={physical_axis}",
                    "relation=direct",
                    path="pi.program_axes",
                )
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\*\s*[A-Za-z_][A-Za-z0-9_]*)+",
                expression,
            ) is None:
                return unavailable(
                    "compound_grid_expression_unsupported",
                    f"physical_axis={physical_axis}",
                    "relation=compound",
                    path="pi.program_bindings",
                )
            factors = tuple(part.strip() for part in expression.split("*"))
            resolved = tuple(factor_coordinates.get(factor) for factor in factors)
            if (
                any(not isinstance(item, str) for item in resolved)
                or len(set(resolved)) != len(resolved)
            ):
                return unavailable(
                    "compound_grid_factor_coordinate_unavailable",
                    f"physical_axis={physical_axis}",
                    "relation=product_linearized",
                    f"factor_count={len(factors)}",
                    f"bound_coordinate_count={len({item for item in resolved if isinstance(item, str)})}",
                    path="pi.program_bindings",
                )
            coordinates = tuple(str(item) for item in resolved)
            relation = "product_linearized"

        dimensions: list[str] = []
        extent_product = 1
        for item in coordinates:
            if (
                re.fullmatch(r"h[0-9]+", item) is None
                or item not in shape_coordinates
            ):
                return unavailable(
                    "shape_coordinate_extent_missing",
                    f"physical_axis={physical_axis}",
                    f"relation={relation}",
                    f"coordinate={item if re.fullmatch(r'h[0-9]+', item) is not None else 'invalid'}",
                    path="pi.program_bindings",
                )
            choices = coordinate_dimensions.get(item, set())
            if len(choices) != 1:
                # No concrete-extent fallback is permitted: same-valued h*
                # axes are not interchangeable logical dimensions.
                return unavailable(
                    "logical_dimension_binding_unavailable",
                    f"physical_axis={physical_axis}",
                    f"relation={relation}",
                    f"coordinate={item}",
                    f"logical_dimension_count={len(choices)}",
                    path="pi.program_bindings",
                )
            dimensions.append(next(iter(choices)))
            extent_product *= shape_coordinates[item]
        if extent_product != grid[physical_axis]:
            return unavailable(
                "grid_extent_mismatch",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"coordinate_count={len(coordinates)}",
                f"logical_dimension_count={len(dimensions)}",
                path="pi.program_bindings",
            )
        if len(set(dimensions)) != len(dimensions):
            return unavailable(
                "duplicate_logical_dimension",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"logical_dimension_count={len(dimensions)}",
                f"unique_logical_dimension_count={len(set(dimensions))}",
                path="pi.program_bindings",
            )
        projections.append((physical_axis, relation, tuple(dimensions)))
    if not projections:
        return unavailable(
            "no_closed_program_domain_projection",
            f"used_axis_count={len(used_launch_axes)}",
            f"grid_rank={len(grid)}",
            path="pi.program_bindings",
        )

    use_program_bindings = any(
        relation != "direct" for _axis, relation, _dimensions in projections
    )
    domain_ordinal = len(domains)
    existing_extent_domains: dict[str, list[str]] = {}
    if provider_domains_present:
        for domain_ref, dimension in _source_domain_extent_symbols(boundary).items():
            existing_extent_domains.setdefault(dimension, []).append(domain_ref)
    domain_by_dimension: dict[str, str] = {}

    def domain_for_dimension(dimension: str) -> str | None:
        nonlocal domain_ordinal
        existing = existing_extent_domains.get(dimension, [])
        if len(existing) > 1:
            return None
        if len(existing) == 1:
            domain_by_dimension[dimension] = existing[0]
            return existing[0]
        if provider_domains_present:
            return None
        domain_ref = f"d{domain_ordinal}"
        coordinate_ref = f"i{domain_ordinal}"
        domains[domain_ref] = (
            f"0 <= {coordinate_ref} and {coordinate_ref} < {dimension}"
        )
        domain_by_dimension[dimension] = domain_ref
        domain_ordinal += 1
        return domain_ref

    claimed_program_dimensions: set[str] = set()
    for physical_axis, relation, dimensions in projections:
        repeated_dimensions = claimed_program_dimensions & set(dimensions)
        if repeated_dimensions:
            return unavailable(
                "duplicate_logical_dimension",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"duplicate_count={len(repeated_dimensions)}",
                path=(
                    "pi.program_bindings"
                    if use_program_bindings
                    else "pi.program_axes"
                ),
            )
        logical_axes: list[dict[str, Any]] = []
        for dimension in dimensions:
            domain_ref = domain_for_dimension(dimension)
            if domain_ref is None:
                reason = (
                    "provider_domain_dimension_ambiguous"
                    if len(existing_extent_domains.get(dimension, [])) > 1
                    else "provider_domain_dimension_unavailable"
                )
                return unavailable(
                    reason,
                    f"dimension={dimension}",
                    f"physical_axis={physical_axis}",
                    path="j.boundary.domains",
                )
            logical_axes.append(
                {"domain_ref": domain_ref, "axis_ref": dimension}
            )
        if use_program_bindings:
            program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": relation,
                    "logical_axes": logical_axes,
                }
            )
        else:
            program_axes.append(
                {
                    "domain_ref": logical_axes[0]["domain_ref"],
                    "axis": physical_axis,
                }
            )
        claimed_program_dimensions.update(dimensions)
    tile_dimensions: set[str] = set()
    for slot_name, coordinate in tile_coordinates.items():
        role = schedule_roles.get(slot_name)
        tile_value = schedule_values.get(slot_name)
        choices = (
            coordinate_dimensions.get(coordinate, set())
            if isinstance(coordinate, str)
            else set()
        )
        if (
            role not in {"tile_extent", "reduction_tile_extent"}
            or not isinstance(coordinate, str)
            or re.fullmatch(r"h[0-9]+", coordinate) is None
        ):
            return unavailable(
                "tile_coordinate_binding_malformed",
                f"slot={slot_name}",
                path="pi.iteration_bindings",
            )
        if len(choices) != 1:
            return unavailable(
                "tile_coordinate_logical_dimension_unavailable",
                f"slot={slot_name}",
                f"logical_dimension_count={len(choices)}",
                path="pi.iteration_bindings",
            )
        if (
            isinstance(tile_value, bool)
            or not isinstance(tile_value, int)
            or tile_value <= 0
        ):
            return unavailable(
                "tile_slot_value_unavailable",
                f"slot={slot_name}",
                path="pi.iteration_bindings",
            )
        dimension = next(iter(choices))
        if dimension in domain_by_dimension:
            return unavailable(
                "tile_program_dimension_overlap",
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="pi.iteration_bindings",
            )
        if dimension in tile_dimensions:
            return unavailable(
                "duplicate_tile_logical_dimension",
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="pi.iteration_bindings",
            )
        domain_ref = domain_for_dimension(dimension)
        if domain_ref is None:
            reason = (
                "provider_domain_dimension_ambiguous"
                if len(existing_extent_domains.get(dimension, [])) > 1
                else "provider_domain_dimension_unavailable"
            )
            return unavailable(
                reason,
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="j.boundary.domains",
            )
        projected_iteration_binding = {
            "relation": "tiled_domain",
            "logical_axes": [
                {"subject_ref": domain_ref, "axis_ref": dimension}
            ],
            "tile_slot": slot_name,
            "mode_slot": None,
        }
        # A preceding host pass may already have emitted this exact
        # Source-proven tile witness.  Preserve it instead of duplicating the
        # same relation while adding the independently proved launch mapping.
        # We deliberately do not merge or reinterpret a non-identical row:
        # that would choose a tile/domain meaning the Source facts have not
        # uniquely established, so strict validation must reject it later.
        if projected_iteration_binding not in iteration_bindings:
            iteration_bindings.append(projected_iteration_binding)
        domain_by_dimension[dimension] = domain_ref
        tile_dimensions.add(dimension)
    result["applied"].append(
        {
            "rule_code": "source_program_domain_projection",
            "path": (
                "j.boundary.domains+pi.program_bindings+pi.iteration_bindings"
                if use_program_bindings and tile_coordinates
                else "j.boundary.domains+pi.program_axes+pi.iteration_bindings"
                if tile_coordinates
                else "j.boundary.domains+pi.program_bindings"
                if use_program_bindings
                else "j.boundary.domains+pi.program_axes"
            ),
            "action": (
                "materialize uniquely Source-bound grid shape coordinates or "
                "closed product grid factors plus Source-proven tile coordinates "
                "as canonical J domains, launch-axis rows, and iteration rows"
            ),
        }
    )
    return result


def bind_source_pi(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    source_runtime_parameter_uses: object = None,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Bind uniquely witnessed Source address facts to the compact pi subwire.

    This is the deterministic Source -> pi producer for the active canonical
    path.  It operates only on an already decoded-shaped ``(J,C,pi)`` packet;
    it never repairs the root envelope and never writes J or C.  A row is added
    only when the compiled address forms, an existing J tensor/access, and an
    existing program-axis witness identify exactly one access and one tensor
    axis.  Any conflict is returned as a reject-only diagnostic and the input
    packet is left unchanged.

    ``ledger`` is the shared Source transfer ledger produced for this exact
    candidate.  When supplied, it is authoritative: this binder does not
    independently rematch a transfer by tensor/kind and thereby create a
    second occurrence relation.  The result is intentionally a plain evidence object so Responses Lift,
    Codex Lift, and the runner can carry the same minimal binding semantics
    while choosing their own transport.  ``artifact`` is the only candidate
    payload; the Source observation is never embedded in it.
    """

    original = copy.deepcopy(dict(packet))
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-pi-binding.v1",
        "status": "pass",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }

    def reject(rule_code: str, path: str, *observed: str) -> dict[str, Any]:
        safe_observed = [
            value
            for value in observed
            if isinstance(value, str)
            and value
            and len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
        ]
        result["status"] = "unavailable"
        result["diagnostics"] = [
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": safe_observed or ["closed=0"],
            }
        ]
        return result

    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass":
        result["status"] = "unavailable"
        result["diagnostics"] = [dict(item) for item in shared_ledger.diagnostics] or [
            {
                "rule_code": "source_transfer_ledger_unavailable",
                "path": "<source-transfer-ledger>",
                "observed_refs_or_types": [shared_ledger.reason],
            }
        ]
        return result
    # A Source occurrence ledger becomes a binding authority only after the
    # effect-order reconciler has closed its one-to-one occurrence -> J access
    # rows.  Do not silently continue with an occurrence-only ledger: that
    # would let this pi binder independently rediscover ownership from tensor
    # and kind and reintroduce a second transfer matcher.
    if shared_ledger.occurrences and len(shared_ledger.bindings) != len(
        shared_ledger.occurrences
    ):
        result["status"] = "unavailable"
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_ledger_bindings_required",
                "path": "<source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    f"occurrences={len(shared_ledger.occurrences)}",
                    f"bindings={len(shared_ledger.bindings)}",
                ],
            }
        ]
        return result
    j = original.get("j")
    pi = original.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        result["status"] = "unavailable"
        return result
    if pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        # The legacy materialized pi path has its own frontend-specific
        # BindSource implementation.  Do not reinterpret it with compact-pi
        # rules merely because the Source observer is available.
        result["status"] = "unavailable"
        return result

    ledger_by_ordinal: dict[int, Mapping[str, Any]] = {
        row["ordinal"]: row
        for row in shared_ledger.bindings
        if isinstance(row, Mapping)
        and isinstance(row.get("ordinal"), int)
        and isinstance(row.get("access_ref"), str)
    }

    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(raw_parameters, list) or not isinstance(raw_accesses, list):
        result["status"] = "unavailable"
        return result

    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return reject(
            "source_runtime_parameter_projection_unavailable",
            "j.signature.parameters",
            "projection=runtime_physical_abi",
        )
    runtime_parameters, parameters_by_position, runtime_position_by_name = (
        parameter_projection
    )

    accesses = [
        item
        for item in raw_accesses
        if isinstance(item, Mapping)
        and isinstance(item.get("access_id"), str)
        and isinstance(item.get("tensor"), str)
        and isinstance(item.get("kind"), str)
    ]
    access_by_id = {
        str(item["access_id"]): item for item in accesses
    }
    domain_coordinates = _source_domain_coordinates(
        boundary if isinstance(boundary, Mapping) else {}
    )
    domain_extent_symbols = _source_domain_extent_symbols(
        boundary if isinstance(boundary, Mapping) else {}
    )
    program_axis_map = _candidate_program_axis_map(pi)

    program_evidence: dict[int, set[tuple[str, int]]] = {}
    access_evidence: dict[tuple[str, str], set[int]] = {}
    source_used_carriers = {
        item
        for item in source_runtime_parameter_uses
        if isinstance(item, str) and re.fullmatch(r"p[0-9]+", item)
    } if isinstance(source_runtime_parameter_uses, list) else set()
    source_forms = tuple(
        (
            occurrence.operation,
            occurrence.pointer_base,
            occurrence.address_terms,
            occurrence.context,
        )
        for occurrence in shared_ledger.occurrences
    )
    for ordinal, (operation, pointer_base, polynomial, _context) in enumerate(
        source_forms
    ):
        physical_axes = _source_program_axes(polynomial)
        if len(physical_axes) != 1:
            continue
        physical_axis = physical_axes[0]
        base_position = _source_pointer_base_position(pointer_base)
        base_parameter = runtime_parameters.get(base_position)
        if not isinstance(base_parameter, Mapping):
            continue
        tensor_ref = base_parameter.get("logical_ref")
        memory_kind = _source_memory_kind(operation)
        source_tensor_ref = shared_ledger.occurrences[ordinal].logical_ref
        if (
            not isinstance(tensor_ref, str)
            or memory_kind is None
            or (
                source_tensor_ref is not None
                and tensor_ref != source_tensor_ref
            )
        ):
            continue
        ledger_row = ledger_by_ordinal.get(ordinal)
        access_ref = (
            ledger_row.get("access_ref")
            if isinstance(ledger_row, Mapping)
            else None
        )
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        if (
            not isinstance(ledger_row, Mapping)
            or not isinstance(access, Mapping)
            or access.get("tensor") != tensor_ref
            or access.get("kind") != memory_kind
            or bool(access.get("predicate") is not None)
            != bool(ledger_row.get("masked"))
        ):
            continue
        if not isinstance(access, Mapping) or not isinstance(access_ref, str):
            continue

        independent_coordinate_axes: set[tuple[str, int]] = set()
        for domain_ref, coordinate in domain_coordinates.items():
            axis = _index_axis_for_coordinate(
                access.get("index_expression"),
                coordinate,
                extent_symbol=domain_extent_symbols.get(domain_ref),
            )
            if axis is not None:
                independent_coordinate_axes.add((domain_ref, axis))

        # Prefer Source + J evidence that does not depend on the candidate's
        # own pi row.  An authored row may be wrong; using it as the only
        # source of its expected axis would make the binder circular.
        if len(independent_coordinate_axes) == 1:
            domain_ref, logical_axis = next(iter(independent_coordinate_axes))
            program_evidence.setdefault(physical_axis, set()).add(
                (domain_ref, logical_axis)
            )
        else:
            candidate_axes = program_axis_map.get(physical_axis, set())
            if not candidate_axes:
                # Without an authored program mapping, a rank>1 or mixed-
                # radix access remains ambiguous and stays provider-authored.
                continue
            logical_axes: set[int] = set()
            for domain_ref, authored_axis in candidate_axes:
                if authored_axis is not None:
                    logical_axes.add(authored_axis)
                    program_evidence.setdefault(physical_axis, set()).add(
                        (domain_ref, authored_axis)
                    )
                    continue
                coordinate = domain_coordinates.get(domain_ref)
                axis = _index_axis_for_coordinate(
                    access.get("index_expression"),
                    coordinate,
                    extent_symbol=domain_extent_symbols.get(domain_ref),
                ) if coordinate is not None else None
                if axis is not None:
                    logical_axes.add(axis)
                    program_evidence.setdefault(physical_axis, set()).add(
                        (domain_ref, axis)
                    )
            if len(logical_axes) != 1:
                continue
            logical_axis = next(iter(logical_axes))

        for position in _source_argument_positions(polynomial):
            if position == base_position:
                continue
            carrier = runtime_parameters.get(position)
            if not isinstance(carrier, Mapping):
                continue
            carrier_name = carrier.get("name")
            if (
                not isinstance(carrier_name, str)
                or carrier.get("kind") != "scalar"
                or carrier.get("logical_ref") is not None
            ):
                continue
            access_evidence.setdefault(
                (carrier_name, access_ref), set()
            ).add(logical_axis)

    # Specialization can fold a unit stride out of the final address
    # polynomial.  The Source occurrence still has one exact ABI pairing:
    # a Source-used scalar p(N) immediately follows the tensor p(N-1) whose
    # sole transfer uses the same direct program coordinate.  Recover only
    # this complete, one-to-one form.  This is not p* positional inference:
    # every precondition below is required so an unrelated scalar or a tensor
    # with multiple transfer occurrences remains provider-authored.
    if source_used_carriers:
        for carrier_name in sorted(source_used_carriers):
            carrier_match = re.fullmatch(r"p([0-9]+)", carrier_name)
            if carrier_match is None:
                continue
            if any(key[0] == carrier_name for key in access_evidence):
                continue
            carrier_runtime_position = runtime_position_by_name.get(carrier_name)
            if carrier_runtime_position is None:
                continue
            carrier_parameter = runtime_parameters.get(carrier_runtime_position)
            if (
                not isinstance(carrier_parameter, Mapping)
                or carrier_parameter.get("kind") != "scalar"
                or carrier_parameter.get("logical_ref") is not None
            ):
                # The unit-stride fallback is only for an ABI scalar whose
                # physical slot follows its tensor pointer.  Consecutive
                # tensor arguments are not implicit strides, even when both
                # names occur in the Source syntax.
                continue
            base_parameter = runtime_parameters.get(carrier_runtime_position - 1)
            if (
                not isinstance(base_parameter, Mapping)
                or base_parameter.get("kind") not in {"tensor", "pointer"}
                or not isinstance((tensor_ref := base_parameter.get("logical_ref")), str)
            ):
                continue
            # This fallback is valid only when the shared occurrence ledger
            # proves the exact Source ordinal/access pair.  The previous
            # implementation reused ``access``/``access_ref`` from the
            # preceding loop above, so the result depended on the last
            # transfer visited and could bind an unrelated carrier.  Resolve
            # the access exclusively through this occurrence's ledger row;
            # never rematch Source forms by tensor/kind or by stale loop state.
            matching_rows: list[tuple[TransferOccurrence, str, Mapping[str, Any]]] = []
            for occurrence in shared_ledger.occurrences:
                if (
                    occurrence.pointer_position != carrier_runtime_position - 1
                    or occurrence.logical_ref != tensor_ref
                    or len(_source_program_axes(occurrence.address_terms)) != 1
                ):
                    continue
                occurrence_bindings = [
                    row
                    for row in shared_ledger.bindings
                    if row.get("ordinal") == occurrence.ordinal
                ]
                if len(occurrence_bindings) != 1:
                    continue
                occurrence_access_ref = occurrence_bindings[0].get("access_ref")
                occurrence_access = (
                    access_by_id.get(occurrence_access_ref)
                    if isinstance(occurrence_access_ref, str)
                    else None
                )
                if (
                    not isinstance(occurrence_access_ref, str)
                    or not isinstance(occurrence_access, Mapping)
                    or occurrence_access.get("tensor") != tensor_ref
                    or occurrence_access.get("kind") != occurrence.kind
                    or (occurrence_access.get("predicate") is not None)
                    != occurrence.masked
                ):
                    continue
                matching_rows.append(
                    (occurrence, occurrence_access_ref, occurrence_access)
                )
            if len(matching_rows) != 1:
                continue
            occurrence, access_ref, access = matching_rows[0]
            coordinates: set[tuple[str, int]] = set()
            for domain_ref, coordinate in domain_coordinates.items():
                axis = _index_axis_for_coordinate(
                    access.get("index_expression"),
                    coordinate,
                    extent_symbol=domain_extent_symbols.get(domain_ref),
                )
                if axis is not None:
                    coordinates.add((domain_ref, axis))
            if len(coordinates) != 1:
                continue
            _domain_ref, axis = next(iter(coordinates))
            access_evidence[(carrier_name, access_ref)] = {axis}

    # Close direct address-carrier ownership only after the same Source
    # occurrence ledger has established its program-coordinate evidence.  The
    # earlier pass intentionally tries to prove an axis from each individual
    # access, but a rank-two J index such as ``[d0, d1]`` has two independent
    # coordinates even when the compiled pointer polynomial proves that a
    # scalar carrier is multiplied by the one direct program axis.  Treating
    # that shape as an ambiguity left a Source-used ``p*`` unowned and made
    # SourceReconcile reject an otherwise canonical candidate.
    #
    # This is not a second transfer matcher: every row below starts from the
    # already closed ordinal -> access binding, verifies the exact pointer
    # base and scalar occurrence in that ordinal's address polynomial, and
    # reads the program axis from the same ledger occurrence.  It therefore
    # cannot bind a carrier to an access selected merely by tensor/kind/order.
    def ledger_program_axis(
        occurrence: TransferOccurrence,
        access: Mapping[str, Any],
    ) -> int | None:
        physical_axes = _source_program_axes(occurrence.address_terms)
        if len(physical_axes) != 1:
            return None
        physical_axis = physical_axes[0]
        witnessed = program_evidence.get(physical_axis, set())
        if len(witnessed) == 1:
            _domain_ref, logical_axis = next(iter(witnessed))
            return logical_axis
        authored = program_axis_map.get(physical_axis, set())
        axes: set[int] = set()
        for domain_ref, axis_ref in authored:
            if axis_ref is not None:
                axes.add(axis_ref)
                continue
            coordinate = domain_coordinates.get(domain_ref)
            if coordinate is None:
                continue
            resolved = _index_axis_for_coordinate(
                access.get("index_expression"),
                coordinate,
                extent_symbol=domain_extent_symbols.get(domain_ref),
            )
            if resolved is not None:
                axes.add(resolved)
        return next(iter(axes)) if len(axes) == 1 else None

    for occurrence in shared_ledger.occurrences:
        ledger_row = ledger_by_ordinal.get(occurrence.ordinal)
        access_ref = (
            ledger_row.get("access_ref")
            if isinstance(ledger_row, Mapping)
            else None
        )
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        if not isinstance(access_ref, str) or not isinstance(access, Mapping):
            continue
        base_position = occurrence.pointer_position
        base_parameter = runtime_parameters.get(base_position)
        if (
            not isinstance(base_position, int)
            or not isinstance(base_parameter, Mapping)
            or not isinstance(base_parameter.get("logical_ref"), str)
            or access.get("tensor") != base_parameter.get("logical_ref")
            or access.get("kind") != occurrence.kind
            or bool(access.get("predicate") is not None) != occurrence.masked
        ):
            continue
        logical_axis = ledger_program_axis(occurrence, access)
        if logical_axis is None:
            continue
        for position in _source_argument_positions(occurrence.address_terms):
            if position == base_position:
                continue
            carrier = runtime_parameters.get(position)
            carrier_name = carrier.get("name") if isinstance(carrier, Mapping) else None
            if (
                not isinstance(carrier_name, str)
                or carrier.get("kind") != "scalar"
                or carrier.get("logical_ref") is not None
            ):
                continue
            access_evidence.setdefault((carrier_name, access_ref), set()).add(
                logical_axis
            )

    diagnostics: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    bound = dict(pi) if isinstance(pi, Mapping) else pi
    if not isinstance(bound, dict):
        result["status"] = "unavailable"
        return result

    raw_program_bindings = bound.get("program_bindings")
    raw_program_axes = bound.get("program_axes")
    if not isinstance(raw_program_bindings, list) or not isinstance(
        raw_program_axes, list
    ):
        result["status"] = "unavailable"
        return result
    if raw_program_bindings and raw_program_axes:
        result["status"] = "unavailable"
        return result

    def existing_program_fact(
        physical_axis: int,
    ) -> set[tuple[str, int]]:
        facts: set[tuple[str, int]] = set()
        if raw_program_bindings:
            for row in raw_program_bindings:
                if not isinstance(row, Mapping) or row.get("axis") != physical_axis:
                    continue
                for logical_axis in row.get("logical_axes", []):
                    if not isinstance(logical_axis, Mapping):
                        continue
                    domain_ref = logical_axis.get("domain_ref")
                    axis = _axis_ref_number(logical_axis.get("axis_ref"))
                    if isinstance(domain_ref, str) and axis is not None:
                        facts.add((domain_ref, axis))
        else:
            for row in raw_program_axes:
                if not isinstance(row, Mapping) or row.get("axis") != physical_axis:
                    continue
                domain_ref = row.get("domain_ref")
                coordinate = domain_coordinates.get(domain_ref)
                if isinstance(domain_ref, str) and coordinate is not None:
                    facts.add((domain_ref, coordinate))
        return facts

    for physical_axis in sorted(program_evidence):
        facts = program_evidence[physical_axis]
        if len(facts) != 1:
            continue
        expected = next(iter(facts))
        existing = existing_program_fact(physical_axis)
        if existing and expected not in existing:
            diagnostics.append(
                {
                    "rule_code": "source_pi_program_binding_conflict",
                    "path": "pi.program_bindings" if raw_program_bindings else "pi.program_axes",
                    "anonymous_handles": [expected[0]],
                    "observed_refs_or_types": [
                        f"source physical axis={physical_axis}",
                        f"source-bound J fact={expected[0]}:{expected[1]}",
                        f"authored facts={sorted(existing)}",
                    ],
                }
            )
            continue
        if existing:
            continue
        domain_ref, axis = expected
        if raw_program_bindings:
            raw_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": "direct",
                    "logical_axes": [
                        {"domain_ref": domain_ref, "axis_ref": axis}
                    ],
                }
            )
            applied.append(
                {
                    "family": "program_bindings",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                    "relation": "direct",
                }
            )
        elif not raw_program_axes and not raw_program_bindings:
            # Prefer the relation form for newly produced rows.  It carries
            # the same direct fact and is consumed by Replay projection; the
            # later consumer-unification step handles both legacy forms.
            raw_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": "direct",
                    "logical_axes": [
                        {"domain_ref": domain_ref, "axis_ref": axis}
                    ],
                }
            )
            applied.append(
                {
                    "family": "program_bindings",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                    "relation": "direct",
                }
            )
        else:
            raw_program_axes.append(
                {"domain_ref": domain_ref, "axis": physical_axis}
            )
            applied.append(
                {
                    "family": "program_axes",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                }
            )

    expected_rows_by_carrier: dict[str, set[tuple[str, int]]] = {}
    for (carrier, access_ref), axes in access_evidence.items():
        if len(axes) == 1:
            expected_rows_by_carrier.setdefault(carrier, set()).add(
                (access_ref, next(iter(axes)))
            )

    raw_access_bindings = bound.get("access_representation_bindings")
    if not isinstance(raw_access_bindings, list):
        result["status"] = "unavailable"
        return result

    # When exact Source syntax says which runtime carriers are used, erase
    # only impossible authored ownership rows before rebuilding the uniquely
    # observed rows below.  This prevents a tensor base p*, an unused scalar,
    # or a conflicting guessed access from competing with the Source ledger.
    # A used carrier with no unique observed occurrence is retained: fixed
    # code must not guess a route in that case.
    if source_used_carriers:
        cleaned_rows: list[Any] = []
        removed_rows = 0
        for row in raw_access_bindings:
            if not isinstance(row, Mapping):
                cleaned_rows.append(row)
                continue
            carrier = row.get("carrier_ref")
            access_ref = row.get("access_ref")
            axis = _axis_ref_number(row.get("axis_ref"))
            parameter_match = (
                next(
                    (
                        parameter
                        for parameter in parameters_by_position.values()
                        if parameter.get("name") == carrier
                    ),
                    None,
                )
                if isinstance(carrier, str)
                else None
            )
            legal_carrier = (
                isinstance(parameter_match, Mapping)
                and parameter_match.get("kind") == "scalar"
                and parameter_match.get("logical_ref") is None
                and carrier in source_used_carriers
            )
            expected_rows = (
                expected_rows_by_carrier.get(carrier, set())
                if isinstance(carrier, str)
                else set()
            )
            exact_row = (
                isinstance(access_ref, str)
                and isinstance(axis, int)
                and (access_ref, axis) in expected_rows
            )
            if not legal_carrier or (expected_rows and not exact_row):
                removed_rows += 1
                continue
            cleaned_rows.append(row)
        if removed_rows:
            raw_access_bindings[:] = cleaned_rows
            applied.append(
                {
                    "family": "access_representation_bindings",
                    "action": (
                        "remove impossible or Source-conflicting authored carrier rows "
                        "before reconstructing unique Source transfer ownership"
                    ),
                }
            )

    existing_access_rows = {
        (
            str(row.get("carrier_ref")),
            str(row.get("access_ref")),
            _axis_ref_number(row.get("axis_ref")),
        )
        for row in bound.get("access_representation_bindings", [])
        if isinstance(row, Mapping)
    }
    existing_access_pair_axes: dict[tuple[str, str], set[int]] = {}
    for carrier, access_ref, axis in existing_access_rows:
        if axis is not None:
            existing_access_pair_axes.setdefault((carrier, access_ref), set()).add(axis)

    for (carrier, access_ref), axes in sorted(access_evidence.items()):
        if len(axes) != 1 or access_ref not in access_by_id:
            continue
        axis = next(iter(axes))
        existing_axes = existing_access_pair_axes.get((carrier, access_ref), set())
        if existing_axes and axis not in existing_axes:
            diagnostics.append(
                {
                    "rule_code": "source_pi_access_binding_conflict",
                    "path": "pi.access_representation_bindings",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        f"source tensor axis={axis}",
                        f"authored axes={sorted(existing_axes)}",
                    ],
                }
            )
            continue
        row_key = (carrier, access_ref, axis)
        if row_key in existing_access_rows:
            continue
        raw_access_bindings.append(
            {
                "relation": "address_recomputed",
                "access_ref": access_ref,
                "carrier_ref": carrier,
                "axis_ref": axis,
            }
        )
        existing_access_rows.add(row_key)
        applied.append(
            {
                "family": "access_representation_bindings",
                "relation": "address_recomputed",
                "access_ref": access_ref,
                "carrier_ref": carrier,
                "axis_ref": axis,
            }
        )

    if diagnostics:
        result["status"] = "fail"
        result["diagnostics"] = diagnostics
        return result
    if applied:
        original["pi"] = bound
        result["artifact"] = original
    result["applied"] = applied
    return result


def reconcile_candidate_source_ttir(
    candidate_text: str,
    observation: Mapping[str, Any] | None,
    *,
    construction: CanonicalConstruction,
) -> dict[str, Any]:
    """Diagnose candidate-authored obligations against Source syntax and TTIR.

    This runs only inside source-aware Lift.  It neither constructs artifact
    fields nor contributes to TypeCheck, Gate1, or certification.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
        "authority": (
            "source-aware diagnostic comparison plus deterministic Source-to-pi binding "
            "of model-authored closed J/compact-pi obligations with the existing anonymous "
            "Source TTIR observation; J/C are never constructed or changed, and no Gate "
            "or certification authority is granted"
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
        python_interface = observation.get("python_interface")
        fingerprint = fingerprint_ttir(
            ttir,
            python_interface=(
                python_interface if isinstance(python_interface, Mapping) else None
            ),
        )
        source_facts = None
        raw_source_facts = observation.get("source_facts")
        if isinstance(raw_source_facts, Mapping):
            source_facts = SourceFacts.from_json(raw_source_facts)
        if source_facts is None:
            raise ValueError("Source observation lacks canonical SourceFacts lineage")
        if source_facts.digest != construction.source_facts.digest:
            raise ValueError("Source observation facts differ from canonical construction")
        construction_ledger = construction.with_observation(fingerprint).transfer_ledger
        # Build one occurrence ledger for this candidate.  Fill validation,
        # pi binding, and effect-order diagnostics all consume this exact
        # object; no later helper rematches a Source transfer independently.
        transfer_ledger = reorder_boundary_effects_from_source_ledger(
            packet,
            fingerprint,
            ledger=construction_ledger,
            source_facts=construction.source_facts,
        )
        binding = bind_source_pi(
            packet,
            fingerprint,
            source_runtime_parameter_uses=observation.get(
                "source_runtime_parameter_uses"
            ),
            ledger=transfer_ledger,
            source_facts=construction.source_facts,
        )
        bound_packet = binding.get("artifact", packet)
        if not isinstance(bound_packet, Mapping):
            bound_packet = packet
        binding_diagnostics = tuple(
            item
            for item in binding.get("diagnostics", [])
            if isinstance(item, Mapping)
        )
        reorder_diagnostics = tuple(
            item
            for item in transfer_ledger.get("diagnostics", ())
            if isinstance(item, Mapping)
        ) if isinstance(transfer_ledger, Mapping) else ()
        if reorder_diagnostics:
            binding_diagnostics = (*reorder_diagnostics, *binding_diagnostics)
        diagnostics = compiled_packet_obligation_diagnostics(
            bound_packet,
            ttir,
            fingerprint=fingerprint,
            path_prefix="<source-ttir-obligation>",
        )
        diagnostics = (
            *diagnostics,
            *_compact_j_mask_presence_diagnostics(bound_packet, fingerprint),
            *_source_access_fill_diagnostics(
                bound_packet,
                fingerprint,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *_compact_j_recurrence_presence_diagnostics(bound_packet, fingerprint),
            *_compact_j_terminal_control_presence_diagnostics(bound_packet, fingerprint),
            *_source_boundary_effect_order_diagnostics(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *validate_source_semantic_authorizations(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *_source_stride_axis_diagnostics(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *_unowned_source_runtime_carrier_diagnostics(
                bound_packet,
                ttir,
                source_runtime_parameter_uses=observation.get(
                    "source_runtime_parameter_uses"
                ),
            ),
            *binding_diagnostics,
        )
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        failing_function = frames[-1].name if frames else "<unknown>"
        return {
            **base,
            "status": "fail",
            "applicable": True,
            "diagnostics": [
                {
                    "rule_code": "source_reconcile_observer_failed",
                    "path": "<source-ttir-obligation>",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [
                        _closed_observer_exception(exc),
                        f"observer_function={failing_function}",
                    ],
                    "line": None,
                }
            ],
        }
    result = {
        **base,
        "status": "fail" if diagnostics else "pass",
        "applicable": True,
        "diagnostics": [dict(item) for item in diagnostics],
    }
    if isinstance(transfer_ledger, Mapping):
        shared_ledger = transfer_ledger.get("source_transfer_ledger")
        if isinstance(shared_ledger, Mapping):
            result["source_transfer_ledger"] = dict(shared_ledger)
    if binding.get("applied"):
        result["source_pi_binding"] = {
            "status": binding.get("status", "pass"),
            "applied": [
                dict(item)
                for item in binding["applied"]
                if isinstance(item, Mapping)
            ],
        }
    elif binding.get("status") == "fail":
        result["source_pi_binding"] = {
            "status": "fail",
            "applied": [],
        }
    return result


def _source_boundary_effect_order_diagnostics(
    packet: Mapping[str, Any],
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject a J boundary ledger whose transfer order contradicts Source.

    ``boundary.effects`` is the ordered observable memory ledger.  The old
    source-aware preflight checked transfer presence and several one-bit
    obligations, but it did not compare this ordering when the same source
    tensors had multiple accesses.  That allowed an artifact to pass Lift and
    TypeCheck even though Replay was required to emit stores in an order that
    the selected Source never uses.

    The SourceTransferLedger is the only occurrence-to-access authority.  This
    diagnostic never falls back to TTIR origin scanning: without a complete
    ledger binding it is unavailable and the formal Gate1 observer remains
    authoritative.  It never constructs or repairs J/C/pi and it does not
    compare arithmetic or address spelling.
    """

    j = packet.get("j")
    if not isinstance(j, Mapping):
        return ()
    boundary = j.get("boundary")
    raw_effects = (
        boundary.get("effects") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(raw_effects, list):
        return ()
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass" or not shared_ledger.bindings:
        return ()
    source_access_order = [
        row.get("access_ref") for row in shared_ledger.bindings
    ]
    authored_access_order = [
        row.get("access_ref")
        for row in raw_effects
        if isinstance(row, Mapping) and isinstance(row.get("access_ref"), str)
    ]
    if (
        not all(isinstance(item, str) for item in source_access_order)
        or len(authored_access_order) != len(source_access_order)
    ):
        return ()
    first_difference = next(
        (
            index
            for index, (expected_access, observed_access) in enumerate(
                zip(authored_access_order, source_access_order, strict=True)
            )
            if expected_access != observed_access
        ),
        None,
    )
    if first_difference is None:
        return ()
    expected_access = authored_access_order[first_difference]
    observed_access = source_access_order[first_difference]
    return (
        {
            "rule_code": "source_boundary_effect_order",
            "path": (
                "<source-ttir-obligation> j.boundary.effects: "
                f"access {expected_access!r} is at ordinal {first_difference}, "
                "but the shared Source transfer ledger resolves a different "
                "logical occurrence"
            ),
            "anonymous_handles": [expected_access, observed_access],
            "observed_refs_or_types": [
                f"expected_access={expected_access}",
                f"observed_access={observed_access}",
                f"ordinal={first_difference}",
            ],
            "source_transfer_order": [
                {"ordinal": index, "access_ref": access_ref}
                for index, access_ref in enumerate(source_access_order)
            ],
            "line": None,
        },
    )


def _split_candidate_index_expression(value: object) -> tuple[str, ...] | None:
    """Split one closed J index expression without interpreting its values."""

    if not isinstance(value, str):
        return None
    expression = value.strip()
    if expression.startswith("[") and expression.endswith("]"):
        expression = expression[1:-1].strip()
    if not expression:
        return None
    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"[": "]", "(": ")", "{": "}"}
    quote: str | None = None
    escaped = False
    for index, char in enumerate(expression):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            item = expression[start:index].strip()
            if not item:
                return None
            pieces.append(item)
            start = index + 1
    if quote is not None or stack:
        return None
    item = expression[start:].strip()
    if not item:
        return None
    pieces.append(item)
    return tuple(pieces)


def _candidate_index_profile(
    index_expression: object,
) -> tuple[int, int, tuple[int, ...]] | None:
    """Return ``(rank, dynamic_components, nonzero_literals)`` for one J index.

    The profile is deliberately syntactic.  A non-literal component is only
    classified as dynamic; this helper never decides which J value it names.
    """

    components = _split_candidate_index_expression(index_expression)
    if components is None:
        return None
    literals: list[int] = []
    dynamic = 0
    for component in components:
        if re.fullmatch(r"[+-]?[0-9]+", component) is not None:
            literals.append(int(component))
        else:
            dynamic += 1
    return len(components), dynamic, tuple(sorted(item for item in literals if item != 0))


def _candidate_tensor_ranks(packet: Mapping[str, Any]) -> dict[str, int] | None:
    """Collect ranks from the already-authored J tensor declarations."""

    j = packet.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    ranks: dict[str, int] = {}
    for field in ("inputs", "outputs"):
        values = function.get(field)
        if not isinstance(values, list):
            return None
        for item in values:
            if not isinstance(item, Mapping):
                return None
            name = item.get("name")
            shape = item.get("shape")
            if not isinstance(name, str) or not isinstance(shape, list):
                return None
            rank = len(shape)
            previous = ranks.get(name)
            if previous is not None and previous != rank:
                return None
            ranks[name] = rank
    return ranks


def _candidate_node_contexts(packet: Mapping[str, Any]) -> dict[str, tuple[str, ...]] | None:
    """Project nested J operation regions to a conservative context class."""

    j = packet.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    contexts: dict[str, tuple[str, ...]] = {}

    def visit(body: object, context: tuple[str, ...]) -> bool:
        if not isinstance(body, list):
            return False
        for operation in body:
            if not isinstance(operation, Mapping):
                return False
            node_id = operation.get("id")
            if not isinstance(node_id, str):
                return False
            contexts[node_id] = context
            op_name = operation.get("op")
            if op_name == "loop":
                if not visit(operation.get("body"), (*context, "loop")):
                    return False
            elif op_name in {"effect_if", "value_if"}:
                for polarity in ("positive", "negative"):
                    arm = operation.get(polarity)
                    if not isinstance(arm, Mapping):
                        return False
                    if not visit(
                        arm.get("body"),
                        (*context, f"branch:{polarity}"),
                    ):
                        return False
        return True

    return contexts if visit(function.get("body"), ()) else None


def _source_context_class(context: object) -> tuple[str, ...] | None:
    if not isinstance(context, (list, tuple)):
        return None
    result: list[str] = []
    for token in context:
        if not isinstance(token, str):
            return None
        if token.startswith("scf.for@"):
            result.append("loop")
            continue
        if token.startswith("scf.if@"):
            result.append("branch")
            continue
        if token.startswith("structured_if_arm="):
            polarity = token.partition("=")[2]
            if polarity not in {"positive", "negative"}:
                return None
            result.append(f"branch:{polarity}")
            continue
        if token.startswith("terminal_fork_arm="):
            polarity = token.partition("=")[2]
            if polarity == "true":
                polarity = "positive"
            elif polarity == "false":
                polarity = "negative"
            else:
                return None
            result.append(f"branch:{polarity}")
            continue
        return None
    return tuple(result)


def _source_j_context_compatible(
    source_context: object,
    j_context: object,
) -> bool:
    source_class = _source_context_class(source_context)
    if not isinstance(j_context, tuple):
        return False
    if source_class is None:
        return False
    # ``scf.if`` and its arm marker are both present in the Source context;
    # collapse the owner marker because J's arm context already implies it.
    collapsed_source: list[str] = []
    for token in source_class:
        if token == "branch" and collapsed_source and collapsed_source[-1].startswith(
            "branch:"
        ):
            continue
        collapsed_source.append(token)
    collapsed_j: list[str] = []
    for token in j_context:
        if token == "branch" and collapsed_j and collapsed_j[-1].startswith("branch:"):
            continue
        collapsed_j.append(token)
    return tuple(collapsed_source) == tuple(collapsed_j)


def reorder_boundary_effects_from_source_ledger(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
    source_order_sealed: bool = False,
) -> dict[str, Any]:
    """Repair a uniquely witnessed J effect order from the Source ledger.

    This is a bounded ingress repair, not a semantic synthesizer.  It only
    moves already-authored access-bound effect objects when every Source
    transfer has one unique J access match using ABI ownership, transfer kind,
    mask state, direct address/index shape, and region context.  Ambiguous or
    incomplete candidates are returned unchanged and remain reject-only
    diagnostics for SourceReconcile/Gate1.
    """

    original = copy.deepcopy(dict(packet))
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-effect-order-repair.v1",
        "status": "unavailable",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
        # This remains an ingress-local observation.  It is never inserted
        # into J/C/pi or a persisted replay packet.
        "source_access_matches": (),
        "source_access_match_reason": "source_memory_effect_ledger_unavailable",
        "source_transfer_ledger": shared_ledger.to_json(),
    }

    def reject(
        rule_code: str,
        path: str,
        *observed: str,
        status: str = "unavailable",
    ) -> dict[str, Any]:
        """Return one bounded structural ledger failure without payload text."""

        safe_observed = [
            value
            for value in observed
            if isinstance(value, str)
            and value
            and len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
        ]
        result["status"] = status
        result["source_access_match_reason"] = rule_code
        result["source_access_matches"] = ()
        result["diagnostics"] = [
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": safe_observed or ["closed=0"],
            }
        ]
        return result

    if shared_ledger.status != "pass":
        result["status"] = shared_ledger.status
        result["source_access_match_reason"] = shared_ledger.reason
        result["diagnostics"] = [dict(item) for item in shared_ledger.diagnostics]
        return result

    j = original.get("j")
    if not isinstance(j, Mapping):
        return reject("source_transfer_candidate_j_missing", "j")
    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses") if isinstance(boundary, Mapping) else None
    )
    raw_effects = (
        boundary.get("effects") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(raw_parameters, list) or not isinstance(raw_accesses, list):
        return reject(
            "source_transfer_boundary_rows_missing",
            "j.boundary.accesses",
            f"parameter_rows={len(raw_parameters) if isinstance(raw_parameters, list) else -1}",
            f"access_rows={len(raw_accesses) if isinstance(raw_accesses, list) else -1}",
        )
    if not isinstance(raw_effects, list):
        return reject(
            "source_transfer_boundary_rows_missing",
            "j.boundary.effects",
            "effect_rows=-1",
        )

    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return reject(
            "source_runtime_parameter_projection_unavailable",
            "j.signature.parameters",
            "projection=runtime_physical_abi",
        )
    runtime_parameters, _physical_parameters, _runtime_position_by_name = (
        parameter_projection
    )
    logical_parameter_positions: dict[str, list[int]] = {}
    for runtime_position, parameter in runtime_parameters.items():
        logical_ref = parameter.get("logical_ref")
        if isinstance(logical_ref, str):
            logical_parameter_positions.setdefault(logical_ref, []).append(
                runtime_position
            )

    accesses: dict[str, Mapping[str, Any]] = {}
    for row_index, access in enumerate(raw_accesses):
        if not isinstance(access, Mapping):
            return reject(
                "source_transfer_access_row_malformed",
                f"j.boundary.accesses[{row_index}]",
                f"row_index={row_index}",
            )
        access_id = access.get("access_id")
        if (
            not isinstance(access_id, str)
            or access_id in accesses
            or access.get("kind") not in {"load", "store", "update"}
            or not isinstance(access.get("tensor"), str)
        ):
            return reject(
                "source_transfer_access_identity_invalid",
                f"j.boundary.accesses[{row_index}]",
                f"row_index={row_index}",
                f"duplicate_access_id={int(isinstance(access_id, str) and access_id in accesses)}",
            )
        accesses[access_id] = access

    effect_by_access: dict[str, Mapping[str, Any]] = {}
    effect_ids: set[str] = set()
    access_effect_positions: list[tuple[int, str]] = []
    non_access_effect_positions: list[int] = []
    for index, effect in enumerate(raw_effects):
        if not isinstance(effect, Mapping):
            return reject(
                "source_transfer_effect_row_malformed",
                f"j.boundary.effects[{index}]",
                f"row_index={index}",
            )
        effect_id = effect.get("effect_id")
        if (
            not isinstance(effect_id, str)
            or re.fullmatch(r"e[0-9]+", effect_id) is None
            or effect_id in effect_ids
        ):
            # Scope membership is expressed in stable effect ids.  Require
            # those ids for every effect, including non-access effects, before
            # moving any access-bound row; otherwise a later scope rewrite
            # could silently drop or duplicate an effect that this helper did
            # not observe as a memory transfer.
            return reject(
                "source_transfer_effect_identity_invalid",
                f"j.boundary.effects[{index}]",
                f"row_index={index}",
                f"duplicate_effect_id={int(isinstance(effect_id, str) and effect_id in effect_ids)}",
            )
        effect_ids.add(effect_id)
        access_ref = effect.get("access_ref")
        if access_ref is None:
            non_access_effect_positions.append(index)
            continue
        if (
            not isinstance(access_ref, str)
            or access_ref in effect_by_access
            or access_ref not in accesses
        ):
            return reject(
                "source_transfer_effect_access_binding_invalid",
                f"j.boundary.effects[{index}].access_ref",
                f"row_index={index}",
                f"duplicate_access_effect={int(isinstance(access_ref, str) and access_ref in effect_by_access)}",
                f"declared_access={int(isinstance(access_ref, str) and access_ref in accesses)}",
            )
        access = accesses[access_ref]
        expected_effect_kind = {
            "load": "read",
            "store": "write",
            "update": "update",
        }[str(access["kind"])]
        if effect.get("kind") != expected_effect_kind:
            return reject(
                "source_transfer_effect_kind_mismatch",
                f"j.boundary.effects[{index}].kind",
                f"row_index={index}",
                f"access_ref={access_ref}",
                f"expected_kind={expected_effect_kind}",
            )
        if effect.get("target") != access.get("tensor"):
            return reject(
                "source_transfer_effect_owner_mismatch",
                f"j.boundary.effects[{index}].target",
                f"row_index={index}",
                f"access_ref={access_ref}",
            )
        if effect.get("node_ref") != access.get("node_ref"):
            return reject(
                "source_transfer_effect_node_mismatch",
                f"j.boundary.effects[{index}].node_ref",
                f"row_index={index}",
                f"access_ref={access_ref}",
            )
        effect_by_access[access_ref] = effect
        access_effect_positions.append((index, access_ref))

    if set(effect_by_access) != set(accesses):
        return reject(
            "source_transfer_access_effect_coverage_mismatch",
            "j.boundary.effects",
            f"access_count={len(accesses)}",
            f"bound_effect_count={len(effect_by_access)}",
        )
    if not access_effect_positions:
        return reject(
            "source_transfer_access_effect_rows_missing",
            "j.boundary.effects",
            f"access_count={len(accesses)}",
        )

    occurrences = shared_ledger.occurrences
    if len(occurrences) != len(access_effect_positions):
        return reject(
            "source_transfer_occurrence_count_mismatch",
            "j.boundary.accesses",
            f"occurrence_count={len(occurrences)}",
            f"access_effect_count={len(access_effect_positions)}",
        )

    tensor_ranks = _candidate_tensor_ranks(original)
    node_contexts = _candidate_node_contexts(original)
    if tensor_ranks is None or node_contexts is None:
        return reject(
            "source_transfer_candidate_topology_unavailable",
            "j.intent.intent_function.body",
            f"tensor_rank_table={int(tensor_ranks is not None)}",
            f"node_context_table={int(node_contexts is not None)}",
        )

    source_rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        ordinal = occurrence.ordinal
        origin_positions = occurrence.origin_positions
        source_context = occurrence.context
        pointer_base = occurrence.pointer_base
        base_position = occurrence.pointer_position
        if (
            base_position is None
            or base_position not in origin_positions
            or len(logical_parameter_positions.get(
                runtime_parameters.get(base_position, {}).get("logical_ref"), []
            )) != 1
        ):
            return reject(
                "source_effect_physical_abi_ambiguous",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        parameter = runtime_parameters[base_position]
        tensor = parameter.get("logical_ref")
        if (
            not isinstance(tensor, str)
            or not tensor.startswith("t")
            or parameter.get("kind") not in {"pointer", "tensor"}
            or (
                occurrence.logical_ref is not None
                and tensor != occurrence.logical_ref
            )
        ):
            return reject(
                "source_effect_logical_owner_unavailable",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        address_profile = occurrence.address_profile
        if address_profile is None:
            return reject(
                "source_effect_address_not_direct",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        kind = occurrence.kind
        if kind not in {"load", "store", "update"}:
            return reject(
                "source_transfer_occurrence_kind_unsupported",
                "<source-transfer-ledger>.occurrences",
                f"ordinal={ordinal}",
            )
        source_rows.append(
            {
                "ordinal": ordinal,
                "kind": kind,
                "tensor": tensor,
                "masked": occurrence.masked,
                "address_profile": address_profile,
                "address_terms": occurrence.address_terms,
                "pointer_base": pointer_base,
                # The closed Source occurrence owns the physical pointer
                # position used later in the access/pi binding witness.  Keep
                # it alongside its validated base/origin facts; omitting it
                # made the successful unique-match path raise KeyError while
                # constructing ``source_access_matches``.
                "pointer_position": base_position,
                "context": source_context,
            }
        )

    def access_mismatch_reasons(
        source_row: Mapping[str, Any], access_ref: str, access: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Classify why one occurrence cannot bind one authored access."""

        reasons: list[str] = []
        if access_ref not in effect_by_access:
            reasons.append("effect_binding_missing")
        if access.get("kind") != source_row["kind"]:
            reasons.append("kind_mismatch")
        if access.get("tensor") != source_row["tensor"]:
            reasons.append("tensor_owner_mismatch")
        if (access.get("predicate") is not None) != source_row["masked"]:
            reasons.append("predicate_presence_mismatch")
        node_ref = access.get("node_ref")
        j_context = (
            node_contexts.get(node_ref, ()) if isinstance(node_ref, str) else ()
        )
        if isinstance(node_ref, str) and node_ref not in node_contexts:
            reasons.append("node_consumer_missing")
        if not _source_j_context_compatible(source_row["context"], j_context):
            reasons.append("node_context_mismatch")
        rank = tensor_ranks.get(str(access.get("tensor")))
        index_profile = _candidate_index_profile(access.get("index_expression"))
        if rank is None:
            reasons.append("tensor_rank_unavailable")
        if index_profile is None:
            reasons.append("index_expression_unavailable")
        elif rank is not None and index_profile[0] != rank:
            reasons.append("index_rank_mismatch")
        return tuple(reasons)

    def compatible_access(
        source_row: Mapping[str, Any], access_ref: str, access: Mapping[str, Any]
    ) -> bool:
        return not access_mismatch_reasons(source_row, access_ref, access)

    # A sealed MCP semantic entry already establishes the one Source-order
    # occurrence identity: ``declare_accesses`` checks each ordinal/label and
    # the constructor then allocates the dense a0..aN table without changing
    # that row order.  Retain that provenance through this private transaction
    # instead of trying to recover it from address *term* counts after labels
    # have deliberately been erased.  A flattened physical address can have
    # more dynamic terms than the number of logical J index components, so
    # comparing those two counts is not a sound occurrence identity test.
    #
    # This path is deliberately opt-in and is enabled only by the caller that
    # has just run the Source-sized semantic submission validator.  It still
    # requires a unique dense canonical table plus kind, tensor, mask, rank,
    # lexical-context, and access/effect checks for every occurrence.  Other
    # callers continue to use the conservative address-profile matcher below.
    declared_access_order = list(accesses)
    canonical_source_order = [
        f"a{index}" for index in range(len(source_rows))
    ]
    effect_access_order = [access_ref for _index, access_ref in access_effect_positions]
    if (
        source_order_sealed
        and declared_access_order == canonical_source_order
        and effect_access_order == canonical_source_order
    ):
        candidates_by_source = [
            [access_ref]
            if compatible_access(source_row, access_ref, accesses[access_ref])
            else []
            for source_row, access_ref in zip(
                source_rows, canonical_source_order, strict=True
            )
        ]
    else:
        candidates_by_source = []
        for source_row in source_rows:
            source_profile = source_row["address_profile"]
            candidates: list[str] = []
            for access_ref, access in accesses.items():
                if not compatible_access(source_row, access_ref, access):
                    continue
                index_profile = _candidate_index_profile(access.get("index_expression"))
                if index_profile is None:
                    continue
                if index_profile[1] != source_profile[0]:
                    continue
                if index_profile[2] != source_profile[2]:
                    continue
                candidates.append(access_ref)
            candidates_by_source.append(candidates)

    if any(len(candidates) != 1 for candidates in candidates_by_source):
        result["status"] = "ambiguous"
        ambiguous_observations: list[str] = []
        for index, candidates in enumerate(candidates_by_source):
            if len(candidates) == 1:
                continue
            ambiguous_observations.extend(
                (f"ordinal={index}", f"candidate_count={len(candidates)}")
            )
            if not candidates:
                expected_access = (
                    canonical_source_order[index]
                    if source_order_sealed
                    and index < len(canonical_source_order)
                    and canonical_source_order[index] in accesses
                    else None
                )
                if expected_access is not None:
                    mismatch_reasons = access_mismatch_reasons(
                        source_rows[index],
                        expected_access,
                        accesses[expected_access],
                    )
                    ambiguous_observations.extend(
                        f"mismatch={reason}" for reason in mismatch_reasons[:4]
                    )
            if len(ambiguous_observations) >= 12:
                break
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_occurrence_access_ambiguous",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": ambiguous_observations[:12],
            }
        ]
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_access_ambiguous"
        )
        return result

    desired_access_order = [candidates[0] for candidates in candidates_by_source]
    if len(set(desired_access_order)) != len(desired_access_order):
        result["status"] = "ambiguous"
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_occurrence_access_not_one_to_one",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": [
                    f"occurrence_count={len(desired_access_order)}",
                    f"unique_access_count={len(set(desired_access_order))}",
                ],
            }
        ]
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_access_not_one_to_one"
        )
        return result

    # Gate1 canonicalizes ``memory_effects`` by sorting it, so its order is
    # deliberately unusable as an occurrence ledger.  This source-only
    # sidecar retains the lexical order shared by the provenance/address
    # ledgers above; atomics and incomplete parser views remain reject-only.
    source_transfers = shared_ledger.occurrences
    # The occurrence ledger issued owner/guard authorization before a candidate
    # existed.  Reorder changes only the candidate-local access/effect rows;
    # it must carry each already-issued receipt through the same ordinal rather
    # than dropping it (or, worse, deriving a replacement from an access row).
    # A pass ledger has already validated one complete, contiguous binding set,
    # but recheck that narrow shape here before its sidecar crosses the private
    # reorder transaction.
    authorizations_by_ordinal: dict[int, Mapping[str, Any]] = {}
    for binding in shared_ledger.bindings:
        ordinal = binding.get("ordinal") if isinstance(binding, Mapping) else None
        authorizations = (
            binding.get("semantic_authorizations")
            if isinstance(binding, Mapping)
            else None
        )
        if (
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and ordinal not in authorizations_by_ordinal
            and isinstance(authorizations, Mapping)
        ):
            authorizations_by_ordinal[ordinal] = copy.deepcopy(dict(authorizations))
        else:
            return reject(
                "source_transfer_authorization_projection_unavailable",
                "<source-transfer-ledger>.bindings",
                "expected=one_authorization_receipt_per_ordinal",
            )
    if set(authorizations_by_ordinal) != set(range(len(source_transfers))):
        return reject(
            "source_transfer_authorization_projection_unavailable",
            "<source-transfer-ledger>.bindings",
            "expected=closed_contiguous_authorization_receipts",
        )
    if len(source_transfers) == len(source_rows) and all(
        effect.kind == row["kind"] and effect.masked == row["masked"]
        for effect, row in zip(source_transfers, source_rows)
    ):
        result["source_access_matches"] = tuple(
            {
                "ordinal": row["ordinal"],
                "access_ref": access_ref,
                "kind": row["kind"],
                "tensor": row["tensor"],
                "masked": row["masked"],
                "source_other": effect.fill_value,
                # This is Source-side provenance.  It must remain separate
                # from the J predicate, whose canonical spelling may be a
                # typed/indexed expression rather than the Source spelling.
                "source_predicate": effect.predicate,
                "source_pointer_base": row["pointer_base"],
                "source_element_type": effect.element_type,
                "source_address_type_profile": effect.address_type_profile,
                "address_profile": row["address_profile"],
                "index_expression": {
                    "address_terms": [
                        {"refs": list(atoms), "coefficient": coefficient}
                        for atoms, coefficient in row["address_terms"]
                    ],
                    "address_profile": {
                        "dynamic_terms": row["address_profile"][0],
                        "constant_terms": row["address_profile"][1],
                        "constant_offsets": list(row["address_profile"][2]),
                    },
                },
                # This comprehension runs after the Source-row scan above,
                # where ``ordinal`` otherwise retains the final occurrence's
                # value.  Bind each row to its own Source occurrence order.
                "effect_order": row["ordinal"],
                "node_ref": accesses[access_ref].get("node_ref"),
                "predicate": accesses[access_ref].get("predicate"),
                "fill_value": accesses[access_ref].get("fill_value"),
                "effect_id": effect_by_access[access_ref].get("effect_id"),
                "semantic_authorizations": authorizations_by_ordinal[
                    row["ordinal"]
                ],
                "pi_binding": {
                    "status": "source_address_owner_bound",
                    "access_ref": access_ref,
                    "effect_id": effect_by_access[access_ref].get("effect_id"),
                    "tensor": row["tensor"],
                    "pointer_position": row["pointer_position"],
                    "element_type": effect.element_type,
                    "address_type_profile": [
                        {
                            "address_terms": [
                                {"refs": list(atoms), "coefficient": coefficient}
                                for atoms, coefficient in terms
                            ],
                            "index_type": index_type,
                            "ssa_type_chain": list(chain),
                        }
                        for terms, index_type, chain in effect.address_type_profile
                    ],
                    "address_profile": {
                        "dynamic_terms": row["address_profile"][0],
                        "constant_terms": row["address_profile"][1],
                        "constant_offsets": list(row["address_profile"][2]),
                    },
                },
            }
            for row, access_ref, effect in zip(
                source_rows,
                desired_access_order,
                source_transfers,
            )
        )
        result["source_access_match_reason"] = (
            "unique_J_access_typed_address_root_and_transfer_occurrence"
        )
        shared_ledger = shared_ledger.with_bindings(
            result["source_access_matches"],
            status="pass",
            reason=result["source_access_match_reason"],
        )
        result["source_transfer_ledger"] = shared_ledger.to_json()

    current_access_order = [access_ref for _index, access_ref in access_effect_positions]
    # ``pass`` is reserved for a closed occurrence -> access/effect binding.
    # An equal effect order is not sufficient: the source rows may still have
    # disagreed in kind/mask or the ledger may not have been closed at all.
    # Returning pass here used to let a binding-less occurrence ledger reach
    # fill validation and pi normalization.
    matches = result.get("source_access_matches")
    if not isinstance(matches, (list, tuple)) or len(matches) != len(source_transfers):
        result["status"] = "ambiguous"
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_binding_unavailable"
        )
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_ledger_bindings_required",
                "path": "<source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    f"occurrences={len(source_transfers)}",
                    f"bindings={len(matches) if isinstance(matches, (list, tuple)) else 0}",
                ],
            }
        ]
        result["source_access_matches"] = ()
        return result
    current_access_declaration_order = list(accesses)
    if (
        current_access_order == desired_access_order
        and current_access_declaration_order == desired_access_order
    ):
        result["status"] = "pass"
        return result

    # Keep every non-access effect at its existing slot.  If an access would
    # cross such an effect, its relative event position is no longer proven.
    desired_at_access_slot = dict(
        zip((index for index, _ref in access_effect_positions), desired_access_order)
    )
    current_prefix: set[str] = set()
    desired_prefix: set[str] = set()
    for index, effect in enumerate(raw_effects):
        if index in desired_at_access_slot:
            current_prefix.add(str(effect["access_ref"]))
            desired_prefix.add(desired_at_access_slot[index])
        elif index in non_access_effect_positions and current_prefix != desired_prefix:
            result["status"] = "ambiguous"
            result["diagnostics"] = [
                {
                    "rule_code": "source_effect_non_access_order_barrier",
                    "path": "j.boundary.effects",
                    "observed_refs_or_types": [
                        f"effect_row={index}",
                        "barrier=non_access_effect",
                    ],
                }
            ]
            # The provisional bijection was sufficient to explain the
            # ambiguity, but it is not an authority once a non-access effect
            # blocks the proven reorder.  Remove both projections so a future
            # consumer cannot accidentally use a half-closed ledger.
            result["source_access_matches"] = ()
            shared_ledger = shared_ledger.with_bindings(
                (),
                status="ambiguous",
                reason="source_effect_non_access_order_barrier",
                diagnostics=result["diagnostics"],
            )
            result["source_transfer_ledger"] = shared_ledger.to_json()
            return result

    desired_effects = {
        access_ref: copy.deepcopy(effect_by_access[access_ref])
        for access_ref in desired_access_order
    }
    rewritten_effects: list[dict[str, Any]] = []
    desired_index = 0
    for effect in raw_effects:
        access_ref = effect.get("access_ref") if isinstance(effect, Mapping) else None
        if isinstance(access_ref, str):
            rewritten_effects.append(desired_effects[desired_access_order[desired_index]])
            desired_index += 1
        else:
            rewritten_effects.append(copy.deepcopy(dict(effect)))

    rewritten = copy.deepcopy(original)
    rewritten_j = copy.deepcopy(dict(j))
    rewritten_boundary = copy.deepcopy(dict(boundary))
    # The occurrence ledger owns not only the effect sequence but also the
    # access declaration sequence from which the canonical constructor assigns
    # a*/e*.  Leaving accesses in provider order while moving only effects
    # creates two mechanical order authorities and lets the later allocator
    # attach a fresh e* sequence to the wrong a* declaration order.
    access_by_ref = {
        str(access["access_id"]): copy.deepcopy(dict(access))
        for access in raw_accesses
        if isinstance(access, Mapping) and isinstance(access.get("access_id"), str)
    }
    if set(access_by_ref) != set(desired_access_order):
        return reject(
            "source_transfer_reorder_access_set_mismatch",
            "j.boundary.accesses",
            f"declared_access_count={len(access_by_ref)}",
            f"desired_access_count={len(desired_access_order)}",
            status="ambiguous",
        )
    rewritten_boundary["accesses"] = [
        access_by_ref[access_ref]
        for access_ref in desired_access_order
    ]
    rewritten_boundary["effects"] = rewritten_effects
    rewritten_j["boundary"] = rewritten_boundary

    raw_scopes = rewritten_j.get("scopes")
    if isinstance(raw_scopes, list):
        for scope in raw_scopes:
            if not isinstance(scope, Mapping) or not isinstance(scope.get("effects"), list):
                return reject(
                    "source_effect_scope_order_unavailable",
                    "j.scopes",
                    "scope_effect_membership=malformed",
                    status="ambiguous",
                )
            members = list(scope["effects"])
            if any(member not in effect_ids for member in members) or len(members) != len(set(members)):
                return reject(
                    "source_effect_scope_membership_unavailable",
                    "j.scopes.effects",
                    f"member_count={len(members)}",
                    f"unique_member_count={len(set(members))}",
                    status="ambiguous",
                )
            scope["effects"] = [
                str(effect["effect_id"])
                for effect in rewritten_effects
                if effect.get("effect_id") in members
            ]
            if set(scope["effects"]) != set(members):
                return reject(
                    "source_effect_scope_membership_changed",
                    "j.scopes.effects",
                    f"member_count={len(members)}",
                    status="ambiguous",
                )

    rewritten["j"] = rewritten_j
    result["status"] = "pass"
    result["artifact"] = rewritten
    result["applied"] = [
        {
            "rule_code": "deterministic_source_transfer_order_repair",
            "path": "j.boundary.accesses + j.boundary.effects",
            "action": (
                "reorder access declarations and access-bound effects to the "
                "unique Source transfer ledger order"
            ),
            "source_transfer_ordinals": list(range(len(source_rows))),
            "previous_access_order": current_access_order,
            "new_access_order": desired_access_order,
            "authority": (
                "unique physical ABI owner, transfer kind, mask, direct address/index, "
                "and region-context correspondence"
            ),
        }
    ]
    return result


def _direct_source_fill_json_literal(value: object) -> bool | int | float | None:
    """Return one direct, finite JSON scalar from a Source ``other=`` value.

    The TTIR fingerprint intentionally keeps arbitrary SSA fills opaque.  The
    canonical transfer ledger can additionally retain the exact lexical
    Source scalar (for example ``0``) before TTIR lowers it to an SSA value.
    Accept only that same closed scalar vocabulary or one ``arith.constant``
    spelling; vectors, compound dense values, references, and non-finite
    values remain unproved instead of being projected into J.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    dense = re.fullmatch(
        r"arith\.constant\s+dense<(?P<literal>[^<>]+)>\s*:\s*.+",
        text,
    )
    scalar = re.fullmatch(
        r"arith\.constant\s+(?P<literal>[^:\s]+)\s*:\s*.+",
        text,
    )
    match = dense or scalar
    # ``TransferOccurrence.fill_value`` is the Source lexical proof, not a
    # TTIR instruction. A direct scalar has no unsafe SSA/reference content to
    # recover, so it is equally closed evidence for authored-fill validation.
    literal = match.group("literal").strip() if match is not None else text
    if len(literal) > 128:
        return None
    if literal == "true":
        return True
    if literal == "false":
        return False
    if re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?",
        literal,
    ) is None:
        return None
    try:
        if "." not in literal and "e" not in literal.lower():
            return int(literal)
        number = float(literal)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _same_json_scalar(left: object, right: object) -> bool:
    """Compare only the closed scalar vocabulary proven by Source facts."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if isinstance(left, float) and not math.isfinite(left):
        return False
    if isinstance(right, float) and not math.isfinite(right):
        return False
    return left == right


def _direct_source_special_fill(value: object) -> str | None:
    """Recognize one exact non-finite TTIR fill without making it repairable.

    J uses a closed string spelling for non-finite values because JSON has no
    infinity literal.  The compiler observation may nevertheless expose the
    exact IEEE bit pattern as a dense tensor constant.  This helper is only a
    validation witness: callers must not use its result to fill a missing J
    value automatically.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if re.fullmatch(
        r"arith\.constant\s+dense<0xFF800000>\s*:\s*"
        r"tensor<[^>]*f32(?:\s*,[^>]*)?>",
        text,
    ):
        return "-inf"
    return None


def _same_source_fill(left: object, source_value: object) -> bool:
    """Compare authored J fill syntax with one closed Source fill witness."""

    if source_value == "-inf":
        return isinstance(left, str) and left in {"-inf", "NEG_INF"}
    return _same_json_scalar(left, source_value)


def _source_access_fill_validation_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate authored fills against exact Source occurrence evidence."""

    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    if not isinstance(raw_accesses, list):
        return ()
    access_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, access in enumerate(raw_accesses):
        access_ref = access.get("access_id") if isinstance(access, Mapping) else None
        if not isinstance(access_ref, str) or access_ref in access_by_id:
            return ()
        access_by_id[access_ref] = (index, access)
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    source_transfers = shared_ledger.occurrences
    source_loads = {
        ordinal: effect
        for ordinal, effect in enumerate(source_transfers)
        if effect.kind == "load"
    }
    source_masked_loads = {
        ordinal: effect
        for ordinal, effect in source_loads.items()
        if effect.masked
    }
    observed_masked_load_count = 0
    if isinstance(observed, TTIRFingerprint):
        # ``TTIRFingerprint`` deliberately stores multiset rows as an
        # immutable tuple rather than a mapping.  This SourceReconcile
        # consumer needs only a read-only count, so materialize the canonical
        # mapping view here instead of treating the tuple as a second wire
        # representation.  The ledger remains the sole occurrence authority.
        observed_masked_load_count = int(
            dict(observed.op_multiset).get("tt.load.masked", 0)
        )
    # ``TransferOccurrence`` is the shared ledger record.  Its canonical
    # Source ``other=`` spelling is ``fill_value``; the TTIR-side observation
    # objects use ``other`` only while the ledger is being built.  Reading
    # ``effect.other`` here bypasses that contract and raises AttributeError
    # precisely on the unavailable-ledger path that must remain diagnostic.
    source_fill_detail_missing = shared_ledger.status != "pass" and any(
        effect.kind == "load" and effect.fill_value is not None
        for effect in shared_ledger.occurrences
    )
    authored_fills = {
        access_ref
        for access_ref, (_index, access) in access_by_id.items()
        if access.get("fill_value") is not None
    }
    def diagnostic(
        code: str,
        *,
        access_ref: str | None = None,
        index: int | None = None,
        facts: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        path = "<source-ttir-obligation> j.boundary.accesses"
        if index is not None:
            path += f"[{index}].fill_value"
        return {
            "rule_code": code,
            "path": path,
            "anonymous_handles": [access_ref] if access_ref is not None else [],
            "observed_refs_or_types": list(facts),
            "line": None,
        }

    if source_fill_detail_missing:
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source masked-load fill has no lexical transfer occurrence ledger",
                ),
            ),
        )
    if observed_masked_load_count > 0 and shared_ledger.status != "pass":
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source exposes masked loads but the canonical transfer ledger "
                    f"is unavailable: {shared_ledger.reason}",
                ),
            ),
        )
    if observed_masked_load_count > 0 and not source_masked_loads:
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source masked-load observation has no corresponding masked "
                    "occurrence in the canonical transfer ledger",
                ),
            ),
        )
    # A masked load with ``other=None`` is still an occurrence-level
    # obligation.  Returning early based only on ``source_fills`` used to
    # mistake "the Source has no other=" for "there is no fill fact", allowing
    # a missing J fill to reach Gate1.  Continue through the bijection and the
    # per-occurrence presence check below whenever Source exposes a masked
    # load, even when its fill value is absent.
    if not source_masked_loads and not authored_fills:
        return ()

    # ``SourceTransferLedger`` is the occurrence authority, while the
    # reorder helper returns its bounded access-match projection.  Always
    # derive that projection from the same ledger; never call ``.get`` on the
    # dataclass or silently treat a bare ledger JSON as a complete match
    # receipt.
    if (
        shared_ledger.status == "pass"
        and len(shared_ledger.bindings) == len(shared_ledger.occurrences)
        and (not shared_ledger.occurrences or shared_ledger.bindings)
    ):
        # A closed dataclass ledger is already the occurrence authority.  Do
        # not invoke reorder a second time merely to obtain its projection;
        # that would create a fresh access matcher for fill validation.
        source_ledger = {
            "status": "pass",
            "source_access_matches": tuple(
                dict(row) for row in shared_ledger.bindings
            ),
            "diagnostics": tuple(shared_ledger.diagnostics),
        }
    elif isinstance(ledger, Mapping) and "source_access_matches" in ledger:
        # A bare access-match projection is ingress-local diagnostic data, not
        # an occurrence ledger.  It must not become an alternate authority for
        # fill validation merely because its outer status says ``pass``.
        source_ledger = {
            "status": "unavailable",
            "source_access_matches": (),
            "source_access_match_reason": (
                "bare_source_access_projection_is_not_transfer_ledger"
            ),
            "diagnostics": (
                {
                    "rule_code": "source_transfer_ledger_bindings_required",
                    "path": "<source-transfer-ledger>",
                    "observed_refs_or_types": [
                        "bare source_access_matches projection is not authoritative"
                    ],
                },
            ),
        }
    else:
        source_ledger = reorder_boundary_effects_from_source_ledger(
            packet,
            observed,
            ledger=shared_ledger,
            source_facts=source_facts,
        )
    ledger_diagnostics = source_ledger.get("diagnostics")
    if not isinstance(ledger_diagnostics, (list, tuple)):
        ledger_diagnostics = ()
    reason = source_ledger.get("source_access_match_reason") or next(
        (
            item.get("rule_code")
            for item in ledger_diagnostics
            if isinstance(item, Mapping) and isinstance(item.get("rule_code"), str)
        ),
        "source_access_occurrence_unavailable",
    )
    reason = str(reason)

    raw_matches = source_ledger.get("source_access_matches")
    matches_by_ordinal: dict[int, str] = {}
    if source_ledger.get("status") != "pass" or not isinstance(
        raw_matches, (list, tuple)
    ):
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=("unique Source transfer/access occurrence unavailable", reason),
            ),
        )
    for match in raw_matches:
        ordinal = match.get("ordinal") if isinstance(match, Mapping) else None
        access_ref = match.get("access_ref") if isinstance(match, Mapping) else None
        valid = (
            not isinstance(ordinal, bool)
            and isinstance(ordinal, int)
            and 0 <= ordinal < len(source_transfers)
            and isinstance(access_ref, str)
            and access_ref in access_by_id
            and ordinal not in matches_by_ordinal
            and access_ref not in matches_by_ordinal.values()
        )
        if not valid:
            return (
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    facts=("malformed unique Source transfer/access occurrence", reason),
                ),
            )
        effect = source_transfers[ordinal]
        if (
            match.get("kind") != effect.kind
            or match.get("masked") != effect.masked
            or match.get("source_other") != effect.fill_value
            or match.get("source_predicate") != effect.predicate
            or match.get("source_pointer_base") != effect.pointer_base
            or match.get("source_element_type") != effect.element_type
            or match.get("source_address_type_profile")
            != effect.address_type_profile
        ):
            return (
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    facts=("Source transfer occurrence changed after matching", reason),
                ),
            )
        matches_by_ordinal[ordinal] = access_ref

    # A pass ledger is a complete occurrence/access bijection, not merely a
    # list containing the occurrences that happen to carry a fill value.  Keep
    # this check here as well as in SourceTransferLedger.__post_init__ because
    # this helper also accepts the ingress-local mapping projection.
    if (
        set(matches_by_ordinal) != set(range(len(source_transfers)))
        or set(matches_by_ordinal.values()) != set(access_by_id)
    ):
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=("Source transfer/access ledger is not a complete bijection", reason),
            ),
        )

    diagnostics: list[Mapping[str, Any]] = [
        diagnostic(
            "source_access_fill_occurrence_unproved",
            facts=(f"Source masked-load ordinal={ordinal} has other=", reason),
        )
        for ordinal in sorted(set(source_masked_loads) - set(matches_by_ordinal))
    ]
    mapped_accesses = set(matches_by_ordinal.values())
    for access_ref in sorted(authored_fills - mapped_accesses):
        diagnostics.append(
            diagnostic(
                "source_access_fill_occurrence_unproved",
                access_ref=access_ref,
                index=access_by_id[access_ref][0],
                facts=("J fill has no unique Source transfer/access occurrence", reason),
            )
        )
    if diagnostics:
        return tuple(diagnostics)

    for ordinal, access_ref in sorted(matches_by_ordinal.items()):
        effect = source_transfers[ordinal]
        if effect.kind != "load":
            continue
        access_index, access = access_by_id[access_ref]
        if access.get("kind") != "load" or (
            effect.fill_value is not None and not effect.masked
        ):
            diagnostics.append(
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(f"Source load ordinal={ordinal} has no compatible J load",),
                )
            )
            continue
        authored_fill = access.get("fill_value")
        if effect.masked and effect.fill_value is None:
            # A masked Source load is not closed merely because its predicate
            # was observed.  The occurrence must also expose the other fill; when
            # that fact is absent, neither a missing nor an authored J fill is
            # safe to accept.  Keep this occurrence-level and fail closed so a
            # different masked load cannot satisfy it by count.
            diagnostics.append(
                diagnostic(
                    (
                        "source_access_fill_presence_mismatch"
                        if authored_fill is not None
                        else "source_access_fill_presence_unproved"
                    ),
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} has no observed other= fill",
                    ),
                )
            )
            continue
        if effect.fill_value is None:
            if authored_fill is not None:
                diagnostics.append(
                    diagnostic(
                        "source_access_fill_presence_mismatch",
                        access_ref=access_ref,
                        index=access_index,
                        facts=(f"Source load ordinal={ordinal} has no other= operand",),
                    )
                )
            continue
        literal = _direct_source_fill_json_literal(effect.fill_value)
        special_literal = _direct_source_special_fill(effect.fill_value)
        source_literal = literal if literal is not None else special_literal
        if source_literal is None:
            diagnostics.append(
                diagnostic(
                    "source_access_fill_literal_unproved",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} other= is not a direct closed JSON scalar literal",
                    ),
                )
            )
        elif authored_fill is None:
            diagnostics.append(
                diagnostic(
                    "source_masked_load_fill_missing",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} requires one provider-authored J fill",
                    ),
                )
            )
        elif not _same_source_fill(authored_fill, source_literal):
            diagnostics.append(
                diagnostic(
                    "source_access_fill_mismatch",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} and authored J fill differ",
                    ),
                )
            )
    return tuple(diagnostics)


def validate_source_semantic_authorizations(
    packet: Mapping[str, Any],
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject owner/guard claims not issued by the exact Source ledger.

    This is a verifier, not an allocator or semantic repair.  It consumes the
    same post-reorder occurrence/access binding used by SourceReconcile and
    verifies that an opaque authorization token was issued for the exact
    occurrence, then that the candidate's already-authored canonical node has
    the approved family.  It never derives a token from source order, node
    numbering, names, shape, or a provider declaration.
    """

    shared = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared.status != "pass":
        return (
            {
                "rule_code": "semantic_authorization_ledger_unavailable",
                "path": "<source-transfer-ledger>",
                "observed_refs_or_types": ["expected:closed Source authorization ledger"],
            },
        )
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    guard_relations = (
        boundary.get("access_guard_relations") if isinstance(boundary, Mapping)
        else None
    )
    body = function.get("body") if isinstance(function, Mapping) else None
    if not isinstance(accesses, list) or not isinstance(guard_relations, list) or not isinstance(body, list):
        return (
            {
                "rule_code": "semantic_authorization_candidate_topology_unavailable",
                "path": "j",
                "observed_refs_or_types": ["expected:boundary access/guard and operation body tables"],
            },
        )

    nodes: dict[str, str] = {}
    value_producers: dict[str, str] = {}

    def visit(rows: object) -> None:
        if not isinstance(rows, list):
            return
        for operation in rows:
            if not isinstance(operation, Mapping):
                continue
            node_id = operation.get("id")
            op = operation.get("op")
            if isinstance(node_id, str) and isinstance(op, str):
                nodes[node_id] = op
                result = operation.get("result")
                results = (result,) if isinstance(result, str) else result
                if isinstance(results, (list, tuple)):
                    for value in results:
                        if isinstance(value, str):
                            value_producers[value] = op
            for child_field in ("body",):
                visit(operation.get(child_field))
            for arm in ("positive", "negative"):
                arm_value = operation.get(arm)
                if isinstance(arm_value, Mapping):
                    visit(arm_value.get("body"))

    visit(body)
    binding_by_access = {
        row.get("access_ref"): row
        for row in shared.bindings
        if isinstance(row, Mapping) and isinstance(row.get("access_ref"), str)
    }
    access_by_id = {
        row.get("access_id"): row
        for row in accesses
        if isinstance(row, Mapping) and isinstance(row.get("access_id"), str)
    }
    diagnostics: list[Mapping[str, Any]] = []
    for index, access in enumerate(accesses):
        if not isinstance(access, Mapping):
            continue
        access_id = access.get("access_id")
        if not isinstance(access_id, str):
            continue
        binding = binding_by_access.get(access_id)
        authorizations = (
            binding.get("semantic_authorizations") if isinstance(binding, Mapping)
            else None
        )
        owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
        access_view = authorizations.get("access_view") if isinstance(authorizations, Mapping) else None
        node_ref = access.get("node_ref")
        if node_ref is None:
            if access.get("kind") == "load" and (
                not isinstance(access_view, Mapping)
                or access_view.get("status") != "pass"
                or access.get("semantic_access_authorization") != access_view.get("token")
            ):
                diagnostics.append(
                    {
                        "rule_code": "semantic_access_authorization_missing",
                        "path": f"j.boundary.accesses[{index}].semantic_access_authorization",
                        "observed_refs_or_types": ["expected:ledger-issued exact Source access receipt"],
                    }
                )
            continue
        if (
            not isinstance(owner, Mapping)
            or owner.get("status") != "pass"
            or access.get("semantic_owner_authorization") != owner.get("token")
        ):
            diagnostics.append(
                {
                    "rule_code": "semantic_owner_authorization_missing",
                    "path": f"j.boundary.accesses[{index}].semantic_owner_authorization",
                    "observed_refs_or_types": ["expected:ledger-issued owner token"],
                }
            )
            continue
        if nodes.get(node_ref) != owner.get("operation_family"):
            diagnostics.append(
                {
                    "rule_code": "semantic_owner_operation_family_mismatch",
                    "path": f"j.boundary.accesses[{index}].node_ref",
                    "observed_refs_or_types": ["expected:ledger-authorized operation family"],
                }
            )
    for relation_index, relation in enumerate(guard_relations):
        if not isinstance(relation, Mapping):
            continue
        access_ref = relation.get("access_ref")
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        binding = binding_by_access.get(access_ref) if isinstance(access_ref, str) else None
        authorizations = (
            binding.get("semantic_authorizations") if isinstance(binding, Mapping)
            else None
        )
        guard = (
            authorizations.get("computed_guard")
            if isinstance(authorizations, Mapping) else None
        )
        if (
            not isinstance(access, Mapping)
            or not isinstance(guard, Mapping)
            or guard.get("status") != "pass"
            or access.get("semantic_guard_authorization") != guard.get("token")
        ):
            diagnostics.append(
                {
                    "rule_code": "semantic_guard_authorization_missing",
                    "path": f"j.boundary.access_guard_relations[{relation_index}]",
                    "observed_refs_or_types": ["expected:ledger-issued computed guard token"],
                }
            )
            continue
        value_ref = relation.get("value_ref")
        if (
            not isinstance(value_ref, str)
            or value_producers.get(value_ref) != guard.get("operation_family")
        ):
            diagnostics.append(
                {
                    "rule_code": "semantic_guard_operation_family_mismatch",
                    "path": f"j.boundary.access_guard_relations[{relation_index}].value_ref",
                    "observed_refs_or_types": [
                        "expected:ledger-authorized guard operation family"
                    ],
                }
            )
    return tuple(diagnostics)


def validate_source_access_fills_from_source_ledger(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Validate provider-authored fills without modifying the candidate."""

    original = copy.deepcopy(dict(packet))
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-access-fill-repair.v1",
        "status": "unavailable",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }
    j = original.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    if not isinstance(raw_accesses, list):
        return result

    diagnostics = _source_access_fill_validation_diagnostics(
        original,
        observed,
        ledger=ledger,
        source_facts=source_facts,
    )
    if diagnostics:
        result["status"] = "unproved"
        result["diagnostics"] = list(diagnostics)
        return result
    result["status"] = "pass"
    return result


def _source_access_fill_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject missing, extra, or non-provable fills per unique access occurrence."""

    diagnostics = _source_access_fill_validation_diagnostics(
        packet,
        observed,
        ledger=ledger,
        source_facts=source_facts,
    )
    if diagnostics:
        return diagnostics
    return ()


def _closed_observer_exception(exc: Exception) -> str:
    """Return bounded observer diagnostics without exporting host-local paths."""

    detail = str(exc).strip()
    detail = re.sub(r"https?://[^\s,;]+", "<url>", detail)
    detail = re.sub(r"(?:[A-Za-z]:)?/[^\s,;]+", "<path>", detail)
    detail = detail[:512]
    return (
        f"{type(exc).__name__}: {detail}"
        if detail
        else type(exc).__name__
    )


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
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject a stride binding whose axis contradicts Source address evidence.

    ``axis_ref`` is the logical tensor axis whose coordinate is multiplied by
    a representation-only stride carrier.  The existing schema checks only
    that the axis is in range, which is insufficient: a row-stride carrier can
    be validly attached to either axis of a rank-2 tensor while only one choice
    matches the compiled Source address polynomial.  This diagnostic is
    intentionally source-aware and necessary-only.  It uses the closed
    SourceTransferLedger occurrence/access binding plus the candidate's
    already-authored program binding; it never constructs or repairs an
    artifact.

    The closed check currently applies when a carrier is visibly multiplied by
    one direct physical ``program_id`` axis.  More complex address forms are
    left to the formal Gate1/Gate2 path rather than guessed here.
    """

    pi = packet.get("pi")
    j = packet.get("j")
    if not isinstance(pi, Mapping) or not isinstance(j, Mapping):
        return ()
    raw_bindings = pi.get("access_representation_bindings")
    if not isinstance(raw_bindings, list):
        return ()
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass" or not shared_ledger.bindings:
        return ()

    try:
        program_projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError:
        return ()

    # Map a physical launch axis to the explicitly authored logical J axes.
    # The shared projection preserves the distinction between direct
    # program_axes rows (no tensor-axis selector) and relation rows (explicit
    # axis_ref), so this diagnostic never falls back to p* order or layout
    # convention.
    program_axis_to_logical_axes: dict[int, set[int]] = {}
    program_axis_to_domains: dict[int, set[str]] = {}
    for row in program_projection.rows:
        for domain_ref, axis_ref in row.logical_axes:
            if axis_ref is None:
                program_axis_to_domains.setdefault(
                    row.physical_axis, set()
                ).add(domain_ref)
                continue
            if isinstance(axis_ref, int):
                logical_axis = axis_ref
            else:
                match = re.fullmatch(r"x([0-9]+)", axis_ref)
                logical_axis = int(match.group(1)) if match is not None else None
            if logical_axis is not None:
                program_axis_to_logical_axes.setdefault(
                    row.physical_axis, set()
                ).add(logical_axis)
    if not program_axis_to_logical_axes and not program_axis_to_domains:
        return ()

    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return ()
    runtime_parameters, _physical_parameters, runtime_position_by_name = (
        parameter_projection
    )
    tensor_to_parameter_positions: dict[str, set[int]] = {}
    for runtime_position, parameter in runtime_parameters.items():
        logical_ref = parameter.get("logical_ref")
        if parameter.get("kind") == "tensor" and isinstance(logical_ref, str):
            tensor_to_parameter_positions.setdefault(logical_ref, set()).add(
                runtime_position
            )

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

    occurrences_by_ordinal = {
        occurrence.ordinal: occurrence
        for occurrence in shared_ledger.occurrences
    }
    occurrence_by_access: dict[str, TransferOccurrence] = {}
    for row in shared_ledger.bindings:
        ordinal = row.get("ordinal")
        access_ref = row.get("access_ref")
        occurrence = occurrences_by_ordinal.get(ordinal)
        if (
            not isinstance(access_ref, str)
            or occurrence is None
            or access_ref in occurrence_by_access
        ):
            return ()
        occurrence_by_access[access_ref] = occurrence
    diagnostics: list[Mapping[str, Any]] = []
    for binding_index, binding in enumerate(raw_bindings):
        if not isinstance(binding, Mapping):
            continue
        carrier = binding.get("carrier_ref")
        access_ref = binding.get("access_ref")
        authored_axis = _axis_ref_number(binding.get("axis_ref"))
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
        occurrence = occurrence_by_access.get(access_ref)
        if occurrence is None:
            continue
        tensor = access.get("tensor")
        tensor_positions = tensor_to_parameter_positions.get(tensor, set())
        if len(tensor_positions) != 1:
            continue
        base_position = next(iter(tensor_positions))
        carrier_runtime_position = runtime_position_by_name.get(carrier)
        if carrier_runtime_position is None:
            continue
        access_kind = access.get("kind")
        if occurrence.kind != access_kind:
            continue
        if occurrence.logical_ref is not None and occurrence.logical_ref != tensor:
            continue
        if occurrence.pointer_position != base_position:
            continue
        expected_axes: set[int] = set()
        polynomial = occurrence.address_terms
        positions = argument_positions(polynomial)
        if carrier_runtime_position not in positions:
            continue
        for physical_axis in program_axes(polynomial):
            expected_axes.update(
                program_axis_to_logical_axes.get(physical_axis, ())
            )
            for domain_ref in program_axis_to_domains.get(physical_axis, ()):
                coordinate = _source_domain_coordinates(boundary).get(domain_ref)
                axis = (
                    _index_axis_for_coordinate(
                        access.get("index_expression"), coordinate
                    )
                    if coordinate is not None
                    else None
                )
                if axis is not None:
                    expected_axes.add(axis)
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
    """Check canonical address-carrier ownership against exact Source use."""

    pi = packet.get("pi")
    j = packet.get("j")
    if (
        not isinstance(j, Mapping)
        or not isinstance(pi, Mapping)
        or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA
    ):
        return ()
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    if not isinstance(parameters, list):
        return ()

    access_carriers = {
        item.get("carrier_ref")
        for item in pi.get("access_representation_bindings", [])
        if isinstance(item, Mapping) and isinstance(item.get("carrier_ref"), str)
    }
    source_syntax_uses = {
        item for item in source_runtime_parameter_uses if isinstance(item, str)
    } if isinstance(source_runtime_parameter_uses, list) else set()
    diagnostics: list[Mapping[str, Any]] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        if (
            not isinstance(name, str)
            or parameter.get("kind") not in {"pointer", "scalar"}
            or parameter.get("logical_ref") is not None
        ):
            continue
        token = re.compile(rf"%{re.escape(name)}(?![A-Za-z0-9_.$])")
        source_used_in_ttir = len(token.findall(ttir)) > 1
        source_used_in_syntax = name in source_syntax_uses
        source_used = source_used_in_ttir or source_used_in_syntax
        observed_use = (
            "source-ttir-runtime-use"
            if source_used_in_ttir
            else "source-syntax-runtime-use"
        )
        owned = name in access_carriers
        if source_used and not owned:
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
        elif owned and not source_used:
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
    return tuple(diagnostics[:64])


__all__ = [
    "SOURCE_TTIR_RECONCILE_PROTOCOL",
    "SOURCE_TTIR_OBSERVATION_PROTOCOL",
    "bind_source_pi",
    "materialize_source_program_domains",
    "observe_source_ttir",
    "observe_anonymous_source_ttir",
    "validate_source_semantic_authorizations",
    "validate_source_access_fills_from_source_ledger",
    "reconcile_candidate_source_ttir",
    "reorder_boundary_effects_from_source_ledger",
]
