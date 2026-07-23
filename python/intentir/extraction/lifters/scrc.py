"""Source-aware LLM Lift producing anonymous ``J + pure C + pi`` directly."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig
from intentir.ir.intent_op_contract import render_intent_op_wire_grammar
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ParameterKind,
    ScheduleSlotKind,
    render_boundary_access_wire_grammar,
    render_gate1_artifact_envelope_grammar,
)
from intentir.verify.gate1.pi import (
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)

from .prompt import SCHEMA as INTENTIR_SCHEMA
from .scrc_schema import (
    render_semantic_archive_wire_grammar,
    scrc_lift_core_text_format,
)


SCRC_RELIFT_PROTOCOL = "llm_direct_pi_structured_source_aware_relift_v2"
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DTYPE_ALIASES = {
    "float16": "f16",
    "half": "f16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
}


def _canonical_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPE_ALIASES.get(spelling, spelling)


@dataclass(frozen=True)
class SCRCHarnessBinding:
    """Privileged source-name sidecar; never enters J/C/pi or Replay."""

    source_to_opaque: tuple[tuple[str, str], ...]
    source_function: str
    replay_function: str = "scrc_k0"

    def opaque_name(self, source_name: str) -> str:
        return dict(self.source_to_opaque)[source_name]


@dataclass(frozen=True)
class SCRCProvenance:
    model: str
    prompt_sha256: str
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class SCRCExtraction:
    artifact: Gate1Artifact
    provenance: SCRCProvenance
    harness_binding: SCRCHarnessBinding
    source: str = field(repr=False)


@dataclass(frozen=True)
class SCRCLiftRequest:
    """Exact source-anonymized prompt material before the single provider call."""

    instructions: str = field(repr=False)
    user_input: str = field(repr=False)
    prompt_sha256: str
    harness_binding: SCRCHarnessBinding
    anonymous_compiler_view: Mapping[str, Any] | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True)
class SCRCReliftContext:
    """Structured TypeCheck/Gate1 evidence visible only to source-aware Lift."""

    relift_round: int
    failure_kind: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    previous_artifact: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.relift_round < 1:
            raise ValueError("relift_round must be positive")
        if self.failure_kind not in {
            "lift",
            "typecheck",
            "gate1",
        }:
            raise ValueError("unsupported relift failure kind")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("relift diagnostics must be a mapping")
        diagnostics = dict(self.diagnostics)
        # The payload is sent only to Lift, but it must still be deterministic
        # JSON data rather than an executable callback or arbitrary object.
        json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.previous_artifact is not None:
            if not isinstance(self.previous_artifact, Mapping):
                raise TypeError("previous relift artifact must be a mapping")
            previous = dict(self.previous_artifact)
            json.dumps(previous, sort_keys=True, allow_nan=False)
            object.__setattr__(self, "previous_artifact", previous)

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "failure_kind": self.failure_kind,
            "diagnostics": dict(self.diagnostics),
            "previous_artifact": (
                dict(self.previous_artifact)
                if self.previous_artifact is not None else None
            ),
        }


@dataclass(frozen=True)
class SCRCReliftRecord:
    relift_round: int
    extraction: SCRCExtraction | None = None
    diagnostic: str | None = None
    rejected_attempts: tuple["SCRCLiftAttempt", ...] = ()


@dataclass(frozen=True)
class SCRCCompilerArgument:
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None
    mode: str | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("compiler argument name must be an identifier")
        if self.kind not in {"tensor", "scalar", "none", "opaque"}:
            raise ValueError(
                "compiler argument kind must be tensor/scalar/none/opaque"
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("compiler argument dtype must be non-empty")
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("compiler argument dimensions must be non-negative")
        if self.kind == "tensor" and self.value is not None:
            raise ValueError("tensor compiler argument forbids scalar value")
        if self.kind == "scalar" and (shape or self.value is None):
            raise ValueError("scalar compiler argument requires value only")
        if self.kind == "none" and (shape or self.value is not None):
            raise ValueError("none compiler argument forbids shape/value")
        if self.kind == "opaque" and (shape or self.value is not None):
            raise ValueError("opaque compiler argument exposes only its public type")
        if self.mode is not None:
            allowed_modes = {
                "tensor": {"input", "output", "inout"},
                "scalar": {"value"},
                "none": {"value"},
                "opaque": {"value"},
            }[self.kind]
            if self.mode not in allowed_modes:
                raise ValueError(
                    f"compiler argument mode for {self.kind} must be one of "
                    + ", ".join(sorted(allowed_modes))
                )
        object.__setattr__(self, "shape", shape)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
        }
        if self.kind == "tensor":
            result["shape"] = list(self.shape)
        elif self.kind == "scalar":
            result["value"] = self.value
        if self.mode is not None:
            result["mode"] = self.mode
        return result


@dataclass(frozen=True)
class SCRCCompilerView:
    """Trusted invocation facts supplied to Lift, never a semantic ABI."""

    schedule_values: Mapping[str, bool | int | float | str]
    bound_constexpr_values: Mapping[str, bool | int | float | str] = field(
        default_factory=dict
    )
    compiler_slots: tuple[str, ...] = ()
    aliases: tuple[AliasSpec, ...] = ()
    arguments: tuple[SCRCCompilerArgument, ...] = ()
    grid: tuple[int, ...] = ()
    grid_contract: tuple[str, ...] = ()
    parameter_order: tuple[str, ...] = ()
    constexpr_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.schedule_values)
        bound_constexpr_values = dict(self.bound_constexpr_values)
        for collection, label in (
            (values, "schedule"),
            (bound_constexpr_values, "bound constexpr"),
        ):
            for name, value in collection.items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"{label} names must be identifiers")
                if not isinstance(value, (bool, int, float, str)):
                    raise TypeError(f"{label} values must be JSON scalars")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{label} float values must be finite")
        if set(values) & set(bound_constexpr_values):
            raise ValueError(
                "bound constexpr and schedule parameters must be disjoint"
            )
        slots = tuple(self.compiler_slots)
        if len(slots) != len(set(slots)) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in slots
        ):
            raise ValueError("compiler slots must contain unique identifiers")
        if set(slots) - set(values):
            raise ValueError("compiler slots require schedule values")
        object.__setattr__(self, "schedule_values", values)
        object.__setattr__(
            self, "bound_constexpr_values", bound_constexpr_values
        )
        object.__setattr__(self, "compiler_slots", slots)
        object.__setattr__(
            self,
            "aliases",
            tuple(
                item if isinstance(item, AliasSpec) else AliasSpec.from_json(item)
                for item in self.aliases
            ),
        )
        object.__setattr__(
            self,
            "arguments",
            tuple(
                item if isinstance(item, SCRCCompilerArgument)
                else SCRCCompilerArgument(**item)
                for item in self.arguments
            ),
        )
        if len({item.name for item in self.arguments}) != len(self.arguments):
            raise ValueError("compiler arguments must have unique names")
        grid = tuple(int(dim) for dim in self.grid)
        if grid and (len(grid) > 3 or any(dim <= 0 for dim in grid)):
            raise ValueError("compiler grid must have one to three positive dimensions")
        grid_contract = tuple(self.grid_contract)
        if any(not isinstance(item, str) or not item for item in grid_contract):
            raise ValueError("grid contract must contain non-empty expressions")
        if grid and grid_contract and len(grid) != len(grid_contract):
            raise ValueError("concrete grid and grid contract ranks differ")
        parameter_order = tuple(self.parameter_order)
        constexpr_parameters = tuple(self.constexpr_parameters)
        for names, label in (
            (parameter_order, "parameter order"),
            (constexpr_parameters, "constexpr parameters"),
        ):
            if len(names) != len(set(names)) or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in names
            ):
                raise ValueError(f"{label} must contain unique identifiers")
        runtime_names = {item.name for item in self.arguments}
        constexpr_names = set(constexpr_parameters)
        if runtime_names & constexpr_names:
            raise ValueError("runtime and constexpr parameters overlap")
        if parameter_order and set(parameter_order) != runtime_names | constexpr_names:
            raise ValueError("parameter order must cover runtime and constexpr inputs")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_contract", grid_contract)
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(
            self, "constexpr_parameters", constexpr_parameters
        )
        bound_names = set(bound_constexpr_values)
        if bound_names - constexpr_names:
            raise ValueError(
                "bound constexpr values must bind constexpr parameters"
            )
        schedule_constexprs = constexpr_names - bound_names
        if schedule_constexprs - set(values):
            raise ValueError("schedule constexpr parameters require C values")

    def to_json(self) -> dict[str, Any]:
        return {
            "schedule_values": dict(self.schedule_values),
            "bound_constexpr_values": dict(self.bound_constexpr_values),
            "compiler_slots": list(self.compiler_slots),
            "aliases": [item.to_json() for item in self.aliases],
            "arguments": [item.to_json() for item in self.arguments],
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
        }


@dataclass(frozen=True)
class SCRCLiftAttempt:
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0


class SCRCLiftError(RuntimeError):
    def __init__(self, attempt: SCRCLiftAttempt) -> None:
        self.attempts = (attempt,)
        super().__init__(f"SCRC Lift response rejected: {attempt.diagnostic}")


def _annotation_is_constexpr(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return ast.unparse(annotation).endswith("constexpr")


def anonymize_source(
    source: str,
    *,
    source_symbol: str | None = None,
    bound_constexpr_parameters: tuple[str, ...] = (),
) -> tuple[str, SCRCHarnessBinding, dict[str, str]]:
    """Alpha-rename source identifiers before the Lift provider sees them.

    ``source_symbol`` is a privileged source-selection coordinate.  It chooses
    the callable whose positional interface becomes ``scrc_k0`` but is never
    placed in J, C, pi, or the provider prompt.  This is required for real
    library modules that contain helpers and several kernels; declaration order
    in a Python file is not a semantic kernel identity.
    """

    tree = ast.parse(source)
    functions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    if not functions:
        raise ValueError("Lift source contains no function")
    if source_symbol is not None:
        if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
            raise ValueError("Lift source_symbol must be a Python identifier")
        selected = [
            function for function in functions if function.name == source_symbol
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Lift source_symbol {source_symbol!r} must resolve exactly once; "
                f"found {len(selected)} definitions"
            )
        primary = selected[0]
        ordered_functions = [
            primary, *(item for item in functions if item is not primary)
        ]
    else:
        primary = functions[0]
        ordered_functions = functions
    primary_source_name = primary.name
    primary_parameter_names = {
        argument.arg
        for argument in [
            *primary.args.posonlyargs,
            *primary.args.args,
            *primary.args.kwonlyargs,
        ]
    }
    mapping: dict[str, str] = {}
    for index, function in enumerate(ordered_functions):
        mapping.setdefault(function.name, f"scrc_k{index}")

    bound_constexprs = set(bound_constexpr_parameters)
    if len(bound_constexprs) != len(bound_constexpr_parameters) or any(
        not isinstance(name, str) or not name.isidentifier()
        for name in bound_constexpr_parameters
    ):
        raise ValueError(
            "bound constexpr parameters must be unique identifiers"
        )
    annotated_constexprs = {
        argument.arg
        for argument in primary.args.args
        if _annotation_is_constexpr(argument.annotation)
    }
    unknown_bound_constexprs = bound_constexprs - annotated_constexprs
    if unknown_bound_constexprs:
        raise ValueError(
            "bound constexpr parameters are absent from the selected ABI: "
            + ", ".join(sorted(unknown_bound_constexprs))
        )

    runtime_index = 0
    constexpr_index = 0
    for argument in primary.args.args:
        if _annotation_is_constexpr(argument.annotation):
            if argument.arg in bound_constexprs:
                opaque = f"p{runtime_index}"
                runtime_index += 1
            else:
                opaque = f"c{constexpr_index}"
                constexpr_index += 1
        else:
            opaque = f"p{runtime_index}"
            runtime_index += 1
        mapping.setdefault(argument.arg, opaque)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
    reserved = (
        set(keyword.kwlist)
        | set(dir(builtins))
        | imported
        | {"triton", "tl", "libdevice", "torch", "math"}
    )
    local_nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id not in reserved
            and node.id not in mapping
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in local_nodes:
        if node.id not in mapping:
            mapping[node.id] = f"v{sum(name.startswith('v') for name in mapping.values())}"

    class _AlphaRenamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = mapping.get(node.arg, node.arg)
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_keyword(self, node: ast.keyword) -> ast.AST:
            if node.arg in primary_parameter_names:
                node.arg = mapping.get(node.arg, node.arg)
            return self.generic_visit(node)

        def visit_Global(self, node: ast.Global) -> ast.AST:
            node.names = [mapping.get(name, name) for name in node.names]
            return node

        def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
            node.names = [mapping.get(name, name) for name in node.names]
            return node

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
            if node.name is not None:
                node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                node.value = _replace_names(node.value, mapping)
            return node

    anonymous_tree = _AlphaRenamer().visit(tree)
    ast.fix_missing_locations(anonymous_tree)
    anonymous = ast.unparse(anonymous_tree) + "\n"
    binding = SCRCHarnessBinding(
        source_to_opaque=tuple(mapping.items()),
        source_function=primary_source_name,
    )
    return anonymous, binding, mapping


def _replace_names(expression: str, mapping: Mapping[str, str]) -> str:
    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: mapping.get(match.group(0), match.group(0)),
        expression,
    )


def _anonymous_compiler_view(
    view: SCRCCompilerView | None, mapping: dict[str, str]
) -> dict[str, Any] | None:
    if view is None:
        return None
    next_c = 0
    used = set(mapping.values())
    for name in [*view.schedule_values, *view.compiler_slots]:
        if name in mapping and mapping[name].startswith("c"):
            continue
        while f"c{next_c}" in used:
            next_c += 1
        mapping[name] = f"c{next_c}"
        used.add(mapping[name])
    return {
        "schedule_values": {
            mapping[name]: value for name, value in view.schedule_values.items()
        },
        "bound_constexpr_values": {
            mapping[name]: value
            for name, value in view.bound_constexpr_values.items()
        },
        "compiler_slots": [mapping[name] for name in view.compiler_slots],
        "compiler_roles": {
            mapping[name]: name for name in view.compiler_slots
        },
        "aliases": [
            {
                "left": mapping[item.left],
                "right": mapping[item.right],
                "kind": item.kind.value,
            }
            for item in view.aliases
        ],
        "arguments": [
            {**item.to_json(), "name": mapping[item.name]}
            for item in view.arguments
        ],
        "grid": list(view.grid),
        "grid_contract": [
            _replace_names(item, mapping) for item in view.grid_contract
        ],
        "parameter_order": [mapping[name] for name in view.parameter_order],
        "constexpr_parameters": [
            mapping[name] for name in view.constexpr_parameters
        ],
    }


def build_scrc_lift_instructions() -> str:
    return f"""You are the source-aware Lift LLM for IntentIR.
Return one complete JSON object matching the supplied schema. You directly
produce anonymous J=A+B, independent pure C, and the realization witness pi.
There is no downstream composer and no rule catalog.

Artifact envelope contract:
{render_gate1_artifact_envelope_grammar()}

Naming is mandatory: kernel scrc_k0; tensors t0,t1,...; J nodes n0,n1,...;
values v0,v1,...; dimensions x0,x1,...; domains d0,...; accesses a0,...;
effects e0,...; scopes q0,...; storage b0,...; parameters p0,...; C slots
c0,...; pi regions r0,... and instances i0,..., all by declaration order.

J tensor handles form their own gap-free sequence over intent.inputs followed
by intent.outputs. Never derive a t* suffix from a p* parameter position. For
example, if p0 is an input pointer, p1 is an output pointer, and p2 is a
runtime extent scalar, the J tensors are input t0 and output t1; p0.logical_ref
is t0, p1.logical_ref is t1, and p2.logical_ref is the symbolic dimension x0.

The semantic IntentFunction name is exactly scrc_j0, never scrc_k0. The
node_ids object maps n0,n1,... to structural paths body[0], body[1], or nested
body[i].body[j], never to v* results. boundary.domains maps each d* key directly
to one portable boolean expression string; it is never an axes/members object.

J boundary access contract:
{render_boundary_access_wire_grammar()}

The trusted harness_compiler_view is byte-authoritative plumbing. Move every
logical p* handle into J.signature and every implementation-selection c*
handle into the single C object. Their relative source positions stay private
to Harness and are not serialized in the artifact. C values and compiler_slots
must be identical to the supplied anonymous values. For each compiler slot,
use exactly the public role in compiler_roles. Do not rename c* slots to p* and
do not turn compiler roles into inferred strategy names.

bound_constexpr_values are non-C invocation bindings. Their p* parameters
remain semantic constexpr handles in J.signature and never become C slots or
values. Lift must classify their source use: any consequence that changes
logical extent/value/predicate/result belongs only to J; representation-only
layout/address realization may be referenced only by pi. A source constexpr
not listed there is pure scheduling, remains a c* parameter, and must have a
matching pure-C constexpr slot.

A runtime scalar p* value in harness_compiler_view is only the current Harness
binding, never permission to specialize J. If source uses that scalar in a
logical extent, index, predicate, or result, create a symbolic x* dimension,
use it in every affected J tensor shape, and set the scalar parameter's
logical_ref to that x*. Symbolic dimensions are type/signature handles, not SSA
values: never list x* in portable scope definitions/captures or storage.

For every observable output, J.boundary.result_relations declares only the
closed semantic comparison class; it never declares numeric thresholds. Use
`exact` only when cross-backend equality is part of the logical contract (for
example integer/boolean results or exactly specified copies). Floating
transcendentals, floating reductions, and contracts that allow reassociation
must use `tolerance`; the trusted external Gate2 policy Q supplies all atol,
rtol, ULP, and NaN rules. Never place a Python comparator or a
candidate-selected tolerance in J, C, or pi.

C contains scheduling slots/values only. It must not contain operators,
predicates, source topology, APIs, policy, or a strategy field.

Pi is not semantics and is not executable. It may record only anonymous
regions, instances referring to J/C handles, routes with operand slots, partial
order, sharing, placement, and the closed source-neutral mechanism tags in the
schema. Never put an operator, combiner, predicate, value, dtype, shape,
frontend API, source identifier, source fragment, prose, or rule identity in pi.

Pi graph reference contract:
{render_realization_witness_wire_grammar()}

Pi mechanism contract:
{render_realization_mechanism_grammar()}

Pi preserves every compiler-visible source realization event not denoted by J
or pure C. Emit multiple instances with the same semantic_ref when one declared
handle has multiple observable realization stages. Every anonymous J.signature
or C handle materially used by address, transfer, synchronization, placement,
or sharing realization must anchor an instance; dependencies then route only
between the declared i* instance IDs.
Every source program-coordinate occurrence must remain a distinct
domain-anchored realization path entering an address_base instance, even when
the current concrete grid extent on that axis is one. One logical domain may
therefore have both a program-coordinate path into address_base and a separate
lane-coordinate path into address_lane; never collapse the former into a loop
or erase it as constant-degenerate topology.
Preserve source event multiplicity, def-use routes, partial order, regions, and
one-producer sharing exactly; do not use duplicate instances as an undocumented
code. Logical operators, predicates, fills, constants, index formulas, shapes,
and effects remain exclusively in J.

The source is already alpha-renamed. Do not invent or recover original names.
Output JSON only.

IntentOp structural contract:
{render_intent_op_wire_grammar()}

Semantic archive cross-reference contract:
{render_semantic_archive_wire_grammar()}

Layer-A operation grammar:
{INTENTIR_SCHEMA}
"""


def _build_user_input(
    anonymous_source: str,
    compiler_view: dict[str, Any] | None,
    relift_context: SCRCReliftContext | None,
) -> str:
    payload: dict[str, Any] = {"anonymous_source": anonymous_source}
    if compiler_view is not None:
        payload["harness_compiler_view"] = compiler_view
    if relift_context is not None:
        payload["relift"] = relift_context.to_json()
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _prompt_material(
    source: str,
    compiler_view: SCRCCompilerView | None,
    relift_context: SCRCReliftContext | None,
    source_symbol: str | None = None,
) -> tuple[str, str, SCRCHarnessBinding, dict[str, Any] | None]:
    anonymous_source, binding, mapping = anonymize_source(
        source,
        source_symbol=source_symbol,
        bound_constexpr_parameters=(
            tuple(compiler_view.bound_constexpr_values)
            if compiler_view is not None
            else ()
        ),
    )
    anonymous_view = _anonymous_compiler_view(compiler_view, mapping)
    instructions = build_scrc_lift_instructions()
    user = _build_user_input(anonymous_source, anonymous_view, relift_context)
    return instructions, user, binding, anonymous_view


def prepare_scrc_lift(
    source: str,
    *,
    compiler_view: SCRCCompilerView | None = None,
    relift_context: SCRCReliftContext | None = None,
    source_symbol: str | None = None,
) -> SCRCLiftRequest:
    instructions, user, binding, anonymous_view = _prompt_material(
        source, compiler_view, relift_context, source_symbol
    )
    return SCRCLiftRequest(
        instructions,
        user,
        hashlib.sha256((instructions + "\0" + user).encode()).hexdigest(),
        binding,
        anonymous_view,
    )


def scrc_lift_prompt_sha256(
    source: str,
    name_hint: str | None = None,
    *,
    compiler_view: SCRCCompilerView | None = None,
    source_symbol: str | None = None,
) -> str:
    del name_hint
    instructions, user, _binding, _view = _prompt_material(
        source, compiler_view, None, source_symbol
    )
    return hashlib.sha256((instructions + "\0" + user).encode()).hexdigest()


def scrc_relift_prompt_sha256(
    source: str,
    context: SCRCReliftContext,
    *,
    compiler_view: SCRCCompilerView | None = None,
    source_symbol: str | None = None,
) -> str:
    instructions, user, _binding, _view = _prompt_material(
        source, compiler_view, context, source_symbol
    )
    return hashlib.sha256((instructions + "\0" + user).encode()).hexdigest()


def _json_object(text: str) -> dict[str, Any]:
    match = _FENCE.search(text)
    if match:
        text = match.group(1)

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Lift response contains non-JSON number {token}")

    parsed = json.loads(text, parse_constant=reject_nonfinite)
    if not isinstance(parsed, dict):
        raise TypeError("Lift response must be a JSON object")
    return parsed


def _response_usage(raw: Mapping[str, Any]) -> dict[str, int]:
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    return {
        key: value
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance((value := usage.get(key)), int)
        and not isinstance(value, bool)
        and value >= 0
    }


def _validate_compiler_view(
    artifact: Gate1Artifact, anonymous_view: dict[str, Any] | None
) -> None:
    if anonymous_view is None:
        return
    if anonymous_view["parameter_order"]:
        expected_p = [
            name for name in anonymous_view["parameter_order"]
            if name.startswith("p")
        ]
        expected_c = [
            name for name in anonymous_view["parameter_order"]
            if name.startswith("c")
        ]
        actual_p = [item.name for item in artifact.j.signature.parameters]
        actual_c = [
            item.name for item in artifact.c.slots
            if item.kind is ScheduleSlotKind.CONSTEXPR
        ]
        if actual_p != expected_p or actual_c != expected_c:
            raise ValueError("Lift changed Harness-declared logical/C handles")
    if artifact.c.values != anonymous_view["schedule_values"]:
        raise ValueError("Lift changed trusted pure-C values")
    actual_compiler_slots = [
        item.name for item in artifact.c.slots
        if item.kind is ScheduleSlotKind.COMPILER
    ]
    if actual_compiler_slots != anonymous_view["compiler_slots"]:
        raise ValueError("Lift changed trusted compiler slots")
    slots = {item.name: item for item in artifact.c.slots}
    for name, role in anonymous_view["compiler_roles"].items():
        slot = slots.get(name)
        if (
            slot is None
            or slot.kind is not ScheduleSlotKind.COMPILER
            or slot.role != role
        ):
            raise ValueError("Lift changed a public compiler-option C role")
    if [item.to_json() for item in artifact.j.boundary.aliases] != anonymous_view["aliases"]:
        raise ValueError("Lift changed trusted alias relation")
    expected_constexprs = set(anonymous_view["constexpr_parameters"])
    actual_constexprs = {
        item.name for item in artifact.j.signature.parameters
        if item.kind is ParameterKind.CONSTEXPR
    } | {
        item.name for item in artifact.c.slots
        if item.kind is ScheduleSlotKind.CONSTEXPR
    }
    if actual_constexprs != expected_constexprs:
        raise ValueError("Lift changed constexpr parameter positions")
    bound_constexprs = set(anonymous_view["bound_constexpr_values"])
    slots = {item.name: item for item in artifact.c.slots}
    for name in expected_constexprs:
        slot = slots.get(name)
        if name in bound_constexprs:
            if slot is not None:
                raise ValueError("Lift moved a bound constexpr into pure C")
        elif slot is None or slot.kind is not ScheduleSlotKind.CONSTEXPR:
            raise ValueError("Lift omitted a pure-C constexpr slot")
    signature = {item.name: item for item in artifact.j.signature.parameters}
    for argument in anonymous_view["arguments"]:
        if argument["name"].startswith("c"):
            if argument["name"] not in slots:
                raise ValueError("Lift omitted a trusted C argument")
            continue
        parameter = signature.get(argument["name"])
        if parameter is None:
            raise ValueError("Lift omitted a trusted compiler argument")
        expected_kind = {
            "tensor": ParameterKind.POINTER,
            "scalar": ParameterKind.SCALAR,
            "opaque": ParameterKind.OPAQUE,
            "none": ParameterKind.SCALAR,
        }[argument["kind"]]
        allowed_kinds = {expected_kind}
        if expected_kind is ParameterKind.POINTER:
            allowed_kinds.add(ParameterKind.TENSOR)
        if argument["kind"] != "none" and parameter.kind not in allowed_kinds:
            raise ValueError("Lift changed a trusted compiler argument kind")
        if _canonical_dtype(parameter.dtype) != _canonical_dtype(argument["dtype"]):
            raise ValueError("Lift changed a trusted compiler argument dtype")
        if argument["kind"] == "tensor" and parameter.rank != len(argument["shape"]):
            raise ValueError("Lift changed a trusted compiler argument rank")
        expected_mode = argument.get("mode")
        if expected_mode is not None and parameter.mode.value != expected_mode:
            raise ValueError("Lift changed a trusted compiler argument mode")
    # Grid/source positions are Harness coordinates.  They are retained in the
    # extraction sidecar and deliberately absent from the proof artifact.


def _validate_anonymous_lift(
    artifact: Gate1Artifact, binding: SCRCHarnessBinding
) -> None:
    if artifact.j.kernel_id != "scrc_k0":
        raise ValueError("Lift kernel id must use scrc_k0")
    if artifact.j.intent.name != "scrc_j0":
        raise ValueError("Lift IntentFunction name must use scrc_j0")
    tensors = [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    actual_tensors = [item.name for item in tensors]
    expected_tensors = [f"t{index}" for index in range(len(tensors))]
    if actual_tensors != expected_tensors:
        raise ValueError(
            "J tensors must use their own declaration-order handles independent "
            f"of p* positions; expected {expected_tensors}, got {actual_tensors}"
        )
    if list(artifact.j.node_ids) != [
        f"n{index}" for index in range(len(artifact.j.node_ids))
    ]:
        raise ValueError("J node ids must use declaration-order handles n0, n1, ...")
    expected_parameters = [
        opaque
        for _source, opaque in binding.source_to_opaque
        if re.fullmatch(r"p[0-9]+", opaque)
    ]
    if [item.name for item in artifact.j.signature.parameters] != expected_parameters:
        raise ValueError("J.signature parameters differ from anonymous source handles")
    dimensions: list[str] = []
    for tensor in tensors:
        for dim in tensor.shape:
            if isinstance(dim, str) and dim not in dimensions:
                dimensions.append(dim)
    if dimensions != [f"x{index}" for index in range(len(dimensions))]:
        raise ValueError("symbolic dimensions must use x0, x1, ...")
    for prefix, values in (
        ("d", list(artifact.j.boundary.domains)),
        ("a", [item.access_id for item in artifact.j.boundary.accesses]),
        ("e", [item.effect_id for item in artifact.j.boundary.effects]),
        ("q", [item.scope_id for item in artifact.j.scopes]),
        ("b", [item.storage_id for item in artifact.j.storage]),
    ):
        if values != [f"{prefix}{index}" for index in range(len(values))]:
            raise ValueError(f"J {prefix} handles are not declaration ordered")
    if any(not re.fullmatch(r"c[0-9]+", item.name) for item in artifact.c.slots):
        raise ValueError("pure C slots must use c0, c1, ...")
    visible = {
        "j": artifact.j.to_json(),
        "c": artifact.c.to_json(),
        "pi": artifact.pi.to_json(),
    }
    visible_strings: list[str] = []
    literal_fields = {
        "dtype", "frontend", "kind", "mechanism", "mode", "op",
        "placement", "producer", "role", "value_type",
    }

    def collect_strings(value: Any, field_name: str | None = None) -> None:
        if isinstance(value, str):
            if field_name not in literal_fields:
                visible_strings.append(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect_strings(item, key)
        elif isinstance(value, list):
            for item in value:
                collect_strings(item, field_name)

    collect_strings(visible)
    for source_name, opaque in binding.source_to_opaque:
        if source_name == opaque:
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])"
        if any(re.search(pattern, item) for item in visible_strings):
            raise ValueError("Lift response retained a source identifier")


def extract_scrc_artifact(
    source: str,
    name_hint: str | None = None,
    *,
    provider: ProviderConfig | None = None,
    compiler_view: SCRCCompilerView | None = None,
    effort: str | None = None,
    relift_context: SCRCReliftContext | None = None,
    source_symbol: str | None = None,
) -> SCRCExtraction:
    """Perform exactly one source-aware provider call for one Lift attempt."""

    del name_hint
    request = prepare_scrc_lift(
        source,
        compiler_view=compiler_view,
        relift_context=relift_context,
        source_symbol=source_symbol,
    )
    instructions = request.instructions
    user = request.user_input
    binding = request.harness_binding
    anonymous_view = request.anonymous_compiler_view
    prompt_sha = request.prompt_sha256
    reply = call_responses(
        instructions,
        user,
        provider=provider,
        effort=effort,
        text_format=scrc_lift_core_text_format(),
    )
    try:
        payload = _json_object(reply.text)
        artifact = Gate1Artifact.from_json(payload)
        _validate_anonymous_lift(artifact, binding)
        _validate_compiler_view(artifact, anonymous_view)
    except Exception as exc:
        attempt = SCRCLiftAttempt(
            model=reply.model,
            diagnostic=f"{type(exc).__name__}: {exc}",
            response_text=reply.text,
            response_id=reply.raw.get("id"),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            usage=_response_usage(reply.raw),
            request_sha256=reply.request_sha256,
            response_sha256=(
                reply.response_sha256
                or hashlib.sha256(reply.text.encode()).hexdigest()
            ),
            raw_response_sha256=reply.raw_response_sha256,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            elapsed_ms=reply.elapsed_ms,
        )
        raise SCRCLiftError(attempt) from exc
    provenance = SCRCProvenance(
        model=reply.model,
        prompt_sha256=prompt_sha,
        response_id=reply.raw.get("id"),
        response_model=(
            reply.raw.get("model")
            if isinstance(reply.raw.get("model"), str) else None
        ),
        usage=_response_usage(reply.raw),
        request_sha256=reply.request_sha256,
        response_sha256=(
            reply.response_sha256
            or hashlib.sha256(reply.text.encode()).hexdigest()
        ),
        raw_response_sha256=reply.raw_response_sha256,
        response_text=reply.text,
        raw_response=reply.raw_response,
        attempts=reply.attempts,
        elapsed_ms=reply.elapsed_ms,
    )
    return SCRCExtraction(artifact, provenance, binding, source)


@dataclass
class ResponsesSCRCRelifter:
    provider: ProviderConfig | None = None
    compiler_view: SCRCCompilerView | None = None
    effort: str | None = None
    source_symbol: str | None = None
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def relift(
        self,
        previous: SCRCExtraction,
        *,
        failure_kind: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> SCRCExtraction:
        round_index = len(self._records) + 1
        context = SCRCReliftContext(
            round_index,
            failure_kind,
            diagnostics or {},
            previous.artifact.to_json(),
        )
        try:
            extraction = extract_scrc_artifact(
                previous.source,
                provider=self.provider,
                compiler_view=self.compiler_view,
                effort=self.effort,
                relift_context=context,
                source_symbol=self.source_symbol,
            )
        except SCRCLiftError as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    rejected_attempts=exc.attempts,
                )
            )
            raise
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        self._records.append(SCRCReliftRecord(round_index, extraction=extraction))
        return extraction


@dataclass
class ResponsesSCRCLiftAdapter:
    """Source-aware provider adapter for the unified SCRC lifecycle."""

    provider: ProviderConfig | None = None
    compiler_view: SCRCCompilerView | None = None
    effort: str | None = None
    source_symbol: str | None = None
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Responses SCRC Lift requires SourceProgram.text")
        round_index = len(self._records) + 1
        context = None
        if feedback is not None:
            kind = getattr(getattr(feedback, "kind", None), "value", None)
            diagnostics = getattr(feedback, "diagnostics", None)
            previous_artifact = getattr(feedback, "previous_artifact", None)
            context = SCRCReliftContext(
                max(1, len(self._records)),
                kind,
                diagnostics if isinstance(diagnostics, Mapping) else {},
                (
                    previous_artifact.to_json()
                    if isinstance(previous_artifact, Gate1Artifact)
                    else None
                ),
            )
        try:
            extraction = extract_scrc_artifact(
                source,
                provider=self.provider,
                compiler_view=self.compiler_view,
                effort=self.effort,
                relift_context=context,
                source_symbol=self.source_symbol,
            )
        except SCRCLiftError as exc:
            diagnostic = "; ".join(
                item.diagnostic for item in exc.attempts
            )
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    rejected_attempts=exc.attempts,
                )
            )
            # Local import avoids making extraction depend on the verify package
            # during module initialization.
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {"diagnostic": diagnostic, "failure_class": "lift_response"},
                physical_calls=max(item.attempts for item in exc.attempts),
            ) from exc
        self._records.append(
            SCRCReliftRecord(round_index, extraction=extraction)
        )
        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            extraction.artifact,
            physical_calls=extraction.provenance.attempts,
            metadata={
                "model": extraction.provenance.model,
                "response_model": extraction.provenance.response_model,
                "response_id": extraction.provenance.response_id,
                "usage": dict(extraction.provenance.usage),
                "elapsed_ms": extraction.provenance.elapsed_ms,
            },
        )


__all__ = [
    "ResponsesSCRCLiftAdapter", "ResponsesSCRCRelifter",
    "SCRC_RELIFT_PROTOCOL", "SCRCCompilerArgument",
    "SCRCCompilerView", "SCRCExtraction", "SCRCHarnessBinding",
    "SCRCLiftAttempt", "SCRCLiftError", "SCRCLiftRequest", "SCRCProvenance", "SCRCReliftContext",
    "SCRCReliftRecord", "anonymize_source", "build_scrc_lift_instructions",
    "extract_scrc_artifact", "prepare_scrc_lift", "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
]
