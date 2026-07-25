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
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ParameterKind,
    ScheduleSlotKind,
    SchemaError,
)
from intentir.verify.typecheck import TYPECHECK_PROTOCOL, type_check

from .scrc_schema import scrc_lift_core_text_format


SCRC_RELIFT_PROTOCOL = "llm_direct_pi_structured_source_aware_relift_v2"
SCRC_DECODE_PROTOCOL = "intentir.scrc.decode.v4"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
_CANDIDATE_DIAGNOSTIC_LIMIT = 20
_CANDIDATE_DIAGNOSTIC_TEXT_LIMIT = 2048
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
    backend: str = "responses"
    termination_status: str | None = None
    trajectory: Mapping[str, Any] | None = field(default=None, repr=False)


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
class SCRCCandidateCheckContext:
    """Source-free, versioned context visible to the local CandidateCheck."""

    expected_parameters: tuple[str, ...]
    anonymous_compiler_view: Mapping[str, Any] | None = field(
        default=None, repr=False
    )
    schema_version: str = Gate1Artifact.SCHEMA
    decode_protocol: str = SCRC_DECODE_PROTOCOL
    typecheck_protocol: str = TYPECHECK_PROTOCOL

    def __post_init__(self) -> None:
        if any(not re.fullmatch(r"p[0-9]+", item) for item in self.expected_parameters):
            raise ValueError("CandidateCheck parameters must be anonymous p* handles")
        if self.anonymous_compiler_view is not None:
            if not isinstance(self.anonymous_compiler_view, Mapping):
                raise TypeError("CandidateCheck compiler view must be a mapping")
            value = dict(self.anonymous_compiler_view)
            json.dumps(value, sort_keys=True, allow_nan=False)
            object.__setattr__(self, "anonymous_compiler_view", value)


def scrc_candidate_check_context(
    request: SCRCLiftRequest,
) -> SCRCCandidateCheckContext:
    """Project a Lift request to the exact source-free checker context."""

    return SCRCCandidateCheckContext(
        expected_parameters=tuple(
            opaque
            for _source, opaque in request.harness_binding.source_to_opaque
            if re.fullmatch(r"p[0-9]+", opaque)
        ),
        anonymous_compiler_view=request.anonymous_compiler_view,
        typecheck_protocol=TYPECHECK_PROTOCOL,
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
    backend_evidence: Mapping[str, Any] | None = field(default=None, repr=False)


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
    backend: str = "responses"
    termination_status: str | None = None
    trajectory: Mapping[str, Any] | None = field(default=None, repr=False)


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

    # Keyword names are identifiers only for the statically selected local
    # callee; they are not ordinary Name nodes.  Record each local function's
    # own formal mapping so calls can be renamed without touching tl.*, library,
    # attribute, dynamic, or unresolved call keywords.
    local_call_keyword_mappings: dict[str, dict[str, str]] = {}
    plain_local_functions = {
        function.name
        for function in functions
        if not function.decorator_list
    }
    for function in functions:
        parameters = (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        for argument in parameters:
            if argument.arg not in mapping:
                mapping[argument.arg] = (
                    f"v{sum(name.startswith('v') for name in mapping.values())}"
                )
        local_call_keyword_mappings[function.name] = {
            argument.arg: mapping[argument.arg] for argument in parameters
        }

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

        def visit_Call(self, node: ast.Call) -> ast.AST:
            plain_callee = (
                isinstance(node.func, ast.Name)
                and node.func.id in plain_local_functions
            )
            formal_mapping = (
                local_call_keyword_mappings.get(node.func.id)
                if isinstance(node.func, ast.Name)
                else None
            )
            if formal_mapping is None:
                return self.generic_visit(node)
            node.func = self.visit(node.func)
            node.args = [
                argument
                if plain_callee
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                else self.visit(argument)
                for argument in node.args
            ]
            for item in node.keywords:
                if item.arg is not None:
                    item.arg = formal_mapping.get(item.arg, item.arg)
                if not (
                    plain_callee
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    item.value = self.visit(item.value)
            return node

        def visit_keyword(self, node: ast.keyword) -> ast.AST:
            # Unknown and attribute/library call keyword spellings are public
            # API, not source identifiers. Local calls are handled by
            # visit_Call with their statically resolved callee formal map.
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
The supplied Responses schema is authoritative for the closed envelope, field
names, enums, optional fields, operation grammar, and structural descriptions.
Do not duplicate, rename, or extend it with convenience fields.

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

The trusted harness_compiler_view is byte-authoritative plumbing. Move every
logical p* handle into J.signature and every implementation-selection c*
handle into the single C object. Their relative source positions stay private
to Harness and are not serialized in the artifact. C values and compiler_slots
must be identical to the supplied anonymous values. For each compiler slot,
use exactly the public role in compiler_roles. Do not rename c* slots to p* and
do not turn compiler roles into inferred strategy names.

bound_constexpr_values are non-C invocation bindings. Their p* parameters
remain semantic constexpr handles in J.signature and never become C slots or
values. They already have logical semantic ownership: bind logical
value/predicate/result use to a dtype-compatible rank-0 J input, or bind an
integral extent-only use to a declared x* dimension. Never serialize a bound
constexpr as a representation-only pi carrier or give it the runtime-scalar
logical_ref=null form. A source constexpr not listed there is pure scheduling,
remains a c* parameter, and must have a matching pure-C constexpr slot. A
constexpr that fits neither logical J nor pure C is a Harness-invalid case, not
an ambiguity for Lift to repair.

Every J.signature logical_ref names only a declared J input/output t* handle or
a symbolic dimension x*; it never names a source/local or J-body v* SSA value.
When a runtime scalar or bound semantic constexpr is logical data used in a
predicate, operator, or result, declare a dtype-matching rank-0 J input tensor
with shape [] and bind that p* parameter to its t* handle. This input participates
in J dataflow. Only an integral extent-only parameter binds an x* dimension.
Never bind a value parameter to an output tensor merely to satisfy handle
existence; tensor dtype, rank, and input/output mode must agree exactly.
Conversely, a runtime scalar used only for representation/address realization
is not logical data: declare it as a rank-0 scalar, mode value, with public
dtype and no logical_ref or J input tensor; pi may reference its p* carrier.
Classify this only from the Source use, never from parameter spelling or
position. Every rank-0 tensor that you do bind through logical_ref must reach an
observable J output dependency or an explicit J-owned boundary/shape semantic
relation. TypeCheck rejects a dead logical input and never auto-unbinds it.
Every trusted compiler-view argument with kind `none` is an absent runtime
carrier: declare that p* exactly once as kind `opaque`, mode `value`, dtype
`none`, with null rank/alias_of/logical_ref. It contributes no J tensor, node,
access, effect, or pi instance; never replace it with a pointer or scalar.

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
For `exact` and `tolerance`, omit `update_node_ref` entirely: an empty string
or null is invalid. Only `one_of_updates` carries one non-empty
`update_node_ref` that names an existing logical update.

Every J.boundary access must have exactly one matching effect. The effect's
kind, target, node_ref (including null), and access_ref must agree exactly with
the access; never attach an otherwise matching access to a different node.

index_expression uses the closed ContractExpr syntax. For a direct dense
projection, k0 and [k0] are equivalent one-axis surface forms, while k1,k0 and
[k1, k0] are equivalent ordered two-axis forms; deterministic J elaboration
normalizes them before TypeCheck and Replay. Do not encode frontend APIs or
free prose as an index expression. For the rectangular functional-domain
profile, explicitly declare every logical axis bound in boundary.domains. If an
access uses a physical padding predicate such as k0 < x0, its direct index must
contain k0 and one domain must declare exactly 0 <= k0 and k0 < x0. A tensor
shape, Source fact, C value, pi route, or runtime value never fills a missing or
different domain relation. General logical predicates remain legal J; do not
delete them merely to satisfy this rectangular profile. A fill_value is legal
only on a load with a non-null predicate.
If a guard-free direct padding predicate and a complete zero-based rectangular
domain use the same-axis bound as the typed tensor extent, and that masked axis
is later eliminated by `reduce(fn=mean)`, a neutral zero fill closes only the
numerator: the current boundary schema has no valid-count/denominator contract.
Do not collapse such a Source program to padded `mean`. Preserve its
compiler-visible logical stages, for example `reduce(fn=sum)` followed by an
explicit logical division when that is what Source computes. Dense/unpredicated
mean, guard-bearing or cross-axis predicates, unrelated non-reduced masks, and
unknown-op propagation are not this TypeCheck error; consumers still fail
closed when they cannot project those boundaries.
For an unconditional access, omit both predicate and fill_value; never attach a
redundant fill to an unconditional access. For a pure-functional boundary with
any access predicate/fill,
the observable store-producing op's result type must be uniquely determined by
the closed op signature and J value graph; an explicit type is optional but, if
present, must agree. A direct same-handle output may omit its identity
output_mapping or spell that redundant identity explicitly; deterministic J
elaboration produces one omission form. Emit a non-identity output_mapping only
when an explicit IntentFunction.returns binding maps an output to a different
value, and make the two relations exactly identical.

When Source contains a compiler-visible semantic recurrence, J must state that
recurrence rather than only an algebraically equivalent final tensor formula.
Use the existing structured `loop` operation to declare the logical trip/scan,
every carried state and initial value, the ordered body update dataflow, and the
matching next/yield/result values. Keep each contraction's canonical subscripts
and result-axis orientation consistent with the values consumed by that update;
insert a transpose only when it is a real logical transformation. Do not replace
an online or loop-carried recurrence with a global contraction/reduction and ask
Replay to rediscover the missing update algorithm. Pi records only realization
topology, multiplicity, placement, and routes; it cannot carry, repair, or infer
an omitted initialization, update formula, yield, or contraction orientation.

Preserve every compiler-visible sensitive representation event in the Source,
including explicit or mandatory implicit integer widening, numeric cast,
accumulation dtype, reassociation, and native/privatized update choice. J owns
one typed realization_authorization for the logical carrier; pi attaches the
corresponding refinement and auth_ref to the exact realization instance. Use
distinct instances/refinements for distinct source stages. These events are
not whole-tensor operators, but omitting them changes Gate1 realization
closure; do not assume TypeCheck can infer an undeclared cast from source code.

C contains scheduling slots/values only. It must not contain operators,
predicates, source topology, APIs, policy, or a strategy field.
When one C prior has a source-neutral J attachment, write optional subject_ref
explicitly as one declared n*/d*/a*/q*/b* handle. Write axis_ref only when the
closed subject contract exposes that logical axis: a reduction input axis, loop
trip axis, canonical domain/index axis, or declared x* dimension. Do not infer
an attachment from role, source names, pi, or physical thread/lane axes. Global
priors may omit both fields; missing local attachment is safer than guessing.

Pi is not semantics and is not executable. It may record only anonymous
regions, instances referring to J/C handles, routes with operand slots, partial
order, sharing, placement, and the closed source-neutral mechanism tags in the
schema. Never put an operator, combiner, predicate, value, dtype, shape,
frontend API, source identifier, source fragment, prose, or rule identity in pi.

For every physical loop region that realizes a structured J `loop` through a
matching C tile slot, emit exactly one pi.iterations relation. The loop_instance
must anchor that J loop and the tile_instance must anchor that C slot. Use
`extent_step` when the physical induction space is zero-to-logical-extent with
the C tile as its step. Use `tile_ordinal` when it is a zero-based unit-step tile
ordinal whose trip count is the ceiling-divided logical extent. Preserve the
source realization distinction even when the induction variable is unused.
An explicit relation may use any role when its C slot is required,
positive-integer, and attached to the same loop/trip axis; axis ordinal 0 and
that loop's declared trip.axis x* are equivalent. The reserved roles
tile_extent and reduction_tile_extent only make a wholly missing relation
mechanically detectable. A custom or misclassified role does not invalidate
an explicit relation, but complete omission remains detectable only by Gate1.
This closed tag never contains a bound expression, numeric value, operator,
source spelling, or frontend API; do not encode untiled loops or invent a tile.

Pi preserves every compiler-visible source realization event not denoted by J
or pure C. Emit multiple instances with the same semantic_ref when one declared
handle has multiple observable realization stages. Every anonymous J.signature
or C handle materially used by address, transfer, synchronization, placement,
or sharing realization must anchor an instance; dependencies then route only
between the declared i* instance IDs.
One Source SSA coordinate or address intermediate that is defined once and
reused by several accesses remains one producer instance with routed consumers
and an explicit one-producer sharing relation. Multiple syntactic uses never
justify recomputing or duplicating that producer. Repeat a semantic_ref instance
only for distinct compiler-visible definitions/stages in Source, not once per
consumer and not to restate an already routed coordinate algebraically.
Every source program-coordinate occurrence must remain a distinct
domain-anchored realization path entering an address_base instance, even when
the current concrete grid extent on that axis is one. One logical domain may
therefore have both a program-coordinate path into address_base and a separate
lane-coordinate path into address_lane; never collapse the former into a loop
or erase it as constant-degenerate topology. Encode the pointer-carrier boundary
without adding fields: when a Source pointer/tensor carrier is applied at the
program-uniform base stage and that pointer result then receives the lane
contribution, route its p* instance directly into address_base and route
address_base to address_lane; this denotes two source pointer/address operations.
When Source instead forms program-uniform base and lane contributions as integers,
merges them, and only then applies the pointer carrier once, route the integer/
domain/C paths through address_base to address_lane while the pointer p* bypasses
address_base and enters address_lane or the final address stage. Do not choose
between these encodings from a base-to-lane edge alone. A lone address_lane is
legal only when Source has no separable program-uniform contribution.
Preserve source event multiplicity, def-use routes, partial order, regions, and
one-producer sharing exactly; do not use duplicate instances as an undocumented
code. Logical operators, predicates, fills, constants, index formulas, shapes,
and effects remain exclusively in J.

The source is already alpha-renamed. Do not invent or recover original names.
Output JSON only.
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

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Lift response contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        text,
        parse_constant=reject_nonfinite,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise TypeError("Lift response must be a JSON object")
    return parsed


def decode_scrc_lift_artifact(
    response_text: str,
    request: SCRCLiftRequest,
) -> Gate1Artifact:
    """Apply the one canonical J/C/pi decoder to any Lift backend output."""

    expected_parameters = tuple(
        opaque
        for _source, opaque in request.harness_binding.source_to_opaque
        if re.fullmatch(r"p[0-9]+", opaque)
    )
    artifact = _decode_scrc_lift_artifact_core(
        response_text,
        expected_parameters=expected_parameters,
        anonymous_compiler_view=request.anonymous_compiler_view,
        required_schema=Gate1Artifact.SCHEMA,
    )
    _validate_source_identifier_privacy(artifact, request.harness_binding)
    return artifact


def _decode_scrc_lift_artifact_core(
    response_text: str,
    *,
    expected_parameters: tuple[str, ...],
    anonymous_compiler_view: Mapping[str, Any] | None,
    required_schema: str,
) -> Gate1Artifact:
    """Shared source-free Decode core used by Runner and CandidateCheck."""

    payload = _json_object(response_text)
    artifact = canonicalize_gate1_artifact(Gate1Artifact.from_json(payload))
    if artifact.wire_schema != required_schema:
        raise ValueError(
            f"Lift artifact uses {artifact.wire_schema!r}; "
            f"this round requires {required_schema!r}"
        )
    _validate_anonymous_lift_core(artifact, expected_parameters)
    _validate_compiler_view(artifact, anonymous_compiler_view)
    return artifact


def check_scrc_lift_candidate(
    response_text: str,
    context: SCRCCandidateCheckContext,
) -> dict[str, Any]:
    """Run the authoritative decode/TypeCheck core and return diagnostics only."""

    if (
        context.schema_version != Gate1Artifact.SCHEMA
        or context.decode_protocol != SCRC_DECODE_PROTOCOL
        or context.typecheck_protocol != TYPECHECK_PROTOCOL
    ):
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_check_context_mismatch",
                    "path": "<context>",
                    "observed_refs_or_types": [
                        context.schema_version,
                        context.decode_protocol,
                        context.typecheck_protocol,
                    ],
                }
            ],
        }
    try:
        artifact = _decode_scrc_lift_artifact_core(
            response_text,
            expected_parameters=context.expected_parameters,
            anonymous_compiler_view=context.anonymous_compiler_view,
            required_schema=context.schema_version,
        )
    except Exception as exc:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": _candidate_decode_diagnostics(exc),
        }

    # This is the same callable used by the formal Runner.  CandidateCheck does
    # not retain or return the canonical artifact and therefore cannot become a
    # second acceptance path.
    report = type_check(artifact.j, artifact.c, artifact.pi)
    return {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass" if report.passed else "fail",
        "diagnostics": [
            {
                "rule_code": item.code.value,
                "path": _bounded_candidate_diagnostic_text(item.path),
                "observed_refs_or_types": (
                    [_bounded_candidate_diagnostic_text(item.ref)]
                    if item.ref is not None
                    else []
                ),
            }
            for item in report.diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]
        ],
    }


def _candidate_decode_diagnostics(exc: Exception) -> list[dict[str, Any]]:
    """Expose only independent errors already found by the strict decoder."""

    if not isinstance(exc, SchemaError) or not exc.issues:
        return [
            {
                "rule_code": "decode_rejected",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            }
        ]

    diagnostics: list[dict[str, Any]] = []
    for issue in exc.issues[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = issue.path
        if path == "LogicalSignature" or path.startswith("LogicalSignature."):
            path = "j.signature" + path.removeprefix("LogicalSignature")
        elif path == "SemanticArchive" or path.startswith("SemanticArchive."):
            path = "j" + path.removeprefix("SemanticArchive")
        diagnostics.append(
            {
                "rule_code": "decode_rejected",
                "path": _bounded_candidate_diagnostic_text(path),
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"SchemaError: {issue.detail}"
                    )
                ],
            }
        )
    return diagnostics


def _bounded_candidate_diagnostic_text(value: Any) -> str:
    """Keep source-free CandidateCheck diagnostics closed and compact."""

    text = str(value)
    marker = "...<truncated>"
    if len(text) <= _CANDIDATE_DIAGNOSTIC_TEXT_LIMIT:
        return text
    return text[: _CANDIDATE_DIAGNOSTIC_TEXT_LIMIT - len(marker)] + marker


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
    artifact: Gate1Artifact, anonymous_view: Mapping[str, Any] | None
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
            "none": ParameterKind.OPAQUE,
        }[argument["kind"]]
        allowed_kinds = {expected_kind}
        if expected_kind is ParameterKind.POINTER:
            allowed_kinds.add(ParameterKind.TENSOR)
        if parameter.kind not in allowed_kinds:
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


def _validate_anonymous_lift_core(
    artifact: Gate1Artifact,
    expected_parameters: tuple[str, ...],
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
    if [item.name for item in artifact.j.signature.parameters] != list(
        expected_parameters
    ):
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


def _validate_source_identifier_privacy(
    artifact: Gate1Artifact,
    binding: SCRCHarnessBinding,
) -> None:
    """Apply the privileged source-name leak check only in the formal decoder."""

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
    prompt_sha = request.prompt_sha256
    reply = call_responses(
        instructions,
        user,
        provider=provider,
        effort=effort,
        text_format=scrc_lift_core_text_format(),
    )
    try:
        artifact = decode_scrc_lift_artifact(reply.text, request)
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
    return SCRCExtraction(
        artifact,
        provenance,
        request.harness_binding,
        source,
    )


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
                "artifact_schema": extraction.artifact.wire_schema,
                "decode_protocol": SCRC_DECODE_PROTOCOL,
                "typecheck_protocol": TYPECHECK_PROTOCOL,
            },
        )


__all__ = [
    "ResponsesSCRCLiftAdapter", "ResponsesSCRCRelifter",
    "CANDIDATE_CHECK_PROTOCOL", "SCRC_DECODE_PROTOCOL", "SCRC_RELIFT_PROTOCOL",
    "SCRCCandidateCheckContext", "SCRCCompilerArgument",
    "SCRCCompilerView", "SCRCExtraction", "SCRCHarnessBinding",
    "SCRCLiftAttempt", "SCRCLiftError", "SCRCLiftRequest", "SCRCProvenance", "SCRCReliftContext",
    "SCRCReliftRecord", "anonymize_source", "build_scrc_lift_instructions",
    "check_scrc_lift_candidate", "decode_scrc_lift_artifact", "extract_scrc_artifact", "prepare_scrc_lift", "scrc_candidate_check_context", "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
]
