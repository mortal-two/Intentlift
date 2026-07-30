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
from intentir.ir.dim_expr import validate_dimension_instantiation
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ParameterKind,
    ScheduleSlotKind,
    SchemaError,
)
from intentir.verify.typecheck import TYPECHECK_PROTOCOL, type_check

from .scrc_schema import scrc_lift_core_json_schema, scrc_lift_core_text_format


SCRC_RELIFT_PROTOCOL = "llm_direct_pi_structured_source_aware_relift_v2"
SCRC_DECODE_PROTOCOL = "intentir.scrc.decode.v16"
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
    """Structured TypeCheck/Gate1/Gate2 evidence visible only to Lift."""

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
            "gate2",
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
    continuation_candidate: Mapping[str, Any] | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True)
class SCRCCompilerArgument:
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None
    mode: str | None = None
    shape_contract: tuple[int | str, ...] | None = None
    shape_coordinate: str | None = None

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
        shape_contract = (
            tuple(self.shape_contract)
            if self.shape_contract is not None
            else None
        )
        if self.kind != "tensor" and shape_contract is not None:
            raise ValueError("only tensor compiler arguments have a shape contract")
        if shape_contract is not None:
            if len(shape_contract) != len(shape):
                raise ValueError(
                    "compiler argument shape contract rank differs from valuation"
                )
            for axis, (contract, extent) in enumerate(
                zip(shape_contract, shape, strict=True)
            ):
                if isinstance(contract, bool) or not isinstance(contract, (int, str)):
                    raise TypeError(
                        "shape contract entries must be positive integers or h* coordinates"
                    )
                if isinstance(contract, int):
                    if contract <= 0 or contract != extent:
                        raise ValueError(
                            "static shape contract extent differs from its valuation"
                        )
                elif not re.fullmatch(r"h[0-9]+", contract):
                    raise ValueError(
                        f"shape contract axis {axis} must use an anonymous h* coordinate"
                    )
        shape_coordinate = self.shape_coordinate
        if shape_coordinate is not None:
            if self.kind != "scalar":
                raise ValueError(
                    "only scalar compiler arguments bind a shape coordinate"
                )
            if not re.fullmatch(r"h[0-9]+", shape_coordinate):
                raise ValueError(
                    "scalar shape-coordinate binding must use an anonymous h* coordinate"
                )
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer valuation"
                )
            if self.dtype.casefold() != "index" and re.fullmatch(
                r"[iu][0-9]+", self.dtype.casefold()
            ) is None:
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer dtype"
                )
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
        object.__setattr__(self, "shape_contract", shape_contract)
        object.__setattr__(self, "shape_coordinate", shape_coordinate)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
        }
        if self.kind == "tensor":
            result["shape"] = list(self.shape)
            if self.shape_contract is not None:
                result["shape_contract"] = list(self.shape_contract)
        elif self.kind == "scalar":
            result["value"] = self.value
            if self.shape_coordinate is not None:
                result["shape_coordinate"] = self.shape_coordinate
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
    bound_constexpr_shape_coordinates: Mapping[str, str] = field(
        default_factory=dict
    )
    compiler_slots: tuple[str, ...] = ()
    aliases: tuple[AliasSpec, ...] = ()
    arguments: tuple[SCRCCompilerArgument, ...] = ()
    grid: tuple[int, ...] = ()
    grid_contract: tuple[str, ...] = ()
    grid_shape_coordinates: tuple[str | None, ...] = ()
    parameter_order: tuple[str, ...] = ()
    constexpr_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.schedule_values)
        bound_constexpr_values = dict(self.bound_constexpr_values)
        bound_constexpr_shape_coordinates = dict(
            self.bound_constexpr_shape_coordinates
        )
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
        object.__setattr__(
            self,
            "bound_constexpr_shape_coordinates",
            bound_constexpr_shape_coordinates,
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
        shape_coordinates: dict[str, int] = {}
        coordinate_order: list[str] = []
        for argument in self.arguments:
            if argument.shape_contract is None:
                continue
            for contract, extent in zip(
                argument.shape_contract, argument.shape, strict=True
            ):
                if not isinstance(contract, str):
                    continue
                if contract not in shape_coordinates:
                    shape_coordinates[contract] = extent
                    coordinate_order.append(contract)
                elif shape_coordinates[contract] != extent:
                    raise ValueError(
                        f"shape coordinate {contract!r} has inconsistent valuations"
                    )
        expected_coordinates = [
            f"h{index}" for index in range(len(coordinate_order))
        ]
        if coordinate_order != expected_coordinates:
            raise ValueError(
                "shape coordinates must use gap-free first-declaration order h0,h1,..."
            )
        for argument in self.arguments:
            coordinate = argument.shape_coordinate
            if coordinate is None:
                continue
            if coordinate not in shape_coordinates:
                raise ValueError(
                    "scalar shape-coordinate binding names an undeclared coordinate"
                )
            if argument.value != shape_coordinates[coordinate]:
                raise ValueError(
                    "scalar and tensor shape-coordinate valuations differ"
                )
        for name, coordinate in bound_constexpr_shape_coordinates.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    "bound constexpr shape-coordinate names must be identifiers"
                )
            if name not in bound_constexpr_values:
                raise ValueError(
                    "bound constexpr shape-coordinate binding requires a bound value"
                )
            if not isinstance(coordinate, str) or coordinate not in shape_coordinates:
                raise ValueError(
                    "bound constexpr shape-coordinate binding names an undeclared coordinate"
                )
            value = bound_constexpr_values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != shape_coordinates[coordinate]
            ):
                raise ValueError(
                    "bound constexpr and tensor shape-coordinate valuations differ"
                )
        grid = tuple(int(dim) for dim in self.grid)
        if grid and (len(grid) > 3 or any(dim <= 0 for dim in grid)):
            raise ValueError("compiler grid must have one to three positive dimensions")
        grid_contract = tuple(self.grid_contract)
        if any(not isinstance(item, str) or not item for item in grid_contract):
            raise ValueError("grid contract must contain non-empty expressions")
        if not grid and grid_contract:
            raise ValueError("grid contract requires a concrete compiler grid")
        if grid and grid_contract and len(grid) != len(grid_contract):
            raise ValueError("concrete grid and grid contract ranks differ")
        grid_shape_coordinates = tuple(self.grid_shape_coordinates)
        if grid_shape_coordinates and len(grid_shape_coordinates) != len(grid):
            raise ValueError(
                "grid shape-coordinate bindings and concrete grid ranks differ"
            )
        for axis, coordinate in enumerate(grid_shape_coordinates):
            if coordinate is None:
                continue
            if not isinstance(coordinate, str) or coordinate not in shape_coordinates:
                raise ValueError(
                    "grid shape-coordinate binding must name a declared h* coordinate"
                )
            if grid[axis] != shape_coordinates[coordinate]:
                raise ValueError(
                    "grid extent differs from its shape-coordinate valuation"
                )
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
        object.__setattr__(
            self, "grid_shape_coordinates", grid_shape_coordinates
        )
        object.__setattr__(self, "_shape_coordinates", shape_coordinates)
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
            "bound_constexpr_shape_coordinates": dict(
                self.bound_constexpr_shape_coordinates
            ),
            "compiler_slots": list(self.compiler_slots),
            "aliases": [item.to_json() for item in self.aliases],
            "arguments": [item.to_json() for item in self.arguments],
            "shape_coordinates": dict(self.shape_coordinates),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "grid_shape_coordinates": list(self.grid_shape_coordinates),
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
        }

    @property
    def shape_coordinates(self) -> Mapping[str, int]:
        """Current valuations for source-grounded anonymous shape coordinates."""

        return dict(getattr(self, "_shape_coordinates", {}))


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


class SourceIdentifierPrivacyError(ValueError):
    """A submitted artifact retained a private Source spelling at one JSON path.

    The diagnostic deliberately exposes only the artifact path.  Returning the
    original spelling to a source-aware owner would defeat the anonymous naming
    boundary that this check protects.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"Lift response retained a source identifier at {path}"
        )


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
        "bound_constexpr_shape_coordinates": {
            mapping[name]: coordinate
            for name, coordinate in view.bound_constexpr_shape_coordinates.items()
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
        "shape_coordinates": dict(view.shape_coordinates),
        "grid": list(view.grid),
        "grid_contract": [
            _replace_names(item, mapping) for item in view.grid_contract
        ],
        "grid_shape_coordinates": list(view.grid_shape_coordinates),
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

Use gap-free declaration-order names: kernel scrc_k0; tensors t*; J nodes n*;
values v*; dimensions x*; domains d*; accesses a*; effects e*; scopes q*;
storage b*; parameters p*; C slots c*; pi regions r* and instances i*.
J t* numbering follows first declaration across inputs then outputs; inout
reuses its input handle. Never derive t* from p* position: p0 input, p1 output,
p2 extent bind t0, t1, x0 respectively.

The semantic IntentFunction is scrc_j0. node_ids maps n* to body paths, never
v* results. boundary.domains maps d* directly to portable boolean strings,
never axes/members objects.

The trusted harness_compiler_view is byte-authoritative plumbing. Put logical
p* in J.signature and implementation-selection c* in the single C; source
positions stay private. C values/compiler_slots equal the anonymous inputs and
each compiler slot uses compiler_roles exactly. Never rename c* to p* or infer
strategy names from roles.

For every tensor, `shape` is only the current Harness binding and is never
permission to specialize J. In `shape_contract`, an
integer stays a static J literal and each source-grounded h* becomes a symbolic
x*: equal h* coordinates share x*, distinct h* remain distinct even when their
values coincide. Choose one x* for each h* and use it in every affected J tensor shape.
h* is Harness-only. An integral `shape_coordinate:h*` scalar or explicitly bound
semantic constexpr
binds directly to that x* when extent/index/domain-only; if it also carries
logical data, bind one rank-0 t* and author the t*=x*
value_dimension_relations equality. Fixed code checks but never creates either
form. Never infer h* from values, shape, grid, C, names, or coincident numbers.
Capacity and active-length coordinates remain distinct: preserve Source padding,
domains, and required non-negative affine constraints, but do not invent a
C-dependent roundup equation. Grid/C/pi may realize rounding, not J ownership.
When Source itself establishes a logical dimension identity, author it in the
complete J.dimension_relations array. The only legal expressions are identity,
add_const, mul_const, ceil_div_const, and round_up_const over declared x* handles
and positive integer constants. Typical examples are a full extent that Source
defines as twice a half extent or a logical rounded capacity that Source defines
from an active extent. Never infer such a relation from equal concrete values,
Harness shapes, grid, C, pi, TTIR layout, or a frontend convention. Emit [] when
Source establishes no dimension relation; fixed code only validates and consumes.

bound_constexpr_values are Harness-owned, non-C JIT bindings; constexprness alone
never makes one a graph selector. Across all Source uses, logical
data/predicate/result becomes one dtype-compatible rank-0 t*, while integral
extent-only becomes x*. A null logical_ref is allowed only when every semantic
use is exclusively compile-time control and the active graph has no shape,
index, mask, arithmetic, boolean, or result use. Such a pure selector stays a
constexpr value carrier: its value stays only in Harness; author only its selected
complete graph. fixed code checks only the authored wire/type contract and never
selects or repairs the graph. String constexpr is pure-selector-only or unsupported.

A J.signature logical_ref names only a declared t* input/output or x*, never p*,
a Source/local name, or body v*. Runtime or constexpr logical data binds one
rank-0 t*. If it also controls an active predicate, reuse that same t* in the
predicate and arithmetic dataflow; never split one Source value across t*, x*,
or null. The equality value(t*)=x* is explicit. Author it only from exact Source
identity, never a valuation, dtype, position, or shape coincidence; an extent-only
scalar binds x* directly.
Emit the complete J.value_dimension_relations array, including []; fixed code and
Replay never add or repair it. Symbolic x* are signature/type handles, not SSA
values; never list x* in portable scope definitions/captures or storage members.
Also emit the complete J.dimension_relations array, including [].

Each physical p* has exactly one logical_ref. Dtype/rank/mode must agree; inout binds one tensor
declared as both input and output, and value parameters never bind outputs merely
to satisfy existence. A representation/address-only runtime scalar is rank-0
mode=value with public dtype and null logical_ref; pi may reference that p*.
Classify only from Source use. Any bound rank-0 t* must reach an observable output
or explicit J boundary/shape semantic. A compiler-view kind=`none` argument is
exactly one opaque mode=value dtype=`none` p* with null rank/alias/logical_ref and
contributes no J tensor, node, access, effect, or pi instance.

For every observable output, J.boundary.result_relations declares only the
closed semantic comparison class; it never declares numeric thresholds. Use
`exact` only when cross-backend equality is part of the logical contract (for
example integer/boolean results, bitwise operations, or exactly specified
copies). Any output reached by floating arithmetic uses `tolerance` unless the
Source itself establishes cross-backend bitwise equality. This includes ordinary
f16/bf16/f32 pointwise add/sub/mul/div as well as casts, transcendentals,
reductions, contraction, and reassociation: target evaluation width and rounding
may differ even when the logical arithmetic graph is the same. The trusted
external Gate2 policy Q supplies all atol, rtol, ULP, and NaN rules. Never place
a Python comparator or a candidate-selected tolerance in J, C, or pi.
For `exact` and `tolerance`, omit `update_node_ref` entirely: an empty string
or null is invalid. Only `one_of_updates` carries one non-empty
`update_node_ref` that names an existing logical update.

Every J.boundary access must have exactly one matching effect. The effect's
kind, target, node_ref (including null), and access_ref must agree exactly with
the access; never attach an otherwise matching access to a different node. A
direct logical-input load may use node_ref=null with the matching null effect;
do not invent map_unary(identity) solely to create an access anchor.
An access attached to one node is consumer-local: it does not silently change
the same tensor operand at a sibling node. When Source reuses one physical
tensor through different logical indices, predicates, fills, or sub-domains at
different Layer-A consumers, author each distinct consumer-local access/effect
explicitly at the node that consumes that view. Conversely, do not duplicate an
access merely because several consumers use the same complete logical tensor.
Fixed code never copies, propagates, or recommends a sibling access relation.

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
only on a load with an authored predicate or access_guard_relation.
For a predicate-bearing Source load, fill_value is that access's actual authored
other value or Layer-A value reference. Never substitute a downstream neutral
element, a value chosen to cancel a later operation, or another observationally
equivalent fill. If Source omits other, omit fill_value. Consumer-specific
outside identities and same-predicate masked stores are separate facts and do
not rewrite the Source load fill.
Trace every predicate-bearing load through closed pointwise/cast/transpose/pad
dataflow to each sum/max/min reduce or inclusive scan that consumes its masked
axis. For every distinct (consumer_ref, operand_index, access_ref), author one
boundary.consumer_extensions relation with that consumer's Source-local tile
axis and typed two-sided outside identity. A load fill or reduction identity is
not this per-consumer relation; one access may require several relations for
different downstream consumers. Author only Source-established outside values.
A computed Layer-A i1 access guard gets exactly one
boundary.access_guard_relations item with increasing target-broadcast axes and
closed polarity; do not copy it into the predicate or boundary.broadcasts. A
guarded load has a fill. Every guarded store targets one tensor authored as
both input and output under one mode=inout plus explicit select. Outside a loop,
inactive is that tensor and returns maps output to select result. Inside a loop,
carry.init_value is that tensor, inactive is current, and select result is
carry.next_value. Positive selects candidate/current; negative selects
current/candidate. One carry has one guarded write.
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
More generally, an explicit Source `reduce(sum)` followed by division remains
those two ordered J nodes even when the divisor equals an extent. Never replace
it with `reduce(mean)`, and never replace its divisor dependency with a generated
`reduce(sum)` over ones. These algebraic rewrites erase compiler-visible stages
and logical scalar dataflow that Gate1 is required to observe.
For an unconditional access, omit both predicate and fill_value; never attach a
redundant fill to an unconditional access. For a pure-functional boundary with
any access predicate/fill,
the observable store-producing op's result type must be uniquely determined by
the closed op signature and J value graph; an explicit type is optional but, if
present, must agree. A direct same-handle output needs no identity
output_mapping. Otherwise author `returns` or `output_mapping`; one suffices and
both must agree. Deterministic J elaboration checks this binding but never
chooses its value.

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
Each scan name retains the complete logical tensor type of its `from` source;
it is not a physical tile and does not remove the trip axis. Pointwise body ops
retain that axis until an explicit J reduce/scan consumes it. Conversely, a
physical strip-mined loop implementing an otherwise ordinary reduction is
Layer C/pi realization rather than evidence for a semantic loop. A
boundary.reductions entry describes an existing J reduce/scan and cannot attach
to a loop to manufacture the missing semantic contraction.

An effect-controlling binary Source branch is one Layer-A `effect_if`: one
visible scalar-i1 predicate and explicit positive/negative `{{scope_id,body}}`
arms (either may be empty) bound to distinct sibling branch_region scopes. It
has no result/yield; arm-local SSA never escapes. scope.effects plus matching
EffectSpec/access_ref is sole membership and cannot overlap access_guard. Pi
only realizes each arm scope in sibling placement=branch regions; it never
copies predicate/polarity or turns the branch into select.
Source early return: final `effect_if`; later work in its continuation arm; one
pi.terminal_controls pair over arm `realize` instances. It names only
terminal/continuation, never predicate/polarity/return/Source/API.
Preserve selected-helper literal memory order in scope.effects/pi; node order is
not effect order. Outline loop-preexisting writes are only a carrier checklist.

A value-yielding Source `scf.if` is one Layer-A `value_if`, never `select`:
author a visible scalar-i1 predicate, one result/type, and two explicit
`{{scope_id,body,yield_value}}` arms. Bodies may be empty/capture outer values
but admit only known pure ordinary ops, no access/effect/unknown/nested region.
Both yields exactly match the result type and only that result escapes. Arm
scopes and pi branch regions are distinct siblings. Fixed code never fills,
derives from select, or rewrites this structure.

Preserve every compiler-visible sensitive representation event in Source. J owns
its typed realization_authorization; pi attaches the matching refinement and
auth_ref to the exact instance. Use distinct instances/refinements for distinct
source stages; fixed code cannot infer an undeclared cast; never propagate one
conversion to sibling parameters, strides, or instances. Signed/unsigned widen
states explicit same-signedness source_dtype and a strictly wider target_dtype;
equal-width or missing dtype pairs are not widening.

For each J `reduce` or inclusive `scan`, boundary.reductions.tile_axis is the
Source frontend's local tile axis after program/grid axes select the program instance. It
is not the whole-tensor logical axis number from reduce.attrs.axes. Preserve the
observed local axis; do not copy a logical rank axis or add program axes to it.
For a Source reduction with omitted frontend axis and a compiler-visible flatten,
keep logical axes in J; add a distinct pi `realize` refined as
`reduction_linearization` (`auth_ref=null`) with one slot-0 route to the reduce
instance. Never use it for an explicit axis or put frontend text/API in pi.

J `const` is exact: finite JSON number, or only `"-inf"`/`"inf"`/`"nan"`
with floating dtype. Never approximate it or move it to C/pi.

C owns scheduling slots/values only: no operator, predicate, Source topology,
API, policy, or strategy. An optional source-neutral attachment names one
declared n*/d*/a*/q*/b* subject_ref. Add axis_ref only for the subject's exact
reduction/loop/contraction/domain/index/x* axis; never infer it from role, names,
pi, or hardware axes. Global priors may omit both. A C attachment never names
a p* parameter, source/local, or J-body v*.

Pi is non-semantic, non-executable anonymous topology. Except for the closed
route-extent divisor/rounding below, it contains no operator/value and never a
predicate, dtype, logical shape, frontend API, Source spelling, prose, or rule ID.

For a tiled physical realization of a structured J `loop`, author exactly one
pi.iterations relation: loop_instance anchors that loop and tile_instance a
required positive-integer C slot on its trip axis. `extent_step` is zero to trip
by tile; `rounded_extent_step` uses the same lower/step and upper
ceil_div(trip,tile)*tile; `tile_ordinal` is zero-based unit-step tile ordinals.
Preserve the Source basis even if unused. Any C role is valid when the explicit
dependent anchors match; reserved tile roles only expose a wholly missing
relation to TypeCheck. The tag has no formula/value/API and never denotes an
untiled loop.

For a physical tiled loop around an existing non-loop J reduce/contraction/access/
domain, author physical_iterations with its same-subject/axis positive-int C tile
and Source basis/direction; structured J loops use iterations. Add physical carriers
only for visible state, in Source tuple order. `pi.order` alone orders events.

For a Source C-extent partition at one existing route slot, author exact
source/target/slot, divisor>1, floor/ceil, and target-domain/axis attachment.
It changes only that representation use, never J or sibling routes.

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
legal only when Source has no separable program-uniform contribution. Every
transfer_direct/async must be route-reachable from a p* pointer/tensor carrier
bound to that access's J tensor.
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
    schema_diagnostics = _candidate_json_schema_diagnostics(response_text)
    if schema_diagnostics:
        occupied_paths = {item["path"] for item in schema_diagnostics}
        for item in _candidate_surface_diagnostics(response_text, context):
            if len(schema_diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                break
            if item["path"] in occupied_paths:
                continue
            schema_diagnostics.append(item)
            occupied_paths.add(item["path"])
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": schema_diagnostics,
        }

    try:
        artifact = _decode_scrc_lift_artifact_core(
            response_text,
            expected_parameters=context.expected_parameters,
            anonymous_compiler_view=context.anonymous_compiler_view,
            required_schema=context.schema_version,
        )
    except Exception as exc:
        diagnostics = _candidate_decode_diagnostics(exc)
        occupied_paths = {item["path"] for item in diagnostics}
        for item in _candidate_surface_diagnostics(response_text, context):
            if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                break
            if item["path"] in occupied_paths:
                continue
            diagnostics.append(item)
            occupied_paths.add(item["path"])
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": diagnostics,
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
                "observed_refs_or_types": _candidate_typecheck_observations(
                    item.ref,
                    item.detail,
                ),
            }
            for item in report.diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]
        ],
    }


def _candidate_json_schema_diagnostics(
    response_text: str,
) -> list[dict[str, Any]]:
    """Aggregate independent structural errors before dependent Decode.

    This is the same closed wire schema already supplied to every Lift backend.
    It never fills a field or changes acceptance: an empty result merely allows
    the existing strict Decode/TypeCheck path to run.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return []
    try:
        import jsonschema
    except ImportError:
        return []

    schema = scrc_lift_core_json_schema()
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (
            tuple(str(item) for item in error.absolute_path),
            tuple(str(item) for item in error.absolute_schema_path),
            error.message,
        ),
    )
    diagnostics: list[dict[str, Any]] = []
    for error in errors[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = ""
        for item in error.absolute_path:
            if isinstance(item, int):
                path += f"[{item}]"
            else:
                path += ("." if path else "") + str(item)
        diagnostics.append(
            {
                "rule_code": "schema_rejected",
                "path": _bounded_candidate_diagnostic_text(
                    path or "<candidate>"
                ),
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(error.message)
                ],
            }
        )
    return diagnostics


def _candidate_surface_diagnostics(
    response_text: str,
    context: SCRCCandidateCheckContext,
) -> list[dict[str, Any]]:
    """Report independent authored-surface mistakes beside a Decode failure.

    The strict decoder remains the only acceptance boundary.  This pass reads
    raw JSON only after that decoder has already rejected the candidate and
    checks closed naming/ABI facts that do not depend on repairing or
    interpreting an invalid graph.  It never mutates, canonicalizes, or fills
    a field, and its diagnostics cannot turn rejection into acceptance.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return []

    diagnostics: list[dict[str, Any]] = []

    def mismatch(
        rule_code: str,
        path: str,
        expected: Any,
        actual: Any,
    ) -> None:
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            return
        diagnostics.append(
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        "expected=" + json.dumps(
                            expected,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    ),
                    _bounded_candidate_diagnostic_text(
                        "actual=" + json.dumps(
                            actual,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    ),
                ],
            }
        )

    j = payload.get("j")
    if not isinstance(j, Mapping):
        return diagnostics
    intent = j.get("intent")
    fn = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    if isinstance(parameters, list) and all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str)
        for item in parameters
    ):
        actual_parameters = [str(item["name"]) for item in parameters]
        if actual_parameters != list(context.expected_parameters):
            mismatch(
                "construction_handle_sequence",
                "j.signature.parameters[*].name",
                list(context.expected_parameters),
                actual_parameters,
            )

    tensors: list[Mapping[str, Any]] = []
    if isinstance(fn, Mapping):
        for collection in ("inputs", "outputs"):
            values = fn.get(collection)
            if isinstance(values, list):
                tensors.extend(
                    item for item in values if isinstance(item, Mapping)
                )
    tensor_names = [item.get("name") for item in tensors]
    if tensor_names and all(isinstance(item, str) for item in tensor_names):
        actual_tensors = list(dict.fromkeys(tensor_names))
        expected_tensors = [f"t{index}" for index in range(len(actual_tensors))]
        if actual_tensors != expected_tensors:
            mismatch(
                "construction_handle_sequence",
                "j.intent.intent_function.inputs+outputs[*].name",
                expected_tensors,
                actual_tensors,
            )

    dimensions: list[str] = []
    for tensor in tensors:
        shape = tensor.get("shape")
        if not isinstance(shape, list):
            continue
        for extent in shape:
            if isinstance(extent, str) and extent not in dimensions:
                dimensions.append(extent)
    expected_dimensions = [f"x{index}" for index in range(len(dimensions))]
    if dimensions != expected_dimensions:
        mismatch(
            "construction_handle_sequence",
            "j.intent.intent_function tensor shape dimensions",
            expected_dimensions,
            dimensions,
        )

    def authored_ops(values: Any) -> list[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        if not isinstance(values, list):
            return result
        for item in values:
            if not isinstance(item, Mapping):
                continue
            result.append(item)
            for arm_name in ("positive", "negative"):
                arm = item.get(arm_name)
                if isinstance(arm, Mapping):
                    result.extend(authored_ops(arm.get("body")))
            result.extend(authored_ops(item.get("body")))
        return result

    body = fn.get("body") if isinstance(fn, Mapping) else None
    operations = authored_ops(body)
    node_ids = [item.get("id") for item in operations]
    if node_ids and all(isinstance(item, str) for item in node_ids):
        expected_nodes = [f"n{index}" for index in range(len(node_ids))]
        if node_ids != expected_nodes:
            mismatch(
                "construction_handle_sequence",
                "j.intent.intent_function.body recursive node ids",
                expected_nodes,
                node_ids,
            )
    authored_node_map = j.get("node_ids")
    if isinstance(authored_node_map, Mapping) and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in authored_node_map.items()
    ):
        path_order = {
            path: index
            for index, operation in enumerate(operations)
            if isinstance((path := authored_node_map.get(operation.get("id"))), str)
        }
        ordered_keys = [
            key
            for key, _value in sorted(
                authored_node_map.items(),
                key=lambda item: path_order.get(item[1], len(path_order)),
            )
        ]
        expected_map_keys = [f"n{index}" for index in range(len(ordered_keys))]
        if ordered_keys != expected_map_keys:
            mismatch(
                "construction_handle_sequence",
                "j.node_ids keys in authored body-path order",
                expected_map_keys,
                ordered_keys,
            )

    boundary = j.get("boundary")
    handle_specs = (
        ("d", "domains", None),
        ("a", "accesses", "access_id"),
        ("e", "effects", "effect_id"),
        ("q", "scopes", "scope_id"),
        ("b", "storage", "storage_id"),
    )
    for prefix, field_name, id_field in handle_specs:
        owner: Any = boundary if field_name in {"domains", "accesses", "effects"} else j
        values = owner.get(field_name) if isinstance(owner, Mapping) else None
        if id_field is None:
            actual = (
                sorted(
                    values,
                    key=lambda item: (
                        int(match.group(1))
                        if isinstance(item, str)
                        and (match := re.fullmatch(r"d([0-9]+)", item))
                        else math.inf,
                        str(item),
                    ),
                )
                if isinstance(values, Mapping)
                else None
            )
        elif isinstance(values, list) and all(
            isinstance(item, Mapping) and isinstance(item.get(id_field), str)
            for item in values
        ):
            actual = [str(item[id_field]) for item in values]
        else:
            actual = None
        if actual is None:
            continue
        expected = [f"{prefix}{index}" for index in range(len(actual))]
        if actual != expected:
            mismatch(
                "construction_handle_sequence",
                f"j.{field_name}" + (f"[*].{id_field}" if id_field else " keys"),
                expected,
                actual,
            )

    pi = payload.get("pi")
    if isinstance(pi, Mapping):
        for field_name, prefix in (("regions", "r"), ("instances", "i")):
            values = pi.get(field_name)
            if not isinstance(values, list) or not all(
                isinstance(item, Mapping) and isinstance(item.get("id"), str)
                for item in values
            ):
                continue
            actual = [str(item["id"]) for item in values]
            expected = [f"{prefix}{index}" for index in range(len(actual))]
            if actual != expected:
                mismatch(
                    "construction_handle_sequence",
                    f"pi.{field_name}[*].id",
                    expected,
                    actual,
                )
        order = pi.get("order")
        if isinstance(order, list):
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in order
            ]
            duplicates = sorted(
                item for item in set(encoded) if encoded.count(item) > 1
            )
            if duplicates:
                mismatch(
                    "construction_duplicate_relation",
                    "pi.order",
                    "unique order edges",
                    [json.loads(item) for item in duplicates],
                )

    view = context.anonymous_compiler_view
    if isinstance(view, Mapping) and isinstance(parameters, list):
        by_name = {
            item.get("name"): item
            for item in parameters
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        arguments = view.get("arguments")
        if isinstance(arguments, list):
            for argument in arguments:
                if not isinstance(argument, Mapping):
                    continue
                name = argument.get("name")
                parameter = by_name.get(name)
                if not isinstance(name, str) or not isinstance(parameter, Mapping):
                    continue
                source_kind = argument.get("kind")
                allowed_kinds = {
                    "tensor": ["pointer", "tensor"],
                    "scalar": ["scalar"],
                    "opaque": ["opaque"],
                    "none": ["opaque"],
                }.get(source_kind)
                expected_fields: dict[str, Any] = {}
                if allowed_kinds is not None:
                    expected_fields["kind"] = allowed_kinds
                if "dtype" in argument:
                    expected_fields["dtype"] = _canonical_dtype(argument.get("dtype"))
                if source_kind == "tensor" and isinstance(argument.get("shape"), list):
                    expected_fields["rank"] = len(argument["shape"])
                if "mode" in argument:
                    expected_fields["mode"] = argument.get("mode")
                for field_name, expected in expected_fields.items():
                    actual = parameter.get(field_name)
                    if field_name == "dtype" and isinstance(actual, str):
                        actual = _canonical_dtype(actual)
                    matched = actual in expected if isinstance(expected, list) else actual == expected
                    if not matched:
                        mismatch(
                            "construction_abi_mismatch",
                            f"j.signature.parameters[{name}].{field_name}",
                            expected,
                            actual,
                        )
        c = payload.get("c")
        if isinstance(c, Mapping) and c.get("values") != view.get("schedule_values"):
            mismatch(
                "construction_abi_mismatch",
                "c.values",
                view.get("schedule_values"),
                c.get("values"),
            )
        if isinstance(c, Mapping):
            raw_slots = c.get("slots")
            slots_by_name = (
                {
                    item.get("name"): item
                    for item in raw_slots
                    if isinstance(item, Mapping)
                    and isinstance(item.get("name"), str)
                }
                if isinstance(raw_slots, list)
                else {}
            )
            bound_non_c = view.get("bound_constexpr_values")
            if isinstance(bound_non_c, Mapping):
                for name in bound_non_c:
                    if name in slots_by_name:
                        mismatch(
                            "construction_semantic_constexpr_in_c",
                            f"c.slots[{name}]",
                            "bound non-C constexpr carrier in J.signature only",
                            slots_by_name[name],
                        )
        actual_aliases = boundary.get("aliases") if isinstance(boundary, Mapping) else None
        if actual_aliases != view.get("aliases"):
            mismatch(
                "construction_abi_mismatch",
                "j.boundary.aliases",
                view.get("aliases"),
                actual_aliases,
            )

    return diagnostics


def _candidate_typecheck_observations(
    ref: str | None,
    detail: str,
) -> list[str]:
    """Expose bounded artifact-only TypeCheck evidence without repairing it."""

    observations: list[str] = []
    if ref is not None:
        observations.append(_bounded_candidate_diagnostic_text(ref))
    if detail:
        observations.append(
            _bounded_candidate_diagnostic_text(f"detail: {detail}")
        )
    return observations


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
    bound_constexprs = set(anonymous_view["bound_constexpr_values"])
    slots = {item.name: item for item in artifact.c.slots}
    misplaced_bound_constexprs = sorted(bound_constexprs & set(slots))
    if misplaced_bound_constexprs:
        raise ValueError(
            "Lift moved bound non-C constexpr carrier(s) into pure C: "
            + ", ".join(misplaced_bound_constexprs)
        )
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
    for name in expected_constexprs:
        slot = slots.get(name)
        if name in bound_constexprs:
            # A misplaced bound carrier was rejected above before the generic
            # handle-order mismatch so CandidateCheck can report the real cause.
            assert slot is None
        elif slot is None or slot.kind is not ScheduleSlotKind.CONSTEXPR:
            raise ValueError("Lift omitted a pure-C constexpr slot")
    signature = {item.name: item for item in artifact.j.signature.parameters}
    def carrier_dtype_matches(value: Any, dtype: str | None) -> bool:
        canonical = _canonical_dtype(dtype)
        if isinstance(value, bool):
            return canonical == "i1"
        if isinstance(value, int):
            return canonical in {"int", "index"} or bool(
                canonical and re.fullmatch(r"[iu][1-9][0-9]*", canonical)
            )
        if isinstance(value, float):
            return canonical in {"bf16", "f16", "f32", "f64", "float", "double"}
        return isinstance(value, str) and canonical in {"str", "string"}

    bound_constexpr_shape_coordinates = anonymous_view.get(
        "bound_constexpr_shape_coordinates", {}
    )
    for name, value in anonymous_view["bound_constexpr_values"].items():
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a bound semantic constexpr")
        if isinstance(value, str) and parameter.logical_ref is not None:
            raise ValueError(
                "A string specialization carrier cannot name a J logical value"
            )
        if parameter.logical_ref is None and not carrier_dtype_matches(
            value, parameter.dtype
        ):
            raise ValueError(
                "Harness specialization value type differs from its constexpr "
                "carrier dtype"
            )
    j_tensors = {
        item.name: item
        for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    dimensions: dict[str, int] = {}
    logical_values: dict[str, Any] = {}

    def bind_dimension(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("J symbolic dimension requires an integer valuation")
        previous = dimensions.setdefault(name, value)
        if previous != value:
            raise ValueError(f"J symbolic extent {name!r} is inconsistent")

    coordinate_to_dimension: dict[str, str] = {}
    dimension_to_coordinate: dict[str, str] = {}

    def bind_shape(
        declared: list[int | str],
        actual: list[Any],
        contract: list[int | str] | None,
        *,
        parameter_name: str,
    ) -> None:
        if len(declared) != len(actual):
            raise ValueError("J tensor rank differs from trusted compiler view")
        if contract is not None and len(contract) != len(actual):
            raise ValueError(
                "trusted shape contract rank differs from its concrete valuation"
            )
        contracts = contract if contract is not None else [None] * len(actual)
        for axis, (expected, extent, coordinate) in enumerate(
            zip(declared, actual, contracts, strict=True)
        ):
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise ValueError("trusted tensor extent must be an integer")
            if isinstance(coordinate, str):
                if not isinstance(expected, str):
                    raise ValueError(
                        "Lift specialized source-grounded shape coordinate "
                        f"{coordinate!r} at {parameter_name}[{axis}] to a literal"
                    )
                previous_dimension = coordinate_to_dimension.setdefault(
                    coordinate, expected
                )
                if previous_dimension != expected:
                    raise ValueError(
                        "Lift mapped one source-grounded shape coordinate to "
                        "multiple J dimensions"
                    )
                previous_coordinate = dimension_to_coordinate.setdefault(
                    expected, coordinate
                )
                if previous_coordinate != coordinate:
                    raise ValueError(
                        "Lift collapsed distinct source-grounded shape coordinates "
                        "into one J dimension"
                    )
            elif isinstance(coordinate, int) and expected != coordinate:
                raise ValueError(
                    "Lift parameterized a Harness-declared static shape extent "
                    f"at {parameter_name}[{axis}]"
                )
            if isinstance(expected, int):
                if expected != extent:
                    raise ValueError(
                        "J static tensor extent differs from trusted compiler view"
                    )
            else:
                bind_dimension(expected, extent)

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
        if argument["kind"] == "tensor" and parameter.logical_ref in j_tensors:
            bind_shape(
                j_tensors[parameter.logical_ref].shape,
                argument["shape"],
                argument.get("shape_contract"),
                parameter_name=argument["name"],
            )
        if (
            argument["kind"] == "scalar"
            and isinstance(parameter.logical_ref, str)
            and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
        ):
            bind_dimension(parameter.logical_ref, argument.get("value"))
        if (
            argument["kind"] == "scalar"
            and isinstance(parameter.logical_ref, str)
            and parameter.logical_ref in j_tensors
            and not j_tensors[parameter.logical_ref].shape
        ):
            logical_values[parameter.logical_ref] = argument.get("value")
        expected_mode = argument.get("mode")
        if expected_mode is not None and parameter.mode.value != expected_mode:
            raise ValueError("Lift changed a trusted compiler argument mode")
    for name, value in anonymous_view["bound_constexpr_values"].items():
        parameter = signature.get(name)
        if (
            parameter is not None
            and isinstance(parameter.logical_ref, str)
            and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
        ):
            bind_dimension(parameter.logical_ref, value)
        if (
            parameter is not None
            and isinstance(parameter.logical_ref, str)
            and parameter.logical_ref in j_tensors
            and not j_tensors[parameter.logical_ref].shape
        ):
            logical_values[parameter.logical_ref] = value
        coordinate = bound_constexpr_shape_coordinates.get(name)
        if coordinate is None:
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a bound-constexpr source-grounded shape coordinate"
            )
        if parameter is not None and parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref if parameter is not None else None
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and any(
                relation.value_ref == logical_ref
                and relation.dimension_ref == dimension
                and relation.relation == "eq"
                for relation in artifact.j.value_dimension_relations
            )
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness bound-constexpr-to-shape-coordinate equality"
        )

    expected_shape_coordinates = set(
        anonymous_view.get("shape_coordinates", {})
    )
    if set(coordinate_to_dimension) != expected_shape_coordinates:
        missing = sorted(expected_shape_coordinates - set(coordinate_to_dimension))
        raise ValueError(
            "Lift omitted source-grounded shape coordinate(s) from J tensor shapes: "
            + ", ".join(missing)
        )

    for argument in anonymous_view["arguments"]:
        coordinate = argument.get("shape_coordinate")
        if not isinstance(coordinate, str):
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a scalar-bound source-grounded shape coordinate"
            )
        parameter = signature[argument["name"]]
        if parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and any(
                relation.value_ref == logical_ref
                and relation.dimension_ref == dimension
                and relation.relation == "eq"
                for relation in artifact.j.value_dimension_relations
            )
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness scalar-to-shape-coordinate equality"
        )

    try:
        validate_dimension_instantiation(artifact.j, dimensions)
        validate_value_dimension_instantiation(
            artifact.j, dimensions, logical_values
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
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
    # One logical inout tensor is deliberately declared in both collections.
    # Tensor ids therefore advance on first declaration, not on collection
    # entries.  SemanticArchive already rejects duplicate tensors within one
    # collection and non-identical input/output contracts.
    actual_tensors = list(dict.fromkeys(item.name for item in tensors))
    expected_tensors = [f"t{index}" for index in range(len(actual_tensors))]
    if actual_tensors != expected_tensors:
        raise ValueError(
            "J tensors must use their own first-declaration-order handles "
            "(an inout output repeats its input handle) independent "
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
    visible_strings: list[tuple[str, str]] = []
    literal_fields = {
        "dtype", "fn", "frontend", "kind", "mechanism", "mode", "op",
        "placement", "producer", "role", "subscripts", "value_type",
    }

    def collect_strings(
        value: Any,
        *,
        path: str,
        field_name: str | None = None,
    ) -> None:
        if isinstance(value, str):
            if field_name not in literal_fields:
                visible_strings.append((path, value))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect_strings(
                    item,
                    path=f"{path}.{key}" if path else str(key),
                    field_name=key,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect_strings(
                    item,
                    path=f"{path}[{index}]",
                    field_name=field_name,
                )

    collect_strings(visible, path="")
    for source_name, opaque in binding.source_to_opaque:
        if source_name == opaque:
            continue
        # These are protocol-public, declaration-ordered handles/binders.  A
        # private Source local may coincidentally have the same spelling, but
        # seeing that spelling in canonical J/C/pi carries no Source identity.
        if re.fullmatch(
            r"(?:scrc_[jk][0-9]+|[abcdeiknpqrtuvx][0-9]+)", source_name
        ):
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])"
        for path, item in visible_strings:
            if re.search(pattern, item):
                raise SourceIdentifierPrivacyError(path)


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
    "SourceIdentifierPrivacyError",
    "SCRCReliftRecord", "anonymize_source", "build_scrc_lift_instructions",
    "check_scrc_lift_candidate", "decode_scrc_lift_artifact", "extract_scrc_artifact", "prepare_scrc_lift", "scrc_candidate_check_context", "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
]
