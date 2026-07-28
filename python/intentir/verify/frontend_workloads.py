"""Prepared SCRC workloads for the external TileLang and CUDA Dataset.

This module is the source-aware Harness boundary for non-Triton frontends.  It
projects a real Dataset source into Lift material, preserves the original
frontend object for Gate1/Gate2, and binds one concrete finite invocation.  It
does not translate either frontend into J; Lift remains the only Source -> J
producer and Replay remains source-blind.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import operator
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.verify.frontend_dataset import (
    FrontendDatasetKernel,
    load_frontend_dataset,
    resolve_cuda_include_dirs,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaKernelSource,
    FrontendParameter,
)
from intentir.verify.gate1.model import (
    Frontend,
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.profile import observer_profile_for
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangKernelSource,
)
from intentir.verify.gate2.native import (
    NativeArgument,
    NativeArgumentRole,
    NativeExecutionSpec,
)
from intentir.verify.gate2.orchestrate import Gate2Case, Gate2Suite
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringSpec
from intentir.verify.scrc_runner import (
    HarnessBindingInstance,
    SCRCRunRequest,
    SourceProgram,
)


FRONTEND_DATASET_HARNESS_PROTOCOL = "intentir.scrc.frontend-dataset-harness.v1"
FRONTEND_DATASET_SOURCE_PROTOCOL = "intentir.scrc.frontend-source-projection.v1"
FRONTEND_DATASET_PROJECTION_PROTOCOL = (
    "intentir.scrc.frontend-dataset-harness-projection.v1"
)
_DEFAULT_DATASET_DIMENSION = 4
_DEFAULT_DATASET_BLOCK_1D = (16, 1, 1)
_DEFAULT_DATASET_BLOCK_2D = (16, 16, 1)


class FrontendDatasetCaseUnavailableError(ValueError):
    """The catalog entry is valid but needs an unsupported explicit sidecar."""


@dataclass(frozen=True)
class FrontendDatasetHarness:
    """One explicit finite binding for a Dataset kernel."""

    case_id: str
    frontend: str
    dimensions: tuple[tuple[str, int], ...]
    scalar_values: tuple[tuple[str, bool | int | float], ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    grid_contract: tuple[str, str, str]
    seed: int
    factory_kwargs: tuple[tuple[str, bool | int | float | str], ...] = ()
    cuda_arch: str = "sm_120"
    source_sha256: str = ""
    descriptor_sha256: str = ""
    protocol: str = FRONTEND_DATASET_HARNESS_PROTOCOL

    def __post_init__(self) -> None:
        if self.frontend not in {"cuda", "tilelang"}:
            raise ValueError("Dataset Harness frontend must be cuda or tilelang")
        if not self.case_id.startswith(f"{self.frontend}:"):
            raise ValueError("Dataset Harness case_id and frontend differ")
        dimensions = tuple((str(name), int(value)) for name, value in self.dimensions)
        if (
            len({name for name, _value in dimensions}) != len(dimensions)
            or any(not name.isidentifier() or value <= 0 for name, value in dimensions)
        ):
            raise ValueError("Dataset dimensions require unique names and positive values")
        scalar_values = tuple(self.scalar_values)
        if (
            len({name for name, _value in scalar_values}) != len(scalar_values)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name, _value in scalar_values
            )
        ):
            raise ValueError("Dataset scalar bindings must be unique")
        for name, value in scalar_values:
            if not isinstance(value, (bool, int, float)):
                raise TypeError(f"Dataset scalar {name!r} is not a numeric scalar")
            if isinstance(value, float) and (
                value != value or value in {float("inf"), float("-inf")}
            ):
                raise ValueError(f"Dataset scalar {name!r} is non-finite")
        for label, values in (("grid", self.grid), ("block", self.block)):
            normalized = tuple(int(value) for value in values)
            if len(normalized) != 3 or any(value <= 0 for value in normalized):
                raise ValueError(f"Dataset {label} must contain three positive values")
            object.__setattr__(self, label, normalized)
        contract = tuple(str(value) for value in self.grid_contract)
        if len(contract) != 3 or any(not value for value in contract):
            raise ValueError("Dataset grid contract must describe all three axes")
        kwargs = tuple(self.factory_kwargs)
        if (
            len({name for name, _value in kwargs}) != len(kwargs)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name, _value in kwargs
            )
        ):
            raise ValueError("TileLang factory kwargs must be unique")
        for name, value in kwargs:
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"TileLang factory kwarg {name!r} is not a JSON scalar")
            if isinstance(value, float) and (
                value != value or value in {float("inf"), float("-inf")}
            ):
                raise ValueError(f"TileLang factory kwarg {name!r} is non-finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Dataset Harness seed must be non-negative")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.source_sha256):
            raise ValueError("Dataset Harness requires a pinned source SHA-256")
        if self.descriptor_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.descriptor_sha256
        ):
            raise ValueError("Dataset Harness descriptor SHA-256 is malformed")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "scalar_values", scalar_values)
        object.__setattr__(self, "grid_contract", contract)
        object.__setattr__(self, "factory_kwargs", kwargs)

    @property
    def dimension_map(self) -> dict[str, int]:
        return dict(self.dimensions)

    @property
    def scalar_map(self) -> dict[str, bool | int | float]:
        return dict(self.scalar_values)

    @property
    def factory_kwarg_map(self) -> dict[str, bool | int | float | str]:
        return dict(self.factory_kwargs)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "case_id": self.case_id,
            "frontend": self.frontend,
            "dimensions": self.dimension_map,
            "scalar_values": self.scalar_map,
            "grid": list(self.grid),
            "block": list(self.block),
            "grid_contract": list(self.grid_contract),
            "seed": self.seed,
            "factory_kwargs": self.factory_kwarg_map,
            "cuda_arch": self.cuda_arch,
            "source_sha256": self.source_sha256,
            "descriptor_sha256": self.descriptor_sha256,
        }


@dataclass(frozen=True)
class FrontendDatasetHarnessDiagnostic:
    """A closed reason why one catalog entry is not projected as a Harness.

    Diagnostics intentionally contain only catalog identity and a bounded
    mechanical reason.  They never contain source bytes, generated J, or a
    semantic guess about the kernel.
    """

    case_id: str
    frontend: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.frontend not in {"cuda", "tilelang", "native"}:
            raise ValueError("Dataset diagnostic frontend must be native")
        if self.status not in {"ready", "case_unavailable", "harness_invalid"}:
            raise ValueError("unsupported Dataset Harness diagnostic status")
        if not self.reason or len(self.reason) > 2048:
            raise ValueError("Dataset diagnostic reason must be bounded")

    def to_json(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "frontend": self.frontend,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FrontendDatasetHarnessProjection:
    """Deterministic ready/unavailable view of the external Dataset."""

    harnesses: Mapping[str, FrontendDatasetHarness]
    diagnostics: tuple[FrontendDatasetHarnessDiagnostic, ...]
    protocol: str = FRONTEND_DATASET_PROJECTION_PROTOCOL

    def __post_init__(self) -> None:
        harnesses = dict(self.harnesses)
        diagnostics = tuple(self.diagnostics)
        if set(harnesses) & {item.case_id for item in diagnostics}:
            raise ValueError("a Dataset case cannot be both ready and diagnosed")
        object.__setattr__(self, "harnesses", harnesses)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "ready": [
                harness.to_json()
                for _case_id, harness in sorted(self.harnesses.items())
            ],
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class DatasetRuntimeParameter:
    source_name: str
    semantic_name: str
    kind: str
    dtype: str
    rank: int
    mode: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None
    shape_contract: tuple[int | str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_name.isidentifier() or not self.semantic_name.isidentifier():
            raise ValueError("Dataset parameter names must be identifiers")
        if self.kind not in {"pointer", "scalar", "opaque"}:
            raise ValueError("unsupported Dataset parameter kind")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("Dataset parameter dtype must be non-empty")
        shape = tuple(int(value) for value in self.shape)
        shape_contract: tuple[int | str, ...] = tuple(self.shape_contract)
        if self.kind == "pointer":
            if self.rank != len(shape) or not shape or any(value <= 0 for value in shape):
                raise ValueError("Dataset pointer shape must match its positive rank")
            if len(shape_contract) != self.rank:
                raise ValueError(
                    "Dataset pointer source shape contract must match its rank"
                )
            for extent in shape_contract:
                if isinstance(extent, bool):
                    raise TypeError("Dataset source shape contract cannot be boolean")
                if isinstance(extent, int):
                    if extent <= 0:
                        raise ValueError(
                            "Dataset static source shape extents must be positive"
                        )
                elif not isinstance(extent, str) or not extent.isidentifier():
                    raise ValueError(
                        "Dataset symbolic source shape extents must be identifiers"
                    )
            if self.mode not in {"input", "output", "inout"}:
                raise ValueError("Dataset pointer requires an effect mode")
            if self.value is not None:
                raise ValueError("Dataset pointer cannot carry a scalar value")
        elif self.kind == "scalar":
            if (
                self.rank != 0
                or shape
                or shape_contract
                or self.mode != "value"
                or self.value is None
            ):
                raise ValueError("Dataset scalar requires one fixed runtime value")
        elif (
            self.rank != 0
            or shape
            or shape_contract
            or self.mode != "value"
            or self.value is not None
        ):
            raise ValueError("Dataset opaque exposes only its public type")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "shape_contract", shape_contract)

    def frontend_parameter(self) -> FrontendParameter:
        return FrontendParameter(self.kind, self.dtype, self.rank, self.mode)

    def compiler_argument(
        self,
        shape_coordinates: Mapping[str, str],
    ) -> SCRCCompilerArgument:
        if self.kind == "pointer":
            return SCRCCompilerArgument(
                self.source_name,
                "tensor",
                self.dtype,
                self.shape,
                mode=self.mode,
                shape_contract=tuple(
                    shape_coordinates[extent]
                    if isinstance(extent, str)
                    else extent
                    for extent in self.shape_contract
                ),
            )
        if self.kind == "scalar":
            return SCRCCompilerArgument(
                self.source_name,
                "scalar",
                self.dtype,
                value=self.value,
                mode="value",
                shape_coordinate=shape_coordinates.get(self.semantic_name),
            )
        return SCRCCompilerArgument(
            self.source_name, "opaque", self.dtype, mode="value"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "semantic_name": self.semantic_name,
            "kind": self.kind,
            "dtype": self.dtype,
            "rank": self.rank,
            "mode": self.mode,
            "shape": list(self.shape),
            "shape_contract": list(self.shape_contract),
            "value": self.value,
        }


@dataclass(frozen=True)
class DatasetSourceProjection:
    code: str
    function: str
    parameters: tuple[str, ...]
    frontend: str
    source_kind: str
    origin_location: str
    protocol: str = FRONTEND_DATASET_SOURCE_PROTOCOL
    constexpr_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.function.isidentifier():
            raise ValueError("Dataset source projection requires code and one function")
        if len(self.parameters) != len(set(self.parameters)) or any(
            not name.isidentifier() for name in self.parameters
        ):
            raise ValueError("Dataset source projection parameters must be unique")

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "function": self.function,
            "parameters": list(self.parameters),
            "constexpr_parameters": list(self.constexpr_parameters),
            "frontend": self.frontend,
            "source_kind": self.source_kind,
            "origin_location": self.origin_location,
        }


@dataclass(frozen=True)
class DatasetWorkloadEntry:
    case_id: str
    frontend: str
    symbol: str
    location: str
    observer_profile: str
    q_policy: str


@dataclass(frozen=True)
class PreparedFrontendDatasetWorkload:
    adapter: FrontendDatasetHarness
    entry: DatasetWorkloadEntry
    source_location: str
    source_slice: DatasetSourceProjection
    compiler_view: SCRCCompilerView
    request: SCRCRunRequest
    source_to_opaque: Mapping[str, str]
    runtime_parameters: tuple[DatasetRuntimeParameter, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


def _safe_integer_expression(expression: Any, values: Mapping[str, int]) -> int:
    if isinstance(expression, bool):
        raise ValueError("launch expressions cannot be boolean")
    if isinstance(expression, int):
        result = expression
    else:
        text = str(expression)
        if len(text) > 256:
            raise ValueError("launch expression exceeds the 256-character limit")
        tree = ast.parse(text, mode="eval")
        if sum(1 for _node in ast.walk(tree)) > 128:
            raise ValueError("launch expression is too complex")
        binary = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }

        def evaluate(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                if abs(node.value) > 2**31:
                    raise ValueError("launch integer exceeds the 32-bit limit")
                return int(node.value)
            if isinstance(node, ast.Name) and node.id in values:
                return int(values[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in binary:
                value = int(
                    binary[type(node.op)](
                        evaluate(node.left), evaluate(node.right)
                    )
                )
                if abs(value) > 2**31:
                    raise ValueError("launch result exceeds the 32-bit limit")
                return value
            raise ValueError(f"unsupported launch expression {expression!r}")

        result = evaluate(tree)
    if result <= 0:
        raise ValueError(f"launch expression must be positive, got {result}")
    return result


def _launch_from_metadata(
    metadata: Mapping[str, Any], dimensions: Mapping[str, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[str, str, str]]:
    launch = metadata.get("launch")
    if not isinstance(launch, Mapping):
        raise ValueError("generated Dataset kernel lacks launch metadata")
    grid_expressions = tuple(str(launch.get(f"blockIdx.{axis}", "1")) for axis in "xyz")
    block_expressions = tuple(str(launch.get(f"threadIdx.{axis}", "1")) for axis in "xyz")
    grid = tuple(_safe_integer_expression(value, dimensions) for value in grid_expressions)
    block = tuple(_safe_integer_expression(value, dimensions) for value in block_expressions)
    return grid, block, grid_expressions  # type: ignore[return-value]


def _resolve_shape(raw: Any, dimensions: Mapping[str, int]) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Dataset pointer parameter requires shape metadata")
    return tuple(_safe_integer_expression(value, dimensions) for value in raw)


def _source_shape_contract(raw: Any) -> tuple[int | str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Dataset pointer parameter requires shape metadata")
    result: list[int | str] = []
    for extent in raw:
        if isinstance(extent, bool):
            raise TypeError("Dataset source shape contract cannot contain booleans")
        if isinstance(extent, int):
            if extent <= 0:
                raise ValueError("Dataset static source shape extent must be positive")
            result.append(extent)
            continue
        if not isinstance(extent, str) or not extent.isidentifier():
            raise ValueError(
                "Dataset source shape contract supports integers and symbolic identifiers"
            )
        result.append(extent)
    return tuple(result)


def _parameter_alias(name: str) -> str:
    """Return a conservative ABI spelling alias for source/metadata matching.

    TileLang's public PrimFunc uses carriers such as ``x_handle`` while the
    exported CUDA ABI records ``x`` (and a few exports use ``x_ptr``).  Only
    these mechanically declared carrier suffixes are removed.  No operator,
    tensor role, J reference, or semantic relation is inferred here.
    """

    pieces = name.lower().split("_")
    while pieces and pieces[-1] in {"handle", "ptr"}:
        pieces.pop()
    return "_".join(pieces)


def _ordered_abi_rows(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Validate and return the public ABI in an unambiguous position order."""

    abi = metadata.get("cuda_abi")
    raw_rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Dataset kernel lacks structured CUDA ABI metadata")
    positioned: list[tuple[int, Mapping[str, Any]]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Dataset ABI parameters must all be objects")
        position = row.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("Dataset ABI parameter positions must be integers")
        positioned.append((position, row))
    positions = sorted(position for position, _row in positioned)
    if positions != list(range(len(positioned))):
        raise ValueError("Dataset ABI parameter positions must be unique and contiguous")
    return tuple(row for _position, row in sorted(positioned, key=lambda item: item[0]))


def _metadata_rows_in_source_order(
    source_names: tuple[str, ...], rows: list[Mapping[str, Any]]
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Bijection-match materialized carriers, then retain source order."""

    by_alias: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        metadata_name = str(row.get("name") or row.get("semantic_name") or "")
        if not metadata_name:
            raise ValueError("Dataset ABI parameter lacks a public name")
        by_alias.setdefault(_parameter_alias(metadata_name), []).append(row)
    ordered: list[tuple[str, Mapping[str, Any]]] = []
    used: set[str] = set()
    for source_name in source_names:
        candidates = [
            row
            for row in by_alias.get(_parameter_alias(source_name), ())
            if str(row.get("name") or row.get("semantic_name") or "") not in used
        ]
        if len(candidates) != 1:
            raise ValueError(
                "materialized TileLang ABI cannot be uniquely aligned with "
                f"source parameter {source_name!r}"
            )
        row = candidates[0]
        metadata_name = str(row.get("name") or row.get("semantic_name") or "")
        used.add(metadata_name)
        ordered.append((source_name, row))
    if len(used) != len(source_names):
        raise ValueError("materialized TileLang ABI contains an unlisted carrier")
    return tuple(ordered)


def _metadata_parameters(
    metadata: Mapping[str, Any],
    *,
    dimensions: Mapping[str, int],
    scalars: Mapping[str, bool | int | float],
    source_names: tuple[str, ...] | None = None,
    include_runtime_scalars: bool,
) -> tuple[DatasetRuntimeParameter, ...]:
    rows = list(_ordered_abi_rows(metadata))
    if not include_runtime_scalars:
        rows = [item for item in rows if item.get("kind") == "pointer"]
    if source_names is not None:
        source_rows = _metadata_rows_in_source_order(source_names, rows)
        if len(source_rows) != len(rows):
            raise ValueError("materialized TileLang ABI differs from generated metadata")
    else:
        source_rows = tuple((str(row.get("name") or ""), row) for row in rows)
    result: list[DatasetRuntimeParameter] = []
    for source_name, row in source_rows:
        semantic_name = str(row.get("semantic_name") or row.get("name") or "")
        kind = str(row.get("kind") or "")
        value = None
        shape: tuple[int, ...] = ()
        shape_contract: tuple[int | str, ...] = ()
        if kind == "pointer":
            shape_contract = _source_shape_contract(row.get("shape"))
            shape = _resolve_shape(shape_contract, dimensions)
        elif kind == "scalar":
            if semantic_name in scalars:
                value = scalars[semantic_name]
            elif semantic_name in dimensions:
                value = dimensions[semantic_name]
            else:
                raise ValueError(f"Dataset scalar {semantic_name!r} lacks a Harness value")
        result.append(
            DatasetRuntimeParameter(
                source_name=source_name,
                semantic_name=semantic_name,
                kind=kind,
                dtype=str(row.get("dtype") or ""),
                rank=int(row.get("rank", 0)),
                mode=str(row.get("mode") or "value"),
                shape=shape,
                value=value,
                shape_contract=shape_contract,
            )
        )
    return tuple(result)


def _shape_coordinate_mapping(
    parameters: tuple[DatasetRuntimeParameter, ...],
) -> dict[str, str]:
    """Assign source-grounded shape identities without exposing source names.

    Coordinates follow first tensor-declaration order and are independent of
    their current valuations.  Equal source symbols share one coordinate;
    distinct symbols remain distinct even when their finite values coincide.
    """

    coordinates: dict[str, str] = {}
    for parameter in parameters:
        if parameter.kind != "pointer":
            continue
        for extent in parameter.shape_contract:
            if isinstance(extent, str) and extent not in coordinates:
                coordinates[extent] = f"h{len(coordinates)}"
    return coordinates


def _grid_shape_coordinates(
    grid_contract: tuple[str, str, str],
    shape_coordinates: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Bind only axes that are exactly one authored shape coordinate.

    Matching by concrete extent would conflate distinct symbolic dimensions;
    compound launch expressions therefore remain unbound expressions.
    """

    return tuple(shape_coordinates.get(expression) for expression in grid_contract)  # type: ignore[return-value]


def _materialize_tilelang_projection(
    kernel: FrontendDatasetKernel,
    factory_kwargs: Mapping[str, bool | int | float | str],
) -> DatasetSourceProjection:
    # Python's process-randomized ``hash`` made this ephemeral import name
    # change on every run without carrying any proof meaning.  A short stable
    # digest keeps module isolation while making preparation/replay receipts
    # reproducible and avoiding a needless per-run hash churn.
    module_digest = hashlib.sha256(
        f"{kernel.path.resolve()}:{kernel.symbol}".encode("utf-8")
    ).hexdigest()[:16]
    module_name = f"_intentir_dataset_lift_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, kernel.path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TileLang Dataset module {kernel.path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, kernel.symbol)
        prim_func = factory(**dict(factory_kwargs))
        code = str(prim_func.script())
    finally:
        sys.modules.pop(module_name, None)
    marker = "# from tvm.script import tir as T"
    if marker not in code:
        raise ValueError("TileLang PrimFunc script lacks its public TIR import marker")
    code = code.replace(marker, "import tilelang.language as T", 1)
    tree = ast.parse(code)
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("materialized TileLang source must contain exactly one PrimFunc")
    function = functions[0]
    parameters = tuple(argument.arg for argument in function.args.args)
    return DatasetSourceProjection(
        code if code.endswith("\n") else code + "\n",
        function.name,
        parameters,
        "tilelang",
        "materialized_tir",
        kernel.location,
    )


def _generated_companion(
    tilelang: FrontendDatasetKernel,
    kernels: tuple[FrontendDatasetKernel, ...],
) -> FrontendDatasetKernel:
    matches = [
        item
        for item in kernels
        if item.frontend == "cuda"
        and item.metadata.get("origin") == "tilelang_export"
        and item.metadata.get("origin_case_id") == tilelang.case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"TileLang Dataset case {tilelang.case_id!r} requires exactly one CUDA ABI companion"
        )
    return matches[0]


def _literal_factory_kwargs(
    kernel: FrontendDatasetKernel,
) -> tuple[tuple[str, bool | int | float | str], ...]:
    """Read only literal factory defaults; never import or execute Dataset code."""

    try:
        tree = ast.parse(kernel.path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"TileLang factory source is not statically readable: {exc}") from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == kernel.symbol
    ]
    if len(functions) != 1:
        raise ValueError("TileLang Dataset factory definition is not unique")
    function = functions[0]
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = [
        *([None] * (len(positional) - len(function.args.defaults))),
        *function.args.defaults,
    ]
    keyword_only = list(function.args.kwonlyargs)
    keyword_defaults = list(function.args.kw_defaults)
    arguments = [*zip(positional, positional_defaults), *zip(keyword_only, keyword_defaults)]
    result: list[tuple[str, bool | int | float | str]] = []
    for argument, default in arguments:
        if default is None:
            raise ValueError(
                f"TileLang factory argument {argument.arg!r} lacks a literal default"
            )
        try:
            value = ast.literal_eval(default)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(
                f"TileLang factory default {argument.arg!r} is not a literal"
            ) from exc
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(
                f"TileLang factory default {argument.arg!r} is not a JSON scalar"
            )
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise ValueError(f"TileLang factory default {argument.arg!r} is non-finite")
        result.append((argument.arg, value))
    return tuple(result)


def _snapshot_canonical_shapes(dataset_root: Path) -> dict[str, dict[str, int]]:
    """Load only concrete invocation extents from optional export snapshots."""

    result: dict[str, dict[str, int]] = {}
    snapshot_root = dataset_root / "cuda" / "ops" / "snapshots"
    for path in sorted(snapshot_root.glob("*.cuda_snapshot.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(loaded, Mapping):
            continue
        name = loaded.get("name")
        canonical = loaded.get("canonical_shapes")
        if not isinstance(name, str) or not isinstance(canonical, Mapping):
            continue
        values: dict[str, int] = {}
        for key, value in canonical.items():
            if (
                isinstance(key, str)
                and key.isidentifier()
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                values[key] = value
        if values:
            result[name] = values
    return result


def _ordered_dimension_symbols(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect authored symbolic extents without interpreting their meaning."""

    seen: set[str] = set()
    ordered: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.isidentifier() and value not in seen:
            seen.add(value)
            ordered.append(value)

    rows = _ordered_abi_rows(metadata)
    for row in rows:
        for extent in row.get("shape") or ():
            add(extent)
        if row.get("kind") == "scalar":
            add(row.get("semantic_name") or row.get("name"))
    launch = metadata.get("launch")
    if not isinstance(launch, Mapping):
        raise ValueError("generated Dataset kernel lacks launch metadata")
    for axis in ("x", "y", "z"):
        for stage in ("blockIdx", "threadIdx"):
            expression = launch.get(f"{stage}.{axis}", "1")
            if isinstance(expression, bool):
                raise ValueError("launch expressions cannot be boolean")
            if isinstance(expression, int):
                continue
            try:
                tree = ast.parse(str(expression), mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"launch expression {expression!r} is not a complete expression"
                ) from exc
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    add(node.id)
    return tuple(ordered)


def _dataset_descriptor_values(
    metadata: Mapping[str, Any],
    *,
    canonical_shapes: Mapping[str, int],
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[str, str, str],
    tuple[tuple[str, bool | int | float], ...],
]:
    symbols = _ordered_dimension_symbols(metadata)
    dimensions = tuple(
        (
            symbol,
            int(canonical_shapes.get(symbol, _DEFAULT_DATASET_DIMENSION)),
        )
        for symbol in symbols
    )
    if any(value <= 0 for _name, value in dimensions):
        raise ValueError("Dataset canonical shape values must be positive")
    dimension_map = dict(dimensions)
    grid, block, contract = _launch_from_metadata(metadata, dimension_map)
    scalar_values: list[tuple[str, bool | int | float]] = []
    rows = _ordered_abi_rows(metadata)
    for row in rows:
        if row.get("kind") != "scalar":
            continue
        name = str(row.get("semantic_name") or row.get("name") or "")
        if not name:
            raise ValueError("Dataset scalar parameter lacks a public name")
        scalar_values.append((name, dimension_map.get(name, _DEFAULT_DATASET_DIMENSION)))
    return dimensions, grid, block, contract, tuple(scalar_values)


def _literal_handwritten_io(kernel: FrontendDatasetKernel) -> Mapping[str, Any]:
    """Read the adjacent CUDA IO dictionary with ``ast.literal_eval`` only."""

    metadata_path = kernel.path.with_suffix(".py")
    try:
        tree = ast.parse(metadata_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(
            f"handwritten CUDA IO metadata is not statically readable: {exc}"
        ) from exc
    candidates: list[Mapping[str, Any]] = []
    for node in tree.body:
        value_node: ast.expr | None = None
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Assign):
            value_node = node.value
            names = tuple(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value_node = node.value
            names = (node.target.id,)
        if value_node is None or not any(name.endswith("_io") for name in names):
            continue
        try:
            loaded = ast.literal_eval(value_node)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(loaded, Mapping) and loaded.get("kernel_name") == kernel.symbol:
            candidates.append(loaded)
    if len(candidates) != 1:
        raise ValueError(
            "handwritten CUDA kernel requires exactly one literal adjacent IO descriptor"
        )
    return candidates[0]


def _matching_parenthesis(code: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(code)):
        character = code[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("CUDA kernel declaration has unbalanced parentheses")


def _split_cuda_parameters(parameters: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(parameters):
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail:
        result.append(tail)
    if not result or any(not item for item in result):
        raise ValueError("CUDA kernel parameter list is empty or malformed")
    return tuple(result)


def _matching_brace(code: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(code)):
        character = code[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("CUDA kernel definition has unbalanced braces")


def _cuda_definition(
    code: str, symbol: str
) -> tuple[tuple[str, ...], str]:
    for match in re.finditer(rf"\b{re.escape(symbol)}\s*\(", code):
        header = code[max(0, match.start() - 512) : match.start()]
        if "__global__" not in header:
            continue
        opening = code.find("(", match.start())
        closing = _matching_parenthesis(code, opening)
        suffix = code[closing + 1 :].lstrip()
        if not suffix.startswith("{"):
            continue
        body_opening = code.find("{", closing + 1)
        body_closing = _matching_brace(code, body_opening)
        return (
            _split_cuda_parameters(code[opening + 1 : closing]),
            code[body_opening + 1 : body_closing],
        )
    raise ValueError("handwritten CUDA kernel definition is unavailable")


def _cuda_declaration_name(declaration: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration)
    if match is None:
        raise ValueError(f"unsupported CUDA parameter declaration {declaration!r}")
    return match.group(1)


def _normalize_dataset_dtype(dtype: Any) -> str:
    aliases = {
        "bool": "i1",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
    }
    value = aliases.get(str(dtype), str(dtype))
    if not re.fullmatch(r"(?:bf16|f16|f32|f64|i1|i8|i16|i32|i64|u8)", value):
        raise ValueError(f"unsupported Dataset ABI dtype {dtype!r}")
    return value


def _cuda_pointer_mode(name: str, declaration: str, body: str) -> str:
    """Conservatively project direct subscript reads/writes for one pointer.

    The handwritten corpus has no explicit effect-mode field.  We accept only
    direct ``p[index]`` accesses and reject bare pointer escapes/calls.  Thus a
    mutable read-only pointer remains an input instead of being mislabeled as
    output merely because its declaration omitted ``const``.
    """

    if "*" not in declaration:
        raise ValueError(f"CUDA tensor parameter {name!r} is not a pointer")
    access_pattern = re.compile(rf"\b{re.escape(name)}\s*\[[^\]]+\]")
    accesses = list(access_pattern.finditer(body))
    remainder = access_pattern.sub("", body)
    if re.search(rf"\b{re.escape(name)}\b", remainder):
        raise ValueError(
            f"CUDA pointer {name!r} escapes direct subscript effect analysis"
        )
    read = False
    write = False
    for access in accesses:
        suffix = body[access.end() :].lstrip()
        assignment = re.match(r"(?P<op>(?:<<|>>|[+\-*/%&|^])?=)(?!=)", suffix)
        if assignment is None:
            read = True
            continue
        write = True
        if assignment.group("op") != "=":
            read = True
    if re.search(r"\bconst\b", declaration):
        if write:
            raise ValueError(f"const CUDA pointer {name!r} is written")
        if not read:
            raise ValueError(f"const CUDA pointer {name!r} has no direct read")
        return "input"
    if read and write:
        return "inout"
    if read:
        return "input"
    if write:
        return "output"
    raise ValueError(f"CUDA pointer {name!r} has no direct readable effect")


def _handwritten_runtime_metadata(
    kernel: FrontendDatasetKernel,
) -> Mapping[str, Any]:
    """Project source ABI plus literal IO metadata into a finite call descriptor.

    Handwritten CUDA sources do not have a fixed launch in Dataset.  The generic
    bounded policy uses one block and dimensions of four.  Rank-1 buffers get a
    16-thread block; rank-2-or-higher buffers get a 16x16 block.  This is a
    Harness invocation choice, not a kernel-family or semantic inference.
    """

    io = _literal_handwritten_io(kernel)
    arg_names = io.get("arg_names")
    tensors = io.get("tensors")
    scalars = io.get("scalars")
    if (
        not isinstance(arg_names, list)
        or not all(isinstance(name, str) and name.isidentifier() for name in arg_names)
        or len(arg_names) != len(set(arg_names))
        or not isinstance(tensors, Mapping)
        or not isinstance(scalars, Mapping)
    ):
        raise ValueError("handwritten CUDA IO descriptor is malformed")
    code = kernel.path.read_text(encoding="utf-8")
    declarations, body = _cuda_definition(code, kernel.symbol)
    declaration_names = tuple(_cuda_declaration_name(item) for item in declarations)
    if tuple(arg_names) != declaration_names:
        raise ValueError("handwritten CUDA IO descriptor differs from source ABI order")
    rows: list[dict[str, Any]] = []
    pointer_ranks: list[int] = []
    for position, (name, declaration) in enumerate(
        zip(arg_names, declarations, strict=True)
    ):
        if name in tensors:
            tensor = tensors[name]
            if not isinstance(tensor, Mapping):
                raise ValueError("handwritten CUDA tensor metadata must be an object")
            shape = tensor.get("shape")
            rank = tensor.get("rank")
            if (
                not isinstance(shape, list)
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank != len(shape)
                or rank <= 0
            ):
                raise ValueError("handwritten CUDA tensor rank/shape metadata differs")
            source_shape = list(_source_shape_contract(shape))
            pointer_ranks.append(rank)
            rows.append(
                {
                    "declaration": declaration,
                    "dtype": _normalize_dataset_dtype(tensor.get("dtype")),
                    "kind": "pointer",
                    "mode": _cuda_pointer_mode(name, declaration, body),
                    "name": name,
                    "position": position,
                    "rank": rank,
                    "semantic_name": name,
                    "shape": source_shape,
                }
            )
            continue
        if name in scalars:
            if "*" in declaration:
                raise ValueError(f"CUDA scalar parameter {name!r} is a pointer")
            rows.append(
                {
                    "declaration": declaration,
                    "dtype": _normalize_dataset_dtype(scalars[name]),
                    "kind": "scalar",
                    "mode": "value",
                    "name": name,
                    "position": position,
                    "rank": 0,
                    "semantic_name": name,
                }
            )
            continue
        raise ValueError(f"handwritten CUDA parameter {name!r} lacks IO metadata")
    if set(tensors) | set(scalars) != set(arg_names):
        raise ValueError("handwritten CUDA IO descriptor contains extra parameters")
    block = (
        _DEFAULT_DATASET_BLOCK_2D
        if pointer_ranks and max(pointer_ranks) >= 2
        else _DEFAULT_DATASET_BLOCK_1D
    )
    return {
        "origin": "handwritten",
        "harness_protocol": FRONTEND_DATASET_PROJECTION_PROTOCOL,
        "cuda_abi": {"parameters": rows},
        "launch": {
            "blockIdx.x": "1",
            "blockIdx.y": "1",
            "blockIdx.z": "1",
            "threadIdx.x": str(block[0]),
            "threadIdx.y": str(block[1]),
            "threadIdx.z": str(block[2]),
        },
        "target": {"name": "cuda"},
    }


def _runtime_metadata(kernel: FrontendDatasetKernel) -> Mapping[str, Any]:
    if kernel.frontend != "cuda":
        raise ValueError("runtime CUDA metadata requested for a non-CUDA kernel")
    if kernel.metadata.get("origin") == "handwritten":
        return _handwritten_runtime_metadata(kernel)
    return kernel.metadata


def prepare_frontend_dataset_workload(
    adapter: FrontendDatasetHarness,
    *,
    dataset_root: str | Path | None = None,
) -> PreparedFrontendDatasetWorkload:
    catalog = load_frontend_dataset(dataset_root)
    validation = catalog.validate()
    if not validation.passed:
        raise ValueError(
            "multi-frontend Dataset validation failed: "
            + "; ".join(validation.errors)
        )
    kernel = catalog.by_id.get(adapter.case_id)
    if kernel is None:
        raise ValueError(f"Dataset case {adapter.case_id!r} is unavailable")
    if kernel.frontend != adapter.frontend:
        raise ValueError("Dataset catalog frontend differs from its Harness")
    actual_sha256 = hashlib.sha256(kernel.path.read_bytes()).hexdigest()
    if actual_sha256 != adapter.source_sha256.lower():
        raise ValueError(
            f"Dataset source digest differs for {adapter.case_id}: "
            f"expected {adapter.source_sha256}, got {actual_sha256}"
        )
    dimensions = adapter.dimension_map
    if adapter.frontend == "tilelang":
        metadata_kernel = _generated_companion(kernel, catalog.kernels)
        projection = _materialize_tilelang_projection(
            kernel, adapter.factory_kwarg_map
        )
        parameters = _metadata_parameters(
            metadata_kernel.metadata,
            dimensions=dimensions,
            scalars=adapter.scalar_map,
            source_names=projection.parameters,
            include_runtime_scalars=False,
        )
        gate_source: Any = TileLangKernelSource.from_factory_file(
            kernel.path,
            kernel.symbol,
            factory_kwargs=adapter.factory_kwarg_map,
        )
        gate_case: Any = TileLangCompileCase(
            tuple(item.frontend_parameter() for item in parameters),
            adapter.grid,
            adapter.block,
            {},
        )
        metadata = metadata_kernel.metadata
    else:
        code = kernel.path.read_text(encoding="utf-8")
        metadata = _runtime_metadata(kernel)
        parameters = _metadata_parameters(
            metadata,
            dimensions=dimensions,
            scalars=adapter.scalar_map,
            include_runtime_scalars=True,
        )
        projection = DatasetSourceProjection(
            code,
            kernel.symbol,
            tuple(item.source_name for item in parameters),
            "cuda",
            kernel.source_kind,
            kernel.location,
        )
        gate_source = CudaKernelSource.from_file(kernel.path, kernel.symbol)
        gate_case = CudaCompileCase(
            tuple(item.frontend_parameter() for item in parameters),
            adapter.grid,
            adapter.block,
            {},
            adapter.cuda_arch,
            include_dirs=resolve_cuda_include_dirs(kernel),
        )
    if (
        adapter.descriptor_sha256
        and _descriptor_sha256(metadata) != adapter.descriptor_sha256.lower()
    ):
        raise ValueError(
            f"Dataset invocation descriptor differs for {adapter.case_id}"
        )
    discovered_grid, discovered_block, discovered_contract = _launch_from_metadata(
        metadata, dimensions
    )
    if discovered_grid != adapter.grid or discovered_block != adapter.block:
        raise ValueError(
            "Dataset Harness launch differs from the captured compiler metadata"
        )
    if discovered_contract != adapter.grid_contract:
        raise ValueError("Dataset Harness grid contract differs from metadata")
    shape_coordinates = _shape_coordinate_mapping(parameters)
    compiler_view = SCRCCompilerView(
        schedule_values={},
        arguments=tuple(
            item.compiler_argument(shape_coordinates) for item in parameters
        ),
        grid=adapter.grid,
        grid_contract=adapter.grid_contract,
        grid_shape_coordinates=_grid_shape_coordinates(
            adapter.grid_contract,
            shape_coordinates,
        ),
        parameter_order=tuple(item.source_name for item in parameters),
    )
    _anonymous, _binding, mapping = anonymize_source(
        projection.code,
        source_symbol=projection.function,
        frontend=adapter.frontend,
    )
    runtime_mapping = {
        item.source_name: mapping[item.source_name] for item in parameters
    }
    profile = observer_profile_for(Frontend(adapter.frontend))
    entry = DatasetWorkloadEntry(
        kernel.case_id,
        kernel.frontend,
        kernel.symbol,
        kernel.location,
        f"{profile.profile_id}@{profile.profile_version}",
        DEFAULT_GATE2_POLICY.policy_id,
    )
    request = SCRCRunRequest(
        SourceProgram(projection.code, gate_source),
        HarnessBindingInstance.create(
            {},
            gate1_case=gate_case,
            gate2_context={
                "test_mode": "finite",
                "seed": adapter.seed,
                "adapter_protocol": adapter.protocol,
                "dataset_case_id": adapter.case_id,
            },
        ),
        adapter.frontend,
    )
    return PreparedFrontendDatasetWorkload(
        adapter,
        entry,
        kernel.location,
        projection,
        compiler_view,
        request,
        runtime_mapping,
        parameters,
        adapter.grid,
        adapter.block,
    )


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    aliases = {
        "bf16": "float32",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "u8": "uint8",
    }
    try:
        return np.dtype(aliases.get(dtype, dtype))
    except TypeError as exc:
        raise ValueError(f"unsupported Dataset tensor dtype {dtype!r}") from exc


def _random_input(
    rng: np.random.Generator, parameter: DatasetRuntimeParameter
) -> np.ndarray:
    dtype = _numpy_dtype(parameter.dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=parameter.shape).astype(dtype)
    if dtype.kind in {"i", "u"}:
        # Non-negative bounded integer fixtures are valid arithmetic/mask data
        # and, unlike the old negative sample range, are also memory-safe for
        # index tensors.  This is dtype-level fixture policy; no parameter name
        # or kernel family is inspected.
        return rng.integers(0, 4, size=parameter.shape).astype(dtype)
    return rng.normal(0.0, 0.25, size=parameter.shape).astype(dtype)


def _bind_shape(
    declared: list[int | str],
    actual: tuple[int, ...],
    dimensions: dict[str, int],
) -> None:
    if len(declared) != len(actual):
        raise ValueError("accepted J rank differs from the Dataset Harness")
    for expected, value in zip(declared, actual, strict=True):
        if isinstance(expected, int):
            if expected != value:
                raise ValueError("accepted J static extent differs from Dataset")
        elif dimensions.setdefault(expected, value) != value:
            raise ValueError("accepted J symbolic extent is inconsistent")


def build_frontend_dataset_gate2_suite(
    prepared: PreparedFrontendDatasetWorkload,
    artifact: Gate1Artifact,
) -> Gate2Suite:
    """Create the finite native Source/Replay/J case after Gate1 closure."""

    mapping = dict(prepared.source_to_opaque)
    runtime = {mapping[item.source_name]: item for item in prepared.runtime_parameters}
    j_inputs = {item.name: item for item in artifact.j.intent.inputs}
    j_outputs = {item.name: item for item in artifact.j.intent.outputs}
    rng = np.random.default_rng(prepared.adapter.seed)
    values: dict[str, Any] = {}
    dimensions: dict[str, int] = {}
    input_bindings: dict[str, str] = {}
    output_bindings: dict[str, str] = {}
    arguments: list[NativeArgument] = []

    for parameter in artifact.j.signature.parameters:
        binding = runtime.get(parameter.name)
        if binding is None:
            raise ValueError(f"accepted J parameter {parameter.name!r} lacks Dataset ABI")
        if parameter.kind is ParameterKind.CONSTEXPR:
            raise ValueError("native Dataset frontends do not expose constexpr parameters")
        if parameter.kind is ParameterKind.SCALAR:
            if binding.kind != "scalar" or binding.value is None:
                raise ValueError("accepted J scalar differs from Dataset ABI")
            key = f"scalar_{parameter.name}"
            values[key] = binding.value
            logical_ref = parameter.logical_ref
            if logical_ref in j_inputs:
                tensor = j_inputs[logical_ref]
                if tensor.shape:
                    raise ValueError("Dataset scalar may bind only a rank-0 J tensor")
                logical_key = f"logical_{parameter.name}"
                values[logical_key] = np.asarray(
                    binding.value, dtype=_numpy_dtype(tensor.dtype)
                )
                input_bindings[logical_ref] = logical_key
            elif logical_ref in j_outputs:
                raise ValueError("Dataset scalar cannot bind a J output")
            elif logical_ref is not None:
                if isinstance(binding.value, bool) or not isinstance(binding.value, int):
                    raise ValueError("J dimension-bound Dataset scalar must be integral")
                previous = dimensions.setdefault(logical_ref, int(binding.value))
                if previous != int(binding.value):
                    raise ValueError(
                        "J dimension-bound Dataset scalar is inconsistent"
                    )
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.SCALAR,
                    value_key=key,
                    dtype=binding.dtype,
                )
            )
            continue
        if parameter.kind is ParameterKind.OPAQUE:
            raise ValueError("opaque native Dataset arguments require an explicit sidecar")
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            raise ValueError("unsupported accepted J Dataset parameter kind")
        if binding.kind != "pointer" or parameter.mode.value != binding.mode:
            raise ValueError("accepted J pointer mode differs from Dataset ABI")
        logical_ref = parameter.logical_ref
        tensor = (
            j_inputs[logical_ref]
            if logical_ref in j_inputs
            else j_outputs.get(logical_ref)
        )
        if tensor is None:
            raise ValueError("Dataset pointer must anchor a J input/output tensor")
        if tensor.dtype != binding.dtype:
            raise ValueError("accepted J tensor dtype differs from Dataset ABI")
        _bind_shape(tensor.shape, binding.shape, dimensions)
        if parameter.mode is ParameterMode.INPUT:
            if logical_ref not in j_inputs:
                raise ValueError("Dataset input pointer does not bind a J input")
            key = f"input_{parameter.name}"
            values[key] = _random_input(rng, binding)
            input_bindings[logical_ref] = key
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.INPUT,
                    value_key=key,
                    dtype=binding.dtype,
                    shape=binding.shape,
                )
            )
        elif parameter.mode is ParameterMode.OUTPUT:
            if logical_ref not in j_outputs:
                raise ValueError("Dataset output pointer does not bind a J output")
            key = f"output_{parameter.name}"
            output_bindings[logical_ref] = key
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.OUTPUT,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    output_key=key,
                )
            )
        else:
            if logical_ref not in j_inputs or logical_ref not in j_outputs:
                raise ValueError("Dataset inout pointer must bind one J inout tensor")
            input_key = f"input_{parameter.name}"
            output_key = f"output_{parameter.name}"
            values[input_key] = _random_input(rng, binding)
            input_bindings[logical_ref] = input_key
            output_bindings[logical_ref] = output_key
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.INOUT,
                    value_key=input_key,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    output_key=output_key,
                )
            )
    if set(input_bindings) != set(j_inputs):
        raise ValueError("Dataset Harness input bindings differ from accepted J")
    if set(output_bindings) != set(j_outputs):
        raise ValueError("Dataset Harness output bindings differ from accepted J")
    refeval = ScalarLoweringSpec.create(
        artifact.j,
        dims=dimensions,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    execution = NativeExecutionSpec(
        tuple(arguments),
        prepared.request.binding.gate1_case,
        prepared.grid,
        prepared.block,
        metadata={
            "adapter_protocol": prepared.adapter.protocol,
            "dataset_case_id": prepared.adapter.case_id,
        },
    )
    output_keys = tuple(output_bindings[item.name] for item in artifact.j.intent.outputs)
    case = Gate2Case(
        "finite-hidden-1",
        refeval.j,
        values,
        execution,  # Gate2Case retains the historical field name `triton`.
        refeval,
        output_keys,
        {
            "test_mode": "finite_hidden_after_gate1",
            "adapter_protocol": prepared.adapter.protocol,
            "frontend": prepared.adapter.frontend,
        },
    )
    return Gate2Suite(
        prepared.adapter.case_id,
        (case,),
        prepared.request.binding.c_values,
    )


FRONTEND_DATASET_HARNESSES: dict[str, FrontendDatasetHarness] = {
    "tilelang:add2d": FrontendDatasetHarness(
        "tilelang:add2d",
        "tilelang",
        (("M", 4),),
        (),
        (4, 1, 1),
        (16, 1, 1),
        ("M", "1", "1"),
        401,
        source_sha256="3beb7fe73950ba50986bb0e8ee7af608a4c92c9d11c6710306414bcb42584733",
    ),
    "cuda:tilelang_add2d": FrontendDatasetHarness(
        "cuda:tilelang_add2d",
        "cuda",
        (("M", 4),),
        (("M", 4),),
        (4, 1, 1),
        (16, 1, 1),
        ("M", "1", "1"),
        409,
        source_sha256="2ffca814f59525234c27ee419e6856dd6488f94d1d32354977407ae43b772a81",
    ),
}


def _stable_dataset_seed(case_id: str, frontend: str) -> int:
    digest = hashlib.sha256(
        f"{FRONTEND_DATASET_PROJECTION_PROTOCOL}:{frontend}:{case_id}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _descriptor_sha256(metadata: Mapping[str, Any]) -> str:
    """Pin the ABI/launch facts consumed by a projected Harness."""

    payload = {
        "cuda_abi": metadata.get("cuda_abi"),
        "launch": metadata.get("launch"),
        "target": metadata.get("target"),
        "include_roots": metadata.get("include_roots"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_for_kernel(
    kernel: FrontendDatasetKernel,
    snapshots: Mapping[str, Mapping[str, int]],
) -> Mapping[str, int]:
    # Snapshot names are an optional invocation-value cache.  Exact stem
    # matching is the only accepted association; aliases are deliberately not
    # guessed from operator names.
    if kernel.frontend == "cuda" and kernel.metadata.get("origin") != "tilelang_export":
        return {}
    return snapshots.get(kernel.path.stem, {})


def _project_tilelang_harness(
    kernel: FrontendDatasetKernel,
    companion: FrontendDatasetKernel | None,
    *,
    snapshots: Mapping[str, Mapping[str, int]],
) -> FrontendDatasetHarness:
    if companion is None:
        raise FrontendDatasetCaseUnavailableError(
            "TileLang Dataset case lacks its generated CUDA ABI companion"
        )
    abi = companion.metadata.get("cuda_abi")
    rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("TileLang Dataset case lacks structured CUDA ABI metadata")
    if any(isinstance(row, Mapping) and row.get("kind") == "opaque" for row in rows):
        raise FrontendDatasetCaseUnavailableError(
            "TileLang Dataset opaque ABI requires an explicit descriptor sidecar"
        )
    factory_kwargs = _literal_factory_kwargs(kernel)
    dimensions, grid, block, contract, _scalars = _dataset_descriptor_values(
        companion.metadata,
        canonical_shapes=_canonical_for_kernel(kernel, snapshots),
    )
    return FrontendDatasetHarness(
        kernel.case_id,
        "tilelang",
        dimensions,
        (),
        grid,
        block,
        contract,
        _stable_dataset_seed(kernel.case_id, "tilelang"),
        factory_kwargs=factory_kwargs,
        source_sha256=hashlib.sha256(kernel.path.read_bytes()).hexdigest(),
        descriptor_sha256=_descriptor_sha256(companion.metadata),
    )


def _project_cuda_harness(
    kernel: FrontendDatasetKernel,
    *,
    snapshots: Mapping[str, Mapping[str, int]],
) -> FrontendDatasetHarness:
    metadata = _runtime_metadata(kernel)
    abi = metadata.get("cuda_abi")
    rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("CUDA Dataset case lacks structured ABI metadata")
    if any(isinstance(row, Mapping) and row.get("kind") == "opaque" for row in rows):
        raise FrontendDatasetCaseUnavailableError(
            "CUDA Dataset opaque ABI requires an explicit descriptor sidecar"
        )
    dimensions, grid, block, contract, scalar_values = _dataset_descriptor_values(
        metadata,
        canonical_shapes=_canonical_for_kernel(kernel, snapshots),
    )
    target = metadata.get("target")
    arch = (
        str(target.get("arch"))
        if isinstance(target, Mapping) and target.get("arch")
        else "sm_120"
    )
    return FrontendDatasetHarness(
        kernel.case_id,
        "cuda",
        dimensions,
        scalar_values,
        grid,
        block,
        contract,
        _stable_dataset_seed(kernel.case_id, "cuda"),
        cuda_arch=arch,
        source_sha256=hashlib.sha256(kernel.path.read_bytes()).hexdigest(),
        descriptor_sha256=_descriptor_sha256(metadata),
    )


def _diagnostic(
    kernel: FrontendDatasetKernel,
    exc: Exception,
) -> FrontendDatasetHarnessDiagnostic:
    reason = f"{type(exc).__name__}: {exc}".strip()
    if len(reason) > 2048:
        reason = reason[:2045] + "..."
    status = (
        "case_unavailable"
        if isinstance(exc, FrontendDatasetCaseUnavailableError)
        else "harness_invalid"
    )
    return FrontendDatasetHarnessDiagnostic(kernel.case_id, kernel.frontend, status, reason)


def project_frontend_dataset_harnesses(
    dataset_root: str | Path | None = None,
) -> FrontendDatasetHarnessProjection:
    """Project every mechanically complete Dataset entry into a Harness.

    The projection is deliberately conservative: it consumes only exported
    ABI/launch facts, literal factory defaults, and optional concrete snapshot
    extents.  Opaque descriptors, malformed launch expressions, missing
    companions, and non-literal factory defaults remain explicit unavailable
    cases instead of receiving semantic guesses.
    """

    from intentir.verify.frontend_dataset import load_frontend_dataset

    catalog = load_frontend_dataset(dataset_root)
    validation = catalog.validate()
    if not validation.passed:
        raise ValueError(
            "multi-frontend Dataset validation failed: "
            + "; ".join(validation.errors)
        )
    snapshots = _snapshot_canonical_shapes(catalog.root)
    generated_by_origin = {
        item.metadata.get("origin_case_id"): item
        for item in catalog.kernels
        if item.frontend == "cuda"
        and item.metadata.get("origin") == "tilelang_export"
    }
    ready: dict[str, FrontendDatasetHarness] = {}
    diagnostics: list[FrontendDatasetHarnessDiagnostic] = []

    for kernel in catalog.select(frontend="tilelang"):
        try:
            companion = generated_by_origin.get(kernel.case_id)
            ready[kernel.case_id] = _project_tilelang_harness(
                kernel,
                companion,
                snapshots=snapshots,
            )
        except Exception as exc:  # fail closed per case, never fabricate a binding
            diagnostics.append(_diagnostic(kernel, exc))

    for kernel in catalog.select(frontend="cuda"):
        try:
            ready[kernel.case_id] = _project_cuda_harness(
                kernel,
                snapshots=snapshots,
            )
        except Exception as exc:  # fail closed per case, never fabricate a binding
            diagnostics.append(_diagnostic(kernel, exc))

    diagnostics.sort(key=lambda item: (item.frontend, item.case_id))
    return FrontendDatasetHarnessProjection(ready, tuple(diagnostics))


__all__ = [
    "DatasetRuntimeParameter",
    "DatasetSourceProjection",
    "DatasetWorkloadEntry",
    "FRONTEND_DATASET_HARNESSES",
    "FRONTEND_DATASET_HARNESS_PROTOCOL",
    "FRONTEND_DATASET_PROJECTION_PROTOCOL",
    "FRONTEND_DATASET_SOURCE_PROTOCOL",
    "FrontendDatasetCaseUnavailableError",
    "FrontendDatasetHarness",
    "FrontendDatasetHarnessDiagnostic",
    "FrontendDatasetHarnessProjection",
    "PreparedFrontendDatasetWorkload",
    "build_frontend_dataset_gate2_suite",
    "project_frontend_dataset_harnesses",
    "prepare_frontend_dataset_workload",
]
