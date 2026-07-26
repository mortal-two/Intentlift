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
        if len({name for name, _value in scalar_values}) != len(scalar_values):
            raise ValueError("Dataset scalar bindings must be unique")
        for label, values in (("grid", self.grid), ("block", self.block)):
            normalized = tuple(int(value) for value in values)
            if len(normalized) != 3 or any(value <= 0 for value in normalized):
                raise ValueError(f"Dataset {label} must contain three positive values")
            object.__setattr__(self, label, normalized)
        contract = tuple(str(value) for value in self.grid_contract)
        if len(contract) != 3 or any(not value for value in contract):
            raise ValueError("Dataset grid contract must describe all three axes")
        kwargs = tuple(self.factory_kwargs)
        if len({name for name, _value in kwargs}) != len(kwargs):
            raise ValueError("TileLang factory kwargs must be unique")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Dataset Harness seed must be non-negative")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.source_sha256):
            raise ValueError("Dataset Harness requires a pinned source SHA-256")
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

    def __post_init__(self) -> None:
        if not self.source_name.isidentifier() or not self.semantic_name.isidentifier():
            raise ValueError("Dataset parameter names must be identifiers")
        if self.kind not in {"pointer", "scalar", "opaque"}:
            raise ValueError("unsupported Dataset parameter kind")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("Dataset parameter dtype must be non-empty")
        shape = tuple(int(value) for value in self.shape)
        if self.kind == "pointer":
            if self.rank != len(shape) or not shape or any(value <= 0 for value in shape):
                raise ValueError("Dataset pointer shape must match its positive rank")
            if self.mode not in {"input", "output", "inout"}:
                raise ValueError("Dataset pointer requires an effect mode")
            if self.value is not None:
                raise ValueError("Dataset pointer cannot carry a scalar value")
        elif self.kind == "scalar":
            if self.rank != 0 or shape or self.mode != "value" or self.value is None:
                raise ValueError("Dataset scalar requires one fixed runtime value")
        elif self.rank != 0 or shape or self.mode != "value" or self.value is not None:
            raise ValueError("Dataset opaque exposes only its public type")
        object.__setattr__(self, "shape", shape)

    def frontend_parameter(self) -> FrontendParameter:
        return FrontendParameter(self.kind, self.dtype, self.rank, self.mode)

    def compiler_argument(self) -> SCRCCompilerArgument:
        if self.kind == "pointer":
            return SCRCCompilerArgument(
                self.source_name,
                "tensor",
                self.dtype,
                self.shape,
                mode=self.mode,
            )
        if self.kind == "scalar":
            return SCRCCompilerArgument(
                self.source_name,
                "scalar",
                self.dtype,
                value=self.value,
                mode="value",
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


def _metadata_parameters(
    metadata: Mapping[str, Any],
    *,
    dimensions: Mapping[str, int],
    scalars: Mapping[str, bool | int | float],
    source_names: tuple[str, ...] | None = None,
    include_runtime_scalars: bool,
) -> tuple[DatasetRuntimeParameter, ...]:
    abi = metadata.get("cuda_abi")
    if not isinstance(abi, Mapping) or not isinstance(abi.get("parameters"), list):
        raise ValueError("Dataset kernel lacks structured CUDA ABI metadata")
    raw_rows = abi["parameters"]
    if not all(isinstance(item, Mapping) for item in raw_rows):
        raise ValueError("Dataset ABI parameters must all be objects")
    rows = sorted(raw_rows, key=lambda item: int(item.get("position", 0)))
    if not include_runtime_scalars:
        rows = [item for item in rows if item.get("kind") == "pointer"]
    if source_names is not None and len(source_names) != len(rows):
        raise ValueError("materialized TileLang ABI differs from generated metadata")
    result: list[DatasetRuntimeParameter] = []
    for index, row in enumerate(rows):
        semantic_name = str(row.get("semantic_name") or row.get("name") or "")
        source_name = source_names[index] if source_names is not None else str(row.get("name") or "")
        kind = str(row.get("kind") or "")
        value = None
        shape: tuple[int, ...] = ()
        if kind == "pointer":
            shape = _resolve_shape(row.get("shape"), dimensions)
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
            )
        )
    return tuple(result)


def _materialize_tilelang_projection(
    kernel: FrontendDatasetKernel,
    factory_kwargs: Mapping[str, bool | int | float | str],
) -> DatasetSourceProjection:
    module_name = f"_intentir_dataset_lift_{abs(hash((str(kernel.path), kernel.symbol))):x}"
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
        parameters = _metadata_parameters(
            kernel.metadata,
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
        metadata = kernel.metadata
    discovered_grid, discovered_block, discovered_contract = _launch_from_metadata(
        metadata, dimensions
    )
    if discovered_grid != adapter.grid or discovered_block != adapter.block:
        raise ValueError(
            "Dataset Harness launch differs from the captured compiler metadata"
        )
    if discovered_contract != adapter.grid_contract:
        raise ValueError("Dataset Harness grid contract differs from metadata")
    compiler_view = SCRCCompilerView(
        schedule_values={},
        arguments=tuple(item.compiler_argument() for item in parameters),
        grid=adapter.grid,
        grid_contract=adapter.grid_contract,
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
        low = 0 if dtype.kind == "u" else -3
        return rng.integers(low, 4, size=parameter.shape).astype(dtype)
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


__all__ = [
    "DatasetRuntimeParameter",
    "DatasetSourceProjection",
    "DatasetWorkloadEntry",
    "FRONTEND_DATASET_HARNESSES",
    "FRONTEND_DATASET_HARNESS_PROTOCOL",
    "FRONTEND_DATASET_SOURCE_PROTOCOL",
    "FrontendDatasetHarness",
    "PreparedFrontendDatasetWorkload",
    "build_frontend_dataset_gate2_suite",
    "prepare_frontend_dataset_workload",
]
