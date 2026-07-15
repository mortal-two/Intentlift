"""Reusable real SCRC lift -> Gate1 -> Gate2 integration harness.

Gate1 remains anchored to one reviewed compile specialization.  Gate2 can run
either that execution contract or a strict declarative shape/value/seed matrix.
It does not add a verifier or an alternate correctness path: all decisions are
still made by the production SCRC lifter, Gate1, and Gate2 implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from intentir.extraction.lifters.scrc import (
    ResponsesSCRCRelifter,
    SCRC_RELIFT_PROTOCOL,
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCLiftError,
    extract_scrc_artifact,
)
from intentir.extraction.providers.provider import get_provider
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.model import (
    Gate1Artifact,
    Gate1Status,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import ResponsesReplayWriter
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate2 import (
    ArgumentRole,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Gate2Suite,
    InputDistribution,
    Lower2CSpec,
    TensorInputSpec,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
    generate_tensor_input,
)

from scrc_probe_sources import (
    add_kernel,
    matmul_kernel,
    rmsnorm_kernel,
    row_sum_kernel,
    softmax_kernel,
)


def _unwrap_python_function(function: Any) -> Any:
    current = function
    visited: set[int] = set()
    while not inspect.isfunction(current):
        marker = id(current)
        if marker in visited:
            raise TypeError("cyclic Triton function wrapper chain")
        visited.add(marker)
        nested = getattr(current, "fn", None)
        if nested is None:
            raise TypeError("Triton function wrapper has no Python function")
        current = nested
    return current


@dataclass(frozen=True)
class Gate2ShapeProfile:
    profile_id: str
    tensor_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    scalar_values: Mapping[str, int | float | bool] = field(default_factory=dict)
    grid: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Gate2 shape profile_id must not be empty")
        if self.grid is not None and (
            not 1 <= len(self.grid) <= 3
            or any(int(extent) <= 0 for extent in self.grid)
        ):
            raise ValueError(
                "Gate2 shape profile grid must be one to three positive extents"
            )


@dataclass(frozen=True)
class Gate2ValueProfile:
    profile_id: str
    distributions: Mapping[str, InputDistribution] = field(default_factory=dict)
    scale: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Gate2 value profile_id must not be empty")
        if self.scale is not None and (
            isinstance(self.scale, bool)
            or not np.isfinite(self.scale)
            or self.scale < 0
        ):
            raise ValueError(
                "Gate2 value profile scale must be finite and non-negative"
            )


@dataclass(frozen=True)
class LiveKernelCase:
    name: str
    function: Any
    arguments: tuple[SCRCCompilerArgument, ...]
    grid: tuple[int, ...]
    schedule_values: dict[str, int | float | bool]
    compiler_slots: tuple[str, ...]
    grid_contract: tuple[str, ...]
    constexpr_parameters: tuple[str, ...]
    input_scale: float
    tolerance: Tolerance
    input_distributions: Mapping[str, InputDistribution] = field(
        default_factory=dict
    )
    input_scales: Mapping[str, float] = field(default_factory=dict)
    gate2_shape_profiles: tuple[Gate2ShapeProfile, ...] = ()
    gate2_value_profiles: tuple[Gate2ValueProfile, ...] = ()

    @property
    def function_name(self) -> str:
        return self.python_function.__name__

    @property
    def python_function(self) -> Any:
        return _unwrap_python_function(self.function)

    @property
    def source_path(self) -> Path:
        path = inspect.getsourcefile(self.python_function)
        if path is None:
            raise ValueError(
                f"cannot locate source file for {self.function_name!r}"
            )
        return Path(path).resolve()

    @property
    def source_text(self) -> str:
        return inspect.getsource(self.python_function)

    @property
    def kernel_source(self) -> KernelSource:
        return KernelSource.from_file(
            self.source_path, self.function_name, trusted=True
        )

    def compiler_view(self) -> SCRCCompilerView:
        return SCRCCompilerView(
            self.schedule_values,
            self.compiler_slots,
            (),
            self.arguments,
            self.grid,
            self.grid_contract,
            tuple(
                [argument.name for argument in self.arguments]
                + list(self.constexpr_parameters)
            ),
            self.constexpr_parameters,
        )


def build_case(args: argparse.Namespace) -> LiveKernelCase:
    n = int(args.n)
    if n <= 0:
        raise ValueError("N must be positive")
    if args.kernel == "add":
        block = int(args.block if args.block is not None else 256)
        arguments = (
            SCRCCompilerArgument("a_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("b_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument(
                "n_elements", "scalar", "i32", value=n
            ),
        )
        return LiveKernelCase(
            "add",
            add_kernel,
            arguments,
            ((n + block - 1) // block,),
            {
                "BLOCK": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(n_elements, BLOCK)",),
            ("BLOCK",),
            1.0,
            Tolerance(rtol=1.0e-5, atol=1.0e-6),
        )
    if args.kernel == "row_sum":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "row_sum first closure requires one BLOCK_N tile covering N"
            )
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m,)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
        )
        return LiveKernelCase(
            "row_sum",
            row_sum_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )
    if args.kernel == "softmax":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "softmax first closure requires one BLOCK_N tile covering N"
            )
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
        )
        return LiveKernelCase(
            "softmax",
            softmax_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            2.0,
            Tolerance(rtol=2.0e-4, atol=2.0e-5),
        )
    if args.kernel == "rmsnorm":
        m = int(args.m)
        if m <= 0:
            raise ValueError("M must be positive")
        block = int(args.block if args.block is not None else 1024)
        if block < n:
            raise ValueError(
                "rmsnorm first closure requires one BLOCK_N tile covering N"
            )
        eps = float(args.eps)
        arguments = (
            SCRCCompilerArgument("x_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("w_ptr", "tensor", "f32", (n,)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
            SCRCCompilerArgument("eps", "scalar", "f32", value=eps),
        )
        return LiveKernelCase(
            "rmsnorm",
            rmsnorm_kernel,
            arguments,
            (m,),
            {
                "BLOCK_N": block,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            ("M",),
            ("BLOCK_N",),
            0.25,
            Tolerance(rtol=3.0e-4, atol=3.0e-5),
        )
    if args.kernel == "matmul":
        m = int(args.m)
        k = int(args.k)
        if m <= 0:
            raise ValueError("M must be positive")
        if k <= 0:
            raise ValueError("K must be positive")
        block_m = int(args.block_m)
        block_n = int(args.block_n)
        block_k = int(args.block_k)
        if min(block_m, block_n, block_k) <= 0:
            raise ValueError("matmul block sizes must be positive")
        arguments = (
            SCRCCompilerArgument("a_ptr", "tensor", "f16", (m, k)),
            SCRCCompilerArgument("b_ptr", "tensor", "f16", (k, n)),
            SCRCCompilerArgument("out_ptr", "tensor", "f32", (m, n)),
            SCRCCompilerArgument("M", "scalar", "i32", value=m),
            SCRCCompilerArgument("N", "scalar", "i32", value=n),
            SCRCCompilerArgument("K", "scalar", "i32", value=k),
        )
        return LiveKernelCase(
            "matmul",
            matmul_kernel,
            arguments,
            (
                (m + block_m - 1) // block_m,
                (n + block_n - 1) // block_n,
            ),
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            },
            ("num_warps", "num_stages"),
            (
                "ceil_div(M, BLOCK_M)",
                "ceil_div(N, BLOCK_N)",
            ),
            ("BLOCK_M", "BLOCK_N", "BLOCK_K"),
            0.25,
            Tolerance(rtol=2.0e-3, atol=2.0e-3),
        )
    raise KeyError(args.kernel)


def _numpy_dtype(value: str) -> np.dtype:
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return np.dtype(aliases.get(value.removeprefix("torch."), value))


def _runtime_dtype(value: str) -> str:
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "i1": "bool",
    }
    return aliases.get(value.removeprefix("torch."), value)


def _compile_case(case: LiveKernelCase) -> CompileCase:
    arguments: dict[str, RuntimeArgument] = {}
    for argument in case.arguments:
        if argument.kind == "tensor":
            arguments[argument.name] = RuntimeArgument.tensor(
                argument.shape, _runtime_dtype(argument.dtype)
            )
        else:
            assert argument.value is not None
            arguments[argument.name] = RuntimeArgument.scalar(
                argument.value, _runtime_dtype(argument.dtype)
            )
    return CompileCase(arguments, case.grid)


def _runtime_by_name(case: LiveKernelCase) -> dict[str, SCRCCompilerArgument]:
    return {argument.name: argument for argument in case.arguments}


def _symbolic_dims(artifact, case: LiveKernelCase) -> dict[str, int]:
    runtime = _runtime_by_name(case)
    tensors = {
        tensor.name: tensor
        for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    dims: dict[str, int] = {}
    for parameter in artifact.interface.parameters:
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            continue
        argument = runtime[parameter.name]
        tensor = tensors[parameter.tensor_ref]
        if len(tensor.shape) != len(argument.shape):
            raise ValueError(
                f"J tensor {tensor.name!r} rank differs from compiler view"
            )
        for declared, concrete in zip(
            tensor.shape, argument.shape, strict=True
        ):
            if isinstance(declared, str):
                previous = dims.setdefault(declared, concrete)
                if previous != concrete:
                    raise ValueError(
                        f"symbolic dim {declared!r} has inconsistent bindings"
                    )
            elif int(declared) != concrete:
                raise ValueError(
                    f"static J extent {declared} differs from {concrete}"
                )
    return dims


def _gate2_case(
    artifact,
    case: LiveKernelCase,
    seed: int,
    *,
    case_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Gate2Case:
    rng = np.random.default_rng(seed)
    runtime = _runtime_by_name(case)
    values: dict[str, Any] = {}
    arguments: list[TritonArgument] = []
    output_keys: list[str] = []

    for parameter in artifact.interface.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            continue
        concrete = runtime[parameter.name]
        if parameter.kind is ParameterKind.SCALAR:
            value_key = parameter.tensor_ref or parameter.name
            scalar_value: Any = concrete.value
            if parameter.tensor_ref is not None:
                dtype = _numpy_dtype(parameter.dtype or concrete.dtype)
                scalar_value = dtype.type(concrete.value)
            values[value_key] = scalar_value
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=value_key,
                )
            )
            continue
        tensor_ref = parameter.tensor_ref
        if tensor_ref is None:
            raise ValueError(f"pointer {parameter.name!r} has no tensor_ref")
        if parameter.mode is ParameterMode.INPUT:
            dtype = _numpy_dtype(parameter.dtype or concrete.dtype)
            distribution = case.input_distributions.get(
                parameter.name, InputDistribution.NORMAL
            )
            scale = case.input_scales.get(parameter.name, case.input_scale)
            values[tensor_ref] = generate_tensor_input(
                TensorInputSpec(
                    concrete.shape,
                    str(dtype),
                    distribution,
                    scale,
                ),
                rng=rng,
            )
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=tensor_ref,
                    dtype=parameter.dtype or concrete.dtype,
                )
            )
        elif parameter.mode is ParameterMode.OUTPUT:
            output_keys.append(tensor_ref)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=parameter.dtype or concrete.dtype,
                    shape=concrete.shape,
                    output_key=tensor_ref,
                )
            )
        else:
            raise ValueError(
                f"live harness does not yet have a case binding for "
                f"{parameter.name}:{parameter.mode.value}"
            )

    if len(output_keys) != 1:
        raise ValueError("current lower2c leg requires exactly one output")
    triton = TritonExecutionSpec(
        tuple(arguments), case.grid, dict(artifact.schedule.values)
    )
    lowering = Lower2CSpec(
        artifact.j.intent,
        _symbolic_dims(artifact, case),
        {tensor.name: tensor.name for tensor in artifact.j.intent.inputs},
        output_keys[0],
    )
    return Gate2Case(
        case_id or f"{case.name}.live",
        values,
        triton,
        lowering,
        tuple(output_keys),
        {output_keys[0]: case.tolerance},
        dict(metadata or {}),
    )


def _apply_gate2_shape_profile(
    case: LiveKernelCase, profile: Gate2ShapeProfile
) -> LiveKernelCase:
    arguments = {argument.name: argument for argument in case.arguments}
    unknown = sorted(
        (set(profile.tensor_shapes) | set(profile.scalar_values)) - set(arguments)
    )
    if unknown:
        raise ValueError(
            f"Gate2 shape profile {profile.profile_id!r} has unknown argument(s): "
            + ", ".join(unknown)
        )
    overlap = sorted(set(profile.tensor_shapes) & set(profile.scalar_values))
    if overlap:
        raise ValueError(
            f"Gate2 shape profile {profile.profile_id!r} has tensor/scalar "
            "override overlap: " + ", ".join(overlap)
        )

    updated: list[SCRCCompilerArgument] = []
    for argument in case.arguments:
        if argument.name in profile.tensor_shapes:
            if argument.kind != "tensor":
                raise ValueError(
                    f"Gate2 shape profile tensor override {argument.name!r} "
                    "does not name a tensor argument"
                )
            updated.append(
                replace(argument, shape=profile.tensor_shapes[argument.name])
            )
        elif argument.name in profile.scalar_values:
            if argument.kind != "scalar":
                raise ValueError(
                    f"Gate2 shape profile scalar override {argument.name!r} "
                    "does not name a scalar argument"
                )
            updated.append(
                replace(argument, value=profile.scalar_values[argument.name])
            )
        else:
            updated.append(argument)
    return replace(
        case,
        arguments=tuple(updated),
        grid=case.grid if profile.grid is None else profile.grid,
    )


def _apply_gate2_value_profile(
    case: LiveKernelCase, profile: Gate2ValueProfile
) -> LiveKernelCase:
    argument_names = {argument.name for argument in case.arguments}
    unknown = sorted(set(profile.distributions) - argument_names)
    if unknown:
        raise ValueError(
            f"Gate2 value profile {profile.profile_id!r} has unknown argument(s): "
            + ", ".join(unknown)
        )
    distributions = dict(case.input_distributions)
    distributions.update(
        {
            name: InputDistribution(distribution)
            for name, distribution in profile.distributions.items()
        }
    )
    return replace(
        case,
        input_scale=case.input_scale if profile.scale is None else profile.scale,
        input_distributions=distributions,
    )


def _gate2_suite(
    artifact,
    case: LiveKernelCase,
    seeds: tuple[int, ...],
) -> Gate2Suite:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Gate2 matrix seeds must be non-empty and unique")
    shapes = case.gate2_shape_profiles or (Gate2ShapeProfile("contract"),)
    values = case.gate2_value_profiles or (Gate2ValueProfile("contract"),)
    if len({profile.profile_id for profile in shapes}) != len(shapes):
        raise ValueError("Gate2 shape profile_id values must be unique")
    if len({profile.profile_id for profile in values}) != len(values):
        raise ValueError("Gate2 value profile_id values must be unique")

    matrix: list[Gate2Case] = []
    for shape_profile in shapes:
        shaped = _apply_gate2_shape_profile(case, shape_profile)
        for value_profile in values:
            profiled = _apply_gate2_value_profile(shaped, value_profile)
            for seed in seeds:
                matrix.append(
                    _gate2_case(
                        artifact,
                        profiled,
                        seed,
                        case_id=(
                            f"{case.name}.{shape_profile.profile_id}."
                            f"{value_profile.profile_id}.seed{seed}"
                        ),
                        metadata={
                            "shape_profile": shape_profile.profile_id,
                            "value_profile": value_profile.profile_id,
                            "seed": seed,
                        },
                    )
                )
    return Gate2Suite(f"{case.name}.matrix", tuple(matrix))


def _execution_json(result) -> dict[str, Any]:
    return {
        "leg": result.leg,
        "ok": result.ok,
        "error": result.error,
        "unsupported": result.unsupported,
        "metadata": dict(result.metadata),
        "outputs": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in result.outputs.items()
        },
    }


def _gate2_evidence_json(result) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "status": result.status.value,
        "metadata": dict(result.metadata),
        "source": _execution_json(result.source),
        "replay": _execution_json(result.replay),
        "lowering": _execution_json(result.lowering),
        "source_replay": None
        if result.source_replay is None
        else asdict(result.source_replay),
        "source_lowering": None
        if result.source_lowering is None
        else asdict(result.source_lowering),
        "replay_lowering": None
        if result.replay_lowering is None
        else asdict(result.replay_lowering),
    }


def _gate2_suite_evidence_json(result) -> dict[str, Any]:
    return {
        "suite_id": result.suite_id,
        "passed": result.passed,
        "status_counts": result.status_counts,
        "failed_case_ids": list(result.failed_case_ids),
        "cases": [_gate2_evidence_json(case) for case in result.cases],
    }


def _progress(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _redact_artifact_source(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    source = result.pop("source", None)
    if isinstance(source, str):
        encoded = source.encode("utf-8")
        result["source_sha256"] = hashlib.sha256(encoded).hexdigest()
        result["source_bytes"] = len(encoded)
    return result


def _artifact_evidence_json(artifact) -> dict[str, Any]:
    return _redact_artifact_source(artifact.to_json())


def _generation_evidence(generation) -> dict[str, Any]:
    encoded = generation.code.encode("utf-8")
    return {
        "code": generation.code,
        "code_bytes": len(encoded),
    }


def _gate1_evidence_json(gate1) -> dict[str, Any]:
    payload = gate1.to_json()
    payload["artifact"] = _redact_artifact_source(payload["artifact"])
    for serialized, candidate in zip(
        payload["candidates"], gate1.candidates, strict=True
    ):
        if candidate.generation is not None:
            serialized["generation"].update(
                _generation_evidence(candidate.generation)
            )
    if gate1.masked_generation is not None:
        payload["masked_generation"].update(
            _generation_evidence(gate1.masked_generation)
        )
    return payload


def _lift_attempt_evidence(item) -> dict[str, Any]:
    encoded = item.response_text.encode("utf-8")
    return {
        "attempt": item.attempt,
        "model": item.model,
        "response_id": item.response_id,
        "diagnostic": item.diagnostic,
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "response_bytes": len(encoded),
    }


def _relift_evidence_json(
    relifter: ResponsesSCRCRelifter,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for record in relifter.records:
        item: dict[str, Any] = {
            "relift_round": record.relift_round,
            "ok": record.extraction is not None,
            "request": dict(record.request),
        }
        if record.extraction is not None:
            extraction = record.extraction
            item.update(
                {
                    "model": extraction.provenance.model,
                    "prompt_sha256": extraction.provenance.prompt_sha256,
                    "attempts": extraction.provenance.attempts,
                    "repairs": list(extraction.provenance.repairs),
                    "response_id": extraction.provenance.response_id,
                    "artifact": _artifact_evidence_json(extraction.artifact),
                }
            )
        else:
            item["diagnostic"] = record.diagnostic
            item["attempts"] = [
                _lift_attempt_evidence(attempt)
                for attempt in record.rejected_attempts
            ]
        evidence.append(item)
    return evidence


def _selected_gate2_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    requested = tuple(getattr(args, "gate2_seeds", ()) or ())
    if requested:
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in requested
        ):
            raise TypeError("--gate2-seed values must be integers")
        if len(requested) != len(set(requested)):
            raise ValueError("--gate2-seed values must be unique")
        return requested
    if getattr(args, "gate2_matrix", False):
        return (args.seed, args.seed + 1, args.seed + 2)
    return (args.seed,)


def _gate2_matrix_policy_json(
    case: LiveKernelCase,
    *,
    enabled: bool,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "seeds": list(seeds),
        "shape_profiles": [
            {
                "profile_id": profile.profile_id,
                "tensor_shapes": {
                    name: list(shape)
                    for name, shape in sorted(profile.tensor_shapes.items())
                },
                "scalar_values": dict(sorted(profile.scalar_values.items())),
                "grid": None if profile.grid is None else list(profile.grid),
            }
            for profile in case.gate2_shape_profiles
        ],
        "value_profiles": [
            {
                "profile_id": profile.profile_id,
                "distributions": {
                    name: InputDistribution(distribution).value
                    for name, distribution in sorted(
                        profile.distributions.items()
                    )
                },
                "scale": profile.scale,
            }
            for profile in case.gate2_value_profiles
        ],
    }


def run_case(
    case: LiveKernelCase,
    args: argparse.Namespace,
    *,
    artifact: Gate1Artifact | None = None,
    lift_evidence: Mapping[str, Any] | None = None,
    lift_origin: str | None = None,
) -> tuple[int, dict[str, Any]]:
    provider = replace(
        get_provider(), model=args.model, reasoning_effort=args.effort
    )
    compiler_view = case.compiler_view()
    source_text = case.source_text
    source_encoded = source_text.encode("utf-8")
    registry = default_triton_rule_registry()
    gate2_matrix_enabled = bool(getattr(args, "gate2_matrix", False))
    gate2_seeds = _selected_gate2_seeds(args)
    relift_rounds = int(getattr(args, "relift_rounds", 3))
    if relift_rounds < 1:
        raise ValueError("relift_rounds must be positive")
    report: dict[str, Any] = {
        "case": case.name,
        "model": args.model,
        "effort": args.effort,
        "seed": args.seed,
        "relift_policy": {
            "protocol": SCRC_RELIFT_PROTOCOL,
            "artifact_rounds": relift_rounds,
            "maximum_semantic_relifts": relift_rounds - 1,
        },
        "gate2_input_policy": {
            "default_distribution": InputDistribution.NORMAL.value,
            "argument_distributions": {
                name: InputDistribution(distribution).value
                for name, distribution in sorted(
                    case.input_distributions.items()
                )
            },
            "scale": case.input_scale,
            "argument_scales": dict(sorted(case.input_scales.items())),
            "tolerance": asdict(case.tolerance),
        },
        "gate2_matrix_policy": _gate2_matrix_policy_json(
            case, enabled=gate2_matrix_enabled, seeds=gate2_seeds
        ),
        "source": {
            "path": str(case.source_path),
            "function": case.function_name,
            "sha256": hashlib.sha256(source_encoded).hexdigest(),
            "bytes": len(source_encoded),
        },
        "compiler_view": compiler_view.to_json(),
    }

    if artifact is None:
        _progress(
            "lift_start", kernel=case.name, model=args.model, effort=args.effort
        )
        try:
            extraction = extract_scrc_artifact(
                source_text,
                case.function_name,
                registry=registry,
                provider=provider,
                compiler_view=compiler_view,
                max_repairs=args.lift_repairs,
                effort=args.effort,
            )
        except SCRCLiftError as exc:
            report["lift"] = {
                "ok": False,
                "attempts": [
                    _lift_attempt_evidence(item) for item in exc.attempts
                ],
            }
            _progress("lift_failed", kernel=case.name, attempts=len(exc.attempts))
            return 2, report

        artifact = extraction.artifact
        _progress(
            "lift_complete",
            kernel=case.name,
            attempts=extraction.provenance.attempts,
            response_id=extraction.provenance.response_id,
        )
        report["lift"] = {
            "ok": True,
            "model": extraction.provenance.model,
            "prompt_sha256": extraction.provenance.prompt_sha256,
            "rule_registry_digest": registry.digest,
            "attempts": extraction.provenance.attempts,
            "repairs": list(extraction.provenance.repairs),
            "response_id": extraction.provenance.response_id,
            "artifact": _artifact_evidence_json(artifact),
        }
    else:
        if lift_evidence is None or not lift_evidence.get("ok"):
            raise ValueError("reused artifact requires successful lift evidence")
        reused = json.loads(json.dumps(dict(lift_evidence)))
        reused["reused"] = True
        reused["reused_from"] = lift_origin
        report["lift"] = reused
        _progress(
            "lift_reused",
            kernel=case.name,
            response_id=reused.get("response_id"),
            origin=lift_origin,
        )

    writer = ResponsesReplayWriter(
        provider=provider,
        model_candidates=(args.model,),
        effort=args.effort,
        max_retries=1,
    )
    relifter = ResponsesSCRCRelifter(
        registry=registry,
        provider=provider,
        compiler_view=compiler_view,
        max_repairs=args.lift_repairs,
        effort=args.effort,
    )
    _progress("gate1_start", kernel=case.name)
    gate1 = Gate1Runner(
        config=Gate1Config(
            max_repairs=args.rewrite_repairs,
            max_relifts=relift_rounds,
            stable_closures=1,
            run_masked_ablation=True,
        )
    ).run(
        case.kernel_source,
        artifact,
        _compile_case(case),
        registry,
        writer,
        relifter=relifter,
    )
    report["gate1"] = _gate1_evidence_json(gate1)
    report["relifts"] = _relift_evidence_json(relifter)
    _progress("gate1_complete", kernel=case.name, status=gate1.status.value)
    closed = next(
        (
            candidate
            for candidate in gate1.candidates
            if candidate.comparison is not None
            and candidate.comparison.hard_pass
            and candidate.generation is not None
        ),
        None,
    )
    if not gate1.replay_closed:
        if (
            getattr(args, "diagnose_rejected", False)
            and gate1.status is Gate1Status.LEAKAGE_REJECTED
            and closed is not None
            and closed.generation is not None
        ):
            replay = KernelSource.from_code(
                closed.generation.code, artifact.interface.function_name
            )
            _progress("gate2_diagnostic_start", kernel=case.name)
            try:
                runner = Gate2Runner()
                if gate2_matrix_enabled:
                    diagnostic_suite = runner.run_suite(
                        case.kernel_source,
                        replay,
                        _gate2_suite(artifact, case, gate2_seeds),
                    )
                    report["gate2_diagnostic_suite"] = {
                        "kind": "offline_gate2_for_gate1_rejected_sample",
                        "gate1_status": gate1.status.value,
                        **_gate2_suite_evidence_json(diagnostic_suite),
                    }
                    _progress(
                        "gate2_diagnostic_complete",
                        kernel=case.name,
                        status=(
                            "gate2_pass"
                            if diagnostic_suite.passed
                            else "gate2_suite_failed"
                        ),
                        cases=len(diagnostic_suite.cases),
                    )
                else:
                    diagnostic = runner.run(
                        case.kernel_source,
                        replay,
                        _gate2_case(artifact, case, args.seed),
                    )
                    report["gate2_diagnostic"] = {
                        "kind": "offline_gate2_for_gate1_rejected_sample",
                        "gate1_status": gate1.status.value,
                        **_gate2_evidence_json(diagnostic),
                    }
                    _progress(
                        "gate2_diagnostic_complete",
                        kernel=case.name,
                        status=diagnostic.status.value,
                    )
            except Exception as exc:
                key = (
                    "gate2_diagnostic_suite"
                    if gate2_matrix_enabled
                    else "gate2_diagnostic"
                )
                report[key] = {
                    "kind": "offline_gate2_for_gate1_rejected_sample",
                    "gate1_status": gate1.status.value,
                    "status": "setup_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return 3, report

    if closed is None or closed.generation is None:
        report["gate1"]["diagnostics"].append(
            "Gate1 closed without a recoverable replay candidate"
        )
        return 3, report
    replay = KernelSource.from_code(
        closed.generation.code, artifact.interface.function_name
    )
    report["replay"] = {
        "response_id": closed.generation.response_id,
        "model": closed.generation.response_model,
        "code_sha256": closed.generation.code_sha256,
        "code": closed.generation.code,
    }

    _progress("gate2_start", kernel=case.name)
    runner = Gate2Runner()
    if gate2_matrix_enabled:
        gate2_suite = runner.run_suite(
            case.kernel_source,
            replay,
            _gate2_suite(artifact, case, gate2_seeds),
        )
        report["gate2_suite"] = _gate2_suite_evidence_json(gate2_suite)
        _progress(
            "gate2_complete",
            kernel=case.name,
            status="gate2_pass" if gate2_suite.passed else "gate2_suite_failed",
            cases=len(gate2_suite.cases),
        )
        return (0 if gate2_suite.passed else 4), report
    gate2 = runner.run(
        case.kernel_source,
        replay,
        _gate2_case(artifact, case, args.seed),
    )
    report["gate2"] = _gate2_evidence_json(gate2)
    _progress("gate2_complete", kernel=case.name, status=gate2.status.value)
    return (0 if gate2.status is Gate2Status.PASS else 4), report


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return run_case(build_case(args), args)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser(default_kernel: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        choices=("add", "row_sum", "softmax", "rmsnorm", "matmul"),
        default=default_kernel,
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--k", type=int, default=19)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--block", type=int)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--gate2-matrix", action="store_true")
    parser.add_argument(
        "--gate2-seed", dest="gate2_seeds", action="append", type=int
    )
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--rewrite-repairs", type=int, default=2)
    parser.add_argument(
        "--relift-rounds",
        type=_positive_int,
        default=3,
        help="total Gate1 artifact rounds, including the initial Lift",
    )
    parser.add_argument("--report")
    if default_kernel == "matmul":
        parser.set_defaults(m=17, n=13, k=19)
    return parser


def main(default_kernel: str = "add") -> int:
    args = _parser(default_kernel).parse_args()
    code, report = run(args)
    _write_report(args.report, report)
    summary = {
        "exit_code": code,
        "kernel": report.get("case"),
        "lift": report.get("lift", {}).get("ok"),
        "gate1": report.get("gate1", {}).get("status"),
        "gate2": report.get("gate2", {}).get("status"),
        "gate2_suite": report.get("gate2_suite", {}).get("passed"),
        "relifts": len(report.get("relifts", [])),
        "report": args.report,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
