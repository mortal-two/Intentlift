"""Reusable real SCRC lift -> Gate1 -> Gate2 integration harness.

This harness invokes the production Gate1 entry point and can additionally run
the exact same immutable replay observation over a declared fixed-C runtime
profile family. Gate2 runs either the reviewed execution contract or a strict
declarative shape/value/seed matrix.  Every case executes the source, the
final-J replay, and the typed scalar MLIR lowering, then requires both
source/replay and source/scalar agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from intentir.ir.program import IntentProgram
from intentir.extraction.lifters.scrc import (
    ResponsesSCRCRelifter,
    SCRC_RELIFT_PROTOCOL,
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCLiftError,
    extract_scrc_artifact,
)
from intentir.extraction.providers.client import ProviderError
from intentir.extraction.providers.provider import get_provider
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.compare import TTIR_SCORE_SEMANTICS
from intentir.verify.gate1.model import (
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.orchestrate import (
    GATE1_FAMILY_ANTI_LEAKAGE_POLICY,
    GATE1_STAGNATION_POLICY,
    GATE1_TERMINATION_POLICY,
    MASKED_J_FAMILY_POLICY,
    MASKED_J_POLICY,
    RELIFT_GUIDANCE_POLICY,
    RELIFT_STRUCTURAL_GUIDANCE_POLICY,
    Gate1Config,
    Gate1Runner,
    canonical_artifact_digest,
)
from intentir.verify.gate1.composer import assert_replay_plan_matches
from intentir.verify.gate1.program_adapter import intent_program_from_gate1
from intentir.verify.gate1.specialization import (
    GATE1_FAMILY_DIFF_POLICY,
    GATE1_FAMILY_RELIFT_GUIDANCE_POLICY,
    GATE1_SPECIALIZATION_POLICY,
    Gate1SpecializationFamily,
    Gate1SpecializationProfile,
)
from intentir.verify.gate1.interventions import (
    PAIRED_INTERVENTION_FAMILY_POLICY,
)
from intentir.verify.gate1.pi_ablations import PiAblationKind, run_pi_ablations
from intentir.verify.gate1.replay import ResponsesReplayWriter
from intentir.verify.gate1.rule_release import load_rule_release
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate1.ttir import TTIR_FINGERPRINT_SCHEMA
from intentir.verify.gate2 import (
    ArgumentRole,
    GATE2_EXECUTION_PROTOCOL,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Gate2Suite,
    GATE2_COMPARISON_PROTOCOL,
    InputDistribution,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
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
    gate1_specialization_profiles: tuple[Gate2ShapeProfile, ...] = ()
    gate1_primary_profile_id: str | None = None

    def __post_init__(self) -> None:
        profiles = tuple(self.gate1_specialization_profiles)
        ids = [profile.profile_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Gate1 specialization profile_id values must be unique"
            )
        if profiles:
            if self.gate1_primary_profile_id not in set(ids):
                raise ValueError(
                    "Gate1 primary profile must be declared in the family"
                )
        elif self.gate1_primary_profile_id is not None:
            raise ValueError(
                "Gate1 primary profile requires specialization profiles"
            )

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
        elif argument.kind == "scalar":
            assert argument.value is not None
            arguments[argument.name] = RuntimeArgument.scalar(
                argument.value, _runtime_dtype(argument.dtype)
            )
        else:
            arguments[argument.name] = RuntimeArgument.none(argument.dtype)
    return CompileCase(arguments, case.grid)


def _runtime_by_name(case: LiveKernelCase) -> dict[str, SCRCCompilerArgument]:
    return {argument.name: argument for argument in case.arguments}


def _symbolic_dims(
    program: IntentProgram, case: LiveKernelCase
) -> dict[str, int]:
    runtime = _runtime_by_name(case)
    tensors = {
        tensor.name: tensor
        for tensor in [*program.function.inputs, *program.function.outputs]
    }
    dims: dict[str, int] = {}
    for parameter in program.interface.arguments:
        if parameter.kind != "tensor" or parameter.tensor_ref is None:
            continue
        argument = runtime.get(parameter.name)
        tensor = tensors.get(parameter.tensor_ref)
        if argument is None or tensor is None:
            raise ValueError(
                f"typed ABI tensor {parameter.name!r}/{parameter.tensor_ref!r} "
                "has no captured runtime binding"
            )
        if len(tensor.shape) != len(argument.shape):
            raise ValueError(
                f"IntentProgram tensor {tensor.name!r} rank differs from capture"
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
    program: IntentProgram | None,
    scalar_handoff_error: str | None = None,
    final_artifact_digest: str,
    gate1_j_digest: str,
    replay_artifact_digest: str,
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
        if parameter.mode in {ParameterMode.INPUT, ParameterMode.INOUT}:
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
                    (
                        ArgumentRole.INOUT
                        if parameter.mode is ParameterMode.INOUT
                        else ArgumentRole.INPUT
                    ),
                    value_key=tensor_ref,
                    dtype=parameter.dtype or concrete.dtype,
                    **(
                        {"output_key": tensor_ref}
                        if parameter.mode is ParameterMode.INOUT
                        else {}
                    ),
                )
            )
            if (
                parameter.mode is ParameterMode.INOUT
                and tensor_ref not in output_keys
            ):
                output_keys.append(tensor_ref)
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

    triton = TritonExecutionSpec(
        tuple(arguments), case.grid, dict(artifact.schedule.values)
    )
    if program is None:
        if not scalar_handoff_error:
            raise ValueError(
                "missing IntentProgram requires a scalar handoff error"
            )
        scalar = ScalarLoweringUnavailable(
            final_artifact_digest,
            gate1_j_digest,
            scalar_handoff_error,
            failure_class="intent_program_handoff",
        )
    else:
        if scalar_handoff_error is not None:
            raise ValueError(
                "scalar handoff error cannot accompany an IntentProgram"
            )
        try:
            scalar = ScalarLoweringSpec.create(
                program,
                final_artifact_digest=final_artifact_digest,
                gate1_j_digest=gate1_j_digest,
                dims=_symbolic_dims(program, case),
                input_bindings={
                    tensor.name: tensor.name
                    for tensor in program.function.inputs
                },
                output_bindings={
                    tensor.name: tensor.name
                    for tensor in program.function.outputs
                },
            )
        except Exception as exc:
            scalar = ScalarLoweringUnavailable(
                final_artifact_digest,
                gate1_j_digest,
                f"{type(exc).__name__}: {exc}",
                failure_class="scalar_spec_binding",
            )
    return Gate2Case(
        case_id or f"{case.name}.live",
        values,
        triton,
        scalar,
        replay_artifact_digest,
        tuple(output_keys),
        {output_key: case.tolerance for output_key in output_keys},
        dict(metadata or {}),
    )


def _gate1_config(
    profile: str,
    *,
    relift_rounds: int,
    semantic_interventions: int,
) -> Gate1Config:
    if profile not in {"formal", "normal"}:
        raise ValueError(f"unsupported Gate1 profile {profile!r}")
    formal = profile == "formal"
    return Gate1Config(
        max_relifts=relift_rounds,
        stable_closures=1,
        run_semantic_interventions=(formal and semantic_interventions > 0),
        max_semantic_interventions=(semantic_interventions if formal else 0),
        run_masked_ablation=formal,
        run_composer_shadow=formal,
    )


def _apply_shape_profile(
    case: LiveKernelCase, profile: Gate2ShapeProfile
) -> LiveKernelCase:
    arguments = {argument.name: argument for argument in case.arguments}
    unknown = sorted(
        (set(profile.tensor_shapes) | set(profile.scalar_values)) - set(arguments)
    )
    if unknown:
        raise ValueError(
            f"shape profile {profile.profile_id!r} has unknown argument(s): "
            + ", ".join(unknown)
        )
    overlap = sorted(set(profile.tensor_shapes) & set(profile.scalar_values))
    if overlap:
        raise ValueError(
            f"shape profile {profile.profile_id!r} has tensor/scalar "
            "override overlap: " + ", ".join(overlap)
        )

    updated: list[SCRCCompilerArgument] = []
    for argument in case.arguments:
        if argument.name in profile.tensor_shapes:
            if argument.kind != "tensor":
                raise ValueError(
                    f"shape profile tensor override {argument.name!r} "
                    "does not name a tensor argument"
                )
            updated.append(
                replace(argument, shape=profile.tensor_shapes[argument.name])
            )
        elif argument.name in profile.scalar_values:
            if argument.kind != "scalar":
                raise ValueError(
                    f"shape profile scalar override {argument.name!r} "
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


def _gate1_specialization_family(
    case: LiveKernelCase,
) -> Gate1SpecializationFamily | None:
    if not case.gate1_specialization_profiles:
        return None
    assert case.gate1_primary_profile_id is not None
    profiles = tuple(
        Gate1SpecializationProfile(
            profile.profile_id,
            replace(
                _compile_case(_apply_shape_profile(case, profile)),
                schedule=dict(case.schedule_values),
            ),
        )
        for profile in case.gate1_specialization_profiles
    )
    return Gate1SpecializationFamily(
        f"{case.name}.fixed_c",
        case.gate1_primary_profile_id,
        profiles,
    )


def _gate1_specialization_policy_json(case: LiveKernelCase) -> dict[str, Any]:
    family = _gate1_specialization_family(case)
    return {
        "enabled": family is not None,
        "role": GATE1_SPECIALIZATION_POLICY,
        "verdict_role": "normal_family_strict_all_pass_gate1_condition",
        "diff_policy": GATE1_FAMILY_DIFF_POLICY,
        "replay_policy": "one_fresh_replay_per_artifact_round",
        "relift_policy": "complete_family_sweep_then_one_source_aware_relift",
        "stagnation_basis": "canonical_artifact_and_exact_family_hard_diff",
        "anti_leakage_scope": (
            "full_fixed_c_family" if family is not None else "primary_profile"
        ),
        "paired_intervention_policy": (
            PAIRED_INTERVENTION_FAMILY_POLICY
        ),
        "masked_j_family_policy": MASKED_J_FAMILY_POLICY,
        "family": None if family is None else family.to_json(),
    }


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
    *,
    program: IntentProgram | None,
    scalar_handoff_error: str | None = None,
    final_artifact_digest: str,
    gate1_j_digest: str,
    replay_artifact_digest: str,
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
        shaped = _apply_shape_profile(case, shape_profile)
        for value_profile in values:
            profiled = _apply_gate2_value_profile(shaped, value_profile)
            for seed in seeds:
                matrix.append(
                    _gate2_case(
                        artifact,
                        profiled,
                        seed,
                        program=program,
                        scalar_handoff_error=scalar_handoff_error,
                        final_artifact_digest=final_artifact_digest,
                        gate1_j_digest=gate1_j_digest,
                        replay_artifact_digest=replay_artifact_digest,
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
        "unsupported": result.unsupported,
        "error": result.error,
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
        "execution_protocol": GATE2_EXECUTION_PROTOCOL,
        "case_id": result.case_id,
        "status": result.status.value,
        "input_digest": result.input_digest,
        "metadata": dict(result.metadata),
        "source": _execution_json(result.source),
        "replay": _execution_json(result.replay),
        "scalar": _execution_json(result.scalar),
        "source_replay": None
        if result.source_replay is None
        else asdict(result.source_replay),
        "source_scalar": None
        if result.source_scalar is None
        else asdict(result.source_scalar),
    }


def _gate2_suite_evidence_json(result) -> dict[str, Any]:
    return {
        "execution_protocol": GATE2_EXECUTION_PROTOCOL,
        "suite_id": result.suite_id,
        "passed": result.passed,
        "status_counts": result.status_counts,
        "failed_case_ids": list(result.failed_case_ids),
        "cases": [_gate2_evidence_json(case) for case in result.cases],
    }


def _progress(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _record_stage_ms(
    stage_ms: dict[str, float], stage: str, started: float
) -> None:
    stage_ms[stage] = round(stage_ms.get(stage, 0.0) + _elapsed_ms(started), 3)


def _lift_response_attempts(lift: Any) -> int:
    if not isinstance(lift, dict):
        return 0
    if lift.get("reused"):
        return 0
    attempts = lift.get("attempts", 0)
    if isinstance(attempts, bool):
        return 0
    if isinstance(attempts, int):
        return max(attempts, 0)
    if isinstance(attempts, list):
        return len(attempts)
    return 0


def _finalize_performance_report(
    report: dict[str, Any],
    *,
    pipeline_started: float,
    stage_ms: Mapping[str, float],
) -> None:
    """Attach observation-only timing and recorded invocation counts.

    These fields never participate in Gate1/Gate2 status selection.  Provider
    SDK retries below one recorded Lift response or writer invocation are not
    observable here and are deliberately not claimed as exact API-call counts.
    """

    gate1 = report.get("gate1")
    gate1 = gate1 if isinstance(gate1, dict) else {}
    relifts = report.get("relifts")
    relifts = relifts if isinstance(relifts, list) else []
    candidates = gate1.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    interventions = gate1.get("interventions")
    interventions = interventions if isinstance(interventions, list) else []
    pi_ablations = report.get("pi_ablations")
    pi_ablations = pi_ablations if isinstance(pi_ablations, list) else []

    initial_lift_responses = _lift_response_attempts(report.get("lift"))
    relift_responses = sum(_lift_response_attempts(item) for item in relifts)
    normal_writer_invocations = len(candidates)
    intervention_writer_invocations = len(interventions)
    masked_writer_invocations = int(gate1.get("masked_outcome") is not None)
    pi_writer_invocations = sum(
        max(int(item.get("attempts", 0)), 0)
        for item in pi_ablations
        if isinstance(item, dict)
        and not isinstance(item.get("attempts", 0), bool)
    )

    normal_compiles = [
        item.get("compile")
        for item in candidates
        if isinstance(item, dict) and isinstance(item.get("compile"), dict)
    ]
    hard_closed = sum(
        bool(item.get("comparison", {}).get("hard_pass"))
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("comparison"), dict)
    )
    family = report.get("gate1_specialization_family")
    family_records = (
        family.get("records", []) if isinstance(family, dict) else []
    )
    family_records = (
        family_records if isinstance(family_records, list) else []
    )
    family_source_compiles = sum(
        isinstance(item, dict)
        and isinstance(item.get("source_compile"), dict)
        and not bool(item.get("source_compile_reused"))
        for item in family_records
    )
    family_replay_compiles = sum(
        isinstance(item, dict)
        and isinstance(item.get("replay_compile"), dict)
        and not bool(item.get("replay_compile_reused"))
        for item in family_records
    )
    intervention_compiles = 0
    intervention_baseline_compiles = 0
    for item in interventions:
        if not isinstance(item, dict):
            continue
        profiles = item.get("profile_records")
        if isinstance(profiles, list) and profiles:
            intervention_compiles += sum(
                isinstance(profile, dict)
                and isinstance(profile.get("compile"), dict)
                for profile in profiles
            )
            intervention_baseline_compiles += sum(
                isinstance(profile, dict)
                and isinstance(profile.get("baseline_compile"), dict)
                and not bool(profile.get("baseline_compile_reused"))
                for profile in profiles
            )
        else:
            intervention_compiles += int(
                isinstance(item.get("compile"), dict)
            )
    pi_compiles = sum(
        isinstance(item, dict) and isinstance(item.get("compile"), dict)
        for item in pi_ablations
    )
    masked_family = gate1.get("masked_family_evidence")
    masked_profiles = (
        masked_family.get("records", [])
        if isinstance(masked_family, dict)
        else []
    )
    if isinstance(masked_profiles, list) and masked_profiles:
        masked_compiles = sum(
            isinstance(item, dict) and isinstance(item.get("compile"), dict)
            for item in masked_profiles
        )
        masked_source_compiles = sum(
            isinstance(item, dict)
            and isinstance(item.get("source_compile"), dict)
            and not bool(item.get("source_compile_reused"))
            for item in masked_profiles
        )
    else:
        masked_compiles = int(isinstance(gate1.get("masked_compile"), dict))
        masked_source_compiles = 0
    gate2_trials = 0
    for key in ("gate2_suite",):
        suite = report.get(key)
        if isinstance(suite, dict) and isinstance(suite.get("cases"), list):
            gate2_trials = len(suite["cases"])
            break
    else:
        gate2_trials = int(isinstance(report.get("gate2"), dict))

    timings: dict[str, float | None] = {
        key: (
            None
            if key not in stage_ms
            else round(float(stage_ms[key]), 3)
        )
        for key in (
            "initial_lift",
            "gate1_loop",
            "specialization_family",
            "pi_ablation",
            "gate2",
        )
    }
    timings["total"] = _elapsed_ms(pipeline_started)
    timings["gate1_core"] = (
        None
        if timings["initial_lift"] is None
        or timings["gate1_loop"] is None
        else round(timings["initial_lift"] + timings["gate1_loop"], 3)
    )
    report["performance"] = {
        "schema": "scrc_case_performance_v1",
        "protocol": "scrc_wall_clock_observation_v1",
        "clock": "time.perf_counter",
        "verdict_role": "observation_only",
        "timing_ms": timings,
        "recorded_invocations": {
            "initial_lift_responses": initial_lift_responses,
            "initial_lift_rounds": int(initial_lift_responses > 0),
            "relift_responses": relift_responses,
            "relift_rounds": len(relifts),
            "lift_responses_total": initial_lift_responses + relift_responses,
            "normal_replay_writer": normal_writer_invocations,
            "semantic_intervention_writer": intervention_writer_invocations,
            "masked_j_writer": masked_writer_invocations,
            "pi_ablation_writer": pi_writer_invocations,
            "writer_total": (
                normal_writer_invocations
                + intervention_writer_invocations
                + masked_writer_invocations
                + pi_writer_invocations
            ),
            "normal_replay_compile": len(normal_compiles),
            "normal_replay_compile_ok": sum(
                bool(item.get("ok")) for item in normal_compiles
            ),
            "specialization_profiles": len(family_records),
            "specialization_source_compile": family_source_compiles,
            "specialization_replay_compile": family_replay_compiles,
            "semantic_intervention_compile": intervention_compiles,
            "semantic_intervention_baseline_compile": (
                intervention_baseline_compiles
            ),
            "masked_j_compile": masked_compiles,
            "masked_j_source_compile": masked_source_compiles,
            "pi_ablation_compile": pi_compiles,
            "gate2_trials": gate2_trials,
        },
        "gate1_observation": {
            "status": gate1.get("status"),
            "outer_loop_termination": gate1.get("termination_reason"),
            "artifact_rounds_observed": len(
                {
                    item.get("relift_round")
                    for item in candidates
                    if isinstance(item, dict)
                    and isinstance(item.get("relift_round"), int)
                }
            ),
            "relift_attempts": len(relifts),
            "normal_replay_candidates": len(candidates),
            "normal_replay_compile_rate": (
                None
                if not normal_compiles
                else round(
                    sum(bool(item.get("ok")) for item in normal_compiles)
                    / len(normal_compiles),
                    6,
                )
            ),
            "hard_closed_candidates": hard_closed,
            "masked_j_outcome": gate1.get("masked_outcome"),
        },
        "limitations": [
            "provider SDK retries below one recorded response/invocation are not exposed",
            "wall time includes local compiler cache and host load effects",
        ],
    }


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
    for serialized, intervention in zip(
        payload.get("interventions", []),
        getattr(gate1, "interventions", ()),
        strict=True,
    ):
        if intervention.generation is not None:
            serialized["generation"].update(
                _generation_evidence(intervention.generation)
            )
    return payload


def _gate1_family_evidence_json(
    evidence: Mapping[str, Any],
    family: Gate1SpecializationFamily,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(evidence)))
    payload.update(
        {
            "role": GATE1_SPECIALIZATION_POLICY,
            "verdict_role": (
                "normal_family_strict_all_pass_gate1_condition"
            ),
            "anti_leakage_scope": "full_fixed_c_family",
            "paired_intervention_policy": (
                PAIRED_INTERVENTION_FAMILY_POLICY
            ),
            "masked_j_family_policy": MASKED_J_FAMILY_POLICY,
            "family": family.to_json(),
        }
    )
    return payload


def _pi_ablation_evidence_json(records) -> list[dict[str, Any]]:
    """Preserve source-blind ablation code for reproducible failure analysis."""

    payloads: list[dict[str, Any]] = []
    for record in records:
        payload = record.to_json()
        if record.generation is not None:
            payload["generation"].update(
                _generation_evidence(record.generation)
            )
        payloads.append(payload)
    return payloads


def _pi_ablation_control_evidence(gate1) -> dict[str, Any]:
    """Require a valid canonical-pi structural control before diagnostics."""

    if getattr(gate1, "source_fingerprint", None) is None:
        return {
            "valid": False,
            "reason": "source_fingerprint_unavailable",
            "gate1_status": gate1.status.value,
        }
    if not gate1.replay_closed:
        return {
            "valid": False,
            "reason": "canonical_pi_control_not_replay_closed",
            "gate1_status": gate1.status.value,
        }
    return {
        "valid": True,
        "reason": "canonical_pi_control_replay_closed",
        "gate1_status": gate1.status.value,
    }


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


def _select_gate2_replay_observation(
    gate1: Any,
    *,
    final_artifact_digest: str | None,
) -> Any | None:
    """Choose an immutable replay observation for independent Gate2 use.

    Only candidates explicitly bound to the final Gate1 artifact are eligible.
    This is stricter than selecting the numerically latest compiled round: a
    Relift round may replace the artifact and fail before producing a replay
    candidate, in which case falling back to an older replay would make Gate2
    evaluate a source/artifact pair that never existed in Gate1.

    Among aligned candidates, prefer a normal hard-closed observation.  When
    Gate1 has no structural closure, use the best compiled aligned observation
    as independent diagnostic evidence.  Gate2 consumes that generated source
    as-is; it never repairs it or feeds it back to the writer.
    """

    if not isinstance(final_artifact_digest, str) or not final_artifact_digest:
        return None
    candidates = tuple(
        candidate
        for candidate in gate1.candidates
        if getattr(candidate, "artifact_digest", None)
        == final_artifact_digest
    )
    hard_closed = [
        candidate
        for candidate in candidates
        if (
            candidate.generation is not None
            and candidate.comparison is not None
            and getattr(
                candidate,
                "overall_hard_pass",
                candidate.comparison.hard_pass,
            )
        )
    ]
    if hard_closed:
        return max(
            hard_closed,
            key=lambda candidate: (
                candidate.relift_round,
                candidate.generation_round,
                candidate.rank_tuple,
            ),
        )
    compiled = [
        candidate
        for candidate in candidates
        if candidate.generation is not None and candidate.compiled
    ]
    if not compiled:
        return None
    final_round = max(candidate.relift_round for candidate in compiled)
    return max(
        (
            candidate
            for candidate in compiled
            if candidate.relift_round == final_round
        ),
        key=lambda candidate: candidate.rank_tuple,
    )


def _select_gate1_specialization_evidence(
    gate1: Any,
    *,
    final_artifact_digest: str | None,
) -> Mapping[str, Any] | None:
    if not isinstance(final_artifact_digest, str) or not final_artifact_digest:
        return None
    candidates = [
        candidate
        for candidate in gate1.candidates
        if getattr(candidate, "specialization_evidence", None) is not None
        and getattr(candidate, "artifact_digest", None)
        == final_artifact_digest
    ]
    if not candidates:
        return None
    final_round = max(candidate.relift_round for candidate in candidates)
    selected = max(
        (
            candidate
            for candidate in candidates
            if candidate.relift_round == final_round
        ),
        key=lambda candidate: candidate.generation_round,
    )
    return selected.specialization_evidence


def run_case(
    case: LiveKernelCase,
    args: argparse.Namespace,
    *,
    artifact: Gate1Artifact | None = None,
    lift_evidence: Mapping[str, Any] | None = None,
    lift_origin: str | None = None,
) -> tuple[int, dict[str, Any]]:
    pipeline_started = time.perf_counter()
    stage_ms: dict[str, float] = {}
    provider = replace(
        get_provider(), model=args.model, reasoning_effort=args.effort
    )
    compiler_view = case.compiler_view()
    source_text = case.source_text
    source_encoded = source_text.encode("utf-8")
    registry = _rule_registry_from_args(args)
    gate1_profile = str(getattr(args, "gate1_profile", "formal"))
    if gate1_profile not in {"formal", "normal"}:
        raise ValueError(f"unsupported Gate1 profile {gate1_profile!r}")
    gate1_formal = gate1_profile == "formal"
    gate2_matrix_enabled = bool(getattr(args, "gate2_matrix", False))
    gate2_seeds = _selected_gate2_seeds(args)
    relift_rounds = int(getattr(args, "relift_rounds", 3))
    if relift_rounds < 1:
        raise ValueError("relift_rounds must be positive")
    semantic_interventions = int(
        getattr(args, "semantic_interventions", 0)
    )
    if semantic_interventions < 0:
        raise ValueError("semantic_interventions must be non-negative")
    pi_ablations_enabled = (
        bool(getattr(args, "pi_ablations", False)) and gate1_formal
    )
    pi_ablation_limit = int(getattr(args, "pi_ablation_limit", 3))
    if pi_ablation_limit < 1:
        raise ValueError("pi_ablation_limit must be positive")
    requested_pi_ablation_kinds = tuple(
        PiAblationKind(value)
        for value in (getattr(args, "pi_ablation_kinds", None) or ())
    )
    pi_ablation_kinds = requested_pi_ablation_kinds or (
        PiAblationKind.PLAN_STEP_DROPOUT,
        PiAblationKind.PLAN_MECHANISM_SUBSTITUTION,
    )
    report: dict[str, Any] = {
        "case": case.name,
        "model": args.model,
        "effort": args.effort,
        "seed": args.seed,
        "provider_request": {
            "name": provider.name,
            "base_url": provider.base_url.rstrip("/"),
            "wire_api": provider.wire_api,
            "model": provider.model,
            "reasoning_effort": provider.reasoning_effort,
            "api_key_recorded": False,
        },
        "gate1_execution_profile": gate1_profile,
        "relift_policy": {
            "protocol": SCRC_RELIFT_PROTOCOL,
            "artifact_rounds": relift_rounds,
            "maximum_semantic_relifts": relift_rounds - 1,
            "normal_recovery": "relift_only_restart_full_pipeline",
            "replay_generation_policy": (
                "fresh_only_no_prior_replay_source"
            ),
            "stagnation": {
                "policy": GATE1_STAGNATION_POLICY,
                "required_consecutive_observations": 2,
                "basis": "canonical_artifact_and_exact_hard_diff",
                "ttir_fingerprint_schema": TTIR_FINGERPRINT_SCHEMA,
            },
            "semantic_guidance": {
                "policy": RELIFT_GUIDANCE_POLICY,
                "structural_policy": RELIFT_STRUCTURAL_GUIDANCE_POLICY,
                "source": (
                    "first_completed_gate1_or_fixed_c_family_hard_comparison"
                ),
                "family_policy": (
                    GATE1_FAMILY_RELIFT_GUIDANCE_POLICY
                ),
                "transport_failure_relift": "disabled_without_hard_diff",
                "replay_source_input": "forbidden",
            },
            "termination": {
                "policy": GATE1_TERMINATION_POLICY,
                "no_closure_requires_structured_reason": True,
                "gate2_verdict_independence": True,
            },
        },
        "gate1_specialization_policy": _gate1_specialization_policy_json(
            case
        ),
        "gate1_anti_leakage_policy": {
            "fixed_c_family_closure_policy": (
                GATE1_FAMILY_ANTI_LEAKAGE_POLICY
            ),
            "paired_semantic_interventions": (
                gate1_formal and semantic_interventions > 0
            ),
            "maximum_interventions": (
                semantic_interventions if gate1_formal else 0
            ),
            "paired_replay_generation": (
                "fresh_from_mutated_lift_artifact"
                if gate1_formal and semantic_interventions > 0
                else "disabled"
            ),
            "paired_intervention_family_policy": (
                PAIRED_INTERVENTION_FAMILY_POLICY
            ),
            "paired_intervention_scope": (
                "full_fixed_c_family"
                if case.gate1_specialization_profiles
                else "primary_profile"
            ),
            "maximum_provider_attempts_per_generation": (
                1 if args.effort.strip().lower() == "xhigh" else 4
            ),
            "provider_request_policy": "effort_aware_no_blind_xhigh_retry",
            "compose_equality": "gate1_verdict_condition",
            "masked_j_role": (
                "actual_lift_pi_terminal_attack"
                if gate1_formal
                else "disabled_for_normal_gate1_closure"
            ),
            "masked_j_policy": MASKED_J_POLICY,
            "masked_j_family_policy": MASKED_J_FAMILY_POLICY,
            "masked_j_scope": (
                "full_fixed_c_family"
                if case.gate1_specialization_profiles
                else "primary_profile"
            ),
            "masked_j_required_outcome": (
                "all_profiles_legal_comparable_hard_mismatch"
                if case.gate1_specialization_profiles
                else "compile_or_hard_nonclosure"
            ),
            "masked_j_score_role": "not_used",
            "ttir_score_semantics": TTIR_SCORE_SEMANTICS,
            "replay_reference_naming": "opaque_replay_refs_v1",
            "pi_ablations": {
                "enabled": pi_ablations_enabled,
                "default_policy": "disabled_by_default_opt_in_diagnostic_only",
                "maximum": pi_ablation_limit,
                "requires_full_pi_replay_closed": True,
                "kinds": [kind.value for kind in pi_ablation_kinds],
                "verdict_role": "diagnostic",
                "pi_mutation_execution": "reject_before_writer",
                "mechanism_substitution": "c_strategy_then_recompose",
                "opaque_reference_discipline": (
                    "resolve_handles_never_emit_v1"
                ),
            },
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
        "gate2_execution_policy": {
            "protocol": GATE2_EXECUTION_PROTOCOL,
            "legs": ["source", "replay", "scalar"],
            "required_comparisons": ["source_replay", "source_scalar"],
            "target_backend_in_verdict": False,
            "source_binding": "captured_source_kernel_v1",
            "replay_binding": "final_gate1_artifact_digest_exact_match_v1",
            "scalar_binding": "typed_intent_program_from_final_gate1_j_v1",
        },
        "gate2_comparison_policy": {
            "protocol": GATE2_COMPARISON_PROTOCOL,
            "integer_and_boolean": "exact",
            "floating": "per_output_tolerance",
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

    def finish(code: int) -> tuple[int, dict[str, Any]]:
        _finalize_performance_report(
            report,
            pipeline_started=pipeline_started,
            stage_ms=stage_ms,
        )
        return code, report

    if artifact is None:
        _progress(
            "lift_start", kernel=case.name, model=args.model, effort=args.effort
        )
        lift_started = time.perf_counter()
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
        except ProviderError as exc:
            report["lift"] = {
                "ok": False,
                "attempts": [],
                "failure": {
                    "kind": "provider_request_failure",
                    "diagnostic": str(exc),
                    "lift_core_response_available": False,
                    "candidate_available": False,
                    "semantic_relift_eligible": False,
                },
            }
            _record_stage_ms(stage_ms, "initial_lift", lift_started)
            _progress(
                "lift_provider_failed",
                kernel=case.name,
                diagnostic=str(exc),
            )
            return finish(2)
        except SCRCLiftError as exc:
            report["lift"] = {
                "ok": False,
                "attempts": [
                    _lift_attempt_evidence(item) for item in exc.attempts
                ],
            }
            _record_stage_ms(stage_ms, "initial_lift", lift_started)
            _progress("lift_failed", kernel=case.name, attempts=len(exc.attempts))
            return finish(2)

        _record_stage_ms(stage_ms, "initial_lift", lift_started)
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
            "response_model": getattr(
                extraction.provenance, "response_model", None
            ),
            "prompt_sha256": extraction.provenance.prompt_sha256,
            "rule_registry_digest": registry.digest,
            "attempts": extraction.provenance.attempts,
            "repairs": list(extraction.provenance.repairs),
            "response_id": extraction.provenance.response_id,
            "response_ids": list(
                getattr(extraction.provenance, "response_ids", ())
            ),
            "usage": dict(getattr(extraction.provenance, "usage", {})),
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
    )
    relifter = ResponsesSCRCRelifter(
        registry=registry,
        provider=provider,
        compiler_view=compiler_view,
        max_repairs=args.lift_repairs,
        effort=args.effort,
    )
    specialization_family = _gate1_specialization_family(case)
    _progress("gate1_start", kernel=case.name)
    gate1_runner = Gate1Runner(
        config=_gate1_config(
            gate1_profile,
            relift_rounds=relift_rounds,
            semantic_interventions=semantic_interventions,
        )
    )
    gate1_started = time.perf_counter()
    try:
        gate1 = gate1_runner.run(
            case.kernel_source,
            artifact,
            _compile_case(case),
            registry,
            writer,
            relifter=relifter,
            specialization_family=specialization_family,
        )
    finally:
        _record_stage_ms(stage_ms, "gate1_loop", gate1_started)
    report["gate1"] = _gate1_evidence_json(gate1)
    report["relifts"] = _relift_evidence_json(relifter)
    _progress("gate1_complete", kernel=case.name, status=gate1.status.value)
    verified_artifact = getattr(gate1, "artifact", artifact)
    try:
        composed = assert_replay_plan_matches(verified_artifact, registry)
        final_artifact_digest = canonical_artifact_digest(
            verified_artifact,
            registry,
        )
        gate1_j_digest = composed.provenance.j_digest
    except Exception as exc:
        final_artifact_digest = None
        gate1_j_digest = None
        artifact_identity_error = f"{type(exc).__name__}: {exc}"
    else:
        artifact_identity_error = None
    gate2_replay = _select_gate2_replay_observation(
        gate1,
        final_artifact_digest=final_artifact_digest,
    )
    report["gate2_replay_alignment"] = {
        "policy": "final_gate1_artifact_digest_exact_match_v1",
        "status": "aligned" if gate2_replay is not None else "coverage_gap",
        "final_artifact_digest": final_artifact_digest,
        "final_gate1_j_digest": gate1_j_digest,
        "candidate_artifact_digest": (
            None
            if gate2_replay is None
            else gate2_replay.artifact_digest
        ),
        "candidate_relift_round": (
            None if gate2_replay is None else gate2_replay.relift_round
        ),
        "reason": (
            None
            if gate2_replay is not None
            else (
                "final_gate1_artifact_digest_unavailable"
                if final_artifact_digest is None
                else "no_executable_candidate_bound_to_final_gate1_artifact"
            )
        ),
        "diagnostic": artifact_identity_error,
    }
    if specialization_family is not None:
        family_evidence = _select_gate1_specialization_evidence(
            gate1,
            final_artifact_digest=final_artifact_digest,
        )
        if family_evidence is None:
            report["gate1_specialization_family"] = {
                "role": GATE1_SPECIALIZATION_POLICY,
                "verdict_role": (
                    "normal_family_strict_all_pass_gate1_condition"
                ),
                "anti_leakage_scope": "full_fixed_c_family",
                "paired_intervention_policy": (
                    PAIRED_INTERVENTION_FAMILY_POLICY
                ),
                "masked_j_family_policy": MASKED_J_FAMILY_POLICY,
                "status": "not_evaluated_before_primary_candidate",
                "family": specialization_family.to_json(),
            }
        else:
            report["gate1_specialization_family"] = (
                _gate1_family_evidence_json(
                    family_evidence, specialization_family
                )
            )
    if pi_ablations_enabled:
        pi_ablation_control = _pi_ablation_control_evidence(gate1)
        report["pi_ablation_control"] = pi_ablation_control
    else:
        pi_ablation_control = None
    if pi_ablation_control is not None and pi_ablation_control["valid"]:
        _progress("pi_ablation_start", kernel=case.name)
        pi_ablation_started = time.perf_counter()
        try:
            pi_ablations = run_pi_ablations(
                verified_artifact,
                gate1.source_fingerprint,
                _compile_case(case),
                registry,
                writer,
                compiler=gate1_runner.compiler,
                comparator=gate1_runner.comparator,
                limit=pi_ablation_limit,
                kinds=pi_ablation_kinds,
            )
        finally:
            _record_stage_ms(
                stage_ms, "pi_ablation", pi_ablation_started
            )
        report["pi_ablations"] = _pi_ablation_evidence_json(pi_ablations)
        _progress(
            "pi_ablation_complete",
            kernel=case.name,
            statuses=[item.status.value for item in pi_ablations],
        )
    elif pi_ablation_control is not None:
        report["pi_ablations"] = []
        _progress(
            "pi_ablation_skipped",
            kernel=case.name,
            reason=pi_ablation_control["reason"],
        )
    if not gate1.replay_closed:
        report["gate2_not_attempted"] = {
            "reason": "gate1_not_accepted",
            "required_binding": "final_accepted_gate1_j",
        }
        return finish(3)

    if gate2_replay is None or gate2_replay.generation is None:
        report["gate1"]["diagnostics"].append(
            "Gate1 closed without an executable replay observation"
        )
        return finish(3)
    if final_artifact_digest is None or gate1_j_digest is None:
        report["gate1"]["diagnostics"].append(
            "Gate1 closed without canonical final J identity"
        )
        return finish(3)
    scalar_handoff_error: str | None = None
    try:
        intent_program: IntentProgram | None = intent_program_from_gate1(
            verified_artifact
        )
    except Exception as exc:
        intent_program = None
        scalar_handoff_error = f"{type(exc).__name__}: {exc}"
    if intent_program is None:
        report["gate2_scalar_binding"] = {
            "status": "unavailable",
            "policy": "typed_intent_program_from_final_gate1_j_v1",
            "final_artifact_digest": final_artifact_digest,
            "gate1_j_digest": gate1_j_digest,
            "error": scalar_handoff_error,
        }
    else:
        report["gate2_scalar_binding"] = {
            "status": "available",
            "policy": "typed_intent_program_from_final_gate1_j_v1",
            "final_artifact_digest": final_artifact_digest,
            "gate1_j_digest": gate1_j_digest,
            "intent_j_digest": intent_program.j_digest,
            "intent_contract_digest": intent_program.contract_digest,
            "intent_program_digest": intent_program.program_digest,
        }
    replay = KernelSource.from_code(
        gate2_replay.generation.code,
        verified_artifact.interface.function_name,
    )
    report["replay"] = {
        "response_id": gate2_replay.generation.response_id,
        "model": gate2_replay.generation.response_model,
        "code_sha256": gate2_replay.generation.code_sha256,
        "code": gate2_replay.generation.code,
    }

    _progress("gate2_start", kernel=case.name)
    runner = Gate2Runner()
    gate2_started = time.perf_counter()
    assert gate2_replay.artifact_digest is not None
    if gate2_matrix_enabled:
        try:
            gate2_suite = runner.run_suite(
                case.kernel_source,
                replay,
                _gate2_suite(
                    verified_artifact,
                    case,
                    gate2_seeds,
                    program=intent_program,
                    scalar_handoff_error=scalar_handoff_error,
                    final_artifact_digest=final_artifact_digest,
                    gate1_j_digest=gate1_j_digest,
                    replay_artifact_digest=gate2_replay.artifact_digest,
                ),
            )
        except Exception as exc:
            report["gate2_suite"] = {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "setup_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _progress(
                "gate2_complete",
                kernel=case.name,
                status="setup_failed",
            )
            return finish(4)
        finally:
            _record_stage_ms(stage_ms, "gate2", gate2_started)
        report["gate2_suite"] = _gate2_suite_evidence_json(gate2_suite)
        _progress(
            "gate2_complete",
            kernel=case.name,
            status="gate2_pass" if gate2_suite.passed else "gate2_suite_failed",
            cases=len(gate2_suite.cases),
        )
        return finish(0 if gate2_suite.passed else 4)
    try:
        gate2 = runner.run(
            case.kernel_source,
            replay,
            _gate2_case(
                verified_artifact,
                case,
                args.seed,
                program=intent_program,
                scalar_handoff_error=scalar_handoff_error,
                final_artifact_digest=final_artifact_digest,
                gate1_j_digest=gate1_j_digest,
                replay_artifact_digest=gate2_replay.artifact_digest,
            ),
        )
    except Exception as exc:
        report["gate2"] = {
            "execution_protocol": GATE2_EXECUTION_PROTOCOL,
            "status": "setup_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _progress(
            "gate2_complete",
            kernel=case.name,
            status="setup_failed",
        )
        return finish(4)
    finally:
        _record_stage_ms(stage_ms, "gate2", gate2_started)
    report["gate2"] = _gate2_evidence_json(gate2)
    _progress("gate2_complete", kernel=case.name, status=gate2.status.value)
    return finish(0 if gate2.status is Gate2Status.PASS else 4)


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return run_case(build_case(args), args)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _rule_registry_from_args(args: argparse.Namespace):
    registry = default_triton_rule_registry()
    release_path = getattr(args, "rule_release", None)
    if release_path is not None:
        registry = load_rule_release(Path(release_path)).apply(registry)
    return registry.frozen_snapshot()


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
    parser.add_argument(
        "--rule-release",
        type=Path,
        help=(
            "explicit approved Gate1 RuleRegistry release; loaded once before "
            "the run and bound by the resulting registry digest"
        ),
    )
    parser.add_argument("--gate2-matrix", action="store_true")
    parser.add_argument(
        "--gate1-profile",
        choices=("formal", "normal"),
        default="formal",
        help=(
            "formal adds the terminal masked-J attack and composer shadow to "
            "the real closure; normal performs closure and relift only"
        ),
    )
    parser.add_argument(
        "--gate2-seed", dest="gate2_seeds", action="append", type=int
    )
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument(
        "--semantic-interventions",
        type=_non_negative_int,
        default=0,
        help=(
            "opt-in diagnostic paired J/B/C interventions after normal closure; "
            "zero on the live acceptance path"
        ),
    )
    parser.add_argument(
        "--pi-ablations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "opt in to exact pi-tamper and C.strategy→recompose diagnostics; "
            "not part of live acceptance"
        ),
    )
    parser.add_argument("--pi-ablation-limit", type=_positive_int, default=3)
    parser.add_argument(
        "--pi-ablation-kind",
        dest="pi_ablation_kinds",
        action="append",
        choices=tuple(kind.value for kind in PiAblationKind),
    )
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
        "gate1_profile": report.get("gate1_execution_profile"),
        "lift": report.get("lift", {}).get("ok"),
        "gate1": report.get("gate1", {}).get("status"),
        "anti_leakage_basis": report.get("gate1", {}).get(
            "anti_leakage_basis"
        ),
        "gate2": report.get("gate2", {}).get("status"),
        "gate2_suite": report.get("gate2_suite", {}).get("passed"),
        "relifts": len(report.get("relifts", [])),
        "report": args.report,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
