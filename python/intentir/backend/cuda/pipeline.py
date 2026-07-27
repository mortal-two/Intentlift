"""Verification-neutral IntentIR -> CUDA lowering entry points.

Gate, paper, command-line, and future target frontends may share this module,
but none of their verdicts participate in lowering authorization.  Callers must
first construct an ``IntentFunction`` (or Intent dialect MLIR) whose A/B/C
contract they accept.  This module emits/probes/compiles that exact program and
returns executable cubins plus a neutral lowering record.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import intentir.backend.cuda.compile as backend_compile
import intentir.backend.cuda.orchestrate as backend_orchestrate
from intentir.backend.cuda.launch_config import (
    KernelLaunchConfig,
    parse_launch_configs,
)
from intentir.backend.cuda.selection import (
    LoweringTarget,
    VariantSelection,
    read_intent_module,
    residual_intent_ops,
    select_variant,
)
from intentir.backend.cuda.schedules import ScheduleDecision
from intentir.frontend import emit_module
from intentir.ir.model import IntentFunction
from intentir.ir.production_snapshot import content_digest


@dataclass(frozen=True)
class MLIRTransformationReceipt:
    """Facts attached by the C++ pass to the concrete post-pass MLIR state."""

    lowering_path_kind: str
    reduction_topology: str
    transformation_trace: tuple[str, ...]
    fma_count: int
    approximate_div_count: int
    shuffle_count: int
    barrier_count: int
    all_reduce_count: int
    block_threads: int
    grid_blocks: int
    grid_stride_loop_count: int
    rows_per_block: int
    stage_count: int
    reduction_parts: int
    predicated_value_if_count: int
    folded_constexpr_select_count: int
    host_value_capture_count: int
    epilogue_load_motion_count: int
    approximation_mode: str
    launch_plans: tuple[KernelLaunchConfig, ...] = ()

    @property
    def observation_changing(self) -> bool:
        return self.fma_count > 0 or self.approximate_div_count > 0 or self.approximation_mode != "none" or self.reduction_topology in {
            "subgroup_shuffle",
            "workgroup_hierarchical",
            "gpu_all_reduce",
            "subgroup_feature",
            "subgroup_rows",
            "multi_stage_partial_finalize",
        }

    @property
    def topology_digest(self) -> str:
        return content_digest(
            {
                "lowering_path_kind": self.lowering_path_kind,
                "reduction_topology": self.reduction_topology,
                "rows_per_block": self.rows_per_block,
                "stage_count": self.stage_count,
                "reduction_parts": self.reduction_parts,
                "transformation_trace": list(self.transformation_trace),
                "approximation_mode": self.approximation_mode,
                "launch_plans": [item.to_dict() for item in self.launch_plans],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lowering_path_kind": self.lowering_path_kind,
            "reduction_topology": self.reduction_topology,
            "transformation_trace": list(self.transformation_trace),
            "fma_count": self.fma_count,
            "approximate_div_count": self.approximate_div_count,
            "shuffle_count": self.shuffle_count,
            "barrier_count": self.barrier_count,
            "all_reduce_count": self.all_reduce_count,
            "block_threads": self.block_threads,
            "grid_blocks": self.grid_blocks,
            "grid_stride_loop_count": self.grid_stride_loop_count,
            "rows_per_block": self.rows_per_block,
            "stage_count": self.stage_count,
            "reduction_parts": self.reduction_parts,
            "predicated_value_if_count": self.predicated_value_if_count,
            "folded_constexpr_select_count": self.folded_constexpr_select_count,
            "host_value_capture_count": self.host_value_capture_count,
            "epilogue_load_motion_count": self.epilogue_load_motion_count,
            "approximation_mode": self.approximation_mode,
            "launch_plans": [item.to_dict() for item in self.launch_plans],
            "observation_changing": self.observation_changing,
            "topology_digest": self.topology_digest,
        }


def _transformation_receipt(mlir: str) -> MLIRTransformationReceipt | None:
    """Decode only C++-authored receipt attrs; never infer from function names."""

    def string_attr(name: str) -> str | None:
        match = re.search(
            rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"",
            mlir,
        )
        return match.group(1) if match is not None else None

    def integer_attr(name: str) -> int | None:
        match = re.search(
            rf"{re.escape(name)}\s*=\s*(-?[0-9]+)\s*:\s*i64",
            mlir,
        )
        return int(match.group(1)) if match is not None else None

    path = string_attr("intent.backend.lowering_path_kind")
    topology = string_attr("intent.backend.reduction_topology")
    approximation_mode = (
        string_attr("intent.backend.approximation_mode") or "none"
    )
    trace_match = re.search(
        r"intent\.backend\.transformation_trace\s*=\s*\[([^\]]*)\]",
        mlir,
    )
    if path is None and topology is None and trace_match is None:
        return None
    if path is None or topology is None or trace_match is None:
        raise ValueError("C++ concrete-state receipt is incomplete")
    trace = tuple(re.findall(r'\"([^\"]+)\"', trace_match.group(1)))
    counts = {
        name: integer_attr(f"intent.backend.{name}")
        for name in (
            "fma_count",
            "shuffle_count",
            "barrier_count",
            "all_reduce_count",
            "block_threads",
            "predicated_value_if_count",
            "folded_constexpr_select_count",
        )
    }
    rows_per_block = integer_attr("intent.backend.rows_per_block")
    if rows_per_block is None:
        rows_per_block = 1
    stage_count = integer_attr("intent.backend.stage_count")
    if stage_count is None:
        stage_count = 1
    reduction_parts = integer_attr("intent.backend.reduction_parts")
    if reduction_parts is None:
        reduction_parts = 1
    grid_blocks = integer_attr("intent.backend.grid_blocks")
    if grid_blocks is None:
        grid_blocks = 0
    grid_stride_loop_count = integer_attr(
        "intent.backend.grid_stride_loop_count"
    )
    if grid_stride_loop_count is None:
        grid_stride_loop_count = 0
    approximate_div_count = integer_attr(
        "intent.backend.approximate_div_count"
    )
    if approximate_div_count is None:
        approximate_div_count = 0
    host_value_capture_count = integer_attr(
        "intent.backend.host_value_capture_count"
    )
    if host_value_capture_count is None:
        host_value_capture_count = 0
    epilogue_load_motion_count = integer_attr(
        "intent.backend.epilogue_load_motion_count"
    )
    if epilogue_load_motion_count is None:
        epilogue_load_motion_count = 0
    if (
        not trace
        or rows_per_block <= 0
        or stage_count <= 0
        or reduction_parts <= 0
        or grid_blocks < 0
        or grid_stride_loop_count not in {0, 1}
        or any(value is None or value < 0 for value in counts.values())
    ):
        raise ValueError("C++ concrete-state receipt has invalid trace/counts")
    assert all(value is not None for value in counts.values())
    return MLIRTransformationReceipt(
        lowering_path_kind=path,
        reduction_topology=topology,
        transformation_trace=trace,
        fma_count=int(counts["fma_count"]),
        approximate_div_count=approximate_div_count,
        shuffle_count=int(counts["shuffle_count"]),
        barrier_count=int(counts["barrier_count"]),
        all_reduce_count=int(counts["all_reduce_count"]),
        block_threads=int(counts["block_threads"]),
        grid_blocks=grid_blocks,
        grid_stride_loop_count=grid_stride_loop_count,
        rows_per_block=rows_per_block,
        stage_count=stage_count,
        reduction_parts=reduction_parts,
        predicated_value_if_count=int(counts["predicated_value_if_count"]),
        folded_constexpr_select_count=int(
            counts["folded_constexpr_select_count"]
        ),
        host_value_capture_count=host_value_capture_count,
        epilogue_load_motion_count=epilogue_load_motion_count,
        approximation_mode=approximation_mode,
        launch_plans=tuple(parse_launch_configs(mlir)),
    )


@dataclass(frozen=True)
class LoweredProgram:
    """One compiled CUDA program and the structural decision that produced it."""

    selection: VariantSelection
    kernels: tuple[backend_compile.CompiledKernel, ...]
    source_mlir_sha256: str
    specialized_mlir_sha256: str | None = None
    transformation_receipt: MLIRTransformationReceipt | None = None
    pipeline_id: str = "cuda.pipeline.direct.v1"
    pipeline_mlir_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("lowering must produce at least one CUDA kernel")
        for kernel in self.kernels:
            if not kernel.cubin.startswith(b"\x7fELF"):
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} is not an ELF cubin"
                )
            residuals = residual_intent_ops(kernel.binary_mlir)
            if residuals:
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} retains Intent ops: "
                    f"{list(residuals)}"
                )
        if self.selection.input_sha256 != self.source_mlir_sha256:
            raise ValueError("selection and compiled source hashes disagree")
        if self.selection.pipeline_kind == "specialized":
            if self.specialized_mlir_sha256 != self.selection.probe_output_sha256:
                raise ValueError(
                    "compiled specialized MLIR differs from its structural probe"
                )
        elif self.specialized_mlir_sha256 is not None:
            raise ValueError("generic lowering cannot carry specialized MLIR")
        if (
            self.transformation_receipt is not None
            and self.selection.pipeline_kind != "specialized"
        ):
            raise ValueError("generic lowering cannot carry a transformation receipt")
        if not self.pipeline_id:
            raise ValueError("lowering pipeline_id must be non-empty")
        if (
            self.pipeline_mlir_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.pipeline_mlir_sha256) is None
        ):
            raise ValueError("pipeline MLIR digest must be SHA-256")

    @property
    def executable_digest(self) -> str:
        return content_digest(
            [hashlib.sha256(kernel.cubin).hexdigest() for kernel in self.kernels]
        )

    @property
    def concrete_mlir_sha256(self) -> str:
        """Digest the real compiler state used for duplicate-state rejection."""

        return (
            self.specialized_mlir_sha256
            or self.pipeline_mlir_sha256
            or self.source_mlir_sha256
        )

    def to_record(self) -> dict[str, object]:
        """Return JSON-safe compiler provenance without embedding binary bytes."""

        return {
            "selection": self.selection.to_dict(),
            "source_mlir_sha256": self.source_mlir_sha256,
            "specialized_mlir_sha256": self.specialized_mlir_sha256,
            "pipeline_id": self.pipeline_id,
            "pipeline_mlir_sha256": self.pipeline_mlir_sha256,
            "transformation_receipt": (
                self.transformation_receipt.to_dict()
                if self.transformation_receipt is not None
                else None
            ),
            "executable_digest": self.executable_digest,
            "kernel_count": len(self.kernels),
            "kernels": [
                {
                    "binary_name": kernel.binary_name,
                    "kernel_name": kernel.kernel_name,
                    "cubin_size_bytes": len(kernel.cubin),
                    "cubin_sha256": hashlib.sha256(kernel.cubin).hexdigest(),
                    "cubin_is_elf": kernel.cubin.startswith(b"\x7fELF"),
                    "binary_mlir_sha256": hashlib.sha256(
                        kernel.binary_mlir.encode("utf-8")
                    ).hexdigest(),
                }
                for kernel in self.kernels
            ],
        }


def _existing_path(intent_module: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(intent_module)
        return path if path.is_file() else None
    except OSError:
        return None


def _compile_generic(
    intent_module: str | os.PathLike[str], source: str
) -> tuple[backend_compile.CompiledKernel, ...]:
    path = _existing_path(intent_module)
    if path is not None:
        return tuple(backend_compile.compile_all_kernels(path))
    with tempfile.TemporaryDirectory(prefix="intentir-lowering-") as directory:
        source_path = Path(directory) / "input.mlir"
        source_path.write_text(source, encoding="utf-8")
        return tuple(backend_compile.compile_all_kernels(source_path))


def _pipeline_mlir_digest(
    kernels: tuple[backend_compile.CompiledKernel, ...],
) -> str:
    return content_digest(
        [
            hashlib.sha256(kernel.binary_mlir.encode("utf-8")).hexdigest()
            for kernel in kernels
        ]
    )


def lower_to_cuda(
    intent_module: str | os.PathLike[str],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    schedule_decision: ScheduleDecision | None = None,
    allow_specialized: bool = True,
    require_specialized: bool = False,
) -> LoweredProgram:
    """Compile accepted Intent dialect MLIR through generic or direct CUDA.

    ``only_fast_path`` is useful for an explicit tuning/ablation candidate.
    With ``require_specialized=True`` a structural probe miss is reported as a
    rejected search candidate instead of silently compiling the generic
    fallback under a specialized candidate id.
    """

    source = read_intent_module(intent_module)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    selected_target = target or LoweringTarget()
    if selected_target.chip != backend_compile.CHIP:
        raise ValueError(
            "lowering target must match the configured CUDA backend: "
            f"requested {selected_target.chip!r}, configured "
            f"{backend_compile.CHIP!r}"
        )
    effective_overrides = dict(pass_overrides or {})
    if schedule_decision is not None:
        if only_fast_path not in (None, schedule_decision.fast_path_id):
            raise ValueError(
                "schedule decision and only_fast_path identify different variants"
            )
        existing = effective_overrides.get(schedule_decision.fast_path_id)
        if existing not in (None, schedule_decision.pass_argument):
            raise ValueError(
                "schedule decision conflicts with an explicit pass override"
            )
        only_fast_path = schedule_decision.fast_path_id
        effective_overrides[only_fast_path] = schedule_decision.pass_argument
    selection = select_variant(
        source,
        target=selected_target,
        pass_overrides=effective_overrides,
        only_fast_path=only_fast_path,
        allow_specialized=allow_specialized,
    )
    if selection.input_sha256 != source_digest:
        raise RuntimeError("selector did not probe the exact lowering input")
    if require_specialized and selection.pipeline_kind != "specialized":
        raise ValueError(
            "required specialized candidate failed whole-graph admission"
        )

    specialized_digest: str | None = None
    transformation_receipt: MLIRTransformationReceipt | None = None
    if selection.pipeline_kind == "specialized":
        assert selection.pass_argument is not None
        specialized = backend_compile._run(
            [backend_compile.INTENT_OPT, selection.pass_argument], stdin=source
        )
        specialized_digest = hashlib.sha256(
            specialized.encode("utf-8")
        ).hexdigest()
        transformation_receipt = _transformation_receipt(specialized)
        if specialized_digest != selection.probe_output_sha256:
            raise RuntimeError(
                "specialized pass output changed between selection and compilation"
            )
        kernels = tuple(
            backend_compile.lower_memref_module_to_kernels(
                specialized, scf_to_cf=selection.scf_to_cf
            )
        )
    else:
        kernels = _compile_generic(intent_module, source)

    return LoweredProgram(
        selection=selection,
        kernels=kernels,
        source_mlir_sha256=source_digest,
        specialized_mlir_sha256=specialized_digest,
        transformation_receipt=transformation_receipt,
        pipeline_id=(
            "cuda.pipeline.direct-specialized.v1"
            if selection.pipeline_kind == "specialized"
            else "cuda.pipeline.linalg-materialized.v1"
        ),
        pipeline_mlir_sha256=_pipeline_mlir_digest(kernels),
    )


def lower_to_cuda_fused(
    intent_module: str | os.PathLike[str],
    *,
    tile: tuple[int, ...] = (1, 256),
    target: LoweringTarget | None = None,
    require_single_kernel: bool = True,
) -> LoweredProgram:
    """Compile a real upstream Linalg fusion+tiling state to CUDA.

    This is the existing MLIR fusion pipeline from ``compile.py`` exposed as a
    production search candidate.  It consumes the same immutable Intent module;
    no Python implementation of the kernel is generated.
    """

    source = read_intent_module(intent_module)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    selected_target = target or LoweringTarget()
    if selected_target.chip != backend_compile.CHIP:
        raise ValueError(
            "lowering target must match the configured CUDA backend: "
            f"requested {selected_target.chip!r}, configured {backend_compile.CHIP!r}"
        )
    selection = select_variant(
        source,
        target=selected_target,
        allow_specialized=False,
    )
    path = _existing_path(intent_module)
    if path is not None:
        kernels = tuple(backend_compile.compile_all_kernels_fused(path, tile))
    else:
        with tempfile.TemporaryDirectory(
            prefix="intentir-lowering-fused-"
        ) as directory:
            source_path = Path(directory) / "input.mlir"
            source_path.write_text(source, encoding="utf-8")
            kernels = tuple(
                backend_compile.compile_all_kernels_fused(source_path, tile)
            )
    if require_single_kernel and len(kernels) != 1:
        raise ValueError(
            "Linalg fusion did not collapse the complete graph to one kernel: "
            f"observed {len(kernels)} kernels"
        )
    return LoweredProgram(
        selection=selection,
        kernels=kernels,
        source_mlir_sha256=source_digest,
        pipeline_id="cuda.pipeline.linalg-elementwise-fused.v1",
        pipeline_mlir_sha256=_pipeline_mlir_digest(kernels),
    )


def lower_function_to_cuda(
    function: IntentFunction,
    concrete_extents: Mapping[str, int],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    schedule_decision: ScheduleDecision | None = None,
    allow_specialized: bool = True,
    require_specialized: bool = False,
    fold: int = 0,
) -> LoweredProgram:
    """Emit an accepted flat ``IntentFunction`` and compile it to CUDA.

    Validation policy intentionally remains outside this function.  Gate1/2,
    another verifier, or a trusted handwritten source can all use the same
    backend once they have accepted the function and supplied concrete extents.
    """

    mlir = emit_module(
        function,
        dict(concrete_extents),
        fold=fold,
    )
    return lower_to_cuda(
        mlir,
        target=target,
        pass_overrides=pass_overrides,
        only_fast_path=only_fast_path,
        schedule_decision=schedule_decision,
        allow_specialized=allow_specialized,
        require_specialized=require_specialized,
    )


def execute_cuda_artifact(
    program: LoweredProgram,
    intent_module: str | os.PathLike[str],
    inputs: Mapping[str, Any],
    *,
    output_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute already-compiled cubins without invoking the compiler again.

    ``intent_module`` supplies the logical ABI used by the host orchestrator.
    Its hash must match the source compiled into ``program``.  The compiler
    callback passed to
    :func:`intentir.backend.cuda.orchestrate.run_program` returns only the
    persisted kernels, so execution cannot silently rebuild or substitute a
    different binary.
    """

    source = read_intent_module(intent_module)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_digest != program.source_mlir_sha256:
        raise ValueError(
            "execution Intent module differs from the compiled artifact source"
        )

    path = _existing_path(intent_module)
    if path is not None:
        return backend_orchestrate.run_program(
            path,
            dict(inputs),
            compile_all=lambda _path: list(program.kernels),
            output_aliases=(
                dict(output_aliases) if output_aliases is not None else None
            ),
        )
    with tempfile.TemporaryDirectory(prefix="intentir-execution-") as directory:
        source_path = Path(directory) / "input.mlir"
        source_path.write_text(source, encoding="utf-8")
        return backend_orchestrate.run_program(
            source_path,
            dict(inputs),
            compile_all=lambda _path: list(program.kernels),
            output_aliases=(
                dict(output_aliases) if output_aliases is not None else None
            ),
        )


__all__ = [
    "LoweredProgram",
    "MLIRTransformationReceipt",
    "execute_cuda_artifact",
    "lower_function_to_cuda",
    "lower_to_cuda",
    "lower_to_cuda_fused",
]
