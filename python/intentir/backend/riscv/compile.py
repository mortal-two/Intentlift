"""True MLIR -> LLVM -> native RVV object compilation.

Python orchestrates registered C++/upstream MLIR passes and LLVM tools.  It
does not emit target kernel source.  The generated object originates from the
same deterministic Intent MLIR M0 consumed by the CUDA backend.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Any

from intentir.backend.contract_capabilities import B_CAP_READONLY_INPUTS
from intentir.backend.materialize import BackendSearchState
from intentir.backend.toolchain.tools import (
    INTENT_OPT,
    LLVM_BIN,
    MLIR_OPT,
    MLIR_TRANSLATE,
)


OPT = os.environ.get("LLVM_OPT", str(LLVM_BIN / "opt"))
LLC = os.environ.get("LLC", str(LLVM_BIN / "llc"))
TRIPLE = os.environ.get("INTENT_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")
MATTR = os.environ.get("INTENT_RISCV_MATTR", "+v,+zfh,+zvfh")

RVV_B_FUSION = "riscv.b.readonly_producer_consumer_fusion.v2"
RVV_B_FUSED_SOFTMAX = "riscv.b.readonly_fused_softmax.v1"
RVV_B_GATHER_COMPOSITION = "riscv.b.readonly_gather_composition.v1"


def _run_text(command: list[str], *, stdin: str) -> str:
    process = subprocess.run(
        command,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with exit {process.returncode}:\n"
            f"{process.stderr}"
        )
    return process.stdout


def _run_bytes(command: list[str], *, stdin: str) -> bytes:
    process = subprocess.run(
        command,
        input=stdin.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with exit {process.returncode}:\n"
            + process.stderr.decode("utf-8", errors="replace")
        )
    return process.stdout


@dataclass(frozen=True)
class RiscvTransformationState:
    fuse_producers: bool
    reassociate_reductions: bool = False
    explicit_vector_reduction: bool = False
    joint_vector_reduction: bool = False
    fused_softmax: bool = False
    rowmask_zero_fill: bool = False
    composed_indexed_gather: bool = False
    fuse_sibling_outputs: bool = False
    approximate_exp: bool = False
    mixed_precision_dot_reduction: bool = False
    paired_rotation: bool = False
    paired_rotation_head_angle_reuse: bool = False

    def __post_init__(self) -> None:
        if type(self.fuse_producers) is not bool:
            raise TypeError("fuse_producers must be bool")
        if type(self.reassociate_reductions) is not bool:
            raise TypeError("reassociate_reductions must be bool")
        if type(self.explicit_vector_reduction) is not bool:
            raise TypeError("explicit_vector_reduction must be bool")
        if type(self.joint_vector_reduction) is not bool:
            raise TypeError("joint_vector_reduction must be bool")
        if type(self.fused_softmax) is not bool:
            raise TypeError("fused_softmax must be bool")
        if type(self.rowmask_zero_fill) is not bool:
            raise TypeError("rowmask_zero_fill must be bool")
        if type(self.composed_indexed_gather) is not bool:
            raise TypeError("composed_indexed_gather must be bool")
        if type(self.fuse_sibling_outputs) is not bool:
            raise TypeError("fuse_sibling_outputs must be bool")
        if type(self.approximate_exp) is not bool:
            raise TypeError("approximate_exp must be bool")
        if type(self.mixed_precision_dot_reduction) is not bool:
            raise TypeError("mixed_precision_dot_reduction must be bool")
        if type(self.paired_rotation) is not bool:
            raise TypeError("paired_rotation must be bool")
        if type(self.paired_rotation_head_angle_reuse) is not bool:
            raise TypeError("paired_rotation_head_angle_reuse must be bool")
        if self.paired_rotation_head_angle_reuse and not self.paired_rotation:
            raise ValueError(
                "paired-rotation head-angle reuse requires paired_rotation"
            )
        if self.explicit_vector_reduction and not self.reassociate_reductions:
            raise ValueError(
                "explicit float vector reduction requires the admitted "
                "reassociation state"
            )
        if self.joint_vector_reduction and not (
            self.fuse_producers
            and self.reassociate_reductions
            and self.explicit_vector_reduction
        ):
            raise ValueError(
                "joint RVV vector reduction requires producer fusion, "
                "reassociation, and explicit vector reduction"
            )
        if self.fused_softmax and not self.reassociate_reductions:
            raise ValueError(
                "fused RVV Softmax requires the admitted reduction "
                "reassociation state"
            )
        if self.fused_softmax and (
            self.fuse_producers
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
        ):
            raise ValueError(
                "fused RVV Softmax is a complete structural state and cannot "
                "be combined with generic fusion or row-reduction states"
            )
        if self.approximate_exp and not self.fused_softmax:
            raise ValueError(
                "approximate RVV exp is only defined inside the fused Softmax state"
            )
        if self.mixed_precision_dot_reduction and not (
            self.fuse_producers and self.reassociate_reductions
        ):
            raise ValueError(
                "mixed-precision RVV dot reduction requires producer fusion "
                "and admitted reduction reassociation"
            )
        if self.mixed_precision_dot_reduction and (
            self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
            or self.fuse_sibling_outputs
        ):
            raise ValueError(
                "mixed-precision RVV dot reduction is a complete structural "
                "state and cannot be combined with another specialized state"
            )
        if self.rowmask_zero_fill and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
        ):
            raise ValueError(
                "row-mask zero-fill is a complete structural state and cannot "
                "be combined with other RVV transformations"
            )
        if self.composed_indexed_gather and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
        ):
            raise ValueError(
                "composed indexed gather is a complete structural state and "
                "cannot be combined with other RVV transformations"
            )
        if self.fuse_sibling_outputs and not self.fuse_producers:
            raise ValueError(
                "sibling-output fusion requires producer fusion within each root"
            )
        if self.fuse_sibling_outputs and (
            self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
        ):
            raise ValueError(
                "sibling-output fusion cannot be combined with a complete "
                "specialized structural state"
            )
        if self.paired_rotation and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
            or self.fuse_sibling_outputs
            or self.approximate_exp
            or self.mixed_precision_dot_reduction
        ):
            raise ValueError(
                "paired rotation is a complete direct-loop structural state"
            )

    @property
    def lowering_path_kind(self) -> str:
        if self.paired_rotation:
            return (
                "paired_rotation_head_angle_reuse"
                if self.paired_rotation_head_angle_reuse
                else "paired_rotation_contiguous"
            )
        if self.composed_indexed_gather:
            return "composed_indexed_gather"
        if self.rowmask_zero_fill:
            return "rowmask_zero_fill"
        if self.fused_softmax:
            return (
                "fused_softmax_vector_approximate_exp"
                if self.approximate_exp
                else "fused_softmax_hybrid"
            )
        if self.mixed_precision_dot_reduction:
            return "mixed_precision_dot_vector_reduction"
        if self.fuse_sibling_outputs:
            return "producer_consumer_sibling_fused"
        if self.joint_vector_reduction:
            return "producer_consumer_fused_joint_vector_reduction"
        if self.explicit_vector_reduction:
            return (
                "producer_consumer_fused_vector_reduction"
                if self.fuse_producers
                else "explicit_vector_reduction"
            )
        return "producer_consumer_fused" if self.fuse_producers else "materialized"

    @property
    def transformation_trace(self) -> tuple[str, ...]:
        structural = (
            "rvv.elementwise.paired_rotation.head_angle_reuse.v1"
            if self.paired_rotation_head_angle_reuse
            else "rvv.elementwise.paired_rotation.contiguous.v1"
            if self.paired_rotation
            else "rvv.gather.composed_indexed.v1"
            if self.composed_indexed_gather
            else "rvv.elementwise.rowmask.zero_fill.v1"
            if self.rowmask_zero_fill
            else "rvv.vector.fused_softmax_hybrid.v1"
            if self.fused_softmax
            else "rvv.linalg.producer_consumer_fusion.v2"
            if self.fuse_producers
            else "rvv.linalg.materialized.v1"
        )
        numeric = (
            ("rvv.numeric.explicit_reduction_reassociation.v1",)
            if self.reassociate_reductions
            else ()
        )
        approximation = (
            ("rvv.numeric.execution_domain_approximate_exp.v1",)
            if self.approximate_exp
            else ()
        )
        vector = (
            ("rvv.vector.explicit_row_reduction.v1",)
            if self.explicit_vector_reduction
            else ()
        )
        joint = (
            ("rvv.vector.joint_row_statistics.v1",)
            if self.joint_vector_reduction
            else ()
        )
        mixed_dot = (
            ("rvv.vector.mixed_precision_dot_reduction.v1",)
            if self.mixed_precision_dot_reduction
            else ()
        )
        sibling = (
            ("rvv.linalg.sibling_output_fusion.v1",)
            if self.fuse_sibling_outputs
            else ()
        )
        return (
            structural,
            *numeric,
            *approximation,
            *vector,
            *joint,
            *mixed_dot,
            *sibling,
        )

    @property
    def b_transformations(self) -> tuple[str, ...]:
        if self.composed_indexed_gather:
            return (RVV_B_GATHER_COMPOSITION,)
        if self.fused_softmax:
            return (RVV_B_FUSED_SOFTMAX,)
        return (RVV_B_FUSION,) if self.fuse_producers else ()

    @property
    def required_b_capabilities(self) -> tuple[str, ...]:
        return (
            (B_CAP_READONLY_INPUTS,)
            if (
                self.fuse_producers
                or self.fused_softmax
                or self.composed_indexed_gather
            )
            else ()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowering_path_kind": self.lowering_path_kind,
            "fuse_producers": self.fuse_producers,
            "reassociate_reductions": self.reassociate_reductions,
            "explicit_vector_reduction": self.explicit_vector_reduction,
            "joint_vector_reduction": self.joint_vector_reduction,
            "fused_softmax": self.fused_softmax,
            "rowmask_zero_fill": self.rowmask_zero_fill,
            "composed_indexed_gather": self.composed_indexed_gather,
            "fuse_sibling_outputs": self.fuse_sibling_outputs,
            "approximate_exp": self.approximate_exp,
            "mixed_precision_dot_reduction": self.mixed_precision_dot_reduction,
            "paired_rotation": self.paired_rotation,
            "paired_rotation_head_angle_reuse": (
                self.paired_rotation_head_angle_reuse
            ),
            "transformation_trace": list(self.transformation_trace),
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(self.required_b_capabilities),
        }


@dataclass(frozen=True)
class RiscvSchedule:
    opt_level: int
    vectorize: bool
    vector_bits: int = 128
    parallel_tile: int = 0
    reduction_tile: int = 0
    thread_count: int = 1
    scalable_vectorization: bool = False
    vector_lanes: int = 4

    def __post_init__(self) -> None:
        if self.opt_level not in {2, 3}:
            raise ValueError("RVV opt_level must be 2 or 3")
        if type(self.vectorize) is not bool:
            raise TypeError("vectorize must be bool")
        if type(self.scalable_vectorization) is not bool:
            raise TypeError("scalable_vectorization must be bool")
        if self.vector_bits not in {0, 128}:
            raise ValueError("RVV vector_bits must be 0 or native VLEN=128")
        if self.scalable_vectorization and (
            not self.vectorize or self.vector_bits != 0
        ):
            raise ValueError(
                "RVV scalable vectorization requires vectorize=true and "
                "vector_bits=0"
            )
        if self.vector_bits == 0 and not self.scalable_vectorization:
            raise ValueError(
                "RVV vector_bits=0 is reserved for scalable vectorization"
            )
        if (
            type(self.vector_lanes) is not int
            or self.vector_lanes < 1
            or self.vector_lanes > 16
            or self.vector_lanes & (self.vector_lanes - 1)
        ):
            raise ValueError("RVV vector_lanes must be a power of two in [1, 16]")
        if self.thread_count != 1:
            raise ValueError("RVV production search is single-thread only")
        for name, value in (
            ("parallel_tile", self.parallel_tile),
            ("reduction_tile", self.reduction_tile),
        ):
            if type(value) is not int:
                raise TypeError(f"RVV {name} must be int")
            if value < 0 or value > 1024:
                raise ValueError(f"RVV {name} must be in [0, 1024]")
            if value and value & (value - 1):
                raise ValueError(f"RVV {name} must be zero or a power of two")

    def to_dict(self) -> dict[str, Any]:
        return {
            "opt_level": self.opt_level,
            "vectorize": self.vectorize,
            "vector_bits": self.vector_bits,
            "vector_policy": (
                "scalable"
                if self.scalable_vectorization
                else "fixed"
                if self.vectorize
                else "scalar"
            ),
            "scalable_vectorization": self.scalable_vectorization,
            "parallel_tile": self.parallel_tile,
            "reduction_tile": self.reduction_tile,
            "thread_count": self.thread_count,
            "vector_lanes": self.vector_lanes,
        }


@dataclass(frozen=True)
class RiscvLoweringCandidate:
    transformation: RiscvTransformationState
    schedule: RiscvSchedule

    @property
    def candidate_id(self) -> str:
        schedule = self.schedule
        path = self.transformation.lowering_path_kind
        return (
            f"rvv.{path}.o{schedule.opt_level}."
            f"vec{int(schedule.vectorize)}.vlen{schedule.vector_bits}."
            f"scalable{int(schedule.scalable_vectorization)}."
            f"lanes{schedule.vector_lanes}."
            f"pt{schedule.parallel_tile}.rt{schedule.reduction_tile}."
            f"thr{schedule.thread_count}."
            f"reassoc{int(self.transformation.reassociate_reductions)}."
            f"explicitvec{int(self.transformation.explicit_vector_reduction)}."
            f"jointvec{int(self.transformation.joint_vector_reduction)}"
            f".fusedsoftmax{int(self.transformation.fused_softmax)}"
            f".approxexp{int(self.transformation.approximate_exp)}"
            f".mixeddot{int(self.transformation.mixed_precision_dot_reduction)}"
            + (".rowmaskzero1" if self.transformation.rowmask_zero_fill else "")
            + (
                ".composedgather1"
                if self.transformation.composed_indexed_gather
                else ""
            )
            + (
                ".siblingfuse1"
                if self.transformation.fuse_sibling_outputs
                else ""
            )
            + (
                ".pairedrotation1.headreuse"
                f"{int(self.transformation.paired_rotation_head_angle_reuse)}"
                if self.transformation.paired_rotation
                else ""
            )
        )

    def _content(self) -> dict[str, Any]:
        return {
            "transformation": self.transformation.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self._content()}


@dataclass(frozen=True)
class RiscvTransformationRecord:
    linalg_ops_before: int
    linalg_ops_after: int
    parallel_tile_requested: int = 0
    reduction_tile_requested: int = 0
    parallel_tile_applied: int = 0
    reduction_tile_applied: int = 0
    linalg_tiled_ops: int = 0
    generated_tile_loops: int = 0
    fused_producers: int = 0
    fusion_outer_parallel_only_ops: int = 0
    fusion_multi_use_rejections: int = 0
    openmp_parallel_ops: int = 0
    reassociated_float_combines: int = 0
    explicit_vector_reductions: int = 0
    explicit_vector_reduction_width: int = 0
    explicit_vector_reduction_loops: int = 0
    explicit_vector_reduction_scalable: int = 0
    explicit_vector_max_reductions: int = 0
    joint_vector_reduction_groups: int = 0
    joint_vector_reductions: int = 0
    joint_vector_reduction_root_loads: int = 0
    joint_raw_moment_variances: int = 0
    fused_softmax_count: int = 0
    fused_softmax_input_traversals: int = 0
    fused_softmax_full_row_traversals: int = 0
    fused_softmax_removed_materializations: int = 0
    fused_softmax_vector_reductions: int = 0
    fused_softmax_scalar_reductions: int = 0
    fused_softmax_vector_width: int = 0
    fused_softmax_scalable: int = 0
    fused_softmax_approximate_exp: int = 0
    mixed_dot_reductions: int = 0
    mixed_dot_reduction_width: int = 0
    mixed_dot_reduction_scalable: int = 0
    rowmask_lowerings: int = 0
    rowmask_zero_fills: int = 0
    rowmask_mask_hoists: int = 0
    rowmask_vector_width: int = 0
    composed_gather_lowerings: int = 0
    composed_gather_chain_depth: int = 0
    composed_gather_vector_width: int = 0
    sibling_output_fusions: int = 0
    sibling_output_eliminated_traversals: int = 0
    paired_rotation_lowerings: int = 0
    paired_rotation_head_angle_reuses: int = 0
    paired_rotation_vector_width: int = 0
    paired_rotation_scalable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "linalg_ops_before": self.linalg_ops_before,
            "linalg_ops_after": self.linalg_ops_after,
            "parallel_tile_requested": self.parallel_tile_requested,
            "reduction_tile_requested": self.reduction_tile_requested,
            "parallel_tile_applied": self.parallel_tile_applied,
            "reduction_tile_applied": self.reduction_tile_applied,
            "linalg_tiled_ops": self.linalg_tiled_ops,
            "generated_tile_loops": self.generated_tile_loops,
            "fused_producers": self.fused_producers,
            "fusion_outer_parallel_only_ops": self.fusion_outer_parallel_only_ops,
            "fusion_multi_use_rejections": self.fusion_multi_use_rejections,
            "openmp_parallel_ops": self.openmp_parallel_ops,
            "reassociated_float_combines": self.reassociated_float_combines,
            "explicit_vector_reductions": self.explicit_vector_reductions,
            "explicit_vector_reduction_width": self.explicit_vector_reduction_width,
            "explicit_vector_reduction_loops": self.explicit_vector_reduction_loops,
            "explicit_vector_reduction_scalable": (
                self.explicit_vector_reduction_scalable
            ),
            "explicit_vector_max_reductions": self.explicit_vector_max_reductions,
            "joint_vector_reduction_groups": self.joint_vector_reduction_groups,
            "joint_vector_reductions": self.joint_vector_reductions,
            "joint_vector_reduction_root_loads": (
                self.joint_vector_reduction_root_loads
            ),
            "joint_raw_moment_variances": self.joint_raw_moment_variances,
            "fused_softmax_count": self.fused_softmax_count,
            "fused_softmax_input_traversals": (
                self.fused_softmax_input_traversals
            ),
            "fused_softmax_full_row_traversals": (
                self.fused_softmax_full_row_traversals
            ),
            "fused_softmax_removed_materializations": (
                self.fused_softmax_removed_materializations
            ),
            "fused_softmax_vector_reductions": (
                self.fused_softmax_vector_reductions
            ),
            "fused_softmax_scalar_reductions": (
                self.fused_softmax_scalar_reductions
            ),
            "fused_softmax_vector_width": self.fused_softmax_vector_width,
            "fused_softmax_scalable": self.fused_softmax_scalable,
            "fused_softmax_approximate_exp": (
                self.fused_softmax_approximate_exp
            ),
            "mixed_dot_reductions": self.mixed_dot_reductions,
            "mixed_dot_reduction_width": self.mixed_dot_reduction_width,
            "mixed_dot_reduction_scalable": self.mixed_dot_reduction_scalable,
            "rowmask_lowerings": self.rowmask_lowerings,
            "rowmask_zero_fills": self.rowmask_zero_fills,
            "rowmask_mask_hoists": self.rowmask_mask_hoists,
            "rowmask_vector_width": self.rowmask_vector_width,
            "composed_gather_lowerings": self.composed_gather_lowerings,
            "composed_gather_chain_depth": self.composed_gather_chain_depth,
            "composed_gather_vector_width": self.composed_gather_vector_width,
            "sibling_output_fusions": self.sibling_output_fusions,
            "sibling_output_eliminated_traversals": (
                self.sibling_output_eliminated_traversals
            ),
            "paired_rotation_lowerings": self.paired_rotation_lowerings,
            "paired_rotation_head_angle_reuses": (
                self.paired_rotation_head_angle_reuses
            ),
            "paired_rotation_vector_width": self.paired_rotation_vector_width,
            "paired_rotation_scalable": self.paired_rotation_scalable,
        }


@dataclass(frozen=True)
class RiscvCompiledObject:
    candidate: RiscvLoweringCandidate
    symbol: str
    object_bytes: bytes
    rvv_instruction_count: int
    rvv_reduction_instruction_count: int
    scalable_vector_ir_count: int
    rvv_vsetvl_modes: tuple[str, ...]
    transformation_record: RiscvTransformationRecord

    @property
    def c_interface_symbol(self) -> str:
        return f"_mlir_ciface_{self.symbol}"

    def to_record(self) -> dict[str, Any]:
        requested_schedule = self.candidate.schedule.to_dict()
        actual_schedule = {
            **requested_schedule,
            "parallel_tile": self.transformation_record.parallel_tile_applied,
            "reduction_tile": self.transformation_record.reduction_tile_applied,
        }
        return {
            "candidate_id": self.candidate.candidate_id,
            "lowering_path_kind": (
                self.candidate.transformation.lowering_path_kind
            ),
            "transformation": self.candidate.transformation.to_dict(),
            "schedule": actual_schedule,
            "requested_schedule": requested_schedule,
            "schedule_realization": {
                "parallel_tile_applied": (
                    self.transformation_record.parallel_tile_applied
                ),
                "reduction_tile_applied": (
                    self.transformation_record.reduction_tile_applied
                ),
                "requested_equals_actual": requested_schedule == actual_schedule,
            },
            "transformation_record": self.transformation_record.to_dict(),
            "symbol": self.symbol,
            "c_interface_symbol": self.c_interface_symbol,
            "object_size_bytes": len(self.object_bytes),
            "rvv_instruction_count": self.rvv_instruction_count,
            "rvv_reduction_instruction_count": (
                self.rvv_reduction_instruction_count
            ),
            "scalable_vector_ir_count": self.scalable_vector_ir_count,
            "rvv_vsetvl_modes": list(self.rvv_vsetvl_modes),
            "true_mlir_pipeline": True,
            "python_target_kernel_emission": False,
        }


def _function_symbol(intent_mlir: str) -> str:
    match = re.search(r"func\.func\s+@([A-Za-z_][A-Za-z0-9_.$]*)", intent_mlir)
    if match is None:
        raise ValueError("Intent MLIR has no func.func symbol")
    return match.group(1)


def _module_i64_attr(mlir: str, name: str, *, default: int = 0) -> int:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*([0-9]+)\s*:\s*i64\b",
        mlir,
    )
    return int(match.group(1)) if match is not None else default


def _linalg_op_count(mlir: str) -> int:
    return len(re.findall(r"\blinalg\.[A-Za-z0-9_]+", mlir))


def _lower_to_llvm_ir(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
) -> tuple[str, str, RiscvTransformationRecord]:
    source = state.base.intent_mlir
    schedule = candidate.schedule
    if candidate.transformation.composed_indexed_gather:
        if not schedule.vectorize or schedule.scalable_vectorization:
            raise ValueError(
                "composed indexed gather requires a fixed-width vector schedule"
            )
        source = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-gather-chain-to-loops="
                    "require-lowering=true "
                    f"vector-width={schedule.vector_lanes}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=source,
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and not schedule.scalable_vectorization
    ):
        raise ValueError(
            "explicit RVV vector lowering requires a scalable-vector schedule"
        )
    linalg = _run_text(
        [
            INTENT_OPT,
            (
                "-convert-intent-to-linalg="
                "reassociate-float-reductions="
                f"{'true' if candidate.transformation.reassociate_reductions else 'false'}"
            ),
            "-expand-rsqrt-for-libm",
        ],
        stdin=source,
    )
    if candidate.transformation.fused_softmax:
        fused_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-fuse-linalg-softmax="
                    f"vector-width={schedule.vector_lanes} scalable-vector="
                    f"{'true' if schedule.scalable_vectorization else 'false'} "
                    "approximate-exp="
                    f"{'true' if candidate.transformation.approximate_exp else 'false'} "
                    "require-fusion=true"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=linalg,
        )
    elif candidate.transformation.fuse_producers:
        fused_structured = _run_text(
            [MLIR_OPT, "--linalg-fuse-elementwise-ops", "--canonicalize"],
            stdin=linalg,
        )
    else:
        fused_structured = linalg
    fusion_changed_ir = bool(
        candidate.transformation.fuse_producers
        and _linalg_op_count(fused_structured) < _linalg_op_count(linalg)
    )
    if candidate.transformation.fuse_sibling_outputs:
        before_sibling_fusion = fused_structured
        fused_structured = _run_text(
            [
                INTENT_OPT,
                "-fuse-linalg-sibling-outputs=require-fusion=true",
                "-canonicalize",
                "-cse",
            ],
            stdin=fused_structured,
        )
        fusion_changed_ir = fusion_changed_ir or (
            _linalg_op_count(fused_structured)
            < _linalg_op_count(before_sibling_fusion)
        )
    pre_schedule_structured = fused_structured
    if candidate.transformation.mixed_precision_dot_reduction:
        pre_schedule_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-vectorize-linalg-mixed-precision-dot-reduction="
                    f"vector-width={schedule.vector_lanes} "
                    "scalable-vector="
                    f"{'true' if schedule.scalable_vectorization else 'false'} "
                    "require-vectorization=true"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.joint_vector_reduction:
        pre_schedule_structured = _run_text(
            [
                MLIR_OPT,
                "--linalg-fold-unit-extent-dims",
                "--canonicalize",
                "--cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.explicit_vector_reduction:
        joint_options = (
            " joint-reductions=true require-joint-reductions=true"
            if candidate.transformation.joint_vector_reduction
            else ""
        )
        pre_schedule_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-vectorize-linalg-row-reduction="
                    f"vector-width={schedule.vector_lanes} scalable-vector=true "
                    f"require-vectorization=true{joint_options}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.joint_vector_reduction:
        before_second_fusion = pre_schedule_structured
        pre_schedule_structured = _run_text(
            [MLIR_OPT, "--linalg-fuse-elementwise-ops", "--canonicalize", "--cse"],
            stdin=pre_schedule_structured,
        )
        fusion_changed_ir = fusion_changed_ir or (
            _linalg_op_count(pre_schedule_structured)
            < _linalg_op_count(before_second_fusion)
        )
    structured = pre_schedule_structured
    if schedule.parallel_tile or schedule.reduction_tile:
        structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-tile-linalg-by-iterator-kind="
                    f"parallel-tile={schedule.parallel_tile} "
                    f"reduction-tile={schedule.reduction_tile} "
                    "fuse-producers="
                    f"{'true' if candidate.transformation.fuse_producers else 'false'}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=structured,
        )
    record = RiscvTransformationRecord(
        linalg_ops_before=_linalg_op_count(linalg),
        linalg_ops_after=_linalg_op_count(structured),
        parallel_tile_requested=schedule.parallel_tile,
        reduction_tile_requested=schedule.reduction_tile,
        parallel_tile_applied=_module_i64_attr(
            structured, "intent.schedule.parallel_tile"
        ),
        reduction_tile_applied=_module_i64_attr(
            structured, "intent.schedule.reduction_tile"
        ),
        linalg_tiled_ops=_module_i64_attr(
            structured, "intent.schedule.linalg_tiled_op_count"
        ),
        generated_tile_loops=_module_i64_attr(
            structured, "intent.schedule.generated_loop_count"
        ),
        fused_producers=max(
            _module_i64_attr(structured, "intent.schedule.fused_producer_count"),
            int(fusion_changed_ir),
        ),
        fusion_outer_parallel_only_ops=_module_i64_attr(
            structured, "intent.schedule.fusion_outer_parallel_only_count"
        ),
        fusion_multi_use_rejections=_module_i64_attr(
            structured, "intent.schedule.fusion_multi_use_rejection_count"
        ),
        reassociated_float_combines=len(
            re.findall(r"fastmath<[^>]*\breassoc\b[^>]*>", linalg)
        ),
        explicit_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_count"
        ),
        explicit_vector_reduction_width=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_width"
        ),
        explicit_vector_reduction_loops=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_loop_count"
        ),
        explicit_vector_reduction_scalable=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_scalable_count"
        ),
        explicit_vector_max_reductions=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_max_reduction_count"
        ),
        joint_vector_reduction_groups=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_group_count"
        ),
        joint_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_count"
        ),
        joint_vector_reduction_root_loads=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_root_load_count"
        ),
        joint_raw_moment_variances=_module_i64_attr(
            structured, "intent.rvv.joint_raw_moment_variance_count"
        ),
        fused_softmax_count=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_count"
        ),
        fused_softmax_input_traversals=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_input_traversal_count"
        ),
        fused_softmax_full_row_traversals=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_full_row_traversal_count"
        ),
        fused_softmax_removed_materializations=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_removed_materialization_count"
        ),
        fused_softmax_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_vector_reduction_count"
        ),
        fused_softmax_scalar_reductions=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_scalar_reduction_count"
        ),
        fused_softmax_vector_width=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_vector_width"
        ),
        fused_softmax_scalable=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_scalable_count"
        ),
        fused_softmax_approximate_exp=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_approximate_exp_count"
        ),
        mixed_dot_reductions=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_count"
        ),
        mixed_dot_reduction_width=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_width"
        ),
        mixed_dot_reduction_scalable=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_scalable_count"
        ),
        sibling_output_fusions=_module_i64_attr(
            structured, "intent.rvv.sibling_output_fusion_count"
        ),
        sibling_output_eliminated_traversals=_module_i64_attr(
            structured, "intent.rvv.sibling_output_eliminated_traversals"
        ),
    )
    if (
        candidate.transformation.reassociate_reductions
        and record.reassociated_float_combines == 0
    ):
        raise RuntimeError(
            "requested reduction reassociation matched no explicit "
            "float_nondeterministic combine"
        )
    if candidate.transformation.fuse_producers and record.fused_producers == 0:
        raise RuntimeError(
            "requested RVV producer-consumer fusion produced no fused Linalg "
            "operation"
        )
    if candidate.transformation.fuse_sibling_outputs and (
        record.sibling_output_fusions != 1
        or record.sibling_output_eliminated_traversals != 1
    ):
        raise RuntimeError(
            "requested RVV sibling-output fusion produced no valid multi-output "
            "elementwise operation"
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and record.explicit_vector_reductions == 0
    ):
        raise RuntimeError(
            "requested explicit RVV vector reduction matched no supported "
            "Linalg row reduction"
        )
    if candidate.transformation.joint_vector_reduction and (
        record.joint_vector_reduction_groups == 0
        or record.joint_vector_reductions < 2
        or record.joint_vector_reduction_root_loads
        != record.joint_vector_reduction_groups
    ):
        raise RuntimeError(
            "requested joint RVV vector reduction produced no valid "
            "same-root statistics group"
        )
    if candidate.transformation.fused_softmax and (
        record.fused_softmax_count != 1
        or record.fused_softmax_input_traversals != 2
        or record.fused_softmax_full_row_traversals != 3
        or record.fused_softmax_removed_materializations < 4
        or record.fused_softmax_vector_reductions
        != (2 if candidate.transformation.approximate_exp else 1)
        or record.fused_softmax_scalar_reductions
        != (0 if candidate.transformation.approximate_exp else 1)
        or record.fused_softmax_vector_width != schedule.vector_lanes
        or record.fused_softmax_scalable
        != int(schedule.scalable_vectorization)
        or record.fused_softmax_approximate_exp
        != int(candidate.transformation.approximate_exp)
    ):
        raise RuntimeError(
            "requested fused RVV Softmax produced no valid hybrid "
            "three-traversal vector state"
        )
    if candidate.transformation.mixed_precision_dot_reduction and (
        record.mixed_dot_reductions != 1
        or record.mixed_dot_reduction_width != schedule.vector_lanes
        or record.mixed_dot_reduction_scalable
        != int(schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested mixed-precision RVV dot produced no widening vector reduction"
        )
    parallel_runtime = schedule.thread_count > 1
    buffer_front = [
        # Intent functions expose logical tensor values.  This backend owns the
        # target ABI and the benchmark/runtime binds those values to dense
        # row-major buffers, so use the same identity-layout boundary
        # conversion as the CPU and CUDA pipelines.  Leaving the default
        # fully-dynamic strided layout prevents the downstream LLVM loop
        # vectorizer from proving unit-stride RVV accesses even though the
        # concrete buffers are contiguous.
        (
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map "
            "allow-return-allocs-from-loops=true"
        ),
        "--buffer-results-to-out-params",
        "--canonicalize",
        "--cse",
    ]
    bufferized = _run_text([MLIR_OPT, *buffer_front], stdin=structured)
    if candidate.transformation.paired_rotation:
        specialized_buffer = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-paired-rotation-to-loops="
                    "require-lowering=true head-angle-reuse="
                    f"{'true' if candidate.transformation.paired_rotation_head_angle_reuse else 'false'} "
                    "vector-width="
                    f"{candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1} "
                    "scalable-vector="
                    f"{'true' if candidate.schedule.scalable_vectorization else 'false'}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=bufferized,
        )
    elif candidate.transformation.rowmask_zero_fill:
        rowmask_vector_width = 4 if schedule.vectorize else 1
        specialized_buffer = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-row-mask-to-loops="
                    "zero-fill=true require-lowering=true "
                    f"vector-width={rowmask_vector_width}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=bufferized,
        )
    else:
        specialized_buffer = bufferized
    memref = _run_text(
        [
            MLIR_OPT,
            (
                "--convert-linalg-to-parallel-loops"
                if parallel_runtime
                else "--convert-linalg-to-loops"
            ),
            "--canonicalize",
            "--cse",
        ],
        stdin=specialized_buffer,
    )
    record = replace(
        record,
        linalg_ops_after=_linalg_op_count(specialized_buffer),
        rowmask_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_lowering_count"
        ),
        rowmask_zero_fills=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_zero_fill_count"
        ),
        rowmask_mask_hoists=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_mask_hoist_count"
        ),
        rowmask_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_vector_width"
        ),
        composed_gather_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_count"
        ),
        composed_gather_chain_depth=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_chain_depth"
        ),
        composed_gather_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_vector_width"
        ),
        paired_rotation_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_lowering_count"
        ),
        paired_rotation_head_angle_reuses=_module_i64_attr(
            specialized_buffer,
            "intent.rvv.paired_rotation_head_angle_reuse_count",
        ),
        paired_rotation_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_vector_width"
        ),
        paired_rotation_scalable=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_scalable_count"
        ),
    )
    if candidate.transformation.paired_rotation and (
        record.paired_rotation_lowerings != 1
        or record.paired_rotation_head_angle_reuses
        != int(candidate.transformation.paired_rotation_head_angle_reuse)
        or record.paired_rotation_vector_width
        != (candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1)
        or record.paired_rotation_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested RVV paired rotation produced no exact direct-loop state"
        )
    if candidate.transformation.rowmask_zero_fill and (
        record.rowmask_lowerings != 1
        or record.rowmask_zero_fills != 1
        or record.rowmask_mask_hoists != 1
        or record.rowmask_vector_width != (4 if schedule.vectorize else 1)
    ):
        raise RuntimeError(
            "requested RVV row-mask zero-fill produced no valid hoisted "
            "typed-broadcast loop"
        )
    if candidate.transformation.composed_indexed_gather and (
        record.composed_gather_lowerings != 1
        or record.composed_gather_chain_depth != 2
        or record.composed_gather_vector_width != schedule.vector_lanes
    ):
        raise RuntimeError(
            "requested composed RVV gather produced no valid direct indexed "
            "vector state"
        )
    forwarded = _run_text(
        [
            INTENT_OPT,
            "-forward-buffer-results-to-out-params",
            "-lower-disjoint-memref-copy-to-loops",
        ],
        stdin=memref,
    )
    cleaned = _run_text(
        [MLIR_OPT, "--buffer-deallocation-pipeline", "--canonicalize", "--cse"],
        stdin=forwarded,
    )
    parallelized = (
        _run_text(
            [MLIR_OPT, "--convert-scf-to-openmp", "--canonicalize", "--cse"],
            stdin=cleaned,
        )
        if parallel_runtime
        else cleaned
    )
    record = replace(
        record,
        openmp_parallel_ops=len(re.findall(r"\bomp\.parallel\b", parallelized)),
    )
    terminal_pipeline = [
        MLIR_OPT,
        "--llvm-request-c-wrappers",
    ]
    if parallel_runtime:
        terminal_pipeline.append("--convert-openmp-to-llvm")
    terminal_pipeline.extend(
        [
            "--expand-strided-metadata",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-math-to-libm",
            *(
                ["--convert-vector-to-llvm"]
                if (
                    candidate.transformation.explicit_vector_reduction
                    or candidate.transformation.fused_softmax
                    or candidate.transformation.mixed_precision_dot_reduction
                    or (
                        candidate.transformation.rowmask_zero_fill
                        and schedule.vectorize
                    )
                    or candidate.transformation.composed_indexed_gather
                    or (
                        candidate.transformation.paired_rotation
                        and schedule.vectorize
                    )
                )
                else []
            ),
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        ]
    )
    lowered = _run_text(
        terminal_pipeline,
        stdin=parallelized,
    )
    if re.search(
        r"^\s*(?:%[^=]+\s*=\s*)?(?:intent|linalg)\.",
        lowered,
        flags=re.MULTILINE,
    ):
        raise RuntimeError("RVV lowering left high-level Intent/Linalg ops")
    llvm_ir = _run_text(
        [MLIR_TRANSLATE, "--mlir-to-llvmir"],
        stdin=lowered,
    )
    return lowered, llvm_ir, record


def compile_riscv_object(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
    *,
    capabilities: tuple[str, ...],
) -> RiscvCompiledObject:
    """Compile one already-admitted RVV search candidate to a PIC object."""

    if state.target.family != "riscv":
        raise ValueError("RVV compiler requires a riscv BackendTarget")
    if not isinstance(candidate, RiscvLoweringCandidate):
        raise TypeError("RVV compiler requires RiscvLoweringCandidate")
    missing = set(candidate.transformation.required_b_capabilities) - set(
        capabilities
    )
    if missing:
        raise ValueError(
            "RVV candidate lacks required B capabilities: "
            + ", ".join(sorted(missing))
        )
    lowered, raw_llvm_ir, transformation_record = _lower_to_llvm_ir(
        state, candidate
    )
    # Keep the production entrypoint fail-closed even when an alternate caller
    # supplies a pre-lowered module.  A requested structural/numeric state is
    # not a distinct candidate unless the compiler record proves it happened.
    if (
        candidate.transformation.reassociate_reductions
        and transformation_record.reassociated_float_combines == 0
    ):
        raise RuntimeError(
            "requested reduction reassociation matched no explicit "
            "float_nondeterministic combine"
        )
    if (
        candidate.transformation.fuse_producers
        and transformation_record.fused_producers == 0
    ):
        raise RuntimeError(
            "requested RVV producer-consumer fusion produced no fused Linalg "
            "operation"
        )
    if candidate.transformation.fuse_sibling_outputs and (
        transformation_record.sibling_output_fusions != 1
        or transformation_record.sibling_output_eliminated_traversals != 1
    ):
        raise RuntimeError(
            "requested RVV sibling-output fusion produced no multi-output "
            "elementwise state"
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and transformation_record.explicit_vector_reductions == 0
    ):
        raise RuntimeError(
            "requested explicit RVV vector reduction produced no vectorized "
            "row reduction"
        )
    if candidate.transformation.joint_vector_reduction and (
        transformation_record.joint_vector_reduction_groups == 0
        or transformation_record.joint_vector_reductions < 2
        or transformation_record.joint_vector_reduction_root_loads
        != transformation_record.joint_vector_reduction_groups
    ):
        raise RuntimeError(
            "requested joint RVV vector reduction produced no valid "
            "same-root statistics group"
        )
    if candidate.transformation.fused_softmax and (
        transformation_record.fused_softmax_count != 1
        or transformation_record.fused_softmax_input_traversals != 2
        or transformation_record.fused_softmax_full_row_traversals != 3
        or transformation_record.fused_softmax_removed_materializations < 4
        or transformation_record.fused_softmax_vector_reductions
        != (2 if candidate.transformation.approximate_exp else 1)
        or transformation_record.fused_softmax_scalar_reductions
        != (0 if candidate.transformation.approximate_exp else 1)
        or transformation_record.fused_softmax_vector_width
        != candidate.schedule.vector_lanes
        or transformation_record.fused_softmax_scalable
        != int(candidate.schedule.scalable_vectorization)
        or transformation_record.fused_softmax_approximate_exp
        != int(candidate.transformation.approximate_exp)
    ):
        raise RuntimeError(
            "requested fused RVV Softmax produced no valid hybrid "
            "three-traversal vector state"
        )
    if candidate.transformation.mixed_precision_dot_reduction and (
        transformation_record.mixed_dot_reductions != 1
        or transformation_record.mixed_dot_reduction_width
        != candidate.schedule.vector_lanes
        or transformation_record.mixed_dot_reduction_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested mixed-precision RVV dot produced no widening vector reduction"
        )
    if candidate.transformation.rowmask_zero_fill and (
        transformation_record.rowmask_lowerings != 1
        or transformation_record.rowmask_zero_fills != 1
        or transformation_record.rowmask_mask_hoists != 1
        or transformation_record.rowmask_vector_width
        != (4 if candidate.schedule.vectorize else 1)
    ):
        raise RuntimeError(
            "requested RVV row-mask zero-fill produced no valid hoisted "
            "typed-broadcast loop"
        )
    if candidate.transformation.composed_indexed_gather and (
        transformation_record.composed_gather_lowerings != 1
        or transformation_record.composed_gather_chain_depth != 2
        or transformation_record.composed_gather_vector_width
        != candidate.schedule.vector_lanes
    ):
        raise RuntimeError(
            "requested composed RVV gather produced no direct indexed vector "
            "state"
        )
    if candidate.transformation.paired_rotation and (
        transformation_record.paired_rotation_lowerings != 1
        or transformation_record.paired_rotation_head_angle_reuses
        != int(candidate.transformation.paired_rotation_head_angle_reuse)
        or transformation_record.paired_rotation_vector_width
        != (candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1)
        or transformation_record.paired_rotation_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested RVV paired rotation produced no exact direct-loop state"
        )
    schedule = candidate.schedule
    vector_shape_flags = (
        [
            "--riscv-v-vector-bits-min=-1",
            "--riscv-v-vector-bits-max=0",
            "--scalable-vectorization=preferred",
        ]
        if schedule.scalable_vectorization
        else [
            f"--riscv-v-vector-bits-min={schedule.vector_bits}",
            f"--riscv-v-vector-bits-max={schedule.vector_bits}",
            "--scalable-vectorization=off",
        ]
    )
    opt_command = [
        OPT,
        f"-O{schedule.opt_level}",
        "-S",
        f"-mtriple={TRIPLE}",
        f"-mattr={MATTR}",
        *vector_shape_flags,
        f"-vectorize-loops={'true' if schedule.vectorize else 'false'}",
        f"-vectorize-slp={'true' if schedule.vectorize else 'false'}",
        "-o",
        "-",
    ]
    optimized = _run_text(opt_command, stdin=raw_llvm_ir)
    llc_common = [
        LLC,
        f"-mtriple={TRIPLE}",
        f"-mattr={MATTR}",
        *vector_shape_flags,
        f"-O{schedule.opt_level}",
        "-relocation-model=pic",
    ]
    assembly = _run_text([*llc_common, "-filetype=asm", "-o", "-"], stdin=optimized)
    object_bytes = _run_bytes(
        [*llc_common, "-filetype=obj", "-o", "-"],
        stdin=optimized,
    )
    rvv_instruction_count = len(
        re.findall(
            r"^\s+v(?:set|l|s|f|m|n|w|r)[a-z0-9_.]*\s",
            assembly,
            flags=re.MULTILINE,
        )
    )
    rvv_reduction_instruction_count = len(
        re.findall(
            r"^\s+vfred[a-z0-9_.]*\s",
            assembly,
            flags=re.MULTILINE,
        )
    )
    if (
        (
            candidate.transformation.explicit_vector_reduction
            and rvv_reduction_instruction_count == 0
        )
        or (
            candidate.transformation.fused_softmax
            and rvv_reduction_instruction_count < 1
        )
        or (
            candidate.transformation.mixed_precision_dot_reduction
            and rvv_reduction_instruction_count < 1
        )
        or (
            candidate.transformation.paired_rotation
            and schedule.vectorize
            and rvv_instruction_count == 0
        )
    ):
        raise RuntimeError(
            "requested explicit RVV vector state produced insufficient RVV "
            "target instructions"
        )
    scalable_vector_ir_count = len(re.findall(r"<vscale\s+x\s+", optimized))
    if schedule.scalable_vectorization and scalable_vector_ir_count == 0:
        raise RuntimeError(
            "requested scalable RVV vectorization produced no scalable-vector "
            "LLVM IR"
        )
    rvv_vsetvl_modes = tuple(
        sorted(
            {
                match.group(1).replace(" ", "")
                for match in re.finditer(
                    r"^\s+vset(?:i?vli|vl)\s+[^\n]*?\b(e(?:8|16|32|64)\s*,\s*m(?:f[248]|[1248]))\b",
                    assembly,
                    flags=re.MULTILINE,
                )
            }
        )
    )
    return RiscvCompiledObject(
        candidate=candidate,
        symbol=_function_symbol(state.base.intent_mlir),
        object_bytes=object_bytes,
        rvv_instruction_count=rvv_instruction_count,
        rvv_reduction_instruction_count=rvv_reduction_instruction_count,
        scalable_vector_ir_count=scalable_vector_ir_count,
        rvv_vsetvl_modes=rvv_vsetvl_modes,
        transformation_record=transformation_record,
    )


__all__ = [
    "MATTR",
    "RVV_B_FUSED_SOFTMAX",
    "RVV_B_FUSION",
    "RiscvCompiledObject",
    "RiscvLoweringCandidate",
    "RiscvSchedule",
    "RiscvTransformationRecord",
    "RiscvTransformationState",
    "TRIPLE",
    "compile_riscv_object",
]
