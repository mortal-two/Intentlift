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
MATTR = os.environ.get("INTENT_RISCV_MATTR", "+v,+zfh")

RVV_B_FUSION = "riscv.b.readonly_producer_consumer_fusion.v2"


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

    def __post_init__(self) -> None:
        if type(self.fuse_producers) is not bool:
            raise TypeError("fuse_producers must be bool")
        if type(self.reassociate_reductions) is not bool:
            raise TypeError("reassociate_reductions must be bool")

    @property
    def lowering_path_kind(self) -> str:
        return (
            "producer_consumer_fused"
            if self.fuse_producers
            else "materialized"
        )

    @property
    def transformation_trace(self) -> tuple[str, ...]:
        structural = (
            "rvv.linalg.producer_consumer_fusion.v2"
            if self.fuse_producers
            else "rvv.linalg.materialized.v1"
        )
        numeric = (
            ("rvv.numeric.explicit_reduction_reassociation.v1",)
            if self.reassociate_reductions
            else ()
        )
        return (structural, *numeric)

    @property
    def b_transformations(self) -> tuple[str, ...]:
        return (RVV_B_FUSION,) if self.fuse_producers else ()

    @property
    def required_b_capabilities(self) -> tuple[str, ...]:
        return (B_CAP_READONLY_INPUTS,) if self.fuse_producers else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowering_path_kind": self.lowering_path_kind,
            "fuse_producers": self.fuse_producers,
            "reassociate_reductions": self.reassociate_reductions,
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

    def __post_init__(self) -> None:
        if self.opt_level not in {2, 3}:
            raise ValueError("RVV opt_level must be 2 or 3")
        if type(self.vectorize) is not bool:
            raise TypeError("vectorize must be bool")
        if self.vector_bits not in {0, 128}:
            raise ValueError("RVV vector_bits must be 0 or native VLEN=128")
        if self.thread_count not in {1, 16}:
            raise ValueError("RVV thread_count must be 1 or 16")
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
            "parallel_tile": self.parallel_tile,
            "reduction_tile": self.reduction_tile,
            "thread_count": self.thread_count,
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
            f"pt{schedule.parallel_tile}.rt{schedule.reduction_tile}."
            f"thr{schedule.thread_count}."
            f"reassoc{int(self.transformation.reassociate_reductions)}"
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
        }


@dataclass(frozen=True)
class RiscvCompiledObject:
    candidate: RiscvLoweringCandidate
    symbol: str
    object_bytes: bytes
    rvv_instruction_count: int
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


def _lower_to_llvm_ir(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
) -> tuple[str, str, RiscvTransformationRecord]:
    source = state.base.intent_mlir
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
    if candidate.transformation.fuse_producers:
        pre_schedule_structured = _run_text(
            [MLIR_OPT, "--linalg-fuse-elementwise-ops", "--canonicalize"],
            stdin=linalg,
        )
    else:
        pre_schedule_structured = linalg
    structured = pre_schedule_structured
    schedule = candidate.schedule
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
        linalg_ops_before=len(re.findall(r"\blinalg\.[A-Za-z0-9_]+", linalg)),
        linalg_ops_after=len(
            re.findall(r"\blinalg\.[A-Za-z0-9_]+", structured)
        ),
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
        fused_producers=_module_i64_attr(
            structured, "intent.schedule.fused_producer_count"
        ),
        fusion_outer_parallel_only_ops=_module_i64_attr(
            structured, "intent.schedule.fusion_outer_parallel_only_count"
        ),
        fusion_multi_use_rejections=_module_i64_attr(
            structured, "intent.schedule.fusion_multi_use_rejection_count"
        ),
        reassociated_float_combines=len(
            re.findall(r"fastmath<[^>]*\breassoc\b[^>]*>", structured)
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
    parallel_runtime = schedule.thread_count > 1
    front = [
        # Intent functions expose logical tensor values.  This backend owns the
        # target ABI and the benchmark/runtime binds those values to dense
        # row-major buffers, so use the same identity-layout boundary
        # conversion as the CPU and CUDA pipelines.  Leaving the default
        # fully-dynamic strided layout prevents the downstream LLVM loop
        # vectorizer from proving unit-stride RVV accesses even though the
        # concrete buffers are contiguous.
        (
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map"
        ),
        "--buffer-results-to-out-params",
        (
            "--convert-linalg-to-parallel-loops"
            if parallel_runtime
            else "--convert-linalg-to-loops"
        ),
        "--canonicalize",
        "--cse",
    ]
    memref = _run_text([MLIR_OPT, *front], stdin=structured)
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
    schedule = candidate.schedule
    opt_command = [
        OPT,
        f"-O{schedule.opt_level}",
        "-S",
        f"-mtriple={TRIPLE}",
        f"-mattr={MATTR}",
        f"--riscv-v-vector-bits-min={schedule.vector_bits}",
        f"--riscv-v-vector-bits-max={schedule.vector_bits}",
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
        f"--riscv-v-vector-bits-min={schedule.vector_bits}",
        f"--riscv-v-vector-bits-max={schedule.vector_bits}",
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
    return RiscvCompiledObject(
        candidate=candidate,
        symbol=_function_symbol(state.base.intent_mlir),
        object_bytes=object_bytes,
        rvv_instruction_count=rvv_instruction_count,
        transformation_record=transformation_record,
    )


__all__ = [
    "MATTR",
    "RVV_B_FUSION",
    "RiscvCompiledObject",
    "RiscvLoweringCandidate",
    "RiscvSchedule",
    "RiscvTransformationRecord",
    "RiscvTransformationState",
    "TRIPLE",
    "compile_riscv_object",
]
