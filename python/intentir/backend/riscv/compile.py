"""True MLIR -> LLVM -> native RVV object compilation.

Python orchestrates registered C++/upstream MLIR passes and LLVM tools.  It
does not emit target kernel source.  The generated object originates from the
same deterministic Intent MLIR M0 consumed by the CUDA backend.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from intentir.backend.contract_capabilities import B_CAP_READONLY_INPUTS
from intentir.backend.materialize import BackendSearchState
from intentir.backend.toolchain.tools import (
    INTENT_OPT,
    LLVM_BIN,
    MLIR_OPT,
    MLIR_TRANSLATE,
)
from intentir.ir.production_snapshot import content_digest


OPT = os.environ.get("LLVM_OPT", str(LLVM_BIN / "opt"))
LLC = os.environ.get("LLC", str(LLVM_BIN / "llc"))
TRIPLE = os.environ.get("INTENT_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")
MATTR = os.environ.get("INTENT_RISCV_MATTR", "+v,+zfh")

RVV_B_FUSION = "riscv.b.readonly_producer_consumer_fusion.v2"


def _sha_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolchain_fingerprint() -> str:
    files = {
        "intent_opt": str(INTENT_OPT),
        "mlir_opt": str(MLIR_OPT),
        "mlir_translate": str(MLIR_TRANSLATE),
        "opt": str(OPT),
        "llc": str(LLC),
    }
    return content_digest(
        {
            "target": {"triple": TRIPLE, "mattr": MATTR},
            "files": {
                name: {"path": path, "sha256": _sha_file(path)}
                for name, path in files.items()
            },
        }
    )


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

    def __post_init__(self) -> None:
        if type(self.fuse_producers) is not bool:
            raise TypeError("fuse_producers must be bool")

    @property
    def lowering_path_kind(self) -> str:
        return (
            "producer_consumer_fused"
            if self.fuse_producers
            else "materialized"
        )

    @property
    def transformation_trace(self) -> tuple[str, ...]:
        return (
            ("rvv.linalg.producer_consumer_fusion.v2",)
            if self.fuse_producers
            else ("rvv.linalg.materialized.v1",)
        )

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

    def __post_init__(self) -> None:
        if self.opt_level not in {2, 3}:
            raise ValueError("RVV opt_level must be 2 or 3")
        if type(self.vectorize) is not bool:
            raise TypeError("vectorize must be bool")
        if self.vector_bits not in {0, 128}:
            raise ValueError("RVV vector_bits must be 0 or native VLEN=128")
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
        }


@dataclass(frozen=True)
class RiscvLoweringCandidate:
    transformation: RiscvTransformationState
    schedule: RiscvSchedule

    @property
    def candidate_id(self) -> str:
        return "rvv.state-request." + content_digest(self._content())[:16]

    def _content(self) -> dict[str, Any]:
        return {
            "transformation": self.transformation.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self._content()}


@dataclass(frozen=True)
class RiscvTransformationReceipt:
    linalg_before_sha256: str
    structured_after_sha256: str
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
    pre_schedule_structured_sha256: str = ""

    @property
    def topology_digest(self) -> str:
        # This digest identifies the realized lowering path after erasing tile,
        # vectorization and optimization parameters.  The concrete post-pass
        # state remains identified separately by structured_after_sha256 and
        # the final object SHA-256.
        return content_digest(
            {
                "linalg_before_sha256": self.linalg_before_sha256,
                "pre_schedule_structured_sha256": (
                    self.pre_schedule_structured_sha256
                    or self.linalg_before_sha256
                ),
                "tile_and_fuse_realized": self.fused_producers > 0,
                "fused_producers": self.fused_producers,
                "fusion_outer_parallel_only_ops": (
                    self.fusion_outer_parallel_only_ops
                ),
                "fusion_multi_use_rejections": self.fusion_multi_use_rejections,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "linalg_before_sha256": self.linalg_before_sha256,
            "structured_after_sha256": self.structured_after_sha256,
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
            "pre_schedule_structured_sha256": (
                self.pre_schedule_structured_sha256
                or self.linalg_before_sha256
            ),
            "topology_digest_semantics": "lowering_path_after_erasing_schedule",
        }


@dataclass(frozen=True)
class RiscvCompiledObject:
    candidate: RiscvLoweringCandidate
    symbol: str
    object_bytes: bytes
    source_mlir_sha256: str
    lowered_mlir_sha256: str
    llvm_ir_sha256: str
    assembly_sha256: str
    rvv_instruction_count: int
    transformation_receipt: RiscvTransformationReceipt

    @property
    def c_interface_symbol(self) -> str:
        return f"_mlir_ciface_{self.symbol}"

    def to_record(self) -> dict[str, Any]:
        requested_schedule = self.candidate.schedule.to_dict()
        actual_schedule = {
            **requested_schedule,
            "parallel_tile": self.transformation_receipt.parallel_tile_applied,
            "reduction_tile": self.transformation_receipt.reduction_tile_applied,
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
                    self.transformation_receipt.parallel_tile_applied
                ),
                "reduction_tile_applied": (
                    self.transformation_receipt.reduction_tile_applied
                ),
                "requested_equals_actual": requested_schedule == actual_schedule,
            },
            "transformation_receipt": self.transformation_receipt.to_dict(),
            "topology_digest": self.transformation_receipt.topology_digest,
            "symbol": self.symbol,
            "c_interface_symbol": self.c_interface_symbol,
            "object_size_bytes": len(self.object_bytes),
            "object_sha256": hashlib.sha256(self.object_bytes).hexdigest(),
            "source_mlir_sha256": self.source_mlir_sha256,
            "lowered_mlir_sha256": self.lowered_mlir_sha256,
            "llvm_ir_sha256": self.llvm_ir_sha256,
            "assembly_sha256": self.assembly_sha256,
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
) -> tuple[str, str, RiscvTransformationReceipt]:
    source = state.base.intent_mlir
    linalg = _run_text(
        [INTENT_OPT, "-convert-intent-to-linalg", "-expand-rsqrt-for-libm"],
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
    receipt = RiscvTransformationReceipt(
        linalg_before_sha256=hashlib.sha256(linalg.encode()).hexdigest(),
        structured_after_sha256=hashlib.sha256(structured.encode()).hexdigest(),
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
        pre_schedule_structured_sha256=hashlib.sha256(
            pre_schedule_structured.encode()
        ).hexdigest(),
    )
    front = [
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--buffer-results-to-out-params",
        "--convert-linalg-to-loops",
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
    lowered = _run_text(
        [
            MLIR_OPT,
            "--llvm-request-c-wrappers",
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
        ],
        stdin=cleaned,
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
    return lowered, llvm_ir, receipt


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
    lowered, raw_llvm_ir, transformation_receipt = _lower_to_llvm_ir(
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
        source_mlir_sha256=state.base.mlir_digest,
        lowered_mlir_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
        llvm_ir_sha256=hashlib.sha256(optimized.encode()).hexdigest(),
        assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
        rvv_instruction_count=rvv_instruction_count,
        transformation_receipt=transformation_receipt,
    )


__all__ = [
    "MATTR",
    "RVV_B_FUSION",
    "RiscvCompiledObject",
    "RiscvLoweringCandidate",
    "RiscvSchedule",
    "RiscvTransformationReceipt",
    "RiscvTransformationState",
    "TRIPLE",
    "compile_riscv_object",
    "toolchain_fingerprint",
]
