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

RVV_B_FUSION = "riscv.b.readonly_whole_graph_fusion.v1"


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
class RiscvSchedule:
    fuse_elementwise: bool
    opt_level: int
    vectorize: bool
    vector_bits: int = 128

    def __post_init__(self) -> None:
        if type(self.fuse_elementwise) is not bool:
            raise TypeError("fuse_elementwise must be bool")
        if self.opt_level not in {2, 3}:
            raise ValueError("RVV opt_level must be 2 or 3")
        if type(self.vectorize) is not bool:
            raise TypeError("vectorize must be bool")
        if self.vector_bits not in {0, 128}:
            raise ValueError("RVV vector_bits must be 0 or native VLEN=128")

    @property
    def b_transformations(self) -> tuple[str, ...]:
        return (RVV_B_FUSION,) if self.fuse_elementwise else ()

    @property
    def required_b_capabilities(self) -> tuple[str, ...]:
        return (B_CAP_READONLY_INPUTS,) if self.fuse_elementwise else ()

    @property
    def candidate_id(self) -> str:
        return "rvv." + content_digest(self.to_dict())[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fuse_elementwise": self.fuse_elementwise,
            "opt_level": self.opt_level,
            "vectorize": self.vectorize,
            "vector_bits": self.vector_bits,
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(
                self.required_b_capabilities
            ),
        }


@dataclass(frozen=True)
class RiscvCompiledObject:
    schedule: RiscvSchedule
    symbol: str
    object_bytes: bytes
    source_mlir_sha256: str
    lowered_mlir_sha256: str
    llvm_ir_sha256: str
    assembly_sha256: str
    rvv_instruction_count: int

    @property
    def c_interface_symbol(self) -> str:
        return f"_mlir_ciface_{self.symbol}"

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.schedule.candidate_id,
            "schedule": self.schedule.to_dict(),
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


def _lower_to_llvm_ir(
    state: BackendSearchState,
    schedule: RiscvSchedule,
) -> tuple[str, str]:
    source = state.base.intent_mlir
    linalg = _run_text(
        [INTENT_OPT, "-convert-intent-to-linalg", "-expand-rsqrt-for-libm"],
        stdin=source,
    )
    front = []
    if schedule.fuse_elementwise:
        front.append("--linalg-fuse-elementwise-ops")
    front.extend(
        [
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--buffer-results-to-out-params",
            "--convert-linalg-to-loops",
            "--canonicalize",
            "--cse",
        ]
    )
    memref = _run_text([MLIR_OPT, *front], stdin=linalg)
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
    return lowered, llvm_ir


def compile_riscv_object(
    state: BackendSearchState,
    schedule: RiscvSchedule,
    *,
    capabilities: tuple[str, ...],
) -> RiscvCompiledObject:
    """Compile one already-admitted RVV search candidate to a PIC object."""

    if state.target.family != "riscv":
        raise ValueError("RVV compiler requires a riscv BackendTarget")
    missing = set(schedule.required_b_capabilities) - set(capabilities)
    if missing:
        raise ValueError(
            "RVV candidate lacks required B capabilities: "
            + ", ".join(sorted(missing))
        )
    lowered, raw_llvm_ir = _lower_to_llvm_ir(state, schedule)
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
        schedule=schedule,
        symbol=_function_symbol(state.base.intent_mlir),
        object_bytes=object_bytes,
        source_mlir_sha256=state.base.mlir_digest,
        lowered_mlir_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
        llvm_ir_sha256=hashlib.sha256(optimized.encode()).hexdigest(),
        assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
        rvv_instruction_count=rvv_instruction_count,
    )


__all__ = [
    "MATTR",
    "RVV_B_FUSION",
    "RiscvCompiledObject",
    "RiscvSchedule",
    "TRIPLE",
    "compile_riscv_object",
]
