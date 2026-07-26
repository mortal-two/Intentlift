"""Prepared no-sync launcher for already-compiled single-launch candidates.

This is the measured-search adapter over the existing compiler/orchestrator.
It deliberately does not compile, allocate, synchronize, or decide candidate
legality inside the timed closure.  ``triton.testing.do_bench`` (or another
event-based timer) therefore measures the same launch interval as a Triton
baseline.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.backend.cuda import orchestrate
from intentir.backend.cuda.launch import load_module
from intentir.backend.cuda.pipeline import LoweredProgram
from intentir.backend.cuda.selection import read_intent_module


EXTRACTED_LAUNCH_PROTOCOL = "intentir.backend.cuda.extracted-launch.v1"


@dataclass(frozen=True)
class PreparedLaunchRecord:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int
    argument_count: int
    tensor_argument_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "block": list(self.block),
            "dynamic_smem_bytes": self.dynamic_smem_bytes,
            "argument_count": self.argument_count,
            "tensor_argument_count": self.tensor_argument_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreparedLaunchRecord":
        return cls(
            grid=tuple(int(item) for item in value["grid"]),
            block=tuple(int(item) for item in value["block"]),
            dynamic_smem_bytes=int(value["dynamic_smem_bytes"]),
            argument_count=int(value["argument_count"]),
            tensor_argument_count=int(value["tensor_argument_count"]),
        )


@dataclass(frozen=True)
class ExtractedSingleLaunch:
    """Portable cubin plus the launch ABI extracted from compiler host IR."""

    cubin: bytes
    kernel_name: str
    binary_name: str
    plan: tuple[tuple[Any, ...], ...]
    record: PreparedLaunchRecord
    source_mlir_sha256: str

    def __post_init__(self) -> None:
        if not self.cubin.startswith(b"\x7fELF"):
            raise ValueError("extracted launch cubin is not ELF")
        if not self.kernel_name or not self.binary_name:
            raise ValueError("extracted launch requires compiler symbols")
        if len(self.plan) != self.record.argument_count:
            raise ValueError("extracted launch plan length mismatch")
        if len(self.source_mlir_sha256) != 64:
            raise ValueError("extracted launch source digest is invalid")

    @property
    def cubin_sha256(self) -> str:
        return hashlib.sha256(self.cubin).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EXTRACTED_LAUNCH_PROTOCOL,
            "kernel_name": self.kernel_name,
            "binary_name": self.binary_name,
            "source_mlir_sha256": self.source_mlir_sha256,
            "cubin_sha256": self.cubin_sha256,
            "cubin_base64": base64.b64encode(self.cubin).decode("ascii"),
            "plan": [list(item) for item in self.plan],
            "launch": self.record.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractedSingleLaunch":
        if value.get("protocol") != EXTRACTED_LAUNCH_PROTOCOL:
            raise ValueError("unsupported extracted launch protocol")
        cubin = base64.b64decode(value["cubin_base64"], validate=True)
        result = cls(
            cubin=cubin,
            kernel_name=str(value["kernel_name"]),
            binary_name=str(value["binary_name"]),
            plan=tuple(tuple(item) for item in value["plan"]),
            record=PreparedLaunchRecord.from_dict(value["launch"]),
            source_mlir_sha256=str(value["source_mlir_sha256"]),
        )
        if value.get("cubin_sha256") != result.cubin_sha256:
            raise ValueError("extracted launch cubin digest mismatch")
        return result


def extract_single_launch(
    program: LoweredProgram,
    intent_module: str,
) -> ExtractedSingleLaunch:
    """Extract a target-portable launch bundle without loading its cubin."""

    if program.selection.pipeline_kind != "specialized":
        raise ValueError("prepared benchmark requires a specialized candidate")
    if len(program.kernels) != 1:
        raise ValueError("prepared benchmark requires exactly one compiled kernel")
    source = read_intent_module(intent_module)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_digest != program.source_mlir_sha256:
        raise ValueError("launch extraction source differs from compiled MLIR")
    inputs = orchestrate._parse_intent_signature(source)
    results = orchestrate._parse_intent_results(source)
    inplace = orchestrate._parse_intent_inplace(source, len(results))
    fresh_outputs = sum(
        1 for item in (inplace or [-1] * len(results)) if item < 0
    )
    tensor_count = len(inputs) + fresh_outputs

    kernel = program.kernels[0]
    _name, host = orchestrate._host_func_text(kernel.binary_mlir)
    _func, _ordered, pointer_blockargs = (
        orchestrate._parse_signature_blockargs(host)
    )
    aligned = orchestrate._map_input_aligned_ptrs(pointer_blockargs)
    constants = orchestrate._parse_constants(host)
    scalar_inputs: dict[str, tuple[int, str]] = {}
    for match in re.finditer(
        r"(%\w+)\s*=\s*llvm\.load\s+(%\w+)\s*:\s*!llvm\.ptr\s*->\s*"
        r"(i32|i64|f16|bf16|f32|f64)",
        host,
    ):
        result_ssa, pointer_ssa, scalar_type = match.groups()
        if pointer_ssa in aligned:
            scalar_inputs[result_ssa] = (aligned[pointer_ssa], scalar_type)
    sites = orchestrate._parse_launches(host)
    if len(sites) != 1:
        raise ValueError(
            "prepared benchmark requires one compiler-emitted launch site"
        )
    site = sites[0]
    if (
        site.binary_module != kernel.binary_name
        or site.func_symbol != kernel.kernel_name
    ):
        raise ValueError("compiled cubin and host launch symbols disagree")

    class _ConstantView:
        pass

    constant_view = _ConstantView()
    constant_view.constants = constants
    grid = tuple(
        orchestrate._resolve_const_index(constant_view, item)
        for item in site.grid_ssa
    )
    block = tuple(
        orchestrate._resolve_const_index(constant_view, item)
        for item in site.block_ssa
    )
    shared = (
        int(constants[site.dynamic_shared_ssa][0])
        if site.dynamic_shared_ssa is not None
        else 0
    )

    plan: list[tuple[Any, ...]] = []
    for operand in site.args:
        if operand.ty.startswith("!llvm.ptr"):
            if operand.ssa not in aligned:
                raise ValueError(
                    "single-launch candidate captures a non-ABI pointer"
                )
            plan.append(("ptr", aligned[operand.ssa]))
            continue
        constant = constants.get(operand.ssa)
        if constant is None:
            dynamic = scalar_inputs.get(operand.ssa)
            if dynamic is None or dynamic[1] != operand.ty:
                raise ValueError(
                    f"unresolved scalar launch operand {operand.ssa!r}"
                )
            plan.append((f"input_{operand.ty}", dynamic[0]))
            continue
        raw, _constant_type = constant
        if operand.ty in {"i64", "index"}:
            plan.append(("i64", int(raw)))
        elif operand.ty == "i32":
            plan.append(("i32", int(raw)))
        elif operand.ty == "f32":
            plan.append(("f32", orchestrate._parse_f32(raw)))
        elif operand.ty == "f64":
            plan.append(("f64", orchestrate._parse_f64(raw)))
        elif operand.ty == "f16":
            plan.append(("f16", orchestrate._parse_f16_bits(raw)))
        elif operand.ty == "bf16":
            plan.append(("bf16", orchestrate._parse_bf16_bits(raw)))
        else:
            raise NotImplementedError(
                f"unhandled scalar launch type {operand.ty!r}"
            )
    record = PreparedLaunchRecord(
        grid=grid,
        block=block,
        dynamic_smem_bytes=shared,
        argument_count=len(plan),
        tensor_argument_count=tensor_count,
    )
    return ExtractedSingleLaunch(
        cubin=kernel.cubin,
        kernel_name=kernel.kernel_name,
        binary_name=kernel.binary_name,
        plan=tuple(plan),
        record=record,
        source_mlir_sha256=source_digest,
    )


class PreparedSingleLaunch:
    """Bind one persisted cubin to its compiler-emitted launch plan."""

    def __init__(self, program: LoweredProgram, intent_module: str):
        self._bind(extract_single_launch(program, intent_module))

    @classmethod
    def from_extracted(
        cls, extracted: ExtractedSingleLaunch
    ) -> "PreparedSingleLaunch":
        instance = cls.__new__(cls)
        instance._bind(extracted)
        return instance

    def _bind(self, extracted: ExtractedSingleLaunch) -> None:
        function = load_module(extracted.cubin).function(extracted.kernel_name)
        shared = extracted.record.dynamic_smem_bytes
        orchestrate._set_dynamic_shared_memory(function, shared)
        self._launcher = orchestrate.CachedLauncher(
            function,
            extracted.record.grid,
            extracted.record.block,
            extracted.plan,
            shared=shared,
        )
        self._tensor_count = extracted.record.tensor_argument_count
        self.record = extracted.record
        self.extracted = extracted

    def thunk(self, tensors: Sequence[Any]):
        """Return a zero-argument no-sync closure over persistent tensors."""

        bound = tuple(tensors)
        if len(bound) != self._tensor_count:
            raise ValueError(
                "prepared launch tensor count mismatch: "
                f"expected {self._tensor_count}, got {len(bound)}"
            )
        self._launcher.bind_input_scalars(bound)

        def launch() -> None:
            self._launcher.launch(bound)

        return launch


def do_bench_us(
    thunk,
    *,
    warmup_ms: int = 25,
    repeat_ms: int = 100,
) -> float:
    """Measure a no-sync closure with Triton's event-based benchmark helper."""

    from triton.testing import do_bench

    result = do_bench(thunk, warmup=warmup_ms, rep=repeat_ms)
    return float(result) * 1000.0


def do_bench_cudagraph_us(thunk, *, repetitions: int = 100) -> float:
    """Measure steady-state device time with host launch overhead amortized.

    Both the compiled MLIR cubin and its comparison baseline must use this same
    helper.  The callable is captured on PyTorch's capture stream, replayed as
    one CUDA graph, and reported as median microseconds per invocation by
    Triton's established benchmark utility.
    """

    if repetitions <= 0:
        raise ValueError("CUDA-graph benchmark repetitions must be positive")
    from triton.testing import do_bench_cudagraph

    result = do_bench_cudagraph(
        thunk,
        rep=repetitions,
        return_mode="median",
    )
    return float(result) * 1000.0


__all__ = [
    "EXTRACTED_LAUNCH_PROTOCOL",
    "ExtractedSingleLaunch",
    "PreparedLaunchRecord",
    "PreparedSingleLaunch",
    "do_bench_cudagraph_us",
    "do_bench_us",
    "extract_single_launch",
]
