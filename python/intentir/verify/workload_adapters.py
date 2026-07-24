"""Declarative Harness adapters for the active real Triton representatives.

These records bind actual buffers, scalars, constexpr values, and launch grids.
They deliberately contain no J operators, predicates, formulas, or Replay text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.scrc_runner import (
    HarnessBindingInstance,
    SCRCRunRequest,
    SourceProgram,
)
from intentir.verify.triton_source_slice import (
    TritonSourceSlice,
    build_triton_source_slice,
    inline_imported_triton_helpers,
)
from intentir.verify.workload_manifest import (
    SCRCWorkloadEntry,
    build_workload_manifest,
)


TRITON_WORKLOAD_ADAPTER_PROTOCOL = "intentir.scrc.triton-workload-adapter.v1"


class BufferRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True)
class BufferBinding:
    formal: str
    role: BufferRole
    dtype: str
    shape: tuple[int, ...]
    output_key: str | None = None

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("buffer formal must be a Python identifier")
        if not isinstance(self.role, BufferRole):
            object.__setattr__(self, "role", BufferRole(self.role))
        if not self.dtype:
            raise ValueError("buffer dtype must be non-empty")
        shape = tuple(int(dim) for dim in self.shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("real workload buffers require positive dimensions")
        object.__setattr__(self, "shape", shape)
        if self.role is BufferRole.INPUT and self.output_key is not None:
            raise ValueError("input-only buffer cannot declare an output key")
        if self.role is not BufferRole.INPUT and self.output_key is None:
            object.__setattr__(self, "output_key", self.formal.casefold())


@dataclass(frozen=True)
class ScalarBinding:
    formal: str
    dtype: str
    value: bool | int | float

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("scalar formal must be a Python identifier")
        if not self.dtype:
            raise ValueError("scalar dtype must be non-empty")
        if not isinstance(self.value, (bool, int, float)):
            raise TypeError("runtime scalar value must be bool/int/float")


@dataclass(frozen=True)
class PreparedTritonWorkload:
    adapter: "TritonWorkloadAdapter"
    entry: SCRCWorkloadEntry
    source_location: str
    source_slice: TritonSourceSlice
    compiler_view: SCRCCompilerView
    request: SCRCRunRequest
    source_to_opaque: Mapping[str, str]


def _pairs(
    values: Iterable[tuple[str, Any]],
    *,
    label: str,
) -> tuple[tuple[str, Any], ...]:
    result = tuple(values)
    names = [name for name, _value in result]
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name.isidentifier() for name in names
    ):
        raise ValueError(f"{label} names must be unique Python identifiers")
    return result


@dataclass(frozen=True)
class TritonWorkloadAdapter:
    case_id: str
    buffers: tuple[BufferBinding, ...]
    scalars: tuple[ScalarBinding, ...]
    semantic_constexprs: tuple[tuple[str, Any], ...]
    c_values: tuple[tuple[str, Any], ...]
    grid: tuple[int, ...]
    grid_contract: tuple[str, ...]
    seed: int
    protocol: str = TRITON_WORKLOAD_ADAPTER_PROTOCOL

    def __post_init__(self) -> None:
        if not self.case_id.startswith("high_performance:"):
            raise ValueError("adapter case_id must reference the active registry")
        buffers = tuple(
            item if isinstance(item, BufferBinding) else BufferBinding(*item)
            for item in self.buffers
        )
        scalars = tuple(
            item if isinstance(item, ScalarBinding) else ScalarBinding(*item)
            for item in self.scalars
        )
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(self, "scalars", scalars)
        semantics = _pairs(
            self.semantic_constexprs, label="semantic constexpr"
        )
        c_values = _pairs(self.c_values, label="C")
        object.__setattr__(self, "semantic_constexprs", semantics)
        object.__setattr__(self, "c_values", c_values)
        names = [item.formal for item in buffers] + [
            item.formal for item in scalars
        ] + [name for name, _ in semantics] + [name for name, _ in c_values]
        if len(names) != len(set(names)):
            raise ValueError("adapter formals must be globally unique")
        grid = tuple(int(dim) for dim in self.grid)
        if not 1 <= len(grid) <= 3 or any(dim <= 0 for dim in grid):
            raise ValueError("adapter grid must contain positive dimensions")
        object.__setattr__(self, "grid", grid)
        contracts = tuple(self.grid_contract)
        if len(contracts) != len(grid) or any(not item for item in contracts):
            raise ValueError("grid contract must describe every launch dimension")
        object.__setattr__(self, "grid_contract", contracts)
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("adapter seed must be a non-negative integer")
        output_keys = [
            item.output_key
            for item in buffers
            if item.role is not BufferRole.INPUT
        ]
        if not output_keys or len(output_keys) != len(set(output_keys)):
            raise ValueError("adapter outputs must have unique observable keys")

    def compiler_view(
        self,
        parameter_order: tuple[str, ...],
        constexpr_parameters: tuple[str, ...],
    ) -> SCRCCompilerView:
        runtime = {
            item.formal: SCRCCompilerArgument(
                item.formal,
                "tensor",
                item.dtype,
                item.shape,
                mode=item.role.value,
            )
            for item in self.buffers
        }
        runtime.update(
            {
                item.formal: SCRCCompilerArgument(
                    item.formal,
                    "scalar",
                    item.dtype,
                    value=item.value,
                    mode="value",
                )
                for item in self.scalars
            }
        )
        expected_runtime = tuple(
            name for name in parameter_order if name not in constexpr_parameters
        )
        if set(expected_runtime) != set(runtime):
            missing = sorted(set(expected_runtime) - set(runtime))
            extra = sorted(set(runtime) - set(expected_runtime))
            raise ValueError(
                f"adapter runtime ABI differs from source; missing={missing}, extra={extra}"
            )
        semantics = dict(self.semantic_constexprs)
        c_values = dict(self.c_values)
        if set(constexpr_parameters) != set(semantics) | set(c_values):
            raise ValueError("adapter constexpr ABI differs from source")
        return SCRCCompilerView(
            schedule_values=c_values,
            bound_constexpr_values=semantics,
            arguments=tuple(runtime[name] for name in expected_runtime),
            grid=self.grid,
            grid_contract=self.grid_contract,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr_parameters,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "case_id": self.case_id,
            "buffers": [
                {
                    "formal": item.formal,
                    "role": item.role.value,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "output_key": item.output_key,
                }
                for item in self.buffers
            ],
            "scalars": [
                {
                    "formal": item.formal,
                    "dtype": item.dtype,
                    "value": item.value,
                }
                for item in self.scalars
            ],
            "semantic_constexprs": dict(self.semantic_constexprs),
            "c_values": dict(self.c_values),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "seed": self.seed,
        }


def _runtime_argument(argument: SCRCCompilerArgument) -> RuntimeArgument:
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    dtype = aliases.get(argument.dtype, argument.dtype)
    if argument.kind == "tensor":
        return RuntimeArgument.tensor(argument.shape, dtype)
    if argument.kind == "scalar":
        assert argument.value is not None
        return RuntimeArgument.scalar(argument.value, dtype)
    raise ValueError("real workload adapter supports tensor/scalar runtime ABI")


def prepare_triton_workload(
    adapter: TritonWorkloadAdapter,
    *,
    dataset_root: str | Path | None = None,
) -> PreparedTritonWorkload:
    manifest = build_workload_manifest()
    entry = manifest.by_id.get(adapter.case_id)
    if entry is None:
        raise ValueError("adapter case is absent from the active 28-kernel registry")
    resolution = entry.resolve_source(dataset_root=dataset_root)
    if not resolution.resolved or resolution.path is None:
        raise ValueError(
            f"real source is unavailable for {entry.case_id}: {resolution.diagnostic}"
        )
    source = resolution.path.read_text(encoding="utf-8")
    source_slice = build_triton_source_slice(
        source, source_symbol=entry.symbol
    )
    module_root = resolution.path
    for _component in entry.source_module.split("."):
        module_root = module_root.parent
    source_slice = inline_imported_triton_helpers(
        source_slice, search_roots=(module_root,)
    )
    view = adapter.compiler_view(
        source_slice.parameters, source_slice.constexpr_parameters
    )
    source_slice.validate_abi(
        parameter_order=view.parameter_order,
        constexpr_parameters=view.constexpr_parameters,
    )
    anonymous, _binding, mapping = anonymize_source(
        source_slice.code,
        source_symbol=entry.symbol,
        bound_constexpr_parameters=tuple(
            name for name, _value in adapter.semantic_constexprs
        ),
    )
    arguments = {
        mapping[item.name]: _runtime_argument(item)
        for item in view.arguments
    }
    all_constexprs = {
        **dict(adapter.semantic_constexprs),
        **dict(adapter.c_values),
    }
    schedule = {
        mapping[name]: all_constexprs[name]
        for name in source_slice.constexpr_parameters
    }
    c0 = {
        mapping[name]: value for name, value in adapter.c_values
    }
    gate_source = KernelSource.from_code(
        anonymous, "scrc_k0", trusted=True
    )
    request = SCRCRunRequest(
        SourceProgram(source_slice.code, gate_source),
        HarnessBindingInstance.create(
            c0,
            gate1_case=CompileCase(arguments, adapter.grid, schedule),
            gate2_context={
                "test_mode": "finite",
                "seed": adapter.seed,
                "adapter_protocol": adapter.protocol,
            },
        ),
        "triton",
    )
    return PreparedTritonWorkload(
        adapter,
        entry,
        resolution.location,
        source_slice,
        view,
        request,
        dict(mapping),
    )


def _buffer(
    formal: str,
    role: BufferRole,
    dtype: str,
    shape: tuple[int, ...],
    output_key: str | None = None,
) -> BufferBinding:
    return BufferBinding(formal, role, dtype, shape, output_key)


def _scalar(formal: str, dtype: str, value: Any) -> ScalarBinding:
    return ScalarBinding(formal, dtype, value)


TRITON_WORKLOAD_ADAPTERS: dict[str, TritonWorkloadAdapter] = {
    "high_performance:liger_dyt": TritonWorkloadAdapter(
        "high_performance:liger_dyt",
        buffers=(
            _buffer("X", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("Y", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("Alpha", BufferRole.INPUT, "f32", (1,)),
            _buffer("Gamma", BufferRole.INPUT, "f32", (127,)),
            _buffer("Beta", BufferRole.INPUT, "f32", (127,)),
        ),
        scalars=(),
        semantic_constexprs=(("HAVE_BETA", True), ("N", 127)),
        c_values=(("BLOCK_N", 128),),
        grid=(1, 2),
        grid_contract=("ceil_div(N,BLOCK_N)", "rows"),
        seed=101,
    ),
    "high_performance:liger_swiglu": TritonWorkloadAdapter(
        "high_performance:liger_swiglu",
        buffers=(
            _buffer("a_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("b_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("c_ptr", BufferRole.OUTPUT, "f32", (2, 127), "c"),
        ),
        scalars=(
            _scalar("stride", "i64", 127),
            _scalar("gate_multiplier", "f32", 1.0),
        ),
        semantic_constexprs=(("n_cols", 127),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=103,
    ),
    (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_preprocess_do_o_dot"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_preprocess_do_o_dot"
        ),
        buffers=(
            _buffer("Out", BufferRole.INPUT, "f16", (1, 2, 33, 16)),
            _buffer("DO", BufferRole.INPUT, "f16", (1, 2, 33, 16)),
            _buffer("Delta", BufferRole.OUTPUT, "f32", (1, 2, 64), "delta"),
        ),
        scalars=(
            _scalar("stride_ob", "i64", 1056),
            _scalar("stride_oh", "i64", 528),
            _scalar("stride_om", "i64", 16),
            _scalar("stride_dob", "i64", 1056),
            _scalar("stride_doh", "i64", 528),
            _scalar("stride_dom", "i64", 16),
            _scalar("nheads", "i32", 2),
            _scalar("seqlen_q", "i32", 33),
            _scalar("seqlen_q_rounded", "i32", 64),
            _scalar("headdim", "i32", 16),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_M", 32), ("BLOCK_HEADDIM", 16)),
        grid=(2, 2),
        grid_contract=("ceil_div(seqlen_q,BLOCK_M)", "batch*nheads"),
        seed=107,
    ),
    "high_performance:liger_softmax": TritonWorkloadAdapter(
        "high_performance:liger_softmax",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 127)),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("n_cols", "i32", 127),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=109,
    ),
    "high_performance:liger_rms_norm": TritonWorkloadAdapter(
        "high_performance:liger_rms_norm",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (127,)),
            _buffer("RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd"),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("offset", "f32", 0.0),
        ),
        semantic_constexprs=(
            ("casting_mode", 0),
            ("elementwise_affine", True),
        ),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=113,
    ),
    (
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
        buffers=(
            _buffer("C", BufferRole.OUTPUT, "f16", (32, 32), "c"),
            _buffer(
                "ACT_INPUT", BufferRole.OUTPUT, "f16", (32, 32), "act_input"
            ),
            _buffer("A", BufferRole.INPUT, "f16", (32, 32)),
            _buffer("B", BufferRole.INPUT, "f16", (32, 32)),
            _buffer("bias", BufferRole.INPUT, "f16", (32,)),
        ),
        scalars=(
            _scalar("M", "i32", 32),
            _scalar("N", "i32", 32),
            _scalar("K", "i32", 32),
            _scalar("CACHE_KEY_M", "i32", 32),
            _scalar("CACHE_KEY_N", "i32", 32),
            _scalar("CACHE_KEY_K", "i32", 32),
            _scalar("stride_cm", "i64", 32),
            _scalar("stride_am", "i64", 32),
            _scalar("stride_ak", "i64", 1),
            _scalar("stride_bn", "i64", 32),
            _scalar("stride_bk", "i64", 1),
        ),
        semantic_constexprs=(
            ("BIAS", True),
            ("SAVE_ACT_INPUT", True),
            ("ACTIVATION", "id"),
        ),
        c_values=(
            ("BLOCK_M", 32),
            ("GROUP_M", 8),
            ("BLOCK_N", 32),
            ("BLOCK_K", 32),
            ("SPLIT_K", 1),
            ("EVEN_K", True),
            ("A_ROWMAJOR", True),
            ("B_COLMAJOR", True),
        ),
        grid=(1,),
        grid_contract=("ceil_div(M,BLOCK_M)*ceil_div(N,BLOCK_N)",),
        seed=127,
    ),
}


__all__ = [
    "BufferBinding",
    "BufferRole",
    "PreparedTritonWorkload",
    "TRITON_WORKLOAD_ADAPTER_PROTOCOL",
    "TRITON_WORKLOAD_ADAPTERS",
    "ScalarBinding",
    "TritonWorkloadAdapter",
    "prepare_triton_workload",
]
