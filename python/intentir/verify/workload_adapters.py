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
    expand_triton_varargs,
    inline_imported_triton_helpers,
    requires_triton_vararg_expansion,
)
from intentir.verify.workload_manifest import (
    SCRCWorkloadEntry,
    build_workload_manifest,
)


TRITON_WORKLOAD_ADAPTER_PROTOCOL = "intentir.scrc.triton-workload-adapter.v3"
TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL = (
    "intentir.scrc.triton-workload-adapter.v4"
)


class BufferRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class BufferInitializer(str, Enum):
    RANDOM = "random"
    ZEROS = "zeros"
    STANDARD_NORMAL = "standard_normal"


class BufferInputTransform(str, Enum):
    """Closed, Harness-only relations between runtime input buffers."""

    SORT_DESCENDING_LAST_AXIS = "sort_descending_last_axis"


@dataclass(frozen=True)
class BufferBinding:
    formal: str
    role: BufferRole
    dtype: str
    shape: tuple[int, ...]
    output_key: str | None = None
    initializer: BufferInitializer | None = None
    source_formal: str | None = None
    input_transform: BufferInputTransform | None = None

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
        if (self.source_formal is None) != (self.input_transform is None):
            raise ValueError(
                "derived input requires both source_formal and input_transform"
            )
        if self.input_transform is not None and not isinstance(
            self.input_transform, BufferInputTransform
        ):
            object.__setattr__(
                self,
                "input_transform",
                BufferInputTransform(self.input_transform),
            )
        if self.source_formal is not None:
            if self.role is not BufferRole.INPUT:
                raise ValueError("only input buffers may derive from another input")
            if not self.source_formal.isidentifier() or self.source_formal == self.formal:
                raise ValueError("derived input source must be a distinct buffer formal")
            if self.initializer is not None:
                raise ValueError("derived input cannot also declare an initializer")
        elif self.role is BufferRole.OUTPUT:
            if self.initializer is not None:
                raise ValueError("output-only buffer cannot declare an initializer")
        elif self.initializer is None:
            object.__setattr__(self, "initializer", BufferInitializer.RANDOM)
        elif not isinstance(self.initializer, BufferInitializer):
            object.__setattr__(
                self, "initializer", BufferInitializer(self.initializer)
            )


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
class NoneBinding:
    formal: str
    dtype: str = "none"

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("None formal must be a Python identifier")
        if self.dtype != "none":
            raise ValueError("None binding dtype must be exactly 'none'")


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
    none_arguments: tuple[NoneBinding, ...] = ()
    vararg_arity: int | None = None
    protocol: str = TRITON_WORKLOAD_ADAPTER_PROTOCOL

    def __post_init__(self) -> None:
        if not self.case_id.startswith(
            ("high_performance:", "scrc_canary:", "tilelang:")
        ):
            raise ValueError(
                "adapter case_id must reference the active or certified-canary "
                "registry"
            )
        buffers = tuple(
            item if isinstance(item, BufferBinding) else BufferBinding(*item)
            for item in self.buffers
        )
        scalars = tuple(
            item if isinstance(item, ScalarBinding) else ScalarBinding(*item)
            for item in self.scalars
        )
        none_arguments = tuple(
            item if isinstance(item, NoneBinding) else NoneBinding(*item)
            for item in self.none_arguments
        )
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(self, "scalars", scalars)
        object.__setattr__(self, "none_arguments", none_arguments)
        semantics = _pairs(
            self.semantic_constexprs, label="semantic constexpr"
        )
        c_values = _pairs(self.c_values, label="C")
        object.__setattr__(self, "semantic_constexprs", semantics)
        object.__setattr__(self, "c_values", c_values)
        names = [item.formal for item in buffers] + [
            item.formal for item in scalars
        ] + [item.formal for item in none_arguments] + [
            name for name, _ in semantics
        ] + [name for name, _ in c_values]
        if len(names) != len(set(names)):
            raise ValueError("adapter formals must be globally unique")
        buffer_by_formal = {item.formal: item for item in buffers}
        for item in buffers:
            if item.source_formal is None:
                continue
            source = buffer_by_formal.get(item.source_formal)
            if source is None or source.role is BufferRole.OUTPUT:
                raise ValueError("derived input source must be a readable buffer")
            if source.source_formal is not None:
                raise ValueError("derived input chains are not supported")
            if source.dtype != item.dtype or source.shape != item.shape:
                raise ValueError("derived input must match source dtype and shape")
        if any(item.source_formal is not None for item in buffers) and (
            self.protocol != TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL
        ):
            raise ValueError("derived inputs require workload adapter protocol v4")
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
        if self.vararg_arity is not None and (
            isinstance(self.vararg_arity, bool)
            or not isinstance(self.vararg_arity, int)
            or not 1 <= self.vararg_arity <= 64
        ):
            raise ValueError("adapter vararg arity must be in [1, 64]")
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
        runtime.update(
            {
                item.formal: SCRCCompilerArgument(
                    item.formal,
                    "none",
                    item.dtype,
                    mode="value",
                )
                for item in self.none_arguments
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
                    "initializer": (
                        item.initializer.value
                        if item.initializer is not None
                        else None
                    ),
                    **(
                        {
                            "source_formal": item.source_formal,
                            "input_transform": item.input_transform.value,
                        }
                        if item.source_formal is not None
                        and item.input_transform is not None
                        else {}
                    ),
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
            "none_arguments": [
                {"formal": item.formal, "dtype": item.dtype}
                for item in self.none_arguments
            ],
            "semantic_constexprs": dict(self.semantic_constexprs),
            "c_values": dict(self.c_values),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "seed": self.seed,
            "vararg_arity": self.vararg_arity,
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
    if argument.kind == "none":
        return RuntimeArgument.none(dtype)
    raise ValueError("real workload adapter supports tensor/scalar/none runtime ABI")


def _prepare_triton_source(
    adapter: TritonWorkloadAdapter,
    entry: SCRCWorkloadEntry,
    *,
    source: str,
    source_location: str,
    search_roots: tuple[Path, ...],
) -> PreparedTritonWorkload:
    source_slice = build_triton_source_slice(
        source, source_symbol=entry.symbol
    )
    source_slice = inline_imported_triton_helpers(
        source_slice, search_roots=search_roots
    )
    requires_varargs = requires_triton_vararg_expansion(source_slice)
    if requires_varargs != (adapter.vararg_arity is not None):
        raise ValueError(
            "adapter vararg arity must be present exactly when Source declares "
            "VAR_ARGS_ARRAY"
        )
    if adapter.vararg_arity is not None:
        source_slice = expand_triton_varargs(
            source_slice, arity=adapter.vararg_arity
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
        source_location,
        source_slice,
        view,
        request,
        dict(mapping),
    )


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
    module_root = resolution.path
    for _component in entry.source_module.split("."):
        module_root = module_root.parent
    return _prepare_triton_source(
        adapter,
        entry,
        source=resolution.path.read_text(encoding="utf-8"),
        source_location=resolution.location,
        search_roots=(module_root,),
    )


def prepare_triton_source_workload(
    adapter: TritonWorkloadAdapter,
    source_path: str | Path,
    *,
    source_symbol: str,
    source_location: str | None = None,
) -> PreparedTritonWorkload:
    """Prepare one exact certified Source outside the active 28-kernel registry."""

    if not adapter.case_id.startswith(("scrc_canary:", "tilelang:")):
        raise ValueError(
            "explicit Source preparation is reserved for certified canaries "
            "and accepted TileLang workload adapters"
        )
    path = Path(source_path).resolve()
    source = path.read_text(encoding="utf-8")
    entry = SCRCWorkloadEntry(
        ordinal=0,
        case_id=adapter.case_id,
        suite=("tilelang" if adapter.case_id.startswith("tilelang:") else "scrc_canary"),
        symbol=source_symbol,
        mechanisms=("certified_source_reuse",),
        source_module="intentir.certified_canary",
        source_alignment="exact archived accepted Source",
        lowering_status="certified",
        evidence_refs=(),
        representative_tier="certified_canary",
        adapter_readiness="declared",
    )
    return _prepare_triton_source(
        adapter,
        entry,
        source=source,
        source_location=(source_location or str(path)),
        search_roots=(path.parent,),
    )


def _buffer(
    formal: str,
    role: BufferRole,
    dtype: str,
    shape: tuple[int, ...],
    output_key: str | None = None,
    initializer: BufferInitializer | None = None,
    source_formal: str | None = None,
    input_transform: BufferInputTransform | None = None,
) -> BufferBinding:
    return BufferBinding(
        formal,
        role,
        dtype,
        shape,
        output_key,
        initializer,
        source_formal,
        input_transform,
    )


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
    (
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel",
        buffers=(
            _buffer("Q", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            _buffer("K", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            _buffer("V", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            # BIAS_TYPE=none makes this ABI carrier intentionally unread.
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer(
                "Out", BufferRole.OUTPUT, "f16", (1, 128, 1, 64), "out"
            ),
            _buffer("Lse", BufferRole.OUTPUT, "f32", (1, 1, 128), "lse"),
            # TMP is explicit source-ABI scratch storage.  The certified J
            # models its store/load recurrence and therefore exposes it as an
            # input/output tensor, even though each tile overwrites before use.
            _buffer(
                "TMP",
                BufferRole.INOUT,
                "f32",
                (1, 1, 128),
                "tmp",
                BufferInitializer.ZEROS,
            ),
        ),
        scalars=(
            _scalar("softmax_scale", "f32", 0.125),
            _scalar("stride_qb", "i64", 8192),
            _scalar("stride_qh", "i64", 64),
            _scalar("stride_qm", "i64", 64),
            _scalar("stride_kb", "i64", 8192),
            _scalar("stride_kh", "i64", 64),
            _scalar("stride_kn", "i64", 64),
            _scalar("stride_vb", "i64", 8192),
            _scalar("stride_vh", "i64", 64),
            _scalar("stride_vn", "i64", 64),
            _scalar("stride_bb", "i64", 0),
            _scalar("stride_bh", "i64", 0),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_ob", "i64", 8192),
            _scalar("stride_oh", "i64", 64),
            _scalar("stride_om", "i64", 64),
            _scalar("nheads", "i32", 1),
            _scalar("seqlen_q", "i32", 128),
            _scalar("seqlen_k", "i32", 128),
            _scalar("seqlen_q_rounded", "i32", 128),
            _scalar("headdim", "i32", 64),
            _scalar("CACHE_KEY_SEQLEN_Q", "i32", 4),
            _scalar("CACHE_KEY_SEQLEN_K", "i32", 4),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("BLOCK_HEADDIM", 64),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            ("BLOCK_M", 128),
            ("BLOCK_N", 128),
        ),
        grid=(1, 1),
        grid_contract=("ceil_div(seqlen_q,BLOCK_M)", "batch*nheads"),
        seed=131,
    ),
    (
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel",
        buffers=(
            _buffer("Q", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            _buffer("K", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            _buffer("V", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            # BIAS_TYPE=none makes this ABI carrier intentionally unread.
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer("DO", BufferRole.INPUT, "f16", (1, 128, 1, 64)),
            _buffer(
                "DQ",
                BufferRole.INOUT,
                "f32",
                (1, 128, 1, 64),
                "dq",
                BufferInitializer.ZEROS,
            ),
            _buffer("DK", BufferRole.OUTPUT, "f32", (1, 128, 1, 64), "dk"),
            _buffer("DV", BufferRole.OUTPUT, "f32", (1, 128, 1, 64), "dv"),
            _buffer("LSE", BufferRole.INPUT, "f32", (1, 1, 128)),
            _buffer("D", BufferRole.INPUT, "f32", (1, 1, 128)),
        ),
        scalars=(
            _scalar("softmax_scale", "f32", 0.125),
            _scalar("stride_qb", "i64", 8192),
            _scalar("stride_qh", "i64", 64),
            _scalar("stride_qm", "i64", 64),
            _scalar("stride_kb", "i64", 8192),
            _scalar("stride_kh", "i64", 64),
            _scalar("stride_kn", "i64", 64),
            _scalar("stride_vb", "i64", 8192),
            _scalar("stride_vh", "i64", 64),
            _scalar("stride_vn", "i64", 64),
            _scalar("stride_bb", "i64", 0),
            _scalar("stride_bh", "i64", 0),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_dob", "i64", 8192),
            _scalar("stride_doh", "i64", 64),
            _scalar("stride_dom", "i64", 64),
            _scalar("stride_dqb", "i64", 8192),
            _scalar("stride_dqh", "i64", 64),
            _scalar("stride_dqm", "i64", 64),
            _scalar("stride_dkb", "i64", 8192),
            _scalar("stride_dkh", "i64", 64),
            _scalar("stride_dkn", "i64", 64),
            _scalar("stride_dvb", "i64", 8192),
            _scalar("stride_dvh", "i64", 64),
            _scalar("stride_dvn", "i64", 64),
            _scalar("nheads", "i32", 1),
            _scalar("seqlen_q", "i32", 128),
            _scalar("seqlen_k", "i32", 128),
            _scalar("seqlen_q_rounded", "i32", 128),
            _scalar("headdim", "i32", 64),
            _scalar("CACHE_KEY_SEQLEN_Q", "i32", 4),
            _scalar("CACHE_KEY_SEQLEN_K", "i32", 4),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("BLOCK_HEADDIM", 64),
            ("SEQUENCE_PARALLEL", False),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            ("BLOCK_M", 128),
            ("BLOCK_N", 128),
        ),
        grid=(1, 1),
        grid_contract=("1", "batch*nheads"),
        seed=137,
    ),
    "high_performance:liger_fused_add_rms_norm": TritonWorkloadAdapter(
        "high_performance:liger_fused_add_rms_norm",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("S_ptr", BufferRole.OUTPUT, "f32", (2, 127), "s"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("R_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (127,)),
            _buffer("RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd"),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("S_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("R_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("offset", "f32", 0.0),
        ),
        semantic_constexprs=(("casting_mode", -1),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=139,
    ),
    "high_performance:liger_sparsemax": TritonWorkloadAdapter(
        "high_performance:liger_sparsemax",
        buffers=(
            _buffer("x_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer(
                "sorted_x_ptr",
                BufferRole.INPUT,
                "f32",
                (2, 127),
                source_formal="x_ptr",
                input_transform=BufferInputTransform.SORT_DESCENDING_LAST_AXIS,
            ),
            _buffer("o_ptr", BufferRole.OUTPUT, "f32", (2, 127), "o"),
        ),
        scalars=(
            _scalar("x_stride_row", "i64", 127),
            _scalar("sorted_x_stride_row", "i64", 127),
            _scalar("o_stride_row", "i64", 127),
            _scalar("n_cols", "i32", 127),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128), ("num_warps", 4)),
        grid=(2,),
        grid_contract=("rows",),
        seed=149,
        protocol=TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL,
    ),
    (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_kernel_one_col_block"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_kernel_one_col_block"
        ),
        buffers=(
            _buffer("Q", BufferRole.INPUT, "f16", (128, 64)),
            _buffer("K", BufferRole.INPUT, "f16", (64, 64)),
            _buffer("V", BufferRole.INPUT, "f16", (64, 64)),
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer("DO", BufferRole.INPUT, "f16", (128, 64)),
            _buffer(
                "DQ",
                BufferRole.INOUT,
                "f32",
                (128, 64),
                "dq",
                BufferInitializer.ZEROS,
            ),
            _buffer("DK", BufferRole.OUTPUT, "f32", (64, 64), "dk"),
            _buffer("DV", BufferRole.OUTPUT, "f32", (64, 64), "dv"),
            _buffer("LSE", BufferRole.INPUT, "f32", (128,)),
            _buffer("D", BufferRole.INPUT, "f32", (128,)),
        ),
        scalars=(
            _scalar("start_n", "i32", 0),
            _scalar("softmax_scale", "f32", 0.125),
            _scalar("stride_qm", "i64", 64),
            _scalar("stride_kn", "i64", 64),
            _scalar("stride_vn", "i64", 64),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_dom", "i64", 64),
            _scalar("stride_dqm", "i64", 64),
            _scalar("stride_dkn", "i64", 64),
            _scalar("stride_dvn", "i64", 64),
            _scalar("seqlen_q", "i32", 128),
            _scalar("seqlen_k", "i32", 64),
            _scalar("headdim", "i32", 64),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("ATOMIC_ADD", False),
            ("BLOCK_HEADDIM", 64),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            ("BLOCK_M", 128),
            ("BLOCK_N", 64),
        ),
        grid=(1,),
        grid_contract=("1",),
        seed=151,
    ),
    "high_performance:liger_qwen2vl_mrope": TritonWorkloadAdapter(
        "high_performance:liger_qwen2vl_mrope",
        buffers=(
            _buffer(
                "q_ptr", BufferRole.INOUT, "f16", (1, 4, 2, 16), "q"
            ),
            _buffer(
                "k_ptr", BufferRole.INOUT, "f16", (1, 4, 1, 16), "k"
            ),
            _buffer("cos", BufferRole.INPUT, "f16", (3, 1, 4, 16)),
            _buffer("sin", BufferRole.INPUT, "f16", (3, 1, 4, 16)),
        ),
        scalars=(_scalar("sl", "i32", 4),),
        semantic_constexprs=(
            ("bs", 1),
            ("n_qh", 2),
            ("n_kh", 1),
            ("hd", 16),
            ("mrope_section_t", 3),
            ("mrope_section_h", 3),
            ("BACKWARD_PASS", False),
        ),
        c_values=(
            ("pad_n_qh", 2),
            ("pad_n_kh", 1),
            ("pad_hd", 16),
            ("BLOCK_SIZE", 2),
        ),
        grid=(4,),
        grid_contract=("bs*sl",),
        seed=157,
    ),
    "high_performance:liger_group_norm": TritonWorkloadAdapter(
        "high_performance:liger_group_norm",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 2, 32), "y"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 2, 32)),
            _buffer("Mean_ptr", BufferRole.OUTPUT, "f32", (2, 2), "mean"),
            _buffer("RSTD_ptr", BufferRole.OUTPUT, "f32", (2, 2), "rstd"),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (4,)),
            _buffer("B_ptr", BufferRole.INPUT, "f32", (4,)),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 64),
            _scalar("Y_col_stride", "i64", 32),
            _scalar("X_row_stride", "i64", 64),
            _scalar("X_col_stride", "i64", 32),
            _scalar("Mean_row_stride", "i64", 2),
            _scalar("Mean_col_stride", "i64", 1),
            _scalar("RSTD_row_stride", "i64", 2),
            _scalar("RSTD_col_stride", "i64", 1),
            _scalar("hidden_size", "i32", 32),
            _scalar("channels_per_group", "i32", 2),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 32),),
        grid=(2, 2),
        grid_contract=("batch", "groups"),
        seed=163,
    ),
    "high_performance:liger_jsd": TritonWorkloadAdapter(
        "high_performance:liger_jsd",
        buffers=(
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("Y_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("loss_ptr", BufferRole.OUTPUT, "f32", (2, 64), "loss"),
            _buffer("dX_ptr", BufferRole.OUTPUT, "f32", (2, 64), "dx"),
            # HAS_LABEL=false makes this required ABI carrier unread.
            _buffer("label_ptr", BufferRole.INPUT, "i64", (2,)),
        ),
        scalars=(
            _scalar("X_stride", "i64", 64),
            _scalar("Y_stride", "i64", 64),
            _scalar("loss_stride", "i64", 64),
            _scalar("dX_stride", "i64", 64),
            _scalar("n_non_ignore", "i32", 2),
            _scalar("n_cols", "i32", 64),
        ),
        semantic_constexprs=(
            ("beta", 0.5),
            ("ignore_index", -100),
            ("HAS_LABEL", False),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        seed=167,
    ),
    "high_performance:liger_cross_entropy": TritonWorkloadAdapter(
        "high_performance:liger_cross_entropy",
        buffers=(
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer(
                "Y_ptr",
                BufferRole.INPUT,
                "i64",
                (2,),
                initializer=BufferInitializer.ZEROS,
            ),
            # HAS_WEIGHT=false makes this required ABI carrier unread.
            _buffer("weight_ptr", BufferRole.INPUT, "f32", (64,)),
            _buffer("loss_ptr", BufferRole.OUTPUT, "f32", (2,), "loss"),
            # The three RETURN_* flags below are false, so these ABI carriers
            # are deliberately inputs rather than observable uninitialized
            # outputs.
            _buffer("z_loss_ptr", BufferRole.INPUT, "f32", (2,)),
            _buffer("token_accuracy_ptr", BufferRole.INPUT, "f32", (2,)),
            _buffer("predicted_tokens_ptr", BufferRole.INPUT, "i64", (2,)),
        ),
        scalars=(
            _scalar("X_stride", "i64", 64),
            _scalar("Y_stride", "i64", 1),
            _scalar("loss_stride", "i64", 1),
            _scalar("token_accuracy_stride", "i64", 0),
            _scalar("predicted_tokens_stride", "i64", 0),
            _scalar("n_cols", "i32", 64),
            _scalar("n_non_ignore", "f32", 2.0),
            _scalar("sum_non_ignore_weight", "f32", 2.0),
            _scalar("weight_sum", "f32", 0.0),
            _scalar("ignore_index", "i64", -100),
            _scalar("softcap", "f32", 0.0),
        ),
        semantic_constexprs=(
            ("lse_square_scale", 0.0),
            ("label_smoothing", 0.0),
            ("reduction", "mean"),
            ("RETURN_Z_LOSS", False),
            ("RETURN_TOKEN_ACCURACY", False),
            ("RETURN_PREDICTED_TOKENS", False),
            ("HAS_WEIGHT", False),
            ("HAS_SOFTCAPPING", False),
            ("HAS_GRADIENTS", False),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        seed=173,
    ),
    "high_performance:liger_geglu": TritonWorkloadAdapter(
        "high_performance:liger_geglu",
        buffers=(
            _buffer("a", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("b", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("c", BufferRole.OUTPUT, "f32", (2, 127), "c"),
        ),
        scalars=(_scalar("stride", "i64", 127),),
        semantic_constexprs=(("n_cols", 127),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=179,
    ),
    "high_performance:liger_poly_norm": TritonWorkloadAdapter(
        "high_performance:liger_poly_norm",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (3,)),
            _buffer("B_ptr", BufferRole.INPUT, "f32", (1,)),
            _buffer("RSTD_ptr", BufferRole.OUTPUT, "f32", (2, 3), "rstd"),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 3),
            _scalar("n_cols", "i32", 127),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=181,
    ),
    "high_performance:liger_llama4_rope": TritonWorkloadAdapter(
        "high_performance:liger_llama4_rope",
        buffers=(
            _buffer("q_ptr", BufferRole.INOUT, "f16", (1, 4, 2, 32), "q"),
            _buffer("k_ptr", BufferRole.INOUT, "f16", (1, 4, 1, 32), "k"),
            _buffer(
                "freqs_complex_ptr", BufferRole.INPUT, "f16", (4, 16, 2)
            ),
        ),
        scalars=(
            _scalar("q_row_stride", "i64", 64),
            _scalar("k_row_stride", "i64", 32),
            _scalar("q_head_stride", "i64", 32),
            _scalar("k_head_stride", "i64", 32),
            _scalar("freqs_row_stride", "i64", 32),
            _scalar("seq_len", "i32", 4),
            _scalar("batch_size", "i32", 1),
            _scalar("imag_sign", "f32", 1.0),
        ),
        semantic_constexprs=(
            ("head_dim_half", 16),
            ("n_q_heads", 2),
            ("n_k_heads", 1),
        ),
        c_values=(("BLOCK_SIZE", 16),),
        grid=(4, 2),
        grid_contract=("batch_size*seq_len", "max(n_q_heads,n_k_heads)"),
        seed=191,
    ),
    (
        "high_performance:flash_attn.ops.triton.layer_norm."
        "_layer_norm_fwd_1pass_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.ops.triton.layer_norm."
            "_layer_norm_fwd_1pass_kernel"
        ),
        buffers=(
            _buffer("X", BufferRole.INPUT, "f16", (2, 127)),
            _buffer("Y", BufferRole.OUTPUT, "f16", (2, 127), "y"),
            _buffer("W", BufferRole.INPUT, "f16", (127,)),
            _buffer("B", BufferRole.INPUT, "f16", (127,)),
            _buffer("RESIDUAL", BufferRole.INPUT, "f16", (2, 127)),
            _buffer("X1", BufferRole.INPUT, "f16", (2, 127)),
            _buffer("W1", BufferRole.INPUT, "f16", (127,)),
            _buffer("B1", BufferRole.INPUT, "f16", (127,)),
            _buffer("Y1", BufferRole.OUTPUT, "f16", (2, 127), "y1"),
            _buffer(
                "RESIDUAL_OUT",
                BufferRole.OUTPUT,
                "f16",
                (2, 127),
                "residual_out",
            ),
            # The fixed flags make these four required ABI carriers unread.
            _buffer("ROWSCALE", BufferRole.INPUT, "f32", (2,)),
            _buffer("SEEDS", BufferRole.INPUT, "i64", (2,)),
            _buffer("DROPOUT_MASK", BufferRole.INPUT, "i1", (2, 127)),
            _buffer("DROPOUT_MASK1", BufferRole.INPUT, "i1", (2, 127)),
            _buffer("Mean", BufferRole.OUTPUT, "f32", (2,), "mean"),
            _buffer("Rstd", BufferRole.OUTPUT, "f32", (2,), "rstd"),
        ),
        scalars=(
            _scalar("stride_x_row", "i64", 127),
            _scalar("stride_y_row", "i64", 127),
            _scalar("stride_res_row", "i64", 127),
            _scalar("stride_res_out_row", "i64", 127),
            _scalar("stride_x1_row", "i64", 127),
            _scalar("stride_y1_row", "i64", 127),
            _scalar("M", "i32", 2),
            _scalar("N", "i32", 127),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("dropout_p", "f32", 0.0),
            _scalar("zero_centered_weight", "i1", False),
        ),
        semantic_constexprs=(
            ("IS_RMS_NORM", False),
            ("HAS_RESIDUAL", True),
            ("STORE_RESIDUAL_OUT", True),
            ("HAS_BIAS", True),
            ("HAS_DROPOUT", False),
            ("STORE_DROPOUT_MASK", False),
            ("HAS_ROWSCALE", False),
            ("HAS_X1", True),
            ("HAS_W1", True),
            ("HAS_B1", True),
        ),
        c_values=(("BLOCK_N", 128),),
        grid=(2,),
        grid_contract=("M",),
        seed=193,
    ),
    "high_performance:liger_rope": TritonWorkloadAdapter(
        "high_performance:liger_rope",
        buffers=(
            _buffer("q_ptr", BufferRole.INOUT, "f16", (1, 4, 2, 16), "q"),
            _buffer("k_ptr", BufferRole.INOUT, "f16", (1, 4, 1, 16), "k"),
            _buffer("cos", BufferRole.INPUT, "f16", (1, 4, 16)),
            _buffer("sin", BufferRole.INPUT, "f16", (1, 4, 16)),
        ),
        scalars=(
            _scalar("q_row_stride", "i64", 32),
            _scalar("k_row_stride", "i64", 16),
            _scalar("cos_row_stride", "i64", 16),
            _scalar("sin_row_stride", "i64", 16),
            _scalar("sl", "i32", 4),
        ),
        semantic_constexprs=(
            ("bs", 1),
            ("cos_bs", 1),
            ("n_qh", 2),
            ("n_kh", 1),
            ("hd", 16),
            ("BACKWARD_PASS", False),
        ),
        c_values=(
            ("pad_n_qh", 2),
            ("pad_n_kh", 1),
            ("pad_hd", 16),
            ("BLOCK_SIZE", 2),
        ),
        grid=(4,),
        grid_contract=("bs*sl",),
        seed=197,
    ),
    "high_performance:liger_layer_norm": TritonWorkloadAdapter(
        "high_performance:liger_layer_norm",
        buffers=(
            _buffer("Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y"),
            _buffer("X_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (127,)),
            _buffer("B_ptr", BufferRole.INPUT, "f32", (127,)),
            _buffer("Mean_ptr", BufferRole.OUTPUT, "f32", (2,), "mean"),
            _buffer("RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd"),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("B_row_stride", "i64", 127),
            _scalar("Mean_row_stride", "i64", 1),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=199,
    ),
    "high_performance:liger_kl_div": TritonWorkloadAdapter(
        "high_performance:liger_kl_div",
        buffers=(
            _buffer("y_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("gt_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("loss_ptr", BufferRole.OUTPUT, "f32", (2,), "loss"),
        ),
        scalars=(
            _scalar("y_stride", "i64", 64),
            _scalar("gt_stride", "i64", 64),
            _scalar("loss_stride", "i64", 1),
            _scalar("n_cols", "i32", 64),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(
            ("log_target", True),
            ("reduction", 3),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        seed=211,
    ),
    "high_performance:liger_tvd": TritonWorkloadAdapter(
        "high_performance:liger_tvd",
        buffers=(
            _buffer("p_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("q_ptr", BufferRole.INPUT, "f32", (2, 64)),
            _buffer("loss_ptr", BufferRole.OUTPUT, "f32", (2, 64), "loss"),
            _buffer("grads_ptr", BufferRole.OUTPUT, "f32", (2, 64), "grads"),
            # HAS_LABEL=false makes this required ABI carrier unread.
            _buffer("label_ptr", BufferRole.INPUT, "i64", (2,)),
        ),
        scalars=(
            _scalar("p_stride", "i64", 64),
            _scalar("q_stride", "i64", 64),
            _scalar("loss_stride", "i64", 64),
            _scalar("grads_stride", "i64", 64),
            _scalar("n_cols", "i32", 64),
            _scalar("scale", "f32", 1.0),
        ),
        semantic_constexprs=(
            ("ignore_index", -100),
            ("HAS_LABEL", False),
            ("reduction", 0),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        seed=223,
    ),
    (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_kernel"
        ),
        buffers=(
            _buffer("x_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("h1_ptr", BufferRole.OUTPUT, "f32", (2, 127), "h1"),
            _buffer("w_ptr", BufferRole.INPUT, "f32", (127,)),
        ),
        scalars=(
            _scalar("eps", "f32", 1.0e-6),
            _scalar("stride", "i64", 127),
        ),
        semantic_constexprs=(
            ("N_COLS", 127),
            ("INCLUDE_WEIGHT", True),
        ),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=227,
    ),
    (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_add_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_add_kernel"
        ),
        buffers=(
            _buffer("x_ptr", BufferRole.INOUT, "f32", (2, 127), "x"),
            _buffer("y_ptr", BufferRole.INPUT, "f32", (2, 127)),
            _buffer("h1_ptr", BufferRole.OUTPUT, "f32", (2, 127), "h1"),
            _buffer("w_ptr", BufferRole.INPUT, "f32", (127,)),
        ),
        scalars=(
            _scalar("eps", "f32", 1.0e-6),
            _scalar("stride", "i64", 127),
        ),
        semantic_constexprs=(
            ("N_COLS", 127),
            ("INCLUDE_WEIGHT", True),
        ),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        seed=229,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_fwd_kernel_splitK"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_fwd_kernel_splitK"
        ),
        buffers=(
            _buffer("Q", BufferRole.INPUT, "f16", (1, 16, 1, 1, 16)),
            _buffer("K", BufferRole.INPUT, "f16", (1, 16, 1, 1, 16)),
            _buffer("V", BufferRole.INPUT, "f16", (1, 16, 1, 1, 16)),
            _buffer(
                "Out_splitK",
                BufferRole.OUTPUT,
                "f32",
                (1, 1, 1, 2, 16, 16),
                "out_splitk",
            ),
            _buffer(
                "LSE_splitk",
                BufferRole.OUTPUT,
                "f32",
                (1, 1, 1, 2, 16),
                "lse_splitk",
            ),
        ),
        scalars=(
            _scalar("sm_scale", "f32", 0.25),
            *(
                _scalar(name, "i64", value)
                for name, value in (
                    ("stride_qz", 256),
                    ("stride_qm", 16),
                    ("stride_qg", 16),
                    ("stride_qh", 16),
                    ("stride_qk", 1),
                    ("stride_kz", 256),
                    ("stride_kn", 16),
                    ("stride_kg", 16),
                    ("stride_kh", 16),
                    ("stride_kk", 1),
                    ("stride_vz", 256),
                    ("stride_vn", 16),
                    ("stride_vg", 16),
                    ("stride_vh", 16),
                    ("stride_vk", 1),
                    ("stride_osk_z", 512),
                    ("stride_osk_g", 512),
                    ("stride_osk_h", 512),
                    ("stride_osk_s", 256),
                    ("stride_osk_m", 16),
                    ("stride_osk_k", 1),
                    ("stride_lsek_z", 32),
                    ("stride_lsek_g", 32),
                    ("stride_lsek_h", 32),
                    ("stride_lsek_s", 16),
                    ("stride_lsek_m", 1),
                    ("stride_blocktablesz", 0),
                    ("stride_blocktablesl", 0),
                    ("stride_bias_b", 0),
                    ("stride_bias_g", 0),
                    ("stride_bias_h", 0),
                    ("stride_bias_qm", 0),
                    ("stride_bias_km", 0),
                )
            ),
        ),
        semantic_constexprs=(
            ("Z", 1),
            ("N_CTX_Q", 16),
            ("N_CTX_K", 16),
            ("BLOCK_N_PER_SPLIT", 8),
            ("H", 1),
            ("G", 1),
            ("BLOCK_DMODEL", 16),
            ("USE_SEQ_LEN", False),
            ("PACKED_PER_VAL", 1),
            ("N_GROUPS", 1),
            ("IS_SPLITK", True),
            ("SPLIT_K_EARLY_EXIT", False),
            ("IS_CAUSAL", False),
            ("IS_LOCAL", False),
            ("NUM_QUERIES_CAUSAL", 16),
            ("USE_PAGED_ATTENTION", False),
            ("PAGE_SIZE", 0),
            ("WINDOW_LEFT", -1),
            ("WINDOW_RIGHT", -1),
            ("WRITE_LSE", True),
            ("HAS_ADDITIVE_BIAS", False),
        ),
        c_values=(
            ("stride_k_fp8_scale_shift_z", 0),
            ("stride_k_fp8_scale_shift_n", 0),
            ("stride_k_fp8_scale_shift_g", 0),
            ("stride_k_fp8_scale_shift_h", 0),
            ("stride_v_fp8_scale_shift_z", 0),
            ("stride_v_fp8_scale_shift_n", 0),
            ("stride_v_fp8_scale_shift_g", 0),
            ("stride_v_fp8_scale_shift_h", 0),
            ("kv_cache_blocks_per_row", 0),
            ("BOUNDS_CHECKS_N", True),
            ("BLOCK_M", 16),
            ("BLOCK_N", 16),
            ("NUM_PROGRAMS_DIM2_CONST", 2),
            ("IS_HIP", False),
        ),
        grid=(1, 1, 2),
        grid_contract=("ceil_div(N_CTX_Q,BLOCK_M)", "Z*G*H", "split_k"),
        seed=241,
        none_arguments=tuple(
            NoneBinding(name)
            for name in (
                "block_tables",
                "Seq_len",
                "Seq_starts_k",
                "Seq_starts_q",
                "Seq_starts_q_multiplier",
                "additive_bias",
                "K_fp8_scale_shift",
                "V_fp8_scale_shift",
            )
        ),
        vararg_arity=1,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_splitK_reduce_varargs"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs"
        ),
        buffers=(
            _buffer("Out_splitK0", BufferRole.INPUT, "f32", (1, 1, 1, 2, 16)),
            _buffer("Out_splitK1", BufferRole.INPUT, "f32", (1, 1, 1, 2, 16)),
            _buffer("LSE_splitK0", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
            _buffer("LSE_splitK1", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
            _buffer("Out", BufferRole.OUTPUT, "f32", (1, 1, 1, 2, 16), "out"),
            _buffer("LSE", BufferRole.OUTPUT, "f32", (1, 1, 1, 2), "lse"),
        ),
        scalars=tuple(
            _scalar(name, "i64", value)
            for name, value in (
                ("stride_osk_z0", 32),
                ("stride_osk_z1", 32),
                ("stride_osk_g0", 32),
                ("stride_osk_g1", 32),
                ("stride_osk_h0", 32),
                ("stride_osk_h1", 32),
                ("stride_osk_m0", 16),
                ("stride_osk_m1", 16),
                ("stride_osk_k0", 1),
                ("stride_osk_k1", 1),
                ("stride_lsek_z0", 2),
                ("stride_lsek_z1", 2),
                ("stride_lsek_g0", 2),
                ("stride_lsek_g1", 2),
                ("stride_lsek_h0", 2),
                ("stride_lsek_h1", 2),
                ("stride_lsek_m0", 1),
                ("stride_lsek_m1", 1),
                ("stride_oz", 32),
                ("stride_og", 16),
                ("stride_oh", 16),
                ("stride_om", 16),
                ("stride_ok", 1),
                ("stride_lse_z", 2),
                ("stride_lse_g", 2),
                ("stride_lse_h", 2),
                ("stride_lse_m", 1),
            )
        ),
        semantic_constexprs=(
            ("head_dim", 16),
            ("H", 1),
            ("G", 1),
            ("WRITE_LSE", True),
        ),
        c_values=(("head_dim_pow_2", 16),),
        grid=(2, 1, 1),
        grid_contract=("M", "B*G*H", "1"),
        seed=233,
        vararg_arity=2,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_splitK_reduce_varargs_backward"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs_backward"
        ),
        buffers=(
            _buffer("Out_splitK0", BufferRole.INPUT, "f32", (1, 1, 1, 2, 16)),
            _buffer("Out_splitK1", BufferRole.INPUT, "f32", (1, 1, 1, 2, 16)),
            _buffer("LSE_splitK0", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
            _buffer("LSE_splitK1", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
            _buffer(
                "Dout_splitK0", BufferRole.OUTPUT, "f32", (1, 1, 1, 2, 16), "dout0"
            ),
            _buffer(
                "Dout_splitK1", BufferRole.OUTPUT, "f32", (1, 1, 1, 2, 16), "dout1"
            ),
            _buffer("DLSE_splitK0", BufferRole.OUTPUT, "f32", (1, 1, 1, 2), "dlse0"),
            _buffer("DLSE_splitK1", BufferRole.OUTPUT, "f32", (1, 1, 1, 2), "dlse1"),
            _buffer("Out", BufferRole.INPUT, "f32", (1, 2, 1, 1, 16)),
            _buffer("LSE", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
            _buffer("DOut", BufferRole.INPUT, "f32", (1, 2, 1, 1, 16)),
            _buffer("DLSE", BufferRole.INPUT, "f32", (1, 1, 1, 2)),
        ),
        scalars=tuple(
            _scalar(name, "i64", value)
            for name, value in (
                ("stride_osk_z0", 32),
                ("stride_osk_z1", 32),
                ("stride_osk_g0", 32),
                ("stride_osk_g1", 32),
                ("stride_osk_h0", 32),
                ("stride_osk_h1", 32),
                ("stride_osk_m0", 16),
                ("stride_osk_m1", 16),
                ("stride_osk_k0", 1),
                ("stride_osk_k1", 1),
                ("stride_lsek_z0", 2),
                ("stride_lsek_z1", 2),
                ("stride_lsek_g0", 2),
                ("stride_lsek_g1", 2),
                ("stride_lsek_h0", 2),
                ("stride_lsek_h1", 2),
                ("stride_lsek_m0", 1),
                ("stride_lsek_m1", 1),
                ("stride_oz", 32),
                ("stride_og", 16),
                ("stride_oh", 16),
                ("stride_om", 16),
                ("stride_ok", 1),
                ("stride_lse_z", 2),
                ("stride_lse_g", 2),
                ("stride_lse_h", 2),
                ("stride_lse_m", 1),
                ("stride_doz", 32),
                ("stride_dog", 16),
                ("stride_doh", 16),
                ("stride_dom", 16),
                ("stride_dok", 1),
                ("stride_dlse_z", 2),
                ("stride_dlse_g", 2),
                ("stride_dlse_h", 2),
                ("stride_dlse_m", 1),
            )
        ),
        semantic_constexprs=(
            ("BLOCK_SIZE", 16),
            ("H", 1),
            ("G", 1),
        ),
        c_values=(),
        grid=(2, 1, 1),
        grid_contract=("M", "B*G*H", "1"),
        seed=239,
        vararg_arity=2,
    ),
}


__all__ = [
    "BufferBinding",
    "BufferInputTransform",
    "BufferInitializer",
    "BufferRole",
    "NoneBinding",
    "PreparedTritonWorkload",
    "TRITON_WORKLOAD_ADAPTER_PROTOCOL",
    "TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL",
    "TRITON_WORKLOAD_ADAPTERS",
    "ScalarBinding",
    "TritonWorkloadAdapter",
    "prepare_triton_source_workload",
    "prepare_triton_workload",
]
