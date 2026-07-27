"""Stable names for the closed production snapshots and their source baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.gate1.model import (
    ImplementationConfig,
    ParameterKind,
    SemanticArchive,
)
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    PreparedTritonWorkload,
    ScalarBinding,
    TRITON_WORKLOAD_ADAPTERS,
    TritonWorkloadAdapter,
    prepare_triton_source_workload,
    prepare_triton_workload,
)


PRODUCTION_CASE_IDS: dict[str, str] = {
    "dyt": "high_performance:liger_dyt",
    "flash_bwd_preprocess": (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_preprocess_do_o_dot"
    ),
    "flash_layer_norm_1pass": (
        "high_performance:flash_attn.ops.triton.layer_norm."
        "_layer_norm_fwd_1pass_kernel"
    ),
    "fused_add_rms_norm": "high_performance:liger_fused_add_rms_norm",
    "geglu": "high_performance:liger_geglu",
    "jsd": "high_performance:liger_jsd",
    "kl_div": "high_performance:liger_kl_div",
    "layer_norm": "high_performance:liger_layer_norm",
    "liger_rms_norm": "high_performance:liger_rms_norm",
    "masked_copy": "scrc_canary:masked_copy",
    "masked_row_square_sum": "scrc_canary:masked_row_square_sum",
    "poly_norm": "high_performance:liger_poly_norm",
    "softmax": "high_performance:liger_softmax",
    "sparsemax": "high_performance:liger_sparsemax",
    "swiglu": "high_performance:liger_swiglu",
    "tvd": "high_performance:liger_tvd",
    "xformers_rms_norm": (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_kernel"
    ),
    "xformers_rms_norm_add": (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_add_kernel"
    ),
}

PRODUCTION_WORKLOAD_PROFILE_CLOSED = "closed-v1"
PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE = "roofline-large-j-only-v1"
PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM = "symbolic-medium-v1"
PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE = "symbolic-large-v1"
_ROOFLINE_MASKED_COPY_EXTENT = 16 * 1024 * 1024

_SYMBOLIC_ROW_PROFILE_KERNELS = frozenset(
    {
        "flash_layer_norm_1pass",
        "jsd",
        "layer_norm",
        "sparsemax",
        "tvd",
    }
)
_SYMBOLIC_PROFILE_KERNELS = frozenset(
    {*_SYMBOLIC_ROW_PROFILE_KERNELS, "flash_bwd_preprocess"}
)
_SYMBOLIC_PROFILES = frozenset(
    {
        PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
        PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    }
)

_ROW = "row"
_COL = "col"
_SYMBOLIC_ROW_BUFFER_AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "flash_layer_norm_1pass": {
        "X": (_ROW, _COL),
        "Y": (_ROW, _COL),
        "W": (_COL,),
        "B": (_COL,),
        "RESIDUAL": (_ROW, _COL),
        "X1": (_ROW, _COL),
        "W1": (_COL,),
        "B1": (_COL,),
        "Y1": (_ROW, _COL),
        "RESIDUAL_OUT": (_ROW, _COL),
        "ROWSCALE": (_ROW,),
        "SEEDS": (_ROW,),
        "DROPOUT_MASK": (_ROW, _COL),
        "DROPOUT_MASK1": (_ROW, _COL),
        "Mean": (_ROW,),
        "Rstd": (_ROW,),
    },
    "jsd": {
        "X_ptr": (_ROW, _COL),
        "Y_ptr": (_ROW, _COL),
        "loss_ptr": (_ROW, _COL),
        "dX_ptr": (_ROW, _COL),
        "label_ptr": (_ROW,),
    },
    "layer_norm": {
        "Y_ptr": (_ROW, _COL),
        "X_ptr": (_ROW, _COL),
        "W_ptr": (_COL,),
        "B_ptr": (_COL,),
        "Mean_ptr": (_ROW,),
        "RSTD_ptr": (_ROW,),
    },
    "sparsemax": {
        "x_ptr": (_ROW, _COL),
        "sorted_x_ptr": (_ROW, _COL),
        "o_ptr": (_ROW, _COL),
    },
    "tvd": {
        "p_ptr": (_ROW, _COL),
        "q_ptr": (_ROW, _COL),
        "loss_ptr": (_ROW, _COL),
        "grads_ptr": (_ROW, _COL),
        "label_ptr": (_ROW,),
    },
}
_SYMBOLIC_ROW_SCALAR_AXES: dict[str, dict[str, str]] = {
    "flash_layer_norm_1pass": {
        "stride_x_row": _COL,
        "stride_y_row": _COL,
        "stride_res_row": _COL,
        "stride_res_out_row": _COL,
        "stride_x1_row": _COL,
        "stride_y1_row": _COL,
        "M": _ROW,
        "N": _COL,
    },
    "jsd": {
        "X_stride": _COL,
        "Y_stride": _COL,
        "loss_stride": _COL,
        "dX_stride": _COL,
        "n_non_ignore": _ROW,
        "n_cols": _COL,
    },
    "layer_norm": {
        "Y_row_stride": _COL,
        "X_row_stride": _COL,
        "W_row_stride": _COL,
        "B_row_stride": _COL,
        "n_cols": _COL,
    },
    "sparsemax": {
        "x_stride_row": _COL,
        "sorted_x_stride_row": _COL,
        "o_stride_row": _COL,
        "n_cols": _COL,
    },
    "tvd": {
        "p_stride": _COL,
        "q_stride": _COL,
        "loss_stride": _COL,
        "grads_stride": _COL,
        "n_cols": _COL,
    },
}


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_MANIFEST = (
    _REPOSITORY_ROOT / "corpus" / "production-source-v1" / "manifest.json"
)
_CERTIFIED_CANARY_ADAPTERS: dict[str, TritonWorkloadAdapter] = {
    "scrc_canary:masked_copy": TritonWorkloadAdapter(
        case_id="scrc_canary:masked_copy",
        buffers=(
            BufferBinding(
                "X",
                BufferRole.INPUT,
                "f32",
                (1000,),
                initializer=BufferInitializer.STANDARD_NORMAL,
            ),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output0"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=11,
    ),
    "scrc_canary:masked_row_square_sum": TritonWorkloadAdapter(
        case_id="scrc_canary:masked_row_square_sum",
        buffers=(
            BufferBinding(
                "X",
                BufferRole.INPUT,
                "f32",
                (2, 1000),
                initializer=BufferInitializer.STANDARD_NORMAL,
            ),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (2,), "output0"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 1024),),
        grid=(2,),
        grid_contract=("2",),
        seed=19,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _symbolic_row_shape(name: str, profile: str) -> tuple[int, int]:
    if profile == PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM:
        return 128, 1024
    if profile == PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE:
        if name == "sparsemax":
            return 1024, 2048
        return 2048, 4096
    raise ValueError(f"unknown symbolic workload profile {profile!r}")


def _scale_symbolic_row_adapter(
    name: str,
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    rows, cols = _symbolic_row_shape(name, profile)
    buffer_axes = _SYMBOLIC_ROW_BUFFER_AXES[name]
    actual_buffer_formals = {item.formal for item in adapter.buffers}
    if set(buffer_axes) != actual_buffer_formals:
        raise ValueError(
            f"symbolic buffer-axis map for {name!r} does not match its adapter"
        )

    def extent(axis: str) -> int:
        if axis == _ROW:
            return rows
        if axis == _COL:
            return cols
        raise ValueError(f"unsupported symbolic profile axis {axis!r}")

    buffers = []
    for item in adapter.buffers:
        axes = buffer_axes[item.formal]
        if len(axes) != len(item.shape):
            raise ValueError(
                f"symbolic axes for {name!r}/{item.formal!r} disagree with rank"
            )
        buffers.append(replace(item, shape=tuple(extent(axis) for axis in axes)))

    scalar_axes = _SYMBOLIC_ROW_SCALAR_AXES[name]
    actual_scalar_formals = {item.formal for item in adapter.scalars}
    unknown_scalar_formals = set(scalar_axes) - actual_scalar_formals
    if unknown_scalar_formals:
        raise ValueError(
            f"symbolic scalar-axis map for {name!r} names missing formals: "
            f"{sorted(unknown_scalar_formals)}"
        )
    scalars = tuple(
        replace(item, value=extent(scalar_axes[item.formal]))
        if item.formal in scalar_axes
        else item
        for item in adapter.scalars
    )
    block_extent = _next_power_of_two(cols)
    c_values = tuple(
        (
            key,
            block_extent if key in {"BLOCK_N", "BLOCK_SIZE"} else value,
        )
        for key, value in adapter.c_values
    )
    return replace(
        adapter,
        buffers=tuple(buffers),
        scalars=scalars,
        c_values=c_values,
        grid=(rows,),
    )


def _scale_symbolic_flash_preprocess_adapter(
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    heads = (
        32
        if profile == PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
        else 256
    )
    out = next(item for item in adapter.buffers if item.formal == "Out")
    if len(out.shape) != 4:
        raise ValueError("Flash preprocess Out buffer must remain rank-4")
    batch, _, sequence, head_dim = out.shape
    buffers = []
    for item in adapter.buffers:
        if item.formal in {"Out", "DO"}:
            if len(item.shape) != 4 or (
                item.shape[0], item.shape[2], item.shape[3]
            ) != (batch, sequence, head_dim):
                raise ValueError("Flash preprocess input layouts disagree")
            shape = (batch, heads, sequence, head_dim)
        elif item.formal == "Delta":
            if len(item.shape) != 3 or item.shape[:2] != (batch, out.shape[1]):
                raise ValueError("Flash preprocess Delta layout disagrees")
            shape = (batch, heads, item.shape[2])
        else:
            raise ValueError(
                f"unexpected Flash preprocess buffer formal {item.formal!r}"
            )
        buffers.append(replace(item, shape=shape))
    batch_stride = heads * sequence * head_dim
    scalars = tuple(
        replace(
            item,
            value=(
                heads
                if item.formal == "nheads"
                else batch_stride
                if item.formal in {"stride_ob", "stride_dob"}
                else item.value
            ),
        )
        for item in adapter.scalars
    )
    block_m = int(dict(adapter.c_values)["BLOCK_M"])
    return replace(
        adapter,
        buffers=tuple(buffers),
        scalars=scalars,
        grid=((sequence + block_m - 1) // block_m, batch * heads),
    )


def _apply_symbolic_workload_profile(
    name: str,
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    if name not in _SYMBOLIC_PROFILE_KERNELS:
        raise ValueError(
            f"symbolic workload profile {profile!r} is not registered for {name!r}"
        )
    if name in _SYMBOLIC_ROW_PROFILE_KERNELS:
        return _scale_symbolic_row_adapter(name, adapter, profile)
    return _scale_symbolic_flash_preprocess_adapter(adapter, profile)


def _apply_workload_profile(
    name: str,
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    if profile == PRODUCTION_WORKLOAD_PROFILE_CLOSED:
        return adapter
    if profile in _SYMBOLIC_PROFILES:
        return _apply_symbolic_workload_profile(name, adapter, profile)
    if profile != PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE:
        raise ValueError(f"unknown production workload profile {profile!r}")
    # Workload profiles define the backend stress domain only.  They never enter
    # compiler path selection.  The first large profile is intentionally a pure
    # copy canary: its 128 MiB J boundary exceeds both RTX 5090D and H100 L2.
    # The accepted Source has a one-block execution contract, so this profile is
    # Target--RefJ-only and must never be reported as Source/Replay closure.
    if name != "masked_copy":
        raise ValueError(
            f"workload profile {profile!r} is not registered for {name!r}"
        )
    extent = _ROOFLINE_MASKED_COPY_EXTENT
    buffers = tuple(
        replace(item, shape=(extent,))
        if item.formal in {"X", "Y"}
        else item
        for item in adapter.buffers
    )
    scalars = tuple(
        replace(item, value=extent) if item.formal == "N" else item
        for item in adapter.scalars
    )
    block_size = dict(adapter.c_values)["BLOCK_SIZE"]
    return replace(
        adapter,
        buffers=buffers,
        scalars=scalars,
        grid=((extent + int(block_size) - 1) // int(block_size),),
        grid_contract=("ceil_div(N,BLOCK_SIZE)",),
    )


def profile_production_snapshot(
    snapshot: ProductionSnapshot,
    prepared: PreparedTritonWorkload,
    profile: str,
) -> tuple[ProductionSnapshot, dict[str, Any]]:
    """Revalue C for a workload profile while preserving the exact accepted J."""

    if profile not in _SYMBOLIC_PROFILES:
        return snapshot, {
            "kind": "accepted_snapshot",
            "base_snapshot_digest": snapshot.snapshot_digest,
            "base_j_digest": snapshot.j_digest,
            "active_snapshot_digest": snapshot.snapshot_digest,
            "active_j_digest": snapshot.j_digest,
            "j_unchanged": True,
        }
    if snapshot.c is None:
        raise ValueError("symbolic workload profile requires accepted C slots")
    c_payload = snapshot.c.to_json()
    source_to_opaque = dict(prepared.source_to_opaque)
    for source_name, value in prepared.adapter.c_values:
        opaque_name = source_to_opaque.get(source_name)
        if opaque_name is None or opaque_name not in c_payload["values"]:
            raise ValueError(
                f"profile C value {source_name!r} has no accepted opaque slot"
            )
        c_payload["values"][opaque_name] = value
    profiled_c = ImplementationConfig.from_json(c_payload)
    active = ProductionSnapshot(j=snapshot.j, c=profiled_c)
    if active.j_digest != snapshot.j_digest:
        raise AssertionError("workload profile modified accepted J")
    return active, {
        "kind": "symbolic_extent_valuation",
        "profile": profile,
        "base_snapshot_digest": snapshot.snapshot_digest,
        "base_j_digest": snapshot.j_digest,
        "base_c_digest": snapshot.c_digest,
        "active_snapshot_digest": active.snapshot_digest,
        "active_j_digest": active.j_digest,
        "active_c_digest": active.c_digest,
        "j_unchanged": True,
    }


def prepare_production_workload(
    name: str,
    *,
    profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
) -> PreparedTritonWorkload:
    """Resolve an active Source or an exact content-checked certified canary."""

    case_id = PRODUCTION_CASE_IDS[name]
    if case_id in TRITON_WORKLOAD_ADAPTERS:
        adapter = _apply_workload_profile(
            name, TRITON_WORKLOAD_ADAPTERS[case_id], profile
        )
        return prepare_triton_workload(adapter)
    adapter = _CERTIFIED_CANARY_ADAPTERS.get(case_id)
    if adapter is None:
        raise KeyError(f"production case {name!r} has no Source adapter")
    payload = json.loads(_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    record = payload.get(name)
    if not isinstance(record, dict) or set(record) != {
        "case_id",
        "path",
        "sha256",
        "symbol",
    }:
        raise ValueError(f"certified Source manifest entry {name!r} is invalid")
    if record["case_id"] != case_id:
        raise ValueError("certified Source manifest case_id mismatch")
    source_path = (_REPOSITORY_ROOT / record["path"]).resolve()
    if _REPOSITORY_ROOT not in source_path.parents:
        raise ValueError("certified Source path escapes the repository")
    if _sha256(source_path) != record["sha256"]:
        raise ValueError("certified Source digest mismatch")
    profiled_adapter = _apply_workload_profile(name, adapter, profile)
    return prepare_triton_source_workload(
        profiled_adapter,
        source_path,
        source_symbol=record["symbol"],
        source_location=f"corpus://production-source-v1/{source_path.name}",
    )


def constexpr_input_bindings(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> dict[str, bool | int | float]:
    """Extract invocation values only for J-declared constexpr inputs."""

    logical_inputs = {item.name for item in j.intent.inputs}
    result: dict[str, bool | int | float] = {}
    for parameter in j.signature.parameters:
        logical_ref = parameter.logical_ref
        if (
            parameter.kind is not ParameterKind.CONSTEXPR
            or logical_ref not in logical_inputs
        ):
            continue
        if logical_ref not in input_bindings:
            raise ValueError(
                f"constexpr logical input {logical_ref!r} has no runtime binding"
            )
        runtime_key = input_bindings[logical_ref]
        if runtime_key not in values:
            raise ValueError(
                f"constexpr runtime binding {runtime_key!r} has no value"
            )
        value = values[runtime_key]
        if hasattr(value, "size") and int(value.size) != 1:
            raise ValueError(
                f"constexpr logical input {logical_ref!r} is not scalar"
            )
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            result[logical_ref] = value
        elif isinstance(value, int):
            result[logical_ref] = value
        elif isinstance(value, float):
            result[logical_ref] = value
        else:
            raise ValueError(
                f"constexpr logical input {logical_ref!r} has unsupported "
                f"value type {type(value).__name__}"
            )
    return result


__all__ = [
    "PRODUCTION_CASE_IDS",
    "PRODUCTION_WORKLOAD_PROFILE_CLOSED",
    "PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM",
    "constexpr_input_bindings",
    "profile_production_snapshot",
    "prepare_production_workload",
]
