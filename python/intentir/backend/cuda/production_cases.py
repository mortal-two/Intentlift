"""Stable names for the closed production snapshots and their source baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from intentir.verify.gate1.model import ParameterKind, SemanticArchive
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


def prepare_production_workload(name: str) -> PreparedTritonWorkload:
    """Resolve an active Source or an exact content-checked certified canary."""

    case_id = PRODUCTION_CASE_IDS[name]
    if case_id in TRITON_WORKLOAD_ADAPTERS:
        return prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[case_id])
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
    return prepare_triton_source_workload(
        adapter,
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
    "constexpr_input_bindings",
    "prepare_production_workload",
]
