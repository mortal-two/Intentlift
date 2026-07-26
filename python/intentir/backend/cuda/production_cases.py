"""Stable names for the closed production snapshots and their source baselines."""

from __future__ import annotations

from typing import Any, Mapping

from intentir.verify.gate1.model import ParameterKind, SemanticArchive


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


__all__ = ["PRODUCTION_CASE_IDS", "constexpr_input_bindings"]
