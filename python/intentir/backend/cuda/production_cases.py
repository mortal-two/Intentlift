"""Stable names for the closed production snapshots and their source baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.workload_valuation import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
    apply_workload_profile,
    case_name_for_id,
    workload_profile_metadata,
)
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
from intentir.verify.gate2.policy import (
    ComparisonPolicy,
    DEFAULT_GATE2_POLICY,
    NumericTolerance,
)


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

# Q is an external workload policy, not a permission encoded in J.  Natural
# headline runs use the same f32 thresholds as the corresponding upstream test
# suites; all other dtype rules remain the project defaults.  Closed SCRC
# valuations keep DEFAULT_GATE2_POLICY byte-for-byte.
_NATURAL_F32_Q: dict[str, NumericTolerance] = {
    "layer_norm": NumericTolerance(1.0e-5, 1.0e-5),
    "poly_norm": NumericTolerance(1.0e-6, 1.0e-4),
    "flash_layer_norm_1pass": NumericTolerance(1.0e-3, 1.0e-4),
}


def comparison_policy_for_workload(
    name: str,
    profile: str,
) -> ComparisonPolicy:
    tolerance = (
        _NATURAL_F32_Q.get(name)
        if profile == PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE
        else None
    )
    if tolerance is None:
        return DEFAULT_GATE2_POLICY
    rules = dict(DEFAULT_GATE2_POLICY.tolerances)
    rules["f32"] = tolerance
    return ComparisonPolicy.create(
        f"intentir.gate2.source-{name.replace('_', '-')}.v1",
        rules.items(),
    )


def _apply_workload_profile(
    name: str,
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    return apply_workload_profile(name, adapter, profile)


def profile_production_snapshot(
    snapshot: ProductionSnapshot,
    prepared: PreparedTritonWorkload,
    profile: str,
) -> tuple[ProductionSnapshot, dict[str, Any]]:
    """Revalue C for a workload profile while preserving the exact accepted J."""

    name = case_name_for_id(prepared.adapter.case_id)
    metadata = workload_profile_metadata(name, profile)
    if profile == PRODUCTION_WORKLOAD_PROFILE_CLOSED:
        return snapshot, {
            "kind": "accepted_snapshot",
            **metadata,
            "j_unchanged": True,
        }
    if snapshot.c is None:
        raise ValueError("workload profile requires accepted C slots")
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
    if active.j.to_json() != snapshot.j.to_json():
        raise AssertionError("workload profile modified accepted J")
    return active, {
        "kind": "symbolic_extent_valuation",
        **metadata,
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
    if not isinstance(record, dict) or not {
        "case_id",
        "path",
        "symbol",
    }.issubset(record):
        raise ValueError(f"certified Source manifest entry {name!r} is invalid")
    if record["case_id"] != case_id:
        raise ValueError("certified Source manifest case_id mismatch")
    source_path = (_REPOSITORY_ROOT / record["path"]).resolve()
    if _REPOSITORY_ROOT not in source_path.parents:
        raise ValueError("certified Source path escapes the repository")
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
    "PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE",
    "PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM",
    "workload_profile_metadata",
    "comparison_policy_for_workload",
    "constexpr_input_bindings",
    "profile_production_snapshot",
    "prepare_production_workload",
]
