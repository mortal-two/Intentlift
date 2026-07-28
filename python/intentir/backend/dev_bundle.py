"""Shared provenance preparation for portable backend J_dev bundles.

This module stops CUDA and RVV exporters from independently reinterpreting a
development J receipt.  It performs no lowering and emits no target code; it
only validates the exact variant/workload pair and returns the immutable
semantic snapshot, real Source adapter invocation, and publication metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.dev_j_variant import JDevVariant
from intentir.backend.dev_workload import DevWorkloadConfig, prepare_dev_workload
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.workload_adapters import PreparedTritonWorkload


_BASELINE_POLICIES = frozenset({"source_and_refj", "target_refj_only"})


@dataclass(frozen=True)
class PreparedDevBundleInput:
    snapshot: ProductionSnapshot
    prepared: PreparedTritonWorkload
    workload_profile: str
    workload_valuation: Mapping[str, Any]
    j_provenance: Mapping[str, Any]
    snapshot_path: Path


def prepare_dev_bundle_input(
    base_snapshot: ProductionSnapshot,
    base_snapshot_path: Path,
    variant: JDevVariant,
    workload: DevWorkloadConfig,
    *,
    variant_path: Path | None,
    baseline_policy: str,
    source_unavailable_reason: str | None,
) -> PreparedDevBundleInput:
    """Validate and prepare one target-independent J_dev bundle input."""

    if not isinstance(base_snapshot, ProductionSnapshot):
        raise TypeError("development bundle input requires ProductionSnapshot")
    if not isinstance(base_snapshot_path, Path):
        raise TypeError("development bundle input requires a base snapshot path")
    if not isinstance(variant, JDevVariant):
        raise TypeError("development bundle input requires JDevVariant")
    if not isinstance(workload, DevWorkloadConfig):
        raise TypeError("development bundle input requires DevWorkloadConfig")
    if baseline_policy not in _BASELINE_POLICIES:
        raise ValueError(
            "development bundle baseline_policy must be one of "
            f"{sorted(_BASELINE_POLICIES)}, got {baseline_policy!r}"
        )
    if source_unavailable_reason is not None and (
        not isinstance(source_unavailable_reason, str)
        or not source_unavailable_reason
    ):
        raise ValueError("source_unavailable_reason must be null or non-empty")

    variant.validate_against_base(base_snapshot)
    if variant.source_kernel_id != workload.case_id:
        raise ValueError("J_dev and development workload kernel identity mismatch")
    if variant.shape_config_id != workload.shape_config_id:
        raise ValueError("J_dev and development workload shape config mismatch")
    if variant.dtype_config_id != workload.dtype_config_id:
        raise ValueError("J_dev and development workload dtype config mismatch")

    prepared = prepare_dev_workload(workload)
    snapshot = ProductionSnapshot(variant.j, variant.c)
    profile = workload.shape_config_id
    variant_record = variant.to_dict()
    workload_valuation = {
        "kind": "manual_dev_j_variant",
        "profile": profile,
        "status": "uncertified_backend_dev",
        "base_snapshot_digest": base_snapshot.snapshot_digest,
        "base_j_digest": base_snapshot.j_digest,
        "base_c_digest": base_snapshot.c_digest,
        "active_snapshot_digest": snapshot.snapshot_digest,
        "active_j_digest": snapshot.j_digest,
        "active_c_digest": snapshot.c_digest,
        "j_unchanged": False,
        "shape_config_id": workload.shape_config_id,
        "dtype_config_id": workload.dtype_config_id,
        "workload_config_digest": workload.config_digest,
        "baseline_policy": baseline_policy,
        "source_unavailable_reason": source_unavailable_reason,
        "upstream_evidence": list(workload.upstream_evidence),
    }
    j_provenance = {
        "protocol": variant_record["protocol"],
        "receipt_digest": variant.receipt_digest,
        "base_snapshot_digest": variant.base_snapshot_digest,
        "base_j_digest": variant.base_j_digest,
        "variant_j_digest": variant.variant_j_digest,
        "variant_c_digest": variant.variant_c_digest,
        "certification_status": variant_record["certification_status"],
        "allowed_claims": list(variant_record["allowed_claims"]),
        "j_unchanged": False,
    }
    return PreparedDevBundleInput(
        snapshot=snapshot,
        prepared=prepared,
        workload_profile=profile,
        workload_valuation=workload_valuation,
        j_provenance=j_provenance,
        snapshot_path=variant_path or base_snapshot_path,
    )


__all__ = ["PreparedDevBundleInput", "prepare_dev_bundle_input"]
