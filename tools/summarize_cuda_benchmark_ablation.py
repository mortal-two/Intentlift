#!/usr/bin/env python3
"""Pair target-side CUDA E2/E3 reports under one measurement contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPORT_PROTOCOL = "intentir.production.cuda-remote-performance.v3"
SUMMARY_PROTOCOL = "intentir.production.cuda-portable-ablation-summary.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != REPORT_PROTOCOL:
        raise ValueError(f"{path} is not a current CUDA performance report")
    return value


def _arm(report: Mapping[str, Any]) -> str:
    axes = report.get("ablation_axes")
    if not isinstance(axes, Mapping) or axes.get("arm") not in {"E2", "E3"}:
        raise ValueError("CUDA report lacks a frozen E2/E3 ablation arm")
    return str(axes["arm"])


def _measurement_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    measurement = report.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("CUDA report lacks measurement metadata")
    keys = (
        "data_regime",
        "effective_timing",
        "graph_repetitions",
        "warmup_ms",
        "cold_candidate_warmup_ms",
        "cold_candidate_warmup_scope",
        "repeat_ms",
        "measurement_rounds",
        "cold_iterations_per_round",
        "cache_flush_bytes",
        "detected_l2_cache_bytes",
        "cache_eviction_in_timed_region",
    )
    return {key: measurement.get(key) for key in keys}


def _device_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    device = report["device"]
    return {
        "device_name": device["device_name"],
        "compute_capability": device["compute_capability"],
        "total_memory_bytes": device["total_memory_bytes"],
        "l2_cache_bytes": device.get("l2_cache_bytes"),
    }


def _records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in report["kernels"]:
        kernel = str(item["kernel"])
        if kernel in result:
            raise ValueError(f"CUDA report contains duplicate kernel {kernel!r}")
        result[kernel] = item
    return result


def _record_status(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "missing"
    if "performance" in record:
        return "success"
    return str(record.get("status", "failed"))


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(item <= 0 for item in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(item) for item in values) / len(values))


def _topology_digests(record: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for candidate in record.get("candidates", []):
        if candidate.get("status") != "measured":
            continue
        state = candidate.get("concrete_state")
        if isinstance(state, Mapping) and state.get("topology_digest"):
            result.add(str(state["topology_digest"]))
    return result


def summarize(e2: Mapping[str, Any], e3: Mapping[str, Any]) -> dict[str, Any]:
    if _arm(e2) != "E2" or _arm(e3) != "E3":
        raise ValueError("reports must be passed as E2 then E3")
    for key in (
        "target",
        "bundle_toolchain_digest",
        "experiment_spec_digest",
        "measurement_spec_protocol",
        "measurement_spec_digest",
        "workload_profile",
        "baseline_policy",
    ):
        if e2.get(key) != e3.get(key):
            raise ValueError(f"E2/E3 differ in {key}")
    if _device_identity(e2) != _device_identity(e3):
        raise ValueError("E2/E3 were not measured on the same CUDA device")
    measurement = _measurement_identity(e2)
    if measurement != _measurement_identity(e3):
        raise ValueError("E2/E3 measurement contracts differ")

    e2_records = _records(e2)
    e3_records = _records(e3)
    kernel_universe = sorted(set(e2_records) | set(e3_records))
    per_kernel: list[dict[str, Any]] = []
    e3_over_e2: list[float] = []
    cross_report_e3_over_e2: list[float] = []
    e2_over_source: list[float] = []
    e3_over_source: list[float] = []
    c_speedups: list[float] = []
    structurally_expanded: list[str] = []
    paired_success_count = 0
    for kernel in kernel_universe:
        left = e2_records.get(kernel)
        right = e3_records.get(kernel)
        e2_status = _record_status(left)
        e3_status = _record_status(right)
        if e2_status != "success" or e3_status != "success":
            identity = left or right or {}
            per_kernel.append(
                {
                    "kernel": kernel,
                    "case_id": identity.get("case_id"),
                    "snapshot_digest": identity.get("snapshot_digest"),
                    "j_digest": identity.get("j_digest"),
                    "c_digest": identity.get("c_digest"),
                    "extents": identity.get("extents"),
                    "workload_valuation": identity.get("workload_valuation"),
                    "workload_profile": identity.get(
                        "workload_profile", e2.get("workload_profile")
                    ),
                    "baseline_policy": identity.get(
                        "baseline_policy", e2.get("baseline_policy")
                    ),
                    "status": "failed_or_missing",
                    "e2_status": e2_status,
                    "e3_status": e3_status,
                    "e2_error": left.get("error") if left is not None else None,
                    "e3_error": right.get("error") if right is not None else None,
                }
            )
            continue
        assert left is not None and right is not None
        for key in (
            "case_id",
            "snapshot_digest",
            "j_digest",
            "c_digest",
            "extents",
            "workload_valuation",
            "workload_profile",
            "baseline_policy",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(f"{kernel} E2/E3 differ in {key}")
        if left.get("j_digest") is None:
            raise ValueError(f"{kernel} CUDA reports lack J identity")
        e2_us = float(left["performance"]["mlir_us"])
        e3_us = float(right["performance"]["mlir_us"])
        e2_source_value = left["performance"].get("source_triton_us")
        e3_source_value = right["performance"].get("source_triton_us")
        if (e2_source_value is None) != (e3_source_value is None):
            raise ValueError(f"{kernel} E2/E3 source baseline availability differs")
        e2_source_us = (
            float(e2_source_value) if e2_source_value is not None else None
        )
        e3_source_us = (
            float(e3_source_value) if e3_source_value is not None else None
        )
        structural = bool(_topology_digests(right) - _topology_digests(left))
        if structural:
            structurally_expanded.append(kernel)
        c_gain = right["performance"]["c_ordering_ablation"].get(
            "c_prior_over_erase_c_speedup"
        )
        if c_gain is not None:
            c_speedups.append(float(c_gain))
        cross_report_gain = e2_us / e3_us
        spine_ablation = right["performance"].get("schedule_spine_ablation")
        same_receipt_gain = None
        if isinstance(spine_ablation, Mapping):
            if not spine_ablation.get("same_candidate_measurement_receipt"):
                raise ValueError(
                    f"{kernel} schedule-spine comparison is not same-receipt"
                )
            same_receipt_gain = float(
                spine_ablation["expanded_over_schedule_spine_speedup"]
            )
            if same_receipt_gain <= 0:
                raise ValueError(f"{kernel} has a non-positive spine speedup")
        primary_gain = (
            same_receipt_gain
            if same_receipt_gain is not None
            else cross_report_gain
        )
        record = {
            "kernel": kernel,
            "case_id": left.get("case_id"),
            "snapshot_digest": left.get("snapshot_digest"),
            "j_digest": left["j_digest"],
            "c_digest": left.get("c_digest"),
            "extents": left.get("extents"),
            "workload_valuation": left.get("workload_valuation"),
            "workload_profile": left.get(
                "workload_profile", e2.get("workload_profile")
            ),
            "baseline_policy": left.get(
                "baseline_policy", e2.get("baseline_policy")
            ),
            "status": "paired_success",
            "e2_status": e2_status,
            "e3_status": e3_status,
            "e2_us": e2_us,
            "e3_us": e3_us,
            "e3_over_e2_speedup": primary_gain,
            "same_receipt_e3_over_e2_speedup": same_receipt_gain,
            "cross_report_e3_over_e2_speedup": cross_report_gain,
            "schedule_spine_best_us_in_e3_receipt": (
                spine_ablation.get("schedule_spine_best_us")
                if isinstance(spine_ablation, Mapping)
                else None
            ),
            "expanded_best_us_in_e3_receipt": (
                spine_ablation.get("expanded_best_us")
                if isinstance(spine_ablation, Mapping)
                else None
            ),
            "e2_speedup_over_source": (
                e2_source_us / e2_us if e2_source_us is not None else None
            ),
            "e3_speedup_over_source": (
                e3_source_us / e3_us if e3_source_us is not None else None
            ),
            "e2_source_us": e2_source_us,
            "e3_source_us": e3_source_us,
            "structural_state_expansion_observed": structural,
            "e2_topology_count": len(_topology_digests(left)),
            "e3_topology_count": len(_topology_digests(right)),
            "c_prior_over_erase_c_speedup": c_gain,
        }
        per_kernel.append(record)
        paired_success_count += 1
        e3_over_e2.append(record["e3_over_e2_speedup"])
        cross_report_e3_over_e2.append(cross_report_gain)
        if record["e2_speedup_over_source"] is not None:
            e2_over_source.append(record["e2_speedup_over_source"])
            e3_over_source.append(record["e3_speedup_over_source"])
    return {
        "protocol": SUMMARY_PROTOCOL,
        "target": e2["target"],
        "workload_profile": e2.get("workload_profile"),
        "baseline_policy": e2.get("baseline_policy"),
        "device": _device_identity(e2),
        "measurement": measurement,
        "experiment_spec_digest": e2["experiment_spec_digest"],
        "measurement_spec_protocol": e2.get("measurement_spec_protocol"),
        "measurement_spec_digest": e2.get("measurement_spec_digest"),
        "bundle_toolchain_digest": e2["bundle_toolchain_digest"],
        "summary": {
            "kernel_count": len(kernel_universe),
            "paired_success_count": paired_success_count,
            "failed_or_missing_count": len(kernel_universe) - paired_success_count,
            "source_baseline_kernel_count": len(e2_over_source),
            "geomean_e3_over_e2_speedup": _geomean(e3_over_e2),
            "geomean_cross_report_e3_over_e2_speedup": _geomean(
                cross_report_e3_over_e2
            ),
            "primary_e3_e2_estimand": (
                "same-E3-receipt schedule-spine/expanded best when available; "
                "cross-report ratio only for legacy reports"
            ),
            "geomean_e2_speedup_over_source": _geomean(e2_over_source),
            "geomean_e3_speedup_over_source": _geomean(e3_over_source),
            "geomean_c_prior_over_erase_c_speedup": _geomean(c_speedups),
            "e3_over_e2_win_count": sum(item > 1 for item in e3_over_e2),
            "structural_state_expansion_count": len(structurally_expanded),
            "structural_state_expansion_kernels": structurally_expanded,
        },
        "per_kernel": per_kernel,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = summarize(_load(args.e2), _load(args.e3))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.out} sha256={hashlib.sha256(encoded.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
