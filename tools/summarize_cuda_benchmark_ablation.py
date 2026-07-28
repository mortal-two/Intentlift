#!/usr/bin/env python3
"""Pair CUDA C-Fixed/Schedule-Only/IntentIR reports under one contract."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


REPORT_PROTOCOL = "intentir.production.cuda-remote-performance.v3"
SUMMARY_PROTOCOL = "intentir.production.cuda-portable-ablation-summary.v3"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != REPORT_PROTOCOL:
        raise ValueError(f"{path} is not a current CUDA performance report")
    return value


def _arm(report: Mapping[str, Any]) -> str:
    axes = report.get("ablation_axes")
    if not isinstance(axes, Mapping) or axes.get("arm") not in {"CF", "E2", "E3"}:
        raise ValueError(
            "CUDA report lacks a C-Fixed/Schedule-Only/IntentIR ablation arm"
        )
    return str(axes["arm"])


def _measurement_config(report: Mapping[str, Any]) -> dict[str, Any]:
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


def _device_config(report: Mapping[str, Any]) -> dict[str, Any]:
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


def _structural_paths(record: Mapping[str, Any]) -> set[tuple[str, tuple[str, ...]]]:
    result: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in record.get("candidates", []):
        if candidate.get("status") != "measured":
            continue
        state = candidate.get("concrete_state")
        if isinstance(state, Mapping):
            result.add(
                (
                    str(state.get("lowering_path_kind", "unknown")),
                    tuple(str(item) for item in state.get("transformation_trace", ())),
                )
            )
    return result


def summarize(
    e2: Mapping[str, Any],
    e3: Mapping[str, Any],
    c_fixed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _arm(e2) != "E2" or _arm(e3) != "E3":
        raise ValueError("reports must be passed as E2 then E3")
    if c_fixed is not None and _arm(c_fixed) != "CF":
        raise ValueError("C-Fixed report must carry the CF arm")
    reports = (e2, e3) if c_fixed is None else (c_fixed, e2, e3)
    for key in (
        "target",
        "measurement_spec_protocol",
        "workload_profile",
        "baseline_policy",
    ):
        if any(report.get(key) != e2.get(key) for report in reports):
            raise ValueError(f"CF/E2/E3 differ in {key}")
    if any(_device_config(report) != _device_config(e2) for report in reports):
        raise ValueError("CF/E2/E3 were not measured on the same CUDA device")
    measurement = _measurement_config(e2)
    if any(_measurement_config(report) != measurement for report in reports):
        raise ValueError("CF/E2/E3 measurement contracts differ")

    cf_records = _records(c_fixed) if c_fixed is not None else {}
    e2_records = _records(e2)
    e3_records = _records(e3)
    kernel_universe = sorted(set(cf_records) | set(e2_records) | set(e3_records))
    per_kernel: list[dict[str, Any]] = []
    schedule_gain: list[float] = []
    lowering_gain: list[float] = []
    cross_report_lowering_gain: list[float] = []
    schedule_only_over_source: list[float] = []
    intentir_over_source: list[float] = []
    c_speedups: list[float] = []
    c_candidates_to_95pct: list[int] = []
    erase_c_candidates_to_95pct: list[int] = []
    structurally_expanded: list[str] = []
    paired_success_count = 0
    for kernel in kernel_universe:
        left = e2_records.get(kernel)
        right = e3_records.get(kernel)
        fixed = cf_records.get(kernel)
        cf_status = _record_status(fixed) if c_fixed is not None else "not_requested"
        e2_status = _record_status(left)
        e3_status = _record_status(right)
        if (
            e2_status != "success"
            or e3_status != "success"
            or (c_fixed is not None and cf_status != "success")
        ):
            available = fixed or left or right or {}
            per_kernel.append(
                {
                    "kernel": kernel,
                    "case_id": available.get("case_id"),
                    "extents": available.get("extents"),
                    "workload_valuation": available.get("workload_valuation"),
                    "workload_profile": available.get(
                        "workload_profile", e2.get("workload_profile")
                    ),
                    "baseline_policy": available.get(
                        "baseline_policy", e2.get("baseline_policy")
                    ),
                    "status": "failed_or_missing",
                    "c_fixed_status": cf_status,
                    "schedule_only_status": e2_status,
                    "intentir_status": e3_status,
                    "schedule_only_error": (
                        left.get("error") if left is not None else None
                    ),
                    "intentir_error": (
                        right.get("error") if right is not None else None
                    ),
                    "c_fixed_error": (
                        fixed.get("error") if fixed is not None else None
                    ),
                }
            )
            continue
        assert left is not None and right is not None
        comparable = (left, right) if fixed is None else (fixed, left, right)
        for key in (
            "case_id",
            "extents",
            "workload_valuation",
            "workload_profile",
            "baseline_policy",
        ):
            if any(item.get(key) != left.get(key) for item in comparable):
                raise ValueError(f"{kernel} CF/E2/E3 differ in {key}")
        schedule_only_us = float(left["performance"]["mlir_us"])
        intentir_us = float(right["performance"]["mlir_us"])
        c_fixed_us = (
            float(fixed["performance"]["mlir_us"])
            if fixed is not None
            else None
        )
        c_fixed_schedule_ablation = left["performance"].get(
            "c_fixed_schedule_ablation"
        )
        same_run_schedule_gain = None
        if isinstance(c_fixed_schedule_ablation, Mapping):
            if not c_fixed_schedule_ablation.get("same_run_measurement"):
                raise ValueError(
                    f"{kernel} C-Fixed/Schedule-Only comparison is not same-run"
                )
            same_run_schedule_gain = float(
                c_fixed_schedule_ablation["schedule_over_c_fixed_speedup"]
            )
            if fixed is not None:
                fixed_winner_id = fixed["winner"]["candidate"]["candidate_id"]
                if fixed_winner_id != c_fixed_schedule_ablation.get(
                    "c_fixed_candidate_id"
                ):
                    raise ValueError(
                        f"{kernel} standalone and same-run C-Fixed selections differ"
                    )
        schedule_source_value = left["performance"].get("source_triton_us")
        intentir_source_value = right["performance"].get("source_triton_us")
        if (schedule_source_value is None) != (intentir_source_value is None):
            raise ValueError(f"{kernel} E2/E3 source baseline availability differs")
        source_us = (
            float(intentir_source_value)
            if intentir_source_value is not None
            else None
        )
        structural = bool(_structural_paths(right) - _structural_paths(left))
        if structural:
            structurally_expanded.append(kernel)
        c_ordering = right["performance"]["c_ordering_ablation"]
        c_gain = c_ordering.get(
            "c_prior_over_erase_c_speedup"
        )
        if c_gain is not None:
            c_speedups.append(float(c_gain))
        c_to_95 = c_ordering.get("with_c_candidates_to_95pct_of_final_best")
        erase_to_95 = c_ordering.get(
            "without_c_candidates_to_95pct_of_final_best"
        )
        if c_to_95 is not None and erase_to_95 is not None:
            c_candidates_to_95pct.append(int(c_to_95))
            erase_c_candidates_to_95pct.append(int(erase_to_95))
        cross_report_gain = schedule_only_us / intentir_us
        spine_ablation = right["performance"].get("schedule_spine_ablation")
        same_run_gain = None
        if isinstance(spine_ablation, Mapping):
            if not spine_ablation.get("same_run_measurement"):
                raise ValueError(
                    f"{kernel} schedule-spine comparison is not same-run"
                )
            same_run_gain = float(
                spine_ablation["expanded_over_schedule_spine_speedup"]
            )
            if same_run_gain <= 0:
                raise ValueError(f"{kernel} has a non-positive spine speedup")
        primary_gain = (
            same_run_gain
            if same_run_gain is not None
            else cross_report_gain
        )
        record = {
            "kernel": kernel,
            "case_id": left.get("case_id"),
            "extents": left.get("extents"),
            "workload_valuation": left.get("workload_valuation"),
            "workload_profile": left.get(
                "workload_profile", e2.get("workload_profile")
            ),
            "baseline_policy": left.get(
                "baseline_policy", e2.get("baseline_policy")
            ),
            "status": "paired_success",
            "c_fixed_status": cf_status,
            "schedule_only_status": e2_status,
            "intentir_status": e3_status,
            "c_fixed_us": c_fixed_us,
            "schedule_only_us": schedule_only_us,
            "intentir_us": intentir_us,
            "g_schedule": (
                same_run_schedule_gain
                if same_run_schedule_gain is not None
                else c_fixed_us / schedule_only_us
                if c_fixed_us is not None
                else None
            ),
            "g_lowering": primary_gain,
            "external_competitiveness": {
                "source": {
                    "status": "measured" if source_us is not None else "unavailable",
                    "latency_us": source_us,
                },
                "intentir_latency_us": intentir_us,
                "intentir_speedup_over_source": (
                    source_us / intentir_us if source_us is not None else None
                ),
            },
            "c_prior_efficiency": {
                "best_at_budget_c_over_erase_c_speedup": c_gain,
                "c_prior_candidates_to_95pct_of_final_best": c_to_95,
                "erase_c_candidates_to_95pct_of_final_best": erase_to_95,
                "unavailable_reason": c_ordering.get("unavailable_reason"),
            },
            "supplementary_diagnostics": {
                "formal_role": "not_a_main_experiment_arm",
                "same_run_g_lowering": same_run_gain,
                "cross_report_g_lowering": cross_report_gain,
                "same_run_g_schedule": same_run_schedule_gain,
                "cross_report_g_schedule": (
                    c_fixed_us / schedule_only_us
                    if c_fixed_us is not None
                    else None
                ),
                "schedule_spine_best_us_in_intentir_run": (
                    spine_ablation.get("schedule_spine_best_us")
                    if isinstance(spine_ablation, Mapping)
                    else None
                ),
                "expanded_best_us_in_intentir_run": (
                    spine_ablation.get("expanded_best_us")
                    if isinstance(spine_ablation, Mapping)
                    else None
                ),
                "structural_state_expansion_observed": structural,
                "schedule_only_structural_path_count": len(_structural_paths(left)),
                "intentir_structural_path_count": len(_structural_paths(right)),
            },
        }
        per_kernel.append(record)
        paired_success_count += 1
        lowering_gain.append(record["g_lowering"])
        if record["g_schedule"] is not None:
            schedule_gain.append(record["g_schedule"])
        cross_report_lowering_gain.append(cross_report_gain)
        source_speedup = record["external_competitiveness"][
            "intentir_speedup_over_source"
        ]
        if source_speedup is not None:
            intentir_over_source.append(source_speedup)
            schedule_only_over_source.append(source_us / schedule_only_us)
    return {
        "protocol": SUMMARY_PROTOCOL,
        "target": e2["target"],
        "workload_profile": e2.get("workload_profile"),
        "baseline_policy": e2.get("baseline_policy"),
        "device": _device_config(e2),
        "measurement": measurement,
        "measurement_spec_protocol": e2.get("measurement_spec_protocol"),
        "summary": {
            "formal_three_arm": c_fixed is not None,
            "kernel_count": len(kernel_universe),
            "paired_success_count": paired_success_count,
            "failed_or_missing_count": len(kernel_universe) - paired_success_count,
            "source_baseline_kernel_count": len(intentir_over_source),
            "geomean_g_schedule": _geomean(schedule_gain),
            "geomean_g_lowering": _geomean(lowering_gain),
            "geomean_intentir_speedup_over_source": _geomean(
                intentir_over_source
            ),
        },
        "c_prior_efficiency_summary": {
            "geomean_best_at_budget_c_over_erase_c_speedup": _geomean(
                c_speedups
            ),
            "oracle_complete_kernel_count": len(c_candidates_to_95pct),
            "median_c_prior_candidates_to_95pct_of_final_best": (
                statistics.median(c_candidates_to_95pct)
                if c_candidates_to_95pct
                else None
            ),
            "median_erase_c_candidates_to_95pct_of_final_best": (
                statistics.median(erase_c_candidates_to_95pct)
                if erase_c_candidates_to_95pct
                else None
            ),
        },
        "supplementary_diagnostics": {
            "formal_role": "not_a_main_experiment_arm",
            "geomean_cross_report_g_lowering": _geomean(
                cross_report_lowering_gain
            ),
            "geomean_schedule_only_speedup_over_source": _geomean(
                schedule_only_over_source
            ),
            "structural_state_expansion_count": len(structurally_expanded),
            "structural_state_expansion_kernels": structurally_expanded,
        },
        "per_kernel": per_kernel,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument(
        "--c-fixed",
        type=Path,
        required=True,
        help="machine-executed C-Fixed report required for formal three-arm output",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = summarize(
        _load(args.e2),
        _load(args.e3),
        _load(args.c_fixed),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
