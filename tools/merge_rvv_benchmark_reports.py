#!/usr/bin/env python3
"""Merge disjoint/incremental RVV target reports and recompute summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v6"


def _geomean(values: list[float]) -> float | None:
    return (
        math.exp(sum(math.log(value) for value in values) / len(values))
        if values
        else None
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payloads = []
    for path in args.reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != REPORT_PROTOCOL:
            raise ValueError(f"unsupported RVV report protocol in {path}")
        payloads.append(payload)
    first = payloads[0]
    measurement = first["measurement"]
    device_config = {
        key: first["device"].get(key)
        for key in ("machine", "vlen_bits", "cache")
    }
    report_config_keys = (
        "bundle_protocol",
        "target",
        "workload_profile",
        "baseline_policy",
        "experiment_spec_protocol",
        "measurement_spec_protocol",
    )
    records: dict[str, dict[str, Any]] = {}
    kernel_configs: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for key in report_config_keys:
            if payload.get(key) != first.get(key):
                raise ValueError(f"RVV reports differ in {key}")
        if payload["measurement"] != measurement:
            raise ValueError("RVV reports use different measurement protocols")
        current_device = {
            key: payload["device"].get(key)
            for key in ("machine", "vlen_bits", "cache")
        }
        if current_device != device_config:
            raise ValueError("RVV reports were produced on incompatible devices")
        for record in payload["kernels"]:
            name = record["kernel"]
            config = {
                key: record.get(key)
                for key in (
                    "case_id",
                    "extents",
                    "workload_valuation",
                )
            }
            if name in kernel_configs and kernel_configs[name] != config:
                raise ValueError(f"duplicate RVV kernel {name} has different configuration")
            kernel_configs[name] = config
            previous = records.get(name)
            if previous is not None and "performance" in previous:
                if "performance" in record:
                    raise ValueError(f"duplicate successful RVV kernel {name}")
                continue
            records[name] = record
        for failure in payload.get("failures", []):
            records.setdefault(failure["kernel"], failure)
    kernels = [records[name] for name in sorted(records)]
    successful = [item for item in kernels if "performance" in item]
    failures = [item for item in kernels if "performance" not in item]
    schedule = [
        item["performance"]["g_schedule"] for item in successful
    ]
    full = [
        item["performance"]["g_full"] for item in successful
    ]
    c_prior = [
        item["c_prior_efficiency"]["best_at_budget_c_over_erase_c_speedup"]
        for item in successful
    ]
    c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "c_prior_candidates_to_95pct_of_final_best"
        ]
        for item in successful
        if item["c_prior_efficiency"].get(
            "c_prior_candidates_to_95pct_of_final_best"
        )
        is not None
    ]
    erase_c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "erase_c_candidates_to_95pct_of_final_best"
        ]
        for item in successful
        if item["c_prior_efficiency"].get(
            "erase_c_candidates_to_95pct_of_final_best"
        )
        is not None
    ]
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": first["bundle_protocol"],
        "target": first["target"],
        "workload_profile": first.get("workload_profile"),
        "baseline_policy": first.get("baseline_policy"),
        "experiment_spec_protocol": first.get("experiment_spec_protocol"),
        "measurement_spec_protocol": first.get("measurement_spec_protocol"),
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": sum(
            float(item.get("elapsed_seconds", 0.0)) for item in payloads
        ),
        "device": payloads[-1]["device"],
        "measurement": measurement,
        "summary": {
            "requested": len(kernels),
            "successful": len(successful),
            "failed": len(failures),
            "geomean_g_schedule": _geomean(schedule),
            "geomean_g_full": _geomean(full),
        },
        "c_prior_efficiency_summary": {
            "geomean_best_at_budget_c_over_erase_c_speedup": _geomean(c_prior),
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
        "kernels": kernels,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
