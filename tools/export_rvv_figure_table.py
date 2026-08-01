#!/usr/bin/env python3
"""Turn RVV benchmark reports into plot-ready long-form data.

The exporter preserves N/A and failed external implementations as coverage
records, while emitting bars only for measured implementations.  It performs no
benchmark selection: every kernel must occur in exactly one input report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "intentir.backend.riscv.figure-table.v1"


def _finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not result > 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def build_figure_table(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    seen: set[str] = set()

    for report_name, payload in reports:
        kernels = payload.get("kernels")
        if not isinstance(kernels, list):
            raise ValueError(f"report {report_name!r} lacks a kernel list")
        for record in kernels:
            if not isinstance(record, Mapping):
                raise ValueError(f"report {report_name!r} has an invalid kernel")
            kernel = record.get("kernel")
            if not isinstance(kernel, str) or not kernel:
                raise ValueError(f"report {report_name!r} has an unnamed kernel")
            if kernel in seen:
                raise ValueError(f"kernel {kernel!r} occurs in multiple reports")
            seen.add(kernel)
            performance = record.get("performance")
            if not isinstance(performance, Mapping):
                coverage.append(
                    {
                        "kernel": kernel,
                        "bar_id": "IntentIR",
                        "display_name": "IntentIR Full",
                        "group": "internal",
                        "status": "failed",
                        "reason": record.get("error", "kernel measurement failed"),
                    }
                )
                continue

            full_us = _finite_positive(performance.get("full_us"), field="full_us")
            internal = (
                ("c_fixed", "C-Fixed", performance.get("c_fixed_us")),
                (
                    "schedule_only",
                    "Schedule-Only",
                    performance.get("schedule_only_us"),
                ),
                ("full", "Full", full_us),
            )
            row: dict[str, Any] = {
                "kernel": kernel,
                "report": report_name,
                "c_fixed_us": _finite_positive(
                    performance.get("c_fixed_us"), field="c_fixed_us"
                ),
                "schedule_only_us": _finite_positive(
                    performance.get("schedule_only_us"),
                    field="schedule_only_us",
                ),
                "full_us": full_us,
                "g_schedule": performance.get("g_schedule"),
                "g_full": performance.get("g_full"),
                "external": {},
            }
            for bar_id, display_name, raw_latency in internal:
                latency = _finite_positive(raw_latency, field=f"{bar_id}_us")
                bars.append(
                    {
                        "kernel": kernel,
                        "bar_id": bar_id,
                        "display_name": display_name,
                        "group": "internal",
                        "status": "measured",
                        "latency_us": latency,
                        "full_speedup_vs_bar": latency / full_us,
                    }
                )
                coverage.append(
                    {
                        "kernel": kernel,
                        "bar_id": bar_id,
                        "display_name": display_name,
                        "group": "internal",
                        "status": "measured",
                    }
                )

            external = record.get("external_baselines", [])
            if not isinstance(external, list):
                raise ValueError(f"kernel {kernel!r} has invalid external baselines")
            for item in external:
                if not isinstance(item, Mapping):
                    raise ValueError(f"kernel {kernel!r} has an invalid baseline")
                baseline_id = item.get("baseline_id")
                display_name = item.get("display_name")
                status = item.get("status")
                if not isinstance(baseline_id, str) or not baseline_id:
                    raise ValueError(f"kernel {kernel!r} has an unnamed baseline")
                if not isinstance(display_name, str) or not display_name:
                    display_name = baseline_id
                if not isinstance(status, str) or not status:
                    raise ValueError(
                        f"kernel {kernel!r} baseline {baseline_id!r} lacks status"
                    )
                coverage_record = {
                    "kernel": kernel,
                    "bar_id": baseline_id,
                    "display_name": display_name,
                    "group": "external",
                    "status": status,
                }
                if status != "measured":
                    if item.get("error") is not None:
                        coverage_record["reason"] = item["error"]
                    coverage.append(coverage_record)
                    row["external"][baseline_id] = {"status": status}
                    continue
                latency = _finite_positive(
                    item.get("latency_us"), field=f"{baseline_id}.latency_us"
                )
                bar = {
                    **coverage_record,
                    "latency_us": latency,
                    "full_speedup_vs_bar": latency / full_us,
                    "paired_full_us": item.get("full_paired_us"),
                    "paired_intentir_speedup": item.get("intentir_speedup"),
                }
                bars.append(bar)
                coverage.append(coverage_record)
                row["external"][baseline_id] = {
                    "status": status,
                    "display_name": display_name,
                    "latency_us": latency,
                    "full_speedup_vs_bar": latency / full_us,
                    "paired_full_us": item.get("full_paired_us"),
                    "paired_intentir_speedup": item.get("intentir_speedup"),
                }

            best_native = record.get("best_native_baseline")
            aggregate_coverage = {
                "kernel": kernel,
                "bar_id": "best_native_rvv",
                "display_name": "Best Native RVV",
                "group": "external_aggregate",
            }
            if isinstance(best_native, Mapping) and best_native.get("status") == "measured":
                latency = _finite_positive(
                    best_native.get("latency_us"), field="best_native_rvv.latency_us"
                )
                selected_id = best_native.get("baseline_id")
                selected_name = best_native.get("display_name")
                aggregate_bar = {
                    **aggregate_coverage,
                    "status": "measured",
                    "latency_us": latency,
                    "full_speedup_vs_bar": latency / full_us,
                    "paired_full_us": best_native.get("full_paired_us"),
                    "paired_intentir_speedup": best_native.get("intentir_speedup"),
                    "selected_baseline_id": selected_id,
                    "selected_baseline_name": selected_name,
                }
                bars.append(aggregate_bar)
                coverage.append({**aggregate_coverage, "status": "measured"})
                row["external"]["best_native_rvv"] = {
                    key: value
                    for key, value in aggregate_bar.items()
                    if key not in {"kernel", "bar_id", "group"}
                }
            else:
                coverage.append(
                    {
                        **aggregate_coverage,
                        "status": "not_applicable",
                        "reason": "no strict Expert RVV or mature-library implementation",
                    }
                )
                row["external"]["best_native_rvv"] = {
                    "status": "not_applicable"
                }
            rows.append(row)

    counts: dict[str, dict[str, int]] = {}
    for item in coverage:
        bar_counts = counts.setdefault(item["bar_id"], {})
        status = item["status"]
        bar_counts[status] = bar_counts.get(status, 0) + 1
    return {
        "protocol": PROTOCOL,
        "kernel_count": len(seen),
        "coverage_counts": counts,
        "rows": sorted(rows, key=lambda item: item["kernel"]),
        "bars": sorted(bars, key=lambda item: (item["kernel"], item["bar_id"])),
        "coverage": sorted(
            coverage, key=lambda item: (item["kernel"], item["bar_id"])
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export plot-ready RVV internal and external baseline data."
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = [
        (path.name, json.loads(path.read_text(encoding="utf-8")))
        for path in args.reports
    ]
    table = build_figure_table(reports)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "bar_id",
        "display_name",
        "group",
        "status",
        "latency_us",
        "full_speedup_vs_bar",
        "paired_full_us",
        "paired_intentir_speedup",
        "selected_baseline_id",
        "selected_baseline_name",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for bar in table["bars"]:
            writer.writerow({field: bar.get(field) for field in fieldnames})
    print(
        f"wrote {len(table['bars'])} measured bars for "
        f"{table['kernel_count']} kernels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
