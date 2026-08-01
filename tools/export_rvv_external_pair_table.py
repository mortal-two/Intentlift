#!/usr/bin/env python3
"""Merge formal RVV external-pair reports into plot-ready JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PAIR_PROTOCOL = "intentir.production.rvv-external-pair.v1"
TABLE_PROTOCOL = "intentir.production.rvv-external-pair-table.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def build_table(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    bars = []
    seen = set()
    external_names = set()
    for report in reports:
        if report.get("protocol") != PAIR_PROTOCOL:
            raise ValueError("unsupported RVV external-pair report")
        kernel = report.get("kernel")
        if not isinstance(kernel, str) or not kernel or kernel in seen:
            raise ValueError(f"invalid or duplicate pair kernel: {kernel!r}")
        seen.add(kernel)
        full = report["full"]
        external = report["external"]
        external_names.add(str(external["display_name"]))
        full_us = float(full["latency_us"])
        external_us = float(external["latency_us"])
        if not math.isfinite(full_us) or not math.isfinite(external_us):
            raise ValueError(f"non-finite latency for {kernel}")
        speedup = external_us / full_us
        workload = report["workload"]
        row = {
            "kernel": kernel,
            "extents": workload["extents"],
            "inputs": workload["inputs"],
            "outputs": workload["outputs"],
            "measurement": report["measurement"],
            "full": full,
            "external": external,
            "external_over_full": speedup,
        }
        rows.append(row)
        for bar_id, display_name, latency in (
            ("full", "IntentIR Full", full_us),
            (external["baseline_id"], external["display_name"], external_us),
        ):
            bars.append(
                {
                    "kernel": kernel,
                    "extents": workload["extents"],
                    "bar_id": bar_id,
                    "display_name": display_name,
                    "latency_us": latency,
                    "full_speedup_vs_bar": latency / full_us,
                }
            )
    geomean = math.exp(
        sum(math.log(row["external_over_full"]) for row in rows) / len(rows)
    )
    if len(external_names) == 1:
        comparison = f"{next(iter(external_names))} versus IntentIR Full"
    else:
        comparison = "External RVV baselines versus IntentIR Full"
    return {
        "protocol": TABLE_PROTOCOL,
        "kernel_count": len(rows),
        "comparison": comparison,
        "geomean_external_over_full": geomean,
        "rows": rows,
        "bars": bars,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    table = build_table(reports)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "kernel",
                "extents",
                "bar_id",
                "display_name",
                "latency_us",
                "full_speedup_vs_bar",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for bar in table["bars"]:
            writer.writerow({**bar, "extents": json.dumps(bar["extents"], sort_keys=True)})
    print(
        f"wrote {table['kernel_count']} RVV external pairs to "
        f"{args.out_json} and {args.out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
