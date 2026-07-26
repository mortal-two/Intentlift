#!/usr/bin/env python3
"""Summarize L0 versus L1 B-conditioned local MLIR transformations.

Both arms retain the exact accepted J=A+B and ContractView.  L0 disables only
registered candidates whose pass metadata declares a B-conditioned
local transformation; L1 adds those candidates.  Candidate timings are read from one
shared full-search receipt, so the comparison cannot be explained by timing
noise between separately ordered searches.  The v1 JSON field names retain
their historical e2/e3 spelling for checked-receipt compatibility; they do not
denote the structural E2/E3 experiment defined by the current backend spec.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.reporting import load_disjoint_kernel_reports


PROTOCOL = "intentir.production.b-conditioned-search-ablation.v1"


def _gmean(values: list[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _candidate_measurements(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "search" in item:
        records = item["search"]["attempts"]
    else:
        records = item["candidates"]
    measured = []
    for record in records:
        latency = record.get("latency_us")
        candidate = record.get("candidate")
        if (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and latency > 0
            and isinstance(candidate, Mapping)
        ):
            if "b_transformations" not in candidate:
                raise ValueError(
                    f"{item['kernel']} receipt predates B-search metadata"
                )
            measured.append(
                {
                    "latency_us": float(latency),
                    "candidate": dict(candidate),
                }
            )
    if not measured:
        raise ValueError(f"{item['kernel']} has no measured candidates")
    return measured


def _one_target(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    kernels = []
    transform_winners: Counter[str] = Counter()
    for item in report["kernels"]:
        measured = _candidate_measurements(item)
        schedule_only = [
            record
            for record in measured
            if not record["candidate"]["b_transformations"]
        ]
        if not schedule_only:
            raise ValueError(
                f"{item['kernel']} has no L0 local-transform-disabled candidate"
            )
        e2 = min(schedule_only, key=lambda record: record["latency_us"])
        e3 = min(measured, key=lambda record: record["latency_us"])
        speedup = e2["latency_us"] / e3["latency_us"]
        for transform in e3["candidate"]["b_transformations"]:
            transform_winners[str(transform)] += 1
        contract = (
            item["search"]["contract_view"]
            if "search" in item
            else item["contract_view"]
        )
        kernels.append(
            {
                "kernel": item["kernel"],
                "contract_view_digest": contract["digest"],
                "b_capabilities": contract.get("capabilities", []),
                "measured_candidate_count": len(measured),
                "e2_schedule_only_candidate_count": len(schedule_only),
                "e3_b_transformation_candidate_count": (
                    len(measured) - len(schedule_only)
                ),
                "e2_schedule_only": {
                    "latency_us": e2["latency_us"],
                    "candidate": e2["candidate"],
                },
                "e3_contract_guided": {
                    "latency_us": e3["latency_us"],
                    "candidate": e3["candidate"],
                },
                "e3_over_e2_speedup": speedup,
                "b_transformation_wins": bool(
                    e3["candidate"]["b_transformations"]
                ),
                "speedup_over_1pct": speedup > 1.01,
            }
        )
    speedups = [item["e3_over_e2_speedup"] for item in kernels]
    affected = [
        item["e3_over_e2_speedup"]
        for item in kernels
        if item["speedup_over_1pct"]
    ]
    return {
        "label": label,
        "device": report["device"],
        "source_protocol": report["protocol"],
        "source_report_count": report.get("source_report_count", 1),
        "invariants": {
            "j_a_plus_b_fixed": True,
            "contract_view_fixed": True,
            "c_fixed": True,
            "workload_fixed": True,
            "shared_candidate_latency_map": True,
            "e2_is_nested_in_e3": True,
            "legacy_e2_e3_fields_mean_l0_l1": True,
            "performance_does_not_decide_legality": True,
        },
        "summary": {
            "kernel_count": len(kernels),
            "winner_uses_b_transformation_count": sum(
                item["b_transformation_wins"] for item in kernels
            ),
            "speedup_over_1pct_kernel_count": len(affected),
            "geomean_e3_over_e2_speedup_all": _gmean(speedups),
            "geomean_e3_over_e2_speedup_affected": (
                _gmean(affected) if affected else 1.0
            ),
            "maximum_e3_over_e2_speedup": max(speedups),
            "winning_transformations": dict(sorted(transform_winners.items())),
        },
        "kernels": kernels,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rtx-report", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--h100-report", type=Path, action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rtx_reports, rtx_kernels = load_disjoint_kernel_reports(args.rtx_report)
    h100_reports, h100_kernels = load_disjoint_kernel_reports(args.h100_report)
    rtx = {
        "protocol": rtx_reports[0]["protocol"],
        "device": rtx_reports[0]["device"],
        "source_report_count": len(rtx_reports),
        "kernels": [rtx_kernels[name] for name in sorted(rtx_kernels)],
    }
    h100 = {
        "protocol": h100_reports[0]["protocol"],
        "device": h100_reports[0]["device"],
        "source_report_count": len(h100_reports),
        "kernels": [h100_kernels[name] for name in sorted(h100_kernels)],
    }
    payload = {
        "protocol": PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "experiment": {
            "l0": "fixed structural reach; B-conditioned local transforms disabled",
            "l1": "same reach plus registered B-conditioned local transforms",
            "legacy_wire_fields": "e2/e3 keys encode L0/L1 in protocol v1",
            "selection": "minimum latency among already-admitted candidates",
        },
        "targets": [
            _one_target(rtx, "rtx5090"),
            _one_target(h100, "h100"),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
