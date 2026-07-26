#!/usr/bin/env python3
"""Summarize paired with-C/without-C measured CUDA search reports.

The paired experiment keeps J=A+B, ContractView, target, compiler, workload,
candidate universe, and timing protocol fixed.  It reports both independently
measured arms and an order-only counterfactual that reuses one latency map while
changing only the candidate order, isolating C's finite-budget effect from
measurement noise.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.cuda.schedules import REGISTERED_DEFAULT_SCHEDULES
from intentir.backend.reporting import load_disjoint_kernel_reports


REPORT_PROTOCOL = "intentir.production.cuda-search-ablation-summary.v2"


def _gmean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(item) for item in values) / len(values))


def _successful(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        item["kernel"]: item
        for item in report["kernels"]
        if "performance" in item
    }


def _attempts(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(item["search"]["attempts"])


def _latencies(item: Mapping[str, Any]) -> dict[str, float]:
    return {
        attempt["candidate"]["candidate_id"]: float(attempt["latency_us"])
        for attempt in _attempts(item)
        if attempt.get("latency_us") is not None
    }


def _order(item: Mapping[str, Any]) -> list[str]:
    return [attempt["candidate"]["candidate_id"] for attempt in _attempts(item)]


def _best_at(
    order: list[str], latencies: Mapping[str, float], budget: int
) -> float | None:
    samples = [latencies[item] for item in order[:budget] if item in latencies]
    return min(samples) if samples else None


def _budget_to_target(
    order: list[str],
    latencies: Mapping[str, float],
    target: float,
) -> int | None:
    best: float | None = None
    for index, candidate_id in enumerate(order, start=1):
        if candidate_id in latencies:
            value = latencies[candidate_id]
            best = value if best is None else min(best, value)
        if best is not None and best <= target:
            return index
    return None


def _default_latency(item: Mapping[str, Any]) -> tuple[str, float]:
    measured = [
        attempt
        for attempt in _attempts(item)
        if attempt.get("latency_us") is not None
    ]
    path_kinds = {attempt["candidate"]["family"] for attempt in measured}
    if len(path_kinds) != 1:
        raise ValueError(
            f"{item['kernel']} has {len(path_kinds)} measured lowering path kinds"
        )
    path_kind = next(iter(path_kinds))
    expected = REGISTERED_DEFAULT_SCHEDULES[path_kind]
    matches = [
        attempt
        for attempt in measured
        if all(
            attempt["candidate"]["schedule"].get(
                name,
                False if value is False else None,
            )
            == value
            for name, value in expected.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{item['kernel']} has {len(matches)} registered-default candidates"
        )
    return path_kind, float(matches[0]["latency_us"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-c", type=Path, action="append", required=True)
    parser.add_argument(
        "--without-c", type=Path, action="append", required=True
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 2, 3, 6, 12, 24, 32],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(item <= 0 for item in args.budgets):
        raise ValueError("all budgets must be positive")
    with_reports, with_items = load_disjoint_kernel_reports(args.with_c)
    without_reports, without_items = load_disjoint_kernel_reports(
        args.without_c
    )
    with_report = {
        "protocol": with_reports[0]["protocol"],
        "device": with_reports[0]["device"],
        "source_report_count": len(with_reports),
        "kernels": [with_items[name] for name in sorted(with_items)],
    }
    without_report = {
        "protocol": without_reports[0]["protocol"],
        "device": without_reports[0]["device"],
        "source_report_count": len(without_reports),
        "kernels": [
            without_items[name] for name in sorted(without_items)
        ],
    }
    with_kernels = _successful(with_report)
    without_kernels = _successful(without_report)
    if set(with_kernels) != set(without_kernels):
        raise ValueError("paired reports have different successful kernel sets")
    if with_report["device"]["target_chip"] != without_report["device"]["target_chip"]:
        raise ValueError("paired reports target different CUDA chips")

    records: list[dict[str, Any]] = []
    for name in sorted(with_kernels):
        seeded = with_kernels[name]
        unseeded = without_kernels[name]
        for field in ("snapshot_digest", "j_digest"):
            left = seeded["snapshot"].get(field)
            right = unseeded["snapshot"].get(field)
            if left != right:
                raise ValueError(f"{name} paired {field} differs")
        left_contract = seeded["search"]["contract_view"]
        right_contract = unseeded["search"]["contract_view"]
        if left_contract["digest"] != right_contract["digest"]:
            raise ValueError(f"{name} ContractView changed after Erase_C")
        seeded_order = _order(seeded)
        unseeded_order = _order(unseeded)
        if set(seeded_order) != set(unseeded_order):
            raise ValueError(f"{name} candidate universe changed after Erase_C")
        seeded_latencies = _latencies(seeded)
        unseeded_latencies = _latencies(unseeded)
        if set(seeded_latencies) != set(unseeded_latencies):
            raise ValueError(f"{name} measured candidate set differs between arms")
        path_kind, default_us = _default_latency(seeded)
        full_best_us = min(seeded_latencies.values())
        actual_unseeded_best_us = min(unseeded_latencies.values())
        within_one_percent = full_best_us * 1.01
        records.append(
            {
                "kernel": name,
                "lowering_path_kind": path_kind,
                "contract_view_digest": left_contract["digest"],
                "b_consumed_fact_count": len(left_contract["consumed"]),
                "candidate_universe_size": len(seeded_order),
                "measured_candidate_count": len(seeded_latencies),
                "registered_default_us": default_us,
                "searched_best_us": full_best_us,
                "search_speedup_over_default": default_us / full_best_us,
                "actual_without_c_best_us": actual_unseeded_best_us,
                "full_budget_best_ratio_with_over_without_c": (
                    full_best_us / actual_unseeded_best_us
                ),
                "attempt_budget_to_within_1pct": {
                    "with_c": _budget_to_target(
                        seeded_order, seeded_latencies, within_one_percent
                    ),
                    "without_c_order_only": _budget_to_target(
                        unseeded_order, seeded_latencies, within_one_percent
                    ),
                    "without_c_actual": _budget_to_target(
                        unseeded_order,
                        unseeded_latencies,
                        actual_unseeded_best_us * 1.01,
                    ),
                },
                "budget_curve": [
                    {
                        "attempt_budget": budget,
                        "with_c_actual_us": _best_at(
                            seeded_order, seeded_latencies, budget
                        ),
                        "without_c_actual_us": _best_at(
                            unseeded_order, unseeded_latencies, budget
                        ),
                        "without_c_order_only_us": _best_at(
                            unseeded_order, seeded_latencies, budget
                        ),
                    }
                    for budget in args.budgets
                ],
            }
        )

    search_speedups = [item["search_speedup_over_default"] for item in records]
    budget_summary = []
    for budget in args.budgets:
        seeded_values = []
        order_only_values = []
        actual_unseeded_values = []
        for item in records:
            point = next(
                value
                for value in item["budget_curve"]
                if value["attempt_budget"] == budget
            )
            if point["with_c_actual_us"] is not None:
                seeded_values.append(point["with_c_actual_us"])
            if point["without_c_order_only_us"] is not None:
                order_only_values.append(point["without_c_order_only_us"])
            if point["without_c_actual_us"] is not None:
                actual_unseeded_values.append(point["without_c_actual_us"])
        budget_summary.append(
            {
                "attempt_budget": budget,
                "with_c_coverage": len(seeded_values),
                "without_c_coverage": len(actual_unseeded_values),
                "with_c_geomean_best_us": _gmean(seeded_values),
                "without_c_actual_geomean_best_us": _gmean(
                    actual_unseeded_values
                ),
                "without_c_order_only_geomean_best_us": _gmean(
                    order_only_values
                ),
            }
        )
    output = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "target": with_report["device"],
        "invariants": {
            "j_a_plus_b_fixed": True,
            "contract_view_fixed": True,
            "candidate_universe_fixed": True,
            "c_changes_legality": False,
            "order_only_counterfactual_reuses_with_c_latency_map": True,
            "with_c_source_report_count": len(with_reports),
            "without_c_source_report_count": len(without_reports),
        },
        "summary": {
            "kernel_count": len(records),
            "search_geomean_speedup_over_registered_default": _gmean(
                search_speedups
            ),
            "search_improved_kernel_count": sum(
                item > 1.0 for item in search_speedups
            ),
            "budget_curves": budget_summary,
        },
        "kernels": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Production CUDA search ablation",
        "",
        f"Target: `{with_report['device']['target_chip']}` / "
        f"{with_report['device']['device_name']}",
        "",
        "J=A+B、ContractView、target、compiler、workload 与候选全集固定；"
        "唯一干预是 Attach_C/Erase_C。",
        "",
        "| Kernel | Lowering path kind | Default us | Search best us | Speedup | "
        "B facts | budget to 1% (C / no-C order) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        budgets = item["attempt_budget_to_within_1pct"]
        lines.append(
            f"| {item['kernel']} | {item['lowering_path_kind']} | "
            f"{item['registered_default_us']:.4f} | "
            f"{item['searched_best_us']:.4f} | "
            f"{item['search_speedup_over_default']:.3f}x | "
            f"{item['b_consumed_fact_count']} | "
            f"{budgets['with_c']} / {budgets['without_c_order_only']} |"
        )
    lines.extend(
        [
            "",
            "Geometric-mean search speedup over the registered default: "
            f"**{output['summary']['search_geomean_speedup_over_registered_default']:.3f}x**.",
            "",
            "The order-only arm uses the identical measured candidate latency map; "
            "therefore its difference is attributable only to C ordering.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
