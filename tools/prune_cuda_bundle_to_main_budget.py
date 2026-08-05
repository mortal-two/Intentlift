#!/usr/bin/env python3
"""Extract the equal-budget main experiment from a full CUDA C-oracle bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v3"


def prune(bundle: dict[str, Any]) -> dict[str, Any]:
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported CUDA benchmark bundle")
    result = dict(bundle)
    kernels = []
    for raw_entry in bundle["kernels"]:
        entry = dict(raw_entry)
        orders = dict(entry["search_orders"])
        selected_ids = set(orders["with_c_concrete_order"])
        selected_ids.update(orders["without_c_concrete_order"])
        selected_ids.update(orders["active_schedule_spine_concrete_order"])
        projection = orders.get("c_fixed_projection", {})
        if projection.get("selected_candidate_id") is not None:
            selected_ids.add(str(projection["selected_candidate_id"]))
        entry["candidates"] = [
            candidate
            for candidate in entry["candidates"]
            if candidate["candidate"]["candidate_id"] in selected_ids
        ]
        available_ids = {
            candidate["candidate"]["candidate_id"]
            for candidate in entry["candidates"]
        }
        if available_ids != selected_ids:
            missing = sorted(selected_ids - available_ids)
            raise ValueError(
                f"{entry['kernel']} main-budget candidates are missing: {missing}"
            )
        orders["c_oracle_transport_complete"] = False
        orders["c_oracle_attempt_trace"] = []
        orders["with_c_full_concrete_order"] = []
        orders["without_c_full_concrete_order"] = []
        orders["transported_union_count"] = len(entry["candidates"])
        entry["search_orders"] = orders
        kernels.append(entry)
    result["kernels"] = kernels
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    result = prune(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(
        f"wrote {len(result['kernels'])} main-budget CUDA workloads to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
