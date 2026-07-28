#!/usr/bin/env python3
"""Merge disjoint/incremental RVV target reports and recompute summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v4"


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
    provenance = []
    for path in args.reports:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if payload.get("protocol") != REPORT_PROTOCOL:
            raise ValueError(f"unsupported RVV report protocol in {path}")
        payloads.append(payload)
        provenance.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    first = payloads[0]
    measurement = first["measurement"]
    device_identity = {
        key: first["device"].get(key)
        for key in ("machine", "vlen_bits", "cache")
    }
    report_identity_keys = (
        "bundle_protocol",
        "bundle_toolchain_digest",
        "target",
        "workload_profile",
        "baseline_policy",
        "experiment_spec_protocol",
        "experiment_spec_digest",
        "measurement_spec_protocol",
        "measurement_spec_digest",
    )
    records: dict[str, dict[str, Any]] = {}
    kernel_identities: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for key in report_identity_keys:
            if payload.get(key) != first.get(key):
                raise ValueError(f"RVV reports differ in {key}")
        if payload["measurement"] != measurement:
            raise ValueError("RVV reports use different measurement protocols")
        current_device = {
            key: payload["device"].get(key)
            for key in ("machine", "vlen_bits", "cache")
        }
        if current_device != device_identity:
            raise ValueError("RVV reports were produced on incompatible devices")
        for record in payload["kernels"]:
            name = record["kernel"]
            identity = {
                key: record.get(key)
                for key in (
                    "case_id",
                    "snapshot_digest",
                    "j_digest",
                    "c_digest",
                    "extents",
                    "workload_valuation",
                    "j_provenance",
                )
            }
            if name in kernel_identities and kernel_identities[name] != identity:
                raise ValueError(f"duplicate RVV kernel {name} has different identity")
            kernel_identities[name] = identity
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
    structural = [
        item["performance"]["e3_over_e2_speedup"] for item in successful
    ]
    retiled = [
        item["performance"]["retiled_over_no_retile_speedup"]
        for item in successful
    ]
    c_prior = [
        item["performance"]["c_prior_over_erase_c_speedup"]
        for item in successful
    ]
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": first["bundle_protocol"],
        "bundle_toolchain_digest": first["bundle_toolchain_digest"],
        "target": first["target"],
        "workload_profile": first.get("workload_profile"),
        "baseline_policy": first.get("baseline_policy"),
        "experiment_spec_protocol": first.get("experiment_spec_protocol"),
        "experiment_spec_digest": first.get("experiment_spec_digest"),
        "measurement_spec_protocol": first.get("measurement_spec_protocol"),
        "measurement_spec_digest": first.get("measurement_spec_digest"),
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": sum(
            float(item.get("elapsed_seconds", 0.0)) for item in payloads
        ),
        "device": payloads[-1]["device"],
        "measurement": measurement,
        "merge_provenance": provenance,
        "summary": {
            "requested": len(kernels),
            "successful": len(successful),
            "failed": len(failures),
            "geomean_e3_over_e2_speedup": _geomean(structural),
            "geomean_retiled_over_no_retile_speedup": _geomean(retiled),
            "geomean_c_prior_over_erase_c_speedup": _geomean(c_prior),
            "b_transformation_winner_count": sum(
                item["performance"]["b_transformation_wins"]
                for item in successful
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
