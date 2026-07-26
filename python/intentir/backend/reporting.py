"""Small helpers for incrementally extending closed-kernel report sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _target_identity(report: Mapping[str, Any]) -> tuple[Any, ...]:
    target = report.get("target")
    if isinstance(target, Mapping):
        return ("target", json.dumps(target, sort_keys=True))
    device = report.get("device", {})
    if not isinstance(device, Mapping):
        raise ValueError("performance report has no target/device identity")
    return (
        "device",
        device.get("target_chip"),
        tuple(device.get("compute_capability", ())),
        device.get("device_name"),
        device.get("machine"),
        device.get("vlen_bits"),
    )


def load_disjoint_kernel_reports(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Load same-target receipts whose kernel sets must be disjoint.

    This is deliberately not a synthetic raw-report merger: each original
    receipt keeps its own toolchain and creation metadata. Consumers receive a
    unified kernel view while retaining the source reports separately.
    """

    if not paths:
        raise ValueError("at least one performance report is required")
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    protocols = {report.get("protocol") for report in reports}
    if len(protocols) != 1:
        raise ValueError("incremental reports use different protocols")
    targets = {_target_identity(report) for report in reports}
    if len(targets) != 1:
        raise ValueError("incremental reports target different devices")
    kernels: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        for item in report.get("kernels", []):
            name = item.get("kernel")
            if not isinstance(name, str) or not name:
                raise ValueError("performance report has an unnamed kernel")
            if name in kernels:
                raise ValueError(
                    f"incremental reports duplicate kernel {name!r}"
                )
            kernels[name] = item
    return reports, kernels


__all__ = ["load_disjoint_kernel_reports"]
