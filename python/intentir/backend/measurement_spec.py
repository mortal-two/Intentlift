"""Machine-readable cache/Roofline performance-measurement contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MEASUREMENT_PROTOCOL = "intentir.backend.performance-measurement.v1"
DEFAULT_MEASUREMENT_SPEC = (
    Path(__file__).resolve().parents[3]
    / "spec"
    / "lowering"
    / "backend-performance-measurement-v1.json"
)


def load_measurement_spec(path: Path | None = None) -> dict[str, Any]:
    location = path or DEFAULT_MEASUREMENT_SPEC
    value = json.loads(location.read_text(encoding="utf-8"))
    expected_root = {
        "protocol",
        "scope",
        "comparison_rule",
        "data_regimes",
        "roofline",
        "search_ablation",
        "reporting",
    }
    if not isinstance(value, dict) or set(value) != expected_root:
        raise ValueError("backend measurement spec has missing or unknown root fields")
    if value["protocol"] != MEASUREMENT_PROTOCOL:
        raise ValueError("unsupported backend measurement protocol")
    regimes = value["data_regimes"]
    if not isinstance(regimes, dict) or set(regimes) != {
        "warm",
        "streaming",
        "cold-data",
    }:
        raise ValueError("backend measurement spec has incomplete data regimes")
    for name, record in regimes.items():
        if not isinstance(record, dict) or not record:
            raise ValueError(f"backend measurement regime {name!r} is invalid")
    required = value["reporting"].get("required")
    forbidden = value["reporting"].get("forbidden")
    if not isinstance(required, list) or not required:
        raise ValueError("backend measurement spec requires report fields")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("backend measurement spec requires forbidden claims")
    return value


__all__ = [
    "DEFAULT_MEASUREMENT_SPEC",
    "MEASUREMENT_PROTOCOL",
    "load_measurement_spec",
]
