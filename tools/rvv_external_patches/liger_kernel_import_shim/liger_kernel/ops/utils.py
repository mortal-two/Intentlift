"""Source-compatible version predicate without importing PyTorch."""

from __future__ import annotations

import re

import triton


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", value)[:3])


def compare_version(package: str, operation, reference: str) -> bool:
    if package != "triton":
        raise ValueError(f"unsupported portability-probe package: {package}")
    return bool(operation(_version(triton.__version__), _version(reference)))
