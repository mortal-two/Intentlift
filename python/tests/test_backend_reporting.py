from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentir.backend.reporting import load_disjoint_kernel_reports


def _report(path: Path, kernel: str, *, chip: str = "sm_120a") -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol": "test.performance.v1",
                "device": {
                    "target_chip": chip,
                    "compute_capability": [12, 0],
                    "device_name": "test",
                },
                "kernels": [{"kernel": kernel}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_incremental_report_view_preserves_disjoint_kernel_receipts(
    tmp_path: Path,
) -> None:
    reports, kernels = load_disjoint_kernel_reports(
        [
            _report(tmp_path / "base.json", "base"),
            _report(tmp_path / "increment.json", "new"),
        ]
    )

    assert len(reports) == 2
    assert set(kernels) == {"base", "new"}


def test_incremental_report_view_rejects_duplicate_or_cross_target_receipts(
    tmp_path: Path,
) -> None:
    first = _report(tmp_path / "first.json", "same")
    duplicate = _report(tmp_path / "duplicate.json", "same")
    other_target = _report(
        tmp_path / "other.json", "other", chip="sm_90a"
    )

    with pytest.raises(ValueError, match="duplicate kernel"):
        load_disjoint_kernel_reports([first, duplicate])
    with pytest.raises(ValueError, match="different devices"):
        load_disjoint_kernel_reports([first, other_target])
