from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test/high_performance_live_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "high_performance_live_benchmark", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_latency_comparison_records_paper_suite_and_extension_thresholds() -> None:
    result = benchmark._compare(
        [10.0, 10.0, 10.0, 10.0],
        [10.5, 10.5, 10.5, 10.5],
        resamples=100,
        seed=1,
    )

    assert result["speedup_source_over_generated"] == pytest.approx(1.05)
    assert result["non_regression_2pct_pass"]
    assert result["extension_3pct_speedup_pass"]
    assert result["paper_suite_target_speedup"] == 1.35
    assert not result["paper_suite_target_is_per_kernel"]


def test_latency_summary_rejects_nonpositive_samples() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark._summarize([1.0, 0.0], resamples=100, seed=1)


def test_output_comparison_rejects_missing_or_extra_bindings() -> None:
    class _Tolerance:
        rtol = 0.0
        atol = 0.0
        equal_nan = False

    with pytest.raises(ValueError, match="output binding mismatch"):
        benchmark._allclose(
            {"wanted": object(), "extra": object()},
            {"wanted": object()},
            _Tolerance(),
            expected_outputs=("wanted",),
        )
