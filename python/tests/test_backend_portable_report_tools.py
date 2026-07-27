import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cuda_summary = _load_tool(
    "intentir_cuda_ablation_summary_tool",
    "tools/summarize_cuda_benchmark_ablation.py",
)
rvv_merge = _load_tool(
    "intentir_rvv_report_merge_tool",
    "tools/merge_rvv_benchmark_reports.py",
)


def _cuda_report(arm: str, latency_us: float) -> dict:
    return {
        "protocol": cuda_summary.REPORT_PROTOCOL,
        "ablation_axes": {"arm": arm},
        "target": {"backend": "cuda", "chip": "sm_120a", "features": []},
        "bundle_toolchain_digest": "a" * 64,
        "experiment_spec_digest": "b" * 64,
        "measurement_spec_protocol": "intentir.backend.performance-measurement.v1",
        "measurement_spec_digest": "c" * 64,
        "workload_profile": "roofline-large-j-only-v1",
        "baseline_policy": "target_refj_only",
        "device": {
            "device_name": "test-gpu",
            "compute_capability": [12, 0],
            "total_memory_bytes": 1,
            "l2_cache_bytes": 1,
        },
        "measurement": {
            "data_regime": "cold-data",
            "effective_timing": "cuda-event-single-shot",
            "graph_repetitions": 1,
            "warmup_ms": 25,
            "repeat_ms": 100,
            "measurement_rounds": 3,
            "cold_iterations_per_round": 16,
            "cache_flush_bytes": 2,
            "detected_l2_cache_bytes": 1,
            "cache_eviction_in_timed_region": False,
        },
        "kernels": [
            {
                "kernel": "masked_copy",
                "j_digest": "d" * 64,
                "c_digest": "e" * 64,
                "extents": {"x0": 16},
                "workload_valuation": {"j_unchanged": True},
                "candidates": [],
                "performance": {
                    "mlir_us": latency_us,
                    "source_triton_us": None,
                    "c_ordering_ablation": {
                        "c_prior_over_erase_c_speedup": 1.0
                    },
                },
            }
        ],
    }


def test_cuda_ablation_summary_accepts_target_refj_only_reports():
    result = cuda_summary.summarize(
        _cuda_report("E2", 2.0),
        _cuda_report("E3", 1.0),
    )

    assert result["summary"]["kernel_count"] == 1
    assert result["summary"]["source_baseline_kernel_count"] == 0
    assert result["summary"]["geomean_e3_over_e2_speedup"] == 2.0
    assert result["summary"]["geomean_e2_speedup_over_source"] is None
    assert result["per_kernel"][0]["e3_speedup_over_source"] is None


def test_cuda_ablation_summary_preserves_failed_horizontal_cells():
    e2 = _cuda_report("E2", 2.0)
    e3 = _cuda_report("E3", 1.0)
    e2["kernels"].append(
        {
            "kernel": "softmax",
            "status": "failed",
            "error": "compile failed",
            "j_digest": "f" * 64,
            "workload_profile": "roofline-large-j-only-v1",
            "baseline_policy": "target_refj_only",
        }
    )
    e3["kernels"].append(
        {
            "kernel": "softmax",
            "status": "failed",
            "error": "QD failed",
            "j_digest": "f" * 64,
            "workload_profile": "roofline-large-j-only-v1",
            "baseline_policy": "target_refj_only",
        }
    )

    result = cuda_summary.summarize(e2, e3)

    assert result["summary"]["kernel_count"] == 2
    assert result["summary"]["paired_success_count"] == 1
    assert result["summary"]["failed_or_missing_count"] == 1
    failed = next(item for item in result["per_kernel"] if item["kernel"] == "softmax")
    assert failed["status"] == "failed_or_missing"
    assert failed["e2_error"] == "compile failed"
    assert failed["e3_error"] == "QD failed"


def _rvv_report(kernel: str, *, toolchain: str = "a" * 64) -> dict:
    return {
        "protocol": rvv_merge.REPORT_PROTOCOL,
        "bundle_protocol": "intentir.production.rvv-search-bundle.v2",
        "bundle_toolchain_digest": toolchain,
        "target": {
            "backend": "riscv",
            "chip": "rv64gcv_zfh",
            "features": ["vlen128"],
        },
        "workload_profile": "closed-v1",
        "baseline_policy": "target_refj_only",
        "experiment_spec_protocol": "intentir.backend.ablation-plan.v2",
        "experiment_spec_digest": "b" * 64,
        "measurement_spec_protocol": "intentir.backend.performance-measurement.v1",
        "measurement_spec_digest": "c" * 64,
        "device": {"machine": "riscv64", "vlen_bits": 128, "cache": {}},
        "measurement": {
            "data_regime": "cold-data",
            "effective_timing": "monotonic-clock-single-shot",
        },
        "elapsed_seconds": 1.0,
        "kernels": [
            {
                "kernel": kernel,
                "case_id": f"case:{kernel}",
                "snapshot_digest": kernel * 4,
                "j_digest": kernel * 4,
                "c_digest": kernel * 4,
                "extents": {"x0": 1},
                "performance": {
                    "e3_over_e2_speedup": 1.1,
                    "retile_over_no_retile_speedup": 1.0,
                    "c_prior_over_erase_c_speedup": 1.0,
                    "b_transformation_wins": 0,
                },
            }
        ],
        "failures": [],
    }


def test_rvv_merge_preserves_report_identity(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "merged.json"
    left.write_text(json.dumps(_rvv_report("left")), encoding="utf-8")
    right.write_text(json.dumps(_rvv_report("right")), encoding="utf-8")

    assert rvv_merge.main([str(left), str(right), "--out", str(out)]) == 0
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["bundle_toolchain_digest"] == "a" * 64
    assert merged["workload_profile"] == "closed-v1"
    assert merged["summary"]["successful"] == 2


def test_rvv_merge_rejects_different_toolchains(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "merged.json"
    left.write_text(json.dumps(_rvv_report("left")), encoding="utf-8")
    right.write_text(
        json.dumps(_rvv_report("right", toolchain="f" * 64)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bundle_toolchain_digest"):
        rvv_merge.main([str(left), str(right), "--out", str(out)])
