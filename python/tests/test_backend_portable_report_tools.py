import importlib.util
import ctypes
import hashlib
import json
import struct
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
rvv_runner = _load_tool(
    "intentir_rvv_target_runner_tool",
    "tools/run_rvv_benchmark_bundle.py",
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
            "cold_candidate_warmup_ms": 25,
            "cold_candidate_warmup_scope": (
                "untimed code/P-state stabilization before L2-evicted samples"
            ),
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
                "winner": {"candidate": {"candidate_id": "c-fixed-state"}},
                "performance": {
                    "mlir_us": latency_us,
                    "source_triton_us": None,
                    "schedule_spine_ablation": None,
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
    assert result["summary"]["geomean_g_lowering"] == 2.0
    assert result["summary"]["geomean_intentir_speedup_over_source"] is None
    assert result["per_kernel"][0]["external_competitiveness"][
        "intentir_speedup_over_source"
    ] is None


def test_cuda_ablation_summary_reports_formal_three_arm_gains():
    result = cuda_summary.summarize(
        _cuda_report("E2", 4.0),
        _cuda_report("E3", 2.0),
        _cuda_report("CF", 8.0),
    )

    assert result["summary"]["formal_three_arm"] is True
    assert result["summary"]["geomean_g_schedule"] == 2.0
    assert result["summary"]["geomean_g_lowering"] == 2.0
    assert result["per_kernel"][0]["c_fixed_us"] == 8.0
    assert result["per_kernel"][0]["schedule_only_us"] == 4.0
    assert result["per_kernel"][0]["intentir_us"] == 2.0
    assert result["per_kernel"][0]["g_schedule"] == 2.0
    assert result["per_kernel"][0]["g_lowering"] == 2.0


def test_cuda_ablation_summary_prefers_same_receipt_schedule_gain():
    c_fixed = _cuda_report("CF", 8.0)
    schedule = _cuda_report("E2", 4.0)
    intentir = _cuda_report("E3", 2.0)
    schedule["kernels"][0]["performance"]["c_fixed_schedule_ablation"] = {
        "same_candidate_measurement_receipt": True,
        "c_fixed_candidate_id": "c-fixed-state",
        "c_fixed_us": 10.0,
        "schedule_only_best_us": 4.0,
        "schedule_over_c_fixed_speedup": 2.5,
    }

    result = cuda_summary.summarize(schedule, intentir, c_fixed)

    assert result["per_kernel"][0]["g_schedule"] == 2.5
    assert result["per_kernel"][0]["supplementary_diagnostics"][
        "cross_report_g_schedule"
    ] == 2.0


def test_cuda_ablation_summary_prefers_same_receipt_schedule_spine_gain():
    e2 = _cuda_report("E2", 4.0)
    e3 = _cuda_report("E3", 2.0)
    e3["kernels"][0]["performance"]["schedule_spine_ablation"] = {
        "same_candidate_measurement_receipt": True,
        "schedule_spine_best_us": 3.0,
        "expanded_best_us": 2.0,
        "expanded_over_schedule_spine_speedup": 1.5,
    }

    result = cuda_summary.summarize(e2, e3)

    assert result["summary"]["geomean_g_lowering"] == 1.5
    assert result["supplementary_diagnostics"][
        "geomean_cross_report_g_lowering"
    ] == 2.0
    assert result["per_kernel"][0]["g_lowering"] == 1.5


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
    assert failed["schedule_only_error"] == "compile failed"
    assert failed["intentir_error"] == "QD failed"


def test_cuda_ablation_summary_rejects_different_stabilization_protocols():
    e2 = _cuda_report("E2", 2.0)
    e3 = _cuda_report("E3", 1.0)
    e3["measurement"]["cold_candidate_warmup_ms"] = 0

    with pytest.raises(ValueError, match="measurement contracts differ"):
        cuda_summary.summarize(e2, e3)


def _rvv_report(kernel: str, *, toolchain: str = "a" * 64) -> dict:
    return {
        "protocol": rvv_merge.REPORT_PROTOCOL,
        "bundle_protocol": "intentir.production.rvv-search-bundle.v3",
        "bundle_toolchain_digest": toolchain,
        "target": {
            "backend": "riscv",
            "chip": "rv64gcv_zfh",
            "features": ["vlen128"],
        },
        "workload_profile": "closed-v1",
        "baseline_policy": "target_refj_only",
        "experiment_spec_protocol": "intentir.backend.ablation-plan.v3",
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
                    "g_schedule": 1.2,
                    "g_lowering": 1.1,
                },
                "c_prior_efficiency": {
                    "best_at_budget_c_over_erase_c_speedup": 1.0
                },
                "supplementary_diagnostics": {
                    "formal_role": "not_a_main_experiment_arm",
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


def test_rvv_bounded_candidate_union_makes_budget_executable() -> None:
    def candidate(candidate_id: str, parallel: int, reduction: int, *, b: bool):
        return {
            "candidate": {
                "candidate_id": candidate_id,
                "object_sha256": candidate_id.encode().hex().ljust(64, "0"),
                "schedule": {
                    "parallel_tile": parallel,
                    "reduction_tile": reduction,
                },
                "transformation": {
                    "lowering_path_kind": (
                        "producer_consumer_fused" if b else "materialized"
                    ),
                    "b_transformations": ["fuse"] if b else [],
                },
            }
        }

    entry = {
        "candidates": [
            candidate("no-retile", 0, 0, b=False),
            candidate("c-fixed", 2, 8, b=False),
            candidate("fused", 64, 32, b=True),
            candidate("erase-only", 8, 8, b=False),
        ],
        "schedule_only_candidate_ids": ["no-retile", "c-fixed", "erase-only"],
        "concrete_search_plan": {
            "schedule_only_c_prior_order": [
                "c-fixed",
                "no-retile",
                "erase-only"
            ],
            "schedule_only_erase_c_order": [
                "no-retile",
                "erase-only",
                "c-fixed"
            ],
            "c_prior_order": [
                "c-fixed",
                "no-retile",
                "erase-only",
                "fused"
            ],
            "erase_c_order": ["no-retile", "erase-only", "c-fixed", "fused"],
            "c_fixed_candidate_id": "c-fixed",
            "candidate_universe_digest": hashlib.sha256(
                json.dumps(
                    sorted(["no-retile", "c-fixed", "fused", "erase-only"]),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "canonical_lowering_path": "materialized",
        },
    }

    orders = rvv_runner._bounded_candidate_orders(entry, 2)

    assert orders["active_e3"] == ["c-fixed", "no-retile"]
    assert orders["active_e2"] == ["c-fixed", "no-retile"]
    assert orders["erase_e3"] == ["no-retile", "erase-only"]
    assert orders["no_retile"] == ["no-retile"]
    assert orders["c_fixed"] == ["c-fixed"]
    assert orders["e2_evaluation"] == ["c-fixed", "no-retile"]
    assert orders["e3_evaluation"] == ["c-fixed", "no-retile"]
    assert orders["schedule_only_retiled"] == ["c-fixed"]
    assert orders["timed"] == ["c-fixed", "no-retile", "erase-only"]

    oracle = rvv_runner._bounded_candidate_orders(
        entry, 2, measure_c_oracle=True
    )
    assert oracle["timed"] == [
        "no-retile",
        "c-fixed",
        "fused",
        "erase-only",
    ]


def test_rvv_bounded_candidate_union_rejects_duplicate_native_objects() -> None:
    candidate = {
        "schedule": {"parallel_tile": 0, "reduction_tile": 0},
        "transformation": {
            "lowering_path_kind": "materialized",
            "b_transformations": [],
        },
        "object_sha256": "a" * 64,
    }
    entry = {
        "candidates": [
            {"candidate": {**candidate, "candidate_id": "left"}},
            {"candidate": {**candidate, "candidate_id": "right"}},
        ],
        "schedule_only_candidate_ids": ["left", "right"],
        "concrete_search_plan": {
            "schedule_only_c_prior_order": ["left", "right"],
            "schedule_only_erase_c_order": ["left", "right"],
            "c_prior_order": ["left", "right"],
            "erase_c_order": ["left", "right"],
            "c_fixed_candidate_id": "left",
            "candidate_universe_digest": hashlib.sha256(
                json.dumps(["left", "right"], separators=(",", ":")).encode()
            ).hexdigest(),
            "canonical_lowering_path": "materialized",
        },
    }

    with pytest.raises(ValueError, match="deduplicate native objects"):
        rvv_runner._bounded_candidate_orders(entry, 1)


def test_rvv_native_comparator_applies_frozen_tolerance_to_all_values(
    tmp_path: Path,
) -> None:
    comparator = rvv_runner._NativeComparator(tmp_path)
    actual = ctypes.create_string_buffer(struct.pack("<2f", 1.0, 2.00001), 8)
    expected = ctypes.create_string_buffer(struct.pack("<2f", 1.0, 2.0), 8)
    rule = {
        "dtype": "f32",
        "tolerance": {"atol": 1e-4, "rtol": 0.0, "equal_nan": False},
    }

    passed, maximum = comparator.close(
        actual,
        expected,
        byte_count=8,
        element_count=2,
        dtype="f32",
        rule=rule,
    )
    assert passed
    assert maximum == pytest.approx(1e-5, rel=2e-2)

    rule["tolerance"]["atol"] = 1e-7
    passed, maximum = comparator.close(
        actual,
        expected,
        byte_count=8,
        element_count=2,
        dtype="f32",
        rule=rule,
    )
    assert not passed
    assert maximum > 1e-7
