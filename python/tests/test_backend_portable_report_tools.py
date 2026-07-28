import importlib.util
import base64
import ctypes
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from intentir.verify.gate1.model import ResultRelationKind
from intentir.verify.gate2.compare import close_q
from intentir.verify.gate2.policy import (
    ComparisonPolicy,
    ComparisonPolicyError,
    NumericTolerance,
)


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
rvv_externalize = _load_tool(
    "intentir_rvv_externalize_payloads_tool",
    "tools/externalize_rvv_bundle_payloads.py",
)
cuda_runner = _load_tool(
    "intentir_cuda_target_runner_tool",
    "tools/run_cuda_benchmark_bundle.py",
)


@pytest.mark.parametrize(
    ("left", "right", "kind", "tolerance"),
    [
        (
            np.array([1.0, -2.0, np.inf], dtype=np.float32),
            np.array([1.0 + 1.0e-6, -2.0, np.inf], dtype=np.float32),
            ResultRelationKind.TOLERANCE,
            NumericTolerance(1.0e-5, 1.0e-6),
        ),
        (
            np.array([np.nan, 2.0], dtype=np.float32),
            np.array([np.nan, 2.0], dtype=np.float32),
            ResultRelationKind.TOLERANCE,
            NumericTolerance(0.0, 0.0, equal_nan=False),
        ),
        (
            np.array([np.nan, 2.0], dtype=np.float32),
            np.array([np.nan, 2.0], dtype=np.float32),
            ResultRelationKind.TOLERANCE,
            NumericTolerance(0.0, 0.0, equal_nan=True),
        ),
        (
            np.array([1.0, np.inf], dtype=np.float32),
            np.array([1.0, -np.inf], dtype=np.float32),
            ResultRelationKind.TOLERANCE,
            NumericTolerance(0.0, 0.0),
        ),
    ],
)
def test_cuda_gpu_q_predicate_matches_canonical_close_q(
    left, right, kind, tolerance
):
    torch = pytest.importorskip("torch")
    policy = ComparisonPolicy.create("test.gpu-q.v1", (("f32", tolerance),))
    rule = policy.rule_for(kind, "f32")

    observed, _ = cuda_runner._gpu_close_q_output(
        torch,
        torch.from_numpy(left),
        torch.from_numpy(right),
        rule,
    )

    assert observed is close_q(left, right, rule)


def test_cuda_gpu_q_exact_integer_reports_max_error():
    torch = pytest.importorskip("torch")
    policy = ComparisonPolicy.create("test.gpu-q.v1", ())
    rule = policy.rule_for(ResultRelationKind.EXACT, "i32")

    passed, max_abs_error = cuda_runner._gpu_close_q_output(
        torch,
        torch.tensor([1, 5], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        rule,
    )

    assert passed is False
    assert max_abs_error == 3.0


def test_gate2_q_from_json_rejects_protocol_substitution():
    with pytest.raises(ComparisonPolicyError, match="unsupported Gate2 Q protocol"):
        ComparisonPolicy.from_json(
            {
                "protocol": "intentir.scrc.gate2-q.v999",
                "policy_id": "test.q.v1",
                "tolerances": {},
            }
        )


def _cuda_report(arm: str, latency_us: float) -> dict:
    return {
        "protocol": cuda_summary.REPORT_PROTOCOL,
        "ablation_axes": {"arm": arm},
        "target": {"backend": "cuda", "chip": "sm_120a", "features": []},
        "measurement_spec_protocol": "intentir.backend.performance-measurement.v1",
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
    assert "best_applicable" not in result["per_kernel"][0][
        "external_competitiveness"
    ]


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


def test_cuda_ablation_summary_prefers_same_run_schedule_gain():
    c_fixed = _cuda_report("CF", 8.0)
    schedule = _cuda_report("E2", 4.0)
    intentir = _cuda_report("E3", 2.0)
    schedule["kernels"][0]["performance"]["c_fixed_schedule_ablation"] = {
        "same_run_measurement": True,
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


def test_cuda_ablation_summary_prefers_same_run_schedule_spine_gain():
    e2 = _cuda_report("E2", 4.0)
    e3 = _cuda_report("E3", 2.0)
    e3["kernels"][0]["performance"]["schedule_spine_ablation"] = {
        "same_run_measurement": True,
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
            "workload_profile": "roofline-large-j-only-v1",
            "baseline_policy": "target_refj_only",
        }
    )
    e3["kernels"].append(
        {
            "kernel": "softmax",
            "status": "failed",
            "error": "QD failed",
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


def _rvv_report(kernel: str) -> dict:
    return {
        "protocol": rvv_merge.REPORT_PROTOCOL,
        "bundle_protocol": "intentir.production.rvv-search-bundle.v3",
        "target": {
            "backend": "riscv",
            "chip": "rv64gcv_zfh",
            "features": ["vlen128"],
        },
        "workload_profile": "closed-v1",
        "baseline_policy": "target_refj_only",
        "experiment_spec_protocol": "intentir.backend.ablation-plan.v3",
        "measurement_spec_protocol": "intentir.backend.performance-measurement.v1",
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


def test_rvv_native_sidecar_attaches_independent_baseline(tmp_path):
    path = tmp_path / "native.json"
    record = {
        "protocol": rvv_runner.NATIVE_BASELINE_PROTOCOL,
        "symbol": "intentir_native_baseline",
        "source": "void intentir_native_baseline(void) {}",
        "implementation": "independent-test",
    }
    path.write_text(
        json.dumps(
            {
                "protocol": rvv_runner.NATIVE_BASELINE_SIDECAR_PROTOCOL,
                "baselines": {"dyt": record},
            }
        ),
        encoding="utf-8",
    )

    sidecar = rvv_runner._load_native_baseline_sidecar(path)
    entries = rvv_runner._attach_native_baselines(
        [{"kernel": "dyt", "baseline_policy": "target_refj_only"}],
        sidecar,
    )

    assert entries[0]["native_baseline"] == record
    assert entries[0]["baseline_policy"] == "native_rvv_and_refj"


def test_rvv_native_sidecar_rejects_missing_kernel_baseline():
    with pytest.raises(RuntimeError, match="lacks an independent Native"):
        rvv_runner._attach_native_baselines(
            [{"kernel": "dyt", "baseline_policy": "target_refj_only"}],
            {},
        )


def test_rvv_external_payload_preserves_bytes_and_logical_size(tmp_path: Path):
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    encoded = {
        "protocol": "intentir.transport.ndarray.v1",
        "dtype": "f32",
        "shape": [4],
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }
    bundle = {
        "protocol": rvv_externalize.BUNDLE_PROTOCOL,
        "kernels": [
            {
                "kernel": "dyt",
                "input_order": ["input"],
                "input_bindings": {"input": "x"},
                "output_specs": [{"name": "output", "dtype": "f32", "shape": [4]}],
                "values": {"x": encoded},
                "reference_outputs": {"y": encoded},
            }
        ],
    }

    compact = rvv_externalize.externalize(
        bundle, output_root=tmp_path, relative_root=Path("arrays")
    )
    entry = compact["kernels"][0]
    record = entry["values"]["x"]

    assert "data_base64" not in record
    assert rvv_runner._decode_payload(record, tmp_path) == payload
    assert rvv_runner._logical_io_bytes(entry, tmp_path) == {
        "boundary_input_bytes": 16,
        "boundary_output_bytes": 16,
        "boundary_io_bytes": 32,
    }


def test_rvv_external_payload_rejects_path_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="stay below"):
        rvv_runner._decode_payload({"data_file": "../outside.bin"}, tmp_path)


def test_rvv_merge_preserves_shared_configuration(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "merged.json"
    left.write_text(json.dumps(_rvv_report("left")), encoding="utf-8")
    right.write_text(json.dumps(_rvv_report("right")), encoding="utf-8")

    assert rvv_merge.main([str(left), str(right), "--out", str(out)]) == 0
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["workload_profile"] == "closed-v1"
    assert merged["target"] == {
        "backend": "riscv",
        "chip": "rv64gcv_zfh",
        "features": ["vlen128"],
    }
    assert merged["summary"]["successful"] == 2


def test_rvv_bounded_candidate_union_makes_budget_executable() -> None:
    def candidate(candidate_id: str, parallel: int, reduction: int, *, b: bool):
        return {
            "candidate": {
                "candidate_id": candidate_id,
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
            candidate("fused", 2, 8, b=True),
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
            "canonical_lowering_path": "materialized",
        },
    }

    orders = rvv_runner._bounded_candidate_orders(entry, 2, thread_count=1)

    assert orders["active_e3"] == ["c-fixed", "fused"]
    assert orders["active_e2"] == ["c-fixed", "no-retile"]
    assert orders["erase_e3"] == ["no-retile", "erase-only"]
    assert orders["no_retile"] == ["no-retile"]
    assert orders["c_fixed"] == ["c-fixed"]
    assert orders["e2_evaluation"] == ["c-fixed", "no-retile"]
    assert orders["e3_evaluation"] == ["c-fixed", "fused"]
    assert orders["schedule_only_retiled"] == ["c-fixed"]
    assert orders["timed"] == ["c-fixed", "no-retile", "fused", "erase-only"]

    oracle = rvv_runner._bounded_candidate_orders(
        entry, 2, thread_count=1, measure_c_oracle=True
    )
    assert oracle["timed"] == [
        "no-retile",
        "c-fixed",
        "fused",
        "erase-only",
    ]


def test_rvv_finalists_include_each_arm_winner_and_global_best() -> None:
    measured = [
        {"candidate": {"candidate_id": "cf"}, "latency_us": 8.0},
        {"candidate": {"candidate_id": "e2"}, "latency_us": 5.0},
        {"candidate": {"candidate_id": "e3"}, "latency_us": 3.0},
        {"candidate": {"candidate_id": "erase"}, "latency_us": 4.0},
        {"candidate": {"candidate_id": "other"}, "latency_us": 2.0},
    ]

    finalists = rvv_runner._finalist_candidate_ids(
        measured,
        [["cf"], ["cf", "e2"], ["cf", "e2", "e3"], ["erase"]],
        5,
    )

    assert {"cf", "e2", "e3", "erase"} <= set(finalists)
    assert "other" in finalists


def test_rvv_native_comparator_applies_declared_tolerance_to_all_values(
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
