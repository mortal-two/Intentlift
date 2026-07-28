from __future__ import annotations

from intentir.backend.ablation_report import (
    mechanism_applicability,
    summarize_ablation_records,
)


def _record(
    arm: str,
    kernel: str,
    latency: float,
    *,
    paths: list[str],
    structural: bool = False,
    reduction: bool = False,
    fma: bool = False,
    trace: list[str] | None = None,
):
    state = {
        "transformation_trace": trace or ["cuda.test.v1"],
        "mlir_digest": "1" * 64,
    }
    return {
        "arm": arm,
        "kernel": kernel,
        "performance": {
            "selected_us": latency,
            "source_triton_us": 10.0,
            "speedup_over_triton": 10.0 / latency,
        },
        "search": {
            "time_to_best_ms": 2.0,
            "actual_state_evidence": {
                "admitted_fast_path_ids": paths,
                "concrete_post_pass_mlir_states": 2,
                "structural_path_expansion_observed": structural,
                "reduction_topology_expansion_observed": reduction,
                "fma_contraction_observed": fma,
                "precision_or_approximation_observed": False,
            },
            "attempts": [
                {
                    "status": "measured",
                    "concrete_state": state,
                }
            ],
            "backend_closure": None,
        },
    }


def test_mechanism_applicability_uses_actual_compiled_receipts() -> None:
    record = _record(
        "E3",
        "k0",
        4.0,
        paths=["cuda.a", "cuda.b"],
        structural=True,
        reduction=True,
        fma=True,
        trace=["cuda.test.v1", "cuda.b.rank0_host_value_capture.v1"],
    )

    applicability = mechanism_applicability(record)

    assert applicability["schedule"]
    assert applicability["structural"]
    assert applicability["local_b"]
    assert applicability["reduction_tree"]
    assert applicability["fma"]
    assert not applicability["precision_approx"]


def test_local_b_applicability_rejects_requested_or_failed_traces() -> None:
    record = _record(
        "L1",
        "k0",
        4.0,
        paths=["cuda.a"],
        trace=["cuda.b.rank0_host_value_capture.v1"],
    )
    attempt = record["search"]["attempts"][0]
    attempt["status"] = "failed"
    assert not mechanism_applicability(record)["local_b"]

    attempt["status"] = "measured"
    attempt["concrete_state"]["transformation_trace"] = [
        "cuda.b.rank0_host_value_capture.request.v1"
    ]
    assert not mechanism_applicability(record)["local_b"]


def test_schedule_applicability_requires_multiple_actual_states_and_path_list() -> None:
    record = _record("E2", "k0", 4.0, paths=["cuda.a"])
    evidence = record["search"]["actual_state_evidence"]

    evidence["concrete_post_pass_mlir_states"] = 1
    assert not mechanism_applicability(record)["schedule"]
    evidence["concrete_post_pass_mlir_states"] = 2
    evidence["admitted_fast_path_ids"] = "cuda.a"
    assert not mechanism_applicability(record)["schedule"]


def test_summary_keeps_all_corpus_and_applicable_subset_separate() -> None:
    records = [
        _record("E2", "k0", 8.0, paths=["cuda.a"]),
        _record("E2", "k1", 9.0, paths=["cuda.a"]),
        _record(
            "E3",
            "k0",
            4.0,
            paths=["cuda.a", "cuda.b"],
            structural=True,
        ),
        _record("E3", "k1", 9.0, paths=["cuda.a"]),
        # A merged input may contain unrelated records; the explicit selected
        # universe is the report boundary and must filter them out.
        _record("E2", "k2", 1.0, paths=["cuda.a"]),
        _record("E3", "k2", 0.1, paths=["cuda.a", "cuda.b"], structural=True),
    ]

    summary = summarize_ablation_records(
        records,
        selected_kernels=("k0", "k1"),
        kernel_universe=("k0", "k1"),
        selected_arms=("E2", "E3"),
    )

    effect = summary["paired_effects"][0]
    assert summary["selection_scope"] == "all_corpus"
    assert effect["paired_all_corpus"]["count"] == 2
    assert effect["paired_applicable_subset"]["count"] == 1
    assert effect["paired_applicable_subset"]["kernels"] == ["k0"]
    assert effect["paired_applicable_subset"]["geomean_gain"] == 2.0


def test_primary_schedule_and_lowering_gains_require_same_receipt() -> None:
    e2 = _record("E2", "k0", 8.0, paths=["cuda.a"])
    e2["performance"]["c_fixed_schedule_ablation"] = {
        "same_candidate_measurement_receipt": True,
        "c_fixed_us": 12.0,
        "schedule_only_best_us": 8.0,
        "schedule_over_c_fixed_speedup": 1.5,
    }
    e3 = _record("E3", "k0", 4.0, paths=["cuda.a"], reduction=True)
    e3["search"]["schedule_spine_ablation"] = {
        "same_candidate_measurement_receipt": True,
        "expanded_state_measured_count": 2,
    }
    e3["performance"]["schedule_spine_ablation"] = {
        "same_candidate_measurement_receipt": True,
        "schedule_spine_best_us": 8.0,
        "expanded_best_us": 4.0,
        "expanded_over_schedule_spine_speedup": 2.0,
    }

    summary = summarize_ablation_records(
        (e2, e3),
        selected_kernels=("k0",),
        kernel_universe=("k0",),
        selected_arms=("E2", "E3"),
    )

    schedule, lowering = summary["same_receipt_effects"]
    assert schedule["same_receipt_all_corpus"]["geomean_gain"] == 1.5
    assert lowering["same_receipt_all_corpus"]["geomean_gain"] == 2.0
    assert lowering["same_receipt_applicable_subset"]["kernels"] == ["k0"]

    e3["performance"]["schedule_spine_ablation"][
        "same_candidate_measurement_receipt"
    ] = False
    rejected = summarize_ablation_records(
        (e2, e3),
        selected_kernels=("k0",),
        kernel_universe=("k0",),
        selected_arms=("E2", "E3"),
    )
    assert rejected["same_receipt_effects"][1]["same_receipt_all_corpus"][
        "count"
    ] == 0
