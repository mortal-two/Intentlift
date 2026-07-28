"""Evidence-preserving aggregation for backend performance ablations."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


REPORT_SUMMARY_PROTOCOL = "intentir.backend.ablation-summary.v1"


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0 else None


def _geomean(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value) and value > 0]
    if not finite:
        return None
    return math.exp(sum(math.log(value) for value in finite) / len(finite))


def _performance(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("performance")
    return value if isinstance(value, Mapping) else None


def _selected_latency(record: Mapping[str, Any]) -> float | None:
    performance = _performance(record)
    if performance is None:
        return None
    return _finite_positive(
        performance.get("selected_us", performance.get("mlir_us"))
    )


def _search(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("search")
    return value if isinstance(value, Mapping) else None


def _actual_state_evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    search = _search(record)
    if search is None:
        return {}
    value = search.get("actual_state_evidence")
    return value if isinstance(value, Mapping) else {}


def _local_b_observed(record: Mapping[str, Any]) -> bool:
    search = _search(record)
    if search is None:
        return False
    attempts = search.get("attempts")
    if not isinstance(attempts, list):
        return False
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or attempt.get("status") not in {
            "compiled",
            "measured",
        }:
            continue
        state = attempt.get("concrete_state")
        if not isinstance(state, Mapping):
            continue
        trace = state.get("transformation_trace")
        if isinstance(trace, list) and any(
            isinstance(item, str)
            and item.startswith("cuda.b.")
            and ".request." not in item
            for item in trace
        ):
            return True
    return False


def mechanism_applicability(record: Mapping[str, Any]) -> dict[str, bool]:
    """Classify applicability only from admitted post-pass compiler states."""

    evidence = _actual_state_evidence(record)
    search = _search(record)
    concrete_count = int(evidence.get("concrete_post_pass_mlir_states", 0) or 0)
    admitted_paths = evidence.get("admitted_fast_path_ids", [])
    return {
        "schedule": (
            concrete_count > 1
            and isinstance(admitted_paths, list)
            and len(admitted_paths) >= 1
        ),
        "structural": bool(
            evidence.get("structural_path_expansion_observed", False)
        ),
        "expanded_state": bool(
            isinstance(search, Mapping)
            and isinstance(search.get("schedule_spine_ablation"), Mapping)
            and int(
                search["schedule_spine_ablation"].get(
                    "expanded_state_measured_count", 0
                )
                or 0
            )
            > 0
        ),
        "fusion_materialization": bool(
            evidence.get("fusion_materialization_pair_observed", False)
        ),
        "local_b": _local_b_observed(record),
        "reduction_tree": bool(
            evidence.get("reduction_topology_expansion_observed", False)
        ),
        "fma": bool(evidence.get("fma_contraction_observed", False)),
        "precision_approx": bool(
            evidence.get("precision_or_approximation_observed", False)
        ),
        "measured": _selected_latency(record) is not None,
        "searched": search is not None,
    }


def _formal_closed(record: Mapping[str, Any]) -> bool:
    search = _search(record)
    if search is not None:
        closure = search.get("backend_closure")
        return isinstance(closure, Mapping) and closure.get("passed") is True
    baseline = record.get("baseline")
    if isinstance(baseline, Mapping):
        closure = baseline.get("backend_closure")
        return isinstance(closure, Mapping) and closure.get("passed") is True
    return False


def _fusion_contrast(record: Mapping[str, Any]) -> dict[str, Any] | None:
    search = _search(record)
    if search is None:
        return None
    attempts = search.get("attempts")
    if not isinstance(attempts, list):
        return None
    materialized: list[tuple[float, str]] = []
    whole_graph: list[tuple[float, str]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        latency = _finite_positive(attempt.get("latency_us"))
        lowering = attempt.get("lowering")
        candidate = attempt.get("candidate")
        if latency is None or not isinstance(lowering, Mapping):
            continue
        if not isinstance(candidate, Mapping):
            continue
        pipeline_id = lowering.get("pipeline_id")
        kernel_count = lowering.get("kernel_count")
        candidate_id = str(candidate.get("candidate_id", ""))
        if (
            pipeline_id == "cuda.pipeline.linalg-materialized.v1"
            and isinstance(kernel_count, int)
            and kernel_count > 1
        ):
            materialized.append((latency, candidate_id))
        elif isinstance(kernel_count, int) and kernel_count == 1:
            whole_graph.append((latency, candidate_id))
    if not materialized or not whole_graph:
        return None
    materialized_best = min(materialized)
    whole_graph_best = min(whole_graph)
    return {
        "kernel": str(record["kernel"]),
        "materialized_candidate": materialized_best[1],
        "materialized_us": materialized_best[0],
        "whole_graph_candidate": whole_graph_best[1],
        "whole_graph_us": whole_graph_best[0],
        "materialization_elimination_gain": (
            materialized_best[0] / whole_graph_best[0]
        ),
    }


def _arm_summary(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    requested_kernels: Sequence[str],
) -> dict[str, Any]:
    successful = [item for item in records if _selected_latency(item) is not None]
    speedups = [
        value
        for item in successful
        for value in (
            _finite_positive((_performance(item) or {}).get("speedup_over_triton")),
        )
        if value is not None
    ]
    elapsed = [
        value
        for item in successful
        for value in (
            _finite_positive((_search(item) or {}).get("time_to_best_ms")),
        )
        if value is not None
    ]
    applicability = {
        mechanism: sorted(
            str(item["kernel"])
            for item in successful
            if mechanism_applicability(item)[mechanism]
        )
        for mechanism in (
            "schedule",
            "expanded_state",
            "structural",
            "fusion_materialization",
            "local_b",
            "reduction_tree",
            "fma",
            "precision_approx",
        )
    }
    fusion_contrasts = [
        contrast
        for item in successful
        for contrast in (_fusion_contrast(item),)
        if contrast is not None
    ]
    return {
        "arm": arm,
        "requested": len(requested_kernels),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "missing_records": len(requested_kernels) - len(records),
        "formal_backend_closed": sum(_formal_closed(item) for item in successful),
        "geomean_speedup_over_source": _geomean(speedups),
        "median_time_to_best_ms": statistics.median(elapsed) if elapsed else None,
        "applicability": {
            mechanism: {
                "count": len(kernels),
                "kernels": kernels,
            }
            for mechanism, kernels in applicability.items()
        },
        "within_arm_fusion_materialization_contrast": {
            "count": len(fusion_contrasts),
            "geomean_gain": _geomean(
                item["materialization_elimination_gain"]
                for item in fusion_contrasts
            ),
            "per_kernel": fusion_contrasts,
        },
    }


def _paired_gain(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline_arm: str,
    enhanced_arm: str,
    applicability: str | None,
) -> dict[str, Any]:
    baseline = {
        str(item["kernel"]): item
        for item in records_by_arm.get(baseline_arm, ())
        if _selected_latency(item) is not None
    }
    enhanced = {
        str(item["kernel"]): item
        for item in records_by_arm.get(enhanced_arm, ())
        if _selected_latency(item) is not None
    }
    kernels = sorted(set(baseline) & set(enhanced))
    rows = []
    for kernel in kernels:
        left = _selected_latency(baseline[kernel])
        right = _selected_latency(enhanced[kernel])
        assert left is not None and right is not None
        applicable = (
            True
            if applicability is None
            else mechanism_applicability(enhanced[kernel])[applicability]
        )
        rows.append(
            {
                "kernel": kernel,
                "baseline_us": left,
                "enhanced_us": right,
                "gain": left / right,
                "applicable": applicable,
            }
        )
    applicable_rows = [item for item in rows if item["applicable"]]
    return {
        "baseline_arm": baseline_arm,
        "enhanced_arm": enhanced_arm,
        "applicability_mechanism": applicability,
        "paired_all_corpus": {
            "count": len(rows),
            "geomean_gain": _geomean(item["gain"] for item in rows),
            "wins": sum(item["gain"] > 1.0 for item in rows),
        },
        "paired_applicable_subset": {
            "count": len(applicable_rows),
            "kernels": [item["kernel"] for item in applicable_rows],
            "geomean_gain": _geomean(
                item["gain"] for item in applicable_rows
            ),
            "wins": sum(item["gain"] > 1.0 for item in applicable_rows),
        },
        "per_kernel": rows,
    }


def _same_receipt_effect(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    arm: str,
    receipt_field: str,
    gain_field: str,
    applicability: str,
) -> dict[str, Any]:
    """Summarize a causal contrast from one search/timing receipt.

    Cross-arm records remain useful diagnostics, but they are not the primary
    attribution because separate launches can drift thermally.  This helper
    only accepts the explicit same-receipt marker emitted by CUDA search.
    """

    rows: list[dict[str, Any]] = []
    for record in records_by_arm.get(arm, ()):
        performance = _performance(record)
        receipt = (
            performance.get(receipt_field)
            if isinstance(performance, Mapping)
            else None
        )
        if not isinstance(receipt, Mapping):
            continue
        if receipt.get("same_candidate_measurement_receipt") is not True:
            continue
        gain = _finite_positive(receipt.get(gain_field))
        if gain is None:
            continue
        rows.append(
            {
                "kernel": str(record["kernel"]),
                "gain": gain,
                "applicable": mechanism_applicability(record)[applicability],
                "receipt": dict(receipt),
            }
        )
    applicable = [item for item in rows if item["applicable"]]
    return {
        "arm": arm,
        "receipt_field": receipt_field,
        "gain_field": gain_field,
        "applicability_mechanism": applicability,
        "same_receipt_all_corpus": {
            "count": len(rows),
            "geomean_gain": _geomean(item["gain"] for item in rows),
            "wins": sum(item["gain"] > 1.0 for item in rows),
        },
        "same_receipt_applicable_subset": {
            "count": len(applicable),
            "kernels": [item["kernel"] for item in applicable],
            "geomean_gain": _geomean(item["gain"] for item in applicable),
            "wins": sum(item["gain"] > 1.0 for item in applicable),
        },
        "per_kernel": rows,
    }


def summarize_ablation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_kernels: Sequence[str],
    kernel_universe: Sequence[str],
    selected_arms: Sequence[str],
) -> dict[str, Any]:
    """Build all-corpus and mechanism-applicable summaries without imputation."""

    selected = tuple(sorted(set(selected_kernels)))
    universe = tuple(sorted(set(kernel_universe)))
    if not selected or not set(selected) <= set(universe):
        raise ValueError("selected kernels must be a non-empty universe subset")
    records_by_arm = {
        arm: tuple(
            item
            for item in records
            if item.get("arm") == arm and item.get("kernel") in selected
        )
        for arm in selected_arms
    }
    pair_specs = (
        ("E1", "E2", "schedule"),
        ("E2", "E3", "structural"),
        ("E3", "E4", None),
        ("L0", "L1", "local_b"),
        ("N0", "N1", "reduction_tree"),
        ("N1", "N2", "fma"),
        ("N2", "N3", "precision_approx"),
    )
    return {
        "protocol": REPORT_SUMMARY_PROTOCOL,
        "kernel_universe": list(universe),
        "selected_kernels": list(selected),
        "selection_scope": (
            "all_corpus" if selected == universe else "explicit_subset"
        ),
        "arms": {
            arm: _arm_summary(arm, records_by_arm[arm], selected)
            for arm in selected_arms
        },
        "paired_effects": [
            _paired_gain(
                records_by_arm,
                baseline_arm=left,
                enhanced_arm=right,
                applicability=mechanism,
            )
            for left, right, mechanism in pair_specs
            if left in records_by_arm and right in records_by_arm
        ],
        "same_receipt_effects": [
            _same_receipt_effect(
                records_by_arm,
                arm="E2",
                receipt_field="c_fixed_schedule_ablation",
                gain_field="schedule_over_c_fixed_speedup",
                applicability="schedule",
            ),
            _same_receipt_effect(
                records_by_arm,
                arm="E3",
                receipt_field="schedule_spine_ablation",
                gain_field="expanded_over_schedule_spine_speedup",
                applicability="expanded_state",
            ),
        ],
        "claim_boundary": (
            "all-corpus aggregates retain failures/not-matched kernels; "
            "applicable-subset gains require an admitted post-pass compiler receipt; "
            "primary schedule/lowering gains require same-receipt measurements"
        ),
    }


__all__ = [
    "REPORT_SUMMARY_PROTOCOL",
    "mechanism_applicability",
    "summarize_ablation_records",
]
