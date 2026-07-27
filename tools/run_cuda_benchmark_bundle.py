#!/usr/bin/env python3
"""Execute and search a precompiled production CUDA bundle on its target GPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from intentir.backend.cuda.benchmark import (
    ExtractedSingleLaunch,
    PreparedSingleLaunch,
    do_bench_cudagraph_us,
    do_bench_us,
)
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    decode_triton_spec,
    decode_values,
)
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.backend.roofline import (
    RooflineCeilings,
    evaluate_logical_roofline,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.triton import PreparedTritonInvocation


REPORT_PROTOCOL = "intentir.production.cuda-remote-performance.v3"
BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v2"


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bool": "bool",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    return getattr(torch, aliases[dtype.removeprefix("torch.").lower()])


def _device_record(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "l2_cache_bytes": int(getattr(properties, "L2_cache_size", 0)),
    }


def _measure_cold_data_us(
    torch: Any,
    thunk: Any,
    *,
    eviction: Any,
    iterations: int,
) -> float:
    """Measure single launches after a target-side L2 eviction.

    The eviction and synchronization are deliberately outside the CUDA-event
    interval.  This is a cold-*data* protocol, not process cold-start and not a
    claim about compulsory instruction-cache misses.
    """

    samples: list[float] = []
    for _ in range(iterations):
        eviction.add_(1.0)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        thunk()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) * 1000.0)
    return float(statistics.median(samples))


def _warmup_thunks(
    torch: Any,
    thunks: tuple[Any, ...],
    *,
    warmup_ms: int,
) -> int:
    """Stabilize kernel code and GPU clocks without adding timed samples.

    Cold-data measurements still evict L2 immediately before every timed launch.
    This controls process/JIT/P-state history, not the measured data-cache state.
    Multiple thunks are traversed round-robin so paired arms receive an equal
    number of untimed launches.
    """

    if warmup_ms <= 0 or not thunks:
        return 0
    deadline = time.perf_counter() + warmup_ms / 1000.0
    rounds = 0
    while True:
        for thunk in thunks:
            thunk()
        rounds += 1
        if rounds % 32 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() >= deadline:
                break
    torch.cuda.synchronize()
    return rounds


def _measure_cuda_roofs(
    torch: Any,
    *,
    memory_mib: int,
    matrix_extent: int,
) -> RooflineCeilings:
    if memory_mib <= 0 or matrix_extent <= 0:
        raise ValueError("Roofline calibration sizes must be positive")
    elements = memory_mib * 1024 * 1024 // 4
    source = torch.empty(elements, dtype=torch.float32, device="cuda").uniform_()
    destination = torch.empty_like(source)
    memory_samples = [
        do_bench_us(
            lambda: destination.copy_(source),
            warmup_ms=50,
            repeat_ms=200,
        )
        for _ in range(3)
    ]
    memory_us = float(statistics.median(memory_samples))
    memory_gbps = 2 * source.numel() * source.element_size() / memory_us / 1_000.0

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        left = torch.randn(
            (matrix_extent, matrix_extent), dtype=torch.float32, device="cuda"
        )
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        compute_samples = [
            do_bench_us(
                lambda: torch.mm(left, right, out=output),
                warmup_ms=50,
                repeat_ms=200,
            )
            for _ in range(3)
        ]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    compute_us = float(statistics.median(compute_samples))
    fp32_gflops = 2 * matrix_extent**3 / compute_us / 1_000.0
    torch.cuda.synchronize()
    return RooflineCeilings(
        memory_gbps=memory_gbps,
        fp32_gflops=fp32_gflops,
        provenance={
            "kind": "target-side-sustained-microbenchmark",
            "memory_method": "device-to-device-fp32-copy-read-plus-write",
            "memory_mib_per_tensor": memory_mib,
            "memory_samples_us": memory_samples,
            "compute_method": "torch.mm-fp32-tf32-disabled",
            "matrix_extent": matrix_extent,
            "compute_samples_us": compute_samples,
            "allocation_in_timed_region": False,
        },
    )


def _actual_outputs(
    output_keys: list[str], outputs: list[Any]
) -> dict[str, np.ndarray]:
    return {
        key: tensor.detach().cpu().numpy().copy()
        for key, tensor in zip(output_keys, outputs, strict=True)
    }


def _compare(
    *,
    name: str,
    actual: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    snapshot: ProductionSnapshot,
    entry: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
):
    return compare_outputs(
        name,
        actual,
        "canonical-J-reference",
        reference,
        j=snapshot.j,
        input_bindings=entry["input_bindings"],
        output_bindings=entry["output_bindings"],
        inputs=values,
        policy=DEFAULT_GATE2_POLICY,
    )


def _one_kernel(
    entry: Mapping[str, Any],
    *,
    timing: str,
    graph_repetitions: int,
    warmup_ms: int,
    repeat_ms: int,
    measurement_rounds: int,
    finalist_count: int,
    finalist_remeasurements: int,
    confirmation_batches: int,
    roofline_ceilings: RooflineCeilings | None,
    data_regime: str,
    cold_iterations: int,
    cache_eviction: Any | None,
) -> dict[str, Any]:
    import torch

    snapshot = ProductionSnapshot.from_dict(entry["snapshot"])
    values = decode_values(entry["values"])
    reference = decode_values(entry["reference_outputs"])
    inputs = [
        torch.as_tensor(
            np.ascontiguousarray(
                values[entry["input_bindings"][name]]
            ),
            device="cuda",
        )
        for name in entry["input_order"]
    ]
    outputs = [
        torch.empty(
            tuple(item["shape"]),
            dtype=_torch_dtype(torch, item["dtype"]),
            device="cuda",
        )
        for item in entry["output_specs"]
    ]
    tensors = [*inputs, *outputs]
    workload_cost = entry.get("workload_cost")
    logical_io_bytes = sum(item.numel() * item.element_size() for item in tensors)
    if (
        workload_cost is not None
        and int(workload_cost["boundary_io_bytes"]) != logical_io_bytes
    ):
        raise ValueError("CUDA tensors disagree with J workload-cost receipt")
    floating_point_ops = (
        int(workload_cost["floating_point_ops"])
        if workload_cost is not None
        else 0
    )

    def measure(thunk) -> float:
        samples = []
        for _ in range(measurement_rounds):
            if data_regime == "cold-data":
                if cache_eviction is None:
                    raise AssertionError("cold-data regime requires cache eviction")
                samples.append(
                    _measure_cold_data_us(
                        torch,
                        thunk,
                        eviction=cache_eviction,
                        iterations=cold_iterations,
                    )
                )
            elif timing == "graph":
                samples.append(
                    do_bench_cudagraph_us(
                        thunk, repetitions=graph_repetitions
                    )
                )
            else:
                samples.append(
                    do_bench_us(
                        thunk,
                        warmup_ms=warmup_ms,
                        repeat_ms=repeat_ms,
                    )
                )
        return float(statistics.median(samples))

    candidate_records: list[dict[str, Any]] = []
    candidate_artifacts: dict[
        str, tuple[Mapping[str, Any], ExtractedSingleLaunch, Any]
    ] = {}
    for candidate in entry["candidates"]:
        extracted = ExtractedSingleLaunch.from_dict(candidate["launch"])
        if (
            extracted.source_mlir_sha256
            != entry["materialization"]["intent_mlir_sha256"]
        ):
            raise ValueError("candidate launch does not belong to bundled M0")
        prepared = PreparedSingleLaunch.from_extracted(extracted)
        thunk = prepared.thunk(tensors)
        thunk()
        torch.cuda.synchronize()
        agreement = _compare(
            name="mlir-cubin",
            actual=_actual_outputs(entry["output_keys"], outputs),
            reference=reference,
            snapshot=snapshot,
            entry=entry,
            values=values,
        )
        if not agreement.passed:
            candidate_records.append(
                {
                    "candidate": candidate["candidate"],
                    "status": "incorrect",
                    "correctness": agreement.to_json(),
                    "concrete_state": candidate.get("concrete_state"),
                    "lowering": candidate.get("lowering"),
                }
            )
            continue
        cold_warmup_rounds = (
            _warmup_thunks(torch, (thunk,), warmup_ms=warmup_ms)
            if data_regime == "cold-data"
            else 0
        )
        latency = measure(thunk)
        candidate_records.append(
            {
                "candidate": candidate["candidate"],
                "status": "measured",
                "latency_us": latency,
                "correctness": agreement.to_json(),
                "concrete_state": candidate.get("concrete_state"),
                "lowering": candidate.get("lowering"),
                "launch": extracted.record.to_dict(),
                "cubin_sha256": extracted.cubin_sha256,
                "cubin_size_bytes": len(extracted.cubin),
                "logical_boundary_gbps": logical_io_bytes / latency / 1_000.0,
                "achieved_gflops": floating_point_ops / latency / 1_000.0,
                "cold_pre_measure_warmup_rounds": cold_warmup_rounds,
            }
        )
        candidate_id = candidate["candidate"]["candidate_id"]
        candidate_artifacts[candidate_id] = (candidate, extracted, thunk)
    measured_indices = [
        index
        for index, record in enumerate(candidate_records)
        if record["status"] == "measured"
    ]
    measured_index_by_id = {
        candidate_records[index]["candidate"]["candidate_id"]: index
        for index in measured_indices
    }
    search_orders = entry["search_orders"]
    default_order = list(measured_index_by_id)
    with_c_order = list(
        search_orders.get("with_c_concrete_order", default_order)
    )
    without_c_order = list(
        search_orders.get("without_c_concrete_order", default_order)
    )
    active_order_name = str(search_orders.get("active_order", "with_c"))
    if active_order_name not in {"with_c", "without_c"}:
        raise ValueError(f"unknown active CUDA search order {active_order_name!r}")
    active_order = (
        with_c_order if active_order_name == "with_c" else without_c_order
    )
    schedule_spine_order = list(
        search_orders.get("active_schedule_spine_concrete_order", [])
    )

    initial_ranked = sorted(
        measured_indices,
        key=lambda item: candidate_records[item]["latency_us"],
    )
    # Always remeasure each ordering's provisional winner.  Otherwise a small
    # finalist_count could accidentally give only one C arm fresh samples.
    mandatory_indices: list[int] = []
    for order in (with_c_order, without_c_order, schedule_spine_order):
        arm_indices = [
            measured_index_by_id[candidate_id]
            for candidate_id in order
            if candidate_id in measured_index_by_id
        ]
        if arm_indices:
            mandatory_indices.append(
                min(
                    arm_indices,
                    key=lambda item: candidate_records[item]["latency_us"],
                )
            )
    ranked_indices = []
    for index in [*mandatory_indices, *initial_ranked]:
        if index not in ranked_indices:
            ranked_indices.append(index)
        if (
            len(ranked_indices) >= finalist_count
            and all(item in ranked_indices for item in mandatory_indices)
        ):
            break
    finalist_samples: dict[int, list[float]] = {
        index: [] for index in ranked_indices
    }
    for round_index in range(finalist_remeasurements):
        round_order = (
            ranked_indices
            if round_index % 2 == 0
            else list(reversed(ranked_indices))
        )
        for index in round_order:
            record = candidate_records[index]
            candidate_id = record["candidate"]["candidate_id"]
            thunk = candidate_artifacts[candidate_id][2]
            finalist_samples[index].append(measure(thunk))
    for index, samples in finalist_samples.items():
        if not samples:
            continue
        record = candidate_records[index]
        # Initial ranking samples are deliberately excluded from final
        # selection.  Fresh alternating round-robin batches remove incumbent
        # and thermal-order bias from the target-side search.
        record["latency_us"] = float(statistics.median(samples))
        record["finalist_batch_us"] = samples
    if not measured_indices:
        raise RuntimeError("no correct bundled MLIR candidate")

    freshly_remeasured = {
        index for index, samples in finalist_samples.items() if samples
    }

    def best_for_order(order: list[str]) -> dict[str, Any] | None:
        records = [
            candidate_records[measured_index_by_id[candidate_id]]
            for candidate_id in order
            if candidate_id in measured_index_by_id
            and (
                not freshly_remeasured
                or measured_index_by_id[candidate_id] in freshly_remeasured
            )
        ]
        return min(records, key=lambda item: item["latency_us"]) if records else None

    with_c_best = best_for_order(with_c_order)
    without_c_best = best_for_order(without_c_order)
    schedule_spine_best = best_for_order(schedule_spine_order)
    best_record = best_for_order(active_order)
    if best_record is None:
        raise RuntimeError(
            f"active CUDA search order {active_order_name!r} has no correct candidate"
        )
    best_id = best_record["candidate"]["candidate_id"]
    winner, extracted, winner_thunk = candidate_artifacts[best_id]

    baseline_policy = str(entry.get("baseline_policy", "source_and_refj"))
    if baseline_policy not in {"source_and_refj", "target_refj_only"}:
        raise ValueError(f"unknown CUDA baseline policy {baseline_policy!r}")
    source_agreement = None
    source_latency: float | None = None
    source_samples: list[float] = []
    mlir_samples: list[float] = []
    paired_batches: list[dict[str, Any]] = []
    confirmation_warmup_rounds = 0
    if baseline_policy == "source_and_refj":
        source_record = entry["source_triton"]
        source_program = KernelSource.from_code(
            source_record["code"],
            source_record["function"],
            trusted=True,
        )
        source_spec = decode_triton_spec(source_record["execution_spec"])
        with PreparedTritonInvocation(
            source_program, source_spec, values
        ) as source:
            source.launch()
            torch.cuda.synchronize()
            source_agreement = _compare(
                name="source-triton",
                actual=source.copied_outputs(),
                reference=reference,
                snapshot=snapshot,
                entry=entry,
                values=values,
            )
            if not source_agreement.passed:
                raise RuntimeError("source Triton baseline differs from canonical J")
            source_thunk = source.thunk()
            confirmation_warmup_rounds = _warmup_thunks(
                torch,
                (source_thunk, winner_thunk),
                warmup_ms=warmup_ms,
            )
            for pair_index in range(confirmation_batches):
                if pair_index % 2 == 0:
                    order = ("source", "target")
                    source_us = measure(source_thunk)
                    target_us = measure(winner_thunk)
                else:
                    order = ("target", "source")
                    target_us = measure(winner_thunk)
                    source_us = measure(source_thunk)
                source_samples.append(source_us)
                mlir_samples.append(target_us)
                paired_batches.append(
                    {
                        "pair_index": pair_index,
                        "order": list(order),
                        "source_us": source_us,
                        "target_us": target_us,
                        "speedup_source_over_target": source_us / target_us,
                    }
                )
            source_latency = float(statistics.median(source_samples))
    else:
        # The J-only large-shape profile extends the backend stress domain beyond
        # the accepted Source's one-block execution contract.  Reconfirm only the
        # already Target--RefJ-correct fixed winner and emit no Source claim.
        confirmation_warmup_rounds = _warmup_thunks(
            torch,
            (winner_thunk,),
            warmup_ms=warmup_ms,
        )
        for batch_index in range(confirmation_batches):
            target_us = measure(winner_thunk)
            mlir_samples.append(target_us)
            paired_batches.append(
                {
                    "batch_index": batch_index,
                    "order": ["target"],
                    "target_us": target_us,
                }
            )

    mlir_latency = float(statistics.median(mlir_samples))
    roofline_evaluations = None
    if workload_cost is not None and roofline_ceilings is not None:
        roofline_evaluations = {
            "mlir": evaluate_logical_roofline(
                workload_cost, mlir_latency, roofline_ceilings
            )
        }
        if source_latency is not None:
            roofline_evaluations["source"] = evaluate_logical_roofline(
                workload_cost, source_latency, roofline_ceilings
            )
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
        "workload_profile": entry.get("workload_profile", "closed-v1"),
        "baseline_policy": baseline_policy,
        "workload_valuation": entry.get("workload_valuation"),
        "snapshot_digest": snapshot.snapshot_digest,
        "j_digest": snapshot.j_digest,
        "c_digest": snapshot.c_digest,
        "extents": entry["extents"],
        "contract_view": entry["contract_view"],
        "search_orders": entry["search_orders"],
        "candidate_count": len(entry["candidates"]),
        "candidates": candidate_records,
        "winner": {
            "candidate": winner["candidate"],
            "concrete_state": winner.get("concrete_state"),
            "lowering": winner.get("lowering"),
            "launch": extracted.record.to_dict(),
            "cubin_sha256": extracted.cubin_sha256,
            "cubin_size_bytes": len(extracted.cubin),
        },
        "correctness": {
            "source_j": (
                source_agreement.to_json()
                if source_agreement is not None
                else None
            ),
            "target_refj_measured_candidate_count": len(measured_indices),
            "formal_source_replay_closure_claimed": False,
            "policy": DEFAULT_GATE2_POLICY.policy_id,
        },
        "performance": {
            "mlir_us": mlir_latency,
            "source_triton_us": source_latency,
            "mlir_over_triton": (
                mlir_latency / source_latency
                if source_latency is not None
                else None
            ),
            "speedup_over_triton": (
                source_latency / mlir_latency
                if source_latency is not None
                else None
            ),
            "meets_or_beats_triton": (
                mlir_latency <= source_latency
                if source_latency is not None
                else None
            ),
            "c_ordering_ablation": {
                "budget": search_orders.get("budget"),
                "budget_unit": search_orders.get("budget_unit"),
                "same_transported_candidate_union": True,
                "active_order": active_order_name,
                "with_c_best_candidate_id": (
                    with_c_best["candidate"]["candidate_id"]
                    if with_c_best is not None
                    else None
                ),
                "without_c_best_candidate_id": (
                    without_c_best["candidate"]["candidate_id"]
                    if without_c_best is not None
                    else None
                ),
                "with_c_best_us": (
                    with_c_best["latency_us"]
                    if with_c_best is not None
                    else None
                ),
                "without_c_best_us": (
                    without_c_best["latency_us"]
                    if without_c_best is not None
                    else None
                ),
                "c_prior_over_erase_c_speedup": (
                    without_c_best["latency_us"] / with_c_best["latency_us"]
                    if with_c_best is not None and without_c_best is not None
                    else None
                ),
            },
            "schedule_spine_ablation": (
                {
                    "same_candidate_measurement_receipt": True,
                    "same_total_search_visit": True,
                    "schedule_spine_candidate_count": len(schedule_spine_order),
                    "schedule_spine_best_candidate_id": (
                        schedule_spine_best["candidate"]["candidate_id"]
                    ),
                    "schedule_spine_best_us": schedule_spine_best["latency_us"],
                    "expanded_best_candidate_id": best_id,
                    "expanded_best_us": best_record["latency_us"],
                    "expanded_over_schedule_spine_speedup": (
                        schedule_spine_best["latency_us"]
                        / best_record["latency_us"]
                    ),
                }
                if schedule_spine_best is not None
                else None
            ),
            "logical_io": {
                "boundary_io_bytes": logical_io_bytes,
                "mlir_logical_boundary_gbps": (
                    logical_io_bytes / mlir_latency / 1_000.0
                ),
                "source_logical_boundary_gbps": (
                    logical_io_bytes / source_latency / 1_000.0
                    if source_latency is not None
                    else None
                ),
                "traffic_definition": (
                    "sum of certified boundary input and output payload bytes; "
                    "not measured DRAM transactions"
                ),
            },
            "roofline_point": {
                "j_workload_cost": workload_cost,
                "arithmetic_intensity_flop_per_byte": (
                    floating_point_ops / logical_io_bytes
                ),
                "mlir_achieved_gflops": (
                    floating_point_ops / mlir_latency / 1_000.0
                ),
                "source_achieved_gflops": (
                    floating_point_ops / source_latency / 1_000.0
                    if source_latency is not None
                    else None
                ),
                "hardware_counter_traffic": None,
                "interpretation": "J-derived logical/minimum-traffic point",
                "evaluations": roofline_evaluations,
            },
            "data_regime": data_regime,
            "timing": (
                "cuda-event-single-shot"
                if data_regime == "cold-data"
                else timing
            ),
            "requested_warm_timing": timing,
            "graph_repetitions": graph_repetitions,
            "warmup_ms": warmup_ms,
            "repeat_ms": repeat_ms,
            "measurement_rounds": measurement_rounds,
            "cold_iterations_per_round": (
                cold_iterations if data_regime == "cold-data" else None
            ),
            "cache_eviction_in_timed_region": False,
            "selection_confirmation": {
                "mlir_batch_us": mlir_samples,
                "source_batch_us": source_samples,
                "batch_aggregation": "median",
                "samples_per_batch": measurement_rounds,
                "finalist_count": len(ranked_indices),
                "finalist_remeasurements": finalist_remeasurements,
                "winner_selection_scope": (
                    "freshly-remeasured-finalists"
                    if freshly_remeasured
                    else "initial-ranking-samples"
                ),
                "comparison_protocol": (
                    "paired-alternating.v1"
                    if source_latency is not None
                    else "target-only-fresh-batches.v1"
                ),
                "paired_batches": paired_batches,
                "warmup_pair_discarded": source_latency is not None,
                "pre_confirmation_warmup_ms": warmup_ms,
                "pre_confirmation_warmup_rounds": confirmation_warmup_rounds,
                "warmup_method": (
                    "untimed-round-robin-before-L2-evicted-samples"
                ),
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*")
    parser.add_argument(
        "--timing", choices=("graph", "event"), default="graph"
    )
    parser.add_argument(
        "--data-regime", choices=("warm", "cold-data"), default="warm"
    )
    parser.add_argument("--cold-iterations", type=int, default=9)
    parser.add_argument("--cache-flush-mib", type=int, default=256)
    parser.add_argument("--graph-repetitions", type=int, default=200)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--finalist-remeasurements", type=int, default=3)
    parser.add_argument("--confirmation-batches", type=int, default=4)
    parser.add_argument("--roofline-calibration", action="store_true")
    parser.add_argument("--roofline-memory-mib", type=int, default=256)
    parser.add_argument("--roofline-matrix-extent", type=int, default=4096)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported CUDA benchmark bundle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.measurement_rounds <= 0:
        raise ValueError("measurement-rounds must be positive")
    if args.finalist_count <= 0:
        raise ValueError("finalist-count must be positive")
    if args.finalist_remeasurements < 0:
        raise ValueError("finalist-remeasurements cannot be negative")
    if args.confirmation_batches <= 0:
        raise ValueError("confirmation-batches must be positive")
    if args.cold_iterations <= 0:
        raise ValueError("cold-iterations must be positive")
    if args.cache_flush_mib <= 0:
        raise ValueError("cache-flush-mib must be positive")
    roofline_ceilings = (
        _measure_cuda_roofs(
            torch,
            memory_mib=args.roofline_memory_mib,
            matrix_extent=args.roofline_matrix_extent,
        )
        if args.roofline_calibration
        else None
    )
    capability = torch.cuda.get_device_capability()
    target = bundle["target"]["chip"]
    expected_capability = cuda_compute_capability_for_chip(target)
    declared_capability = tuple(bundle["target"].get("compute_capability", ()))
    if declared_capability != expected_capability:
        raise RuntimeError(
            "bundle CUDA target metadata is inconsistent: "
            f"chip {target} names {expected_capability}, metadata names "
            f"{declared_capability}"
        )
    if capability != expected_capability:
        raise RuntimeError(
            f"{target} bundle requires compute capability "
            f"{expected_capability}, got {capability}"
        )
    device_properties = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    )
    l2_cache_bytes = int(getattr(device_properties, "L2_cache_size", 0))
    cache_flush_bytes = args.cache_flush_mib * 1024 * 1024
    if args.data_regime == "cold-data" and (
        l2_cache_bytes <= 0 or cache_flush_bytes <= l2_cache_bytes
    ):
        raise ValueError(
            "cold-data cache-flush working set must exceed the detected L2: "
            f"flush={cache_flush_bytes}, l2={l2_cache_bytes}"
        )
    cache_eviction = None
    if args.data_regime == "cold-data":
        cache_eviction = torch.empty(
            cache_flush_bytes // 4,
            dtype=torch.float32,
            device="cuda",
        ).uniform_()
        # Initialize the eviction kernel/runtime before any measured candidate.
        cache_eviction.add_(1.0)
        torch.cuda.synchronize()
    selected = set(args.kernels or [])
    entries = [
        item
        for item in bundle["kernels"]
        if not selected or item["kernel"] in selected
    ]
    records: list[dict[str, Any]] = []
    started = time.time_ns()
    for entry in entries:
        try:
            record = _one_kernel(
                entry,
                timing=args.timing,
                graph_repetitions=args.graph_repetitions,
                warmup_ms=args.warmup_ms,
                repeat_ms=args.repeat_ms,
                measurement_rounds=args.measurement_rounds,
                finalist_count=args.finalist_count,
                finalist_remeasurements=args.finalist_remeasurements,
                confirmation_batches=args.confirmation_batches,
                roofline_ceilings=roofline_ceilings,
                data_regime=args.data_regime,
                cold_iterations=args.cold_iterations,
                cache_eviction=cache_eviction,
            )
            records.append(record)
            perf = record["performance"]
            if perf["source_triton_us"] is None:
                print(
                    f"{entry['kernel']}: {perf['mlir_us']:.4f} us "
                    "(Target--RefJ-only; no Source claim)",
                    flush=True,
                )
            else:
                print(
                    f"{entry['kernel']}: {perf['mlir_us']:.4f} us vs "
                    f"{perf['source_triton_us']:.4f} us "
                    f"({perf['mlir_over_triton']:.3f}x)",
                    flush=True,
                )
        except Exception as exc:
            records.append(
                {
                    "kernel": entry["kernel"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"{entry['kernel']}: FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                break
    successful = [item for item in records if "performance" in item]
    c_ordering_speedups = [
        item["performance"]["c_ordering_ablation"][
            "c_prior_over_erase_c_speedup"
        ]
        for item in successful
        if item["performance"]["c_ordering_ablation"][
            "c_prior_over_erase_c_speedup"
        ]
        is not None
    ]
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "bundle_toolchain_digest": bundle["toolchain_digest"],
        "target": bundle["target"],
        "workload_profile": bundle.get("workload_profile", "closed-v1"),
        "baseline_policy": bundle.get("baseline_policy", "source_and_refj"),
        "ablation_axes": bundle.get("ablation_axes"),
        "experiment_spec_digest": bundle.get("experiment_spec_digest"),
        "measurement_spec_protocol": bundle.get("measurement_spec_protocol"),
        "measurement_spec_digest": bundle.get("measurement_spec_digest"),
        "device": _device_record(torch),
        "roofline_ceilings": (
            roofline_ceilings.to_record()
            if roofline_ceilings is not None
            else None
        ),
        "measurement": {
            "data_regime": args.data_regime,
            "requested_warm_timing": args.timing,
            "effective_timing": (
                "cuda-event-single-shot"
                if args.data_regime == "cold-data"
                else args.timing
            ),
            "graph_repetitions": args.graph_repetitions,
            "warmup_ms": args.warmup_ms,
            "cold_candidate_warmup_ms": (
                args.warmup_ms if args.data_regime == "cold-data" else None
            ),
            "cold_candidate_warmup_scope": (
                "untimed code/P-state stabilization before L2-evicted samples"
                if args.data_regime == "cold-data"
                else None
            ),
            "repeat_ms": args.repeat_ms,
            "measurement_rounds": args.measurement_rounds,
            "cold_iterations_per_round": (
                args.cold_iterations
                if args.data_regime == "cold-data"
                else None
            ),
            "cache_flush_bytes": (
                cache_flush_bytes if args.data_regime == "cold-data" else None
            ),
            "detected_l2_cache_bytes": l2_cache_bytes,
            "cache_eviction_in_timed_region": False,
            "cold_scope": (
                "data-cache state only; process/module/JIT cold start excluded"
                if args.data_regime == "cold-data"
                else None
            ),
        },
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "summary": {
            "requested": len(entries),
            "successful": len(successful),
            "failed": len(records) - len(successful),
            "meets_or_beats_triton": sum(
                item["performance"]["meets_or_beats_triton"] is True
                for item in successful
            ),
            "source_baseline_count": sum(
                item["performance"]["source_triton_us"] is not None
                for item in successful
            ),
            "geomean_c_prior_over_erase_c_speedup": (
                float(
                    np.exp(
                        np.mean(np.log(np.asarray(c_ordering_speedups)))
                    )
                )
                if c_ordering_speedups
                else None
            ),
        },
        "kernels": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
