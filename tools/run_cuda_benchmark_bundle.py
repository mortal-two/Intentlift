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
    decode_triton_spec,
    decode_values,
)
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.triton import PreparedTritonInvocation


REPORT_PROTOCOL = "intentir.production.cuda-remote-performance.v1"
BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v1"


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
    }


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

    def measure(thunk) -> float:
        samples = []
        for _ in range(measurement_rounds):
            if timing == "graph":
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
                }
            )
            continue
        latency = measure(thunk)
        candidate_records.append(
            {
                "candidate": candidate["candidate"],
                "status": "measured",
                "latency_us": latency,
                "correctness": agreement.to_json(),
                "launch": extracted.record.to_dict(),
                "cubin_sha256": extracted.cubin_sha256,
                "cubin_size_bytes": len(extracted.cubin),
            }
        )
        candidate_id = candidate["candidate"]["candidate_id"]
        candidate_artifacts[candidate_id] = (candidate, extracted, thunk)
    measured_indices = [
        index
        for index, record in enumerate(candidate_records)
        if record["status"] == "measured"
    ]
    for index in sorted(
        measured_indices,
        key=lambda item: candidate_records[item]["latency_us"],
    )[:finalist_count]:
        record = candidate_records[index]
        candidate_id = record["candidate"]["candidate_id"]
        thunk = candidate_artifacts[candidate_id][2]
        samples = [float(record["latency_us"])]
        samples.extend(measure(thunk) for _ in range(finalist_remeasurements))
        record["latency_us"] = float(statistics.median(samples))
        record["finalist_batch_us"] = samples
    if not measured_indices:
        raise RuntimeError("no correct bundled MLIR candidate")
    best_record = min(
        (candidate_records[index] for index in measured_indices),
        key=lambda item: item["latency_us"],
    )
    best_id = best_record["candidate"]["candidate_id"]
    winner, extracted, winner_thunk = candidate_artifacts[best_id]

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
        source_samples = [
            measure(source.thunk()) for _ in range(confirmation_batches)
        ]
        source_latency = float(statistics.median(source_samples))

    mlir_samples = [float(best_record["latency_us"])]
    mlir_samples.extend(
        measure(winner_thunk) for _ in range(confirmation_batches - 1)
    )
    mlir_latency = float(statistics.median(mlir_samples))
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
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
            "launch": extracted.record.to_dict(),
            "cubin_sha256": extracted.cubin_sha256,
            "cubin_size_bytes": len(extracted.cubin),
        },
        "correctness": {
            "source_j": source_agreement.to_json(),
            "policy": DEFAULT_GATE2_POLICY.policy_id,
        },
        "performance": {
            "mlir_us": mlir_latency,
            "source_triton_us": source_latency,
            "mlir_over_triton": mlir_latency / source_latency,
            "speedup_over_triton": source_latency / mlir_latency,
            "meets_or_beats_triton": mlir_latency <= source_latency,
            "timing": timing,
            "graph_repetitions": graph_repetitions,
            "warmup_ms": warmup_ms,
            "repeat_ms": repeat_ms,
            "measurement_rounds": measurement_rounds,
            "selection_confirmation": {
                "mlir_batch_us": mlir_samples,
                "source_batch_us": source_samples,
                "batch_aggregation": "median",
                "samples_per_batch": measurement_rounds,
                "finalist_count": min(finalist_count, len(measured_indices)),
                "finalist_remeasurements": finalist_remeasurements,
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
    parser.add_argument("--graph-repetitions", type=int, default=200)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--finalist-remeasurements", type=int, default=1)
    parser.add_argument("--confirmation-batches", type=int, default=2)
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
    capability = torch.cuda.get_device_capability()
    target = bundle["target"]["chip"]
    if target == "sm_90a" and capability != (9, 0):
        raise RuntimeError(
            f"sm_90a bundle requires compute capability 9.0, got {capability}"
        )
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
            )
            records.append(record)
            perf = record["performance"]
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
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "bundle_toolchain_digest": bundle["toolchain_digest"],
        "target": bundle["target"],
        "device": _device_record(torch),
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "summary": {
            "requested": len(entries),
            "successful": len(successful),
            "failed": len(records) - len(successful),
            "meets_or_beats_triton": sum(
                item["performance"]["meets_or_beats_triton"]
                for item in successful
            ),
        },
        "kernels": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
