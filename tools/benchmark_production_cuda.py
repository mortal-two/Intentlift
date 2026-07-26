#!/usr/bin/env python3
"""Lower, execute, search, and benchmark frozen production J+C snapshots.

The measured implementation is always a cubin emitted from the snapshot's
materialized Intent MLIR through a registered C++ MLIR pass.  Python owns only
snapshot transport, candidate orchestration, CUDA Driver launch binding, and
event timing.  The comparison baseline is the exact declarative Triton workload
adapter already used by SCRC, not a newly authored substitute kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.benchmark import (
    PreparedSingleLaunch,
    do_bench_cudagraph_us,
    do_bench_us,
)
from intentir.backend.cuda.search import PerformanceCache, search_cuda
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    constexpr_input_bindings,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.ir.production_snapshot import (
    ProductionSnapshot,
    load_production_snapshot,
)
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.gate2.triton import PreparedTritonInvocation
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


REPORT_PROTOCOL = "intentir.production.cuda-performance.v1"

CASE_IDS = PRODUCTION_CASE_IDS


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


def _shape(shape: list[int | str], dims: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(dims[item]) if isinstance(item, str) else item for item in shape)


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
        "target_chip": backend_compile.CHIP,
    }


def _tensor_inputs(
    torch: Any,
    snapshot: ProductionSnapshot,
    case: Any,
    runtime_input_names: tuple[str, ...],
) -> list[Any]:
    bindings = case.refeval.input_binding_map
    tensors = {item.name: item for item in snapshot.j.intent.inputs}
    result = []
    for name in runtime_input_names:
        tensor = tensors[name]
        raw = np.ascontiguousarray(np.asarray(case.values[bindings[tensor.name]]))
        result.append(
            torch.as_tensor(raw, dtype=_torch_dtype(torch, tensor.dtype), device="cuda")
        )
    return result


def _tensor_outputs(
    torch: Any,
    snapshot: ProductionSnapshot,
    dims: Mapping[str, int],
) -> list[Any]:
    return [
        torch.empty(
            _shape(tensor.shape, dims),
            dtype=_torch_dtype(torch, tensor.dtype),
            device="cuda",
        )
        for tensor in snapshot.j.intent.outputs
    ]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _one_kernel(
    *,
    name: str,
    snapshot_dir: Path,
    budget: int,
    warmup_ms: int,
    repeat_ms: int,
    timing: str,
    graph_repetitions: int,
    measurement_rounds: int,
    search_prior: str,
    b_search_mode: str,
    cache: PerformanceCache,
) -> dict[str, Any]:
    import torch

    snapshot, snapshot_path = load_production_snapshot(snapshot_dir, name)
    if snapshot.c is None:
        raise ValueError("performance search requires the optional frozen C prior")
    adapter = TRITON_WORKLOAD_ADAPTERS[CASE_IDS[name]]
    prepared_workload = prepare_triton_workload(adapter)
    suite = build_workload_jc_suite(prepared_workload, snapshot.j, snapshot.c)
    case = suite.cases[0]
    dims = case.refeval.dims_map
    constexpr_bindings = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    materialized = materialize_j(
        snapshot.j,
        dims,
        constexpr_bindings=constexpr_bindings,
    )
    state = attach_c(
        materialized,
        snapshot.c if search_prior == "with-c" else None,
        BackendTarget("cuda", backend_compile.CHIP),
    )
    inputs = _tensor_inputs(
        torch,
        snapshot,
        case,
        materialized.runtime_input_names,
    )
    outputs = _tensor_outputs(torch, snapshot, dims)
    tensors = [*inputs, *outputs]

    launch_records: dict[str, Any] = {}

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

    def benchmark(program, candidate) -> float:
        prepared = PreparedSingleLaunch(program, materialized.intent_mlir)
        launch_records[candidate.candidate_id] = prepared.record.to_dict()
        return measure(prepared.thunk(tensors))

    search = search_cuda(
        state,
        benchmark=benchmark,
        benchmark_key=(
            f"{name}:accepted-shape:{sorted(dims.items())}:"
            f"timing={timing}:graph-repetitions={graph_repetitions}"
            f":measurement-rounds={measurement_rounds}"
        ),
        budget=budget,
        include_generic=False,
        cache=cache,
        enable_b_transformations=b_search_mode == "contract-guided",
    )
    if search.winner is None or search.winner_program is None:
        raise RuntimeError("measured search produced no executable winner")
    winner = search.winner_program
    mlir_prepared = PreparedSingleLaunch(winner, materialized.intent_mlir)
    mlir_thunk = mlir_prepared.thunk(tensors)
    mlir_thunk()
    torch.cuda.synchronize()

    actual = {
        runtime_key: tensor.detach().cpu().numpy().copy()
        for runtime_key, tensor in zip(case.output_keys, outputs, strict=True)
    }
    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")
    mlir_j = compare_outputs(
        "mlir-cubin",
        actual,
        "canonical-J",
        reference.outputs,
        j=snapshot.j,
        input_bindings=case.refeval.input_binding_map,
        output_bindings=case.refeval.output_binding_map,
        inputs=case.values,
        policy=DEFAULT_GATE2_POLICY,
    )
    if not mlir_j.passed:
        raise RuntimeError("compiled MLIR result differs from canonical J")

    source_program = prepared_workload.request.source.gate_source
    with PreparedTritonInvocation(
        source_program, case.triton, case.values
    ) as source:
        source.launch()
        torch.cuda.synchronize()
        source_actual = source.copied_outputs()
        source_j = compare_outputs(
            "source-triton",
            source_actual,
            "canonical-J",
            reference.outputs,
            j=snapshot.j,
            input_bindings=case.refeval.input_binding_map,
            output_bindings=case.refeval.output_binding_map,
            inputs=case.values,
            policy=DEFAULT_GATE2_POLICY,
        )
        if not source_j.passed:
            raise RuntimeError("source Triton result differs from canonical J")
        mlir_samples = [float(search.winner_latency_us)]
        mlir_samples.extend(measure(mlir_thunk) for _ in range(2))
        source_samples = [measure(source.thunk()) for _ in range(3)]
        mlir_latency = float(statistics.median(mlir_samples))
        source_latency = float(statistics.median(source_samples))

    kernel = winner.kernels[0]
    return {
        "kernel": name,
        "case_id": CASE_IDS[name],
        "snapshot": {
            "path": str(snapshot_path),
            "snapshot_digest": snapshot.snapshot_digest,
            "j_digest": snapshot.j_digest,
            "c_digest": snapshot.c_digest,
        },
        "extents": dims,
        "materialization": materialized.to_record(),
        "search_prior": {
            "mode": search_prior,
            "c_attached_to_backend_search": search_prior == "with-c",
            "accepted_c_used_only_to_reconstruct_exact_workload": True,
        },
        "b_search_mode": b_search_mode,
        "search": search.to_record(),
        "winner": {
            "candidate": search.winner.to_dict(),
            "lowering": winner.to_record(),
            "launch": mlir_prepared.record.to_dict(),
            "cubin_sha256": _sha256(kernel.cubin),
            "cubin_size_bytes": len(kernel.cubin),
        },
        "correctness": {
            "mlir_j": mlir_j.to_json(),
            "source_j": source_j.to_json(),
            "policy": DEFAULT_GATE2_POLICY.policy_id,
        },
        "performance": {
            "mlir_us": mlir_latency,
            "source_triton_us": source_latency,
            "mlir_over_triton": mlir_latency / source_latency,
            "speedup_over_triton": source_latency / mlir_latency,
            "meets_or_beats_triton": mlir_latency <= source_latency,
            "warmup_ms": warmup_ms,
            "repeat_ms": repeat_ms,
            "timing": timing,
            "graph_repetitions": graph_repetitions,
            "measurement_rounds": measurement_rounds,
            "selection_confirmation": {
                "mlir_batch_us": mlir_samples,
                "source_batch_us": source_samples,
                "batch_aggregation": "median",
                "samples_per_batch": measurement_rounds,
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(CASE_IDS),
        default=sorted(CASE_IDS),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--search-budget", type=int, default=96)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument(
        "--timing", choices=("graph", "event"), default="graph"
    )
    parser.add_argument("--graph-repetitions", type=int, default=100)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument(
        "--search-prior",
        choices=("with-c", "without-c"),
        default="with-c",
        help="Attach or erase C for candidate ordering; J=A+B is unchanged.",
    )
    parser.add_argument(
        "--b-search-mode",
        choices=("contract-guided", "schedule-only"),
        default="contract-guided",
        help=(
            "E3 enables B-conditioned MLIR transformations; E2 keeps the "
            "same complete B contract but searches schedule-only candidates."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.measurement_rounds <= 0:
        raise ValueError("measurement-rounds must be positive")
    if args.graph_repetitions <= 0:
        raise ValueError("graph-repetitions must be positive")
    cache = PerformanceCache(args.cache)
    records: list[dict[str, Any]] = []
    started = time.time_ns()
    for name in args.kernels:
        try:
            record = _one_kernel(
                name=name,
                snapshot_dir=args.snapshot_dir,
                budget=args.search_budget,
                warmup_ms=args.warmup_ms,
                repeat_ms=args.repeat_ms,
                timing=args.timing,
                graph_repetitions=args.graph_repetitions,
                measurement_rounds=args.measurement_rounds,
                search_prior=args.search_prior,
                b_search_mode=args.b_search_mode,
                cache=cache,
            )
            records.append(record)
            perf = record["performance"]
            print(
                f"{name}: {perf['mlir_us']:.4f} us vs "
                f"{perf['source_triton_us']:.4f} us "
                f"({perf['mlir_over_triton']:.3f}x)",
                flush=True,
            )
        except Exception as exc:
            records.append(
                {
                    "kernel": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{name}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    successful = [item for item in records if "performance" in item]
    report = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": _device_record(torch),
        "toolchain": {
            "intent_opt": str(backend_compile.INTENT_OPT),
            "mlir_opt": str(backend_compile.MLIR_OPT),
            "cuda_bin": backend_compile.CUDA_BIN,
        },
        "summary": {
            "requested": len(args.kernels),
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
    return 0 if len(successful) == len(args.kernels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
