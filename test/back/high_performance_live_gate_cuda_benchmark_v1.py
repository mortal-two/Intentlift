"""Archived benchmark for a historical Gate1/Gate2 CUDA-lowering report.

This source snapshot expects the pre-2026-07-20 three-way live helper contract.
It is retained for audit; reproduce the full CLI from commit ``c59bd69`` rather
than importing the current source/replay-only live harness.

This first performance adapter deliberately handles only automatically selected
single-launch CUDA programs.  Unsupported multi-launch/generic cases remain
explicitly unmeasured; they are never credited from historical handwritten P7
receipts.  The adapter recompiles the accepted Gate1 artifact, verifies that the
same structural selector direct-hits, performs a fresh correctness check, then
times preloaded cubin and Triton launchers with identical methodology.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from intentir.backend import orchestrate as backend_orchestrate
from intentir.backend.launch import load_module
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.back.gate2_production_lowering_v1 import CudaLoweringRunner

import high_performance_live_batch as batch
import scrc_live_e2e as live


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def _summarize(
    samples_us: Sequence[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(samples_us, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("latency summary requires at least two samples")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("latency samples must be finite and positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "samples_us": [float(value) for value in values],
        "min_us": float(np.min(values)),
        "median_us": median,
        "p95_us": _percentile(values, 95),
        "max_us": float(np.max(values)),
        "mean_us": float(np.mean(values)),
        "stddev_us": float(np.std(values, ddof=1)),
        "median_bootstrap_95ci_us": [
            _percentile(medians, 2.5),
            _percentile(medians, 97.5),
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _compare(
    generated_us: Sequence[float],
    source_us: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    generated = np.asarray(generated_us, dtype=np.float64)
    source = np.asarray(source_us, dtype=np.float64)
    if generated.shape != source.shape or generated.ndim != 1:
        raise ValueError("latency samples must be paired one-dimensional arrays")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, generated.size, size=(resamples, generated.size)
    )
    generated_boot = np.median(generated[indices], axis=1)
    source_boot = np.median(source[indices], axis=1)
    speedup_boot = source_boot / generated_boot
    generated_median = float(np.median(generated))
    source_median = float(np.median(source))
    speedup = source_median / generated_median
    ratio_ci = generated_boot / source_boot
    speedup_ci = [
        _percentile(speedup_boot, 2.5),
        _percentile(speedup_boot, 97.5),
    ]
    non_regression = _percentile(ratio_ci, 97.5) <= 1.02
    extension_speedup = speedup >= 1.03 and speedup_ci[0] > 1.0
    return {
        "paired_sample_count": int(generated.size),
        "speedup_source_over_generated": speedup,
        "speedup_bootstrap_95ci": speedup_ci,
        "generated_over_source": generated_median / source_median,
        "generated_over_source_bootstrap_95ci": [
            _percentile(ratio_ci, 2.5),
            _percentile(ratio_ci, 97.5),
        ],
        "non_regression_2pct_pass": bool(non_regression),
        "extension_3pct_speedup_pass": bool(extension_speedup),
        "paper_suite_target_speedup": 1.35,
        "paper_suite_target_is_per_kernel": False,
    }


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return getattr(torch, aliases.get(dtype.removeprefix("torch."), dtype))


def _single_launch_plan(
    binary_mlir: str, kernel: Any, *, abi_count: int
) -> tuple[dict[str, Any], list[tuple[str, int | float]]]:
    host_name, host_text = backend_orchestrate._host_func_text(binary_mlir)
    _signature, _ordered, pointer_args = (
        backend_orchestrate._parse_signature_blockargs(host_text)
    )
    aligned = backend_orchestrate._map_input_aligned_ptrs(pointer_args)
    if len(aligned) != abi_count:
        raise ValueError(
            f"single-launch benchmark needs {abi_count} memrefs, got {len(aligned)}"
        )
    constants = backend_orchestrate._parse_constants(host_text)
    sites = backend_orchestrate._parse_launches(host_text)
    if len(sites) != 1:
        raise ValueError(f"generated program has {len(sites)} launches, not one")
    site = sites[0]
    if site.binary_module != kernel.binary_name or site.func_symbol != kernel.kernel_name:
        raise ValueError("generated host launch does not match the compiled cubin")

    class _Constants:
        pass

    host = _Constants()
    host.constants = constants
    grid = tuple(
        backend_orchestrate._resolve_const_index(host, value)
        for value in site.grid_ssa
    )
    block = tuple(
        backend_orchestrate._resolve_const_index(host, value)
        for value in site.block_ssa
    )
    shared = (
        int(constants[site.dynamic_shared_ssa][0])
        if site.dynamic_shared_ssa is not None
        else 0
    )
    plan: list[tuple[str, int | float]] = []
    pointer_indices: list[int] = []
    for operand in site.args:
        if operand.ty.startswith("!llvm.ptr"):
            if operand.ssa not in aligned:
                raise ValueError("launch pointer is not a host ABI memref")
            index = aligned[operand.ssa]
            pointer_indices.append(index)
            plan.append(("ptr", index))
        elif operand.ty in {"i64", "index"}:
            plan.append(("i64", int(constants[operand.ssa][0])))
        elif operand.ty == "i32":
            plan.append(("i32", int(constants[operand.ssa][0])))
        elif operand.ty == "f32":
            plan.append(
                ("f32", backend_orchestrate._parse_f32(constants[operand.ssa][0]))
            )
        else:
            raise ValueError(f"unsupported single-launch argument {operand.ty!r}")
    if sorted(pointer_indices) != list(range(abi_count)):
        raise ValueError(
            "single-launch benchmark requires the complete uncompressed Intent ABI"
        )
    return (
        {
            "host_function": host_name,
            "binary_name": site.binary_module,
            "kernel_name": site.func_symbol,
            "grid": list(grid),
            "block": list(block),
            "dynamic_shared_memory_bytes": shared,
            "pointer_indices": pointer_indices,
        },
        plan,
    )


def _load_source_module(path: Path, function: str) -> tuple[Any, str]:
    name = f"_intentir_hp_bench_{function}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load captured source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, name


def _source_launcher(case: Any, gate2_case: Any) -> tuple[Callable[[], None], dict[str, Any]]:
    import torch

    module, module_name = _load_source_module(case.source_path, case.function_name)
    kernel = getattr(module, case.function_name)
    arguments: list[Any] = []
    outputs: dict[str, Any] = {}
    for argument in gate2_case.triton.arguments:
        if argument.role.value == "scalar":
            value = (
                gate2_case.values[argument.value_key]
                if argument.value_key is not None
                else argument.value
            )
            if isinstance(value, np.generic):
                value = value.item()
            arguments.append(value)
            continue
        dtype = _torch_dtype(torch, argument.dtype or "f32")
        if argument.role.value == "output":
            tensor = torch.empty(
                tuple(argument.shape or ()), dtype=dtype, device="cuda"
            )
            outputs[argument.output_key or argument.name] = tensor
        else:
            raw = np.asarray(gate2_case.values[argument.value_key])
            tensor = torch.as_tensor(raw, dtype=dtype, device="cuda").clone()
            if argument.role.value == "inout":
                outputs[argument.output_key or argument.name] = tensor
        arguments.append(tensor)

    def launch() -> None:
        kernel[gate2_case.triton.grid](
            *arguments, **dict(gate2_case.triton.schedule)
        )

    launch.__intentir_module_name__ = module_name
    return launch, outputs


def _generated_launcher(
    lowering_spec: Any,
    lowering_artifact: Any,
    values: Mapping[str, Any],
) -> tuple[Callable[[], None], dict[str, Any], dict[str, Any]]:
    import torch

    bundle = lowering_spec.bundle
    input_tensors: dict[str, Any] = {}
    abi_tensors: list[Any] = []
    for tensor in bundle.intent.inputs:
        raw = values[bundle.input_bindings[tensor.name]]
        value = torch.as_tensor(
            np.asarray(raw), dtype=_torch_dtype(torch, tensor.dtype), device="cuda"
        ).clone()
        input_tensors[tensor.name] = value
        abi_tensors.append(value)
    aliases = dict(lowering_artifact.output_aliases)
    outputs: dict[str, Any] = {}
    for index, tensor in enumerate(bundle.intent.outputs):
        key = f"out{index}"
        if key in aliases:
            value = input_tensors[aliases[key]]
        else:
            shape = tuple(
                int(extent) if isinstance(extent, int) else bundle.dims[extent]
                for extent in tensor.shape
            )
            value = torch.empty(
                shape, dtype=_torch_dtype(torch, tensor.dtype), device="cuda"
            )
            abi_tensors.append(value)
        outputs[dict(lowering_artifact.output_bindings)[key]] = value
    program = lowering_artifact.program
    if program.selection.pipeline_kind != "specialized":
        raise ValueError("performance requires an automatic specialized direct hit")
    if len(program.kernels) != 1:
        raise ValueError("single-launch benchmark requires exactly one cubin")
    kernel = program.kernels[0]
    launch_plan, argument_plan = _single_launch_plan(
        kernel.binary_mlir, kernel, abi_count=len(abi_tensors)
    )
    function = load_module(kernel.cubin).function(kernel.kernel_name)
    if launch_plan["dynamic_shared_memory_bytes"]:
        backend_orchestrate._set_dynamic_shared_memory(
            function, launch_plan["dynamic_shared_memory_bytes"]
        )
    launcher = backend_orchestrate.CachedLauncher(
        function,
        tuple(launch_plan["grid"]),
        tuple(launch_plan["block"]),
        argument_plan,
        shared=launch_plan["dynamic_shared_memory_bytes"],
    )

    def launch() -> None:
        launcher.launch(abi_tensors)

    return launch, outputs, launch_plan


def _allclose(
    generated: Mapping[str, Any],
    source: Mapping[str, Any],
    tolerance: Any,
    *,
    expected_outputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected = set(expected_outputs if expected_outputs is not None else source)
    source_names = set(source)
    generated_names = set(generated)
    if source_names != expected or generated_names != expected:
        raise ValueError(
            "output binding mismatch: "
            f"expected={sorted(expected)!r}, source={sorted(source_names)!r}, "
            f"generated={sorted(generated_names)!r}"
        )
    result: dict[str, Any] = {}
    for name in sorted(expected):
        if generated[name].shape != source[name].shape:
            result[name] = {
                "passed": False,
                "detail": "shape mismatch",
                "generated_shape": list(generated[name].shape),
                "source_shape": list(source[name].shape),
            }
            continue
        if generated[name].dtype != source[name].dtype:
            result[name] = {
                "passed": False,
                "detail": "dtype mismatch",
                "generated_dtype": str(generated[name].dtype),
                "source_dtype": str(source[name].dtype),
            }
            continue
        if generated[name].device != source[name].device:
            result[name] = {
                "passed": False,
                "detail": "device mismatch",
                "generated_device": str(generated[name].device),
                "source_device": str(source[name].device),
            }
            continue
        left = generated[name].detach().cpu().float().numpy()
        right = source[name].detach().cpu().float().numpy()
        passed = bool(
            np.allclose(
                left,
                right,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                equal_nan=tolerance.equal_nan,
            )
        )
        result[name] = {
            "passed": passed,
            "shape": list(left.shape),
            "max_abs_error": (
                float(np.nanmax(np.abs(left - right))) if left.size else 0.0
            ),
        }
    return result


def benchmark_report(
    report_path: Path,
    *,
    registry: Path,
    capture_manifest: Path,
    paired_samples: int,
    warmup_ms: int,
    rep_ms: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    import torch
    import triton

    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed, failures = batch._fresh_contract(report)
    if not passed:
        raise ValueError(f"report is not an official full pipeline pass: {failures}")
    kernel_id = str(report["campaign"]["kernel_id"])
    cases = {
        str(case.origin["kernel_id"]): case
        for case in batch.load_cases(registry, capture_manifest)
    }
    case = cases[kernel_id]
    artifact_payload = dict(report["gate1"]["artifact"])
    artifact_payload.pop("source_sha256", None)
    artifact_payload.pop("source_bytes", None)
    artifact_payload["source"] = case.source_text
    artifact = Gate1Artifact.from_json(artifact_payload)
    gate2_case = live._gate2_case(
        artifact,
        case,
        int(report["seed"]),
        lowering_backend="cuda",
    )
    lowering_artifact = CudaLoweringRunner().lower(gate2_case.lowering)
    source_launch, source_outputs = _source_launcher(case, gate2_case)
    generated_launch, generated_outputs, launch_plan = _generated_launcher(
        gate2_case.lowering, lowering_artifact, gate2_case.values
    )
    source_launch()
    generated_launch()
    torch.cuda.synchronize()
    correctness = _allclose(
        generated_outputs,
        source_outputs,
        case.tolerance,
        expected_outputs=gate2_case.output_keys,
    )
    if not all(item["passed"] for item in correctness.values()):
        raise RuntimeError(f"pre-benchmark correctness failed: {correctness}")

    def bench(launch: Callable[[], None]) -> float:
        return float(
            triton.testing.do_bench(
                launch, warmup=warmup_ms, rep=rep_ms, return_mode="median"
            )
            * 1000.0
        )

    probes = {"generated": bench(generated_launch), "source": bench(source_launch)}
    graph_mode = min(probes.values()) <= 10.0
    unroll = 100 if graph_mode else 1

    def capture(launch: Callable[[], None]) -> Any:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(unroll):
                launch()
        torch.cuda.synchronize()
        return graph

    generated_timed = generated_launch
    source_timed = source_launch
    if graph_mode:
        generated_timed = capture(generated_launch).replay
        source_timed = capture(source_launch).replay

    def measure(launch: Callable[[], None]) -> float:
        return bench(launch) / unroll

    generated_samples: list[float] = []
    source_samples: list[float] = []
    paired_order: list[str] = []
    paired_records: list[dict[str, Any]] = []
    for index in range(paired_samples):
        if index % 2 == 0:
            order = "source_then_generated"
            source = measure(source_timed)
            generated = measure(generated_timed)
        else:
            order = "generated_then_source"
            generated = measure(generated_timed)
            source = measure(source_timed)
        paired_order.append(order)
        generated_samples.append(generated)
        source_samples.append(source)
        paired_records.append(
            {
                "pair_id": index,
                "order": order,
                "generated_us": generated,
                "source_us": source,
            }
        )
    comparison = _compare(
        generated_samples,
        source_samples,
        resamples=bootstrap_resamples,
        seed=20260726,
    )
    result = {
        "schema": "intentir.high-performance.live-gate-cuda-performance.v1",
        "kernel_id": kernel_id,
        "status": "measured",
        "model": batch.OFFICIAL_MODEL,
        "effort": batch.OFFICIAL_EFFORT,
        "gate1_status": report["gate1"]["status"],
        "gate2_status": report["gate2"]["status"],
        "selection": lowering_artifact.program.selection.to_dict(),
        "schedule_decision": (
            lowering_artifact.schedule_decision.to_dict()
            if lowering_artifact.schedule_decision is not None
            else None
        ),
        "launch_plan": launch_plan,
        "correctness_before_timing": correctness,
        "correctness_pass": True,
        "methodology": {
            "timing_domain": (
                "equal_unroll_cuda_graph_replay_amortized_per_launch"
                if graph_mode
                else "raw_kernel_only_triton_do_bench"
            ),
            "direct_probe_us": probes,
            "graph_replay_unroll": unroll,
            "paired_samples": paired_samples,
            "paired_order": paired_order,
            "paired_records": paired_records,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "bootstrap_resamples": bootstrap_resamples,
            "compilation_excluded_from_samples": True,
            "inputs_and_outputs_preallocated": True,
        },
        "generated": _summarize(
            generated_samples, resamples=bootstrap_resamples, seed=20260724
        ),
        "source": _summarize(
            source_samples, resamples=bootstrap_resamples, seed=20260725
        ),
        "comparison": comparison,
        "paper_suite_target": {
            "metric": "geomean_speedup",
            "target": 1.35,
            "scope": "28-kernel-suite",
            "per_kernel_applicable": False,
        },
    }
    report["gpu_performance"] = result
    batch._atomic_json(report_path, report)
    batch._atomic_json(report_path.with_suffix(".performance.json"), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=batch.DEFAULT_REGISTRY)
    parser.add_argument(
        "--capture-manifest", type=Path, default=batch.DEFAULT_CAPTURE_MANIFEST
    )
    parser.add_argument("--paired-samples", type=int, default=50)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=50_000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.paired_samples < 10:
        raise SystemExit("paired-samples must be at least 10")
    if args.warmup_ms < 0:
        raise SystemExit("warmup-ms must be non-negative")
    if args.rep_ms <= 0:
        raise SystemExit("rep-ms must be positive")
    if args.bootstrap_resamples < 100:
        raise SystemExit("bootstrap-resamples must be at least 100")
    result = benchmark_report(
        args.report,
        registry=args.registry,
        capture_manifest=args.capture_manifest,
        paired_samples=args.paired_samples,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
