#!/usr/bin/env python3
"""Execute one anonymous J-only CUDA runtime case against the RefJ oracle.

The process sees no Source, pi, Replay, frontend label, case id, Gate result, or
historical winner.  It compiles from the exact production J+C snapshot before
loading Q values, executes the resulting cubins, compares target outputs to
RefJ, and optionally measures the already-prepared whole launch graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

from intentir.backend.cuda.benchmark import PreparedGraphLaunch, do_bench_us
from intentir.backend.cuda.pipeline import lower_to_cuda
from intentir.backend.materialize import materialize_j
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.orchestrate import gate2_input_digest
from intentir.verify.gate2.policy import ComparisonPolicy
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


INPUT_PROTOCOL = "intentir.experiment.anonymous-j-runtime-input.v1"
RESULT_PROTOCOL = "intentir.experiment.anonymous-j-runtime-result.v1"
ROOT_FIELDS = frozenset({"protocol", "entry_count", "entries", "input_digest"})
ENTRY_FIELDS = frozenset(
    {
        "run_id",
        "snapshot",
        "concrete_extents",
        "input_file",
        "input_file_sha256",
        "input_digest",
        "comparison_policy",
    }
)
FORBIDDEN_FIELDS = frozenset(
    {
        "source",
        "pi",
        "replay",
        "frontend",
        "case",
        "case_id",
        "gate1",
        "gate2",
        "winner",
        "history",
        "stem",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_entry(manifest_path: Path, run_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != ROOT_FIELDS:
        raise ValueError("runtime input root does not match the exact whitelist")
    if manifest["protocol"] != INPUT_PROTOCOL:
        raise ValueError("unsupported runtime input protocol")
    entries = manifest["entries"]
    if not isinstance(entries, list) or manifest["entry_count"] != len(entries):
        raise ValueError("runtime input entry_count mismatch")
    content = {
        "protocol": manifest["protocol"],
        "entry_count": manifest["entry_count"],
        "entries": entries,
    }
    if manifest["input_digest"] != _digest(content):
        raise ValueError("runtime input manifest digest mismatch")
    selected = None
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or set(raw) != ENTRY_FIELDS:
            raise ValueError(f"runtime input entry {index} violates the exact whitelist")
        if set(raw) & FORBIDDEN_FIELDS:
            raise ValueError(f"runtime input entry {index} contains forbidden identity data")
        current = raw["run_id"]
        if not isinstance(current, str) or _SHA256.fullmatch(current) is None:
            raise ValueError(f"runtime input entry {index} has invalid run_id")
        if current in seen:
            raise ValueError("runtime input run_id must be unique")
        seen.add(current)
        if current == run_id:
            selected = raw
    if selected is None:
        raise ValueError(f"anonymous runtime run_id {run_id!r} is absent")
    return selected


def _load_values(manifest_path: Path, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    filename = entry["input_file"]
    if filename != f"{entry['run_id']}.npz" or Path(filename).name != filename:
        raise ValueError("runtime input file is not content-addressed by run_id")
    path = manifest_path.parent / "runtime_values_current" / filename
    if _file_digest(path) != entry["input_file_sha256"]:
        raise ValueError("runtime input file SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if gate2_input_digest(values) != entry["input_digest"]:
        raise ValueError("logical runtime input digest mismatch")
    return values


_TORCH_DTYPES = {
    "f16": torch.float16,
    "float16": torch.float16,
    "f32": torch.float32,
    "float32": torch.float32,
    "f64": torch.float64,
    "float64": torch.float64,
    "i1": torch.bool,
    "bool": torch.bool,
    "i8": torch.int8,
    "int8": torch.int8,
    "i16": torch.int16,
    "int16": torch.int16,
    "i32": torch.int32,
    "int32": torch.int32,
    "i64": torch.int64,
    "int64": torch.int64,
}


def _torch_input(value: np.ndarray, dtype: str) -> torch.Tensor:
    key = dtype.removeprefix("torch.").lower()
    if key not in _TORCH_DTYPES:
        raise ValueError(f"runtime CUDA tensor dtype {dtype!r} is unsupported")
    return torch.as_tensor(value, device="cuda", dtype=_TORCH_DTYPES[key]).contiguous()


def _shape_records(values: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        result[name] = {"shape": list(array.shape), "dtype": array.dtype.name}
    return result


def _failure(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "fail",
        "stage": stage,
        "error_type": type(error).__name__,
        "diagnostic": str(error),
    }


def _pass(stage: str, **fields: Any) -> dict[str, Any]:
    return {"status": "pass", "stage": stage, **fields}


def _run(
    manifest_path: Path,
    entry: Mapping[str, Any],
    *,
    benchmark: bool,
    warmup_ms: int,
    repeat_ms: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    result: dict[str, Any] = {
        "protocol": RESULT_PROTOCOL,
        "run_id": entry["run_id"],
        "stages": {},
    }
    try:
        snapshot = ProductionSnapshot.from_dict(entry["snapshot"])
        extents = {str(k): int(v) for k, v in entry["concrete_extents"].items()}
        policy = ComparisonPolicy.from_json(entry["comparison_policy"])
        result["stages"]["decode"] = _pass(
            "decode",
            snapshot_digest=snapshot.snapshot_digest,
            j_digest=snapshot.j_digest,
            c_digest=snapshot.c_digest,
            policy_id=policy.policy_id,
        )
    except Exception as error:
        result["stages"]["decode"] = _failure("decode", error)
        return result

    # Compile before Q values are loaded.  Runtime values cannot influence J
    # materialization, variant selection, cubin generation, or kernel topology.
    compile_started = time.perf_counter_ns()
    try:
        materialized = materialize_j(snapshot.j, extents)
        program = lower_to_cuda(materialized.intent_mlir, allow_specialized=False)
        binaries = [
            {
                "kernel_name": kernel.kernel_name,
                "binary_name": kernel.binary_name,
                "cubin_size_bytes": len(kernel.cubin),
                "cubin_sha256": hashlib.sha256(kernel.cubin).hexdigest(),
                "cubin_is_elf": kernel.cubin.startswith(b"\x7fELF"),
            }
            for kernel in program.kernels
        ]
        if not binaries or not all(item["cubin_is_elf"] for item in binaries):
            raise RuntimeError("runtime compilation did not produce non-empty ELF cubins")
        result["stages"]["compile"] = _pass(
            "compile",
            elapsed_ms=(time.perf_counter_ns() - compile_started) / 1_000_000.0,
            intent_mlir_sha256=hashlib.sha256(
                materialized.intent_mlir.encode("utf-8")
            ).hexdigest(),
            intent_mlir_bytes=len(materialized.intent_mlir.encode("utf-8")),
            materialization_form=materialized.form,
            kernel_count=len(program.kernels),
            total_cubin_size_bytes=sum(item["cubin_size_bytes"] for item in binaries),
            binaries=binaries,
            q_values_loaded_before_compile=False,
        )
    except Exception as error:
        result["stages"]["compile"] = _failure("compile", error)
        return result

    try:
        values = _load_values(manifest_path, entry)
        input_names = tuple(item.name for item in snapshot.j.intent.inputs)
        if set(values) != set(input_names):
            raise ValueError("runtime Q values do not match canonical J inputs")
        input_bindings = {name: name for name in input_names}
        output_bindings = {
            item.name: f"out{index}"
            for index, item in enumerate(snapshot.j.intent.outputs)
        }
        spec = ScalarLoweringSpec.create(
            snapshot.j,
            dims=extents,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
        )
        result["stages"]["input"] = _pass(
            "input",
            input_digest=entry["input_digest"],
            input_count=len(values),
            inputs=_shape_records(values),
        )
    except Exception as error:
        result["stages"]["input"] = _failure("input", error)
        return result

    ref_started = time.perf_counter_ns()
    reference = ScalarLoweringRunner().run(spec, values)
    if not reference.ok:
        result["stages"]["refj"] = {
            "status": "fail",
            "stage": "refj",
            "unsupported": reference.unsupported,
            "diagnostic": reference.error,
        }
        return result
    result["stages"]["refj"] = _pass(
        "refj",
        elapsed_ms=(time.perf_counter_ns() - ref_started) / 1_000_000.0,
        unsupported=False,
        outputs=_shape_records(reference.outputs),
    )

    execute_started = time.perf_counter_ns()
    try:
        tensors_by_name = {
            tensor.name: _torch_input(values[tensor.name], tensor.dtype)
            for tensor in snapshot.j.intent.inputs
        }
        ordered_inputs = [
            tensors_by_name[item.name] for item in snapshot.j.intent.inputs
        ]
        prepared = PreparedGraphLaunch(
            program,
            materialized.intent_mlir,
            ordered_inputs,
        )
        prepared.thunk()()
        torch.cuda.synchronize()
        target_outputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in prepared.outputs.items()
        }
        agreement = compare_outputs(
            "cuda_target",
            target_outputs,
            "ref_j",
            reference.outputs,
            j=snapshot.j,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
            inputs=values,
            policy=policy,
        )
        result["stages"]["execute_compare"] = {
            "status": "pass" if agreement.passed else "fail",
            "stage": "execute_compare",
            "elapsed_ms": (
                time.perf_counter_ns() - execute_started
            ) / 1_000_000.0,
            "graph": prepared.record.to_dict(),
            "outputs": _shape_records(target_outputs),
            "agreement": agreement.to_json(),
        }
        if not agreement.passed:
            return result
    except Exception as error:
        result["stages"]["execute_compare"] = _failure("execute_compare", error)
        return result

    if benchmark:
        try:
            median_us = do_bench_us(
                prepared.thunk(),
                warmup_ms=warmup_ms,
                repeat_ms=repeat_ms,
            )
            result["stages"]["benchmark"] = _pass(
                "benchmark",
                method="triton.testing.do_bench event timing",
                warmup_ms=warmup_ms,
                repeat_ms=repeat_ms,
                median_us=median_us,
                prepared_no_allocation_no_sync=True,
                source_baseline_available=False,
            )
        except Exception as error:
            result["stages"]["benchmark"] = _failure("benchmark", error)
    else:
        result["stages"]["benchmark"] = {
            "status": "skipped",
            "stage": "benchmark",
        }
    result["total_elapsed_ms"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000.0
    result["machine"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_capability": list(torch.cuda.get_device_capability()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--repeat-ms", type=int, default=30)
    args = parser.parse_args()
    if args.warmup_ms <= 0 or args.repeat_ms <= 0:
        raise ValueError("benchmark windows must be positive")
    entry = _validated_entry(args.manifest, args.run_id)
    result = _run(
        args.manifest,
        entry,
        benchmark=not args.no_benchmark,
        warmup_ms=args.warmup_ms,
        repeat_ms=args.repeat_ms,
    )
    result["result_digest"] = _digest(result)
    _write(args.output, result)
    passed = (
        result.get("stages", {}).get("execute_compare", {}).get("status") == "pass"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "runtime_pass": passed,
                "stages": {
                    name: value.get("status")
                    for name, value in result.get("stages", {}).items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
