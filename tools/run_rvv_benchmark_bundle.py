#!/usr/bin/env python3
"""Link, execute, validate, and search a native RVV object bundle.

This target-side runner intentionally uses only the Python standard library.
The target computation lives in precompiled objects produced by the MLIR/LLVM
pipeline; Python owns descriptor binding, correctness audit, and timing only.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v1"
REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v1"


class KernelFailure(RuntimeError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _decode_payload(record: Mapping[str, Any]) -> bytes:
    payload = base64.b64decode(record["data_base64"], validate=True)
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("portable array digest mismatch")
    return payload


def _numel(shape: list[int]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _dtype_size(dtype: str) -> int:
    key = dtype.removeprefix("torch.").lower()
    return {
        "bool": 1,
        "i1": 1,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "f64": 8,
    }[key]


def _encoded_values(record: Mapping[str, Any]) -> list[Any]:
    payload = _decode_payload(record)
    dtype = str(record["dtype"])
    count = _numel([int(item) for item in record["shape"]])
    normalized = dtype.lstrip("<>=|")
    formats = {
        "b1": "?",
        "i1": "b",
        "u1": "B",
        "i2": "h",
        "u2": "H",
        "i4": "i",
        "u4": "I",
        "i8": "q",
        "u8": "Q",
        "f2": "e",
        "f4": "f",
        "f8": "d",
    }
    if normalized not in formats:
        raise ValueError(f"unsupported portable dtype {dtype!r}")
    fmt = "<" + formats[normalized]
    values = [item[0] for item in struct.iter_unpack(fmt, payload)]
    if len(values) != count:
        raise ValueError("portable array size differs from shape")
    return values


def _output_values(payload: bytes, dtype: str) -> list[Any]:
    key = dtype.removeprefix("torch.").lower()
    formats = {
        "bool": "?",
        "i1": "?",
        "i8": "b",
        "i16": "h",
        "i32": "i",
        "i64": "q",
        "f16": "e",
        "f32": "f",
        "f64": "d",
    }
    if key == "bf16":
        result = []
        for (bits,) in struct.iter_unpack("<H", payload):
            result.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
        return result
    if key not in formats:
        raise ValueError(f"unsupported output dtype {dtype!r}")
    return [item[0] for item in struct.iter_unpack("<" + formats[key], payload)]


@lru_cache(maxsize=None)
def _descriptor_type(rank: int) -> type[ctypes.Structure]:
    fields: list[tuple[str, Any]] = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank:
        fields.extend(
            [
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]
        )
    return type(
        f"MemRefDescriptor{rank}",
        (ctypes.Structure,),
        {"_fields_": fields},
    )


def _descriptor(buffer: Any, shape: list[int]) -> ctypes.Structure:
    descriptor_type = _descriptor_type(len(shape))
    result = descriptor_type()
    address = ctypes.addressof(buffer)
    result.allocated = address
    result.aligned = address
    result.offset = 0
    if shape:
        stride = 1
        strides = [0] * len(shape)
        for index in range(len(shape) - 1, -1, -1):
            strides[index] = stride
            stride *= int(shape[index])
        for index, extent in enumerate(shape):
            result.sizes[index] = int(extent)
            result.strides[index] = int(strides[index])
    return result


def _close(
    actual: list[Any],
    expected: list[Any],
    rule: Mapping[str, Any],
) -> tuple[bool, float]:
    if len(actual) != len(expected):
        return False, float("inf")
    tolerance = rule.get("tolerance")
    if tolerance is None or not rule["dtype"].startswith(("f", "bf")):
        return actual == expected, 0.0 if actual == expected else float("inf")
    max_error = 0.0
    for left, right in zip(actual, expected, strict=True):
        left = float(left)
        right = float(right)
        if math.isnan(left) or math.isnan(right):
            if not (
                math.isnan(left)
                and math.isnan(right)
                and tolerance["equal_nan"]
            ):
                return False, float("inf")
            continue
        if math.isinf(left) or math.isinf(right):
            if left != right:
                return False, float("inf")
            continue
        error = abs(left - right)
        max_error = max(max_error, error)
        bound = float(tolerance["atol"]) + float(tolerance["rtol"]) * max(
            abs(left), abs(right)
        )
        if error > bound:
            return False, max_error
    return True, max_error


def _link_object(directory: Path, candidate: Mapping[str, Any]) -> Path:
    candidate_id = candidate["candidate"]["candidate_id"]
    object_path = directory / f"{candidate_id}.o"
    library_path = directory / f"{candidate_id}.so"
    object_bytes = base64.b64decode(candidate["object_base64"], validate=True)
    expected = candidate["candidate"]["object_sha256"]
    if hashlib.sha256(object_bytes).hexdigest() != expected:
        raise ValueError("RVV object digest mismatch")
    object_path.write_bytes(object_bytes)
    process = subprocess.run(
        [
            "clang",
            "-shared",
            "-march=rv64gcv_zfh",
            str(object_path),
            "-lm",
            "-o",
            str(library_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"target link failed: {process.stderr}")
    return library_path


def _measure(
    function: Any,
    arguments: list[Any],
    *,
    warmup: int,
    iterations: int,
    batches: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function(*arguments)
    samples = []
    for _ in range(batches):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function(*arguments)
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / iterations / 1_000.0)
    return float(statistics.median(samples)), samples


def _one_kernel(
    entry: Mapping[str, Any],
    *,
    warmup: int,
    iterations: int,
    batches: int,
) -> dict[str, Any]:
    input_buffers = []
    descriptors = []
    for name in entry["input_order"]:
        value_record = entry["values"][entry["input_bindings"][name]]
        payload = _decode_payload(value_record)
        buffer = ctypes.create_string_buffer(payload, len(payload))
        input_buffers.append(buffer)
        descriptors.append(
            _descriptor(buffer, [int(item) for item in value_record["shape"]])
        )
    output_buffers = []
    for spec in entry["output_specs"]:
        size = _numel(spec["shape"]) * _dtype_size(spec["dtype"])
        buffer = ctypes.create_string_buffer(size)
        output_buffers.append(buffer)
        descriptors.append(_descriptor(buffer, list(spec["shape"])))
    arguments = [ctypes.byref(item) for item in descriptors]

    records = []
    with tempfile.TemporaryDirectory(prefix="intentir-rvv-") as temporary:
        directory = Path(temporary)
        for candidate in entry["candidates"]:
            try:
                library_path = _link_object(directory, candidate)
                library = ctypes.CDLL(str(library_path))
                symbol = candidate["candidate"]["c_interface_symbol"]
                function = getattr(library, symbol)
                function.argtypes = [
                    ctypes.POINTER(type(item)) for item in descriptors
                ]
                function.restype = None
                function(*arguments)
                correctness = []
                passed = True
                for spec, buffer in zip(
                    entry["output_specs"], output_buffers, strict=True
                ):
                    output_name = spec["name"]
                    actual = _output_values(bytes(buffer.raw), spec["dtype"])
                    reference_record = entry["reference_outputs"][
                        entry["output_bindings"][output_name]
                    ]
                    expected = _encoded_values(reference_record)
                    agrees, max_error = _close(
                        actual,
                        expected,
                        entry["output_rules"][output_name],
                    )
                    correctness.append(
                        {
                            "output": output_name,
                            "passed": agrees,
                            "max_abs_error": max_error,
                        }
                    )
                    passed = passed and agrees
                if not passed:
                    records.append(
                        {
                            "candidate": candidate["candidate"],
                            "status": "incorrect",
                            "correctness": correctness,
                        }
                    )
                    continue
                latency, samples = _measure(
                    function,
                    arguments,
                    warmup=warmup,
                    iterations=iterations,
                    batches=batches,
                )
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "measured",
                        "latency_us": latency,
                        "samples_us": samples,
                        "correctness": correctness,
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    measured = [item for item in records if item["status"] == "measured"]
    if not measured:
        raise KernelFailure("no correct RVV candidate", records)
    schedule_only_ids = set(entry["schedule_only_candidate_ids"])
    e2 = min(
        (
            item
            for item in measured
            if item["candidate"]["candidate_id"] in schedule_only_ids
        ),
        key=lambda item: item["latency_us"],
    )
    e3 = min(measured, key=lambda item: item["latency_us"])
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
        "snapshot_digest": entry["snapshot_digest"],
        "j_digest": entry["j_digest"],
        "c_digest": entry["c_digest"],
        "contract_view": entry["contract_view"],
        "extents": entry["extents"],
        "candidates": records,
        "winner": e3["candidate"],
        "performance": {
            "e2_schedule_only_us": e2["latency_us"],
            "e3_contract_guided_us": e3["latency_us"],
            "e3_over_e2_speedup": e2["latency_us"] / e3["latency_us"],
            "b_transformation_wins": bool(
                e3["candidate"]["schedule"]["b_transformations"]
            ),
            "timing": "pinned-process perf_counter_ns via ctypes C ABI",
            "warmup": warmup,
            "iterations": iterations,
            "batches": batches,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(args.warmup, args.iterations, args.batches) <= 0:
        raise ValueError("warmup, iterations, and batches must be positive")
    os.sched_setaffinity(0, {args.cpu})
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported RVV benchmark bundle")
    selected = set(args.kernels or [])
    entries = [
        item
        for item in bundle["kernels"]
        if not selected or item["kernel"] in selected
    ]
    started = time.time_ns()
    kernels = []
    failures = []
    for entry in entries:
        try:
            record = _one_kernel(
                entry,
                warmup=args.warmup,
                iterations=args.iterations,
                batches=args.batches,
            )
            kernels.append(record)
            performance = record["performance"]
            print(
                f"{entry['kernel']}: E2={performance['e2_schedule_only_us']:.4f} "
                f"us E3={performance['e3_contract_guided_us']:.4f} us "
                f"speedup={performance['e3_over_e2_speedup']:.3f}x",
                flush=True,
            )
        except Exception as error:
            failure = {
                "kernel": entry["kernel"],
                "error": f"{type(error).__name__}: {error}",
            }
            if isinstance(error, KernelFailure):
                failure["candidate_diagnostics"] = error.diagnostics
            failures.append(failure)
            print(
                f"{entry['kernel']}: FAILED: {type(error).__name__}: {error}",
                flush=True,
            )
    speedups = [item["performance"]["e3_over_e2_speedup"] for item in kernels]
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": {
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "vlen_bits": 128,
        },
        "summary": {
            "requested": len(entries),
            "successful": len(kernels),
            "failed": len(failures),
            "geomean_e3_over_e2_speedup": (
                math.exp(sum(math.log(item) for item in speedups) / len(speedups))
                if speedups
                else None
            ),
            "b_transformation_winner_count": sum(
                item["performance"]["b_transformation_wins"]
                for item in kernels
            ),
        },
        "kernels": kernels,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
