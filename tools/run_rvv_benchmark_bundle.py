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
import shutil
import statistics
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v3"
REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v5"
NATIVE_COMPARATOR_PROTOCOL = "intentir.backend.native-output-comparator.v1"

_NATIVE_COMPARATOR_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>

int intentir_equal_bytes(const void *left, const void *right, uint64_t bytes) {
  return memcmp(left, right, (size_t)bytes) == 0;
}

int intentir_close_f32(const float *left, const float *right, uint64_t count,
                       double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = (double)left[index];
    const double b = (double)right[index];
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

int intentir_close_f64(const double *left, const double *right, uint64_t count,
                       double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = left[index];
    const double b = right[index];
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

static float intentir_half_to_float(uint16_t value) {
  const uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
  uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
  uint32_t mantissa = (uint32_t)value & 0x03ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static float intentir_bf16_to_float(uint16_t value) {
  const uint32_t bits = (uint32_t)value << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static int intentir_close_16(const uint16_t *left, const uint16_t *right,
                             uint64_t count, double atol, double rtol,
                             int equal_nan, double *max_abs_error,
                             int bfloat) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = bfloat ? intentir_bf16_to_float(left[index])
                            : intentir_half_to_float(left[index]);
    const double b = bfloat ? intentir_bf16_to_float(right[index])
                            : intentir_half_to_float(right[index]);
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

int intentir_close_f16(const uint16_t *left, const uint16_t *right,
                       uint64_t count, double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  return intentir_close_16(left, right, count, atol, rtol, equal_nan,
                           max_abs_error, 0);
}

int intentir_close_bf16(const uint16_t *left, const uint16_t *right,
                        uint64_t count, double atol, double rtol, int equal_nan,
                        double *max_abs_error) {
  return intentir_close_16(left, right, count, atol, rtol, equal_nan,
                           max_abs_error, 1);
}
"""


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


def _cache_size_bytes(text: str) -> int:
    value = text.strip().upper()
    multiplier = 1
    if value.endswith("K"):
        multiplier = 1024
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1024 * 1024
        value = value[:-1]
    return int(value) * multiplier


def _cpu_cache_record() -> dict[str, Any]:
    caches = []
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    for index in sorted(root.glob("index*")):
        try:
            cache_type = (index / "type").read_text().strip()
            level = int((index / "level").read_text().strip())
            size_bytes = _cache_size_bytes((index / "size").read_text())
            line_bytes = int((index / "coherency_line_size").read_text().strip())
        except (OSError, ValueError):
            continue
        caches.append(
            {
                "level": level,
                "type": cache_type,
                "size_bytes": size_bytes,
                "coherency_line_bytes": line_bytes,
            }
        )
    data_caches = [
        item for item in caches if item["type"] in {"Data", "Unified"}
    ]
    last_level_bytes = max(
        (item["size_bytes"] for item in data_caches), default=0
    )
    return {"levels": caches, "last_level_data_cache_bytes": last_level_bytes}


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


def _descriptor_from_address(address: int, shape: list[int]) -> ctypes.Structure:
    descriptor_type = _descriptor_type(len(shape))
    result = descriptor_type()
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


def _descriptor(buffer: Any, shape: list[int]) -> ctypes.Structure:
    return _descriptor_from_address(ctypes.addressof(buffer), shape)


class _NativeComparator:
    def __init__(self, directory: Path):
        source = directory / "intentir_native_compare.c"
        library_path = directory / "intentir_native_compare.so"
        source.write_text(_NATIVE_COMPARATOR_SOURCE, encoding="utf-8")
        compiler = shutil.which("clang") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("native comparator requires target clang or cc")
        process = subprocess.run(
            [
                compiler,
                "-shared",
                "-O3",
                "-fPIC",
                str(source),
                "-lm",
                "-o",
                str(library_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"native comparator compile failed: {process.stderr}")
        self.library = ctypes.CDLL(str(library_path))
        self.equal_bytes = self.library.intentir_equal_bytes
        self.equal_bytes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.equal_bytes.restype = ctypes.c_int
        self.close_functions = {}
        for dtype in ("f16", "bf16", "f32", "f64"):
            function = getattr(self.library, f"intentir_close_{dtype}")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
            ]
            function.restype = ctypes.c_int
            self.close_functions[dtype] = function
        self.receipt = {
            "protocol": NATIVE_COMPARATOR_PROTOCOL,
            "source_sha256": hashlib.sha256(
                _NATIVE_COMPARATOR_SOURCE.encode("utf-8")
            ).hexdigest(),
            "compiler": f"{Path(compiler).name} -O3 -shared -fPIC",
            "compares_all_output_elements": True,
            "python_elementwise_comparison": False,
        }

    def close(
        self,
        actual: Any,
        expected: Any,
        *,
        byte_count: int,
        element_count: int,
        dtype: str,
        rule: Mapping[str, Any],
    ) -> tuple[bool, float]:
        actual_pointer = ctypes.c_void_p(ctypes.addressof(actual))
        expected_pointer = ctypes.c_void_p(ctypes.addressof(expected))
        key = dtype.removeprefix("torch.").lower()
        tolerance = rule.get("tolerance")
        if tolerance is None or key not in self.close_functions:
            equal = bool(
                self.equal_bytes(actual_pointer, expected_pointer, byte_count)
            )
            return equal, 0.0 if equal else float("inf")
        maximum = ctypes.c_double(0.0)
        passed = bool(
            self.close_functions[key](
                actual_pointer,
                expected_pointer,
                element_count,
                float(tolerance["atol"]),
                float(tolerance["rtol"]),
                int(bool(tolerance["equal_nan"])),
                ctypes.byref(maximum),
            )
        )
        return passed, float(maximum.value)


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


class _ArgumentRotation:
    def __init__(self, argument_sets: list[list[Any]]):
        if not argument_sets:
            raise ValueError("argument rotation requires at least one replica")
        self.argument_sets = argument_sets
        self.cursor = 0

    def next(self) -> list[Any]:
        result = self.argument_sets[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.argument_sets)
        return result


def _logical_io_bytes(entry: Mapping[str, Any]) -> dict[str, int]:
    input_bytes = sum(
        len(_decode_payload(entry["values"][entry["input_bindings"][name]]))
        for name in entry["input_order"]
    )
    output_bytes = sum(
        _numel(spec["shape"]) * _dtype_size(spec["dtype"])
        for spec in entry["output_specs"]
    )
    return {
        "boundary_input_bytes": input_bytes,
        "boundary_output_bytes": output_bytes,
        "boundary_io_bytes": input_bytes + output_bytes,
    }


def _argument_replicas(
    entry: Mapping[str, Any], replicas: int
) -> tuple[list[list[Any]], list[Any], list[list[Any]]]:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    all_buffers: list[Any] = []
    output_buffers: list[list[Any]] = []
    argument_sets: list[list[Any]] = []
    for _ in range(replicas):
        descriptors = []
        replica_outputs = []
        for name in entry["input_order"]:
            value_record = entry["values"][entry["input_bindings"][name]]
            payload = _decode_payload(value_record)
            buffer = ctypes.create_string_buffer(payload, len(payload))
            all_buffers.append(buffer)
            descriptors.append(
                _descriptor(buffer, [int(item) for item in value_record["shape"]])
            )
        for spec in entry["output_specs"]:
            size = _numel(spec["shape"]) * _dtype_size(spec["dtype"])
            buffer = ctypes.create_string_buffer(size)
            all_buffers.append(buffer)
            replica_outputs.append(buffer)
            descriptors.append(_descriptor(buffer, list(spec["shape"])))
        output_buffers.append(replica_outputs)
        argument_sets.append([ctypes.byref(item) for item in descriptors])
        # byref objects keep descriptors alive, and all_buffers keeps payloads
        # alive.  Retain the descriptor list explicitly for PyPy/CPython parity.
        all_buffers.extend(descriptors)
    return argument_sets, all_buffers, output_buffers


def _evict(eviction_buffer: Any, fill: int) -> float:
    started = time.perf_counter_ns()
    ctypes.memset(
        ctypes.addressof(eviction_buffer),
        fill & 0xFF,
        ctypes.sizeof(eviction_buffer),
    )
    return (time.perf_counter_ns() - started) / 1_000.0


def _measure(
    function: Any,
    rotation: _ArgumentRotation,
    *,
    regime: str,
    eviction_buffer: Any | None,
    warmup: int,
    iterations: int,
    batches: int,
) -> tuple[float, list[float], float]:
    for _ in range(warmup):
        if regime == "cold-data":
            assert eviction_buffer is not None
            _evict(eviction_buffer, rotation.cursor)
        function(*rotation.next())
    samples = []
    prep_samples = []
    for _ in range(batches):
        if regime == "cold-data":
            call_samples = []
            for iteration in range(iterations):
                assert eviction_buffer is not None
                prep_samples.append(_evict(eviction_buffer, iteration))
                started = time.perf_counter_ns()
                function(*rotation.next())
                call_samples.append((time.perf_counter_ns() - started) / 1_000.0)
            samples.append(float(statistics.median(call_samples)))
            continue
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function(*rotation.next())
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / iterations / 1_000.0)
    prep_us = float(statistics.median(prep_samples)) if prep_samples else 0.0
    return float(statistics.median(samples)), samples, prep_us


def _bounded_candidate_orders(
    entry: Mapping[str, Any], search_budget: int, *, measure_c_oracle: bool = False
) -> dict[str, list[str]]:
    """Select the exact bounded timing union before any candidate executes.

    C-Fixed owns exactly one target-projected state. Schedule-Only and IntentIR
    receive the same caller-supplied total budget; the IntentIR order starts with
    the complete fixed-structure schedule spine. The independent C table compares
    C/Erase-C prefixes over the same IntentIR universe. Only this exact union is
    executed unless ``measure_c_oracle`` requests the complete latency map.
    """

    if search_budget <= 0:
        raise ValueError("RVV search budget must be positive")
    candidate_by_id = {
        item["candidate"]["candidate_id"]: item for item in entry["candidates"]
    }
    object_owner: dict[str, str] = {}
    for candidate_id, item in candidate_by_id.items():
        object_sha256 = item["candidate"].get("object_sha256")
        if not isinstance(object_sha256, str) or len(object_sha256) != 64:
            raise ValueError(
                f"RVV candidate {candidate_id!r} has no canonical object SHA-256"
            )
        previous = object_owner.setdefault(object_sha256, candidate_id)
        if previous != candidate_id:
            raise ValueError(
                "RVV bundle must deduplicate native objects before budgeting: "
                f"{candidate_id!r} duplicates {previous!r}"
            )
    available = set(candidate_by_id)
    plan = entry.get("concrete_search_plan", {})

    def ordered(raw: Any) -> list[str]:
        result = []
        seen = set()
        for candidate_id in raw:
            if candidate_id in available and candidate_id not in seen:
                result.append(candidate_id)
                seen.add(candidate_id)
        return result

    c_order = ordered(plan.get("c_prior_order", tuple(candidate_by_id)))
    erase_order = ordered(plan.get("erase_c_order", tuple(candidate_by_id)))
    c_schedule_order = ordered(plan.get("schedule_only_c_prior_order", ()))
    erase_schedule_order = ordered(plan.get("schedule_only_erase_c_order", ()))
    if not c_schedule_order or not erase_schedule_order:
        raise ValueError("RVV bundle lacks explicit Schedule-Only search orders")
    actual_universe_digest = hashlib.sha256(
        json.dumps(sorted(available), separators=(",", ":")).encode()
    ).hexdigest()
    if plan.get("candidate_universe_digest") != actual_universe_digest:
        raise ValueError("RVV concrete candidate-universe digest drifted")
    if set(c_order) != available or set(erase_order) != available:
        raise ValueError("RVV C/Erase-C orders do not cover the same full universe")
    if set(c_schedule_order) != set(erase_schedule_order):
        raise ValueError("RVV C changed the concrete Schedule-Only universe")
    canonical_path = plan.get("canonical_lowering_path")
    if not isinstance(canonical_path, str) or not canonical_path:
        raise ValueError("RVV bundle lacks canonical lowering-path identity")
    for candidate_id in c_schedule_order:
        actual_path = candidate_by_id[candidate_id]["candidate"][
            "transformation"
        ].get("lowering_path_kind")
        if actual_path != canonical_path:
            raise ValueError("RVV Schedule-Only escaped the canonical lowering path")
    if c_order[: len(c_schedule_order)] != c_schedule_order:
        raise ValueError("RVV IntentIR C order does not start with the E2 spine")
    if erase_order[: len(erase_schedule_order)] != erase_schedule_order:
        raise ValueError("RVV IntentIR Erase-C order does not start with the E2 spine")
    active_e2 = c_schedule_order[:search_budget]
    active_e3 = c_order[:search_budget]
    erase_e3 = erase_order[:search_budget]
    c_fixed_id = plan.get("c_fixed_candidate_id")
    if c_fixed_id not in available:
        raise ValueError("RVV bundle lacks the concrete C-Fixed candidate")
    no_retile = [
        candidate_id
        for candidate_id in c_schedule_order
        if int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "parallel_tile", 0
        ))
        == 0
        and int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "reduction_tile", 0
        ))
        == 0
    ]
    def unique(items: Any) -> list[str]:
        result = []
        seen = set()
        for candidate_id in items:
            if candidate_id in available and candidate_id not in seen:
                result.append(candidate_id)
                seen.add(candidate_id)
        return result

    e2_evaluation = unique(active_e2)
    e3_evaluation = unique(active_e3)
    if not set(e2_evaluation).issubset(set(c_order)):
        raise AssertionError("RVV E2 candidates escaped the IntentIR universe")
    if search_budget >= len(c_schedule_order) and not set(
        c_schedule_order
    ).issubset(set(e3_evaluation)):
        raise AssertionError("RVV IntentIR budget omitted the complete E2 spine")
    # This arm isolates schedule retile: do not mix B/structural states into it.
    schedule_only_retiled = [
        candidate_id
        for candidate_id in e2_evaluation
        if int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "parallel_tile", 0
        ))
        != 0
        or int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "reduction_tile", 0
        ))
        != 0
    ]

    timed = []
    seen = set()
    timed_source = (
        tuple(candidate_by_id)
        if measure_c_oracle
        else (c_fixed_id, *e2_evaluation, *e3_evaluation, *erase_e3)
    )
    for candidate_id in timed_source:
        if candidate_id not in seen:
            timed.append(candidate_id)
            seen.add(candidate_id)
    return {
        "active_e3": active_e3,
        "active_e2": active_e2,
        "erase_e3": erase_e3,
        "c_order": c_order,
        "erase_order": erase_order,
        "no_retile": no_retile,
        "c_fixed": [c_fixed_id],
        "e2_evaluation": e2_evaluation,
        "e3_evaluation": e3_evaluation,
        "schedule_only_retiled": schedule_only_retiled,
        "timed": timed,
    }


def _one_kernel(
    entry: Mapping[str, Any],
    *,
    regime: str,
    working_set_bytes: int,
    max_replicas: int,
    search_budget: int,
    last_level_cache_bytes: int,
    warmup: int,
    iterations: int,
    batches: int,
    measure_c_oracle: bool,
) -> dict[str, Any]:
    logical_io = _logical_io_bytes(entry)
    io_bytes = logical_io["boundary_io_bytes"]
    if io_bytes <= 0:
        raise ValueError("RVV benchmark has no logical boundary traffic")
    workload_cost = entry.get("workload_cost")
    if (
        workload_cost is not None
        and int(workload_cost["boundary_io_bytes"]) != io_bytes
    ):
        raise ValueError("runner logical bytes disagree with J workload-cost receipt")
    required_replicas = (
        max(2, math.ceil(working_set_bytes / io_bytes))
        if regime == "streaming"
        else 1
    )
    if required_replicas > max_replicas:
        raise ValueError(
            f"streaming regime needs {required_replicas} replicas, exceeding "
            f"--max-replicas={max_replicas}"
        )
    argument_sets, buffer_keepalive, output_buffer_sets = _argument_replicas(
        entry, required_replicas
    )
    output_buffers = output_buffer_sets[0]
    descriptors = [item._obj for item in argument_sets[0]]
    eviction_buffer = (
        ctypes.create_string_buffer(working_set_bytes)
        if regime == "cold-data"
        else None
    )

    candidate_by_id = {
        item["candidate"]["candidate_id"]: item for item in entry["candidates"]
    }
    concrete_plan = entry.get("concrete_search_plan", {})
    requested_order = concrete_plan.get(
        "c_prior_order", entry.get("candidate_order", tuple(candidate_by_id))
    )
    ordered_ids = []
    seen_ids = set()
    for candidate_id in (*requested_order, *candidate_by_id):
        if candidate_id in candidate_by_id and candidate_id not in seen_ids:
            seen_ids.add(candidate_id)
            ordered_ids.append(candidate_id)
    bounded_orders = _bounded_candidate_orders(
        entry, search_budget, measure_c_oracle=measure_c_oracle
    )
    timed_ids = set(bounded_orders["timed"])

    expected_outputs: dict[str, tuple[Any, int, int, str]] = {}
    for spec in entry["output_specs"]:
        output_name = spec["name"]
        reference_record = entry["reference_outputs"][
            entry["output_bindings"][output_name]
        ]
        payload = _decode_payload(reference_record)
        element_count = _numel([int(item) for item in spec["shape"]])
        byte_count = element_count * _dtype_size(spec["dtype"])
        if len(payload) != byte_count:
            raise ValueError(
                f"reference payload size differs from output {output_name!r}"
            )
        expected_outputs[output_name] = (
            ctypes.create_string_buffer(payload, len(payload)),
            element_count,
            byte_count,
            spec["dtype"],
        )

    records = []
    candidate_artifacts: dict[str, tuple[Any, Any, _ArgumentRotation]] = {}
    with tempfile.TemporaryDirectory(prefix="intentir-rvv-") as temporary:
        directory = Path(temporary)
        comparator = _NativeComparator(directory)
        comparator_receipt = comparator.receipt
        for candidate_id in ordered_ids:
            candidate = candidate_by_id[candidate_id]
            if candidate_id not in timed_ids:
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "not_visited_due_budget",
                    }
                )
                continue
            try:
                library_path = _link_object(directory, candidate)
                library = ctypes.CDLL(str(library_path))
                symbol = candidate["candidate"]["c_interface_symbol"]
                function = getattr(library, symbol)
                function.argtypes = [
                    ctypes.POINTER(type(item)) for item in descriptors
                ]
                function.restype = None
                function(*argument_sets[0])
                correctness = []
                passed = True
                for spec, buffer in zip(
                    entry["output_specs"], output_buffers, strict=True
                ):
                    output_name = spec["name"]
                    expected, element_count, byte_count, dtype = expected_outputs[
                        output_name
                    ]
                    agrees, max_error = comparator.close(
                        buffer,
                        expected,
                        byte_count=byte_count,
                        element_count=element_count,
                        dtype=dtype,
                        rule=entry["output_rules"][output_name],
                    )
                    correctness.append(
                        {
                            "output": output_name,
                            "passed": agrees,
                            "max_abs_error": max_error,
                            "comparator_protocol": NATIVE_COMPARATOR_PROTOCOL,
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
                rotation = _ArgumentRotation(argument_sets)
                latency, samples, prep_us = _measure(
                    function,
                    rotation,
                    regime=regime,
                    eviction_buffer=eviction_buffer,
                    warmup=warmup,
                    iterations=iterations,
                    batches=1,
                )
                candidate_artifacts[candidate_id] = (library, function, rotation)
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "measured",
                        "latency_us": latency,
                        "initial_ranking_samples_us": samples,
                        "initial_data_prep_us_per_call": prep_us,
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
    # Exclude the sequential initial ranking sample.  Every correct state gets
    # the same number of fresh batches in alternating round-robin order, so a
    # fused state cannot win merely because it was measured earlier or later.
    fresh_samples: dict[str, list[float]] = {
        item["candidate"]["candidate_id"]: [] for item in measured
    }
    for round_index in range(batches):
        round_order = measured if round_index % 2 == 0 else list(reversed(measured))
        for item in round_order:
            candidate_id = item["candidate"]["candidate_id"]
            function = candidate_artifacts[candidate_id][1]
            rotation = candidate_artifacts[candidate_id][2]
            latency, samples, prep_us = _measure(
                function,
                rotation,
                regime=regime,
                eviction_buffer=eviction_buffer,
                warmup=0,
                iterations=iterations,
                batches=1,
            )
            assert len(samples) == 1
            fresh_samples[candidate_id].append(latency)
            item.setdefault("data_prep_samples_us", []).append(prep_us)
    for item in measured:
        candidate_id = item["candidate"]["candidate_id"]
        samples = fresh_samples[candidate_id]
        item["latency_us"] = float(statistics.median(samples))
        item["selection_samples_us"] = samples
        item["selection_protocol"] = "fresh-alternating-round-robin.v1"
        item["logical_boundary_gbps"] = io_bytes / item["latency_us"] / 1_000.0
    measured_by_id = {
        item["candidate"]["candidate_id"]: item for item in measured
    }

    def best(candidate_ids: list[str]) -> dict[str, Any]:
        available = [
            measured_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in measured_by_id
        ]
        if not available:
            raise KernelFailure("ablation arm has no correct RVV candidate", records)
        return min(available, key=lambda item: item["latency_us"])

    def optional_best(candidate_ids: list[str]) -> dict[str, Any] | None:
        available = [
            measured_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in measured_by_id
        ]
        return min(available, key=lambda item: item["latency_us"]) if available else None

    e2 = best(bounded_orders["e2_evaluation"])
    e3 = best(bounded_orders["e3_evaluation"])
    c_fixed = best(bounded_orders["c_fixed"])
    no_retile = optional_best(bounded_orders["no_retile"])
    schedule_only_retiled = optional_best(
        bounded_orders["schedule_only_retiled"]
    )
    erase_budget_ids = bounded_orders["erase_e3"]
    c_prior_budget_ids = bounded_orders["active_e3"]
    erase_budget_winner = best(erase_budget_ids)
    c_prior_budget_winner = best(c_prior_budget_ids)
    c_fixed_id = concrete_plan.get("c_fixed_candidate_id")
    if c_fixed_id != c_fixed["candidate"]["candidate_id"]:
        raise KernelFailure("C-Fixed candidate identity drifted", records)

    oracle_complete = measure_c_oracle and not any(
        item["status"] == "not_visited_due_budget" for item in records
    )
    oracle_best = optional_best(bounded_orders["c_order"]) if oracle_complete else None

    def candidates_to_95pct(order: list[str]) -> int | None:
        if oracle_best is None:
            return None
        threshold_us = float(oracle_best["latency_us"]) / 0.95
        best_so_far = float("inf")
        for count, candidate_id in enumerate(order, start=1):
            measured_candidate = measured_by_id.get(candidate_id)
            if measured_candidate is not None:
                best_so_far = min(
                    best_so_far, float(measured_candidate["latency_us"])
                )
            if best_so_far <= threshold_us:
                return count
        return None

    c_candidates_to_95pct = candidates_to_95pct(bounded_orders["c_order"])
    erase_candidates_to_95pct = candidates_to_95pct(
        bounded_orders["erase_order"]
    )
    achieved_working_set_bytes = required_replicas * io_bytes
    floating_point_ops = (
        int(workload_cost["floating_point_ops"])
        if workload_cost is not None
        else 0
    )
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
        "workload_profile": entry.get("workload_profile"),
        "workload_valuation": entry.get("workload_valuation"),
        "j_provenance": entry.get("j_provenance"),
        "baseline_policy": entry.get("baseline_policy"),
        "snapshot_digest": entry["snapshot_digest"],
        "j_digest": entry["j_digest"],
        "c_digest": entry["c_digest"],
        "contract_view": entry["contract_view"],
        "extents": entry["extents"],
        "candidates": records,
        "winner": e3["candidate"],
        "performance": {
            "c_fixed_us": c_fixed["latency_us"],
            "schedule_only_us": e2["latency_us"],
            "intentir_us": e3["latency_us"],
            "g_schedule": c_fixed["latency_us"] / e2["latency_us"],
            "g_lowering": e2["latency_us"] / e3["latency_us"],
            "c_fixed_candidate_id": c_fixed_id,
            "schedule_only_winner_candidate_id": e2["candidate"]["candidate_id"],
            "intentir_winner_candidate_id": e3["candidate"]["candidate_id"],
        },
        "c_prior_efficiency": {
            "candidate_budget": search_budget,
            "budget_unit": "distinct_native_object_sha256",
            "c_prior_best_us": c_prior_budget_winner["latency_us"],
            "erase_c_best_us": erase_budget_winner["latency_us"],
            "best_at_budget_c_over_erase_c_speedup": (
                erase_budget_winner["latency_us"]
                / c_prior_budget_winner["latency_us"]
            ),
            "c_prior_candidates_to_95pct_of_final_best": c_candidates_to_95pct,
            "erase_c_candidates_to_95pct_of_final_best": (
                erase_candidates_to_95pct
            ),
            "oracle_complete": oracle_complete,
            "oracle_best_us": (
                oracle_best["latency_us"] if oracle_best is not None else None
            ),
            "unavailable_reason": (
                None
                if oracle_complete
                else "complete_latency_map_not_requested"
            ),
            "c_prior_budget_winner_candidate_id": c_prior_budget_winner[
                "candidate"
            ]["candidate_id"],
            "erase_c_budget_winner_candidate_id": erase_budget_winner[
                "candidate"
            ]["candidate_id"],
        },
        "supplementary_diagnostics": {
            "formal_role": "not_a_main_experiment_arm",
            "b_transformation_wins": bool(
                e3["candidate"]["transformation"]["b_transformations"]
            ),
            "no_retile_schedule_us": (
                no_retile["latency_us"] if no_retile is not None else None
            ),
            "retiled_schedule_us": (
                schedule_only_retiled["latency_us"]
                if schedule_only_retiled is not None
                else None
            ),
            "retiled_over_no_retile_speedup": (
                no_retile["latency_us"] / schedule_only_retiled["latency_us"]
                if no_retile is not None and schedule_only_retiled is not None
                else None
            ),
            "retiled_over_c_fixed_speedup": (
                c_fixed["latency_us"] / schedule_only_retiled["latency_us"]
                if schedule_only_retiled is not None
                else None
            ),
            "intentir_over_c_fixed_speedup": (
                c_fixed["latency_us"] / e3["latency_us"]
            ),
            "no_retile_winner_candidate_id": (
                no_retile["candidate"]["candidate_id"]
                if no_retile is not None
                else None
            ),
            "retiled_winner_candidate_id": (
                schedule_only_retiled["candidate"]["candidate_id"]
                if schedule_only_retiled is not None
                else None
            ),
        },
        "search_execution": {
            "candidate_budget": search_budget,
            "budget_unit": "distinct_native_object_sha256",
            "candidate_correctness_pass_count": len(
                [
                    item
                    for item in records
                    if item["status"] == "measured"
                ]
            ),
            "candidate_correctness_fail_count": len(
                [item for item in records if item["status"] == "incorrect"]
            ),
            "candidate_execution_failure_count": len(
                [item for item in records if item["status"] == "failed"]
            ),
            "timed_candidate_count": len(measured),
            "not_visited_candidate_count": len(
                [
                    item
                    for item in records
                    if item["status"] == "not_visited_due_budget"
                ]
            ),
            "active_e3_budget_candidate_ids": bounded_orders["active_e3"],
            "active_e2_budget_candidate_ids": bounded_orders["active_e2"],
            "erase_c_e3_budget_candidate_ids": bounded_orders["erase_e3"],
            "e3_evaluation_candidate_ids": bounded_orders["e3_evaluation"],
            "e2_evaluation_candidate_ids": bounded_orders["e2_evaluation"],
            "retiled_schedule_candidate_ids": bounded_orders[
                "schedule_only_retiled"
            ],
            "c_fixed_candidate_ids": bounded_orders["c_fixed"],
            "timed_candidate_ids": bounded_orders["timed"],
            "correctness_comparator": comparator_receipt,
        },
        "measurement_analysis": {
            "logical_io": {
                **logical_io,
                "e2_logical_boundary_gbps": (
                    io_bytes / e2["latency_us"] / 1_000.0
                ),
                "e3_logical_boundary_gbps": (
                    io_bytes / e3["latency_us"] / 1_000.0
                ),
                "traffic_definition": (
                    "sum of certified boundary input and output payload bytes; "
                    "not measured DRAM transactions"
                ),
            },
            "roofline_point": {
                "j_workload_cost": workload_cost,
                "arithmetic_intensity_flop_per_byte": (
                    floating_point_ops / io_bytes
                ),
                "e2_achieved_gflops": (
                    floating_point_ops / e2["latency_us"] / 1_000.0
                ),
                "e3_achieved_gflops": (
                    floating_point_ops / e3["latency_us"] / 1_000.0
                ),
                "hardware_counter_traffic": None,
                "interpretation": "J-derived logical/minimum-traffic point",
            },
            "cache_regime": {
                "name": regime,
                "buffer_policy": (
                    "rotating-certified-invocation-replicas"
                    if regime == "streaming"
                    else "eviction-before-each-single-shot"
                    if regime == "cold-data"
                    else "persistent-single-address"
                ),
                "replicas": required_replicas,
                "requested_working_set_bytes": working_set_bytes,
                "achieved_working_set_bytes": achieved_working_set_bytes,
                "eviction_bytes": (
                    working_set_bytes if regime == "cold-data" else 0
                ),
                "data_prep_in_timed_region": False,
                "streaming_target_achieved": (
                    achieved_working_set_bytes >= working_set_bytes
                    if regime == "streaming"
                    else None
                ),
                "last_level_cache_bytes": last_level_cache_bytes,
                "working_set_exceeds_last_level_cache": (
                    achieved_working_set_bytes > last_level_cache_bytes
                    if regime == "streaming" and last_level_cache_bytes
                    else working_set_bytes > last_level_cache_bytes
                    if regime == "cold-data" and last_level_cache_bytes
                    else None
                ),
            },
            "timing": "pinned-process perf_counter_ns via ctypes C ABI",
            "comparison_protocol": "fresh-alternating-round-robin.v1",
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
    parser.add_argument(
        "--regime",
        choices=("warm", "streaming", "cold-data"),
        default="warm",
    )
    parser.add_argument("--working-set-mib", type=int, default=128)
    parser.add_argument("--max-replicas", type=int, default=262144)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument(
        "--measure-c-oracle",
        action="store_true",
        help=(
            "execute the complete concrete object universe so C/no-C can report "
            "candidates-to-95%-of-final-best"
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cold-iterations", type=int, default=16)
    parser.add_argument("--batches", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(
        args.warmup,
        args.iterations,
        args.cold_iterations,
        args.batches,
        args.working_set_mib,
        args.max_replicas,
        args.search_budget,
    ) <= 0:
        raise ValueError("all measurement counts and budgets must be positive")
    os.sched_setaffinity(0, {args.cpu})
    cache_record = _cpu_cache_record()
    working_set_bytes = args.working_set_mib * 1024 * 1024
    last_level_cache_bytes = cache_record["last_level_data_cache_bytes"]
    if (
        args.regime != "warm"
        and last_level_cache_bytes
        and working_set_bytes <= last_level_cache_bytes
    ):
        raise ValueError(
            f"{args.regime} working set ({working_set_bytes} bytes) must exceed "
            f"detected last-level cache ({last_level_cache_bytes} bytes)"
        )
    bundle_bytes = args.bundle.read_bytes()
    bundle = json.loads(bundle_bytes)
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
    measured_iterations = (
        args.cold_iterations if args.regime == "cold-data" else args.iterations
    )
    for entry in entries:
        try:
            record = _one_kernel(
                entry,
                regime=args.regime,
                working_set_bytes=working_set_bytes,
                max_replicas=args.max_replicas,
                search_budget=args.search_budget,
                last_level_cache_bytes=last_level_cache_bytes,
                warmup=args.warmup,
                iterations=measured_iterations,
                batches=args.batches,
                measure_c_oracle=args.measure_c_oracle,
            )
            kernels.append(record)
            performance = record["performance"]
            print(
                f"{entry['kernel']}: CF={performance['c_fixed_us']:.4f} us "
                f"Schedule={performance['schedule_only_us']:.4f} us "
                f"IntentIR={performance['intentir_us']:.4f} us "
                f"G_schedule={performance['g_schedule']:.3f}x "
                f"G_lowering={performance['g_lowering']:.3f}x "
                "C="
                f"{record['c_prior_efficiency']['best_at_budget_c_over_erase_c_speedup']:.3f}x",
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
    schedule_speedups = [item["performance"]["g_schedule"] for item in kernels]
    lowering_speedups = [item["performance"]["g_lowering"] for item in kernels]
    c_speedups = [
        item["c_prior_efficiency"]["best_at_budget_c_over_erase_c_speedup"]
        for item in kernels
    ]
    c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "c_prior_candidates_to_95pct_of_final_best"
        ]
        for item in kernels
        if item["c_prior_efficiency"][
            "c_prior_candidates_to_95pct_of_final_best"
        ]
        is not None
    ]
    erase_c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "erase_c_candidates_to_95pct_of_final_best"
        ]
        for item in kernels
        if item["c_prior_efficiency"][
            "erase_c_candidates_to_95pct_of_final_best"
        ]
        is not None
    ]

    def geomean(values: list[float]) -> float | None:
        return (
            math.exp(sum(math.log(item) for item in values) / len(values))
            if values
            else None
        )

    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bundle_toolchain_digest": bundle["toolchain_digest"],
        "target": bundle["target"],
        "workload_profile": bundle.get("workload_profile", "closed-v1"),
        "baseline_policy": bundle.get("baseline_policy", "target_refj_only"),
        "experiment_spec_protocol": bundle.get("experiment_spec_protocol"),
        "experiment_spec_digest": bundle.get("experiment_spec_digest"),
        "measurement_spec_protocol": bundle.get("measurement_spec_protocol"),
        "measurement_spec_digest": bundle.get("measurement_spec_digest"),
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": {
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "vlen_bits": 128,
            "cache": cache_record,
        },
        "measurement": {
            "data_regime": args.regime,
            "effective_timing": (
                "monotonic-clock-single-shot"
                if args.regime == "cold-data"
                else "monotonic-clock-batch"
            ),
            "timer": "perf_counter_ns",
            "correctness_policy": bundle.get("correctness_policy"),
            "working_set_bytes": working_set_bytes,
            "warmup_iterations": args.warmup,
            "iterations_per_batch": measured_iterations,
            "measurement_batches": args.batches,
            "detected_last_level_data_cache_bytes": last_level_cache_bytes,
            "cache_eviction_in_timed_region": False,
            "regime": args.regime,
            "working_set_mib": args.working_set_mib,
            "warmup": args.warmup,
            "iterations": measured_iterations,
            "batches": args.batches,
            "search_budget": args.search_budget,
            "measure_c_oracle": args.measure_c_oracle,
        },
        "summary": {
            "requested": len(entries),
            "successful": len(kernels),
            "failed": len(failures),
            "geomean_g_schedule": geomean(schedule_speedups),
            "geomean_g_lowering": geomean(lowering_speedups),
        },
        "c_prior_efficiency_summary": {
            "geomean_best_at_budget_c_over_erase_c_speedup": geomean(
                c_speedups
            ),
            "oracle_complete_kernel_count": len(c_candidates_to_95pct),
            "median_c_prior_candidates_to_95pct_of_final_best": (
                statistics.median(c_candidates_to_95pct)
                if c_candidates_to_95pct
                else None
            ),
            "median_erase_c_candidates_to_95pct_of_final_best": (
                statistics.median(erase_c_candidates_to_95pct)
                if erase_c_candidates_to_95pct
                else None
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
