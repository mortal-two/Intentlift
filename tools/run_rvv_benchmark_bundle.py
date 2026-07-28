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


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v2"
REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v4"


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
    entry: Mapping[str, Any], search_budget: int
) -> dict[str, list[str]]:
    """Select the exact bounded timing union before any candidate executes.

    Only visited native objects receive Q/D and timing.  The visited union is the
    active C-prior E3 prefix, both schedule-only C/Erase-C prefixes, and the
    mandatory frozen/no-retile controls.  Other compiled objects remain explicit
    ``not_visited_due_budget`` records.  This makes the CLI budget executable
    instead of an after-the-fact report cut.
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
    schedule_only = set(entry["schedule_only_candidate_ids"]) & available
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
    c_schedule_order = [item for item in c_order if item in schedule_only]
    erase_schedule_order = [item for item in erase_order if item in schedule_only]
    active_e3 = c_order[:search_budget]
    active_e2 = c_schedule_order[:search_budget]
    erase_e2 = erase_schedule_order[:search_budget]
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
    frozen_id = plan.get("frozen_c_candidate_id")
    mandatory = [*no_retile]
    if frozen_id in available:
        mandatory.append(frozen_id)

    def unique(items: Any) -> list[str]:
        result = []
        seen = set()
        for candidate_id in items:
            if candidate_id in available and candidate_id not in seen:
                result.append(candidate_id)
                seen.add(candidate_id)
        return result

    e2_evaluation = unique((*active_e2, *mandatory))
    # E3 is nested: it can never forget the E2 controls under a finite budget.
    e3_evaluation = unique((*active_e3, *e2_evaluation))
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
    for candidate_id in (*e3_evaluation, *erase_e2):
        if candidate_id not in seen:
            timed.append(candidate_id)
            seen.add(candidate_id)
    return {
        "active_e3": active_e3,
        "active_e2": active_e2,
        "erase_e2": erase_e2,
        "no_retile": no_retile,
        "mandatory": unique(mandatory),
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
    bounded_orders = _bounded_candidate_orders(entry, search_budget)
    timed_ids = set(bounded_orders["timed"])

    records = []
    candidate_artifacts: dict[str, tuple[Any, Any, _ArgumentRotation]] = {}
    with tempfile.TemporaryDirectory(prefix="intentir-rvv-") as temporary:
        directory = Path(temporary)
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

    e2 = best(bounded_orders["e2_evaluation"])
    e3 = best(bounded_orders["e3_evaluation"])
    no_retile = best(bounded_orders["no_retile"])
    schedule_only_retiled = best(bounded_orders["schedule_only_retiled"])
    erase_budget_ids = bounded_orders["erase_e2"]
    c_prior_budget_ids = bounded_orders["active_e2"]
    erase_budget_winner = best(erase_budget_ids)
    c_prior_budget_winner = best(c_prior_budget_ids)
    frozen_id = concrete_plan.get("frozen_c_candidate_id")
    if frozen_id not in measured_by_id:
        raise KernelFailure(
            "frozen-C projection has no correct executable candidate", records
        )
    frozen_c = measured_by_id[frozen_id]
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
            "e2_schedule_only_us": e2["latency_us"],
            "e3_contract_guided_us": e3["latency_us"],
            "e3_over_e2_speedup": e2["latency_us"] / e3["latency_us"],
            "b_transformation_wins": bool(
                e3["candidate"]["transformation"]["b_transformations"]
            ),
            "no_retile_schedule_us": no_retile["latency_us"],
            "retiled_schedule_us": schedule_only_retiled["latency_us"],
            "retiled_over_no_retile_speedup": (
                no_retile["latency_us"] / schedule_only_retiled["latency_us"]
            ),
            "frozen_c_us": frozen_c["latency_us"],
            "schedule_only_over_frozen_c_speedup": (
                frozen_c["latency_us"] / e2["latency_us"]
            ),
            "retiled_over_frozen_c_speedup": (
                frozen_c["latency_us"] / schedule_only_retiled["latency_us"]
            ),
            "full_search_over_frozen_c_speedup": (
                frozen_c["latency_us"] / e3["latency_us"]
            ),
            "search_budget": search_budget,
            "search_budget_unit": "distinct_native_object_sha256",
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
            "e3_evaluation_candidate_ids": bounded_orders["e3_evaluation"],
            "e2_evaluation_candidate_ids": bounded_orders["e2_evaluation"],
            "retiled_schedule_candidate_ids": bounded_orders[
                "schedule_only_retiled"
            ],
            "erase_c_e2_budget_candidate_ids": bounded_orders["erase_e2"],
            "mandatory_control_candidate_ids": bounded_orders["mandatory"],
            "timed_candidate_ids": bounded_orders["timed"],
            "erase_c_budget_us": erase_budget_winner["latency_us"],
            "c_prior_budget_us": c_prior_budget_winner["latency_us"],
            "c_prior_over_erase_c_speedup": (
                erase_budget_winner["latency_us"]
                / c_prior_budget_winner["latency_us"]
            ),
            "no_retile_winner_candidate_id": no_retile["candidate"][
                "candidate_id"
            ],
            "retiled_winner_candidate_id": schedule_only_retiled["candidate"][
                "candidate_id"
            ],
            "frozen_c_candidate_id": frozen_id,
            "erase_c_budget_winner_candidate_id": erase_budget_winner[
                "candidate"
            ]["candidate_id"],
            "c_prior_budget_winner_candidate_id": c_prior_budget_winner[
                "candidate"
            ]["candidate_id"],
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
            )
            kernels.append(record)
            performance = record["performance"]
            print(
                f"{entry['kernel']}: E2={performance['e2_schedule_only_us']:.4f} "
                f"us E3={performance['e3_contract_guided_us']:.4f} us "
                f"structural={performance['e3_over_e2_speedup']:.3f}x "
                f"retile={performance['retiled_over_no_retile_speedup']:.3f}x "
                f"C={performance['c_prior_over_erase_c_speedup']:.3f}x",
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
    retiled_speedups = [
        item["performance"]["retiled_over_no_retile_speedup"] for item in kernels
    ]
    c_speedups = [
        item["performance"]["c_prior_over_erase_c_speedup"] for item in kernels
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
        },
        "summary": {
            "requested": len(entries),
            "successful": len(kernels),
            "failed": len(failures),
            "geomean_e3_over_e2_speedup": geomean(speedups),
            "geomean_retiled_over_no_retile_speedup": geomean(
                retiled_speedups
            ),
            "geomean_c_prior_over_erase_c_speedup": geomean(c_speedups),
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
