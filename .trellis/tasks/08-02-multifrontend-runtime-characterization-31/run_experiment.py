#!/usr/bin/env python3
"""Reconstruct exact Gate2 Q, run anonymous CUDA runtime, and summarize Task31."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.frontend_workloads import build_frontend_dataset_gate2_suite
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate2.orchestrate import gate2_input_digest
from intentir.verify.gate2.policy import gate2_policy_for
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.workload_registry import prepare_registered_scrc_workload


TASK_ROOT = Path(__file__).resolve().parent
TASKS_ROOT = TASK_ROOT.parent
TASK28 = TASKS_ROOT / "08-01-j-only-replay-multifrontend-batch-28"
TASK29 = TASKS_ROOT / "08-01-tilelang-cuda-paired-j-analysis-29"
TASK30 = TASKS_ROOT / "08-02-artifact-only-multifrontend-backend-reuse-30"
TASK30_INPUT = TASK30 / "backend_inputs_current.json"
TASK30_PROVENANCE = TASK30 / "provenance_current.json"
TASK30_BACKEND = TASK30 / "backend_results_current.json"
TASK29_RESULTS = TASK29 / "results_current.json"

VALUES_DIR = TASK_ROOT / "runtime_values_current"
RUNS_DIR = TASK_ROOT / "runtime_runs_current"
INPUT_MANIFEST = TASK_ROOT / "runtime_inputs_current.json"
PROVENANCE_PATH = TASK_ROOT / "runtime_provenance_current.json"
RUNTIME_RESULTS = TASK_ROOT / "runtime_results_current.json"
SUMMARY_JSON = TASK_ROOT / "results_current.json"
SUMMARY_MD = TASK_ROOT / "results_current.md"
CONSUMER = TASK_ROOT / "runtime_consume.py"

INPUT_PROTOCOL = "intentir.experiment.anonymous-j-runtime-input.v1"
PROVENANCE_PROTOCOL = "intentir.experiment.runtime-input-provenance.v1"
COMBINED_PROTOCOL = "intentir.experiment.anonymous-j-runtime-batch.v1"
SUMMARY_PROTOCOL = "intentir.experiment.multifrontend-runtime-characterization.v2"


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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


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


def _shape_records(values: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        result[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.name,
        }
    return result


def _write_deterministic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    """Write an np.load-compatible archive with byte-stable member metadata."""

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, value in sorted(values.items()):
            if not name.isidentifier():
                raise ValueError(f"runtime logical input name is unsafe: {name!r}")
            buffer = io.BytesIO()
            array = np.asarray(value)
            # np.ascontiguousarray promotes rank-0 arrays to shape (1,), which
            # silently breaks J value-dimension relations for scalar inputs.
            # Preserve true scalars while canonicalizing higher-rank storage.
            serialized = array if array.ndim == 0 else np.ascontiguousarray(array)
            np.lib.format.write_array(
                buffer,
                serialized,
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _verify_digest_field(value: Mapping[str, Any], field: str) -> None:
    content = dict(value)
    observed = content.pop(field, None)
    if observed != _digest(content):
        raise ValueError(f"{field} mismatch")


def export_inputs() -> dict[str, Any]:
    task30_inputs = _read(TASK30_INPUT)
    task30_provenance = _read(TASK30_PROVENANCE)
    _verify_digest_field(task30_provenance, "provenance_digest")
    inputs_by_id = {item["run_id"]: item for item in task30_inputs["entries"]}
    provenance_by_id = {
        item["run_id"]: item for item in task30_provenance["entries"]
    }
    if set(inputs_by_id) != set(provenance_by_id) or len(inputs_by_id) != 62:
        raise ValueError("Task30 input/provenance run sets are not exact full-62")
    VALUES_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    seed_routes: Counter[str] = Counter()

    for run_id in sorted(inputs_by_id):
        source = provenance_by_id[run_id]
        run_dir = TASK28 / source["run_dir"]
        artifact_path = run_dir / "artifact.json"
        adapter_path = run_dir / "adapter.json"
        gate2_path = run_dir / "gate2.json"
        if _file_digest(artifact_path) != source["artifact_sha256"]:
            raise ValueError(f"Task28 artifact digest drifted: {run_dir}")
        artifact = canonicalize_gate1_artifact(
            Gate1Artifact.from_json(_read(artifact_path))
        )
        adapter = _read(adapter_path)
        gate2 = _read(gate2_path)
        if gate2.get("passed") is not True or len(gate2.get("cases", [])) != 1:
            raise ValueError(f"Task28 Gate2 is not one exact passing case: {run_dir}")
        accepted_input_digest = gate2["cases"][0]["input_digest"]
        prepared = prepare_registered_scrc_workload(source["case"])
        candidate_seeds = [
            ("accepted_adapter_seed", int(adapter["seed"])),
            ("current_projection_seed", int(prepared.adapter.seed)),
        ]
        selected_suite = None
        selected_route = None
        tried: set[int] = set()
        for route, seed in candidate_seeds:
            if seed in tried:
                continue
            tried.add(seed)
            suite = build_frontend_dataset_gate2_suite(
                prepared,
                artifact,
                seed=seed,
            )
            if gate2_input_digest(suite.cases[0].values) == accepted_input_digest:
                selected_suite = suite
                selected_route = route
                break
        if selected_suite is None or selected_route is None:
            raise ValueError(f"cannot reconstruct accepted Gate2 Q: {run_dir}")
        seed_routes[selected_route] += 1
        case = selected_suite.cases[0]
        reference = ScalarLoweringRunner().run(case.refeval, case.values)
        if not reference.ok or reference.unsupported:
            raise ValueError(
                f"accepted Gate2 RefJ no longer executes: {run_dir}: {reference.error}"
            )
        logical_values = {
            logical: np.asarray(case.values[runtime_key])
            for logical, runtime_key in case.refeval.input_binding_map.items()
        }
        expected_inputs = {item.name for item in artifact.j.intent.inputs}
        if set(logical_values) != expected_inputs:
            raise ValueError(f"logical runtime values differ from J inputs: {run_dir}")
        logical_input_digest = gate2_input_digest(logical_values)
        value_path = VALUES_DIR / f"{run_id}.npz"
        _write_deterministic_npz(value_path, logical_values)
        task30_entry = inputs_by_id[run_id]
        if case.refeval.dims_map != task30_entry["concrete_extents"]:
            raise ValueError(f"runtime Q dimensions differ from Task30 lower: {run_dir}")
        if artifact.j.to_json() != task30_entry["snapshot"]["j"]:
            raise ValueError(f"runtime artifact J differs from Task30 snapshot: {run_dir}")
        entry = {
            "run_id": run_id,
            "snapshot": task30_entry["snapshot"],
            "concrete_extents": task30_entry["concrete_extents"],
            "input_file": value_path.name,
            "input_file_sha256": _file_digest(value_path),
            "input_digest": logical_input_digest,
            "comparison_policy": gate2_policy_for(prepared.entry.q_policy).to_json(),
        }
        entries.append(entry)
        provenance_rows.append(
            {
                "run_id": run_id,
                "stem": source["stem"],
                "frontend": source["frontend"],
                "case": source["case"],
                "run_dir": source["run_dir"],
                "artifact_sha256": source["artifact_sha256"],
                "gate1_artifact_digest": source["gate1_artifact_digest"],
                "accepted_gate2_input_digest": accepted_input_digest,
                "reconstructed_gate2_input_digest": gate2_input_digest(
                    case.values
                ),
                "exact_accepted_gate2_input": True,
                "seed_route": selected_route,
                "runtime_input_digest": logical_input_digest,
                "runtime_input_file_sha256": _file_digest(value_path),
                "runtime_inputs": _shape_records(logical_values),
                "refj_export_precheck": "pass",
                "backend_receives_this_provenance": False,
            }
        )
    if len(entries) != 62 or len({item["run_id"] for item in entries}) != 62:
        raise ValueError("runtime export denominator is not 62 unique artifacts")
    manifest: dict[str, Any] = {
        "protocol": INPUT_PROTOCOL,
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest["input_digest"] = _digest(manifest)
    provenance: dict[str, Any] = {
        "protocol": PROVENANCE_PROTOCOL,
        "entry_count": len(provenance_rows),
        "entries": provenance_rows,
        "seed_route_counts": dict(sorted(seed_routes.items())),
        "task30_input_digest": task30_inputs["input_digest"],
        "task30_provenance_digest": task30_provenance["provenance_digest"],
        "backend_receives_this_file": False,
        "claim_boundary": (
            "Source/case/Gate provenance reconstructs Q outside the backend; "
            "runtime_consume.py receives only anonymous J+C, extents, Q arrays, "
            "and the public comparison policy"
        ),
    }
    provenance["provenance_digest"] = _digest(provenance)
    _write(INPUT_MANIFEST, manifest)
    _write(PROVENANCE_PATH, provenance)
    return manifest


def run_runtime(
    *,
    benchmark: bool,
    warmup_ms: int,
    repeat_ms: int,
    case_timeout_s: int,
    limit: int | None,
) -> dict[str, Any]:
    manifest = _read(INPUT_MANIFEST)
    entries = list(manifest["entries"])
    if limit is not None:
        entries = entries[:limit]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, entry in enumerate(entries, start=1):
        output = RUNS_DIR / f"{entry['run_id']}.json"
        with tempfile.NamedTemporaryFile(
            prefix=f".{entry['run_id']}.",
            suffix=".json",
            dir=RUNS_DIR,
            delete=False,
        ) as temporary:
            temporary_output = Path(temporary.name)
        command = [
            sys.executable,
            str(CONSUMER),
            "--manifest",
            str(INPUT_MANIFEST),
            "--run-id",
            entry["run_id"],
            "--output",
            str(temporary_output),
            "--warmup-ms",
            str(warmup_ms),
            "--repeat-ms",
            str(repeat_ms),
        ]
        if not benchmark:
            command.append("--no-benchmark")
        completed = None
        driver_error: BaseException | None = None
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=case_timeout_s,
            )
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise RuntimeError("runtime consumer produced no result artifact")
            result = _read(temporary_output)
            _verify_digest_field(result, "result_digest")
            if result.get("run_id") != entry["run_id"]:
                raise ValueError("runtime consumer result run_id mismatch")
            runtime_pass = (
                result.get("stages", {})
                .get("execute_compare", {})
                .get("status")
                == "pass"
            )
            expected_returncode = 0 if runtime_pass else 2
            if completed.returncode != expected_returncode:
                raise RuntimeError(
                    "runtime consumer return code disagrees with its result: "
                    f"returncode={completed.returncode}, expected={expected_returncode}"
                )
            temporary_output.replace(output)
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
            driver_error = error
        finally:
            temporary_output.unlink(missing_ok=True)
        if driver_error is not None:
            stderr = completed.stderr if completed is not None else ""
            result = {
                "protocol": "intentir.experiment.runtime-driver-failure.v1",
                "run_id": entry["run_id"],
                "stages": {
                    "driver": {
                        "status": "fail",
                        "stage": "driver",
                        "error_type": type(driver_error).__name__,
                        "diagnostic": f"{driver_error}\n{stderr[-4000:]}",
                    }
                },
            }
            result["result_digest"] = _digest(result)
            _write(output, result)
        results.append(result)
        execute_status = result.get("stages", {}).get("execute_compare", {}).get(
            "status", "missing"
        )
        benchmark_status = result.get("stages", {}).get("benchmark", {}).get(
            "status", "missing"
        )
        print(
            f"[{index}/{len(entries)}] {entry['run_id'][:12]} "
            f"runtime={execute_status} benchmark={benchmark_status}",
            flush=True,
        )
    stage_names = (
        "decode",
        "compile",
        "input",
        "refj",
        "execute_compare",
        "benchmark",
    )
    stage_counts = {}
    for stage in stage_names:
        counts = Counter(
            str(item.get("stages", {}).get(stage, {}).get("status", "missing"))
            for item in results
        )
        stage_counts[stage] = dict(sorted(counts.items()))
    combined: dict[str, Any] = {
        "protocol": COMBINED_PROTOCOL,
        "input_digest": manifest["input_digest"],
        "entry_count": len(results),
        "benchmark_enabled": benchmark,
        "benchmark_windows": {"warmup_ms": warmup_ms, "repeat_ms": repeat_ms},
        "case_timeout_s": case_timeout_s,
        "stage_status_counts": stage_counts,
        "entries": results,
    }
    combined["result_digest"] = _digest(combined)
    _write(RUNTIME_RESULTS, combined)
    return combined


def _status(entry: Mapping[str, Any], stage: str) -> str:
    return str(entry.get("stages", {}).get(stage, {}).get("status", "missing"))


def _numbers(values: Iterable[float]) -> dict[str, float | int | None]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
    }


def _geomean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if float(value) > 0.0]
    return math.exp(statistics.fmean(math.log(item) for item in data)) if data else None


def _frontend_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "artifact_count": len(rows),
        "runtime_pass": sum(_status(row, "execute_compare") == "pass" for row in rows),
        "benchmark_pass": sum(_status(row, "benchmark") == "pass" for row in rows),
        "serial_compile_ms": _numbers(
            row["stages"]["compile"]["elapsed_ms"]
            for row in rows
            if _status(row, "compile") == "pass"
        ),
        "prepared_graph_median_us": _numbers(
            row["stages"]["benchmark"]["median_us"]
            for row in rows
            if _status(row, "benchmark") == "pass"
        ),
        "cuda_kernel_count": _numbers(
            row["stages"]["compile"]["kernel_count"]
            for row in rows
            if _status(row, "compile") == "pass"
        ),
        "cuda_total_cubin_bytes": _numbers(
            row["stages"]["compile"]["total_cubin_size_bytes"]
            for row in rows
            if _status(row, "compile") == "pass"
        ),
        "rvv_object_bytes": _numbers(row["rvv_object_bytes"] for row in rows),
        "rvv_instruction_count": _numbers(
            row["rvv_instruction_count"] for row in rows
        ),
    }


def summarize() -> dict[str, Any]:
    runtime = _read(RUNTIME_RESULTS)
    provenance = _read(PROVENANCE_PATH)
    task30 = _read(TASK30_BACKEND)
    task29 = _read(TASK29_RESULTS)
    _verify_digest_field(provenance, "provenance_digest")
    _verify_digest_field(runtime, "result_digest")
    runtime_by_id = {item["run_id"]: item for item in runtime["entries"]}
    provenance_by_id = {item["run_id"]: item for item in provenance["entries"]}
    task30_by_id = {item["run_id"]: item for item in task30["entries"]}
    if not (
        set(runtime_by_id) == set(provenance_by_id) == set(task30_by_id)
        and len(runtime_by_id) == 62
    ):
        raise ValueError("Task31 runtime/provenance/Task30 run sets differ")
    pair_analysis = {item["stem"]: item for item in task29["pairs"]}
    joined = []
    for run_id in sorted(runtime_by_id):
        row = dict(runtime_by_id[run_id])
        source = provenance_by_id[run_id]
        old = task30_by_id[run_id]
        new_cubins = [
            item["cubin_sha256"]
            for item in row.get("stages", {}).get("compile", {}).get("binaries", [])
        ]
        old_cubins = [
            item["cubin_sha256"]
            for item in old["targets"]["cuda"]["lower"]["binaries"]
        ]
        row.update(
            {
                "stem": source["stem"],
                "frontend": source["frontend"],
                "exact_accepted_gate2_input": source[
                    "exact_accepted_gate2_input"
                ],
                "task30_cubin_digests_exact": new_cubins == old_cubins,
                "rvv_object_bytes": old["targets"]["rvv"]["lower"]["candidate"][
                    "object_size_bytes"
                ],
                "rvv_instruction_count": old["targets"]["rvv"]["lower"][
                    "candidate"
                ]["rvv_instruction_count"],
                "rvv_object_sha256": old["targets"]["rvv"]["lower"][
                    "object_sha256"
                ],
            }
        )
        joined.append(row)
    by_frontend = {
        frontend: [row for row in joined if row["frontend"] == frontend]
        for frontend in ("tilelang", "cuda")
    }
    frontend_metrics = {
        frontend: _frontend_metrics(rows) for frontend, rows in by_frontend.items()
    }
    pair_rows = []
    for stem in sorted(pair_analysis):
        fronts = {
            row["frontend"]: row for row in joined if row["stem"] == stem
        }
        if set(fronts) != {"tilelang", "cuda"}:
            raise ValueError(f"runtime pair {stem} is incomplete")
        tile = fronts["tilelang"]
        cuda = fronts["cuda"]
        tile_compile = tile["stages"]["compile"]
        cuda_compile = cuda["stages"]["compile"]
        tile_bench = tile["stages"].get("benchmark", {})
        cuda_bench = cuda["stages"].get("benchmark", {})
        tile_cubin_digests = [
            binary["cubin_sha256"] for binary in tile_compile.get("binaries", [])
        ]
        cuda_cubin_digests = [
            binary["cubin_sha256"] for binary in cuda_compile.get("binaries", [])
        ]
        compiles_pass = all(
            _status(item, "compile") == "pass" for item in fronts.values()
        )
        tile_cubin_bytes = tile_compile.get("total_cubin_size_bytes")
        cuda_cubin_bytes = cuda_compile.get("total_cubin_size_bytes")
        cubin_denominator = (
            max(tile_cubin_bytes, cuda_cubin_bytes, 1)
            if compiles_pass
            else None
        )
        rvv_denominator = max(tile["rvv_object_bytes"], cuda["rvv_object_bytes"], 1)
        pair_rows.append(
            {
                "stem": stem,
                "semantic_core_exact": bool(
                    pair_analysis[stem]["semantic_core_exact"]
                ),
                "runtime_both_pass": all(
                    _status(item, "execute_compare") == "pass"
                    for item in fronts.values()
                ),
                "benchmark_both_pass": all(
                    _status(item, "benchmark") == "pass" for item in fronts.values()
                ),
                "task30_cubin_identity_both": all(
                    item["task30_cubin_digests_exact"] for item in fronts.values()
                ),
                "cross_frontend_intent_mlir_digest_equal": (
                    tile_compile.get("intent_mlir_sha256")
                    == cuda_compile.get("intent_mlir_sha256")
                ),
                "cross_frontend_cuda_cubin_sequence_equal": (
                    tile_cubin_digests == cuda_cubin_digests
                ),
                "cross_frontend_rvv_object_digest_equal": (
                    tile["rvv_object_sha256"] == cuda["rvv_object_sha256"]
                ),
                "cross_frontend_graph_topology_equal": (
                    tile["stages"]["execute_compare"].get("graph")
                    == cuda["stages"]["execute_compare"].get("graph")
                ),
                "cross_frontend_output_signature_equal": (
                    tile["stages"]["execute_compare"].get("outputs")
                    == cuda["stages"]["execute_compare"].get("outputs")
                ),
                "cuda_kernel_count_equal": (
                    compiles_pass
                    and tile_compile.get("kernel_count")
                    == cuda_compile.get("kernel_count")
                ),
                "cuda_cubin_relative_delta": (
                    abs(tile_cubin_bytes - cuda_cubin_bytes) / cubin_denominator
                    if compiles_pass and cubin_denominator is not None
                    else None
                ),
                "rvv_object_relative_delta": abs(
                    tile["rvv_object_bytes"] - cuda["rvv_object_bytes"]
                )
                / rvv_denominator,
                "rvv_instruction_count_equal": (
                    tile["rvv_instruction_count"] == cuda["rvv_instruction_count"]
                ),
                "prepared_graph_latency_ratio_cuda_over_tilelang": (
                    float(cuda_bench["median_us"]) / float(tile_bench["median_us"])
                    if tile_bench.get("status") == cuda_bench.get("status") == "pass"
                    else None
                ),
            }
        )

    def stratum(exact: bool) -> dict[str, Any]:
        rows = [row for row in pair_rows if row["semantic_core_exact"] is exact]
        ratios = [
            row["prepared_graph_latency_ratio_cuda_over_tilelang"]
            for row in rows
            if row["prepared_graph_latency_ratio_cuda_over_tilelang"] is not None
        ]
        return {
            "pair_count": len(rows),
            "runtime_both_pass": sum(row["runtime_both_pass"] for row in rows),
            "intent_mlir_digest_equal": sum(
                row["cross_frontend_intent_mlir_digest_equal"] for row in rows
            ),
            "cuda_cubin_sequence_equal": sum(
                row["cross_frontend_cuda_cubin_sequence_equal"] for row in rows
            ),
            "rvv_object_digest_equal": sum(
                row["cross_frontend_rvv_object_digest_equal"] for row in rows
            ),
            "graph_topology_equal": sum(
                row["cross_frontend_graph_topology_equal"] for row in rows
            ),
            "output_signature_equal": sum(
                row["cross_frontend_output_signature_equal"] for row in rows
            ),
            "kernel_count_equal": sum(row["cuda_kernel_count_equal"] for row in rows),
            "rvv_instruction_count_equal": sum(
                row["rvv_instruction_count_equal"] for row in rows
            ),
            "cuda_cubin_relative_delta": _numbers(
                row["cuda_cubin_relative_delta"]
                for row in rows
                if row["cuda_cubin_relative_delta"] is not None
            ),
            "rvv_object_relative_delta": _numbers(
                row["rvv_object_relative_delta"] for row in rows
            ),
            "paired_latency_ratio_cuda_over_tilelang": {
                **_numbers(ratios),
                "geomean": _geomean(ratios),
            },
        }

    all_cubins = [
        binary["cubin_sha256"]
        for row in joined
        for binary in row.get("stages", {}).get("compile", {}).get("binaries", [])
    ]
    failures = Counter()
    for row in joined:
        for stage in (
            "decode",
            "compile",
            "input",
            "refj",
            "execute_compare",
            "benchmark",
        ):
            record = row.get("stages", {}).get(stage, {})
            if record.get("status") == "pass":
                continue
            failures[f"{stage}:{record.get('error_type', record.get('status', 'missing'))}"] += 1
    policy_case_counts = Counter(
        row["stages"]["decode"]["policy_id"] for row in joined
    )
    output_comparisons = [
        output
        for row in joined
        for output in row["stages"]["execute_compare"]["agreement"]["outputs"]
    ]
    policies = sorted({output["policy_id"] for output in output_comparisons})
    numeric_by_policy = {}
    for policy_id in policies:
        rows = [
            output for output in output_comparisons if output["policy_id"] == policy_id
        ]
        tolerance_errors = [
            output["max_abs_error"]
            for output in rows
            if output["relation_kind"] == "tolerance"
            and output["max_abs_error"] is not None
        ]
        numeric_by_policy[policy_id] = {
            "case_count": policy_case_counts[policy_id],
            "output_comparison_count": len(rows),
            "exact_output_count": sum(
                output["relation_kind"] == "exact" for output in rows
            ),
            "tolerance_output_count": len(tolerance_errors),
            "max_abs_error": _numbers(tolerance_errors),
            "max_observed_abs_error": max(tolerance_errors, default=None),
        }
    machine_records = {
        _canonical_bytes(row["machine"]): row["machine"] for row in joined
    }
    if len(machine_records) != 1:
        raise ValueError("Task31 runtime entries do not share one execution environment")
    execution_environment = next(iter(machine_records.values()))
    content: dict[str, Any] = {
        "protocol": SUMMARY_PROTOCOL,
        "denominator": {
            "same_origin_pairs": 31,
            "frontend_artifacts": 62,
            "tilelang_artifacts": 31,
            "cuda_artifacts": 31,
            "primary20_used": False,
        },
        "runtime_correctness": {
            "exact_gate2_q_reconstructed": sum(
                row["exact_accepted_gate2_input"] for row in joined
            ),
            "refj_pass": sum(_status(row, "refj") == "pass" for row in joined),
            "cuda_target_refj_agreement": sum(
                _status(row, "execute_compare") == "pass" for row in joined
            ),
            "task30_cubin_digest_identity": sum(
                row["task30_cubin_digests_exact"] for row in joined
            ),
            "benchmark_pass": sum(
                _status(row, "benchmark") == "pass" for row in joined
            ),
        },
        "frontend_metrics": frontend_metrics,
        "pair_metrics": {
            "runtime_both_pass": sum(row["runtime_both_pass"] for row in pair_rows),
            "benchmark_both_pass": sum(
                row["benchmark_both_pass"] for row in pair_rows
            ),
            "task30_cubin_identity_both": sum(
                row["task30_cubin_identity_both"] for row in pair_rows
            ),
            "cross_frontend_intent_mlir_digest_equal": sum(
                row["cross_frontend_intent_mlir_digest_equal"] for row in pair_rows
            ),
            "cross_frontend_cuda_cubin_sequence_equal": sum(
                row["cross_frontend_cuda_cubin_sequence_equal"] for row in pair_rows
            ),
            "cross_frontend_rvv_object_digest_equal": sum(
                row["cross_frontend_rvv_object_digest_equal"] for row in pair_rows
            ),
            "cross_frontend_graph_topology_equal": sum(
                row["cross_frontend_graph_topology_equal"] for row in pair_rows
            ),
            "cross_frontend_output_signature_equal": sum(
                row["cross_frontend_output_signature_equal"] for row in pair_rows
            ),
            "cuda_kernel_count_equal": sum(
                row["cuda_kernel_count_equal"] for row in pair_rows
            ),
            "rvv_instruction_count_equal": sum(
                row["rvv_instruction_count_equal"] for row in pair_rows
            ),
            "semantic_core_exact": stratum(True),
            "semantic_core_nonexact": stratum(False),
        },
        "numeric_agreement": {
            "case_policy_counts": dict(sorted(policy_case_counts.items())),
            "output_comparison_count": len(output_comparisons),
            "all_outputs_pass": sum(output["passed"] for output in output_comparisons),
            "by_policy": numeric_by_policy,
        },
        "execution_environment": execution_environment,
        "benchmark_protocol": {
            **runtime["benchmark_windows"],
            "case_timeout_s": runtime["case_timeout_s"],
            "independent_process_per_artifact": True,
            "prepared_no_allocation_no_sync": True,
            "gpu_clocks_pinned": False,
        },
        "artifact_diversity": {
            "cuda_cubin_records": len(all_cubins),
            "unique_cuda_cubin_digests": len(set(all_cubins)),
            "rvv_object_records": len(joined),
            "unique_rvv_object_digests": len(
                {row["rvv_object_sha256"] for row in joined}
            ),
        },
        "failure_taxonomy": dict(sorted(failures.items())),
        "pair_rows": pair_rows,
        "claim_boundary": {
            "runtime_edge": "CUDA target versus canonical RefJ on the exact Task28 Gate2 Q",
            "source_executed_in_task31": False,
            "replay_executed_in_task31": False,
            "target_binary_recompiled_from_exact_task30_snapshot": True,
            "task30_binary_digest_identity_checked": True,
            "benchmark_is_absolute_prepared_graph_latency": True,
            "source_or_replay_speedup_claimed": False,
            "paired_latency_speedup_claimed": False,
            "timing_is_descriptive_not_inferential": True,
            "rvv_runtime_executed": False,
            "compile_ms_is_serial_pipeline_wall_time_not_kernel_latency": True,
        },
        "runtime_input_digest": runtime["input_digest"],
        "runtime_result_digest": runtime["result_digest"],
        "provenance_digest": provenance["provenance_digest"],
        "task30_backend_result_digest": task30["result_digest"],
    }
    content["summary_digest"] = _digest(content)
    _write(SUMMARY_JSON, content)
    _write_markdown(content)
    return content


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _write_markdown(result: Mapping[str, Any]) -> None:
    runtime = result["runtime_correctness"]
    fm = result["frontend_metrics"]
    pm = result["pair_metrics"]
    lines = [
        "# Task31 multi-frontend runtime and artifact characterization",
        "",
        "- denominator: 31 same-origin pairs / 62 accepted frontend artifacts",
        "- Q: exact Task28 Gate2 input digest reconstructed for every artifact",
        "- runtime edge: recompiled Task30 CUDA cubin graph versus canonical RefJ",
        "- Source / Replay / pi / explicit frontend label are unavailable to runtime_consume.py",
        "",
        "## Runtime closure",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        f"| Exact accepted Gate2 Q reconstructed | {runtime['exact_gate2_q_reconstructed']}/62 |",
        f"| RefJ execution | {runtime['refj_pass']}/62 |",
        f"| CUDA target vs RefJ agreement | {runtime['cuda_target_refj_agreement']}/62 |",
        f"| Recompiled cubin digests identical to Task30 | {runtime['task30_cubin_digest_identity']}/62 |",
        f"| Prepared graph benchmark available | {runtime['benchmark_pass']}/62 |",
        f"| Both frontends runtime-close per pair | {pm['runtime_both_pass']}/31 |",
        "",
        "## Per-frontend backend/runtime characterization",
        "",
        "| Frontend | N | Serial compile median / p90 (ms) | Kernels median | Cubin bytes median | Prepared graph median / p90 (us) | RVV object median (B) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for frontend in ("tilelang", "cuda"):
        row = fm[frontend]
        compile_ms = row["serial_compile_ms"]
        kernels = row["cuda_kernel_count"]
        cubin = row["cuda_total_cubin_bytes"]
        latency = row["prepared_graph_median_us"]
        rvv = row["rvv_object_bytes"]
        lines.append(
            f"| {frontend} | 31 | {_fmt(compile_ms['median'])} / {_fmt(compile_ms['p90'])} | "
            f"{_fmt(kernels['median'])} | {_fmt(cubin['median'], 1)} | "
            f"{_fmt(latency['median'])} / {_fmt(latency['p90'])} | "
            f"{_fmt(rvv['median'], 1)} |"
        )
    lines.extend(
        [
            "",
            "Compile time is serial compiler-pipeline wall time; prepared-graph latency uses `triton.testing.do_bench` on an allocation-free/no-sync whole graph. Artifacts run in independent processes without pinned GPU clocks, so timings are descriptive distributions only: neither a Source/Replay baseline nor a paired frontend speedup.",
            "",
            "## Pair stability by J correspondence",
            "",
            "| Stratum | Pairs | Runtime both | CUDA kernel-count equal | RVV instruction-count equal | Median cubin-size rel. delta | Median RVV-object rel. delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("semantic_core_exact", "Semantic core exact"),
        ("semantic_core_nonexact", "Semantic core non-exact"),
    ):
        row = pm[name]
        lines.append(
            f"| {label} | {row['pair_count']} | {row['runtime_both_pass']}/{row['pair_count']} | "
            f"{row['kernel_count_equal']}/{row['pair_count']} | "
            f"{row['rvv_instruction_count_equal']}/{row['pair_count']} | "
            f"{_fmt(row['cuda_cubin_relative_delta']['median'])} | "
            f"{_fmt(row['rvv_object_relative_delta']['median'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-frontend generated-artifact identity",
            "",
            "| Stratum | Pairs | Intent MLIR digest equal | CUDA cubin sequence equal | RVV object digest equal | Runtime graph topology equal | Output signature equal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("semantic_core_exact", "Semantic core exact"),
        ("semantic_core_nonexact", "Semantic core non-exact"),
    ):
        row = pm[name]
        denominator = row["pair_count"]
        lines.append(
            f"| {label} | {denominator} | "
            f"{row['intent_mlir_digest_equal']}/{denominator} | "
            f"{row['cuda_cubin_sequence_equal']}/{denominator} | "
            f"{row['rvv_object_digest_equal']}/{denominator} | "
            f"{row['graph_topology_equal']}/{denominator} | "
            f"{row['output_signature_equal']}/{denominator} |"
        )
    numeric = result["numeric_agreement"]
    lines.extend(
        [
            "",
            "## Numeric agreement receipts",
            "",
            f"- output comparisons: {numeric['all_outputs_pass']}/{numeric['output_comparison_count']} pass",
            "",
            "| Policy | Cases | Output comparisons | Exact outputs | Tolerance outputs | Max observed absolute error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_id, row in numeric["by_policy"].items():
        lines.append(
            f"| `{policy_id}` | {row['case_count']} | "
            f"{row['output_comparison_count']} | {row['exact_output_count']} | "
            f"{row['tolerance_output_count']} | "
            f"{_fmt(row['max_observed_abs_error'], 9)} |"
        )
    diversity = result["artifact_diversity"]
    lines.extend(
        [
            "",
            "## Artifact diversity",
            "",
            f"- CUDA cubin records: {diversity['cuda_cubin_records']}; unique digests: {diversity['unique_cuda_cubin_digests']}",
            f"- RVV objects: {diversity['rvv_object_records']}; unique digests: {diversity['unique_rvv_object_digests']}",
            "",
            "## Failure taxonomy",
            "",
        ]
    )
    if result["failure_taxonomy"]:
        lines.extend(
            f"- `{name}`: {count}"
            for name, count in result["failure_taxonomy"].items()
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Per-pair runtime status",
            "",
            "| Stem | Core exact | Runtime both | Task30 recompile stable | Cross-frontend target artifacts equal | CUDA kernels equal | RVV instructions equal |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in result["pair_rows"]:
        yes = lambda value: "yes" if value else "no"
        lines.append(
            f"| `{row['stem']}` | {yes(row['semantic_core_exact'])} | "
            f"{yes(row['runtime_both_pass'])} | {yes(row['task30_cubin_identity_both'])} | "
            f"{yes(row['cross_frontend_intent_mlir_digest_equal'] and row['cross_frontend_cuda_cubin_sequence_equal'] and row['cross_frontend_rvv_object_digest_equal'])} | "
            f"{yes(row['cuda_kernel_count_equal'])} | {yes(row['rvv_instruction_count_equal'])} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("export", "runtime", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--repeat-ms", type=int, default=30)
    parser.add_argument("--case-timeout-s", type=int, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.case_timeout_s <= 0:
        raise ValueError("case timeout must be positive")
    if args.phase in {"export", "all"}:
        manifest = export_inputs()
        print(
            json.dumps(
                {
                    "exported": manifest["entry_count"],
                    "input_digest": manifest["input_digest"],
                },
                sort_keys=True,
            )
        )
    if args.phase in {"runtime", "all"}:
        runtime = run_runtime(
            benchmark=not args.no_benchmark,
            warmup_ms=args.warmup_ms,
            repeat_ms=args.repeat_ms,
            case_timeout_s=args.case_timeout_s,
            limit=args.limit,
        )
        print(json.dumps(runtime["stage_status_counts"], sort_keys=True))
    if args.phase in {"summarize", "all"}:
        result = summarize()
        print(
            json.dumps(
                {
                    "runtime_correctness": result["runtime_correctness"],
                    "pair_runtime": result["pair_metrics"]["runtime_both_pass"],
                    "failure_taxonomy": result["failure_taxonomy"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
