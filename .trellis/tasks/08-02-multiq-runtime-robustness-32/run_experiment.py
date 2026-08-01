#!/usr/bin/env python3
"""Pre-register, export, execute, and summarize Task32 multi-Q robustness."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
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
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.workload_registry import prepare_registered_scrc_workload


TASK_ROOT = Path(__file__).resolve().parent
TASKS_ROOT = TASK_ROOT.parent
TASK28 = TASKS_ROOT / "08-01-j-only-replay-multifrontend-batch-28"
TASK29 = TASKS_ROOT / "08-01-tilelang-cuda-paired-j-analysis-29"
TASK31 = TASKS_ROOT / "08-02-multifrontend-runtime-characterization-31"

TASK29_RESULTS = TASK29 / "results_current.json"
TASK31_INPUT = TASK31 / "runtime_inputs_current.json"
TASK31_PROVENANCE = TASK31 / "runtime_provenance_current.json"
TASK31_RUNTIME = TASK31 / "runtime_results_current.json"
TASK31_VALUES = TASK31 / "runtime_values_current"

SCHEDULE_PATH = TASK_ROOT / "q_schedule_current.json"
VALUES_DIR = TASK_ROOT / "multiq_values_current"
INPUT_MANIFEST = TASK_ROOT / "multiq_inputs_current.json"
PROVENANCE_PATH = TASK_ROOT / "multiq_provenance_current.json"
RUNS_DIR = TASK_ROOT / "multiq_runs_current"
RUNTIME_RESULTS = TASK_ROOT / "multiq_runtime_results_current.json"
SUMMARY_JSON = TASK_ROOT / "results_current.json"
SUMMARY_MD = TASK_ROOT / "results_current.md"
CONSUMER = TASK_ROOT / "multiq_consume.py"

SCHEDULE_PROTOCOL = "intentir.experiment.multiq-schedule.v1"
INPUT_PROTOCOL = "intentir.experiment.anonymous-j-multiq-input.v1"
PROVENANCE_PROTOCOL = "intentir.experiment.multiq-provenance.v1"
RUNTIME_PROTOCOL = "intentir.experiment.anonymous-j-multiq-batch.v1"
SUMMARY_PROTOCOL = "intentir.experiment.multiq-runtime-robustness.v1"
Q_PER_ARTIFACT = 5
ROBUSTNESS_Q = 4


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
        raise TypeError(f"expected one JSON object: {path}")
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


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    content = dict(value)
    content.pop(field, None)
    return content


def _verify_digest_field(value: Mapping[str, Any], field: str) -> None:
    if value.get(field) != _digest(_without(value, field)):
        raise ValueError(f"{field} mismatch")


def _shape_records(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.name,
        }
        for name, value in sorted(values.items())
    }


def _write_deterministic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, value in sorted(values.items()):
            if not name.isidentifier():
                raise ValueError(f"multi-Q logical input name is unsafe: {name!r}")
            array = np.asarray(value)
            serialized = array if array.ndim == 0 else np.ascontiguousarray(array)
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, serialized, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _q_id(run_id: str, ordinal: int) -> str:
    return hashlib.sha256(
        f"intentir.task32.q-id.v1:{run_id}:{ordinal}".encode("utf-8")
    ).hexdigest()


def _seed(run_id: str, ordinal: int) -> int:
    raw = hashlib.sha256(
        f"intentir.task32.q-seed.v1:{run_id}:{ordinal}".encode("utf-8")
    ).digest()
    return int.from_bytes(raw[:4], "big", signed=False)


def _expected_schedule() -> dict[str, Any]:
    task31_inputs = _read(TASK31_INPUT)
    task31_provenance = _read(TASK31_PROVENANCE)
    _verify_digest_field(task31_inputs, "input_digest")
    _verify_digest_field(task31_provenance, "provenance_digest")
    inputs_by_id = {item["run_id"]: item for item in task31_inputs["entries"]}
    provenance_by_id = {
        item["run_id"]: item for item in task31_provenance["entries"]
    }
    if set(inputs_by_id) != set(provenance_by_id) or len(inputs_by_id) != 62:
        raise ValueError("Task31 does not contain the exact full-62 denominator")
    entries = []
    seen_q: set[str] = set()
    for run_id in sorted(inputs_by_id):
        q_cases = []
        seeds: set[int] = set()
        for ordinal in range(Q_PER_ARTIFACT):
            qid = _q_id(run_id, ordinal)
            if qid in seen_q:
                raise ValueError("Task32 q_id schedule collision")
            seen_q.add(qid)
            seed = None if ordinal == 0 else _seed(run_id, ordinal)
            if seed is not None and seed in seeds:
                raise ValueError("Task32 seed schedule collision within artifact")
            if seed is not None:
                seeds.add(seed)
            q_cases.append(
                {
                    "q_id": qid,
                    "ordinal": ordinal,
                    "q_kind": "accepted" if ordinal == 0 else "robustness",
                    "seed": seed,
                }
            )
        entries.append(
            {
                "run_id": run_id,
                "snapshot_digest": inputs_by_id[run_id]["snapshot"]["snapshot_digest"],
                "artifact_sha256": provenance_by_id[run_id]["artifact_sha256"],
                "q_cases": q_cases,
            }
        )
    schedule: dict[str, Any] = {
        "protocol": SCHEDULE_PROTOCOL,
        "artifact_count": len(entries),
        "q_per_artifact": Q_PER_ARTIFACT,
        "accepted_q_per_artifact": 1,
        "robustness_q_per_artifact": ROBUSTNESS_Q,
        "seed_derivation": "sha256(intentir.task32.q-seed.v1:run_id:ordinal)[:4]",
        "selection_uses_runtime_results": False,
        "entries": entries,
    }
    schedule["schedule_digest"] = _digest(schedule)
    return schedule


def create_or_verify_schedule() -> dict[str, Any]:
    expected = _expected_schedule()
    if SCHEDULE_PATH.exists():
        observed = _read(SCHEDULE_PATH)
        _verify_digest_field(observed, "schedule_digest")
        if observed != expected:
            raise ValueError("frozen Task32 Q schedule differs from deterministic schedule")
        return observed
    if any(path.exists() for path in (INPUT_MANIFEST, RUNTIME_RESULTS, SUMMARY_JSON)):
        raise RuntimeError("cannot pre-register Task32 schedule after experiment outputs exist")
    _write(SCHEDULE_PATH, expected)
    return expected


def export_inputs() -> dict[str, Any]:
    schedule = _read(SCHEDULE_PATH)
    _verify_digest_field(schedule, "schedule_digest")
    if schedule != _expected_schedule():
        raise ValueError("Task32 schedule is not the exact deterministic pre-registration")
    task31_inputs = _read(TASK31_INPUT)
    task31_provenance = _read(TASK31_PROVENANCE)
    _verify_digest_field(task31_inputs, "input_digest")
    _verify_digest_field(task31_provenance, "provenance_digest")
    inputs_by_id = {item["run_id"]: item for item in task31_inputs["entries"]}
    provenance_by_id = {
        item["run_id"]: item for item in task31_provenance["entries"]
    }
    schedule_by_id = {item["run_id"]: item for item in schedule["entries"]}
    VALUES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    provenance_entries: list[dict[str, Any]] = []
    precheck_counts: Counter[str] = Counter()

    for run_id in sorted(schedule_by_id):
        base = inputs_by_id[run_id]
        source = provenance_by_id[run_id]
        run_dir = TASK28 / source["run_dir"]
        artifact_path = run_dir / "artifact.json"
        if _file_digest(artifact_path) != source["artifact_sha256"]:
            raise ValueError(f"Task28 artifact digest drifted: {run_dir}")
        artifact = canonicalize_gate1_artifact(
            Gate1Artifact.from_json(_read(artifact_path))
        )
        if artifact.j.to_json() != base["snapshot"]["j"]:
            raise ValueError(f"Task32 artifact J differs from Task31 snapshot: {run_id}")
        prepared = prepare_registered_scrc_workload(source["case"])
        manifest_q_cases = []
        provenance_q_cases = []
        logical_digests = []
        for q_schedule in schedule_by_id[run_id]["q_cases"]:
            ordinal = int(q_schedule["ordinal"])
            qid = q_schedule["q_id"]
            if ordinal == 0:
                task31_path = TASK31_VALUES / base["input_file"]
                if _file_digest(task31_path) != base["input_file_sha256"]:
                    raise ValueError(f"Task31 accepted-Q NPZ drifted: {run_id}")
                logical_values = _load_npz(task31_path)
                dims = base["concrete_extents"]
                ref_status = "inherited_task31_pass"
                ref_error = None
            else:
                suite = build_frontend_dataset_gate2_suite(
                    prepared,
                    artifact,
                    seed=int(q_schedule["seed"]),
                    case_id=f"finite-robustness-{ordinal}",
                )
                case = suite.cases[0]
                dims = case.refeval.dims_map
                logical_values = {
                    logical: np.asarray(case.values[runtime_key])
                    for logical, runtime_key in case.refeval.input_binding_map.items()
                }
                reference = ScalarLoweringRunner().run(case.refeval, case.values)
                ref_status = (
                    "pass" if reference.ok and not reference.unsupported else "fail"
                )
                ref_error = reference.error
            if dims != base["concrete_extents"]:
                raise ValueError(f"Task32 Q dimensions differ from exact snapshot: {run_id}")
            expected_inputs = {item.name for item in artifact.j.intent.inputs}
            if set(logical_values) != expected_inputs:
                raise ValueError(f"Task32 logical values differ from J inputs: {run_id}")
            logical_digest = gate2_input_digest(logical_values)
            logical_digests.append(logical_digest)
            value_path = VALUES_DIR / f"{qid}.npz"
            _write_deterministic_npz(value_path, logical_values)
            q_manifest = {
                "q_id": qid,
                "input_file": value_path.name,
                "input_file_sha256": _file_digest(value_path),
                "input_digest": logical_digest,
            }
            manifest_q_cases.append(q_manifest)
            provenance_q_cases.append(
                {
                    **q_schedule,
                    "input_digest": logical_digest,
                    "input_file_sha256": q_manifest["input_file_sha256"],
                    "inputs": _shape_records(logical_values),
                    "refj_export_precheck": ref_status,
                    "refj_export_precheck_error": ref_error,
                }
            )
            precheck_counts[ref_status] += 1
        entry: dict[str, Any] = {
            "run_id": run_id,
            "snapshot": base["snapshot"],
            "concrete_extents": base["concrete_extents"],
            "comparison_policy": base["comparison_policy"],
            "q_cases": manifest_q_cases,
        }
        entry["entry_digest"] = _digest(entry)
        manifest_entries.append(entry)
        provenance_entries.append(
            {
                "run_id": run_id,
                "stem": source["stem"],
                "frontend": source["frontend"],
                "case": source["case"],
                "run_dir": source["run_dir"],
                "artifact_sha256": source["artifact_sha256"],
                "semantic_input_digest_count": len(set(logical_digests)),
                "all_five_inputs_distinct": len(set(logical_digests)) == Q_PER_ARTIFACT,
                "q_cases": provenance_q_cases,
                "consumer_receives_this_entry": False,
            }
        )
    if len(manifest_entries) != 62 or len(provenance_entries) != 62:
        raise ValueError("Task32 export denominator is not exact full-62")
    manifest: dict[str, Any] = {
        "protocol": INPUT_PROTOCOL,
        "artifact_count": len(manifest_entries),
        "q_per_artifact": Q_PER_ARTIFACT,
        "entries": manifest_entries,
        "schedule_digest": schedule["schedule_digest"],
    }
    manifest["manifest_digest"] = _digest(manifest)
    provenance: dict[str, Any] = {
        "protocol": PROVENANCE_PROTOCOL,
        "artifact_count": len(provenance_entries),
        "q_case_count": len(provenance_entries) * Q_PER_ARTIFACT,
        "entries": provenance_entries,
        "schedule_digest": schedule["schedule_digest"],
        "task31_input_digest": task31_inputs["input_digest"],
        "task31_provenance_digest": task31_provenance["provenance_digest"],
        "refj_export_precheck_counts": dict(sorted(precheck_counts.items())),
        "consumer_receives_this_file": False,
    }
    provenance["provenance_digest"] = _digest(provenance)
    _write(INPUT_MANIFEST, manifest)
    _write(PROVENANCE_PATH, provenance)
    return manifest


def run_runtime(*, case_timeout_s: int, limit: int | None) -> dict[str, Any]:
    manifest = _read(INPUT_MANIFEST)
    _verify_digest_field(manifest, "manifest_digest")
    entries = list(manifest["entries"])
    if limit is not None:
        entries = entries[:limit]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    driver_returncodes: Counter[str] = Counter()
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
        ]
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
                raise RuntimeError("multi-Q consumer produced no result artifact")
            result = _read(temporary_output)
            _verify_digest_field(result, "result_digest")
            if result.get("run_id") != entry["run_id"]:
                raise ValueError("multi-Q consumer result run_id mismatch")
            passed = result.get("stages", {}).get("multiq", {}).get("status") == "pass"
            expected_returncode = 0 if passed else 2
            if completed.returncode != expected_returncode:
                raise RuntimeError(
                    "multi-Q return code disagrees with receipt: "
                    f"{completed.returncode} != {expected_returncode}"
                )
            driver_returncodes[str(completed.returncode)] += 1
            temporary_output.replace(output)
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
            driver_error = error
        finally:
            temporary_output.unlink(missing_ok=True)
        if driver_error is not None:
            stderr = completed.stderr if completed is not None else ""
            result = {
                "protocol": "intentir.experiment.multiq-driver-failure.v1",
                "run_id": entry["run_id"],
                "stages": {
                    "driver": {
                        "status": "fail",
                        "stage": "driver",
                        "error_type": type(driver_error).__name__,
                        "diagnostic": f"{driver_error}\n{stderr[-4000:]}",
                    }
                },
                "q_cases": [],
            }
            result["result_digest"] = _digest(result)
            _write(output, result)
            driver_returncodes["driver_failure"] += 1
        results.append(result)
        multiq = result.get("stages", {}).get("multiq", {})
        print(
            f"[{index}/{len(entries)}] {entry['run_id'][:12]} "
            f"multiq={multiq.get('status', 'missing')} "
            f"q={multiq.get('q_pass_count', 0)}/{Q_PER_ARTIFACT}",
            flush=True,
        )
    combined: dict[str, Any] = {
        "protocol": RUNTIME_PROTOCOL,
        "manifest_digest": manifest["manifest_digest"],
        "schedule_digest": manifest["schedule_digest"],
        "artifact_count": len(results),
        "q_per_artifact": Q_PER_ARTIFACT,
        "case_timeout_s": case_timeout_s,
        "driver_returncode_counts": dict(sorted(driver_returncodes.items())),
        "entries": results,
    }
    combined["result_digest"] = _digest(combined)
    _write(RUNTIME_RESULTS, combined)
    return combined


def _numbers(values: Iterable[float]) -> dict[str, float | int | None]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": max(data),
    }


def summarize() -> dict[str, Any]:
    schedule = _read(SCHEDULE_PATH)
    manifest = _read(INPUT_MANIFEST)
    provenance = _read(PROVENANCE_PATH)
    runtime = _read(RUNTIME_RESULTS)
    task29 = _read(TASK29_RESULTS)
    task31_runtime = _read(TASK31_RUNTIME)
    for value, field in (
        (schedule, "schedule_digest"),
        (manifest, "manifest_digest"),
        (provenance, "provenance_digest"),
        (runtime, "result_digest"),
        (task31_runtime, "result_digest"),
    ):
        _verify_digest_field(value, field)
    runtime_by_id = {item["run_id"]: item for item in runtime["entries"]}
    provenance_by_id = {item["run_id"]: item for item in provenance["entries"]}
    task31_by_id = {item["run_id"]: item for item in task31_runtime["entries"]}
    if not (
        set(runtime_by_id) == set(provenance_by_id) == set(task31_by_id)
        and len(runtime_by_id) == 62
    ):
        raise ValueError("Task32 runtime/provenance/Task31 run sets differ")
    pair_analysis = {item["stem"]: item for item in task29["pairs"]}
    q_rows = []
    artifact_rows = []
    failures: Counter[str] = Counter()
    for run_id in sorted(runtime_by_id):
        result = runtime_by_id[run_id]
        source = provenance_by_id[run_id]
        q_provenance = {item["q_id"]: item for item in source["q_cases"]}
        old_cubins = [
            item["cubin_sha256"]
            for item in task31_by_id[run_id]["stages"]["compile"]["binaries"]
        ]
        compile_stage = result.get("stages", {}).get("compile", {})
        new_cubins = [
            item["cubin_sha256"] for item in compile_stage.get("binaries", [])
        ]
        artifact_pass = result.get("stages", {}).get("multiq", {}).get("status") == "pass"
        artifact_rows.append(
            {
                "run_id": run_id,
                "stem": source["stem"],
                "frontend": source["frontend"],
                "semantic_core_exact": bool(
                    pair_analysis[source["stem"]]["semantic_core_exact"]
                ),
                "artifact_pass": artifact_pass,
                "compile_once": (
                    compile_stage.get("status") == "pass"
                    and compile_stage.get("program_compile_count") == 1
                    and compile_stage.get("q_values_loaded_before_compile") is False
                ),
                "task31_cubin_sequence_equal": new_cubins == old_cubins,
                "all_five_inputs_distinct": source["all_five_inputs_distinct"],
            }
        )
        if compile_stage.get("status") != "pass":
            failures[
                f"compile:{compile_stage.get('error_type', compile_stage.get('status', 'missing'))}"
            ] += 1
        for q_result in result.get("q_cases", []):
            qid = q_result["q_id"]
            q_source = q_provenance[qid]
            execute = q_result.get("stages", {}).get("execute_compare", {})
            refj = q_result.get("stages", {}).get("refj", {})
            q_pass = execute.get("status") == "pass"
            row = {
                "run_id": run_id,
                "q_id": qid,
                "ordinal": q_source["ordinal"],
                "q_kind": q_source["q_kind"],
                "frontend": source["frontend"],
                "stem": source["stem"],
                "semantic_core_exact": bool(
                    pair_analysis[source["stem"]]["semantic_core_exact"]
                ),
                "policy_id": result.get("stages", {})
                .get("decode", {})
                .get("policy_id"),
                "input_status": q_result.get("stages", {})
                .get("input", {})
                .get("status", "missing"),
                "refj_status": refj.get("status", "missing"),
                "execute_compare_status": execute.get("status", "missing"),
                "q_pass": q_pass,
                "output_receipts": execute.get("agreement", {}).get("outputs", []),
            }
            q_rows.append(row)
            for stage in ("input", "refj", "execute_compare"):
                record = q_result.get("stages", {}).get(stage, {})
                if record.get("status") != "pass":
                    failures[
                        f"{stage}:{record.get('error_type', record.get('status', 'missing'))}"
                    ] += 1
    if len(artifact_rows) != 62 or len(q_rows) != 310:
        raise ValueError("Task32 result denominator is incomplete")
    output_rows = [output for row in q_rows for output in row["output_receipts"]]

    def q_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        outputs = [output for row in rows for output in row["output_receipts"]]
        tolerance_errors = [
            output["max_abs_error"]
            for output in outputs
            if output["relation_kind"] == "tolerance"
            and output["max_abs_error"] is not None
        ]
        return {
            "q_evaluations": len(rows),
            "q_pass": sum(row["q_pass"] for row in rows),
            "output_comparisons": len(outputs),
            "output_pass": sum(output["passed"] for output in outputs),
            "exact_outputs": sum(output["relation_kind"] == "exact" for output in outputs),
            "tolerance_outputs": len(tolerance_errors),
            "max_abs_error": _numbers(tolerance_errors),
        }

    frontend_metrics = {
        frontend: q_group([row for row in q_rows if row["frontend"] == frontend])
        for frontend in ("tilelang", "cuda")
    }
    policy_metrics = {
        policy: q_group([row for row in q_rows if row["policy_id"] == policy])
        for policy in sorted({row["policy_id"] for row in q_rows})
    }
    q_kind_metrics = {
        kind: q_group([row for row in q_rows if row["q_kind"] == kind])
        for kind in ("accepted", "robustness")
    }
    pair_rows = []
    for stem in sorted(pair_analysis):
        artifacts = [row for row in artifact_rows if row["stem"] == stem]
        if len(artifacts) != 2 or {row["frontend"] for row in artifacts} != {
            "tilelang",
            "cuda",
        }:
            raise ValueError(f"Task32 pair {stem} is incomplete")
        pair_rows.append(
            {
                "stem": stem,
                "semantic_core_exact": bool(
                    pair_analysis[stem]["semantic_core_exact"]
                ),
                "both_artifacts_all_five_pass": all(
                    row["artifact_pass"] for row in artifacts
                ),
                "both_compile_once": all(row["compile_once"] for row in artifacts),
                "both_task31_cubin_stable": all(
                    row["task31_cubin_sequence_equal"] for row in artifacts
                ),
            }
        )

    def pair_stratum(exact: bool) -> dict[str, Any]:
        rows = [row for row in pair_rows if row["semantic_core_exact"] is exact]
        q_subset = [row for row in q_rows if row["semantic_core_exact"] is exact]
        return {
            "pair_count": len(rows),
            "pairs_all_ten_evaluations_pass": sum(
                row["both_artifacts_all_five_pass"] for row in rows
            ),
            "artifact_count": len(rows) * 2,
            **q_group(q_subset),
        }

    machine_records = {
        _canonical_bytes(item["machine"]): item["machine"]
        for item in runtime_by_id.values()
        if "machine" in item
    }
    if len(machine_records) != 1:
        raise ValueError("Task32 runtime entries do not share one machine")
    content: dict[str, Any] = {
        "protocol": SUMMARY_PROTOCOL,
        "denominator": {
            "same_origin_pairs": 31,
            "frontend_artifacts": 62,
            "q_per_artifact": Q_PER_ARTIFACT,
            "accepted_q_evaluations": 62,
            "new_robustness_q_evaluations": 248,
            "artifact_q_evaluations": 310,
        },
        "runtime_correctness": {
            "artifact_compile_once_receipts": sum(
                row["compile_once"] for row in artifact_rows
            ),
            "task31_cubin_sequence_identity": sum(
                row["task31_cubin_sequence_equal"] for row in artifact_rows
            ),
            "artifacts_all_five_q_pass": sum(
                row["artifact_pass"] for row in artifact_rows
            ),
            "pairs_all_ten_evaluations_pass": sum(
                row["both_artifacts_all_five_pass"] for row in pair_rows
            ),
            "q_evaluations_pass": sum(row["q_pass"] for row in q_rows),
            "output_comparisons_pass": sum(
                output["passed"] for output in output_rows
            ),
        },
        "input_diversity": {
            "artifacts_with_five_distinct_q": sum(
                row["all_five_inputs_distinct"] for row in artifact_rows
            ),
            "unique_logical_input_digests": len(
                {
                    q_case["input_digest"]
                    for entry in manifest["entries"]
                    for q_case in entry["q_cases"]
                }
            ),
        },
        "frontend_metrics": frontend_metrics,
        "policy_metrics": policy_metrics,
        "q_kind_metrics": q_kind_metrics,
        "pair_metrics": {
            "all_pairs": {
                "pair_count": len(pair_rows),
                "pairs_all_ten_evaluations_pass": sum(
                    row["both_artifacts_all_five_pass"] for row in pair_rows
                ),
            },
            "semantic_core_exact": pair_stratum(True),
            "semantic_core_nonexact": pair_stratum(False),
        },
        "failure_taxonomy": dict(sorted(failures.items())),
        "pair_rows": pair_rows,
        "execution_environment": next(iter(machine_records.values())),
        "claim_boundary": {
            "runtime_edge": "CUDA target versus canonical RefJ on one accepted plus four pre-registered value sets per exact J",
            "source_executed_in_task32": False,
            "replay_executed_in_task32": False,
            "rvv_runtime_executed": False,
            "shape_robustness_claimed": False,
            "value_robustness_within_fixed_extents": True,
            "seeds_selected_using_runtime_results": False,
            "performance_claimed": False,
        },
        "schedule_digest": schedule["schedule_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "provenance_digest": provenance["provenance_digest"],
        "runtime_result_digest": runtime["result_digest"],
        "task31_runtime_result_digest": task31_runtime["result_digest"],
    }
    content["summary_digest"] = _digest(content)
    _write(SUMMARY_JSON, content)
    _write_markdown(content)
    return content


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.9g}"


def _write_markdown(result: Mapping[str, Any]) -> None:
    denominator = result["denominator"]
    runtime = result["runtime_correctness"]
    lines = [
        "# Task32 pre-registered multi-Q runtime robustness",
        "",
        "- denominator: 31 same-origin pairs / 62 exact accepted artifacts",
        "- Q schedule: one accepted Q plus four SHA256-derived Q seeds per artifact",
        "- consumer compiles each exact J once before opening any of its five Q files",
        "- Source / Replay / pi / frontend / case / seed are unavailable to multiq_consume.py",
        "",
        "## Runtime closure",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        f"| Compile-once receipts | {runtime['artifact_compile_once_receipts']}/62 |",
        f"| Cubin sequence identical to Task31 | {runtime['task31_cubin_sequence_identity']}/62 |",
        f"| Artifacts passing all five Q | {runtime['artifacts_all_five_q_pass']}/62 |",
        f"| Pairs passing all ten frontend-Q evaluations | {runtime['pairs_all_ten_evaluations_pass']}/31 |",
        f"| Artifact-Q evaluations | {runtime['q_evaluations_pass']}/{denominator['artifact_q_evaluations']} |",
        f"| Output comparisons | {runtime['output_comparisons_pass']}/{sum(row['output_comparisons'] for row in result['policy_metrics'].values())} |",
        f"| Artifacts with five distinct logical-Q digests | {result['input_diversity']['artifacts_with_five_distinct_q']}/62 |",
        "",
        "## By Q provenance",
        "",
        "| Q set | Evaluations | Q pass | Outputs pass | Max abs error median / p95 / max |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("accepted", "robustness"):
        row = result["q_kind_metrics"][kind]
        errors = row["max_abs_error"]
        lines.append(
            f"| {kind} | {row['q_evaluations']} | {row['q_pass']} | "
            f"{row['output_pass']}/{row['output_comparisons']} | "
            f"{_fmt(errors['median'])} / {_fmt(errors['p95'])} / {_fmt(errors['max'])} |"
        )
    lines.extend(
        [
            "",
            "## By numeric policy",
            "",
            "| Policy | Q evaluations | Outputs | Exact / tolerance | Pass | Max observed abs error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, row in result["policy_metrics"].items():
        lines.append(
            f"| `{policy}` | {row['q_evaluations']} | {row['output_comparisons']} | "
            f"{row['exact_outputs']} / {row['tolerance_outputs']} | "
            f"{row['output_pass']}/{row['output_comparisons']} | "
            f"{_fmt(row['max_abs_error']['max'])} |"
        )
    lines.extend(
        [
            "",
            "## J-correspondence strata",
            "",
            "| Stratum | Pairs | Pairs all ten pass | Artifact-Q pass | Output pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("semantic_core_exact", "Semantic core exact"),
        ("semantic_core_nonexact", "Source-grounded non-exact"),
    ):
        row = result["pair_metrics"][name]
        lines.append(
            f"| {label} | {row['pair_count']} | "
            f"{row['pairs_all_ten_evaluations_pass']}/{row['pair_count']} | "
            f"{row['q_pass']}/{row['q_evaluations']} | "
            f"{row['output_pass']}/{row['output_comparisons']} |"
        )
    lines.extend(["", "## Failure taxonomy", ""])
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
            "This is fixed-extent value robustness, not shape robustness, Source/Replay execution, RVV runtime, or a performance experiment.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("schedule", "export", "runtime", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--case-timeout-s", type=int, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.case_timeout_s <= 0:
        raise ValueError("case timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.phase in {"schedule", "all"}:
        schedule = create_or_verify_schedule()
        print(
            json.dumps(
                {
                    "scheduled_artifacts": schedule["artifact_count"],
                    "q_per_artifact": schedule["q_per_artifact"],
                    "schedule_digest": schedule["schedule_digest"],
                },
                sort_keys=True,
            )
        )
    if args.phase in {"export", "all"}:
        manifest = export_inputs()
        print(
            json.dumps(
                {
                    "exported_artifacts": manifest["artifact_count"],
                    "q_per_artifact": manifest["q_per_artifact"],
                    "manifest_digest": manifest["manifest_digest"],
                },
                sort_keys=True,
            )
        )
    if args.phase in {"runtime", "all"}:
        runtime = run_runtime(
            case_timeout_s=args.case_timeout_s,
            limit=args.limit,
        )
        print(
            json.dumps(
                {
                    "artifacts": runtime["artifact_count"],
                    "driver_returncode_counts": runtime["driver_returncode_counts"],
                },
                sort_keys=True,
            )
        )
    if args.phase in {"summarize", "all"}:
        result = summarize()
        print(
            json.dumps(
                {
                    "runtime_correctness": result["runtime_correctness"],
                    "failure_taxonomy": result["failure_taxonomy"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
