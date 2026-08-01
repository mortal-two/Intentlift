#!/usr/bin/env python3
"""Compile one anonymous exact J once and execute five pre-registered Q cases."""

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

from intentir.backend.cuda.benchmark import PreparedGraphLaunch
from intentir.backend.cuda.pipeline import lower_to_cuda
from intentir.backend.materialize import materialize_j
from intentir.ir.production_snapshot import ProductionSnapshot
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.orchestrate import gate2_input_digest
from intentir.verify.gate2.policy import ComparisonPolicy
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


INPUT_PROTOCOL = "intentir.experiment.anonymous-j-multiq-input.v1"
RESULT_PROTOCOL = "intentir.experiment.anonymous-j-multiq-result.v1"
ROOT_FIELDS = frozenset(
    {
        "protocol",
        "artifact_count",
        "q_per_artifact",
        "entries",
        "schedule_digest",
        "manifest_digest",
    }
)
ENTRY_FIELDS = frozenset(
    {
        "run_id",
        "snapshot",
        "concrete_extents",
        "comparison_policy",
        "q_cases",
        "entry_digest",
    }
)
Q_FIELDS = frozenset(
    {"q_id", "input_file", "input_file_sha256", "input_digest"}
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
        "seed",
        "q_kind",
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


def _without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    content = dict(value)
    content.pop(field, None)
    return content


def _validated_entry(manifest_path: Path, run_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != ROOT_FIELDS:
        raise ValueError("multi-Q manifest root violates the exact whitelist")
    if manifest["protocol"] != INPUT_PROTOCOL:
        raise ValueError("unsupported multi-Q input protocol")
    if manifest["manifest_digest"] != _digest(
        _without(manifest, "manifest_digest")
    ):
        raise ValueError("multi-Q manifest digest mismatch")
    entries = manifest["entries"]
    if (
        not isinstance(entries, list)
        or manifest["artifact_count"] != len(entries)
        or manifest["q_per_artifact"] != 5
    ):
        raise ValueError("multi-Q manifest denominator mismatch")
    selected = None
    seen_runs: set[str] = set()
    seen_q: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or set(raw) != ENTRY_FIELDS:
            raise ValueError(f"multi-Q entry {index} violates the exact whitelist")
        if set(raw) & FORBIDDEN_FIELDS:
            raise ValueError(f"multi-Q entry {index} contains forbidden provenance")
        if raw["entry_digest"] != _digest(_without(raw, "entry_digest")):
            raise ValueError(f"multi-Q entry {index} digest mismatch")
        current = raw["run_id"]
        if not isinstance(current, str) or _SHA256.fullmatch(current) is None:
            raise ValueError(f"multi-Q entry {index} has an invalid run_id")
        if current in seen_runs:
            raise ValueError("multi-Q run_id must be unique")
        seen_runs.add(current)
        q_cases = raw["q_cases"]
        if not isinstance(q_cases, list) or len(q_cases) != manifest["q_per_artifact"]:
            raise ValueError(f"multi-Q entry {index} does not contain five Q cases")
        for q_index, q_case in enumerate(q_cases):
            if not isinstance(q_case, dict) or set(q_case) != Q_FIELDS:
                raise ValueError(
                    f"multi-Q entry {index} Q {q_index} violates the whitelist"
                )
            if set(q_case) & FORBIDDEN_FIELDS:
                raise ValueError("multi-Q Q case contains forbidden provenance")
            q_id = q_case["q_id"]
            if not isinstance(q_id, str) or _SHA256.fullmatch(q_id) is None:
                raise ValueError("multi-Q q_id must be one SHA256")
            if q_id in seen_q:
                raise ValueError("multi-Q q_id must be globally unique")
            seen_q.add(q_id)
        if current == run_id:
            selected = raw
    if selected is None:
        raise ValueError(f"anonymous multi-Q run_id {run_id!r} is absent")
    return selected


def _load_values(
    manifest_path: Path, q_case: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    filename = q_case["input_file"]
    if filename != f"{q_case['q_id']}.npz" or Path(filename).name != filename:
        raise ValueError("multi-Q input filename is not content-addressed by q_id")
    path = manifest_path.parent / "multiq_values_current" / filename
    if _file_digest(path) != q_case["input_file_sha256"]:
        raise ValueError("multi-Q NPZ SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if gate2_input_digest(values) != q_case["input_digest"]:
        raise ValueError("multi-Q logical input digest mismatch")
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
        raise ValueError(f"multi-Q CUDA tensor dtype {dtype!r} is unsupported")
    return torch.as_tensor(value, device="cuda", dtype=_TORCH_DTYPES[key]).contiguous()


def _shape_records(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.name,
        }
        for name, value in sorted(values.items())
    }


def _failure(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "fail",
        "stage": stage,
        "error_type": type(error).__name__,
        "diagnostic": str(error),
    }


def _pass(stage: str, **fields: Any) -> dict[str, Any]:
    return {"status": "pass", "stage": stage, **fields}


def _run(manifest_path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter_ns()
    result: dict[str, Any] = {
        "protocol": RESULT_PROTOCOL,
        "run_id": entry["run_id"],
        "stages": {},
        "q_cases": [],
    }
    try:
        snapshot = ProductionSnapshot.from_dict(entry["snapshot"])
        extents = {str(key): int(value) for key, value in entry["concrete_extents"].items()}
        policy = ComparisonPolicy.from_json(entry["comparison_policy"])
        input_names = tuple(item.name for item in snapshot.j.intent.inputs)
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
        result["stages"]["decode"] = _pass(
            "decode",
            snapshot_digest=snapshot.snapshot_digest,
            j_digest=snapshot.j_digest,
            c_digest=snapshot.c_digest,
            policy_id=policy.policy_id,
            q_case_count=len(entry["q_cases"]),
        )
    except Exception as error:
        result["stages"]["decode"] = _failure("decode", error)
        return result

    # The only target compilation happens before the consumer opens any Q file.
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
            raise RuntimeError("multi-Q compilation did not produce ELF cubins")
        result["stages"]["compile"] = _pass(
            "compile",
            elapsed_ms=(time.perf_counter_ns() - compile_started) / 1_000_000.0,
            program_compile_count=1,
            q_values_loaded_before_compile=False,
            materialization_form=materialized.form,
            intent_mlir_sha256=hashlib.sha256(
                materialized.intent_mlir.encode("utf-8")
            ).hexdigest(),
            kernel_count=len(program.kernels),
            total_cubin_size_bytes=sum(item["cubin_size_bytes"] for item in binaries),
            binaries=binaries,
        )
    except Exception as error:
        result["stages"]["compile"] = _failure("compile", error)
        return result

    prepared_graph_bind_count = 0
    for q_case in entry["q_cases"]:
        q_started = time.perf_counter_ns()
        q_result: dict[str, Any] = {"q_id": q_case["q_id"], "stages": {}}
        try:
            values = _load_values(manifest_path, q_case)
            if set(values) != set(input_names):
                raise ValueError("multi-Q values do not match canonical J inputs")
            q_result["stages"]["input"] = _pass(
                "input",
                input_digest=q_case["input_digest"],
                inputs=_shape_records(values),
            )
        except Exception as error:
            q_result["stages"]["input"] = _failure("input", error)
            q_result["total_elapsed_ms"] = (
                time.perf_counter_ns() - q_started
            ) / 1_000_000.0
            result["q_cases"].append(q_result)
            continue

        ref_started = time.perf_counter_ns()
        reference = ScalarLoweringRunner().run(spec, values)
        if not reference.ok:
            q_result["stages"]["refj"] = {
                "status": "fail",
                "stage": "refj",
                "unsupported": reference.unsupported,
                "diagnostic": reference.error,
            }
            q_result["total_elapsed_ms"] = (
                time.perf_counter_ns() - q_started
            ) / 1_000_000.0
            result["q_cases"].append(q_result)
            continue
        q_result["stages"]["refj"] = _pass(
            "refj",
            elapsed_ms=(time.perf_counter_ns() - ref_started) / 1_000_000.0,
            outputs=_shape_records(reference.outputs),
        )

        execute_started = time.perf_counter_ns()
        try:
            tensors_by_name = {
                tensor.name: _torch_input(values[tensor.name], tensor.dtype)
                for tensor in snapshot.j.intent.inputs
            }
            prepared = PreparedGraphLaunch(
                program,
                materialized.intent_mlir,
                [tensors_by_name[item.name] for item in snapshot.j.intent.inputs],
            )
            prepared_graph_bind_count += 1
            prepared.thunk()()
            torch.cuda.synchronize()
            target_outputs = {
                name: tensor.detach().cpu().numpy().copy()
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
            q_result["stages"]["execute_compare"] = {
                "status": "pass" if agreement.passed else "fail",
                "stage": "execute_compare",
                "elapsed_ms": (
                    time.perf_counter_ns() - execute_started
                ) / 1_000_000.0,
                "graph": prepared.record.to_dict(),
                "outputs": _shape_records(target_outputs),
                "agreement": agreement.to_json(),
            }
        except Exception as error:
            q_result["stages"]["execute_compare"] = _failure(
                "execute_compare", error
            )
        q_result["total_elapsed_ms"] = (
            time.perf_counter_ns() - q_started
        ) / 1_000_000.0
        result["q_cases"].append(q_result)

    all_q_pass = (
        len(result["q_cases"]) == len(entry["q_cases"])
        and all(
            item.get("stages", {}).get("execute_compare", {}).get("status")
            == "pass"
            for item in result["q_cases"]
        )
    )
    result["stages"]["multiq"] = {
        "status": "pass" if all_q_pass else "fail",
        "stage": "multiq",
        "q_case_count": len(result["q_cases"]),
        "q_pass_count": sum(
            item.get("stages", {}).get("execute_compare", {}).get("status")
            == "pass"
            for item in result["q_cases"]
        ),
        "prepared_graph_bind_count": prepared_graph_bind_count,
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
    args = parser.parse_args()
    entry = _validated_entry(args.manifest, args.run_id)
    result = _run(args.manifest, entry)
    result["result_digest"] = _digest(result)
    _write(args.output, result)
    passed = result.get("stages", {}).get("multiq", {}).get("status") == "pass"
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "multiq_pass": passed,
                "q_pass_count": result.get("stages", {})
                .get("multiq", {})
                .get("q_pass_count", 0),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
