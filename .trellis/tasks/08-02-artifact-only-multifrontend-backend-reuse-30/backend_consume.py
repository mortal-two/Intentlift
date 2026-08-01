#!/usr/bin/env python3
"""Consume anonymous production J+C snapshots through the real backends.

This process intentionally has no Task28, Dataset, Source, Replay, pi, frontend,
or case-id dependency.  Its input schema is a strict whitelist containing only
content-addressed ``ProductionSnapshot`` objects and concrete logical extents.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Mapping

from intentir.backend.cuda import compile as cuda_compile
from intentir.backend.cuda.pipeline import lower_to_cuda
from intentir.backend.cuda.search import derive_cuda_contract_view
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.compile import compile_riscv_object
from intentir.backend.riscv.contract import derive_riscv_contract_view
from intentir.backend.riscv.search import plan_riscv_search
from intentir.backend.toolchain.tools import INTENT_OPT, MLIR_OPT, MLIR_TRANSLATE
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import ProductionSnapshot


INPUT_PROTOCOL = "intentir.experiment.artifact-only-backend-input.v1"
RESULT_PROTOCOL = "intentir.experiment.artifact-only-backend-result.v1"
ENTRY_FIELDS = frozenset(
    {"run_id", "snapshot", "concrete_extents", "constexpr_bindings"}
)
ROOT_FIELDS = frozenset({"protocol", "entry_count", "entries", "input_digest"})
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "source",
        "pi",
        "replay",
        "frontend",
        "case",
        "case_id",
        "kernel_name",
        "gate1",
        "gate2",
        "winner",
        "history",
        "evidence",
    }
)
CUDA_TARGET = BackendTarget("cuda", cuda_compile.CHIP)
RVV_TARGET = BackendTarget("riscv", "rv64gcv_zfh_zvfh", ("vlen128",))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _validated_input(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != ROOT_FIELDS:
        raise ValueError(
            "backend input root must contain exactly " + ", ".join(sorted(ROOT_FIELDS))
        )
    if value["protocol"] != INPUT_PROTOCOL:
        raise ValueError(f"unsupported backend input protocol {value['protocol']!r}")
    entries = value["entries"]
    if not isinstance(entries, list) or value["entry_count"] != len(entries):
        raise ValueError("backend input entry_count does not match entries")
    content = {
        "protocol": value["protocol"],
        "entry_count": value["entry_count"],
        "entries": entries,
    }
    if value["input_digest"] != _digest(content):
        raise ValueError("backend input digest mismatch")
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or set(raw) != ENTRY_FIELDS:
            raise ValueError(
                f"backend input entry {index} must contain exactly "
                + ", ".join(sorted(ENTRY_FIELDS))
            )
        forbidden = sorted(set(raw) & FORBIDDEN_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"backend input entry {index} has forbidden fields {forbidden}")
        run_id = raw["run_id"]
        if not isinstance(run_id, str) or _SHA256.fullmatch(run_id) is None:
            raise ValueError(f"backend input entry {index} has invalid run_id")
        if run_id in seen:
            raise ValueError(f"duplicate anonymous run_id {run_id}")
        seen.add(run_id)
        snapshot = raw["snapshot"]
        if not isinstance(snapshot, dict):
            raise ValueError(f"backend input entry {index} snapshot must be an object")
        extents = raw["concrete_extents"]
        if not isinstance(extents, dict) or any(
            not isinstance(name, str)
            or _IDENTIFIER.fullmatch(name) is None
            or isinstance(extent, bool)
            or not isinstance(extent, int)
            or extent < 0
            for name, extent in extents.items()
        ):
            raise ValueError(f"backend input entry {index} has invalid concrete_extents")
        constexprs = raw["constexpr_bindings"]
        if not isinstance(constexprs, dict) or any(
            not isinstance(name, str)
            or _IDENTIFIER.fullmatch(name) is None
            or not isinstance(item, (bool, int, float))
            for name, item in constexprs.items()
        ):
            raise ValueError(f"backend input entry {index} has invalid constexpr_bindings")
        snapshot_digest = snapshot.get("snapshot_digest")
        expected_run_id = _digest(
            {
                "snapshot_digest": snapshot_digest,
                "concrete_extents": extents,
                "constexpr_bindings": constexprs,
            }
        )
        if run_id != expected_run_id:
            raise ValueError(f"backend input entry {index} run_id is not content derived")
    return value


def _failure(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "fail",
        "stage": stage,
        "error_type": type(error).__name__,
        "diagnostic": str(error),
    }


def _pass(stage: str, **fields: Any) -> dict[str, Any]:
    return {"status": "pass", "stage": stage, **fields}


def _cuda_lower(materialized: Any, snapshot: ProductionSnapshot) -> dict[str, Any]:
    state = attach_c(materialized, snapshot.c, CUDA_TARGET)
    if state.erase_c() != materialized.intent_mlir:
        raise AssertionError("CUDA Attach_C violated Erase_C")
    contract_started = time.perf_counter_ns()
    contract = derive_cuda_contract_view(state)
    legal = _pass(
        "cuda_legal",
        elapsed_ms=(time.perf_counter_ns() - contract_started) / 1_000_000.0,
        contract_version=contract.version,
        consumed_count=len(contract.consumed),
        capability_count=len(contract.capabilities),
        capabilities=list(contract.capabilities),
        c_prior_count=len(state.priors),
        ignored_c_slot_count=len(state.ignored_slots),
        erase_c_matches_base=True,
    )
    lower_started = time.perf_counter_ns()
    try:
        program = lower_to_cuda(materialized.intent_mlir, allow_specialized=False)
        program_record = program.to_record()
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
        lower = _pass(
            "cuda_lower",
            elapsed_ms=(time.perf_counter_ns() - lower_started) / 1_000_000.0,
            pipeline_id=program.pipeline_id,
            selection=program_record["selection"],
            kernel_count=len(program.kernels),
            binaries=binaries,
            runtime_executed=False,
            qd_certified=False,
        )
    except Exception as error:
        lower = _failure("cuda_lower", error)
    return {"legal": legal, "lower": lower}


def _rvv_lower(materialized: Any, snapshot: ProductionSnapshot) -> dict[str, Any]:
    state = attach_c(materialized, snapshot.c, RVV_TARGET)
    if state.erase_c() != materialized.intent_mlir:
        raise AssertionError("RVV Attach_C violated Erase_C")
    contract_started = time.perf_counter_ns()
    contract = derive_riscv_contract_view(state)
    legal = _pass(
        "rvv_legal",
        elapsed_ms=(time.perf_counter_ns() - contract_started) / 1_000_000.0,
        contract_version=contract.version,
        consumed_count=len(contract.consumed),
        capability_count=len(contract.capabilities),
        capabilities=list(contract.capabilities),
        c_prior_count=len(state.priors),
        ignored_c_slot_count=len(state.ignored_slots),
        erase_c_matches_base=True,
    )
    plan_started = time.perf_counter_ns()
    try:
        plan = plan_riscv_search(state, enable_b_transformations=False)
        candidates = {
            candidate.candidate_id: candidate for candidate in plan.universe
        }
        candidate = candidates[plan.c_fixed_candidate_id]
        plan_ms = (time.perf_counter_ns() - plan_started) / 1_000_000.0
        lower_started = time.perf_counter_ns()
        compiled = compile_riscv_object(
            state,
            candidate,
            capabilities=contract.capabilities,
        )
        compiled_record = compiled.to_record()
        if not compiled_record["schedule_realization"]["requested_equals_actual"]:
            raise RuntimeError("C-Fixed RVV schedule was not realized by MLIR")
        if not compiled.object_bytes.startswith(b"\x7fELF"):
            raise RuntimeError("RVV compiler did not produce a non-empty ELF object")
        lower = _pass(
            "rvv_lower",
            plan_elapsed_ms=plan_ms,
            elapsed_ms=(time.perf_counter_ns() - lower_started) / 1_000_000.0,
            candidate_universe_count=len(plan.universe),
            c_fixed_candidate_id=plan.c_fixed_candidate_id,
            candidate=compiled_record,
            object_sha256=hashlib.sha256(compiled.object_bytes).hexdigest(),
            object_is_elf=True,
            runtime_executed=False,
            qd_certified=False,
        )
    except Exception as error:
        lower = _failure("rvv_lower", error)
    return {"legal": legal, "lower": lower}


def _one(raw: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(raw["run_id"])
    result: dict[str, Any] = {"run_id": run_id, "targets": {}}
    started = time.perf_counter_ns()
    try:
        snapshot = ProductionSnapshot.from_dict(raw["snapshot"])
        result["decode"] = _pass(
            "decode",
            snapshot_schema=snapshot.schema_version,
            snapshot_digest=snapshot.snapshot_digest,
            j_digest=snapshot.j_digest,
            c_digest=snapshot.c_digest,
        )
    except Exception as error:
        result["decode"] = _failure("decode", error)
        result["emit"] = {"status": "skipped", "stage": "emit"}
        result["total_elapsed_ms"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000.0
        return result
    emit_started = time.perf_counter_ns()
    try:
        materialized = materialize_j(
            snapshot.j,
            raw["concrete_extents"],
            constexpr_bindings=raw["constexpr_bindings"],
        )
        try:
            workload_cost: dict[str, Any] = {
                "status": "pass",
                "record": derive_j_workload_cost(materialized).to_record(),
            }
        except Exception as error:
            workload_cost = _failure("workload_cost", error)
        result["emit"] = _pass(
            "emit",
            elapsed_ms=(time.perf_counter_ns() - emit_started) / 1_000_000.0,
            materializer_version=materialized.materializer_version,
            materialization_form=materialized.form,
            intent_mlir_sha256=hashlib.sha256(
                materialized.intent_mlir.encode("utf-8")
            ).hexdigest(),
            intent_mlir_bytes=len(materialized.intent_mlir.encode("utf-8")),
            intent_mlir_lines=len(materialized.intent_mlir.splitlines()),
            concrete_extents=materialized.extents,
            consumed_structure_count=len(materialized.consumed_structures),
            workload_cost=workload_cost,
        )
    except Exception as error:
        result["emit"] = _failure("emit", error)
        result["total_elapsed_ms"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000.0
        return result

    for target_name, consumer in (("cuda", _cuda_lower), ("rvv", _rvv_lower)):
        try:
            result["targets"][target_name] = consumer(materialized, snapshot)
        except Exception as error:
            target_result = result["targets"].setdefault(target_name, {})
            if "legal" not in target_result:
                target_result["legal"] = _failure(f"{target_name}_legal_or_attach", error)
                target_result["lower"] = {
                    "status": "skipped",
                    "stage": f"{target_name}_lower",
                }
            else:
                target_result["lower"] = _failure(f"{target_name}_lower", error)
    result["total_elapsed_ms"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000.0
    return result


def _status(entry: Mapping[str, Any], path: tuple[str, ...]) -> str:
    current: Any = entry
    for item in path:
        if not isinstance(current, Mapping):
            return "missing"
        current = current.get(item)
    return str(current) if current is not None else "missing"


def _aggregate(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    paths = {
        "decode": ("decode", "status"),
        "emit": ("emit", "status"),
        "cuda_legal": ("targets", "cuda", "legal", "status"),
        "cuda_lower": ("targets", "cuda", "lower", "status"),
        "rvv_legal": ("targets", "rvv", "legal", "status"),
        "rvv_lower": ("targets", "rvv", "lower", "status"),
    }
    counts: dict[str, Any] = {}
    for name, path in paths.items():
        statuses: dict[str, int] = {}
        for entry in entries:
            status = _status(entry, path)
            statuses[status] = statuses.get(status, 0) + 1
        counts[name] = dict(sorted(statuses.items()))
    fully_lowered = sum(
        all(_status(entry, path) == "pass" for path in paths.values())
        for entry in entries
    )
    return {
        "entry_count": len(entries),
        "stage_status_counts": counts,
        "fully_lowered_both_targets": fully_lowered,
    }


def _machine_record(workers: int) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "workers": workers,
        "intent_opt": INTENT_OPT,
        "mlir_opt": MLIR_OPT,
        "mlir_translate": MLIR_TRANSLATE,
        "cuda_chip": cuda_compile.CHIP,
        "cuda_bin": cuda_compile.CUDA_BIN,
        "rvv_target": RVV_TARGET.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    bundle = _validated_input(args.input)
    indexed = list(enumerate(bundle["entries"]))
    results: list[dict[str, Any] | None] = [None] * len(indexed)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_one, raw): index for index, raw in indexed}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    exact_results = [entry for entry in results if entry is not None]
    if len(exact_results) != len(indexed):
        raise AssertionError("backend worker omitted an entry")
    content: dict[str, Any] = {
        "protocol": RESULT_PROTOCOL,
        "input_protocol": bundle["protocol"],
        "input_digest": bundle["input_digest"],
        "entry_count": len(exact_results),
        "policy": {
            "backend_input_fields_exact": sorted(ENTRY_FIELDS),
            "production_snapshot_fields_exact": [
                "c",
                "c_digest",
                "j",
                "j_digest",
                "schema_version",
                "snapshot_digest",
            ],
            "forbidden_fields": sorted(FORBIDDEN_INPUT_FIELDS),
            "source_or_certification_evidence_available_to_backend": False,
            "pi_available_to_backend": False,
            "explicit_frontend_label_or_case_identity_available_to_backend": False,
            "historical_winner_available_to_backend": False,
            "cuda_lower_policy": "generic materialized state; no search, execution, or Q/D",
            "rvv_lower_policy": "C-Fixed candidate from canonical schedule universe; no execution or Q/D",
        },
        "machine": _machine_record(args.workers),
        "entries": exact_results,
        "aggregate": _aggregate(exact_results),
    }
    content["result_digest"] = _digest(content)
    _write(args.output, content)
    print(json.dumps(content["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
