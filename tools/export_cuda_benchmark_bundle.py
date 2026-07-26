#!/usr/bin/env python3
"""Compile production J snapshots into a portable target-cubin search bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.benchmark import extract_single_launch
from intentir.backend.cuda.portable_bundle import (
    encode_triton_spec,
    encode_values,
)
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    constexpr_input_bindings,
)
from intentir.backend.cuda.pipeline import lower_to_cuda
from intentir.backend.cuda.search import (
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
    toolchain_fingerprint,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v1"


def _shape(shape: list[int | str], dims: dict[str, int]) -> list[int]:
    return [int(dims[item]) if isinstance(item, str) else item for item in shape]


def _entry(name: str, snapshot_dir: Path, budget: int) -> dict[str, Any]:
    snapshot, path = load_production_snapshot(snapshot_dir, name)
    if snapshot.c is None:
        raise ValueError("portable search bundle requires frozen C")
    adapter = TRITON_WORKLOAD_ADAPTERS[PRODUCTION_CASE_IDS[name]]
    prepared = prepare_triton_workload(adapter)
    suite = build_workload_jc_suite(prepared, snapshot.j, snapshot.c)
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
        snapshot.c,
        BackendTarget("cuda", backend_compile.CHIP),
    )
    contract_view = derive_cuda_contract_view(state)
    unseeded_state = attach_c(
        materialized,
        None,
        BackendTarget("cuda", backend_compile.CHIP),
    )
    unseeded_contract = derive_cuda_contract_view(unseeded_state)
    if contract_view.digest != unseeded_contract.digest:
        raise AssertionError("ContractView changed when C was erased")
    seeded_candidates = list(enumerate_cuda_candidates(state)[1:])
    unseeded_candidates = list(enumerate_cuda_candidates(unseeded_state)[1:])
    if {item.candidate_id for item in seeded_candidates} != {
        item.candidate_id for item in unseeded_candidates
    }:
        raise AssertionError("C changed the candidate universe")
    candidates = seeded_candidates[:budget]
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        started = time.perf_counter_ns()
        try:
            program = lower_to_cuda(
                materialized.intent_mlir,
                only_fast_path=candidate.fast_path_id,
                pass_overrides={
                    candidate.fast_path_id: candidate.pass_argument
                },
                require_specialized=True,
            )
            extracted = extract_single_launch(
                program, materialized.intent_mlir
            )
            admitted.append(
                {
                    "candidate": candidate.to_dict(),
                    "compile_ms": (
                        time.perf_counter_ns() - started
                    )
                    / 1_000_000.0,
                    "lowering": program.to_record(),
                    "launch": extracted.to_dict(),
                }
            )
        except Exception as exc:
            rejected.append(
                {
                    "candidate": candidate.to_dict(),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not admitted:
        raise RuntimeError(f"{name} has no admitted specialized candidate")

    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")
    source = prepared.request.source.gate_source
    source_code, source_origin = source.read_code()
    admitted_ids = {
        item["candidate"]["candidate_id"] for item in admitted
    }
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "snapshot": snapshot.to_dict(),
        "snapshot_path": str(path),
        "materialization": materialized.to_record(),
        "contract_view": contract_view.to_dict(),
        "search_orders": {
            "budget": budget,
            "with_c_attempted": [
                item.candidate_id for item in seeded_candidates[:budget]
            ],
            "without_c_attempted": [
                item.candidate_id for item in unseeded_candidates[:budget]
            ],
            "with_c_admitted": [
                item.candidate_id
                for item in seeded_candidates
                if item.candidate_id in admitted_ids
            ],
            "without_c_admitted": [
                item.candidate_id
                for item in unseeded_candidates
                if item.candidate_id in admitted_ids
            ],
            "candidate_universe_equal": True,
            "contract_view_equal": True,
        },
        "extents": dims,
        "input_order": list(materialized.runtime_input_names),
        "output_specs": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": _shape(item.shape, dims),
            }
            for item in snapshot.j.intent.outputs
        ],
        "input_bindings": case.refeval.input_binding_map,
        "output_bindings": case.refeval.output_binding_map,
        "output_keys": list(case.output_keys),
        "values": encode_values(case.values),
        "reference_outputs": encode_values(reference.outputs),
        "source_triton": {
            "function": source.function,
            "code": source_code,
            "code_sha256": hashlib.sha256(
                source_code.encode("utf-8")
            ).hexdigest(),
            "origin_at_export": source_origin,
            "execution_spec": encode_triton_spec(case.triton),
        },
        "candidates": admitted,
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(PRODUCTION_CASE_IDS),
        default=sorted(PRODUCTION_CASE_IDS),
    )
    parser.add_argument("--search-budget", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time_ns()
    entries = []
    for name in args.kernels:
        entry = _entry(
            name,
            args.snapshot_dir,
            args.search_budget,
        )
        entries.append(entry)
        print(
            f"{name}: {len(entry['candidates'])} target cubins",
            flush=True,
        )
    bundle = {
        "protocol": BUNDLE_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": {
            "backend": "cuda",
            "chip": backend_compile.CHIP,
        },
        "toolchain_digest": toolchain_fingerprint(),
        "toolchain": {
            "intent_opt": str(backend_compile.INTENT_OPT),
            "mlir_opt": str(backend_compile.MLIR_OPT),
            "cuda_bin": backend_compile.CUDA_BIN,
        },
        "kernels": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
