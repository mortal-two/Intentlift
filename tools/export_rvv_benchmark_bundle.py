#!/usr/bin/env python3
"""Compile accepted production J snapshots into native RVV object bundles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from intentir.backend.cuda.portable_bundle import encode_values
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    constexpr_input_bindings,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.compile import (
    RiscvLoweringCandidate,
    RiscvSchedule,
    RiscvTransformationState,
    compile_riscv_object,
)
from intentir.backend.riscv.contract import derive_riscv_contract_view
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import ResultRelationKind
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v2"
TARGET = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))


def _shape(shape: list[int | str], dims: dict[str, int]) -> list[int]:
    return [int(dims[item]) if isinstance(item, str) else int(item) for item in shape]


def _candidates(
    enable_b_transformations: bool,
) -> list[RiscvLoweringCandidate]:
    candidates = [
        RiscvLoweringCandidate(
            RiscvTransformationState(fuse),
            RiscvSchedule(opt, vectorize, 128),
        )
        for fuse in (False, True)
        for opt in (2, 3)
        for vectorize in (False, True)
        if enable_b_transformations or not fuse
    ]
    return sorted(
        candidates,
        key=lambda item: (
            not item.transformation.fuse_elementwise,
            not item.schedule.vectorize,
            -item.schedule.opt_level,
        ),
    )


def _output_rules(snapshot: Any) -> dict[str, Any]:
    relations = {
        item.output: item.kind for item in snapshot.j.boundary.result_relations
    }
    result = {}
    for output in snapshot.j.intent.outputs:
        kind = relations.get(output.name, ResultRelationKind.EXACT)
        rule = DEFAULT_GATE2_POLICY.rule_for(kind, output.dtype)
        result[output.name] = {
            "comparison_class": kind.value,
            "dtype": rule.dtype,
            "tolerance": (
                rule.tolerance.to_json()
                if rule.tolerance is not None
                else None
            ),
        }
    return result


def _entry(name: str, snapshot_dir: Path) -> dict[str, Any]:
    snapshot, path = load_production_snapshot(snapshot_dir, name)
    if snapshot.c is None:
        raise ValueError("RVV bundle requires frozen accepted C")
    prepared = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS[PRODUCTION_CASE_IDS[name]]
    )
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
    state = attach_c(materialized, snapshot.c, TARGET)
    contract = derive_riscv_contract_view(state)
    candidates = []
    rejected = []
    for candidate in _candidates(enable_b_transformations=True):
        started = time.perf_counter_ns()
        try:
            compiled = compile_riscv_object(
                state,
                candidate,
                capabilities=contract.capabilities,
            )
        except Exception as error:
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_dict(),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        candidates.append(
            {
                "candidate": compiled.to_record(),
                "compile_ms": (
                    time.perf_counter_ns() - started
                )
                / 1_000_000.0,
                "object_base64": base64.b64encode(
                    compiled.object_bytes
                ).decode("ascii"),
            }
        )
    if not candidates:
        reasons = list(
            dict.fromkeys(item["reason"] for item in rejected)
        )
        raise RuntimeError(
            f"{name} has no compiled RVV candidates: "
            + " | ".join(reasons[:3])
        )
    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "snapshot_path": str(path),
        "snapshot_digest": snapshot.snapshot_digest,
        "j_digest": snapshot.j_digest,
        "c_digest": snapshot.c_digest,
        "materialization": materialized.to_record(),
        "contract_view": contract.to_dict(),
        "extents": dims,
        "input_order": list(materialized.runtime_input_names),
        "input_bindings": case.refeval.input_binding_map,
        "output_bindings": case.refeval.output_binding_map,
        "input_specs": {
            item.name: {
                "dtype": item.dtype,
                "shape": _shape(item.shape, dims),
            }
            for item in snapshot.j.intent.inputs
            if item.name in materialized.runtime_input_names
        },
        "output_specs": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": _shape(item.shape, dims),
            }
            for item in snapshot.j.intent.outputs
        ],
        "output_rules": _output_rules(snapshot),
        "comparison_policy": DEFAULT_GATE2_POLICY.to_json(),
        "values": encode_values(case.values),
        "reference_outputs": encode_values(reference.outputs),
        "candidate_order": [
            item["candidate"]["candidate_id"] for item in candidates
        ],
        "schedule_only_candidate_ids": [
            item["candidate"]["candidate_id"]
            for item in candidates
            if not item["candidate"]["transformation"]["b_transformations"]
        ],
        "candidates": candidates,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time_ns()
    entries = []
    failures = []
    for name in args.kernels:
        try:
            entry = _entry(name, args.snapshot_dir)
            entries.append(entry)
            print(
                f"{name}: {len(entry['candidates'])} native RVV objects",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "kernel": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"{name}: FAILED: {type(error).__name__}: {error}",
                flush=True,
            )
    payload = {
        "protocol": BUNDLE_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": TARGET.to_dict(),
        "true_mlir_pipeline": True,
        "python_target_kernel_emission": False,
        "summary": {
            "requested": len(args.kernels),
            "compiled": len(entries),
            "failed": len(failures),
        },
        "kernels": entries,
        "failures": failures,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {args.out} sha256="
        f"{hashlib.sha256(encoded.encode()).hexdigest()}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
