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

from intentir.backend.ablation import ABLATION_PROTOCOL, ablation_spec_digest
from intentir.backend.cuda.portable_bundle import encode_values
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    constexpr_input_bindings,
    prepare_production_workload,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.measurement_spec import (
    MEASUREMENT_PROTOCOL,
    measurement_spec_digest,
)
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.compile import compile_riscv_object, toolchain_fingerprint
from intentir.backend.riscv.contract import derive_riscv_contract_view
from intentir.backend.riscv.search import plan_riscv_search
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import ResultRelationKind
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v2"
TARGET = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))


def _shape(shape: list[int | str], dims: dict[str, int]) -> list[int]:
    return [int(dims[item]) if isinstance(item, str) else int(item) for item in shape]


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
    prepared = prepare_production_workload(name)
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
    workload_cost = derive_j_workload_cost(materialized)
    search_plan = plan_riscv_search(state, enable_b_transformations=True)
    candidates = []
    rejected = []
    for candidate in search_plan.universe:
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
    concrete_candidates = []
    duplicate_candidates = []
    object_owner: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for item in candidates:
        candidate_id = item["candidate"]["candidate_id"]
        object_digest = item["candidate"]["object_sha256"]
        owner = object_owner.get(object_digest)
        if owner is not None:
            aliases[candidate_id] = owner
            duplicate_candidates.append(
                {
                    "candidate": item["candidate"],
                    "duplicate_of": owner,
                    "reason": "identical_native_object_sha256",
                }
            )
            continue
        object_owner[object_digest] = candidate_id
        aliases[candidate_id] = candidate_id
        concrete_candidates.append(item)
    candidates = concrete_candidates
    available_ids = {
        item["candidate"]["candidate_id"] for item in candidates
    }

    def concrete_order(requested_order: list[str] | tuple[str, ...]) -> list[str]:
        result = []
        seen = set()
        for requested_id in requested_order:
            candidate_id = aliases.get(requested_id)
            if (
                candidate_id is None
                or candidate_id not in available_ids
                or candidate_id in seen
            ):
                continue
            seen.add(candidate_id)
            result.append(candidate_id)
        return result

    erase_c_order = concrete_order(search_plan.erase_c_order)
    c_prior_order = concrete_order(search_plan.c_prior_order)
    frozen_c_candidate_id = aliases.get(search_plan.frozen_c_candidate_id)
    if frozen_c_candidate_id not in available_ids:
        frozen_c_candidate_id = next(
            candidate_id
            for candidate_id in erase_c_order
            if not next(
                item["candidate"]
                for item in candidates
                if item["candidate"]["candidate_id"] == candidate_id
            )["transformation"]["fuse_elementwise"]
        )
    concrete_universe_digest = hashlib.sha256(
        json.dumps(sorted(available_ids), separators=(",", ":")).encode()
    ).hexdigest()
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
        "workload_cost": workload_cost.to_record(),
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
        "search_plan": search_plan.to_record(),
        "concrete_search_plan": {
            "candidate_universe_digest": concrete_universe_digest,
            "candidate_universe_count": len(candidates),
            "erase_c_order": erase_c_order,
            "c_prior_order": c_prior_order,
            "frozen_c_requested_candidate_id": (
                search_plan.frozen_c_candidate_id
            ),
            "frozen_c_candidate_id": frozen_c_candidate_id,
            "same_universe_for_erase_c_and_c_prior": True,
            "budget_unit": "distinct_native_object_sha256",
        },
        "candidate_order": c_prior_order,
        "schedule_only_candidate_ids": [
            item["candidate"]["candidate_id"]
            for item in candidates
            if not item["candidate"]["transformation"]["b_transformations"]
        ],
        "candidates": candidates,
        "duplicate_candidates": duplicate_candidates,
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
        "toolchain_digest": toolchain_fingerprint(),
        "experiment_spec_protocol": ABLATION_PROTOCOL,
        "experiment_spec_digest": ablation_spec_digest(),
        "measurement_spec_protocol": MEASUREMENT_PROTOCOL,
        "measurement_spec_digest": measurement_spec_digest(),
        "workload_profile": PRODUCTION_WORKLOAD_PROFILE_CLOSED,
        "baseline_policy": "target_refj_only",
        "correctness_policy": DEFAULT_GATE2_POLICY.to_json(),
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
