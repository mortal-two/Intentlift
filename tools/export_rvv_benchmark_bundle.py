#!/usr/bin/env python3
"""Compile accepted production J snapshots into native RVV object bundles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.ablation import ABLATION_PROTOCOL, ablation_spec_digest
from intentir.backend.cuda.portable_bundle import encode_values
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
    constexpr_input_bindings,
    profile_production_snapshot,
    prepare_production_workload,
)
from intentir.backend.dev_bundle import prepare_dev_bundle_input
from intentir.backend.dev_j_variant import JDevVariant, load_j_dev_variant
from intentir.backend.dev_workload import (
    DevWorkloadConfig,
    load_dev_workload_config,
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


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v3"
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


def _entry(
    name: str,
    snapshot_dir: Path,
    workload_profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    j_dev_variant: JDevVariant | None = None,
    dev_workload: DevWorkloadConfig | None = None,
    j_dev_path: Path | None = None,
) -> dict[str, Any]:
    base_snapshot, path = load_production_snapshot(snapshot_dir, name)
    if base_snapshot.c is None:
        raise ValueError("RVV bundle requires frozen accepted C")
    if (j_dev_variant is None) != (dev_workload is None):
        raise ValueError("J_dev variant and development workload must be paired")
    j_provenance: Mapping[str, Any] | None = None
    if j_dev_variant is not None and dev_workload is not None:
        if PRODUCTION_CASE_IDS[name] != dev_workload.case_id:
            raise ValueError("development workload does not match selected kernel")
        dev_input = prepare_dev_bundle_input(
            base_snapshot,
            path,
            j_dev_variant,
            dev_workload,
            variant_path=j_dev_path,
            baseline_policy="target_refj_only",
            source_unavailable_reason="rvv_bundle_has_no_triton_source_executable",
        )
        prepared = dev_input.prepared
        snapshot = dev_input.snapshot
        workload_profile = dev_input.workload_profile
        workload_valuation = dev_input.workload_valuation
        j_provenance = dev_input.j_provenance
        path = dev_input.snapshot_path
    else:
        prepared = prepare_production_workload(name, profile=workload_profile)
        snapshot, workload_valuation = profile_production_snapshot(
            base_snapshot,
            prepared,
            workload_profile,
        )
    if snapshot.c is None:
        raise AssertionError("workload valuation erased required C")
    suite = build_workload_jc_suite(prepared, snapshot.j, snapshot.c)
    case = suite.cases[0]
    dims = case.refeval.dims_map
    if dev_workload is not None and dims != dict(dev_workload.concrete_extents):
        raise ValueError(
            "development workload extents disagree with J/Source binding: "
            f"declared {dict(dev_workload.concrete_extents)}, resolved {dims}"
        )
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
    aliases: dict[str, str] = {}
    object_groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        object_digest = item["candidate"]["object_sha256"]
        object_groups.setdefault(object_digest, []).append(item)

    requested_candidate_by_id = {
        item.candidate_id: item for item in search_plan.universe
    }
    canonical_requested_path = requested_candidate_by_id[
        search_plan.c_fixed_candidate_id
    ].transformation.lowering_path_kind

    def representative_key(item: Mapping[str, Any]) -> tuple[int, int, str]:
        candidate = item["candidate"]
        requested = candidate["requested_schedule"]
        actual = candidate["schedule"]
        path_penalty = int(
            candidate["transformation"]["lowering_path_kind"]
            != canonical_requested_path
        )
        mismatch_count = sum(
            requested[name] != actual[name]
            for name in ("parallel_tile", "reduction_tile")
        )
        return path_penalty, mismatch_count, str(candidate["candidate_id"])

    for group in object_groups.values():
        owner_item = min(group, key=representative_key)
        owner = owner_item["candidate"]["candidate_id"]
        concrete_candidates.append(owner_item)
        for item in group:
            candidate_id = item["candidate"]["candidate_id"]
            aliases[candidate_id] = owner
            if candidate_id == owner:
                continue
            duplicate_candidates.append(
                {
                    "candidate": item["candidate"],
                    "duplicate_of": owner,
                    "reason": (
                        "identical_native_object_sha256; representative preserves "
                        "the canonical path first, then minimizes requested-vs-actual "
                        "schedule mismatch"
                    ),
                }
            )
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
    schedule_only_erase_c_order = concrete_order(
        search_plan.schedule_only_erase_c_order
    )
    schedule_only_c_prior_order = concrete_order(
        search_plan.schedule_only_c_prior_order
    )
    c_fixed_candidate_id = aliases.get(search_plan.c_fixed_candidate_id)
    if c_fixed_candidate_id not in available_ids:
        c_fixed_candidate_id = next(
            candidate_id
            for candidate_id in schedule_only_c_prior_order
            if not next(
                item["candidate"]
                for item in candidates
                if item["candidate"]["candidate_id"] == candidate_id
            )["transformation"]["fuse_producers"]
        )
    candidate_by_id = {
        item["candidate"]["candidate_id"]: item["candidate"] for item in candidates
    }
    schedule_only_ids = set(schedule_only_c_prior_order)
    if schedule_only_ids != set(schedule_only_erase_c_order):
        raise AssertionError("C changed the concrete Schedule-Only universe")
    c_fixed_candidate = candidate_by_id[c_fixed_candidate_id]
    canonical_lowering_path = str(
        c_fixed_candidate["transformation"]["lowering_path_kind"]
    )
    if any(
        str(candidate_by_id[candidate_id]["transformation"]["lowering_path_kind"])
        != canonical_lowering_path
        for candidate_id in schedule_only_ids
    ):
        raise AssertionError("Schedule-Only escaped the canonical lowering path")
    concrete_universe_digest = hashlib.sha256(
        json.dumps(sorted(available_ids), separators=(",", ":")).encode()
    ).hexdigest()
    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "workload_profile": workload_profile,
        "workload_valuation": workload_valuation,
        "j_provenance": j_provenance,
        # RVV has no Triton Source executable.  The Source-availability field in
        # workload_valuation is retained for cross-target identity only.
        "baseline_policy": "target_refj_only",
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
            "schedule_only_erase_c_order": schedule_only_erase_c_order,
            "schedule_only_c_prior_order": schedule_only_c_prior_order,
            "erase_c_order": erase_c_order,
            "c_prior_order": c_prior_order,
            "c_fixed_requested_candidate_id": (
                search_plan.c_fixed_candidate_id
            ),
            "c_fixed_candidate_id": c_fixed_candidate_id,
            "c_fixed_exactly_one": True,
            "c_projection": dict(search_plan.c_projection),
            "canonical_lowering_path": canonical_lowering_path,
            "same_universe_for_erase_c_and_c_prior": True,
            "same_schedule_only_universe_for_erase_c_and_c_prior": True,
            "budget_unit": "distinct_native_object_sha256",
        },
        "candidate_order": c_prior_order,
        "schedule_only_candidate_ids": [
            candidate_id
            for candidate_id in schedule_only_c_prior_order
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
        "--j-dev-variant",
        type=Path,
        help=(
            "uncertified dev-j-variant envelope; requires exactly one kernel "
            "and --dev-workload-config"
        ),
    )
    parser.add_argument(
        "--dev-workload-config",
        type=Path,
        help=(
            "digest-pinned natural Source/Harness invocation paired with "
            "--j-dev-variant"
        ),
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(PRODUCTION_CASE_IDS),
        default=sorted(PRODUCTION_CASE_IDS),
    )
    parser.add_argument(
        "--workload-profile",
        choices=(
            PRODUCTION_WORKLOAD_PROFILE_CLOSED,
            PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
            PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
            PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
        ),
        default=PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time_ns()
    entries = []
    failures = []
    if (args.j_dev_variant is None) != (args.dev_workload_config is None):
        raise ValueError(
            "--j-dev-variant and --dev-workload-config must be supplied together"
        )
    if args.j_dev_variant is not None:
        if len(args.kernels) != 1:
            raise ValueError("development J export currently requires exactly one kernel")
        if args.workload_profile != PRODUCTION_WORKLOAD_PROFILE_CLOSED:
            raise ValueError(
                "development workload owns its shape; --workload-profile must be closed-v1"
            )
        j_dev_variant = load_j_dev_variant(args.j_dev_variant)
        dev_workload = load_dev_workload_config(args.dev_workload_config)
    else:
        j_dev_variant = None
        dev_workload = None
    if (
        args.workload_profile == PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
        and args.kernels != ["masked_copy"]
    ):
        raise ValueError(
            "roofline-large-j-only-v1 currently requires exactly "
            "--kernels masked_copy"
        )
    for name in args.kernels:
        try:
            entry = _entry(
                name,
                args.snapshot_dir,
                args.workload_profile,
                j_dev_variant,
                dev_workload,
                args.j_dev_variant,
            )
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
    baseline_policies = {str(entry["baseline_policy"]) for entry in entries}
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
        "workload_profile": (
            next(iter({entry["workload_profile"] for entry in entries}))
            if len({entry["workload_profile"] for entry in entries}) == 1
            else "mixed_by_kernel"
        ),
        "baseline_policy": (
            next(iter(baseline_policies))
            if len(baseline_policies) == 1
            else "mixed_by_kernel"
        ),
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
