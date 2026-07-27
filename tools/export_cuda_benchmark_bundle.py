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
from intentir.backend.ablation import (
    ExperimentArm,
    SearchAxes,
    ablation_spec_digest,
    standard_ablation_plan,
)
from intentir.backend.cuda.benchmark import extract_single_launch
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    encode_triton_spec,
    encode_values,
)
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
from intentir.backend.cuda.search import (
    admit_cuda_candidate_paths,
    canonical_cuda_candidate,
    compile_cuda_candidate,
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
    toolchain_fingerprint,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.measurement_spec import (
    MEASUREMENT_PROTOCOL,
    measurement_spec_digest,
)
from intentir.backend.protocol import BackendTarget
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite


BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v2"


def _shape(shape: list[int | str], dims: dict[str, int]) -> list[int]:
    return [int(dims[item]) if isinstance(item, str) else item for item in shape]


def _fixed_path_for_axes(
    state: Any,
    axes: SearchAxes | None,
) -> tuple[str | None, str | None]:
    if axes is None or axes.search_structural_states:
        return None, None
    canonical = canonical_cuda_candidate(state)
    if canonical.generic:
        return "generic", None
    return canonical.lowering_path_kind, canonical.fast_path_id


def _entry(
    name: str,
    snapshot_dir: Path,
    budget: int,
    axes: SearchAxes | None = None,
    workload_profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
) -> dict[str, Any]:
    base_snapshot, path = load_production_snapshot(snapshot_dir, name)
    if base_snapshot.c is None:
        raise ValueError("portable search bundle requires frozen C")
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
    workload_cost = derive_j_workload_cost(materialized)
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
    seeded_kind, seeded_fast_path = _fixed_path_for_axes(state, axes)
    unseeded_kind, unseeded_fast_path = _fixed_path_for_axes(
        unseeded_state, axes
    )
    if (seeded_kind, seeded_fast_path) != (
        unseeded_kind,
        unseeded_fast_path,
    ):
        raise AssertionError("C changed the fixed concrete lowering path")
    seeded_universe = tuple(
        candidate
        for candidate in enumerate_cuda_candidates(
            state,
            axes=axes,
            fixed_lowering_path_kind=seeded_kind,
            fixed_fast_path_id=seeded_fast_path,
        )
        if seeded_fast_path is None or not candidate.generic
    )
    unseeded_universe = tuple(
        candidate
        for candidate in enumerate_cuda_candidates(
            unseeded_state,
            axes=axes,
            fixed_lowering_path_kind=unseeded_kind,
            fixed_fast_path_id=unseeded_fast_path,
        )
        if unseeded_fast_path is None or not candidate.generic
    )
    if {item.candidate_id for item in seeded_universe} != {
        item.candidate_id for item in unseeded_universe
    }:
        raise AssertionError("C changed the candidate universe")
    seeded_candidates, seeded_path_admission = admit_cuda_candidate_paths(
        state,
        seeded_universe,
    )
    unseeded_candidates, unseeded_path_admission = admit_cuda_candidate_paths(
        unseeded_state,
        unseeded_universe,
    )
    if {item.candidate_id for item in seeded_candidates} != {
        item.candidate_id for item in unseeded_candidates
    }:
        raise AssertionError("C changed hard path admission")

    candidates_by_id = {item.candidate_id: item for item in seeded_candidates}
    compiled: dict[str, dict[str, Any]] = {}

    def compile_once(candidate: Any) -> dict[str, Any]:
        cached = compiled.get(candidate.candidate_id)
        if cached is not None:
            return cached
        started = time.perf_counter_ns()
        try:
            program = compile_cuda_candidate(state, candidate)
            concrete_state = candidate.concrete_state(program)
            if concrete_state.mlir_digest is None:
                raise RuntimeError("CUDA candidate formed no canonical MLIR digest")
            extracted = extract_single_launch(program, materialized.intent_mlir)
            result = {
                "status": "compiled_state",
                "candidate_id": candidate.candidate_id,
                "mlir_digest": concrete_state.mlir_digest,
                "admitted": {
                    "candidate": candidate.to_dict(),
                    "compile_ms": (
                        time.perf_counter_ns() - started
                    )
                    / 1_000_000.0,
                    "lowering": program.to_record(),
                    "concrete_state": concrete_state.to_dict(),
                    "launch": extracted.to_dict(),
                },
            }
        except Exception as exc:
            result = {
                "status": "compile_or_transport_failed",
                "candidate_id": candidate.candidate_id,
                "rejection": {
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_dict(),
                    "status": "compile_or_transport_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "budget_consumed": True,
                },
            }
        compiled[candidate.candidate_id] = result
        return result

    def visit_order(ordered: tuple[Any, ...]) -> dict[str, Any]:
        seen_states: dict[str, str] = {}
        trace: list[dict[str, Any]] = []
        concrete_ids: list[str] = []
        budget_consumed = 0
        for candidate in ordered:
            if budget_consumed >= budget:
                break
            outcome = compile_once(candidate)
            if outcome["status"] != "compiled_state":
                budget_consumed += 1
                trace.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": outcome["status"],
                        "budget_consumed": True,
                    }
                )
                continue
            digest = str(outcome["mlir_digest"])
            duplicate_of = seen_states.get(digest)
            if duplicate_of is not None:
                trace.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "duplicate_state",
                        "mlir_digest": digest,
                        "duplicate_of": duplicate_of,
                        "budget_consumed": False,
                    }
                )
                continue
            seen_states[digest] = candidate.candidate_id
            budget_consumed += 1
            concrete_ids.append(candidate.candidate_id)
            trace.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "admitted_state",
                    "mlir_digest": digest,
                    "budget_consumed": True,
                }
            )
        return {
            "budget_consumed": budget_consumed,
            "trace": trace,
            "concrete_ids": concrete_ids,
        }

    with_c_visit = visit_order(seeded_candidates)
    without_c_visit = visit_order(unseeded_candidates)

    # The portable bundle carries the union of both equal-budget visits.  A
    # concrete MLIR state is compiled/transported once even when two requested
    # candidates or both orderings reach it.
    selected_ids = [
        *with_c_visit["concrete_ids"],
        *without_c_visit["concrete_ids"],
    ]
    aliases: dict[str, str] = {}
    state_owner: dict[str, str] = {}
    admitted: list[dict[str, Any]] = []
    duplicate_candidates: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        outcome = compiled[candidate_id]
        digest = str(outcome["mlir_digest"])
        owner = state_owner.get(digest)
        if owner is None:
            state_owner[digest] = candidate_id
            aliases[candidate_id] = candidate_id
            admitted.append(outcome["admitted"])
        else:
            aliases[candidate_id] = owner
            duplicate_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": candidates_by_id[candidate_id].to_dict(),
                    "status": "duplicate_state_across_search_orders",
                    "mlir_digest": digest,
                    "duplicate_of": owner,
                    "budget_consumed": False,
                }
            )

    def concrete_order(requested_ids: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for requested_id in requested_ids:
            owner = aliases[requested_id]
            if owner not in seen:
                seen.add(owner)
                result.append(owner)
        return result

    with_c_order = concrete_order(with_c_visit["concrete_ids"])
    without_c_order = concrete_order(without_c_visit["concrete_ids"])
    use_c_prior = axes.use_c_prior if axes is not None else True
    active_order = with_c_order if use_c_prior else without_c_order
    rejected_by_id = {
        candidate_id: outcome["rejection"]
        for candidate_id, outcome in compiled.items()
        if outcome["status"] == "compile_or_transport_failed"
    }
    rejected = [*rejected_by_id.values(), *duplicate_candidates]
    if not admitted:
        raise RuntimeError(f"{name} has no portable executable candidate")

    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")
    source = prepared.request.source.gate_source
    source_code, source_origin = source.read_code()
    baseline_policy = str(workload_valuation["baseline_policy"])
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "workload_profile": workload_profile,
        "baseline_policy": baseline_policy,
        "snapshot": snapshot.to_dict(),
        "snapshot_path": str(path),
        "workload_valuation": workload_valuation,
        "materialization": materialized.to_record(),
        "contract_view": contract_view.to_dict(),
        "workload_cost": workload_cost.to_record(),
        "ablation_axes": axes.to_dict() if axes is not None else None,
        "fixed_concrete_path": {
            "lowering_path_kind": seeded_kind,
            "fast_path_id": seeded_fast_path,
        },
        "search_orders": {
            "budget": budget,
            "budget_unit": "distinct post-pass MLIR state or exact failure",
            "with_c_budget_consumed": with_c_visit["budget_consumed"],
            "without_c_budget_consumed": without_c_visit["budget_consumed"],
            "with_c_attempt_trace": with_c_visit["trace"],
            "without_c_attempt_trace": without_c_visit["trace"],
            "with_c_concrete_order": with_c_order,
            "without_c_concrete_order": without_c_order,
            "active_order": "with_c" if use_c_prior else "without_c",
            "active_concrete_order": active_order,
            "transported_union_count": len(admitted),
            "candidate_universe_equal": True,
            "contract_view_equal": True,
            "hard_admission_equal": True,
            "with_c_hard_admission": list(seeded_path_admission),
            "without_c_hard_admission": list(unseeded_path_admission),
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
    parser.add_argument(
        "--arm",
        choices=[
            item.value
            for item in ExperimentArm
            if item
            not in {ExperimentArm.E0_SOURCE, ExperimentArm.E1_CANONICAL}
        ],
        help=(
            "freeze one machine-specified search arm; omitted preserves the "
            "full production candidate universe"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    axes = None
    if args.arm is not None:
        axes = next(
            item
            for item in standard_ablation_plan(args.search_budget)
            if item.arm.value == args.arm
        )
    started = time.time_ns()
    if (
        args.workload_profile == PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
        and args.kernels != ["masked_copy"]
    ):
        raise ValueError(
            "roofline-large-j-only-v1 currently requires exactly "
            "--kernels masked_copy"
        )
    entries = []
    for name in args.kernels:
        entry = _entry(
            name,
            args.snapshot_dir,
            args.search_budget,
            axes,
            args.workload_profile,
        )
        entries.append(entry)
        print(
            f"{name}: {len(entry['candidates'])} target cubins",
            flush=True,
        )
    baseline_policies = {str(entry["baseline_policy"]) for entry in entries}
    bundle = {
        "protocol": BUNDLE_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": {
            "backend": "cuda",
            "chip": backend_compile.CHIP,
            "compute_capability": list(
                cuda_compute_capability_for_chip(backend_compile.CHIP)
            ),
            "binary_format": "elf-cubin",
            "load_policy": "exact-compute-capability",
        },
        "workload_profile": args.workload_profile,
        "baseline_policy": (
            next(iter(baseline_policies))
            if len(baseline_policies) == 1
            else "mixed_by_kernel"
        ),
        "ablation_axes": axes.to_dict() if axes is not None else None,
        "experiment_spec_digest": (
            ablation_spec_digest() if axes is not None else None
        ),
        "measurement_spec_protocol": MEASUREMENT_PROTOCOL,
        "measurement_spec_digest": measurement_spec_digest(),
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
