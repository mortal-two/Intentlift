#!/usr/bin/env python3
"""Audit Layer-B admission and C-prior decoupling on frozen production J+C.

This is a compiler experiment, not another correctness gate.  J=A+B remains
immutable in both arms.  The only intervention is Attach_C versus Erase_C;
both arms enumerate registered MLIR pass candidates, run complete Layer-B
admission, and compile the bounded candidate prefix to target cubins.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    constexpr_input_bindings,
)
from intentir.backend.cuda.search import (
    CudaContractError,
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
    search_cuda,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import AliasKind, AliasSpec
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


REPORT_PROTOCOL = "intentir.production.cuda-b-c-search-ablation.v1"


def _candidate_ids(
    state: Any,
    *,
    enable_b_transformations: bool = True,
) -> list[str]:
    return [
        item.candidate_id
        for item in enumerate_cuda_candidates(
            state,
            enable_b_transformations=enable_b_transformations,
        )
        if not item.generic
    ]


def _unsupported_b_control(
    seeded: Any,
    unseeded: Any,
) -> dict[str, Any]:
    """Prove that C cannot authorize an unsupported Layer-B relation."""

    tensors = list(
        dict.fromkeys(
            access.tensor for access in seeded.base.j.boundary.accesses
        )
    )
    if len(tensors) < 2:
        raise AssertionError("unsupported-B control requires two tensors")
    alias = AliasSpec(tensors[0], tensors[1], AliasKind.MAY_ALIAS)
    arms: dict[str, Any] = {}
    for label, state in (("with_c", seeded), ("without_c", unseeded)):
        state.base.j.boundary.aliases.append(alias)
        try:
            try:
                enumerate_cuda_candidates(state)
            except CudaContractError as error:
                arms[label] = {
                    "rejected_before_codegen": True,
                    "reason": str(error),
                }
            else:
                raise AssertionError(
                    f"{label} accepted unsupported may_alias relation"
                )
        finally:
            state.base.j.boundary.aliases.pop()
    return {
        "intervention": {
            "kind": "may_alias",
            "lhs": tensors[0],
            "rhs": tensors[1],
        },
        "with_c": arms["with_c"],
        "without_c": arms["without_c"],
        "both_rejected_before_codegen": True,
    }


def _one(name: str, snapshot_dir: Path, budget: int) -> dict[str, Any]:
    snapshot, snapshot_path = load_production_snapshot(snapshot_dir, name)
    if snapshot.c is None:
        raise ValueError("C ablation requires the accepted optional C object")
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
    base = materialize_j(
        snapshot.j,
        dims,
        constexpr_bindings=constexpr_bindings,
    )
    target = BackendTarget("cuda", backend_compile.CHIP)
    seeded = attach_c(base, snapshot.c, target)
    unseeded = attach_c(base, None, target)
    seeded_contract = derive_cuda_contract_view(seeded)
    unseeded_contract = derive_cuda_contract_view(unseeded)
    if seeded_contract.digest != unseeded_contract.digest:
        raise AssertionError("ContractView changed when C was erased")

    seeded_order = _candidate_ids(seeded)
    unseeded_order = _candidate_ids(unseeded)
    if set(seeded_order) != set(unseeded_order):
        raise AssertionError("C changed the legal candidate universe")
    seeded_search = search_cuda(
        seeded,
        budget=budget,
        include_generic=False,
    )
    unseeded_search = search_cuda(
        unseeded,
        budget=budget,
        include_generic=False,
    )
    schedule_only_order = _candidate_ids(
        seeded,
        enable_b_transformations=False,
    )
    schedule_only_unseeded_order = _candidate_ids(
        unseeded,
        enable_b_transformations=False,
    )
    if set(schedule_only_order) != set(schedule_only_unseeded_order):
        raise AssertionError("C changed the L0 candidate universe")
    if not set(schedule_only_order) <= set(seeded_order):
        raise AssertionError("L0 local-transform-disabled set is not nested inside L1")
    schedule_only_search = search_cuda(
        seeded,
        budget=budget,
        include_generic=False,
        enable_b_transformations=False,
    )
    unsupported_b_control = _unsupported_b_control(seeded, unseeded)
    prefix_seeded = seeded_order[:budget]
    prefix_unseeded = unseeded_order[:budget]
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "snapshot_path": str(snapshot_path),
        "snapshot_digest": snapshot.snapshot_digest,
        "extents": dims,
        "materialization": base.to_record(),
        "contract_view": seeded_contract.to_dict(),
        "invariants": {
            "j_a_plus_b_unchanged": True,
            "contract_view_equal_after_erase_c": True,
            "candidate_universe_equal_after_erase_c": True,
            "c_used_for_legality": False,
            "b_used_for_legality": True,
        },
        "finite_budget": {
            "budget": budget,
            "with_c_prefix": prefix_seeded,
            "without_c_prefix": prefix_unseeded,
            "prefix_overlap": len(set(prefix_seeded) & set(prefix_unseeded)),
            "order_changed": seeded_order != unseeded_order,
        },
        "b_search_ablation": {
            "fixed": "accepted J=A+B, C, ContractView, target, compiler",
            "l0_control": "fixed structural reach; local transformations disabled",
            "l1_intervention": "enable B-conditioned local MLIR transformations",
            "contract_view_digest": seeded_contract.digest,
            "schedule_only_candidate_count": len(schedule_only_order),
            "contract_guided_candidate_count": len(seeded_order),
            "b_transformation_candidate_count": len(
                set(seeded_order) - set(schedule_only_order)
            ),
            "schedule_only_is_nested": True,
            "schedule_only": schedule_only_search.to_record(),
        },
        "unsupported_b_control": unsupported_b_control,
        "with_c": seeded_search.to_record(),
        "without_c": unseeded_search.to_record(),
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
    parser.add_argument("--search-budget", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.search_budget <= 0:
        raise ValueError("search-budget must be positive")
    started = time.time_ns()
    kernels = []
    for name in args.kernels:
        record = _one(name, args.snapshot_dir, args.search_budget)
        kernels.append(record)
        seeded = record["with_c"]["counts"]
        unseeded = record["without_c"]["counts"]
        print(
            f"{name}: B={len(record['contract_view']['consumed'])} facts; "
            f"with-C admitted={seeded['admitted']}, "
            f"without-C admitted={unseeded['admitted']}",
            flush=True,
        )
    report = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": {"backend": "cuda", "chip": backend_compile.CHIP},
        "experiment": {
            "fixed": "accepted J=A+B, target, compiler, candidate universe, budget",
            "intervention": "Attach_C versus Erase_C",
            "b_search_intervention": (
                "L0 versus L1 B-conditioned local transformations; "
                "J=A+B and ContractView remain fixed"
            ),
            "selection": "compile/admission only; no correctness-by-testing",
        },
        "kernels": kernels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
