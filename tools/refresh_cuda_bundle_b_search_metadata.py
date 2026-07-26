#!/usr/bin/env python3
"""Attach current B-search metadata to an unchanged precompiled CUDA bundle.

Every candidate id, pass argument, schedule, cubin digest, M0 digest, target,
and candidate universe is cross-checked before metadata is refreshed.  This is
safe when only search receipt metadata changed and avoids pretending that an
unchanged cubin was recompiled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from intentir.backend.cuda.search import (
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.ir.production_snapshot import ProductionSnapshot


BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported CUDA bundle protocol")
    chip = bundle["target"]["chip"]
    refreshed = 0
    for entry in bundle["kernels"]:
        snapshot = ProductionSnapshot.from_dict(entry["snapshot"])
        bindings = entry["materialization"].get("constexpr_bindings", {})
        materialized = materialize_j(
            snapshot.j,
            entry["extents"],
            constexpr_bindings=bindings,
        )
        if (
            materialized.mlir_digest
            != entry["materialization"]["intent_mlir_sha256"]
        ):
            raise AssertionError("metadata refresh changed M0")
        target = BackendTarget("cuda", chip)
        state = attach_c(materialized, snapshot.c, target)
        unseeded = attach_c(materialized, None, target)
        current = {
            item.candidate_id: item
            for item in enumerate_cuda_candidates(state)
            if not item.generic
        }
        unseeded_ids = {
            item.candidate_id
            for item in enumerate_cuda_candidates(unseeded)
            if not item.generic
        }
        if set(current) != unseeded_ids:
            raise AssertionError("C changed candidate universe during refresh")
        for record in entry["candidates"]:
            old = record["candidate"]
            candidate_id = old["candidate_id"]
            if candidate_id not in current:
                raise AssertionError(
                    f"unknown historical candidate {candidate_id}"
                )
            new = current[candidate_id].to_dict()
            for field in (
                "candidate_id",
                "family",
                "fast_path_id",
                "pass_argument",
                "schedule",
                "c_order_score",
            ):
                if old[field] != new[field]:
                    raise AssertionError(
                        f"candidate {candidate_id} changed field {field}"
                    )
            record["candidate"] = new
            refreshed += 1
        contract = derive_cuda_contract_view(state)
        if entry["contract_view"]["boundary_sha256"] != contract.boundary_sha256:
            raise AssertionError("metadata refresh changed B projection")
        entry["contract_view"] = contract.to_dict()
        entry["metadata_refresh"] = {
            "kind": "search-metadata-only",
            "candidate_binary_recompiled": False,
            "candidate_binary_fields_unchanged": True,
            "m0_unchanged": True,
            "b_search_metadata_added": True,
        }
    bundle["metadata_refresh"] = {
        "created_unix_ns": time.time_ns(),
        "source": str(args.input),
        "refreshed_candidate_count": refreshed,
        "candidate_binaries_recompiled": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"refreshed {refreshed} candidates into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
