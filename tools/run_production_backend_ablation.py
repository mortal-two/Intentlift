#!/usr/bin/env python3
"""Run E0/E1 baselines and fair-budget CUDA search ablations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.ablation import (
    ExperimentArm,
    standard_ablation_plan,
)
from intentir.backend.ablation_report import summarize_ablation_records
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
)
from intentir.verify.gate1.compiler import KernelSource
from benchmark_production_cuda import _device_record, _one_kernel


REPORT_PROTOCOL = "intentir.production.cuda-ablation.v2"
_ARMS = tuple(ExperimentArm)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return result


def _json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _replay_program(
    manifest: Mapping[str, Any], name: str
) -> KernelSource | None:
    raw = manifest.get(name)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not {"path", "function"}.issubset(raw):
        raise ValueError(
            f"Replay manifest entry {name!r} requires path and function"
        )
    path = Path(str(raw["path"]))
    return KernelSource.from_replay(
        path.read_text(encoding="utf-8"),
        str(raw["function"]),
    )


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
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[item.value for item in _ARMS],
        default=[item.value for item in _ARMS],
    )
    parser.add_argument("--candidate-budget", type=int, default=96)
    parser.add_argument(
        "--workload-profile",
        choices=(
            PRODUCTION_WORKLOAD_PROFILE_CLOSED,
            PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
        ),
        default=PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    )
    parser.add_argument("--fixed-path-map", type=Path)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--qd-search-seeds", type=_seeds, default=(2001, 2003))
    parser.add_argument("--qd-cert-seeds", type=_seeds, default=(3001, 3003))
    parser.add_argument("--require-backend-closure", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--timing", choices=("graph", "event"), default="graph")
    parser.add_argument("--graph-repetitions", type=int, default=100)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if set(args.qd_search_seeds) & set(args.qd_cert_seeds):
        raise ValueError("Q/D search and certification seeds must be disjoint")
    plans = {item.arm: item for item in standard_ablation_plan(args.candidate_budget)}
    selected_arms = tuple(ExperimentArm(item) for item in args.arms)
    fixed_paths = _json_object(args.fixed_path_map)
    replay_manifest = _json_object(args.replay_manifest)
    closure_arms = {
        arm for arm in selected_arms if arm is not ExperimentArm.E0_SOURCE
    }
    if args.require_backend_closure and closure_arms:
        missing = sorted(set(args.kernels) - set(replay_manifest))
        if missing:
            raise ValueError(
                "formal closure requires opaque Replay entries for: "
                + ", ".join(missing)
            )

    records: list[dict[str, Any]] = []
    started = time.time_ns()
    stop = False
    for arm in selected_arms:
        axes = plans[arm]
        for name in args.kernels:
            try:
                record = _one_kernel(
                    name=name,
                    snapshot_dir=args.snapshot_dir,
                    budget=axes.candidate_budget,
                    warmup_ms=args.warmup_ms,
                    repeat_ms=args.repeat_ms,
                    timing=args.timing,
                    graph_repetitions=args.graph_repetitions,
                    measurement_rounds=args.measurement_rounds,
                    search_prior="with-c" if axes.use_c_prior else "without-c",
                    b_search_mode=(
                        "contract-guided"
                        if axes.enable_local_b_transformations
                        else "schedule-only"
                    ),
                    axes=axes,
                    fixed_lowering_path_kind=(
                        str(fixed_paths[name])
                        if not axes.search_structural_states and name in fixed_paths
                        else None
                    ),
                    qd_search_seeds=(
                        ()
                        if arm is ExperimentArm.E0_SOURCE
                        else args.qd_search_seeds
                    ),
                    qd_cert_seeds=(
                        ()
                        if arm is ExperimentArm.E0_SOURCE
                        else args.qd_cert_seeds
                    ),
                    replay_program=_replay_program(replay_manifest, name),
                    require_backend_closure=args.require_backend_closure,
                    workload_profile=args.workload_profile,
                )
                records.append({"arm": arm.value, **record})
                performance = record["performance"]
                selected_us = performance["selected_us"]
                print(
                    f"{arm.value}/{name}: {selected_us:.4f} us "
                    f"({performance['speedup_over_triton']:.3f}x source)",
                    flush=True,
                )
            except Exception as exc:
                records.append(
                    {
                        "arm": arm.value,
                        "kernel": name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"{arm.value}/{name}: FAILED: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    stop = True
                    break
        if stop:
            break

    successful = [item for item in records if "performance" in item]
    selected_arm_names = [item.value for item in selected_arms]
    report = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": _device_record(torch),
        "target_chip": backend_compile.CHIP,
        "candidate_budget": args.candidate_budget,
        "workload_profile": args.workload_profile,
        "kernel_universe": sorted(PRODUCTION_CASE_IDS),
        "selected_kernels": sorted(args.kernels),
        "selection_scope": (
            "all_corpus"
            if set(args.kernels) == set(PRODUCTION_CASE_IDS)
            else "explicit_subset"
        ),
        "replay_manifest": {
            "path": str(args.replay_manifest) if args.replay_manifest else None,
            "indexed_kernels": sorted(replay_manifest),
            "missing_selected_kernels": sorted(
                set(args.kernels) - set(replay_manifest)
            ),
        },
        "qd": {
            "search_seeds": list(args.qd_search_seeds),
            "certification_seeds": list(args.qd_cert_seeds),
            "direct_final_edges": [
                "target_source",
                "target_replay",
                "target_ref_j",
            ],
            "formal_closure_required": args.require_backend_closure,
        },
        "arms": [plans[item].to_dict() for item in selected_arms],
        "summary": {
            "requested": len(selected_arms) * len(args.kernels),
            "successful": len(successful),
            "failed": len(records) - len(successful),
        },
        "evidence_summary": summarize_ablation_records(
            records,
            selected_kernels=args.kernels,
            kernel_universe=PRODUCTION_CASE_IDS,
            selected_arms=selected_arm_names,
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
