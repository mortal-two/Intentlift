#!/usr/bin/env python3
"""Run fair-budget E2/E3/E4, L0/L1, and N0-N3 CUDA ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.ablation import ExperimentArm, standard_ablation_plan
from intentir.backend.cuda.production_cases import PRODUCTION_CASE_IDS
from intentir.backend.cuda.search import PerformanceCache
from intentir.verify.gate1.compiler import KernelSource
from benchmark_production_cuda import _device_record, _one_kernel


REPORT_PROTOCOL = "intentir.production.cuda-ablation.v1"
_SEARCH_ARMS = tuple(
    arm
    for arm in ExperimentArm
    if arm not in {ExperimentArm.E0_SOURCE, ExperimentArm.E1_CANONICAL}
)


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
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "function"}:
        raise ValueError(
            f"Replay manifest entry {name!r} requires path, sha256, function"
        )
    path = Path(str(raw["path"]))
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != raw["sha256"]:
        raise ValueError(f"Replay manifest digest mismatch for {name}")
    return KernelSource.from_replay(
        payload.decode("utf-8"),
        str(raw["function"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(PRODUCTION_CASE_IDS),
        default=sorted(PRODUCTION_CASE_IDS),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[item.value for item in _SEARCH_ARMS],
        default=[item.value for item in _SEARCH_ARMS],
    )
    parser.add_argument("--candidate-budget", type=int, default=96)
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
    if args.require_backend_closure:
        missing = sorted(set(args.kernels) - set(replay_manifest))
        if missing:
            raise ValueError(
                "formal closure requires opaque Replay entries for: "
                + ", ".join(missing)
            )

    cache = PerformanceCache(args.cache)
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
                    cache=cache,
                    axes=axes,
                    fixed_lowering_path_kind=(
                        str(fixed_paths[name])
                        if not axes.search_structural_states and name in fixed_paths
                        else None
                    ),
                    qd_search_seeds=args.qd_search_seeds,
                    qd_cert_seeds=args.qd_cert_seeds,
                    replay_program=_replay_program(replay_manifest, name),
                    require_backend_closure=args.require_backend_closure,
                )
                records.append({"arm": arm.value, **record})
                performance = record["performance"]
                print(
                    f"{arm.value}/{name}: {performance['mlir_us']:.4f} us "
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
    report = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": _device_record(torch),
        "target_chip": backend_compile.CHIP,
        "candidate_budget": args.candidate_budget,
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
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
