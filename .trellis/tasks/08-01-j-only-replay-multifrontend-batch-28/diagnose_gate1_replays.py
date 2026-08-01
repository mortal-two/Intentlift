"""Run rejected, compiled Replays through diagnostic three-edge execution.

This deliberately does not upgrade or bypass Gate1 evidence.  Results are saved
as diagnostic attribution only, so a numerically passing Replay remains formally
unaccepted until a fresh/rechecked Gate1 and the ordinary Gate2 path both pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

from manual_g1_artifacts import CASE_BUILDERS as G1_CASE_BUILDERS
from manual_g2_artifacts import (
    CASE_BUILDERS as G2_CASE_BUILDERS,
    CASE_IDS as G2_CASE_IDS,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompiler,
    CudaKernelSource,
)
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.tilelang_compiler import TileLangKernelSource
from intentir.verify.gate2.native import CudaRunner, TileLangRunner
from intentir.verify.gate2.orchestrate import Gate2Runner
from intentir.verify.frontend_workloads import build_frontend_dataset_gate2_suite
from intentir.verify.workload_registry import prepare_registered_scrc_workload


TASK_ROOT = Path(__file__).resolve().parent
CASE_BUILDERS = {**G1_CASE_BUILDERS, **G2_CASE_BUILDERS}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _latest_rejected_run(case_id: str) -> Path:
    case_root = TASK_ROOT / "runs" / case_id.replace(":", "_")
    candidates: list[Path] = []
    for run_dir in case_root.glob("run-*"):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file() or not (run_dir / "replay.source").is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("gate1") == "no_closure":
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no rejected compiled Replay exists for {case_id}")
    return max(candidates, key=lambda item: int(item.name.removeprefix("run-")))


def _diagnose(case_id: str) -> dict[str, Any]:
    run_dir = _latest_rejected_run(case_id)
    try:
        prepared = prepare_registered_scrc_workload(case_id)
        artifact = Gate1Artifact.from_json(CASE_BUILDERS[case_id]())
        suite = build_frontend_dataset_gate2_suite(
            prepared,
            artifact,
            test_mode="diagnostic_after_gate1_reject_not_certificate",
        )
        code = (run_dir / "replay.source").read_text(encoding="utf-8")
        source = prepared.request.source.gate_source
        if prepared.request.frontend == "cuda":
            replay = CudaKernelSource.from_replay(code, "scrc_k0")
            source_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
            replay_runner: Any = CudaRunner(CudaCompiler(timeout=240.0))
        else:
            replay = TileLangKernelSource.from_replay(code, "scrc_k0")
            source_runner = TileLangRunner()
            replay_runner = TileLangRunner()
        result = Gate2Runner(
            source_runner=source_runner,
            replay_runner=replay_runner,
        ).run_suite(source, replay, suite)
        payload = result.to_json()
        payload["diagnostic_only"] = True
        payload["formal_acceptance"] = False
        payload["reason"] = "Gate1 rejected this exact Replay"
        _write_json(run_dir / "gate2_diagnostic.json", payload)
        return {
            "case": case_id,
            "run_dir": str(run_dir),
            "diagnostic_gate2_passed": result.passed,
            "status_counts": dict(result.status_counts),
            "formal_acceptance": False,
        }
    except Exception as exc:
        row = {
            "case": case_id,
            "run_dir": str(run_dir),
            "diagnostic_gate2_passed": False,
            "formal_acceptance": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(run_dir / "gate2_diagnostic_failure.json", row)
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASE_BUILDERS))
    parser.add_argument("--all-g2", action="store_true")
    args = parser.parse_args()
    selected = list(G2_CASE_IDS) if args.all_g2 else list(dict.fromkeys(args.case or ()))
    if not selected:
        parser.error("select --case or --all-g2")
    rows: list[dict[str, Any]] = []
    for case_id in selected:
        try:
            row = _diagnose(case_id)
        except FileNotFoundError:
            continue
        rows.append(row)
        print(json.dumps(row, sort_keys=True, ensure_ascii=False), flush=True)
    summary = {
        "protocol": "intentir.experiment.rejected-replay-gate2-diagnostic.v1",
        "diagnostic_only": True,
        "formal_acceptance": False,
        "case_count": len(rows),
        "diagnostic_pass_count": sum(
            row.get("diagnostic_gate2_passed") is True for row in rows
        ),
        "results": rows,
    }
    _write_json(TASK_ROOT / "gate2_diagnostic_latest.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
