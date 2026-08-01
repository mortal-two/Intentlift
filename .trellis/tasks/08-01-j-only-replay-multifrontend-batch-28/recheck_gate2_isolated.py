"""Re-run exact Gate1-accepted native Replays in per-case Gate2 processes.

This is a formal evidence repair for batches whose shared CUDA context was
poisoned by an earlier illegal-memory-access kernel.  It never invokes Replay,
changes J/C/pi, or exposes Gate2 feedback to an agent: the exact accepted Replay
source recorded by Gate1 is executed again in a clean spawned interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import canonical_artifact_digest

from manual_g2_artifacts import CASE_IDS as G2_CASE_IDS
from run_manual_batch import CASE_IDS, TASK_ROOT, _execute_gate2, _write_json


def _latest_gate1_closed_run(case_id: str) -> Path:
    root = TASK_ROOT / "runs" / case_id.replace(":", "_")
    candidates: list[Path] = []
    for run_dir in root.glob("run-*"):
        gate1_path = run_dir / "gate1.json"
        if not gate1_path.is_file() or not (run_dir / "replay.source").is_file():
            continue
        gate1 = json.loads(gate1_path.read_text(encoding="utf-8"))
        if gate1.get("status") == "replay_closed":
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no Gate1-closed Replay exists for {case_id}")
    return max(candidates, key=lambda item: int(item.name.removeprefix("run-")))


def _recheck(case_id: str, semaphore: threading.Semaphore) -> dict[str, Any]:
    run_dir = _latest_gate1_closed_run(case_id)
    artifact = Gate1Artifact.from_json(
        json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))
    )
    gate1_json = json.loads((run_dir / "gate1.json").read_text(encoding="utf-8"))
    if gate1_json.get("artifact_digest") != canonical_artifact_digest(artifact):
        raise ValueError(f"{case_id} Gate1 artifact digest does not bind artifact.json")
    visibility = json.loads(
        (run_dir / "pi_visibility.json").read_text(encoding="utf-8")
    )
    if visibility.get("pi_visible_to_replay") is not False:
        raise ValueError(f"{case_id} Replay evidence is not J-only")
    generation = gate1_json.get("normal", {}).get("generation")
    if not isinstance(generation, dict) or not isinstance(
        generation.get("function_name"), str
    ):
        raise ValueError(f"{case_id} Gate1 evidence lacks the accepted function name")
    replay_code = (run_dir / "replay.source").read_text(encoding="utf-8")
    gate1 = SimpleNamespace(
        normal=SimpleNamespace(
            generation=SimpleNamespace(
                code=replay_code,
                function_name=generation["function_name"],
            )
        )
    )

    old_path = run_dir / "gate2.json"
    if old_path.is_file() and not (run_dir / "gate2_pre_isolation.json").exists():
        _write_json(
            run_dir / "gate2_pre_isolation.json",
            json.loads(old_path.read_text(encoding="utf-8")),
        )
    started = time.time()
    result = _execute_gate2(case_id, artifact, gate1, semaphore)
    elapsed_ms = (time.time() - started) * 1000.0
    payload = result.to_json()
    _write_json(old_path, payload)
    _write_json(
        run_dir / "gate2_isolation.json",
        {
            "protocol": "intentir.experiment.gate2-process-isolation.v1",
            "case": case_id,
            "start_method": "spawn",
            "one_process_per_case": True,
            "exact_gate1_replay_reused": True,
            "replay_agent_calls": 0,
            "gate2_feedback_visible_to_replay": False,
            "elapsed_ms": elapsed_ms,
        },
    )

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "gate2": "gate2_pass" if result.passed else "gate2_fail",
            "gate2_passed": result.passed,
            "gate2_status_counts": result.status_counts,
            "gate2_elapsed_ms": elapsed_ms,
            "gate2_process_isolated": True,
        }
    )
    _write_json(summary_path, summary)
    return {
        "case": case_id,
        "run_dir": str(run_dir),
        "gate1": "replay_closed",
        "gate2_passed": result.passed,
        "gate2_status_counts": result.status_counts,
        "gate2_elapsed_ms": elapsed_ms,
        "process_isolated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASE_IDS))
    parser.add_argument("--all-g2", action="store_true")
    parser.add_argument("--all-authored", action="store_true")
    parser.add_argument("--gate2-workers", type=int, default=1)
    args = parser.parse_args()
    if args.gate2_workers < 1:
        parser.error("--gate2-workers must be positive")
    if args.all_authored:
        selected = list(CASE_IDS)
    elif args.all_g2:
        selected = list(G2_CASE_IDS)
    else:
        selected = list(dict.fromkeys(args.case or ()))
    if not selected:
        parser.error("select --case, --all-g2, or --all-authored")
    semaphore = threading.Semaphore(args.gate2_workers)
    rows: list[dict[str, Any]] = []
    for case_id in selected:
        try:
            row = _recheck(case_id, semaphore)
        except FileNotFoundError:
            continue
        rows.append(row)
        print(json.dumps(row, sort_keys=True, ensure_ascii=False), flush=True)
    report = {
        "protocol": "intentir.experiment.gate2-isolated-recheck.v1",
        "selected": selected,
        "case_count": len(rows),
        "passed_count": sum(row["gate2_passed"] is True for row in rows),
        "process_isolated": True,
        "replay_agent_calls": 0,
        "results": rows,
    }
    _write_json(TASK_ROOT / "gate2_isolated_latest.json", report)
    if selected == list(G2_CASE_IDS):
        summaries = [
            json.loads(
                Path(row["run_dir"]).joinpath("summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for row in rows
        ]
        batch = {
            "protocol": "intentir.experiment.manual-batch-summary.v1",
            "selected": selected,
            "workers": 0,
            "gate2_workers": args.gate2_workers,
            "gate2_process_isolated": True,
            "lift_agent_calls": 0,
            "results": summaries,
            "accepted": [
                row["case"]
                for row in summaries
                if row.get("gate1") == "replay_closed"
                and row.get("gate2_passed") is True
            ],
        }
        batch["accepted_count"] = len(batch["accepted"])
        _write_json(TASK_ROOT / "batch_latest.json", batch)
    return 0 if report["passed_count"] == report["case_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
