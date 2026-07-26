#!/usr/bin/env python3
"""Run one registered Triton, TileLang, or CUDA workload through full SCRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.extraction.providers.provider import get_provider
from intentir.verify.scrc_workload_run import (
    WorkloadExecutionError,
    classify_scrc_report,
    execute_prepared_workload,
    initialize_workload_evidence,
    next_workload_run_dir,
    save_workload_failure,
    save_workload_execution,
)
from intentir.verify.workload_registry import (
    prepare_registered_scrc_workload,
    registered_scrc_case_ids,
    registered_scrc_workloads,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", help="registered real-workload case id")
    selection.add_argument(
        "--list",
        action="store_true",
        help="list registered cases without loading a provider",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument(
        "--provider-config",
        type=Path,
        default=None,
        help="provider JSON path; API keys are never copied into evidence",
    )
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--lift-backend",
        choices=("responses", "miniswe"),
        default="responses",
    )
    parser.add_argument(
        "--replay-backend",
        choices=("responses", "miniswe"),
        default="responses",
    )
    parser.add_argument(
        "--no-masked-j",
        action="store_true",
        help="diagnostic mode: skip the J-necessity masked Replay arm",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "artifacts" / "scrc-frontends",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _emit(payload: dict[str, object], *, json_mode: bool) -> None:
    if json_mode:
        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _listed_cases() -> dict[str, object]:
    rows = registered_scrc_workloads()
    cases = [
        {
            "case_id": case_id,
            "frontend": getattr(rows[case_id], "frontend", "triton"),
        }
        for case_id in registered_scrc_case_ids()
    ]
    return {"count": len(cases), "cases": cases}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        _emit(_listed_cases(), json_mode=args.json)
        return 0

    assert args.case is not None
    run_dir: Path | None = None
    try:
        prepared = prepare_registered_scrc_workload(
            args.case,
            dataset_root=args.dataset_root,
        )
        provider = get_provider(args.provider, args.provider_config)
        run_dir = next_workload_run_dir(args.run_root, args.case)
        initialize_workload_evidence(
            run_dir,
            prepared,
            provider=provider,
            effort=args.effort,
            run_masked_j=not args.no_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
        )
        execution = execute_prepared_workload(
            prepared,
            provider=provider,
            effort=args.effort,
            run_masked_j=not args.no_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
        )
        save_workload_execution(run_dir, prepared, execution)
    except Exception as exc:
        if run_dir is not None:
            runtime_error = exc if isinstance(exc, WorkloadExecutionError) else None
            try:
                save_workload_failure(
                    run_dir,
                    exc,
                    evidence=(
                        runtime_error.evidence if runtime_error is not None else None
                    ),
                    runtime=runtime_error,
                )
            except Exception as save_error:  # pragma: no cover - defensive path
                (run_dir / "failure_save_error.txt").write_text(
                    f"{type(save_error).__name__}: {save_error}\n",
                    encoding="utf-8",
                )
            payload = {
                "failure_class": "workload_infrastructure",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, WorkloadExecutionError):
                payload["evidence"] = exc.evidence
            (run_dir / "infrastructure_error.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"SCRC frontend run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    classification = classify_scrc_report(execution.report)
    payload = {
        "case_id": args.case,
        "frontend": prepared.request.frontend,
        "classification": classification.value,
        "main_status": execution.report.main_status.value,
        "evidence_path": str(run_dir),
        "lift_backend": args.lift_backend,
        "replay_backend": args.replay_backend,
    }
    _emit(payload, json_mode=args.json)
    return 0 if execution.report.main_status.value == "scrc_certified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
