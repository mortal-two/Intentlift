#!/usr/bin/env python3
"""List, validate, or dry-run the active 28-kernel SCRC workload catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.verify.scrc_batch import (
    SequentialSCRCBatchRunner,
    dry_run_executor,
)
from intentir.verify.workload_manifest import build_workload_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--validate-manifest", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--case", default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--skip-masked-j", action="store_true")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / ".trellis/tasks/07-22-scrc-real-kernel-batch-03/runs",
    )
    return parser


def _emit(value: dict, *, output: Path | None, json_mode: bool) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if json_mode or output is None:
        print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_workload_manifest()
    if args.validate_manifest and (args.case is not None or args.tier is not None):
        print(
            "--validate-manifest always validates the complete 28-item projection; "
            "--case/--tier are not accepted",
            file=sys.stderr,
        )
        return 2
    if args.run and args.case is None and args.tier is None:
        print(
            "--run requires --case or --tier; Task03 does not implicitly run all 28",
            file=sys.stderr,
        )
        return 2
    try:
        entries = manifest.select(case_id=args.case, tier=args.tier)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list:
        value = {
            "protocol": manifest.to_json()["protocol"],
            "total": len(entries),
            "entries": [entry.to_json() for entry in entries],
        }
        if args.json or args.output is not None:
            _emit(value, output=args.output, json_mode=args.json)
        else:
            for entry in entries:
                print(
                    f"{entry.ordinal:02d}\t{entry.case_id}\t"
                    f"{entry.representative_tier}\t{entry.source_module}:"
                    f"{entry.symbol}"
                )
        return 0

    if args.validate_manifest:
        validation = manifest.validate(dataset_root=args.dataset_root)
        value = validation.to_json()
        _emit(value, output=args.output, json_mode=args.json)
        return 0 if validation.passed else 2

    if args.dry_run:
        report = SequentialSCRCBatchRunner().run(
            entries,
            dry_run_executor(dataset_root=args.dataset_root),
            mode="dry_run",
        )
    else:
        # Live-only dependencies stay out of list/validate/dry-run imports.
        from intentir.extraction.providers.provider import get_provider
        from intentir.verify.scrc_batch import (
            BatchCaseResult,
            BatchClassification,
        )
        from intentir.verify.scrc_workload_run import (
            classify_scrc_report,
            execute_prepared_workload,
            initialize_workload_evidence,
            next_workload_run_dir,
            save_workload_execution,
            save_workload_failure,
            scrc_attempt_stage_counts,
            scrc_report_diagnostic,
            WorkloadExecutionError,
        )
        from intentir.verify.workload_adapters import (
            TRITON_WORKLOAD_ADAPTERS,
            prepare_triton_workload,
        )

        provider = get_provider(args.provider)
        run_root = args.run_root.expanduser().resolve()

        def live_executor(entry):
            adapter = TRITON_WORKLOAD_ADAPTERS.get(entry.case_id)
            if adapter is None:
                return BatchCaseResult(
                    entry.case_id,
                    entry.ordinal,
                    BatchClassification.HARNESS_INVALID,
                    entry.representative_tier,
                    diagnostic="no real SCRC Harness adapter is registered",
                )
            try:
                prepared = prepare_triton_workload(
                    adapter, dataset_root=args.dataset_root
                )
            except Exception as exc:
                return BatchCaseResult(
                    entry.case_id,
                    entry.ordinal,
                    BatchClassification.HARNESS_INVALID,
                    entry.representative_tier,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            run_dir = next_workload_run_dir(run_root, entry.case_id)
            evidence_path = "run-root://" + run_dir.relative_to(run_root).as_posix()
            initialize_workload_evidence(
                run_dir,
                prepared,
                provider=provider,
                effort=args.effort,
                run_masked_j=not args.skip_masked_j,
            )
            try:
                execution = execute_prepared_workload(
                    prepared,
                    provider=provider,
                    effort=args.effort,
                    run_masked_j=not args.skip_masked_j,
                )
            except Exception as exc:
                runtime = exc if isinstance(exc, WorkloadExecutionError) else None
                evidence = getattr(exc, "evidence", None)
                save_workload_failure(
                    run_dir,
                    exc,
                    evidence=evidence if isinstance(evidence, dict) else None,
                    runtime=runtime,
                )
                cause = runtime.cause if runtime is not None else exc
                return BatchCaseResult(
                    entry.case_id,
                    entry.ordinal,
                    BatchClassification.INFRASTRUCTURE_FAILED,
                    entry.representative_tier,
                    diagnostic=f"{type(cause).__name__}: {cause}",
                    source_file=prepared.source_location,
                    evidence_path=evidence_path,
                )
            save_workload_execution(run_dir, prepared, execution)
            result = execution.report
            classification = classify_scrc_report(result)
            gate2_counts = (
                result.gate2.status_counts if result.gate2 is not None else {}
            )
            return BatchCaseResult(
                entry.case_id,
                entry.ordinal,
                classification,
                entry.representative_tier,
                diagnostic=scrc_report_diagnostic(result),
                source_file=prepared.source_location,
                main_status=result.main_status.value,
                j_necessity_status=(
                    result.j_necessity_status.value
                    if result.j_necessity_status is not None
                    else None
                ),
                evidence_path=evidence_path,
                call_budget=result.call_budget.to_json(),
                details={
                    "accepted_attempt": result.accepted_attempt,
                    "attempt_stage_counts": scrc_attempt_stage_counts(result),
                    "gate2_status_counts": gate2_counts,
                    "observer_profile": entry.observer_profile,
                    "q_policy": entry.q_policy,
                },
            )

        report = SequentialSCRCBatchRunner().run(
            entries, live_executor, mode="live"
        )
    value = report.to_json()
    _emit(value, output=args.output, json_mode=args.json)
    accepted = {"ready"} if args.dry_run else {"scrc_certified"}
    return 0 if set(report.counts) <= accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
