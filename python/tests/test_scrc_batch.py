from __future__ import annotations

import json

from intentir.verify.scrc_batch import (
    BatchCaseResult,
    BatchClassification,
    SequentialSCRCBatchRunner,
    dry_run_executor,
)
from intentir.verify.workload_manifest import build_workload_manifest


def test_sequential_batch_isolates_one_case_failure_and_preserves_order() -> None:
    entries = build_workload_manifest().entries[:3]
    calls: list[str] = []

    def executor(entry):
        calls.append(entry.case_id)
        if entry is entries[1]:
            raise RuntimeError("case-local boom")
        return BatchCaseResult(
            entry.case_id,
            entry.ordinal,
            BatchClassification.READY,
            entry.representative_tier,
        )

    report = SequentialSCRCBatchRunner().run(
        entries, executor, mode="unit_test"
    )

    assert calls == [entry.case_id for entry in entries]
    assert [item.classification for item in report.cases] == [
        BatchClassification.READY,
        BatchClassification.INFRASTRUCTURE_FAILED,
        BatchClassification.READY,
    ]
    assert report.counts == {"infrastructure_failed": 1, "ready": 2}
    assert "case-local boom" in report.cases[1].diagnostic


def test_representative_dry_run_resolves_six_without_gate_execution() -> None:
    entries = build_workload_manifest().select(tier="representative")

    report = SequentialSCRCBatchRunner().run(
        entries, dry_run_executor(), mode="dry_run"
    )

    assert len(report.cases) == 6
    assert report.counts == {"ready": 6}
    assert all(item.main_status is None for item in report.cases)
    assert all(item.call_budget == {} for item in report.cases)
    assert all(item.source_file for item in report.cases)


def test_batch_report_is_json_safe_and_aggregates_only_existing_facts() -> None:
    entries = build_workload_manifest().entries[:1]
    report = SequentialSCRCBatchRunner().run(
        entries, dry_run_executor(), mode="dry_run"
    )

    payload = report.to_json()
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)

    assert payload["total"] == 1
    assert payload["status_counts"] == {"ready": 1}
    assert "source_code" not in encoded
    assert "artifact" not in encoded
