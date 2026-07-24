from __future__ import annotations

from dataclasses import asdict
import json
import threading

import pytest

from intentir.verify.scrc_batch import (
    BatchCaseResult,
    BatchClassification,
    ConcurrentSCRCBatchRunner,
    MAX_CASE_WORKERS,
    SCRC_BATCH_PROTOCOL,
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
    assert payload["protocol"] == "intentir.scrc.batch.v3"
    assert payload["protocol"] == SCRC_BATCH_PROTOCOL
    assert payload["scheduling"] == {}
    assert "source_code" not in encoded
    assert "artifact" not in encoded


def test_concurrent_batch_overlaps_cases_but_reports_registry_order() -> None:
    registry_entries = build_workload_manifest().entries[:3]
    entries = (
        registry_entries[2],
        registry_entries[0],
        registry_entries[1],
    )
    barrier = threading.Barrier(len(entries) + 1, timeout=5)
    release = [threading.Event() for _entry in entries]
    returned = [threading.Event() for _entry in entries]
    completion_order: list[int] = []

    def coordinate_reverse_completion() -> None:
        barrier.wait()
        for index in reversed(range(len(entries))):
            release[index].set()
            assert returned[index].wait(timeout=5)

    coordinator = threading.Thread(target=coordinate_reverse_completion)
    coordinator.start()

    def executor(entry):
        index = entries.index(entry)
        barrier.wait()
        assert release[index].wait(timeout=5)
        completion_order.append(entry.ordinal)
        returned[index].set()
        return BatchCaseResult(
            entry.case_id,
            entry.ordinal,
            BatchClassification.READY,
            entry.representative_tier,
        )

    report = ConcurrentSCRCBatchRunner(case_workers=3).run(
        entries,
        executor,
        mode="unit_test",
    )
    coordinator.join(timeout=5)

    assert not coordinator.is_alive()
    assert completion_order == [entry.ordinal for entry in reversed(entries)]
    assert [item.ordinal for item in report.cases] == sorted(
        entry.ordinal for entry in entries
    )


def test_concurrent_batch_isolates_exception_and_wrong_case_result() -> None:
    entries = build_workload_manifest().entries[:4]

    def executor(entry):
        if entry is entries[0]:
            raise RuntimeError("isolated boom")
        if entry is entries[1]:
            return BatchCaseResult(
                entries[2].case_id,
                entry.ordinal,
                BatchClassification.READY,
                entry.representative_tier,
            )
        if entry is entries[2]:
            return BatchCaseResult(
                entry.case_id,
                entries[3].ordinal,
                BatchClassification.READY,
                entry.representative_tier,
            )
        return BatchCaseResult(
            entry.case_id,
            entry.ordinal,
            BatchClassification.READY,
            entry.representative_tier,
        )

    report = ConcurrentSCRCBatchRunner(case_workers=4).run(
        entries,
        executor,
        mode="unit_test",
    )

    assert [item.classification for item in report.cases] == [
        BatchClassification.INFRASTRUCTURE_FAILED,
        BatchClassification.INFRASTRUCTURE_FAILED,
        BatchClassification.INFRASTRUCTURE_FAILED,
        BatchClassification.READY,
    ]
    assert "isolated boom" in report.cases[0].diagnostic
    assert "another case" in report.cases[1].diagnostic
    assert "another case" in report.cases[2].diagnostic


@pytest.mark.parametrize(
    "case_workers",
    [0, -1, 1.5, True, MAX_CASE_WORKERS + 1],
)
def test_concurrent_batch_rejects_invalid_case_workers(case_workers) -> None:
    with pytest.raises(ValueError, match="case_workers must be an integer"):
        ConcurrentSCRCBatchRunner(case_workers=case_workers)


def test_batch_scheduling_metadata_is_closed_validated_and_json_safe() -> None:
    entries = build_workload_manifest().entries[:1]
    scheduling = {
        "case_workers": 2,
        "candidates": 3,
        "max_branch_workers": 6,
        "gate2_workers": 1,
    }
    expected = dict(scheduling)
    report = ConcurrentSCRCBatchRunner(case_workers=2).run(
        entries,
        dry_run_executor(),
        mode="dry_run",
        scheduling=scheduling,
    )
    scheduling["gate2_workers"] = 2

    assert report.to_json()["scheduling"] == expected
    assert asdict(report)["scheduling"] == expected
    json.dumps(report.to_json(), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("scheduling", "message"),
    [
        (
            {
                "case_workers": 2,
                "candidates": 1,
                "max_branch_workers": 2,
            },
            "exactly",
        ),
        (
            {
                "case_workers": 2,
                "candidates": 1,
                "max_branch_workers": 2,
                "gate2_workers": 1,
                "unexpected": 1,
            },
            "exactly",
        ),
        (
            {
                "case_workers": 2,
                "candidates": True,
                "max_branch_workers": 2,
                "gate2_workers": 1,
            },
            "positive integer",
        ),
        (
            {
                "case_workers": 2,
                "candidates": 2,
                "max_branch_workers": 3,
                "gate2_workers": 1,
            },
            r"case_workers \* candidates",
        ),
        (
            {
                "case_workers": 1,
                "candidates": 1,
                "max_branch_workers": 2,
                "gate2_workers": 1,
            },
            "must equal runner",
        ),
    ],
)
def test_concurrent_batch_rejects_invalid_scheduling_metadata(
    scheduling,
    message,
) -> None:
    entries = build_workload_manifest().entries[:1]
    with pytest.raises(ValueError, match=message):
        ConcurrentSCRCBatchRunner(case_workers=2).run(
            entries,
            dry_run_executor(),
            mode="dry_run",
            scheduling=scheduling,
        )
