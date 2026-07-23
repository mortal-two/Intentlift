from __future__ import annotations

from threading import Barrier

from intentir.verify.candidate_fanout import (
    CANDIDATE_SELECTION_RULE,
    CandidateBranchOutcome,
    CandidateBranchStatus,
    ConcurrentCandidateFanoutRunner,
)
from intentir.verify.workload_manifest import build_workload_manifest


def _entry():
    return build_workload_manifest().by_id["high_performance:liger_dyt"]


def test_candidate_fanout_is_concurrent_and_orders_results_by_branch() -> None:
    entry = _entry()
    barrier = Barrier(3, timeout=2.0)

    def execute(item, branch_id):
        assert item is entry
        barrier.wait()
        status = (
            CandidateBranchStatus.GATE1_CLOSED
            if branch_id in {2, 3}
            else CandidateBranchStatus.NO_CLOSURE
        )
        return CandidateBranchOutcome(
            branch_id,
            item.case_id,
            item.ordinal,
            status,
            evidence_path=f"run-root://candidate-{branch_id:02d}",
            call_budget={"lift_calls": branch_id},
        )

    report = ConcurrentCandidateFanoutRunner().run(
        entry, execute, candidates=3
    )

    assert [item.branch_id for item in report.branches] == [1, 2, 3]
    assert report.winner is report.branches[1]
    assert report.selection_rule == CANDIDATE_SELECTION_RULE
    assert report.counts == {"gate1_closed": 2, "no_closure": 1}
    assert report.aggregate_call_budget == {"lift_calls": 6}


def test_candidate_fanout_isolates_one_branch_exception() -> None:
    entry = _entry()

    def execute(item, branch_id):
        if branch_id == 2:
            raise RuntimeError("provider unavailable")
        return CandidateBranchOutcome(
            branch_id,
            item.case_id,
            item.ordinal,
            CandidateBranchStatus.NO_CLOSURE,
        )

    report = ConcurrentCandidateFanoutRunner().run(
        entry, execute, candidates=3
    )

    assert report.branches[1].status is CandidateBranchStatus.INFRASTRUCTURE_FAILED
    assert "provider unavailable" in report.branches[1].diagnostic
    assert report.branches[0].status is CandidateBranchStatus.NO_CLOSURE
    assert report.branches[2].status is CandidateBranchStatus.NO_CLOSURE


def test_candidate_fanout_validates_width_and_executor_identity() -> None:
    entry = _entry()
    runner = ConcurrentCandidateFanoutRunner()

    for invalid in (0, 33):
        try:
            runner.run(entry, lambda _entry, _branch: None, candidates=invalid)
        except ValueError as exc:
            assert "[1, 32]" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("invalid candidate width was accepted")

    report = runner.run(
        entry,
        lambda item, branch_id: CandidateBranchOutcome(
            branch_id + 1,
            item.case_id,
            item.ordinal,
            CandidateBranchStatus.NO_CLOSURE,
        ),
        candidates=1,
    )
    assert report.branches[0].status is CandidateBranchStatus.INFRASTRUCTURE_FAILED
    assert "another branch id" in report.branches[0].diagnostic
