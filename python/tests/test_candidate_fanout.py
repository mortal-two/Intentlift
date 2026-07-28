from __future__ import annotations

from threading import Barrier, Event, Lock
from time import monotonic

import pytest

from intentir.verify.candidate_fanout import (
    CANDIDATE_SELECTION_RULE,
    CandidateBranchOutcome,
    CandidateBranchStatus,
    CandidateFanoutSearchReport,
    ConcurrentCandidateFanoutRunner,
)
from intentir.verify.workload_manifest import build_workload_manifest


def _entry():
    return build_workload_manifest().by_id["high_performance:liger_dyt"]


def test_candidate_fanout_is_concurrent_and_orders_results_by_branch() -> None:
    entry = _entry()
    barrier = Barrier(3, timeout=2.0)
    branch_two_completed = Event()
    completion_order = []
    completion_lock = Lock()

    def execute(item, branch_id, stop_requested):
        assert item is entry
        barrier.wait()
        if branch_id == 2:
            with completion_lock:
                completion_order.append(branch_id)
            branch_two_completed.set()
            status = CandidateBranchStatus.GATE1_CLOSED
            formal_round = 1
        else:
            assert branch_two_completed.wait(2.0)
            deadline = monotonic() + 2.0
            while stop_requested() is None and monotonic() < deadline:
                Event().wait(0.001)
            assert stop_requested() == 2
            with completion_lock:
                completion_order.append(branch_id)
            status = (
                CandidateBranchStatus.GATE1_CLOSED
                if branch_id == 1
                else CandidateBranchStatus.NO_CLOSURE
            )
            formal_round = 3 if branch_id == 1 else None
        return CandidateBranchOutcome(
            branch_id,
            item.case_id,
            item.ordinal,
            status,
            evidence_path=f"run-root://candidate-{branch_id:02d}",
            call_budget={"lift_calls": branch_id},
            formal_round=formal_round,
        )

    report = ConcurrentCandidateFanoutRunner().run(
        entry,
        execute,
        candidates=3,
    )

    assert completion_order[0] == 2
    assert [item.branch_id for item in report.branches] == [1, 2, 3]
    assert report.winner is report.branches[1]
    assert report.winner_branch_id == 2
    assert report.winner_formal_round == 1
    assert report.selection_rule == CANDIDATE_SELECTION_RULE
    assert report.counts == {
        "gate1_closed": 1,
        "gate1_closed_not_selected": 1,
        "no_closure": 1,
    }
    assert report.branches[0].formal_round == 3
    assert report.branches[0].details["winner_branch_id"] == 2
    assert report.branches[0].details["winner_formal_round"] == 1
    assert report.aggregate_call_budget == {"lift_calls": 6}


def test_candidate_fanout_isolates_one_branch_exception() -> None:
    entry = _entry()

    def execute(item, branch_id, _stop_requested):
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
    assert report.winner is None
    assert report.winner_formal_round is None


def test_candidate_fanout_b1_selects_its_only_closed_coordinate() -> None:
    entry = _entry()
    report = ConcurrentCandidateFanoutRunner().run(
        entry,
        lambda item, branch_id, _stop_requested: CandidateBranchOutcome(
            branch_id,
            item.case_id,
            item.ordinal,
            CandidateBranchStatus.GATE1_CLOSED,
            formal_round=2,
        ),
        candidates=1,
    )

    assert report.winner is report.branches[0]
    assert report.winner_branch_id == 1
    assert report.winner_formal_round == 2
    assert report.counts == {"gate1_closed": 1}


def test_candidate_fanout_requires_closed_formal_round_only() -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="positive formal_round"):
        CandidateBranchOutcome(
            1,
            entry.case_id,
            entry.ordinal,
            CandidateBranchStatus.GATE1_CLOSED,
        )
    with pytest.raises(ValueError, match="only a Gate1-closed"):
        CandidateBranchOutcome(
            1,
            entry.case_id,
            entry.ordinal,
            CandidateBranchStatus.NO_CLOSURE,
            formal_round=1,
        )


def test_candidate_fanout_report_accepts_first_observed_winner() -> None:
    entry = _entry()
    branches = (
        CandidateBranchOutcome(
            1,
            entry.case_id,
            entry.ordinal,
            CandidateBranchStatus.GATE1_CLOSED_NOT_SELECTED,
            formal_round=3,
        ),
        CandidateBranchOutcome(
            2,
            entry.case_id,
            entry.ordinal,
            CandidateBranchStatus.GATE1_CLOSED,
            formal_round=1,
        ),
    )

    report = CandidateFanoutSearchReport(
        entry.case_id,
        entry.ordinal,
        branches,
        winner_branch_id=2,
        winner_formal_round=1,
    )
    assert report.winner is branches[1]


def test_candidate_fanout_validates_width_and_executor_identity() -> None:
    entry = _entry()
    runner = ConcurrentCandidateFanoutRunner()

    for invalid in (0, 33):
        try:
            runner.run(
                entry,
                lambda _entry, _branch, _stop_requested: None,
                candidates=invalid,
            )
        except ValueError as exc:
            assert "[1, 32]" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("invalid candidate width was accepted")

    report = runner.run(
        entry,
        lambda item, branch_id, _stop_requested: CandidateBranchOutcome(
            branch_id + 1,
            item.case_id,
            item.ordinal,
            CandidateBranchStatus.NO_CLOSURE,
        ),
        candidates=1,
    )
    assert report.branches[0].status is CandidateBranchStatus.INFRASTRUCTURE_FAILED
    assert "another branch id" in report.branches[0].diagnostic
