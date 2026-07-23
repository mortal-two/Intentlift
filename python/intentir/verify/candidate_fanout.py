"""Bounded same-case candidate concurrency before hidden Gate2 selection."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock
from typing import Any, Callable, Mapping

from .workload_manifest import SCRCWorkloadEntry


CANDIDATE_FANOUT_PROTOCOL = "intentir.scrc.candidate-fanout.v2"
CANDIDATE_SELECTION_RULE = "first_gate1_closure_cooperative_stop.v2"


class CandidateBranchStatus(str, Enum):
    GATE1_CLOSED = "gate1_closed"
    GATE1_CLOSED_NOT_SELECTED = "gate1_closed_not_selected"
    NO_CLOSURE = "no_closure"
    SUPERSEDED = "superseded"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class FirstClosureControl:
    """Exactly-once winner selection plus a cooperative branch stop signal."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._winner_branch_id: int | None = None

    @property
    def winner_branch_id(self) -> int | None:
        with self._lock:
            return self._winner_branch_id

    def try_claim(self, branch_id: int) -> bool:
        if not isinstance(branch_id, int) or branch_id < 1:
            raise ValueError("winner branch_id must be positive")
        with self._lock:
            if self._winner_branch_id is not None:
                return False
            self._winner_branch_id = branch_id
            self._event.set()
            return True

    def stop_requested(self) -> int | None:
        if not self._event.is_set():
            return None
        return self.winner_branch_id


@dataclass(frozen=True)
class CandidateBranchOutcome:
    branch_id: int
    case_id: str
    ordinal: int
    status: CandidateBranchStatus
    diagnostic: str = ""
    evidence_path: str | None = None
    call_budget: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    runtime: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, int) or self.branch_id < 1:
            raise ValueError("candidate branch_id must be positive")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("candidate case_id must be non-empty")
        if not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ValueError("candidate ordinal must be positive")
        if not isinstance(self.status, CandidateBranchStatus):
            object.__setattr__(self, "status", CandidateBranchStatus(self.status))

    def to_json(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "status": self.status.value,
            "diagnostic": self.diagnostic,
            "evidence_path": self.evidence_path,
            "call_budget": dict(self.call_budget),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CandidateFanoutSearchReport:
    case_id: str
    ordinal: int
    branches: tuple[CandidateBranchOutcome, ...]
    winner_branch_id: int | None = None
    selection_rule: str = CANDIDATE_SELECTION_RULE

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("candidate fan-out requires at least one branch")
        branch_ids = [item.branch_id for item in self.branches]
        if branch_ids != list(range(1, len(self.branches) + 1)):
            raise ValueError("candidate branches must be ordered and gap-free")
        if any(
            item.case_id != self.case_id or item.ordinal != self.ordinal
            for item in self.branches
        ):
            raise ValueError("candidate branch belongs to another workload")
        selected = [
            item.branch_id
            for item in self.branches
            if item.status is CandidateBranchStatus.GATE1_CLOSED
        ]
        if self.winner_branch_id is None:
            if selected:
                raise ValueError("fan-out report omitted its selected winner")
        elif selected != [self.winner_branch_id]:
            raise ValueError("fan-out report winner differs from selected branch")

    @property
    def winner(self) -> CandidateBranchOutcome | None:
        if self.winner_branch_id is None:
            return None
        return next(
            item for item in self.branches
            if item.branch_id == self.winner_branch_id
        )

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(item.status.value for item in self.branches)
        return dict(sorted(counts.items()))

    @property
    def aggregate_call_budget(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for branch in self.branches:
            totals.update(branch.call_budget)
        return dict(sorted(totals.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CANDIDATE_FANOUT_PROTOCOL,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "candidate_count": len(self.branches),
            "selection_rule": self.selection_rule,
            "winner_branch_id": self.winner_branch_id,
            "status_counts": self.counts,
            "aggregate_call_budget": self.aggregate_call_budget,
            "branches": [item.to_json() for item in self.branches],
        }


CandidateBranchExecutor = Callable[
    [SCRCWorkloadEntry, int], CandidateBranchOutcome
]
FirstClosedCallback = Callable[[CandidateBranchOutcome], None]


class ConcurrentCandidateFanoutRunner:
    """Run independent branches concurrently and select without Gate2 evidence."""

    def run(
        self,
        entry: SCRCWorkloadEntry,
        executor: CandidateBranchExecutor,
        *,
        candidates: int,
        control: FirstClosureControl | None = None,
        on_first_closed: FirstClosedCallback | None = None,
    ) -> CandidateFanoutSearchReport:
        if not isinstance(candidates, int) or not 1 <= candidates <= 32:
            raise ValueError("candidates must be an integer in [1, 32]")
        control = control or FirstClosureControl()
        if control.winner_branch_id is not None:
            raise ValueError("candidate fan-out control was already used")
        completed: dict[int, CandidateBranchOutcome] = {}
        with ThreadPoolExecutor(
            max_workers=candidates,
            thread_name_prefix="scrc-candidate",
        ) as pool:
            future_to_branch = {
                pool.submit(executor, entry, branch_id): branch_id
                for branch_id in range(1, candidates + 1)
            }
            for future in as_completed(future_to_branch):
                branch_id = future_to_branch[future]
                try:
                    outcome = future.result()
                    if outcome.branch_id != branch_id:
                        raise ValueError(
                            "candidate executor returned another branch id"
                        )
                    if outcome.case_id != entry.case_id:
                        raise ValueError(
                            "candidate executor returned another case id"
                        )
                    if outcome.ordinal != entry.ordinal:
                        raise ValueError(
                            "candidate executor returned another ordinal"
                        )
                except Exception as exc:
                    outcome = CandidateBranchOutcome(
                        branch_id,
                        entry.case_id,
                        entry.ordinal,
                        CandidateBranchStatus.INFRASTRUCTURE_FAILED,
                        diagnostic=f"{type(exc).__name__}: {exc}",
                    )
                if outcome.status is CandidateBranchStatus.GATE1_CLOSED:
                    if control.try_claim(branch_id):
                        if on_first_closed is not None:
                            on_first_closed(outcome)
                    else:
                        outcome = CandidateBranchOutcome(
                            outcome.branch_id,
                            outcome.case_id,
                            outcome.ordinal,
                            CandidateBranchStatus.GATE1_CLOSED_NOT_SELECTED,
                            diagnostic=outcome.diagnostic,
                            evidence_path=outcome.evidence_path,
                            call_budget=outcome.call_budget,
                            details={
                                **dict(outcome.details),
                                "winner_branch_id": control.winner_branch_id,
                            },
                            runtime=outcome.runtime,
                        )
                completed[branch_id] = outcome
        branches = tuple(
            completed[branch_id] for branch_id in range(1, candidates + 1)
        )
        return CandidateFanoutSearchReport(
            entry.case_id,
            entry.ordinal,
            branches,
            winner_branch_id=control.winner_branch_id,
        )


__all__ = [
    "CANDIDATE_FANOUT_PROTOCOL",
    "CANDIDATE_SELECTION_RULE",
    "CandidateBranchOutcome",
    "CandidateBranchStatus",
    "CandidateFanoutSearchReport",
    "ConcurrentCandidateFanoutRunner",
    "FirstClosureControl",
]
