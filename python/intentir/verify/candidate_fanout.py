"""Bounded same-case candidate concurrency before hidden Gate2 selection."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .workload_manifest import SCRCWorkloadEntry


CANDIDATE_FANOUT_PROTOCOL = "intentir.scrc.candidate-fanout.v1"
CANDIDATE_SELECTION_RULE = "lowest_branch_after_gate1_search_barrier.v1"


class CandidateBranchStatus(str, Enum):
    GATE1_CLOSED = "gate1_closed"
    NO_CLOSURE = "no_closure"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


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

    @property
    def winner(self) -> CandidateBranchOutcome | None:
        return next(
            (
                item
                for item in self.branches
                if item.status is CandidateBranchStatus.GATE1_CLOSED
            ),
            None,
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
        winner = self.winner
        return {
            "protocol": CANDIDATE_FANOUT_PROTOCOL,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "candidate_count": len(self.branches),
            "selection_rule": self.selection_rule,
            "winner_branch_id": winner.branch_id if winner is not None else None,
            "status_counts": self.counts,
            "aggregate_call_budget": self.aggregate_call_budget,
            "branches": [item.to_json() for item in self.branches],
        }


CandidateBranchExecutor = Callable[
    [SCRCWorkloadEntry, int], CandidateBranchOutcome
]


class ConcurrentCandidateFanoutRunner:
    """Run independent branches concurrently and select without Gate2 evidence."""

    def run(
        self,
        entry: SCRCWorkloadEntry,
        executor: CandidateBranchExecutor,
        *,
        candidates: int,
    ) -> CandidateFanoutSearchReport:
        if not isinstance(candidates, int) or not 1 <= candidates <= 32:
            raise ValueError("candidates must be an integer in [1, 32]")
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
                completed[branch_id] = outcome
        branches = tuple(
            completed[branch_id] for branch_id in range(1, candidates + 1)
        )
        return CandidateFanoutSearchReport(
            entry.case_id,
            entry.ordinal,
            branches,
        )


__all__ = [
    "CANDIDATE_FANOUT_PROTOCOL",
    "CANDIDATE_SELECTION_RULE",
    "CandidateBranchOutcome",
    "CandidateBranchStatus",
    "CandidateFanoutSearchReport",
    "ConcurrentCandidateFanoutRunner",
]
