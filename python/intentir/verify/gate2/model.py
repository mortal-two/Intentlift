"""Result model for source-anchored Gate2 dynamic execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class Gate2Status(str, Enum):
    PASS = "gate2_pass"
    REPLAY_DIVERGED = "replay_diverged"
    SCALAR_DIVERGED = "scalar_diverged"
    REPLAY_AND_SCALAR_DIVERGED = "replay_and_scalar_diverged"
    RESULT_RELATION_DIVERGED = "result_relation_diverged"
    SCALAR_UNSUPPORTED = "scalar_unsupported"
    EXECUTION_FAILED = "execution_failed"


@dataclass
class ExecutionResult:
    leg: str
    ok: bool
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    error: str = ""
    unsupported: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputAgreement:
    output: str
    passed: bool
    max_abs_error: float | None
    detail: str = ""
    relation_kind: str = "exact"
    equality_required: bool = True


@dataclass(frozen=True)
class PairwiseAgreement:
    left: str
    right: str
    passed: bool
    outputs: tuple[OutputAgreement, ...]


@dataclass
class Gate2Result:
    case_id: str
    status: Gate2Status
    source: ExecutionResult
    replay: ExecutionResult
    scalar: ExecutionResult
    source_replay: PairwiseAgreement | None = None
    source_scalar: PairwiseAgreement | None = None
    input_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is Gate2Status.PASS


@dataclass(frozen=True)
class Gate2SuiteResult:
    suite_id: str
    cases: tuple[Gate2Result, ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case result")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite result contains duplicate case_id values")

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases if not case.passed)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(case.status.value for case in self.cases).items())
        )


__all__ = [
    "ExecutionResult",
    "Gate2Result",
    "Gate2Status",
    "Gate2SuiteResult",
    "OutputAgreement",
    "PairwiseAgreement",
]
