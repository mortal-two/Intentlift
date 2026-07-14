"""Result model for SCRC Gate2 three-way execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class Gate2Status(str, Enum):
    PASS = "gate2_pass"
    LOWERING_DIVERGED = "lowering_diverged"
    REPLAY_DIVERGED = "replay_diverged"
    LIFT_DIVERGED = "lift_diverged"
    ALL_DIVERGED = "all_diverged"
    INCONCLUSIVE = "inconclusive"
    LOWERING_UNSUPPORTED = "lowering_unsupported"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class Tolerance:
    rtol: float = 1.0e-5
    atol: float = 1.0e-6
    equal_nan: bool = True

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("Gate2 tolerances must be non-negative")


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
    lowering: ExecutionResult
    source_replay: PairwiseAgreement | None = None
    source_lowering: PairwiseAgreement | None = None
    replay_lowering: PairwiseAgreement | None = None
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
    "Tolerance",
]
