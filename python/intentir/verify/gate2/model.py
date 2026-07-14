"""Result model for SCRC Gate2 three-way execution."""

from __future__ import annotations

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

    @property
    def passed(self) -> bool:
        return self.status is Gate2Status.PASS


__all__ = [
    "ExecutionResult",
    "Gate2Result",
    "Gate2Status",
    "OutputAgreement",
    "PairwiseAgreement",
    "Tolerance",
]
