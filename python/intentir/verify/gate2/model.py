"""Result model for source-anchored Gate2 dynamic execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class Gate2Status(str, Enum):
    PASS = "gate2_pass"
    PAIRWISE_DIVERGED = "pairwise_diverged"
    COMPARISON_UNSUPPORTED = "comparison_unsupported"
    REFERENCE_UNSUPPORTED = "reference_unsupported"
    EXECUTION_FAILED = "execution_failed"


@dataclass
class ExecutionResult:
    leg: str
    ok: bool
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    error: str = ""
    unsupported: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "ok": self.ok,
            "unsupported": self.unsupported,
            "error": self.error,
            "outputs": {
                name: {
                    "dtype": str(np.asarray(value).dtype),
                    "shape": list(np.asarray(value).shape),
                }
                for name, value in self.outputs.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OutputAgreement:
    output: str
    passed: bool
    max_abs_error: float | None
    detail: str = ""
    relation_kind: str = "exact"
    equality_required: bool = True
    policy_id: str = ""
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "detail": self.detail,
            "relation_kind": self.relation_kind,
            "equality_required": self.equality_required,
            "policy_id": self.policy_id,
            "rtol": self.rtol,
            "atol": self.atol,
            "equal_nan": self.equal_nan,
        }


@dataclass(frozen=True)
class PairwiseAgreement:
    left: str
    right: str
    passed: bool
    outputs: tuple[OutputAgreement, ...]
    policy_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "passed": self.passed,
            "policy_id": self.policy_id,
            "outputs": [item.to_json() for item in self.outputs],
        }


@dataclass
class Gate2Result:
    case_id: str
    status: Gate2Status
    source: ExecutionResult
    replay: ExecutionResult
    reference: ExecutionResult
    source_replay: PairwiseAgreement | None = None
    source_j: PairwiseAgreement | None = None
    replay_j: PairwiseAgreement | None = None
    input_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is Gate2Status.PASS

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "source": self.source.to_json(),
            "replay": self.replay.to_json(),
            "reference": self.reference.to_json(),
            "source_replay": (
                self.source_replay.to_json()
                if self.source_replay is not None else None
            ),
            "source_j": (
                self.source_j.to_json() if self.source_j is not None else None
            ),
            "replay_j": (
                self.replay_j.to_json() if self.replay_j is not None else None
            ),
            "input_digest": self.input_digest,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Gate2SuiteResult:
    suite_id: str
    cases: tuple[Gate2Result, ...]
    c_values: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case result")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite result contains duplicate case_id values")
        names = [name for name, _ in self.c_values]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(
                "Gate2 suite result c0 bindings must be unique and sorted"
            )

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

    def to_json(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "passed": self.passed,
            "failed_case_ids": list(self.failed_case_ids),
            "status_counts": self.status_counts,
            "c0": dict(self.c_values),
            "cases": [case.to_json() for case in self.cases],
        }


__all__ = [
    "ExecutionResult",
    "Gate2Result",
    "Gate2Status",
    "Gate2SuiteResult",
    "OutputAgreement",
    "PairwiseAgreement",
]
