"""Independent Gate1/Gate2 observations and paired conditional summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_THREE_WAY_STATUSES = frozenset(
    {
        "gate2_pass",
        "lowering_diverged",
        "replay_diverged",
        "lift_diverged",
        "all_diverged",
        "inconclusive",
    }
)


def extract_gate1_gate2_observation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract both gate verdicts without collapsing one into the other.

    A missing diagnostic, setup failure, unsupported lowering, or execution
    failure is not silently counted as a semantic Gate2 rejection. Conditional
    probabilities use only completed three-way comparisons and expose attempt
    and completion coverage alongside the probability.
    """

    gate1 = report.get("gate1")
    gate1_status = gate1.get("status") if isinstance(gate1, Mapping) else None
    observation: dict[str, Any] = {
        "gate1_status": gate1_status,
        "gate1_closed": gate1_status == "replay_closed",
        "gate2_evidence_kind": None,
        "gate2_attempted": False,
        "gate2_three_way_completed": False,
        "gate2_status": None,
        "gate2_pass": None,
        "gate2_case_statuses": [],
    }

    evidence: Mapping[str, Any] | None = None
    is_suite = False
    for report_key, evidence_kind, suite in (
        ("gate2_suite", "production_suite", True),
        ("gate2", "production", False),
        ("gate2_diagnostic_suite", "diagnostic_suite", True),
        ("gate2_diagnostic", "diagnostic", False),
    ):
        candidate = report.get(report_key)
        if isinstance(candidate, Mapping):
            observation["gate2_evidence_kind"] = evidence_kind
            evidence = candidate
            is_suite = suite
            break

    if evidence is None:
        return observation

    observation["gate2_attempted"] = True
    if is_suite:
        cases = evidence.get("cases")
        case_items = (
            cases
            if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes))
            else ()
        )
        statuses = [
            str(case["status"])
            for case in case_items
            if isinstance(case, Mapping)
            and isinstance(case.get("status"), str)
        ]
        observation["gate2_case_statuses"] = statuses
        suite_passed = evidence.get("passed")
        completed = (
            bool(statuses)
            and isinstance(suite_passed, bool)
            and all(status in _THREE_WAY_STATUSES for status in statuses)
        )
        observation["gate2_three_way_completed"] = completed
        raw_status = evidence.get("status")
        if isinstance(raw_status, str) and not statuses:
            observation["gate2_status"] = raw_status
        else:
            observation["gate2_status"] = (
                "gate2_pass" if suite_passed is True else "gate2_suite_failed"
            )
        if completed:
            observation["gate2_pass"] = suite_passed is True
        return observation

    status = evidence.get("status")
    observation["gate2_status"] = status if isinstance(status, str) else None
    if isinstance(status, str):
        observation["gate2_case_statuses"] = [status]
        completed = status in _THREE_WAY_STATUSES
        observation["gate2_three_way_completed"] = completed
        if completed:
            observation["gate2_pass"] = status == "gate2_pass"
    return observation


def _conditional_bucket(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = [item for item in observations if item["gate2_attempted"]]
    completed = [
        item for item in observations if item["gate2_three_way_completed"]
    ]
    passed = [item for item in completed if item["gate2_pass"] is True]
    total = len(observations)
    return {
        "samples": total,
        "gate2_attempted": len(attempted),
        "gate2_three_way_completed": len(completed),
        "gate2_pass": len(passed),
        "gate2_nonpass": len(completed) - len(passed),
        "attempt_coverage": None if total == 0 else len(attempted) / total,
        "completion_coverage": None if total == 0 else len(completed) / total,
        "pass_probability_given_completed": (
            None if not completed else len(passed) / len(completed)
        ),
    }


def summarize_gate1_gate2(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report Gate2 pass probabilities conditioned on the independent Gate1 result."""

    observations: list[Mapping[str, Any]] = []
    for row in rows:
        observation = row.get("gate1_gate2")
        if isinstance(observation, Mapping) and isinstance(
            observation.get("gate1_status"), str
        ):
            observations.append(observation)
    closed = [item for item in observations if item["gate1_closed"]]
    not_closed = [item for item in observations if not item["gate1_closed"]]
    statuses = sorted({str(item["gate1_status"]) for item in observations})
    return {
        "schema": "scrc_gate1_gate2_conditional_v1",
        "gate1_observed_samples": len(observations),
        "rows_without_gate1_verdict": len(rows) - len(observations),
        "replay_closed": _conditional_bucket(closed),
        "not_replay_closed": _conditional_bucket(not_closed),
        "by_gate1_status": {
            status: _conditional_bucket(
                [item for item in observations if item["gate1_status"] == status]
            )
            for status in statuses
        },
    }


__all__ = ["extract_gate1_gate2_observation", "summarize_gate1_gate2"]
