"""Independent Gate1/Gate2 observations and paired conditional summaries."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .orchestrate import GATE2_EXECUTION_PROTOCOL


_COMPLETED_STATUSES = frozenset(
    {
        "gate2_pass",
        "replay_diverged",
        "scalar_diverged",
        "replay_and_scalar_diverged",
    }
)


def _has_source_anchored_evidence(case: Mapping[str, Any]) -> bool:
    source = case.get("source")
    replay = case.get("replay")
    scalar = case.get("scalar")
    source_replay = case.get("source_replay")
    source_scalar = case.get("source_scalar")
    status = case.get("status")
    input_digest = case.get("input_digest")
    if not (
        isinstance(source, Mapping)
        and source.get("ok") is True
        and isinstance(replay, Mapping)
        and replay.get("ok") is True
        and isinstance(scalar, Mapping)
        and scalar.get("ok") is True
        and isinstance(source_replay, Mapping)
        and isinstance(source_replay.get("passed"), bool)
        and isinstance(source_scalar, Mapping)
        and isinstance(source_scalar.get("passed"), bool)
        and isinstance(input_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", input_digest) is not None
    ):
        return False
    replay_passed = source_replay["passed"]
    scalar_passed = source_scalar["passed"]
    if replay_passed and scalar_passed:
        expected = "gate2_pass"
    elif not replay_passed and scalar_passed:
        expected = "replay_diverged"
    elif replay_passed and not scalar_passed:
        expected = "scalar_diverged"
    else:
        expected = "replay_and_scalar_diverged"
    return status == expected


def extract_gate1_gate2_observation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract both gate verdicts without collapsing one into the other."""

    gate1 = report.get("gate1")
    gate1_status = gate1.get("status") if isinstance(gate1, Mapping) else None
    observation: dict[str, Any] = {
        "gate1_status": gate1_status,
        "gate1_closed": gate1_status == "replay_closed",
        "gate2_evidence_kind": None,
        "gate2_execution_protocol": None,
        "gate2_attempted": False,
        "gate2_comparison_completed": False,
        "gate2_status": None,
        "gate2_pass": None,
        "gate2_case_statuses": [],
    }

    # Gate2 v3 is digest-bound to an accepted final J.  Rejected-sample
    # diagnostics from older report schemas are not Gate2 observations.
    if not observation["gate1_closed"]:
        return observation

    evidence: Mapping[str, Any] | None = None
    is_suite = False
    for report_key, evidence_kind, suite in (
        ("gate2_suite", "suite", True),
        ("gate2", "single", False),
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
    execution_protocol = evidence.get("execution_protocol")
    observation["gate2_execution_protocol"] = (
        execution_protocol if isinstance(execution_protocol, str) else None
    )
    if execution_protocol != GATE2_EXECUTION_PROTOCOL:
        observation["gate2_status"] = "gate2_protocol_mismatch"
        return observation

    if is_suite:
        cases = evidence.get("cases")
        case_items = list(
            cases
            if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes))
            else ()
        )
        statuses: list[str] = []
        case_mappings: list[Mapping[str, Any]] = []
        all_cases_are_mappings = bool(case_items)
        for case in case_items:
            if not isinstance(case, Mapping):
                all_cases_are_mappings = False
                continue
            case_mappings.append(case)
            status = case.get("status")
            if isinstance(status, str):
                statuses.append(status)
        observation["gate2_case_statuses"] = statuses

        all_statuses_present = (
            all_cases_are_mappings
            and len(case_mappings) == len(case_items)
            and len(statuses) == len(case_items)
        )
        statuses_terminal = all(
            status in _COMPLETED_STATUSES for status in statuses
        )
        comparisons_complete = bool(case_mappings) and all(
            _has_source_anchored_evidence(case) for case in case_mappings
        )

        any_case_id = any("case_id" in case for case in case_mappings)
        case_ids = [case.get("case_id") for case in case_mappings]
        case_ids_valid = (
            not any_case_id
            or (
                len(case_ids) == len(case_items)
                and all(
                    isinstance(case_id, str) and bool(case_id)
                    for case_id in case_ids
                )
                and len(case_ids) == len(set(case_ids))
            )
        )

        status_counts_valid = True
        declared_status_counts = evidence.get("status_counts")
        if declared_status_counts is not None:
            status_counts_valid = bool(
                isinstance(declared_status_counts, Mapping)
                and all(
                    isinstance(key, str)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for key, value in declared_status_counts.items()
                )
                and dict(Counter(statuses)) == dict(declared_status_counts)
            )

        suite_passed = evidence.get("passed")
        derived_pass = bool(statuses) and all(
            status == "gate2_pass" for status in statuses
        )
        completed = (
            all_statuses_present
            and statuses_terminal
            and comparisons_complete
            and case_ids_valid
            and status_counts_valid
            and isinstance(suite_passed, bool)
            and suite_passed is derived_pass
        )
        observation["gate2_comparison_completed"] = completed
        raw_status = evidence.get("status")
        if completed:
            observation["gate2_status"] = (
                "gate2_pass" if derived_pass else "gate2_suite_failed"
            )
            observation["gate2_pass"] = derived_pass
        elif isinstance(raw_status, str):
            observation["gate2_status"] = raw_status
        else:
            observation["gate2_status"] = "gate2_suite_incomplete"
        return observation

    status = evidence.get("status")
    observation["gate2_status"] = status if isinstance(status, str) else None
    if isinstance(status, str):
        observation["gate2_case_statuses"] = [status]
        completed = (
            status in _COMPLETED_STATUSES
            and _has_source_anchored_evidence(evidence)
        )
        observation["gate2_comparison_completed"] = completed
        if completed:
            observation["gate2_pass"] = status == "gate2_pass"
    return observation


def _conditional_bucket(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = [item for item in observations if item["gate2_attempted"]]
    completed = [
        item for item in observations if item["gate2_comparison_completed"]
    ]
    passed = [item for item in completed if item["gate2_pass"] is True]
    total = len(observations)
    return {
        "samples": total,
        "gate2_attempted": len(attempted),
        "gate2_comparison_completed": len(completed),
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
    """Report independent Gate1/Gate2 quadrants and completion coverage."""

    observations: list[Mapping[str, Any]] = []
    for row in rows:
        observation = row.get("gate1_gate2")
        if isinstance(observation, Mapping) and isinstance(
            observation.get("gate1_status"), str
        ):
            observations.append(observation)
    closed = [item for item in observations if item["gate1_closed"]]
    not_closed = [item for item in observations if not item["gate1_closed"]]
    completed = [
        item for item in observations if item["gate2_comparison_completed"]
    ]
    unattempted = [
        item for item in observations if not item["gate2_attempted"]
    ]
    attempted_incomplete = [
        item
        for item in observations
        if item["gate2_attempted"]
        and not item["gate2_comparison_completed"]
    ]

    def quadrant(*, gate1_closed: bool, gate2_pass: bool) -> int:
        return sum(
            item["gate1_closed"] is gate1_closed
            and item["gate2_pass"] is gate2_pass
            for item in completed
        )

    statuses = sorted({str(item["gate1_status"]) for item in observations})
    return {
        "schema": "scrc_gate1_gate2_conditional_v5",
        "gate1_observed_samples": len(observations),
        "rows_without_gate1_verdict": len(rows) - len(observations),
        "quadrants": {
            "gate1_closed_gate2_pass": quadrant(
                gate1_closed=True, gate2_pass=True
            ),
            "gate1_closed_gate2_nonpass": quadrant(
                gate1_closed=True, gate2_pass=False
            ),
            "gate1_not_closed_gate2_pass": quadrant(
                gate1_closed=False, gate2_pass=True
            ),
            "gate1_not_closed_gate2_nonpass": quadrant(
                gate1_closed=False, gate2_pass=False
            ),
        },
        "quadrant_coverage": {
            "eligible_completed_samples": len(completed),
            "attempted_but_incomplete": len(attempted_incomplete),
            "unattempted": len(unattempted),
            "completion_coverage": (
                None
                if not observations
                else len(completed) / len(observations)
            ),
        },
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
