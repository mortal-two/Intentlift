"""Gate1 capability evaluation against a downstream Gate2 oracle.

Gate1 is intentionally conservative: its primary experimental objective is a
high conditional Gate2 pass rate among replay-closed kernels, not maximal
coverage.  This module keeps that distinction explicit and classifies the four
possible Gate1/Gate2 outcomes without overstating any of them as proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class FailureCause(str, Enum):
    """Stable failure taxonomy used by diagnostics and capability reports."""

    ARTIFACT_SCHEMA = "artifact_schema"
    PLAN_LEAKAGE = "plan_leakage"
    RULE_CARD = "rule_card"
    REPLAY_POLICY = "replay_policy"
    REPLAY_COMPILE = "replay_compile"
    INTERFACE = "interface"
    OPERATOR = "operator"
    DATAFLOW = "dataflow"
    REDUCTION = "reduction"
    PREDICATE_EFFECT = "predicate_effect"
    SCHEDULE = "schedule"
    NORMALIZATION_GAP = "normalization_gap"
    COMPARATOR_BLIND_SPOT = "comparator_blind_spot"
    J_SELF_CONSISTENT_ERROR = "j_self_consistent_error"
    LOWERING = "lowering"
    UNKNOWN = "unknown"


class OutcomeClass(str, Enum):
    TRUE_ACCEPT = "gate1_closed_gate2_pass"
    FALSE_ACCEPT = "gate1_closed_gate2_fail"
    FALSE_REJECT = "gate1_rejected_gate2_pass"
    TRUE_REJECT = "gate1_rejected_gate2_fail"
    UNLABELED = "gate2_unknown"


@dataclass(frozen=True)
class CapabilityObservation:
    case_id: str
    gate1_closed: bool
    gate2_pass: bool | None
    gate1_failure: FailureCause | None = None
    expected_cause: FailureCause | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> OutcomeClass:
        if self.gate2_pass is None:
            return OutcomeClass.UNLABELED
        if self.gate1_closed:
            return (
                OutcomeClass.TRUE_ACCEPT
                if self.gate2_pass
                else OutcomeClass.FALSE_ACCEPT
            )
        return (
            OutcomeClass.FALSE_REJECT
            if self.gate2_pass
            else OutcomeClass.TRUE_REJECT
        )


@dataclass(frozen=True)
class FailureAnalysis:
    case_id: str
    outcome: OutcomeClass
    primary_cause: FailureCause
    explanation: str
    priority: str


@dataclass(frozen=True)
class CapabilitySummary:
    total: int
    labeled: int
    gate1_closed: int
    gate2_pass: int
    true_accept: int
    false_accept: int
    false_reject: int
    true_reject: int
    gate2_precision_given_closed: float | None
    gate1_coverage_of_gate2_pass: float | None
    outcome_counts: dict[str, int]
    failure_cause_counts: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "labeled": self.labeled,
            "gate1_closed": self.gate1_closed,
            "gate2_pass": self.gate2_pass,
            "true_accept": self.true_accept,
            "false_accept": self.false_accept,
            "false_reject": self.false_reject,
            "true_reject": self.true_reject,
            "gate2_precision_given_closed": self.gate2_precision_given_closed,
            "gate1_coverage_of_gate2_pass": self.gate1_coverage_of_gate2_pass,
            "outcome_counts": dict(self.outcome_counts),
            "failure_cause_counts": dict(self.failure_cause_counts),
        }


def summarize_capability(
    observations: Iterable[CapabilityObservation],
) -> CapabilitySummary:
    items = list(observations)
    labeled = [item for item in items if item.gate2_pass is not None]
    outcomes = {kind: 0 for kind in OutcomeClass}
    causes = {cause: 0 for cause in FailureCause}
    for item in items:
        outcomes[item.outcome] += 1
        if item.gate1_failure is not None:
            causes[item.gate1_failure] += 1

    true_accept = outcomes[OutcomeClass.TRUE_ACCEPT]
    false_accept = outcomes[OutcomeClass.FALSE_ACCEPT]
    false_reject = outcomes[OutcomeClass.FALSE_REJECT]
    true_reject = outcomes[OutcomeClass.TRUE_REJECT]
    accepted_labeled = true_accept + false_accept
    gate2_pass = true_accept + false_reject
    precision = (
        true_accept / accepted_labeled if accepted_labeled else None
    )
    coverage = true_accept / gate2_pass if gate2_pass else None
    return CapabilitySummary(
        total=len(items),
        labeled=len(labeled),
        gate1_closed=sum(item.gate1_closed for item in items),
        gate2_pass=gate2_pass,
        true_accept=true_accept,
        false_accept=false_accept,
        false_reject=false_reject,
        true_reject=true_reject,
        gate2_precision_given_closed=precision,
        gate1_coverage_of_gate2_pass=coverage,
        outcome_counts={key.value: value for key, value in outcomes.items()},
        failure_cause_counts={
            key.value: value for key, value in causes.items() if value
        },
    )


def analyze_failure(observation: CapabilityObservation) -> FailureAnalysis | None:
    """Translate a non-ideal outcome into an actionable root-cause hypothesis."""

    outcome = observation.outcome
    if outcome in {OutcomeClass.TRUE_ACCEPT, OutcomeClass.UNLABELED}:
        return None

    if outcome is OutcomeClass.FALSE_ACCEPT:
        cause = observation.gate1_failure or FailureCause.COMPARATOR_BLIND_SPOT
        return FailureAnalysis(
            observation.case_id,
            outcome,
            cause,
            "Gate1 判闭合但 Gate2 失败：优先检查指纹是否遗漏运行时值、"
            "launch/grid、别名/副作用或同算子不同数据流；若重放与 lowering "
            "一致而源偏离，则属于 J 与 π 自洽地理解错。",
            "critical",
        )

    if outcome is OutcomeClass.FALSE_REJECT:
        cause = (
            observation.expected_cause
            or observation.gate1_failure
            or FailureCause.NORMALIZATION_GAP
        )
        if cause in {
            FailureCause.ARTIFACT_SCHEMA,
            FailureCause.PLAN_LEAKAGE,
            FailureCause.RULE_CARD,
            FailureCause.REPLAY_POLICY,
            FailureCause.INTERFACE,
        }:
            explanation = (
                "Gate2 在当前输入上通过，但 Gate1 因 SCRC 的接口签名或信息"
                "分区规则拒绝。若是 π/API/source 泄漏，这是有意拒绝：否则无法"
                "证明语义来自 J；不能把它当成普通 TTIR 规范化缺口。"
            )
        else:
            explanation = (
                "Gate2 行为正确但 Gate1 拒绝：通常是有限规范化规则不足、"
                "规则卡过窄或合法代数分解不同。它损失覆盖率，但不直接降低"
                "已闭合样本的 Gate2 条件精确率；只应根据该真实 diff 增加有限规则。"
            )
        return FailureAnalysis(
            observation.case_id,
            outcome,
            cause,
            explanation,
            "medium",
        )

    cause = observation.gate1_failure or observation.expected_cause or FailureCause.UNKNOWN
    return FailureAnalysis(
        observation.case_id,
        outcome,
        cause,
        "Gate1 拒绝且 Gate2 也失败：这是正确拒绝。根因应按首个硬失败项归入"
        " J、π、接口/C、重放编译、归约、数据流或 predicate/effect。",
        "low",
    )
