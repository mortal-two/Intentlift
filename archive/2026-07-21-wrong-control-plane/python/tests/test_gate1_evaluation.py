from intentir.verify.gate1.evaluation import (
    CapabilityObservation,
    FailureCause,
    OutcomeClass,
    analyze_failure,
    summarize_capability,
)


def test_capability_summary_prioritizes_conditional_gate2_precision() -> None:
    observations = [
        CapabilityObservation("ok", True, True),
        CapabilityObservation("caught", False, False, FailureCause.REDUCTION),
        CapabilityObservation("conservative", False, True, FailureCause.NORMALIZATION_GAP),
        CapabilityObservation("blind", True, False),
    ]
    summary = summarize_capability(observations)
    assert summary.gate2_precision_given_closed == 0.5
    assert summary.gate1_coverage_of_gate2_pass == 0.5
    assert summary.true_accept == 1
    assert summary.false_accept == 1
    assert summary.false_reject == 1
    assert summary.true_reject == 1


def test_failure_analysis_distinguishes_precision_and_coverage_failures() -> None:
    false_accept = CapabilityObservation("fa", True, False)
    false_reject = CapabilityObservation("fr", False, True)
    true_reject = CapabilityObservation(
        "tr", False, False, FailureCause.PREDICATE_EFFECT
    )

    fa = analyze_failure(false_accept)
    fr = analyze_failure(false_reject)
    tr = analyze_failure(true_reject)
    assert fa is not None and fa.outcome is OutcomeClass.FALSE_ACCEPT
    assert fa.priority == "critical"
    assert fr is not None and fr.primary_cause is FailureCause.NORMALIZATION_GAP
    assert fr.priority == "medium"
    assert tr is not None and tr.primary_cause is FailureCause.PREDICATE_EFFECT
    assert tr.priority == "low"


def test_curated_expected_cause_separates_policy_reject_from_normalization() -> None:
    normalization = analyze_failure(
        CapabilityObservation(
            "equivalent",
            False,
            True,
            FailureCause.OPERATOR,
            FailureCause.NORMALIZATION_GAP,
        )
    )
    policy = analyze_failure(
        CapabilityObservation(
            "leak",
            False,
            True,
            FailureCause.PLAN_LEAKAGE,
            FailureCause.PLAN_LEAKAGE,
        )
    )

    assert normalization is not None
    assert normalization.primary_cause is FailureCause.NORMALIZATION_GAP
    assert "真实 diff" in normalization.explanation
    assert policy is not None
    assert policy.primary_cause is FailureCause.PLAN_LEAKAGE
    assert "语义来自 J" in policy.explanation
