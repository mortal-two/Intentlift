from __future__ import annotations

import pytest

from intentir.backend.ablation import (
    ExperimentArm,
    assert_fair_pair,
    standard_ablation_plan,
)


def _arms(budget: int = 24):
    return {item.arm: item for item in standard_ablation_plan(budget)}


def test_standard_plan_freezes_equal_search_budgets() -> None:
    arms = _arms(17)

    assert arms[ExperimentArm.E0_SOURCE].candidate_budget == 0
    assert arms[ExperimentArm.E1_CANONICAL].candidate_budget == 0
    assert all(
        item.candidate_budget == 17
        for arm, item in arms.items()
        if arm not in {ExperimentArm.E0_SOURCE, ExperimentArm.E1_CANONICAL}
    )


def test_e3_e4_differ_only_by_erasable_c_prior() -> None:
    arms = _arms()

    assert_fair_pair(
        arms[ExperimentArm.E3_STATE_SEARCH],
        arms[ExperimentArm.E4_ERASE_C],
        varying="use_c_prior",
    )
    assert arms[ExperimentArm.E3_STATE_SEARCH].use_c_prior
    assert not arms[ExperimentArm.E4_ERASE_C].use_c_prior


def test_l0_l1_differ_only_by_local_b_transformations() -> None:
    arms = _arms()

    assert_fair_pair(
        arms[ExperimentArm.L0_LOCAL_OFF],
        arms[ExperimentArm.L1_LOCAL_ON],
        varying="enable_local_b_transformations",
    )


def test_numeric_arms_are_monotonic_concrete_state_expansions() -> None:
    arms = _arms()
    n0 = arms[ExperimentArm.N0_NUMERIC_OFF]
    n1 = arms[ExperimentArm.N1_REDUCTION_TREE]
    n2 = arms[ExperimentArm.N2_FMA]
    n3 = arms[ExperimentArm.N3_PRECISION_APPROX]

    assert n0.numeric_state_level == "off"
    assert n1.search_reduction_topology and not n1.search_fma
    assert n2.search_reduction_topology and n2.search_fma
    assert n3.search_precision and n3.search_approximation


def test_fair_pair_rejects_hidden_budget_changes() -> None:
    arms = _arms()
    changed = type(arms[ExperimentArm.E4_ERASE_C])(
        **{
            **arms[ExperimentArm.E4_ERASE_C].__dict__,
            "candidate_budget": 25,
        }
    )

    with pytest.raises(ValueError, match="candidate_budget"):
        assert_fair_pair(
            arms[ExperimentArm.E3_STATE_SEARCH],
            changed,
            varying="use_c_prior",
        )
