from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentir.backend.ablation import (
    ExperimentArm,
    ablation_spec_digest,
    assert_fair_pair,
    load_ablation_spec,
    standard_ablation_plan,
)


def _arms(budget: int = 24):
    return {item.arm: item for item in standard_ablation_plan(budget)}


def test_standard_plan_freezes_equal_search_budgets() -> None:
    arms = _arms(17)

    assert arms[ExperimentArm.E0_SOURCE].candidate_budget == 0
    assert arms[ExperimentArm.E1_CANONICAL].candidate_budget == 0
    assert arms[ExperimentArm.C_FIXED].candidate_budget == 1
    assert arms[ExperimentArm.C_FIXED].baseline_kind == "c_fixed"
    assert all(
        item.candidate_budget == 17
        for arm, item in arms.items()
        if arm
        not in {
            ExperimentArm.E0_SOURCE,
            ExperimentArm.E1_CANONICAL,
            ExperimentArm.C_FIXED,
        }
    )


def test_machine_spec_is_the_consumed_qd_experiment_contract() -> None:
    spec = load_ablation_spec()

    assert len(ablation_spec_digest()) == 64
    assert spec["correctness"]["search_domain_edge"] == "target_ref_j"
    assert spec["correctness"]["certification_domain_edges"] == [
        "target_source",
        "target_replay",
        "target_ref_j",
    ]
    assert spec["candidate_identity"]["deduplication_key"] == (
        "canonical post-pass MLIR digest"
    )
    assert spec["measurement"]["confirmation_rule"].startswith(
        "equal-count fresh matched-baseline and target"
    )
    assert spec["measurement"]["formal_panels"]["cuda_external"] == [
        "Source",
        "IntentIR",
    ]
    assert spec["measurement"]["formal_panels"]["rvv_external"] == [
        "Native RVV",
        "IntentIR",
    ]
    assert spec["measurement"]["formal_panels"]["internal_attribution"] == [
        "C-Fixed",
        "Schedule-Only",
        "IntentIR",
    ]
    assert _arms()[ExperimentArm.E3_STATE_SEARCH].to_dict()[
        "experiment_name"
    ] == "IntentIR"
    assert {item["arm"] for item in spec["arms"]} == {
        item.value for item in ExperimentArm
    }


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


def test_machine_spec_rejects_non_boolean_axis(tmp_path: Path) -> None:
    spec = load_ablation_spec()
    spec["arms"][2]["search_fma"] = 0
    path = tmp_path / "non-boolean.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="must be boolean"):
        standard_ablation_plan(8, spec_path=path)


def test_machine_spec_rejects_qd_protocol_drift(tmp_path: Path) -> None:
    spec = load_ablation_spec()
    spec["correctness"]["search_domain_edge"] = "target_source"
    path = tmp_path / "bad-qd.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="Q/D implementation"):
        standard_ablation_plan(8, spec_path=path)


def test_machine_spec_rejects_unknown_root_field(tmp_path: Path) -> None:
    spec = load_ablation_spec()
    spec["posthoc_override"] = True
    path = tmp_path / "unknown-root.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="root fields"):
        standard_ablation_plan(8, spec_path=path)
