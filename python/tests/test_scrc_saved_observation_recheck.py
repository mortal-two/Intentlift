from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

import scrc_saved_observation_recheck as recheck  # noqa: E402
from scrc_saved_observation_recheck import (  # noqa: E402
    rehydrate_saved_artifact,
    run_saved_gate2,
    run_saved_gate2_suite,
    saved_anti_leakage_sources,
    select_saved_candidate,
)


def test_saved_candidate_prefers_latest_hard_closed_observation() -> None:
    candidates = [
        {
            "relift_round": 0,
            "artifact_digest": "final-artifact",
            "generation": {"code": "first"},
            "comparison": {"hard_pass": True},
        },
        {
            "relift_round": 1,
            "artifact_digest": "different-artifact",
            "generation": {"code": "failed"},
            "comparison": {"hard_pass": False},
        },
        {
            "relift_round": 2,
            "artifact_digest": "final-artifact",
            "generation": {"code": "latest-closed"},
            "comparison": {"hard_pass": True},
        },
    ]

    index, selected = select_saved_candidate(
        candidates,
        final_artifact_digest="final-artifact",
    )
    assert index == 2
    assert selected["generation"]["code"] == "latest-closed"


def test_saved_candidate_uses_last_candidate_in_final_failed_round() -> None:
    candidates = [
        {
            "relift_round": 0,
            "artifact_digest": "old-artifact",
            "generation": {"code": "old"},
            "comparison": {"hard_pass": False},
        },
        {
            "relift_round": 1,
            "artifact_digest": "final-artifact",
            "generation": {"code": "final-a"},
            "comparison": {"hard_pass": False},
        },
        {
            "relift_round": 1,
            "artifact_digest": "final-artifact",
            "generation": {"code": "final-b"},
            "comparison": {"hard_pass": False},
        },
    ]

    index, selected = select_saved_candidate(
        candidates,
        final_artifact_digest="final-artifact",
    )
    assert index == 2
    assert selected["generation"]["code"] == "final-b"


def test_saved_candidate_does_not_pair_old_closed_code_with_new_artifact() -> None:
    candidates = [
        {
            "relift_round": 0,
            "artifact_digest": "old-artifact",
            "generation": {"code": "old-closed"},
            "comparison": {"hard_pass": True},
        },
        {
            "relift_round": 1,
            "artifact_digest": "final-artifact",
            "generation": {"code": "final-failed"},
            "comparison": {"hard_pass": False},
        },
    ]

    index, selected = select_saved_candidate(
        candidates,
        final_artifact_digest="final-artifact",
    )
    assert index == 1
    assert selected["generation"]["code"] == "final-failed"


def test_saved_candidate_rejects_reports_without_replay_source() -> None:
    with pytest.raises(ValueError, match="no replay source"):
        select_saved_candidate(
            [{"generation": None}],
            final_artifact_digest="final-artifact",
        )


def test_saved_candidate_rejects_old_replay_without_final_artifact_binding() -> None:
    with pytest.raises(ValueError, match="bound to the final artifact"):
        select_saved_candidate(
            [
                {
                    "relift_round": 0,
                    "artifact_digest": "artifact-before-relift",
                    "generation": {"code": "old-compiled"},
                    "comparison": {"hard_pass": True},
                }
            ],
            final_artifact_digest="artifact-after-relift",
        )


def test_saved_candidate_accepts_legacy_same_round_shadow_binding() -> None:
    provenance = {
        "composer_version": "deterministic_composer_v2",
        "j_digest": "j",
        "interface_digest": "interface",
        "schedule_digest": "schedule",
        "strategy_digest": "strategy",
        "registry_digest": "registry",
        "plan_digest": "plan",
    }
    digest = recheck._artifact_digest_from_compose_provenance(provenance)
    assert digest is not None

    index, selected = select_saved_candidate(
        [
            {
                "relift_round": 0,
                "generation": {"code": "legacy-aligned"},
                "comparison": {"hard_pass": True},
            }
        ],
        final_artifact_digest=digest,
        composer_shadow=(
            {
                "relift_round": 0,
                "status": "match",
                "compose_provenance": provenance,
            },
        ),
    )

    assert index == 0
    assert selected["generation"]["code"] == "legacy-aligned"


def test_saved_anti_leakage_sources_preserve_saved_arm_order() -> None:
    mutants, masked = saved_anti_leakage_sources(
        {
            "interventions": [
                {"generation": {"code": "MUTANT_A"}},
                {"generation": {"code": "MUTANT_B"}},
            ],
            "masked_generation": {"code": "MASKED"},
        }
    )

    assert mutants == ("MUTANT_A", "MUTANT_B")
    assert masked == "MASKED"


def test_saved_anti_leakage_sources_require_auditable_masked_arm() -> None:
    with pytest.raises(ValueError, match="masked-J replay source"):
        saved_anti_leakage_sources(
            {"interventions": [], "masked_generation": None}
        )


def test_saved_artifact_rehydrates_actual_source_without_metadata_fields(
    monkeypatch,
) -> None:
    captured = {}

    def fake_from_json(payload):
        captured.update(payload)
        return payload

    monkeypatch.setattr(
        recheck.Gate1Artifact,
        "from_json",
        staticmethod(fake_from_json),
    )
    payload = {
        "j": {"kernel_id": "saved"},
        "source_sha256": "digest",
        "source_bytes": 123,
    }

    result = rehydrate_saved_artifact(payload, source="ACTUAL_SOURCE")

    assert result == captured
    assert captured["source"] == "ACTUAL_SOURCE"
    assert "source_sha256" not in captured
    assert "source_bytes" not in captured
    assert payload["source_sha256"] == "digest"


def test_saved_gate2_requires_current_digest_bound_reacceptance(
    monkeypatch,
) -> None:
    captured = {}
    case = SimpleNamespace(kernel_source="SOURCE")
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name="kernel")
    )

    class FakeRunner:
        def run(self, replay, gate2_case):
            captured.update(
                replay=replay,
                gate2_case=gate2_case,
            )
            return SimpleNamespace(status=SimpleNamespace(value="gate2_pass"))

    monkeypatch.setattr(
        recheck, "_gate2_case", lambda *args, **kwargs: "GATE2_CASE"
    )
    monkeypatch.setattr(
        recheck,
        "_gate2_evidence_json",
        lambda result: {"status": result.status.value},
    )

    payload = run_saved_gate2(
        case,
        artifact,
        "REPLAY",
        seed=7,
        final_artifact_digest="a" * 64,
        gate1_j_digest="b" * 64,
        program="PROGRAM",
        runner=FakeRunner(),
    )

    assert payload == {
        "saved_observation_recheck": True,
        "required_gate1_state": "replay_closed",
        "status": "gate2_pass",
    }
    assert captured["gate2_case"] == "GATE2_CASE"


def test_saved_gate2_suite_uses_current_digest_bound_matrix(
    monkeypatch,
) -> None:
    captured = {}
    case = SimpleNamespace(kernel_source="SOURCE")
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name="kernel")
    )

    class FakeRunner:
        def run_suite(self, replay, suite):
            captured.update(replay=replay, suite=suite)
            return SimpleNamespace(passed=True)

    monkeypatch.setattr(
        recheck, "_gate2_suite", lambda *args, **kwargs: "SUITE"
    )
    monkeypatch.setattr(
        recheck,
        "_gate2_suite_evidence_json",
        lambda result: {"passed": result.passed, "cases": []},
    )

    payload = run_saved_gate2_suite(
        case,
        artifact,
        "REPLAY",
        seeds=(1, 2, 3),
        final_artifact_digest="a" * 64,
        gate1_j_digest="b" * 64,
        program="PROGRAM",
        runner=FakeRunner(),
    )

    assert payload == {
        "saved_observation_recheck": True,
        "required_gate1_state": "replay_closed",
        "passed": True,
        "cases": [],
    }
    assert captured["suite"] == "SUITE"
