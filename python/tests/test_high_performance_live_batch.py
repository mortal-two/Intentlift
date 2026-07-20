from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test/high_performance_live_batch.py"
SPEC = importlib.util.spec_from_file_location("high_performance_live_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def test_official_high_performance_campaign_resolves_28_authentic_sources() -> None:
    cases = batch.load_cases()

    assert len(cases) == 28
    assert len({case.function_name for case in cases}) == 28
    assert all(case.source_path.is_file() for case in cases)
    assert all(case.compiler_view().grid for case in cases)


def test_official_campaign_configuration_is_not_cli_overridable() -> None:
    parsed = batch._parser().parse_args([])
    official = batch._official_args(parsed)

    assert official.model == "gpt-5.6-sol"
    assert official.effort == "xhigh"
    assert official.gate1_profile == "formal"
    assert official.lowering_backend == "cuda"
    assert official.semantic_interventions == 0
    assert not official.pi_ablations
    assert not official.diagnose_rejected


def test_fresh_contract_rejects_wrong_model_cache_and_missing_cuda() -> None:
    report = {
        "model": "gpt-5.5",
        "effort": "high",
        "lift": {
            "ok": True,
            "reused": True,
            "response_model": "gpt-5.5",
            "response_ids": [],
            "usage": {},
        },
        "gate1": {
            "status": "replay_closed",
            "interventions": [],
            "masked_generation": {
                "masked": True,
                "response_id": "resp_masked",
                "requested_model": "gpt-5.6-sol",
                "response_model": "gpt-5.6-sol",
            },
            "masked_outcome": "hard_mismatch",
        },
        "gate1_anti_leakage_policy": {
            "paired_semantic_interventions": False,
            "maximum_interventions": 0,
            "pi_ablations": {"enabled": False},
        },
        "gate2": {"status": "gate2_pass"},
        "gate2_lowering_policy": {"backend": "cpu"},
    }

    passed, failures = batch._fresh_contract(report)

    assert not passed
    assert set(failures) == {
        "lift_was_reused",
        "response_model_mismatch",
        "missing_provider_response_id",
        "requested_model_mismatch",
        "requested_effort_mismatch",
        "not_cuda_lowering",
    }


def test_fresh_contract_accepts_only_the_locked_full_pipeline() -> None:
    report = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "provider_request": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "xhigh",
            "api_key_recorded": False,
        },
        "lift": {
            "ok": True,
            "response_model": "gpt-5.6-sol",
            "response_ids": ["resp_fresh"],
            "usage": {"total_tokens": 1},
        },
        "gate1": {"status": "replay_closed"},
        "gate1_anti_leakage_policy": {
            "paired_semantic_interventions": False,
            "maximum_interventions": 0,
            "pi_ablations": {"enabled": False},
        },
        "gate2": {"status": "gate2_pass"},
        "gate2_lowering_policy": {"backend": "cuda"},
    }

    report["gate1"].update(
        {
            "interventions": [],
            "masked_generation": {
                "masked": True,
                "response_id": "resp_masked",
                "requested_model": "gpt-5.6-sol",
                "response_model": "gpt-5.6-sol",
            },
            "masked_outcome": "hard_mismatch",
        }
    )

    assert batch._fresh_contract(report) == (True, [])


def test_fresh_contract_rejects_interventions_or_missing_masked_j() -> None:
    report = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "lift": {
            "ok": True,
            "response_model": "gpt-5.6-sol",
            "response_ids": ["resp_fresh"],
            "usage": {"total_tokens": 1},
        },
        "gate1": {
            "status": "replay_closed",
            "interventions": [{"status": "pass"}],
            "masked_generation": None,
            "masked_outcome": None,
        },
        "gate1_anti_leakage_policy": {
            "paired_semantic_interventions": True,
            "maximum_interventions": 3,
            "pi_ablations": {"enabled": True},
        },
        "pi_ablations": [{"status": "rejected"}],
        "gate2": {"status": "gate2_pass"},
        "gate2_lowering_policy": {"backend": "cuda"},
    }

    passed, failures = batch._fresh_contract(report)

    assert not passed
    assert {
        "paired_interventions_present",
        "missing_masked_j_generation",
        "masked_j_did_not_nonclose",
        "paired_interventions_not_disabled",
        "paired_intervention_budget_not_zero",
        "pi_ablations_not_disabled",
        "pi_ablation_evidence_present",
    }.issubset(failures)


def test_performance_state_requires_measured_correct_output() -> None:
    report = {
        "gpu_performance": {
            "status": "measured",
            "correctness_pass": True,
            "comparison": {
                "speedup_source_over_generated": 1.04,
                "non_regression_2pct_pass": True,
                "extension_3pct_speedup_pass": True,
            },
        }
    }

    assert batch._performance_state(report) == {
        "measured": True,
        "speedup_source_over_generated": 1.04,
        "non_regression_2pct_pass": True,
        "extension_3pct_speedup_pass": True,
    }
    report["gpu_performance"]["correctness_pass"] = False
    assert not batch._performance_state(report)["measured"]
