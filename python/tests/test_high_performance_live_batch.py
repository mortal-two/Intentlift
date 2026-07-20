from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


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
    assert all(case.origin["capture_abi_exact"] is True for case in cases)


def test_tvd_capture_preserves_the_interleaved_constexpr_abi() -> None:
    case = next(
        item for item in batch.load_cases()
        if item.origin["kernel_id"] == "high_performance:liger_tvd"
    )

    assert case.parameter_order[8:15] == (
        "label_ptr", "ignore_index", "n_cols", "scale",
        "BLOCK_SIZE", "HAS_LABEL", "reduction",
    )
    assert case.constexpr_parameters == (
        "ignore_index", "BLOCK_SIZE", "HAS_LABEL", "reduction"
    )


def test_capture_abi_mismatch_fails_before_provider_use(tmp_path: Path) -> None:
    source = tmp_path / "captured.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def kernel(x, FLAG: tl.constexpr, out):\n"
        "    pass\n",
        encoding="utf-8",
    )
    launch = {
        "preflight": {
            "python_interface": {"arg_names": ["x", "FLAG", "out"]}
        },
        "parameters": {"x": {}, "FLAG": {}, "out": {}},
    }
    compiler = {
        "parameter_order": ["x", "out", "FLAG"],
        "arguments": [{"name": "x"}, {"name": "out"}],
        "constexpr_parameters": ["FLAG"],
    }

    with pytest.raises(ValueError, match="ABI order mismatch before LLM"):
        batch._validate_capture_abi(
            kernel_id="high_performance:synthetic",
            function_name="kernel",
            source_path=source,
            launch=launch,
            compiler=compiler,
        )


def test_official_campaign_configuration_is_not_cli_overridable() -> None:
    parsed = batch._parser().parse_args([])
    official = batch._official_args(parsed)

    assert official.model == "gpt-5.6-sol"
    assert official.effort == "xhigh"
    assert official.gate1_profile == "formal"
    assert not hasattr(official, "lowering_backend")
    assert official.semantic_interventions == 0
    assert not official.pi_ablations
    assert not hasattr(official, "diagnose_rejected")


def test_fresh_contract_rejects_wrong_model_cache_and_gate2_protocol() -> None:
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
        "gate2": {
            "execution_protocol": "legacy_three_way",
            "status": "gate2_pass",
        },
        "gate2_execution_policy": {
            "protocol": "legacy_three_way",
            "legs": ["source", "replay", "lowering"],
            "target_backend_in_verdict": True,
        },
    }

    passed, failures = batch._fresh_contract(report)

    assert not passed
    assert set(failures) == {
        "lift_was_reused",
        "response_model_mismatch",
        "missing_provider_response_id",
        "requested_model_mismatch",
        "requested_effort_mismatch",
        "gate2_execution_protocol_mismatch",
        "gate2_execution_legs_mismatch",
        "target_backend_entered_gate2_verdict",
        "gate2_required_comparisons_mismatch",
        "gate2_evidence_protocol_mismatch",
        "missing_gate2_scalar_binding",
        "missing_source_execution_evidence",
        "missing_replay_execution_evidence",
        "missing_source_replay_comparison_evidence",
        "missing_source_scalar_comparison_evidence",
        "missing_scalar_execution_metadata",
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
        "gate2": {
            "execution_protocol": batch.GATE2_EXECUTION_PROTOCOL,
            "status": "gate2_pass",
            "source": {"ok": True},
            "replay": {"ok": True},
            "scalar": {
                "ok": True,
                "metadata": {
                    "protocol": batch.SCALAR_LOWERING_PROTOCOL,
                    "target_backend_in_verdict": False,
                    "execution_evidence": {
                        "schema": batch.CPU_EXECUTION_EVIDENCE_SCHEMA,
                        "abi_protocol": batch.CPU_MEMREF_ABI_PROTOCOL,
                        "runtime_protocol": batch.CPU_RUNTIME_PROTOCOL,
                        "native_call_count": 1,
                        "isolated_process": True,
                        "module_depends_on_runtime_values": False,
                    },
                },
            },
            "source_replay": {"passed": True},
            "source_scalar": {"passed": True},
        },
        "gate2_execution_policy": {
            "protocol": batch.GATE2_EXECUTION_PROTOCOL,
            "legs": ["source", "replay", "scalar"],
            "required_comparisons": ["source_replay", "source_scalar"],
            "target_backend_in_verdict": False,
        },
        "gate2_scalar_binding": {"intent_program_digest": "typed"},
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
        "gate2": {
            "execution_protocol": "source_replay_dynamic_equivalence_v1",
            "status": "gate2_pass",
        },
        "gate2_execution_policy": {
            "protocol": "source_replay_dynamic_equivalence_v1",
            "legs": ["source", "replay"],
            "production_lowering_in_verdict": False,
        },
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
