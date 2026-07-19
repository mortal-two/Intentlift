from __future__ import annotations

import json
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.model import (
    Gate1Status,
    KernelParameter,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.replay import ReplayGeneration
from intentir.verify.gate2 import Gate2Status


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

import scrc_live_e2e as live  # noqa: E402
from scrc_live_e2e import (  # noqa: E402
    Gate2ShapeProfile,
    _compile_case,
    _gate1_family_evidence_json,
    _gate1_specialization_family,
    _gate1_specialization_policy_json,
    _gate2_case,
    _gate1_evidence_json,
    _select_gate2_replay_observation,
    _select_gate1_specialization_evidence,
    _lift_attempt_evidence,
    _parser,
    _pi_ablation_control_evidence,
    _pi_ablation_evidence_json,
    _relift_evidence_json,
    _write_report,
    build_case,
)


def _args(kernel: str, **overrides) -> Namespace:
    values = {
        "kernel": kernel,
        "m": 7,
        "n": 1003,
        "k": 19,
        "block": None,
        "block_m": 16,
        "block_n": 16,
        "block_k": 16,
        "num_warps": 4,
        "num_stages": 2,
        "eps": 1.0e-5,
    }
    values.update(overrides)
    return Namespace(**values)


def test_row_sum_case_carries_full_trusted_compiler_view() -> None:
    case = build_case(_args("row_sum"))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "x_ptr",
        "out_ptr",
        "M",
        "N",
    ]
    assert view["arguments"][0]["shape"] == [7, 1003]
    assert view["arguments"][1]["shape"] == [7]
    assert view["grid"] == [7]
    assert view["grid_contract"] == ["M"]
    assert view["parameter_order"] == [
        "x_ptr",
        "out_ptr",
        "M",
        "N",
        "BLOCK_N",
    ]
    assert view["constexpr_parameters"] == ["BLOCK_N"]
    assert view["schedule_values"]["BLOCK_N"] == 1024
    assert compile_case.grid == (7,)
    assert compile_case.arguments["M"].dtype == "int32"


def test_row_sum_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("row_sum", block=512))


def test_softmax_case_preserves_two_dimensional_output_and_tail_tile() -> None:
    case = build_case(_args("softmax", m=4, n=37, block=64))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["shape"] for item in view["arguments"][:2]] == [
        [4, 37],
        [4, 37],
    ]
    assert view["grid"] == [4]
    assert view["grid_contract"] == ["M"]
    assert view["schedule_values"]["BLOCK_N"] == 64
    assert compile_case.arguments["N"].value == 37
    assert case.tolerance.rtol == 2.0e-4
    assert case.tolerance.atol == 2.0e-5


def test_softmax_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("softmax", block=512))


def test_rmsnorm_case_preserves_runtime_eps_and_tail_tile() -> None:
    case = build_case(_args("rmsnorm", m=4, n=37, block=64, eps=1.0e-5))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "x_ptr",
        "w_ptr",
        "out_ptr",
        "M",
        "N",
        "eps",
    ]
    assert [item["shape"] for item in view["arguments"][:3]] == [
        [4, 37],
        [37],
        [4, 37],
    ]
    assert view["parameter_order"] == [
        "x_ptr",
        "w_ptr",
        "out_ptr",
        "M",
        "N",
        "eps",
        "BLOCK_N",
    ]
    assert view["constexpr_parameters"] == ["BLOCK_N"]
    assert view["schedule_values"]["BLOCK_N"] == 64
    assert compile_case.arguments["eps"].dtype == "float32"
    assert compile_case.arguments["eps"].value == 1.0e-5
    assert case.tolerance.rtol == 3.0e-4
    assert case.tolerance.atol == 3.0e-5


def test_rmsnorm_case_rejects_multi_program_column_reduction() -> None:
    with pytest.raises(ValueError, match="one BLOCK_N tile"):
        build_case(_args("rmsnorm", block=512))


def test_rmsnorm_gate2_case_binds_eps_as_f32_semantic_scalar() -> None:
    case = build_case(_args("rmsnorm", m=4, n=37, block=64, eps=1.0e-5))
    intent = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/rms_norm/intent.json").read_text())
    )
    parameters = [
        KernelParameter(
            "x_ptr", ParameterKind.POINTER, ParameterMode.INPUT,
            "f32", 2, tensor_ref="X",
        ),
        KernelParameter(
            "w_ptr", ParameterKind.POINTER, ParameterMode.INPUT,
            "f32", 1, tensor_ref="W",
        ),
        KernelParameter(
            "out_ptr", ParameterKind.POINTER, ParameterMode.OUTPUT,
            "f32", 2, tensor_ref="Y",
        ),
        KernelParameter("M", ParameterKind.SCALAR, dtype="i32", rank=0),
        KernelParameter("N", ParameterKind.SCALAR, dtype="i32", rank=0),
        KernelParameter(
            "eps", ParameterKind.SCALAR, dtype="f32", rank=0,
            tensor_ref="eps",
        ),
        KernelParameter("BLOCK_N", ParameterKind.CONSTEXPR, dtype="i32", rank=0),
    ]
    artifact = SimpleNamespace(
        interface=SimpleNamespace(parameters=parameters),
        schedule=SimpleNamespace(values=case.schedule_values),
        j=SimpleNamespace(intent=intent),
    )

    gate2 = _gate2_case(artifact, case, 20260715)

    assert isinstance(gate2.values["eps"], np.float32)
    assert gate2.values["eps"] == np.float32(1.0e-5)
    eps_arg = next(item for item in gate2.triton.arguments if item.name == "eps")
    assert eps_arg.value_key == "eps"
    assert gate2.lowering.input_bindings["eps"] == "eps"
    assert gate2.lowering.dims == {"M": 4, "N": 37}


def test_matmul_case_preserves_runtime_dims_three_c_slots_and_2d_grid() -> None:
    case = build_case(_args("matmul", m=17, n=13, k=19))
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert [item["name"] for item in view["arguments"]] == [
        "a_ptr",
        "b_ptr",
        "out_ptr",
        "M",
        "N",
        "K",
    ]
    assert [item["dtype"] for item in view["arguments"][:3]] == [
        "f16",
        "f16",
        "f32",
    ]
    assert [item["shape"] for item in view["arguments"][:3]] == [
        [17, 19],
        [19, 13],
        [17, 13],
    ]
    assert view["grid"] == [2, 1]
    assert view["grid_contract"] == [
        "ceil_div(M, BLOCK_M)",
        "ceil_div(N, BLOCK_N)",
    ]
    assert view["parameter_order"] == [
        "a_ptr",
        "b_ptr",
        "out_ptr",
        "M",
        "N",
        "K",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
    ]
    assert view["constexpr_parameters"] == [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
    ]
    assert view["schedule_values"] == {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 16,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert compile_case.arguments["M"].value == 17
    assert compile_case.arguments["N"].value == 13
    assert compile_case.arguments["K"].value == 19
    assert case.tolerance.rtol == 2.0e-3
    assert case.tolerance.atol == 2.0e-3


def test_matmul_case_rejects_nonpositive_logical_or_block_extent() -> None:
    with pytest.raises(ValueError, match="K must be positive"):
        build_case(_args("matmul", k=0))
    with pytest.raises(ValueError, match="block sizes must be positive"):
        build_case(_args("matmul", block_k=0))


def test_matmul_entry_uses_reviewed_probe_defaults() -> None:
    args = _parser("matmul").parse_args([])
    assert (args.m, args.n, args.k) == (17, 13, 19)
    assert (args.block_m, args.block_n, args.block_k) == (16, 16, 16)
    assert args.model == "gpt-5.6-sol"
    assert args.effort == "xhigh"
    assert args.relift_rounds == 3


def test_live_parser_rejects_nonpositive_relift_rounds() -> None:
    with pytest.raises(SystemExit):
        _parser("add").parse_args(["--relift-rounds", "0"])


def test_live_case_builds_an_ordered_fixed_c_family() -> None:
    base = build_case(_args("add"))
    profiles = (
        Gate2ShapeProfile(
            "one",
            {"a_ptr": (1,), "b_ptr": (1,), "out_ptr": (1,)},
            {"n_elements": 1},
            (1,),
        ),
        Gate2ShapeProfile(
            "contract",
            {
                "a_ptr": (1003,),
                "b_ptr": (1003,),
                "out_ptr": (1003,),
            },
            {"n_elements": 1003},
            (4,),
        ),
    )
    case = replace(
        base,
        gate1_specialization_profiles=profiles,
        gate1_primary_profile_id="contract",
    )

    family = _gate1_specialization_family(case)

    assert family is not None
    assert family.primary_profile_id == "contract"
    assert [profile.profile_id for profile in family.profiles] == [
        "one",
        "contract",
    ]
    assert all(
        profile.case.schedule == base.schedule_values
        for profile in family.profiles
    )
    assert family.profiles[0].case.arguments["n_elements"].value == 1
    assert family.profiles[1].case.arguments["n_elements"].value == 1003
    policy = _gate1_specialization_policy_json(case)
    assert policy["enabled"]
    assert policy["family"]["primary_profile_id"] == "contract"


def test_live_report_selects_final_round_family_evidence() -> None:
    base = build_case(_args("add"))
    profiles = (
        Gate2ShapeProfile(
            "one",
            {"a_ptr": (1,), "b_ptr": (1,), "out_ptr": (1,)},
            {"n_elements": 1},
            (1,),
        ),
        Gate2ShapeProfile(
            "contract",
            {
                "a_ptr": (1003,),
                "b_ptr": (1003,),
                "out_ptr": (1003,),
            },
            {"n_elements": 1003},
            (4,),
        ),
    )
    case = replace(
        base,
        gate1_specialization_profiles=profiles,
        gate1_primary_profile_id="contract",
    )
    family = _gate1_specialization_family(case)
    assert family is not None
    failed = {
        "policy": "fixed_c_family_full_sweep_relift_v1",
        "status": "no_closure",
        "all_passed": False,
        "records": [],
    }
    closed = {
        "policy": "fixed_c_family_full_sweep_relift_v1",
        "status": "replay_closed",
        "all_passed": True,
        "records": [],
    }
    gate1 = SimpleNamespace(
        candidates=(
            SimpleNamespace(
                relift_round=0,
                generation_round=0,
                artifact_digest="artifact-v0",
                specialization_evidence=failed,
            ),
            SimpleNamespace(
                relift_round=1,
                generation_round=0,
                artifact_digest="artifact-v1",
                specialization_evidence=closed,
            ),
        )
    )

    selected = _select_gate1_specialization_evidence(
        gate1,
        final_artifact_digest="artifact-v1",
    )
    assert selected == closed
    rendered = _gate1_family_evidence_json(selected, family)
    assert rendered["all_passed"] is True
    assert rendered["verdict_role"] == (
        "normal_family_strict_all_pass_gate1_condition"
    )
    assert rendered["anti_leakage_scope"] == "full_fixed_c_family"
    assert rendered["paired_intervention_policy"] == (
        "fixed_c_family_exact_delta_all_profiles_v1"
    )
    assert rendered["masked_j_family_policy"] == (
        "actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1"
    )
    assert rendered["family"]["primary_profile_id"] == "contract"


def test_live_case_rejects_an_undeclared_gate1_primary_profile() -> None:
    with pytest.raises(ValueError, match="primary profile"):
        replace(
            build_case(_args("add")),
            gate1_specialization_profiles=(Gate2ShapeProfile("one"),),
            gate1_primary_profile_id="missing",
        )


def test_live_case_wires_real_relifter_policy_into_gate1(monkeypatch) -> None:
    case = build_case(_args("add"))
    provider = ProviderConfig(
        "test",
        "https://example.invalid",
        "secret",
        "base-model",
    )
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name=case.function_name),
        to_json=lambda: {
            "kernel_id": case.function_name,
            "source": case.source_text,
        },
    )
    extraction = SimpleNamespace(
        artifact=artifact,
        provenance=SimpleNamespace(
            model="gpt-5.6-sol",
            prompt_sha256="lift-prompt",
            attempts=1,
            repairs=(),
            response_id="lift-response",
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "get_provider", lambda: provider)

    def fake_extract(*args, **kwargs):
        captured["lift_provider"] = kwargs["provider"]
        captured["lift_effort"] = kwargs["effort"]
        return extraction

    monkeypatch.setattr(live, "extract_scrc_artifact", fake_extract)
    monkeypatch.setattr(
        live,
        "ResponsesReplayWriter",
        lambda **kwargs: SimpleNamespace(),
    )

    class FakeRelifter:
        def __init__(self, **kwargs):
            captured["relifter_kwargs"] = kwargs
            self.records = ()

    class FakeRunner:
        def __init__(self, *, config):
            captured["gate1_config"] = config

        def run(self, *args, relifter=None, **kwargs):
            captured["gate1_relifter"] = relifter
            return SimpleNamespace(
                status=Gate1Status.NO_CLOSURE,
                replay_closed=False,
                candidates=(),
                masked_generation=None,
                    to_json=lambda: {
                        "status": Gate1Status.NO_CLOSURE.value,
                        "termination_reason": "relift_budget_exhausted",
                        "termination": {
                            "reason": "relift_budget_exhausted",
                            "artifact_round": 3,
                            "normal_candidate_count": 0,
                            "remaining_artifact_round_budget": 0,
                            "hard_difference_categories": [],
                        },
                        "artifact": artifact.to_json(),
                    "candidates": [],
                    "masked_generation": None,
                    "diagnostics": [],
                },
            )

    monkeypatch.setattr(live, "ResponsesSCRCRelifter", FakeRelifter)
    monkeypatch.setattr(live, "Gate1Runner", FakeRunner)
    monkeypatch.setattr(
        live,
        "canonical_artifact_digest",
        lambda artifact, registry: "final-artifact",
    )
    args = Namespace(
        model="gpt-5.6-sol",
        effort="xhigh",
        seed=20260715,
        lift_repairs=2,
        relift_rounds=4,
        gate2_matrix=False,
        gate2_seeds=None,
        diagnose_rejected=False,
    )

    code, report = live.run_case(case, args)

    configured_provider = captured["lift_provider"]
    assert isinstance(configured_provider, ProviderConfig)
    assert configured_provider.model == "gpt-5.6-sol"
    assert configured_provider.reasoning_effort == "xhigh"
    assert captured["lift_effort"] == "xhigh"
    relifter_kwargs = captured["relifter_kwargs"]
    assert isinstance(relifter_kwargs, dict)
    assert relifter_kwargs["provider"] is configured_provider
    assert relifter_kwargs["effort"] == "xhigh"
    assert captured["gate1_relifter"] is not None
    assert captured["gate1_config"].max_relifts == 4
    assert captured["gate1_config"].run_semantic_interventions
    assert captured["gate1_config"].max_semantic_interventions == 3
    assert report["relift_policy"]["protocol"] == (
        "gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11"
    )
    assert report["relift_policy"]["maximum_semantic_relifts"] == 3
    assert report["relift_policy"]["normal_recovery"] == (
        "relift_only_restart_full_pipeline"
    )
    assert report["relift_policy"]["replay_generation_policy"] == (
        "fresh_only_no_prior_replay_source"
    )
    assert report["relift_policy"]["stagnation"] == {
        "policy": "exact_artifact_and_hard_diff_v1",
        "required_consecutive_observations": 2,
        "basis": "canonical_artifact_and_exact_hard_diff",
        "ttir_fingerprint_schema": "triton-ttir-3.6-gate1-v3",
    }
    assert report["relift_policy"]["semantic_guidance"] == {
        "policy": "deterministic_gate1_hard_diff_guidance_v2",
        "structural_policy": (
            "deterministic_scoped_fingerprint_structure_guidance_v2"
        ),
        "source": (
            "first_completed_gate1_or_fixed_c_family_hard_comparison"
        ),
        "family_policy": (
            "deterministic_fixed_c_family_hard_diff_guidance_v2"
        ),
        "transport_failure_relift": "disabled_without_hard_diff",
        "replay_source_input": "forbidden",
    }
    assert report["relift_policy"]["termination"] == {
        "policy": "structured_no_closure_termination_v1",
        "no_closure_requires_structured_reason": True,
        "gate2_verdict_independence": True,
    }
    assert report["gate1_specialization_policy"] == {
        "enabled": False,
        "role": "fixed_c_family_full_sweep_relift_v1",
        "verdict_role": "normal_family_strict_all_pass_gate1_condition",
        "diff_policy": "exact_ordered_fixed_c_family_hard_diff_v1",
        "replay_policy": "one_fresh_replay_per_artifact_round",
        "relift_policy": (
            "complete_family_sweep_then_one_source_aware_relift"
        ),
        "stagnation_basis": (
            "canonical_artifact_and_exact_family_hard_diff"
        ),
        "anti_leakage_scope": "primary_profile",
        "paired_intervention_policy": (
            "fixed_c_family_exact_delta_all_profiles_v1"
        ),
        "masked_j_family_policy": (
            "actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1"
        ),
        "family": None,
    }
    assert report["gate2_lowering_policy"] == {
        "protocol": (
            "intent_linalg_cpu_symbolic_iota_integer_readback_"
            "boundary_broadcast_compressed_gather_v5"
        ),
        "unsupported_status": "lowering_unsupported",
        "legacy_interpreter_fallback": "disabled",
    }
    assert report["gate2_comparison_policy"] == {
        "protocol": "exact_integer_tolerant_float_v2",
        "integer_and_boolean": "exact",
        "floating": "per_output_tolerance",
    }
    assert report["gate1_anti_leakage_policy"] == {
        "fixed_c_family_closure_policy": (
            "normal_paired_and_masked_fixed_c_family_all_required_v1"
        ),
        "paired_semantic_interventions": True,
        "maximum_interventions": 3,
        "paired_replay_generation": "fresh_from_mutated_lift_artifact",
        "paired_intervention_family_policy": (
            "fixed_c_family_exact_delta_all_profiles_v1"
        ),
        "paired_intervention_scope": "primary_profile",
        "maximum_provider_attempts_per_generation": 3,
        "compose_equality": "gate1_verdict_condition",
        "masked_j_role": "actual_lift_pi_terminal_attack",
        "masked_j_policy": (
            "actual_lift_pi_masked_j_hard_nonclosure_v3"
        ),
        "masked_j_family_policy": (
            "actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1"
        ),
        "masked_j_scope": "primary_profile",
        "masked_j_required_outcome": "compile_or_hard_nonclosure",
        "masked_j_score_role": "not_used",
        "ttir_score_semantics": "active_evidence_weighted_v2",
        "replay_reference_naming": "opaque_replay_refs_v1",
        "pi_ablations": {
            "enabled": True,
            "default_policy": "enabled_for_formal_live_evidence",
            "maximum": 3,
            "requires_full_pi_replay_closed": True,
            "kinds": [
                "plan_step_dropout",
                "plan_mechanism_substitution",
            ],
            "verdict_role": "diagnostic",
            "pi_mutation_execution": "reject_before_writer",
            "mechanism_substitution": "c_strategy_then_recompose",
            "opaque_reference_discipline": "resolve_handles_never_emit_v1",
        },
    }
    assert report["pi_ablation_control"] == {
        "valid": False,
        "reason": "source_fingerprint_unavailable",
        "gate1_status": "no_closure",
    }
    assert report["pi_ablations"] == []
    assert report["relifts"] == []
    assert report["performance"]["schema"] == "scrc_case_performance_v1"
    assert report["performance"]["verdict_role"] == "observation_only"
    assert report["performance"]["timing_ms"]["initial_lift"] is not None
    assert report["performance"]["timing_ms"]["gate1_loop"] is not None
    assert report["performance"]["timing_ms"]["gate2"] is None
    assert report["performance"]["gate1_observation"]["status"] == (
        "no_closure"
    )
    assert report["performance"]["gate1_observation"][
        "outer_loop_termination"
    ] == "relift_budget_exhausted"
    assert report["performance"]["recorded_invocations"][
        "normal_replay_writer"
    ] == 0
    assert code == 3


def test_add_compatibility_case_keeps_existing_defaults() -> None:
    case = build_case(_args("add"))
    view = case.compiler_view().to_json()

    assert view["grid"] == [4]
    assert view["grid_contract"] == ["ceil_div(n_elements, BLOCK)"]
    assert view["schedule_values"]["BLOCK"] == 256
    assert case.source_path == ROOT / "test" / "scrc_probe_sources.py"
    assert case.function_name == "add_kernel"


def test_performance_counts_family_intervention_and_masked_compiles() -> None:
    report = {
        "gate1": {
            "status": "replay_closed",
            "termination_reason": "replay_closed",
            "candidates": [{"compile": {"ok": True}}],
            "interventions": [
                {
                    "generation": {"generation_id": "mutant"},
                    "profile_records": [
                        {
                            "baseline_compile": {"ok": True},
                            "baseline_compile_reused": True,
                            "compile": {"ok": True},
                        },
                        {
                            "baseline_compile": {"ok": True},
                            "baseline_compile_reused": False,
                            "compile": {"ok": True},
                        },
                    ],
                }
            ],
            "masked_outcome": "family_hard_mismatch",
            "masked_family_evidence": {
                "records": [
                    {
                        "source_compile": {"ok": True},
                        "source_compile_reused": True,
                        "compile": {"ok": True},
                    },
                    {
                        "source_compile": {"ok": True},
                        "source_compile_reused": False,
                        "compile": {"ok": True},
                    },
                ]
            },
        },
        "gate1_specialization_family": {
            "records": [
                {
                    "source_compile": {"ok": True},
                    "source_compile_reused": True,
                    "replay_compile": {"ok": True},
                    "replay_compile_reused": True,
                },
                {
                    "source_compile": {"ok": True},
                    "source_compile_reused": False,
                    "replay_compile": {"ok": True},
                    "replay_compile_reused": False,
                },
            ]
        },
        "relifts": [],
    }

    live._finalize_performance_report(
        report,
        pipeline_started=live.time.perf_counter(),
        stage_ms={},
    )

    counts = report["performance"]["recorded_invocations"]
    assert counts["semantic_intervention_writer"] == 1
    assert counts["specialization_source_compile"] == 1
    assert counts["specialization_replay_compile"] == 1
    assert counts["semantic_intervention_baseline_compile"] == 1
    assert counts["semantic_intervention_compile"] == 2
    assert counts["masked_j_writer"] == 1
    assert counts["masked_j_source_compile"] == 1
    assert counts["masked_j_compile"] == 2


def test_live_parser_records_independent_gate2_for_rejected_replays_by_default() -> None:
    assert _parser("add").parse_args([]).diagnose_rejected
    assert not _parser("add").parse_args(
        ["--no-diagnose-rejected"]
    ).diagnose_rejected


def test_live_parser_enables_final_pi_audit_by_default() -> None:
    assert _parser("add").parse_args([]).pi_ablations
    assert not _parser("add").parse_args(
        ["--no-pi-ablations"]
    ).pi_ablations


def test_pi_ablation_report_keeps_source_blind_generated_code() -> None:
    generation = ReplayGeneration(
        code="import triton\n",
        prompt_sha256="prompt",
        requested_model="model",
        generation_id="generation",
        code_sha256="code",
    )
    record = SimpleNamespace(
        generation=generation,
        to_json=lambda: {"generation": {"code_sha256": "code"}},
    )

    payload = _pi_ablation_evidence_json((record,))[0]
    assert payload["generation"]["code"] == "import triton\n"
    assert payload["generation"]["code_bytes"] == 14


def test_pi_diagnostics_require_a_closed_canonical_pi_control() -> None:
    rejected = SimpleNamespace(
        source_fingerprint=object(),
        replay_closed=False,
        status=Gate1Status.NO_CLOSURE,
    )
    accepted = SimpleNamespace(
        source_fingerprint=object(),
        replay_closed=True,
        status=Gate1Status.REPLAY_CLOSED,
    )

    assert _pi_ablation_control_evidence(rejected) == {
        "valid": False,
        "reason": "canonical_pi_control_not_replay_closed",
        "gate1_status": "no_closure",
    }
    assert _pi_ablation_control_evidence(accepted)["valid"]


def test_gate2_replay_selection_prefers_hard_close_then_final_round_best() -> None:
    earlier_best = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=0,
        generation_round=0,
        artifact_digest="final-artifact",
        rank_tuple=(1,),
    )
    final_weaker = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=1,
        generation_round=0,
        artifact_digest="final-artifact",
        rank_tuple=(2,),
    )
    final_best = SimpleNamespace(
        generation=object(),
        comparison=None,
        compiled=True,
        relift_round=1,
        generation_round=1,
        artifact_digest="final-artifact",
        rank_tuple=(3,),
    )
    gate1 = SimpleNamespace(
        candidates=(earlier_best, final_weaker, final_best)
    )
    assert (
        _select_gate2_replay_observation(
            gate1,
            final_artifact_digest="final-artifact",
        )
        is final_best
    )

    hard_closed = SimpleNamespace(
        generation=object(),
        comparison=SimpleNamespace(hard_pass=True),
        compiled=True,
        relift_round=0,
        generation_round=0,
        artifact_digest="final-artifact",
        rank_tuple=(0,),
    )
    gate1 = SimpleNamespace(candidates=(final_best, hard_closed))
    assert (
        _select_gate2_replay_observation(
            gate1,
            final_artifact_digest="final-artifact",
        )
        is hard_closed
    )


def test_gate2_replay_selection_never_falls_back_across_artifacts() -> None:
    round_zero = SimpleNamespace(
        generation=object(),
        comparison=SimpleNamespace(hard_pass=True),
        compiled=True,
        relift_round=0,
        generation_round=0,
        artifact_digest="artifact-before-relift",
        rank_tuple=(9,),
    )
    gate1 = SimpleNamespace(candidates=(round_zero,))

    assert (
        _select_gate2_replay_observation(
            gate1,
            final_artifact_digest="artifact-after-relift",
        )
        is None
    )
    assert (
        _select_gate1_specialization_evidence(
            SimpleNamespace(
                candidates=(
                    SimpleNamespace(
                        relift_round=0,
                        generation_round=0,
                        artifact_digest="artifact-before-relift",
                        specialization_evidence={"all_passed": True},
                    ),
                )
            ),
            final_artifact_digest="artifact-after-relift",
        )
        is None
    )


def test_rejected_gate1_records_independent_gate2_without_overwriting_status(
    monkeypatch,
) -> None:
    case = build_case(_args("add"))
    provider = ProviderConfig(
        "test", "https://example.invalid", "secret", "base-model"
    )
    artifact = SimpleNamespace(
        interface=SimpleNamespace(function_name=case.function_name),
        to_json=lambda: {
            "kernel_id": case.function_name,
            "source": case.source_text,
        },
    )
    extraction = SimpleNamespace(
        artifact=artifact,
        provenance=SimpleNamespace(
            model="gpt-5.6-sol",
            prompt_sha256="lift-prompt",
            attempts=1,
            repairs=(),
            response_id="lift-response",
        ),
    )
    generation = _generation(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef add_kernel(a_ptr, b_ptr, out_ptr, n_elements, "
        "BLOCK: tl.constexpr):\n    return\n",
        masked=False,
    )
    candidate = SimpleNamespace(
        generation=generation,
        comparison=SimpleNamespace(hard_pass=True),
        compiled=True,
        relift_round=0,
        generation_round=0,
        artifact_digest="final-artifact",
        rank_tuple=(1,),
    )
    gate1 = SimpleNamespace(
        status=Gate1Status.NO_CLOSURE,
        replay_closed=False,
        artifact=artifact,
        candidates=(candidate,),
        masked_generation=None,
        to_json=lambda: {
            "status": Gate1Status.NO_CLOSURE.value,
            "termination_reason": "deterministic_stagnation",
            "termination": {
                "reason": "deterministic_stagnation",
                "artifact_round": 1,
                "normal_candidate_count": 2,
                "remaining_artifact_round_budget": 1,
                "hard_difference_categories": ["dataflow"],
            },
            "artifact": artifact.to_json(),
            "candidates": [{"generation": {"code_sha256": "digest"}}],
            "masked_generation": None,
            "diagnostics": [],
        },
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "get_provider", lambda: provider)
    monkeypatch.setattr(live, "extract_scrc_artifact", lambda *a, **k: extraction)
    monkeypatch.setattr(
        live, "ResponsesReplayWriter", lambda **kwargs: SimpleNamespace()
    )

    class FakeRelifter:
        def __init__(self, **kwargs):
            self.records = ()

    class FakeGate1Runner:
        def __init__(self, *, config):
            pass

        def run(self, *args, **kwargs):
            return gate1

    class FakeGate2Runner:
        def run(self, source, replay, gate2_case):
            captured["gate2_case"] = gate2_case
            captured["replay"] = replay
            return SimpleNamespace(status=Gate2Status.PASS)

    monkeypatch.setattr(live, "ResponsesSCRCRelifter", FakeRelifter)
    monkeypatch.setattr(live, "Gate1Runner", FakeGate1Runner)
    monkeypatch.setattr(live, "Gate2Runner", FakeGate2Runner)
    monkeypatch.setattr(
        live,
        "canonical_artifact_digest",
        lambda artifact, registry: "final-artifact",
    )
    monkeypatch.setattr(live, "_gate2_case", lambda *args: "gate2-case")
    monkeypatch.setattr(
        live,
        "_gate2_evidence_json",
        lambda result: {"status": result.status.value},
    )
    args = Namespace(
        model="gpt-5.6-sol",
        effort="xhigh",
        seed=20260715,
        lift_repairs=2,
        relift_rounds=2,
        gate2_matrix=False,
        gate2_seeds=None,
        diagnose_rejected=True,
    )

    code, report = live.run_case(case, args)

    assert code == 3
    assert report["gate1"]["status"] == Gate1Status.NO_CLOSURE.value
    assert report["performance"]["gate1_observation"][
        "outer_loop_termination"
    ] == "deterministic_stagnation"
    assert report["gate2_diagnostic"] == {
        "kind": "offline_gate2_for_gate1_rejected_sample",
        "independent": True,
        "gate1_status": Gate1Status.NO_CLOSURE.value,
        "status": Gate2Status.PASS.value,
    }
    assert captured["gate2_case"] == "gate2-case"
    assert report["gate2_replay_alignment"]["status"] == "aligned"

    # A later Relift may replace the final artifact and fail before producing
    # any candidate.  The older compiled replay must then remain unavailable
    # to Gate2 instead of being selected as a fallback.
    candidate.artifact_digest = "artifact-before-relift"
    captured.clear()
    code, report = live.run_case(case, args)

    assert code == 3
    assert "gate2_case" not in captured
    assert "gate2_diagnostic" not in report
    assert report["gate2_replay_alignment"] == {
        "policy": "final_gate1_artifact_digest_exact_match_v1",
        "status": "coverage_gap",
        "final_artifact_digest": "final-artifact",
        "candidate_artifact_digest": None,
        "candidate_relift_round": None,
        "reason": (
            "no_executable_candidate_bound_to_final_gate1_artifact"
        ),
        "diagnostic": None,
    }


def _generation(code: str, *, masked: bool) -> ReplayGeneration:
    return ReplayGeneration(
        code=code,
        prompt_sha256="prompt",
        requested_model="model",
        generation_id=f"generation:{masked}",
        code_sha256="digest",
        masked=masked,
    )


def test_failure_evidence_keeps_candidates_but_redacts_source() -> None:
    normal = _generation("NORMAL_REPLAY\n", masked=False)
    masked = _generation("OPAQUE_MASKED_REPLAY\n", masked=True)
    gate1 = SimpleNamespace(
        to_json=lambda: {
            "artifact": {"source": "SECRET_SOURCE", "kernel_id": "k"},
            "candidates": [{"generation": {"code_sha256": "digest"}}],
            "masked_generation": {"code_sha256": "digest"},
        },
        candidates=(SimpleNamespace(generation=normal),),
        masked_generation=masked,
    )

    evidence = _gate1_evidence_json(gate1)
    assert "source" not in evidence["artifact"]
    assert evidence["artifact"]["source_bytes"] == len("SECRET_SOURCE")
    assert evidence["candidates"][0]["generation"]["code"] == normal.code
    assert evidence["masked_generation"]["code"] == masked.code


def test_lift_failure_evidence_hashes_raw_model_response() -> None:
    evidence = _lift_attempt_evidence(
        SimpleNamespace(
            attempt=0,
            model="model",
            response_id="response",
            diagnostic="invalid artifact",
            response_text="SECRET_MODEL_RESPONSE",
        )
    )

    assert "response_text" not in evidence
    assert evidence["response_bytes"] == len("SECRET_MODEL_RESPONSE")
    assert len(evidence["response_sha256"]) == 64


def test_relift_report_links_the_normal_no_closure_request() -> None:
    relifter = SimpleNamespace(
        records=(
            SimpleNamespace(
                relift_round=1,
                extraction=None,
                request={
                    "trigger": "normal_no_closure",
                    "evaluation_round": 0,
                    "candidate_count": 3,
                    "compiled_candidate_count": 3,
                    "best_generation_round": 0,
                    "hard_difference_categories": ["dataflow"],
                },
                diagnostic="relift failed",
                rejected_attempts=(),
            ),
        )
    )

    evidence = _relift_evidence_json(relifter)

    assert evidence[0]["request"]["trigger"] == "normal_no_closure"
    assert evidence[0]["request"]["hard_difference_categories"] == [
        "dataflow"
    ]
    assert evidence[0]["diagnostic"] == "relift failed"


def test_report_write_is_atomic_private_and_creates_explicit_parent(tmp_path) -> None:
    target = tmp_path / "nested" / "report.json"
    _write_report(str(target), {"status": "ok"})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(target.parent.iterdir()) == [target]
