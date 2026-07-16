from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate2 import (
    extract_gate1_gate2_observation,
    summarize_gate1_gate2,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

from scrc_heldout_batch import (  # noqa: E402
    CANDIDATES,
    READY_CASES,
    _batch_identity,
    _case_args,
    _load_cached_lift,
    _parser,
    _relift_response_ids,
    _response_ids,
    _resume_row,
    _result_status,
    build_heldout_case,
)
from scrc_live_e2e import _compile_case  # noqa: E402


def test_first_batch_manifest_has_reviewed_20_to_40_real_candidates() -> None:
    assert 20 <= len(CANDIDATES) <= 40
    assert len({item.name for item in CANDIDATES}) == len(CANDIDATES)
    assert READY_CASES == (
        "fg_dot",
        "fg_addr",
        "fg_ones",
        "fg_arange",
        "fg_embedding",
        "fg_sum",
        "fg_mean",
        "fg_max",
        "fg_min",
        "fg_prod",
        "fg_vector_l2",
        "fg_vector_l1",
    )
    assert all(item.module.startswith("flag_gems.") for item in CANDIDATES)
    assert all(item.function.isidentifier() for item in CANDIDATES)


def test_real_flag_gems_dot_case_preserves_rank_zero_output_contract() -> None:
    case = build_heldout_case("fg_dot")
    view = case.compiler_view().to_json()
    compile_case = _compile_case(case)

    assert case.source_path.name == "dot.py"
    assert case.function_name == "dot_kernel"
    assert "def dot_kernel" in case.source_text
    assert view["arguments"][2] == {
        "name": "out_ptr",
        "kind": "tensor",
        "dtype": "f32",
        "shape": [],
    }
    assert view["grid"] == [1]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert compile_case.arguments["out_ptr"].shape == ()


def test_real_flag_gems_arange_case_records_iota_runtime_contract() -> None:
    case = build_heldout_case("fg_arange")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "arange.py"
    assert case.function_name == "arange_func"
    assert "arange_val = cols * step + step_offset + start" in case.source_text
    assert by_name["y_ptr"]["shape"] == [1003]
    assert by_name["start"]["value"] == -3.25
    assert by_name["end"]["value"] == 122.125
    assert by_name["step"]["value"] == 0.125
    assert by_name["size"]["value"] == 1003
    assert view["grid"] == [8]
    assert view["grid_contract"] == ["ceil_div(size, BLOCK_SIZE)"]
    assert view["constexpr_parameters"] == ["BLOCK_SIZE"]
    assert view["schedule_values"]["BLOCK_SIZE"] == 128
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "forward_fractional",
        "descending_fractional",
        "forward_unit",
    ]
    assert [profile.scalar_values["step"] for profile in case.gate2_shape_profiles] == [
        0.125,
        -0.5,
        1.0,
    ]


def test_real_flag_gems_addr_case_records_runtime_scalars_and_strides() -> None:
    case = build_heldout_case("fg_addr")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "addr.py"
    assert view["grid"] == [1, 1]
    assert view["grid_contract"] == [
        "ceil_div(M, BLOCK_SIZE_M)",
        "ceil_div(N, BLOCK_SIZE_N)",
    ]
    by_name = {item["name"]: item for item in view["arguments"]}
    assert by_name["beta"]["value"] == 0.75
    assert by_name["alpha"]["value"] == -1.25
    assert by_name["stride_input_m"]["value"] == 13
    assert by_name["stride_output_n"]["value"] == 1


def test_real_flag_gems_ones_case_has_multi_program_tail_contract() -> None:
    case = build_heldout_case("fg_ones")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "ones.py"
    assert case.function_name == "ones_kernel"
    assert "def ones_kernel" in case.source_text
    assert view["arguments"] == [
        {
            "name": "output_ptr",
            "kind": "tensor",
            "dtype": "f32",
            "shape": [2051],
        },
        {
            "name": "n_elements",
            "kind": "scalar",
            "dtype": "i32",
            "value": 2051,
        },
    ]
    assert view["grid"] == [3]
    assert view["grid_contract"] == [
        "ceil_div(n_elements, BLOCK_SIZE)"
    ]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one",
        "block_minus_one",
        "block",
        "block_plus_one",
        "non_divisible",
        "far_multitile",
    ]
    assert [
        profile.scalar_values["n_elements"]
        for profile in case.gate2_shape_profiles
    ] == [1, 1023, 1024, 1025, 2051, 8209]


def test_real_flag_gems_sum_case_uses_final_rank_zero_stage() -> None:
    case = build_heldout_case("fg_sum")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "sum.py"
    assert case.function_name == "sum_kernel_2"
    assert "def sum_kernel_2" in case.source_text
    assert view["arguments"][0]["shape"] == [1003]
    assert view["arguments"][1] == {
        "name": "out",
        "kind": "tensor",
        "dtype": "f32",
        "shape": [],
    }
    assert view["arguments"][2]["value"] == 1003
    assert view["grid"] == [1, 1, 1]
    assert view["schedule_values"]["BLOCK_MID"] == 1024


def test_real_flag_gems_mean_case_keeps_total_distinct_from_mid_size() -> None:
    case = build_heldout_case("fg_mean")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "mean.py"
    assert case.function_name == "mean_kernel_2"
    assert "mean_val = sum_val / M" in case.source_text
    assert by_name["mid"]["shape"] == [37]
    assert by_name["out"]["shape"] == []
    assert by_name["M"]["value"] == 1003
    assert by_name["MID_SIZE"]["value"] == 37
    assert view["schedule_values"]["BLOCK_MID"] == 64


@pytest.mark.parametrize(
    ("name", "source_file", "function", "mid_size", "block", "grid"),
    [
        ("fg_max", "max.py", "max_kernel_2", 63, 64, [1, 1, 1]),
        ("fg_min", "min.py", "min_kernel_2", 64, 64, [1, 1, 1]),
        (
            "fg_prod",
            "prod.py",
            "prod_kernel_result",
            17,
            32,
            [1, 1, 1],
        ),
        (
            "fg_vector_l2",
            "vector_norm.py",
            "l2_norm_kernel_2",
            37,
            64,
            [1],
        ),
    ],
)
def test_next_reduction_cases_use_whole_tensor_final_stages(
    name: str,
    source_file: str,
    function: str,
    mid_size: int,
    block: int,
    grid: list[int],
) -> None:
    case = build_heldout_case(name)
    view = case.compiler_view().to_json()

    assert case.source_path.name == source_file
    assert case.function_name == function
    assert f"def {function}" in case.source_text
    assert view["arguments"][0]["shape"] == [mid_size]
    assert view["arguments"][1]["shape"] == []
    assert view["arguments"][2]["value"] == mid_size
    assert view["grid"] == grid
    assert view["schedule_values"]["BLOCK_MID"] == block


def test_prod_and_l2_gate2_inputs_respect_intermediate_value_domains() -> None:
    prod = build_heldout_case("fg_prod")
    l2 = build_heldout_case("fg_vector_l2")

    assert prod.input_distributions == {
        "mid": "near_one"
    }
    assert l2.input_distributions == {
        "Mid": "positive"
    }


def test_real_flag_gems_vector_l1_case_keeps_runtime_order_unspecialized() -> None:
    case = build_heldout_case("fg_vector_l1")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "vector_norm.py"
    assert case.function_name == "l1_norm_kernel_1"
    assert "@triton.jit(do_not_specialize=[\"ord\"])" in case.source_text
    assert "tl.sum(pow(tl.abs(x), ord))" in case.source_text
    assert by_name["X"]["shape"] == [1003]
    assert by_name["Mid"]["shape"] == [1]
    assert by_name["ord"] == {
        "name": "ord",
        "kind": "scalar",
        "dtype": "f32",
        "value": 1.5,
    }
    assert by_name["M"]["value"] == 1003
    assert view["grid"] == [1]
    assert view["grid_contract"] == ["ceil_div(M, BLOCK_SIZE)"]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one_ord1",
        "block_minus_one_ord15",
        "block_ord2",
    ]
    assert [
        (profile.tensor_shapes["X"][0], profile.scalar_values["ord"])
        for profile in case.gate2_shape_profiles
    ] == [(1, 1.0), (1023, 1.5), (1024, 2.0)]
    assert [profile.profile_id for profile in case.gate2_value_profiles] == [
        "normal",
        "uniform",
    ]


def test_real_flag_gems_embedding_case_records_take_gather_contract() -> None:
    case = build_heldout_case("fg_embedding")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "embedding.py"
    assert case.function_name == "embedding_kernel"
    assert "weight_ptr += row_idx * N" in case.source_text
    assert by_name["out_ptr"]["shape"] == [17, 13]
    assert by_name["in_ptr"] == {
        "name": "in_ptr",
        "kind": "tensor",
        "dtype": "i64",
        "shape": [17],
    }
    assert by_name["weight_ptr"]["shape"] == [97, 13]
    assert view["grid"] == [17]
    assert view["grid_contract"] == ["K"]
    assert view["constexpr_parameters"] == ["N", "BLOCK_SIZE"]
    assert view["schedule_values"]["N"] == 13
    assert view["schedule_values"]["BLOCK_SIZE"] == 16
    assert case.input_distributions["in_ptr"] == "positive"
    assert case.input_scales["in_ptr"] == 96.0
    assert case.gate2_shape_profiles == ()
    assert [profile.profile_id for profile in case.gate2_value_profiles] == [
        "random_index_normal_weight",
        "boundary_index_uniform_weight",
    ]


def test_min_gate2_matrix_declares_shape_value_cross_product() -> None:
    case = build_heldout_case("fg_min")

    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one",
        "block_minus_one",
        "block",
    ]
    assert [profile.profile_id for profile in case.gate2_value_profiles] == [
        "normal",
        "uniform",
        "large_dynamic",
        "ieee_special",
    ]


def test_batch_status_uses_existing_gate_verdicts() -> None:
    assert _result_status(
        {"preflight": {"ok": False}}
    ) == "source_compile_failed"
    assert _result_status({"lift": {"ok": False}}) == "lift_failed"
    assert _result_status(
        {"lift": {"ok": True}, "gate1": {"status": "no_closure"}}
    ) == "no_closure"
    assert _result_status(
        {
            "lift": {"ok": True},
            "gate1": {"status": "leakage_rejected"},
            "gate2_diagnostic": {"status": "gate2_pass"},
        }
    ) == "leakage_rejected"
    assert _result_status(
        {
            "lift": {"ok": True},
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "lowering_unsupported"},
        }
    ) == "lowering_unsupported"
    assert _result_status(
        {
            "lift": {"ok": True},
            "gate1": {"status": "replay_closed"},
            "gate2_suite": {"passed": True},
        }
    ) == "gate2_pass"
    assert _result_status(
        {
            "lift": {"ok": True},
            "gate1": {"status": "replay_closed"},
            "gate2_suite": {"passed": False},
        }
    ) == "gate2_suite_failed"


def test_batch_extracts_gate1_rejection_and_independent_gate2_pass() -> None:
    observation = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "leakage_rejected"},
            "gate2_diagnostic": {
                "kind": "offline_gate2_for_gate1_rejected_sample",
                "independent": True,
                "status": "gate2_pass",
            },
        }
    )

    assert observation == {
        "gate1_status": "leakage_rejected",
        "gate1_closed": False,
        "gate2_evidence_kind": "diagnostic",
        "gate2_attempted": True,
        "gate2_three_way_completed": True,
        "gate2_status": "gate2_pass",
        "gate2_pass": True,
        "gate2_case_statuses": ["gate2_pass"],
    }


def test_batch_does_not_count_unsupported_or_setup_failure_as_semantic_nonpass() -> None:
    unsupported = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "lowering_unsupported"},
        }
    )
    setup_failed = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "no_closure"},
            "gate2_diagnostic_suite": {"status": "setup_failed"},
        }
    )

    assert unsupported["gate2_attempted"]
    assert not unsupported["gate2_three_way_completed"]
    assert unsupported["gate2_pass"] is None
    assert setup_failed["gate2_attempted"]
    assert not setup_failed["gate2_three_way_completed"]
    assert setup_failed["gate2_status"] == "setup_failed"
    assert setup_failed["gate2_pass"] is None


def test_batch_conditional_metrics_report_probability_and_evaluation_coverage() -> None:
    reports = [
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "gate2_pass"},
        },
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "replay_diverged"},
        },
        {
            "gate1": {"status": "leakage_rejected"},
            "gate2_diagnostic_suite": {
                "passed": True,
                "cases": [
                    {"status": "gate2_pass"},
                    {"status": "gate2_pass"},
                ],
            },
        },
        {"gate1": {"status": "no_closure"}},
    ]
    rows = [
        {"gate1_gate2": extract_gate1_gate2_observation(report)}
        for report in reports
    ]

    metrics = summarize_gate1_gate2(rows)

    assert metrics["replay_closed"] == {
        "samples": 2,
        "gate2_attempted": 2,
        "gate2_three_way_completed": 2,
        "gate2_pass": 1,
        "gate2_nonpass": 1,
        "attempt_coverage": 1.0,
        "completion_coverage": 1.0,
        "pass_probability_given_completed": 0.5,
    }
    assert metrics["not_replay_closed"] == {
        "samples": 2,
        "gate2_attempted": 1,
        "gate2_three_way_completed": 1,
        "gate2_pass": 1,
        "gate2_nonpass": 0,
        "attempt_coverage": 0.5,
        "completion_coverage": 0.5,
        "pass_probability_given_completed": 1.0,
    }
    assert metrics["by_gate1_status"]["no_closure"][
        "pass_probability_given_completed"
    ] is None


def test_batch_response_ids_tolerate_failed_generation_and_absent_masked() -> None:
    normal, masked = _response_ids(
        {
            "gate1": {
                "candidates": [
                    {"generation": None},
                    {"generation": {"response_id": "resp-normal"}},
                ],
                "masked_generation": None,
            }
        }
    )
    assert normal == ["resp-normal"]
    assert masked is None


def _batch_args(**overrides) -> Namespace:
    values = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "seed": 20260715,
        "lift_repairs": 2,
        "relift_rounds": 3,
        "semantic_interventions": 3,
        "pi_ablations": True,
        "pi_ablation_limit": 3,
        "pi_ablation_kinds": None,
        "diagnose_rejected": False,
        "gate2_matrix": False,
        "gate2_seeds": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resume_requires_the_exact_current_batch_identity() -> None:
    identity = _batch_identity(build_heldout_case("fg_dot"), _batch_args())
    row = {"case": "fg_dot", "status": "leakage_rejected"}
    saved = {
        "batch": {"terminal": True, "identity": identity, "row": row}
    }

    assert _resume_row(saved, identity) == row
    stale = dict(identity)
    stale["seed"] = identity["seed"] + 1
    assert _resume_row(saved, stale) is None
    assert (
        _resume_row({"batch": {"terminal": True, "row": row}}, identity)
        is None
    )
    changed_input = _batch_identity(
        replace(build_heldout_case("fg_dot"), input_scale=0.5),
        _batch_args(),
    )
    assert changed_input != identity
    assert _resume_row(saved, changed_input) is None
    matrix_identity = _batch_identity(
        build_heldout_case("fg_dot"),
        _batch_args(gate2_matrix=True),
    )
    assert matrix_identity["gate2_matrix_policy"]["enabled"]
    assert matrix_identity["gate2_matrix_policy"]["seeds"] == [
        20260715,
        20260716,
        20260717,
    ]
    assert _resume_row(saved, matrix_identity) is None
    relift_identity = _batch_identity(
        build_heldout_case("fg_dot"),
        _batch_args(relift_rounds=1),
    )
    assert identity["schema"] == 17
    assert identity["relift_rounds"] == 3
    assert identity["semantic_interventions"] == 3
    assert identity["normal_recovery"] == "relift_only_restart_full_pipeline"
    assert identity["replay_code_repair"] == "disabled"
    assert identity["pi_ablations"]
    assert identity["pi_ablation_limit"] == 3
    assert identity["pi_ablation_kinds"] == []
    assert identity["pi_ablation_replay_repairs"] == 0
    assert identity["pi_ablation_requires_closed_control"] is True
    assert identity["pi_ablation_opaque_reference_discipline"] == (
        "resolve_handles_never_emit_v1"
    )
    assert identity["pi_ablation_replay_code_repair"] == "disabled"
    assert identity["maximum_provider_attempts_per_generation"] == 3
    assert identity["ttir_score_semantics"] == "active_evidence_weighted_v2"
    assert identity["masked_j_fallback_policy"] == (
        "exact_semantic_hard_degradation_v1"
    )
    assert identity["masked_j_score_role"] == "diagnostic_only"
    assert identity["replay_reference_naming"] == "opaque_replay_refs_v1"
    assert identity["relift_protocol"] == (
        "relift_only_pi_audit_on_strict_opaque_refs_causal_v7"
    )
    assert relift_identity["relift_rounds"] == 1
    assert _resume_row(saved, relift_identity) is None


def test_batch_parser_and_case_args_preserve_relift_policy() -> None:
    parsed = _parser().parse_args([])
    assert parsed.model == "gpt-5.6-sol"
    assert parsed.effort == "xhigh"
    assert parsed.relift_rounds == 3
    assert parsed.pi_ablations
    assert not _parser().parse_args(["--no-pi-ablations"]).pi_ablations
    assert _case_args(parsed).relift_rounds == 3
    with pytest.raises(SystemExit):
        _parser().parse_args(["--relift-rounds", "0"])


def test_batch_collects_relift_response_ids() -> None:
    assert _relift_response_ids(
        {
            "relifts": [
                {"ok": True, "response_id": "relift-1"},
                {"ok": False},
                {"ok": True, "response_id": "relift-2"},
            ]
        }
    ) == ["relift-1", "relift-2"]


def test_lift_cache_rejects_a_different_rule_registry(tmp_path: Path) -> None:
    case = build_heldout_case("fg_dot")
    path = tmp_path / "fg_dot.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.6-sol",
                "effort": "xhigh",
                "source": {
                    "sha256": hashlib.sha256(
                        case.source_text.encode("utf-8")
                    ).hexdigest()
                },
                "compiler_view": case.compiler_view().to_json(),
                "lift": {
                    "ok": True,
                    "rule_registry_digest": "stale-registry",
                    "artifact": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rule registry differs"):
        _load_cached_lift(
            path,
            case,
            Namespace(model="gpt-5.6-sol", effort="xhigh"),
        )


def test_lift_cache_rejects_a_different_initial_prompt(tmp_path: Path) -> None:
    case = build_heldout_case("fg_dot")
    path = tmp_path / "fg_dot.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.6-sol",
                "effort": "xhigh",
                "source": {
                    "sha256": hashlib.sha256(
                        case.source_text.encode("utf-8")
                    ).hexdigest()
                },
                "compiler_view": case.compiler_view().to_json(),
                "lift": {
                    "ok": True,
                    "rule_registry_digest": (
                        default_triton_rule_registry().digest
                    ),
                    "prompt_sha256": "stale-prompt",
                    "artifact": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt differs"):
        _load_cached_lift(
            path,
            case,
            Namespace(model="gpt-5.6-sol", effort="xhigh"),
        )
