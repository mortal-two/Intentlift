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
    GATE2_EXECUTION_PROTOCOL,
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
    _summarize_performance,
    _timing_distribution,
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
        "fg_zeros",
        "fg_arange",
        "fg_eye",
        "fg_trace",
        "fg_embedding",
        "fg_index_select",
        "fg_diag_extract",
        "fg_amax",
        "fg_sum",
        "fg_mean",
        "fg_max",
        "fg_min",
        "fg_prod",
        "fg_vector_l2",
        "fg_vector_l1",
        "fg_log_softmax",
        "fg_skip_layer_norm",
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
    assert case.gate1_specialization_profiles == case.gate2_shape_profiles
    assert case.gate1_primary_profile_id == "non_divisible"


def test_real_flag_gems_zeros_case_adds_an_independent_fill_constant() -> None:
    case = build_heldout_case("fg_zeros")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "zeros.py"
    assert case.function_name == "zeros_kernel"
    assert "tl.store(output_ptr + offsets, 0.0, mask=mask)" in case.source_text
    assert view["arguments"][0]["shape"] == [2051]
    assert view["arguments"][1]["value"] == 2051
    assert view["grid"] == [3]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one",
        "block_minus_one",
        "block",
        "block_plus_one",
        "non_divisible",
        "far_multitile",
    ]


def test_real_flag_gems_eye_case_covers_2d_select_mask_and_fixed_c_family() -> None:
    case = build_heldout_case("fg_eye")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "eye_m.py"
    assert case.function_name == "eye_kernel"
    assert "tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)" in case.source_text
    assert view["arguments"][0]["shape"] == [37, 29]
    assert view["grid"] == [2, 1]
    assert view["grid_contract"] == [
        "ceil_div(N, BLOCK_i)",
        "ceil_div(M, BLOCK_j)",
    ]
    assert view["schedule_values"]["BLOCK_i"] == 32
    assert view["schedule_values"]["BLOCK_j"] == 32
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one_by_one",
        "block_edge",
        "both_axes_tail",
        "rectangular_nondivisible",
    ]
    assert case.gate1_specialization_profiles == case.gate2_shape_profiles
    assert case.gate1_primary_profile_id == "rectangular_nondivisible"


def test_real_flag_gems_trace_case_exercises_a_multi_iteration_diagonal_reduce() -> None:
    case = build_heldout_case("fg_trace")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "trace.py"
    assert case.function_name == "trace_kernel"
    assert "for i in range(0, tl.cdiv(num_diag, BLOCK_SIZE))" in case.source_text
    assert by_name["inp_ptr"]["shape"] == [1031, 1025]
    assert by_name["out_ptr"]["shape"] == []
    assert by_name["num_diag"]["value"] == 1025
    assert by_name["stride0"]["value"] == 1025
    assert by_name["stride1"]["value"] == 1
    assert view["grid"] == [1]
    assert view["schedule_values"]["BLOCK_SIZE"] == 1024
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "short_trip",
        "exact_block_trip",
        "block_plus_one_trip",
    ]
    assert case.gate1_specialization_profiles == case.gate2_shape_profiles
    assert case.gate1_primary_profile_id == "block_plus_one_trip"
    exact_block = case.gate2_shape_profiles[1]
    assert exact_block.tensor_shapes["inp_ptr"] == (1031, 1025)
    assert exact_block.scalar_values == {
        "num_diag": 1024,
        "stride0": 1025,
        "stride1": 1,
    }


def test_real_flag_gems_log_softmax_case_covers_loop_carry_and_two_tail_axes() -> None:
    case = build_heldout_case("fg_log_softmax")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "log_softmax.py"
    assert case.function_name == "log_softmax_kernel"
    assert "for start_n in range(0, N, BLOCK_N)" in case.source_text
    assert "z = tl.where(all_neg_inf, z," in case.source_text
    assert by_name["output_ptr"]["shape"] == [5, 513, 3]
    assert by_name["input_ptr"]["shape"] == [5, 513, 3]
    assert by_name["M"]["value"] == 5
    assert by_name["N"]["value"] == 513
    assert by_name["K"]["value"] == 3
    assert view["grid"] == [1, 3]
    assert view["grid_contract"] == ["ceil_div(M, BLOCK_M)", "K"]
    assert view["schedule_values"]["BLOCK_M"] == 8
    assert view["schedule_values"]["BLOCK_N"] == 256
    assert view["schedule_values"]["num_warps"] == 8
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "single_n_tile",
        "exact_tiles",
        "multi_n_tail",
        "multi_m_and_n_tail",
    ]
    assert case.gate1_specialization_profiles == case.gate2_shape_profiles
    assert case.gate1_primary_profile_id == "multi_n_tail"


def test_real_flag_gems_amax_case_uses_the_reviewed_final_reduce_stage() -> None:
    case = build_heldout_case("fg_amax")
    view = case.compiler_view().to_json()

    assert case.source_path.name == "amax.py"
    assert case.function_name == "amax_kernel_2"
    assert "amax_val = tl.max(mid_val)" in case.source_text
    assert view["arguments"][0]["shape"] == [63]
    assert view["arguments"][1]["shape"] == []
    assert view["arguments"][2]["value"] == 63
    assert view["grid"] == [1, 1]
    assert view["schedule_values"]["BLOCK_MID"] == 64
    assert [profile.profile_id for profile in case.gate2_shape_profiles] == [
        "one",
        "block_minus_one",
        "block",
    ]
    assert [profile.profile_id for profile in case.gate2_value_profiles] == [
        "normal",
        "uniform",
        "large_dynamic",
    ]


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


def test_real_index_select_adds_two_axis_indirect_gather_family() -> None:
    case = build_heldout_case("fg_index_select")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "index_select.py"
    assert case.function_name == "index_select_kernel"
    assert "inp_off = rows_offsets * N + indices[None, :]" in case.source_text
    assert "final_mask = out_mask & index_valid_mask" in case.source_text
    assert by_name["inp"]["shape"] == [19, 23]
    assert by_name["out"]["shape"] == [19, 17]
    assert by_name["index"] == {
        "name": "index",
        "kind": "tensor",
        "dtype": "i64",
        "shape": [17],
    }
    assert view["grid"] == [3, 1]
    assert view["grid_contract"] == [
        "ceil_div(M, BLOCK_M)",
        "ceil_div(index_len, BLOCK_N)",
    ]
    assert view["constexpr_parameters"] == ["BLOCK_M", "BLOCK_N"]
    assert view["schedule_values"]["BLOCK_M"] == 8
    assert view["schedule_values"]["BLOCK_N"] == 32
    assert case.input_distributions["index"] == "positive"
    assert case.input_scales["index"] == 22.0
    assert [
        profile.profile_id for profile in case.gate1_specialization_profiles
    ] == ["single", "rectangular_tail", "two_axis_multitile"]
    assert case.gate1_primary_profile_id == "rectangular_tail"
    assert case.gate2_shape_profiles == case.gate1_specialization_profiles
    assert [profile.profile_id for profile in case.gate2_value_profiles] == [
        "random_index_normal_input",
        "boundary_index_uniform_input",
    ]


def test_real_diag_extract_adds_strided_gather_and_fixed_c_family() -> None:
    case = build_heldout_case("fg_diag_extract")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "diag.py"
    assert case.function_name == "diag_2d_to_1d_kernel"
    assert "row_idx * stride0 + col_idx * stride1" in case.source_text
    assert by_name["data_ptr"]["shape"] == [11, 13]
    assert by_name["output_ptr"]["shape"] == [11]
    assert by_name["stride0"]["value"] == 13
    assert view["grid"] == [1]
    assert view["grid_contract"] == ["ceil_div(N, BLOCK_SIZE)"]
    assert view["constexpr_parameters"] == ["diagonal", "BLOCK_SIZE"]
    assert view["schedule_values"]["diagonal"] == 2
    assert [
        profile.profile_id for profile in case.gate1_specialization_profiles
    ] == ["single", "rectangular_tail", "multi_block"]
    assert case.gate1_primary_profile_id == "rectangular_tail"
    assert case.gate2_shape_profiles == case.gate1_specialization_profiles


def test_real_skip_layer_norm_adds_non_reduce_norm_contract() -> None:
    case = build_heldout_case("fg_skip_layer_norm")
    view = case.compiler_view().to_json()
    by_name = {item["name"]: item for item in view["arguments"]}

    assert case.source_path.name == "skip_layernorm.py"
    assert case.function_name == "skip_layer_norm_kernel"
    assert "mean = tl.sum(x, axis=0) / N" in case.source_text
    assert "x_hat = (x - mean) * rstd" in case.source_text
    assert by_name["Y"]["shape"] == [7, 257]
    assert by_name["X"]["shape"] == [7, 257]
    assert by_name["W"]["shape"] == [257]
    assert by_name["y_stride_r"]["value"] == 257
    assert by_name["eps"]["value"] == 1.0e-5
    assert view["grid"] == [7]
    assert view["grid_contract"] == ["7"]
    assert view["constexpr_parameters"] == ["BLOCK_SIZE"]
    assert view["schedule_values"]["BLOCK_SIZE"] == 512
    assert [
        profile.profile_id for profile in case.gate1_specialization_profiles
    ] == [
        "single_width",
        "rectangular_tail",
        "block_minus_one",
        "block",
    ]
    assert case.gate1_primary_profile_id == "rectangular_tail"
    assert case.gate2_shape_profiles == case.gate1_specialization_profiles
    assert [
        profile.profile_id for profile in case.gate2_value_profiles
    ] == ["normal", "large_dynamic", "near_constant"]


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
            "gate2": {"status": "execution_failed"},
        }
    ) == "execution_failed"
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
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
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
        "gate2_execution_protocol": GATE2_EXECUTION_PROTOCOL,
        "gate2_attempted": True,
        "gate2_comparison_completed": True,
        "gate2_status": "gate2_pass",
        "gate2_pass": True,
        "gate2_case_statuses": ["gate2_pass"],
    }


@pytest.mark.parametrize(
    "suite",
    [
        {
            "passed": True,
            "cases": [
                {"case_id": "c0", "status": "gate2_pass"},
                {"case_id": "c1"},
            ],
        },
        {
            "passed": True,
            "cases": [
                {"case_id": "same", "status": "gate2_pass"},
                {"case_id": "same", "status": "gate2_pass"},
            ],
        },
        {
            "passed": True,
            "cases": [
                {"case_id": "c0", "status": "gate2_pass"},
                {"status": "gate2_pass"},
            ],
        },
        {
            "passed": True,
            "cases": [
                {"case_id": "c0", "status": "gate2_pass"},
                {"case_id": "c1", "status": "replay_diverged"},
            ],
        },
        {
            "passed": True,
            "status_counts": {"gate2_pass": 1},
            "cases": [
                {"case_id": "c0", "status": "gate2_pass"},
                {"case_id": "c1", "status": "gate2_pass"},
            ],
        },
    ],
    ids=(
        "missing-status",
        "duplicate-case-id",
        "partially-missing-case-id",
        "inconsistent-top-level-passed",
        "incomplete-status-counts",
    ),
)
def test_gate2_suite_metrics_fail_closed_on_incomplete_or_inconsistent_cases(
    suite,
) -> None:
    suite = {"execution_protocol": GATE2_EXECUTION_PROTOCOL, **suite}
    observation = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "replay_closed"},
            "gate2_suite": suite,
        }
    )

    assert observation["gate2_attempted"] is True
    assert observation["gate2_comparison_completed"] is False
    assert observation["gate2_pass"] is None
    assert observation["gate2_status"] == "gate2_suite_incomplete"


def test_gate2_suite_metrics_accept_a_complete_all_pass_suite() -> None:
    observation = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "replay_closed"},
            "gate2_suite": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "passed": True,
                "status_counts": {"gate2_pass": 2},
                "cases": [
                    {"case_id": "c0", "status": "gate2_pass"},
                    {"case_id": "c1", "status": "gate2_pass"},
                ],
            },
        }
    )

    assert observation["gate2_comparison_completed"] is True
    assert observation["gate2_pass"] is True
    assert observation["gate2_status"] == "gate2_pass"


def test_gate2_suite_metrics_accept_a_complete_semantic_nonpass_suite() -> None:
    observation = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "no_closure"},
            "gate2_diagnostic_suite": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "passed": False,
                "status_counts": {
                    "gate2_pass": 1,
                    "replay_diverged": 1,
                },
                "cases": [
                    {"case_id": "c0", "status": "gate2_pass"},
                    {"case_id": "c1", "status": "replay_diverged"},
                ],
            },
        }
    )

    assert observation["gate2_comparison_completed"] is True
    assert observation["gate2_pass"] is False
    assert observation["gate2_status"] == "gate2_suite_failed"


def test_gate2_metrics_reject_legacy_three_way_status_without_active_protocol() -> None:
    observation = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {"status": "gate2_pass"},
        }
    )

    assert observation["gate2_attempted"] is True
    assert observation["gate2_execution_protocol"] is None
    assert observation["gate2_comparison_completed"] is False
    assert observation["gate2_status"] == "gate2_protocol_mismatch"
    assert observation["gate2_pass"] is None


def test_batch_does_not_count_execution_or_setup_failure_as_semantic_nonpass() -> None:
    execution_failed = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "execution_failed",
            },
        }
    )
    setup_failed = extract_gate1_gate2_observation(
        {
            "gate1": {"status": "no_closure"},
            "gate2_diagnostic_suite": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "setup_failed",
            },
        }
    )

    assert execution_failed["gate2_attempted"]
    assert not execution_failed["gate2_comparison_completed"]
    assert execution_failed["gate2_pass"] is None
    assert setup_failed["gate2_attempted"]
    assert not setup_failed["gate2_comparison_completed"]
    assert setup_failed["gate2_status"] == "setup_failed"
    assert setup_failed["gate2_pass"] is None


def test_batch_conditional_metrics_report_probability_and_evaluation_coverage() -> None:
    reports = [
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "gate2_pass",
            },
        },
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "replay_diverged",
            },
        },
        {
            "gate1": {"status": "leakage_rejected"},
            "gate2_diagnostic_suite": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
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

    assert metrics["schema"] == "scrc_gate1_gate2_conditional_v3"
    assert metrics["quadrants"] == {
        "gate1_closed_gate2_pass": 1,
        "gate1_closed_gate2_nonpass": 1,
        "gate1_not_closed_gate2_pass": 1,
        "gate1_not_closed_gate2_nonpass": 0,
    }
    assert metrics["quadrant_coverage"] == {
        "eligible_completed_samples": 3,
        "attempted_but_incomplete": 0,
        "unattempted": 1,
        "completion_coverage": 0.75,
    }
    assert metrics["replay_closed"] == {
        "samples": 2,
        "gate2_attempted": 2,
        "gate2_comparison_completed": 2,
        "gate2_pass": 1,
        "gate2_nonpass": 1,
        "attempt_coverage": 1.0,
        "completion_coverage": 1.0,
        "pass_probability_given_completed": 0.5,
    }
    assert metrics["not_replay_closed"] == {
        "samples": 2,
        "gate2_attempted": 1,
        "gate2_comparison_completed": 1,
        "gate2_pass": 1,
        "gate2_nonpass": 0,
        "attempt_coverage": 0.5,
        "completion_coverage": 0.5,
        "pass_probability_given_completed": 1.0,
    }
    assert metrics["by_gate1_status"]["no_closure"][
        "pass_probability_given_completed"
    ] is None


def test_gate1_gate2_quadrants_exclude_execution_and_setup_failure() -> None:
    reports = [
        {
            "gate1": {"status": "replay_closed"},
            "gate2": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "execution_failed",
            },
        },
        {
            "gate1": {"status": "no_closure"},
            "gate2_diagnostic_suite": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "setup_failed",
            },
        },
        {
            "gate1": {"status": "no_closure"},
            "gate2_diagnostic": {
                "execution_protocol": GATE2_EXECUTION_PROTOCOL,
                "status": "replay_diverged",
            },
        },
    ]
    metrics = summarize_gate1_gate2(
        [
            {"gate1_gate2": extract_gate1_gate2_observation(report)}
            for report in reports
        ]
    )

    assert metrics["quadrants"] == {
        "gate1_closed_gate2_pass": 0,
        "gate1_closed_gate2_nonpass": 0,
        "gate1_not_closed_gate2_pass": 0,
        "gate1_not_closed_gate2_nonpass": 1,
    }
    assert metrics["quadrant_coverage"] == {
        "eligible_completed_samples": 1,
        "attempted_but_incomplete": 2,
        "unattempted": 0,
        "completion_coverage": 1 / 3,
    }


def test_performance_distribution_does_not_turn_an_empty_sample_into_zero_time() -> None:
    assert _timing_distribution([]) == {
        "count": 0,
        "total_ms": None,
        "mean_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "min_ms": None,
        "max_ms": None,
    }
    assert _timing_distribution([1.0, 2.0, 100.0]) == {
        "count": 3,
        "total_ms": 103.0,
        "mean_ms": 34.333,
        "p50_ms": 2.0,
        "p95_ms": 100.0,
        "min_ms": 1.0,
        "max_ms": 100.0,
    }


def test_batch_performance_summary_keeps_gate1_rates_and_missing_stages_explicit() -> None:
    rows = [
        {
            "case": "closed",
            "status": "gate2_pass",
            "performance": {
                "timing_ms": {
                    "preflight_compile": 3.0,
                    "initial_lift": 10.0,
                    "gate1_loop": 20.0,
                    "specialization_family": None,
                    "pi_ablation": 2.0,
                    "gate2": 5.0,
                    "total": 37.0,
                    "heldout_total": 40.0,
                },
                "recorded_invocations": {
                    "normal_replay_compile": 2,
                    "normal_replay_compile_ok": 1,
                    "writer_total": 4,
                },
                "gate1_observation": {
                    "status": "replay_closed",
                    "outer_loop_termination": "replay_closed",
                    "relift_attempts": 1,
                    "masked_j_outcome": "hard_mismatch",
                },
            },
        },
        {"case": "unmeasured", "status": "harness_error"},
    ]

    summary = _summarize_performance(rows)

    assert summary["selected_cases"] == 2
    assert summary["measured_cases"] == 1
    assert summary["missing_measurement_cases"] == 1
    assert summary["timing"]["specialization_family"]["count"] == 0
    assert summary["timing"]["specialization_family"]["mean_ms"] is None
    assert summary["gate1"]["status_counts"] == {"replay_closed": 1}
    assert summary["gate1"]["outer_loop_termination_counts"] == {
        "replay_closed": 1
    }
    assert summary["gate1"]["replay_compile_attempts"] == 2
    assert summary["gate1"]["replay_compile_ok"] == 1
    assert summary["gate1"]["replay_compile_rate"] == 0.5
    assert summary["gate1"]["cases_with_relift"] == 1
    assert summary["gate1"]["masked_j_outcomes"] == {
        "hard_mismatch": 1
    }


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
    assert identity["schema"] == 35
    assert identity["composer_version"] == "deterministic_composer_v5"
    assert identity["performance_schema"] == "scrc_case_performance_v1"
    assert identity["gate1_gate2_metrics_schema"] == (
        "scrc_gate1_gate2_conditional_v3"
    )
    assert identity["relift_rounds"] == 3
    assert identity["semantic_interventions"] == 3
    assert identity["fixed_c_family_closure_policy"] == (
        "normal_paired_and_masked_fixed_c_family_all_required_v1"
    )
    assert identity["paired_intervention_family_policy"] == (
        "fixed_c_family_exact_delta_all_profiles_v1"
    )
    assert identity["paired_intervention_scope"] == "primary_profile"
    assert identity["normal_recovery"] == "relift_only_restart_full_pipeline"
    assert identity["replay_generation_policy"] == (
        "fresh_only_no_prior_replay_source"
    )
    assert identity["gate1_stagnation_policy"] == {
        "policy": "exact_artifact_and_hard_diff_v1",
        "required_consecutive_observations": 2,
        "basis": "canonical_artifact_and_exact_hard_diff",
    }
    assert identity["ttir_fingerprint_schema"] == (
        "triton-ttir-3.6-gate1-v3"
    )
    assert identity["relift_guidance_policy"] == {
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
    assert identity["gate1_termination_policy"] == {
        "policy": "structured_no_closure_termination_v1",
        "no_closure_requires_structured_reason": True,
        "gate2_verdict_independence": True,
    }
    assert identity["gate2_execution_protocol"] == (
        "source_replay_dynamic_equivalence_v1"
    )
    assert identity["gate2_execution_legs"] == ["source", "replay"]
    assert identity["gate2_comparison_protocol"] == (
        "exact_integer_tolerant_float_v2"
    )
    assert identity["pi_ablations"]
    assert identity["pi_ablation_limit"] == 3
    assert identity["pi_ablation_kinds"] == []
    assert identity["pi_ablation_requires_closed_control"] is True
    assert identity["pi_ablation_opaque_reference_discipline"] == (
        "resolve_handles_never_emit_v1"
    )
    assert identity["maximum_provider_attempts_per_generation"] == 3
    assert identity["ttir_score_semantics"] == "active_evidence_weighted_v2"
    assert identity["masked_j_policy"] == (
        "actual_lift_pi_masked_j_hard_nonclosure_v3"
    )
    assert identity["masked_j_family_policy"] == (
        "actual_lift_pi_fixed_c_family_all_profile_hard_mismatch_v1"
    )
    assert identity["masked_j_scope"] == "primary_profile"
    assert identity["masked_j_required_outcome"] == (
        "compile_or_hard_nonclosure"
    )
    assert identity["masked_j_score_role"] == "not_used"
    assert identity["replay_reference_naming"] == "opaque_replay_refs_v1"
    assert identity["relift_protocol"] == (
        "gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11"
    )
    assert identity["gate1_specialization_policy"] == {
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
    ones_identity = _batch_identity(
        build_heldout_case("fg_ones"), _batch_args()
    )
    family_policy = ones_identity["gate1_specialization_policy"]
    assert family_policy["enabled"]
    assert family_policy["anti_leakage_scope"] == "full_fixed_c_family"
    assert ones_identity["paired_intervention_scope"] == (
        "full_fixed_c_family"
    )
    assert ones_identity["masked_j_scope"] == "full_fixed_c_family"
    assert ones_identity["masked_j_required_outcome"] == (
        "all_profiles_legal_comparable_hard_mismatch"
    )
    assert family_policy["family"]["primary_profile_id"] == "non_divisible"
    assert [
        profile["profile_id"]
        for profile in family_policy["family"]["profiles"]
    ] == [
        "one",
        "block_minus_one",
        "block",
        "block_plus_one",
        "non_divisible",
        "far_multitile",
    ]
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
