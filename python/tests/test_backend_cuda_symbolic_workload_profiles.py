from pathlib import Path

import pytest

from intentir.backend.cuda.production_cases import (
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
    constexpr_input_bindings,
    prepare_production_workload,
    profile_production_snapshot,
)
from intentir.backend.cuda.search import enumerate_cuda_candidates
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.scrc_workload_run import build_workload_jc_suite


SNAPSHOTS = Path("corpus/production-jc-v1")
FRESH_SNAPSHOTS = Path("corpus/production-jc-v2")


@pytest.mark.parametrize(
    ("name", "expected_dims", "expected_grid", "baseline_policy"),
    (
        ("dyt", {"x0": 1024}, (1, 2), "source_and_refj"),
        ("flash_bwd_preprocess", {"x0": 32}, (2, 32), "source_and_refj"),
        (
            "flash_layer_norm_1pass",
            {"x0": 128, "x1": 1024},
            (128,),
            "source_and_refj",
        ),
        ("fused_add_rms_norm", {"x0": 1024}, (2,), "source_and_refj"),
        ("geglu", {"x0": 1024}, (2,), "source_and_refj"),
        ("jsd", {"x0": 128, "x1": 1024}, (128,), "source_and_refj"),
        ("kl_div", {"x0": 1024}, (2,), "source_and_refj"),
        ("layer_norm", {"x0": 128, "x1": 1024}, (128,), "source_and_refj"),
        ("liger_rms_norm", {"x0": 1024}, (2,), "source_and_refj"),
        ("masked_copy", {"x0": 1024 * 1024}, (1024,), "target_refj_only"),
        ("masked_row_square_sum", {"x0": 1024}, (2,), "source_and_refj"),
        ("poly_norm", {"x0": 1024}, (2,), "source_and_refj"),
        ("softmax", {"x0": 1024}, (2,), "source_and_refj"),
        ("sparsemax", {"x0": 128, "x1": 1024}, (128,), "source_and_refj"),
        ("swiglu", {"x0": 1024}, (2,), "source_and_refj"),
        ("tvd", {"x0": 128, "x1": 1024}, (128,), "source_and_refj"),
        ("xformers_rms_norm", {"x0": 1024}, (2,), "source_and_refj"),
        ("xformers_rms_norm_add", {"x0": 1024}, (2,), "source_and_refj"),
    ),
)
def test_symbolic_medium_profile_revalues_harness_without_changing_j(
    name, expected_dims, expected_grid, baseline_policy
):
    base, _ = load_production_snapshot(SNAPSHOTS, name)
    prepared = prepare_production_workload(
        name, profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
    )
    active, record = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
    )
    suite = build_workload_jc_suite(prepared, active.j, active.c)

    assert active.j_digest == base.j_digest
    assert record["j_unchanged"] is True
    assert record["kind"] == "symbolic_extent_valuation"
    assert suite.cases[0].refeval.dims_map == expected_dims
    assert prepared.adapter.grid == expected_grid
    assert record["baseline_policy"] == baseline_policy


@pytest.mark.parametrize(
    ("name", "buffer_shapes", "scalar_values"),
    (
        (
            "flash_layer_norm_1pass",
            {
                "X": (128, 1024),
                "W": (1024,),
                "ROWSCALE": (128,),
                "DROPOUT_MASK": (128, 1024),
                "Mean": (128,),
            },
            {"stride_x_row": 1024, "M": 128, "N": 1024, "eps": 1.0e-6},
        ),
        (
            "jsd",
            {"X_ptr": (128, 1024), "label_ptr": (128,)},
            {"X_stride": 1024, "n_non_ignore": 128, "n_cols": 1024},
        ),
        (
            "layer_norm",
            {"X_ptr": (128, 1024), "W_ptr": (1024,), "Mean_ptr": (128,)},
            {"X_row_stride": 1024, "W_row_stride": 1024, "n_cols": 1024},
        ),
        (
            "sparsemax",
            {"x_ptr": (128, 1024), "o_ptr": (128, 1024)},
            {"x_stride_row": 1024, "o_stride_row": 1024, "n_cols": 1024},
        ),
        (
            "tvd",
            {"p_ptr": (128, 1024), "label_ptr": (128,)},
            {"p_stride": 1024, "grads_stride": 1024, "n_cols": 1024, "scale": 1.0},
        ),
    ),
)
def test_symbolic_row_profiles_use_explicit_adapter_axis_bindings(
    name, buffer_shapes, scalar_values
):
    prepared = prepare_production_workload(
        name, profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
    )
    actual_shapes = {item.formal: item.shape for item in prepared.adapter.buffers}
    actual_scalars = {item.formal: item.value for item in prepared.adapter.scalars}

    assert {key: actual_shapes[key] for key in buffer_shapes} == buffer_shapes
    assert {key: actual_scalars[key] for key in scalar_values} == scalar_values


def test_flash_preprocess_symbolic_profile_derives_strides_from_literal_layout():
    prepared = prepare_production_workload(
        "flash_bwd_preprocess",
        profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
    )
    shapes = {item.formal: item.shape for item in prepared.adapter.buffers}
    scalars = {item.formal: item.value for item in prepared.adapter.scalars}

    assert shapes == {
        "Out": (1, 32, 33, 16),
        "DO": (1, 32, 33, 16),
        "Delta": (1, 32, 64),
    }
    assert scalars["stride_ob"] == 32 * 33 * 16
    assert scalars["stride_dob"] == 32 * 33 * 16
    assert scalars["stride_oh"] == 33 * 16
    assert scalars["stride_om"] == 16
    assert prepared.adapter.grid == (2, 32)


@pytest.mark.parametrize(
    ("name", "expected_dims", "expected_grid"),
    (
        ("flash_bwd_preprocess", {"x0": 256}, (2, 256)),
        ("layer_norm", {"x0": 2048, "x1": 4096}, (2048,)),
        ("sparsemax", {"x0": 2048, "x1": 4096}, (2048,)),
    ),
)
def test_symbolic_large_profile_uses_only_declared_j_dimensions(
    name, expected_dims, expected_grid
):
    base, _ = load_production_snapshot(SNAPSHOTS, name)
    prepared = prepare_production_workload(
        name, profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    active, _ = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    suite = build_workload_jc_suite(prepared, active.j, active.c)

    assert active.j_digest == base.j_digest
    assert suite.cases[0].refeval.dims_map == expected_dims
    assert prepared.adapter.grid == expected_grid


def test_symbolic_profile_preserves_literal_row_softmax_while_revaluing_feature():
    base, _ = load_production_snapshot(SNAPSHOTS, "softmax")
    prepared = prepare_production_workload(
        "softmax", profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
    )
    active, record = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM
    )

    shapes = {item.formal: item.shape for item in prepared.adapter.buffers}
    assert shapes == {"Y_ptr": (2, 1024), "X_ptr": (2, 1024)}
    assert prepared.adapter.grid == (2,)
    assert active.j_digest == base.j_digest
    assert record["literal_constraints"] == ["rows=2"]


def test_large_reduction_c_prior_orders_exact_cuda_retile_first():
    base, _ = load_production_snapshot(SNAPSHOTS, "layer_norm")
    prepared = prepare_production_workload(
        "layer_norm", profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    active, _ = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    case = build_workload_jc_suite(prepared, active.j, active.c).cases[0]
    materialized = materialize_j(active.j, case.refeval.dims_map)
    state = attach_c(materialized, active.c, BackendTarget("cuda", "sm_120a"))
    candidates = enumerate_cuda_candidates(state)
    workgroup_plain = [
        item
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
        and item.schedule.get("block_threads", 32) > 32
        and item.schedule.get("fuse_fma") is False
        and item.schedule.get("approximate_div") is False
        and not item.b_transformations
    ]

    assert workgroup_plain[0].schedule["block_threads"] == 512
    assert workgroup_plain[0].schedule["tile_cols"] == 32


def test_dense_transcendental_j_retiles_same_c_to_more_cuda_threads():
    base, _ = load_production_snapshot(SNAPSHOTS, "jsd")
    prepared = prepare_production_workload(
        "jsd", profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    active, _ = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    case = build_workload_jc_suite(prepared, active.j, active.c).cases[0]
    materialized = materialize_j(active.j, case.refeval.dims_map)
    state = attach_c(materialized, active.c, BackendTarget("cuda", "sm_120a"))
    candidates = enumerate_cuda_candidates(state)
    workgroup_plain = [
        item
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
        and item.schedule.get("block_threads", 32) > 32
        and item.schedule.get("fuse_fma") is False
        and item.schedule.get("approximate_div") is False
        and not item.b_transformations
    ]

    assert workgroup_plain[0].schedule["block_threads"] == 1024


def test_tiny_row_elementwise_c_retile_targets_grid_stride_transition():
    base, _ = load_production_snapshot(SNAPSHOTS, "dyt")
    prepared = prepare_production_workload(
        "dyt", profile=PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    active, _ = profile_production_snapshot(
        base, prepared, PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE
    )
    case = build_workload_jc_suite(prepared, active.j, active.c).cases[0]
    materialized = materialize_j(active.j, case.refeval.dims_map)
    state = attach_c(materialized, active.c, BackendTarget("cuda", "sm_120a"))
    candidates = enumerate_cuda_candidates(state)
    elementwise_base = [
        item
        for item in candidates
        if item.lowering_path_kind == "elementwise"
        and not item.b_transformations
        and item.schedule.get("tanh_approximation") == "stable-ex2"
    ]

    assert elementwise_base[0].schedule["block_threads"] == 64


@pytest.mark.parametrize(
    ("profile", "expected_threads"),
    (
        (PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM, 512),
        (PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE, 64),
    ),
)
def test_elementwise_c_retile_uses_j_rows_for_target_occupancy(
    profile, expected_threads
):
    base, _ = load_production_snapshot(SNAPSHOTS, "tvd")
    prepared = prepare_production_workload("tvd", profile=profile)
    active, _ = profile_production_snapshot(base, prepared, profile)
    case = build_workload_jc_suite(prepared, active.j, active.c).cases[0]
    constexprs = constexpr_input_bindings(
        active.j,
        case.refeval.input_binding_map,
        case.values,
    )
    materialized = materialize_j(
        active.j,
        case.refeval.dims_map,
        constexpr_bindings=constexprs,
    )
    state = attach_c(materialized, active.c, BackendTarget("cuda", "sm_120a"))
    candidates = enumerate_cuda_candidates(state)
    elementwise_base = [
        item
        for item in candidates
        if item.lowering_path_kind == "elementwise"
        and not item.b_transformations
        and item.schedule.get("tanh_approximation") == "stable-ex2"
    ]

    assert elementwise_base[0].schedule["block_threads"] == expected_threads


def test_tvd_functional_loop_cost_scales_linearly_with_symbolic_tensor_domain():
    costs = []
    for profile in (
        PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
        PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    ):
        base, _ = load_production_snapshot(SNAPSHOTS, "tvd")
        prepared = prepare_production_workload("tvd", profile=profile)
        active, _ = profile_production_snapshot(base, prepared, profile)
        case = build_workload_jc_suite(prepared, active.j, active.c).cases[0]
        constexprs = constexpr_input_bindings(
            active.j,
            case.refeval.input_binding_map,
            case.values,
        )
        materialized = materialize_j(
            active.j,
            case.refeval.dims_map,
            constexpr_bindings=constexprs,
        )
        costs.append(derive_j_workload_cost(materialized))

    medium, large = costs
    element_ratio = (2048 * 4096) // (128 * 1024)
    # Two scalar operations are shape-independent; tensor work is exactly linear.
    assert large.floating_point_ops - 2 == (
        medium.floating_point_ops - 2
    ) * element_ratio
    assert large.select_ops == medium.select_ops * element_ratio
    assert large.arithmetic_intensity_flop_per_byte < 2.0


@pytest.mark.parametrize(
    ("name", "formal", "shape", "grid"),
    (
        ("swiglu", "a_ptr", (16384, 11008), (16384,)),
        ("liger_rms_norm", "X_ptr", (2048, 4096), (2048,)),
        ("fused_add_rms_norm", "X_ptr", (2048, 4096), (2048,)),
        ("geglu", "a", (32768, 11008), (32768,)),
        ("layer_norm", "X_ptr", (4096, 4096), (4096,)),
        ("softmax", "X_ptr", (2048, 4096), (2048,)),
        ("dyt", "X", (4096, 4096), (1, 4096)),
        ("sparsemax", "x_ptr", (2048, 4096), (2048,)),
        ("kl_div", "y_ptr", (4096, 4096), (4096,)),
        ("poly_norm", "X_ptr", (2048, 4096), (2048,)),
        (
            "flash_bwd_preprocess",
            "Out",
            (4, 16, 8192, 128),
            (256, 64),
        ),
        ("flash_layer_norm_1pass", "X", (4096, 4096), (4096,)),
        ("xformers_rms_norm_add", "x_ptr", (4096, 12288), (4096,)),
        (
            "splitk_reduce_varargs",
            "Out_splitK0",
            (1, 1, 16, 1, 256),
            (1, 16, 1),
        ),
    ),
)
def test_fresh14_natural_headline_revalues_declared_symbols_only(
    name, formal, shape, grid
):
    base, _ = load_production_snapshot(FRESH_SNAPSHOTS, name)
    prepared = prepare_production_workload(
        name,
        profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    active, receipt = profile_production_snapshot(
        base,
        prepared,
        PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    shapes = {item.formal: item.shape for item in prepared.adapter.buffers}

    assert shapes[formal] == shape
    assert prepared.adapter.grid == grid
    assert active.j_digest == base.j_digest
    assert receipt["j_unchanged"] is True
    assert receipt["profile"] == PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE


def test_natural_flash_and_splitk_contiguous_stride_products_are_revalued():
    flash = prepare_production_workload(
        "flash_bwd_preprocess",
        profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    ).adapter
    flash_scalars = {item.formal: item.value for item in flash.scalars}
    assert flash_scalars["stride_ob"] == 16 * 8192 * 128
    assert flash_scalars["stride_oh"] == 8192 * 128
    assert flash_scalars["stride_om"] == 128
    assert flash_scalars["seqlen_q"] == 8192
    assert flash_scalars["headdim"] == 128

    splitk = prepare_production_workload(
        "splitk_reduce_varargs",
        profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    ).adapter
    splitk_scalars = {item.formal: item.value for item in splitk.scalars}
    splitk_constexprs = dict(splitk.semantic_constexprs)
    assert splitk_scalars["stride_osk_z0"] == 16 * 1 * 256
    assert splitk_scalars["stride_osk_h0"] == 256
    assert splitk_scalars["stride_lsek_z0"] == 16
    assert splitk_constexprs["head_dim"] == 256
    assert splitk_constexprs["H"] == 16
    assert splitk.vararg_arity == 2
