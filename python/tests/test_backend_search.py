from __future__ import annotations

from pathlib import Path

import pytest

import intentir.backend.cuda.search as cuda_search
from direct_fixtures import make_reduce_artifact
from intentir.backend.cuda.search import (
    B_CAP_RANK0_HOST_CAPTURE,
    CudaContractError,
    SEARCH_PARTITION_EXPANDED,
    SEARCH_PARTITION_SCHEDULE_SPINE,
    canonical_cuda_candidate,
    derive_cuda_contract_view,
    derive_default_fast_path_id,
    enumerate_cuda_candidates,
    project_cuda_c_fixed_candidate,
    schedule_only_cuda_path_identity,
)
from intentir.backend.cuda.production_cases import (
    constexpr_input_bindings,
    prepare_production_workload,
)
from intentir.backend.cuda.schedules import (
    ELEMENTWISE_FAST_PATH_ID,
    ROW_FAST_PATH_ID,
    ElementwiseSchedule,
    PaddedRowSchedule,
    RowSchedule,
)
from intentir.backend.materialize import CPrior, attach_c, materialize_j
from intentir.backend.workload_valuation import (
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
)
from intentir.backend.ablation import ExperimentArm, standard_ablation_plan
from intentir.backend.protocol import BackendTarget
from intentir.backend.search_and_close import ObservationClass
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import AliasKind, AliasSpec
from intentir.verify.scrc_workload_run import build_workload_jc_suite


def _states():
    artifact = canonicalize_gate1_artifact(
        make_reduce_artifact(explicit_accesses=False)
    )
    base = materialize_j(artifact.j, {"x0": 2, "x1": 128})
    target = BackendTarget("cuda", "sm_120a")
    return (
        attach_c(base, artifact.c, target),
        attach_c(base, None, target),
    )


def _fresh_splitk_state():
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v2",
        "splitk_reduce_varargs",
    )
    prepared = prepare_production_workload("splitk_reduce_varargs")
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _fresh_softmax_state():
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v2",
        "softmax",
    )
    prepared = prepare_production_workload("softmax")
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _fresh_dyt_state():
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v2",
        "dyt",
    )
    prepared = prepare_production_workload("dyt")
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _fresh_fused_add_rms_norm_state():
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v2",
        "fused_add_rms_norm",
    )
    prepared = prepare_production_workload("fused_add_rms_norm")
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _stage2_flash_linear_state():
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v3",
        "flash_linear",
    )
    prepared = prepare_production_workload(
        "flash_linear",
        profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _stage2_state(name: str):
    root = Path(__file__).parents[2]
    snapshot, _path = load_production_snapshot(
        root / "corpus" / "production-jc-v3",
        name,
    )
    prepared = prepare_production_workload(name)
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    base = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def test_c_changes_only_candidate_order() -> None:
    with_c, without_c = _states()
    ordered = enumerate_cuda_candidates(with_c)
    unseeded = enumerate_cuda_candidates(without_c)

    assert ordered[0].lowering_path_kind == "generic"
    assert unseeded[0].lowering_path_kind == "generic"
    assert {item.candidate_id for item in ordered} == {
        item.candidate_id for item in unseeded
    }
    assert [item.lowering_path_kind for item in ordered[1:4]] == [
        "elementwise",
        "row_reduction",
        "padded_row_reduction",
    ]
    row_arguments = [
        item.pass_argument
        for item in ordered
        if item.lowering_path_kind == "row_reduction"
    ]
    assert any("fuse-fma=false" in argument for argument in row_arguments)
    assert any("fuse-fma=true" in argument for argument in row_arguments)
    assert not any("approximate-div=true" in argument for argument in row_arguments)


def test_numeric_states_require_a_visible_operation() -> None:
    softmax = _fresh_softmax_state()
    axes = next(
        item
        for item in standard_ablation_plan(96)
        if item.arm is ExperimentArm.N3_PRECISION_APPROX
    )
    candidates = enumerate_cuda_candidates(softmax, axes=axes)

    assert any(
        item.schedule.get("approximate_div") is True
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
    )
    assert all(
        item.schedule.get("fuse_fma") is not True
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
    )


def test_intentir_complete_space_crosses_softmax_numeric_lowering_with_schedules() -> None:
    softmax = _fresh_softmax_state()
    axes = next(
        item
        for item in standard_ablation_plan(32)
        if item.arm is ExperimentArm.E3_STATE_SEARCH
    )
    candidates = enumerate_cuda_candidates(softmax, axes=axes)
    schedule_threads: list[int] = []
    for item in candidates:
        if item.search_partition != SEARCH_PARTITION_SCHEDULE_SPINE:
            break
        block_threads = int(item.schedule["block_threads"])
        if block_threads not in schedule_threads:
            schedule_threads.append(block_threads)
    approximate_threads = {
        int(item.schedule["block_threads"])
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
        and item.schedule.get("approximate_div") is True
        and item.schedule.get("materialize_shared_row_output") is True
    }
    leader_threads = {
        int(item.schedule["block_threads"])
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
        and item.schedule.get("leader_finalize") is True
        and item.schedule.get("materialize_shared_row_output") is True
        and item.schedule.get("approximate_div") is not True
    }
    combined_threads = {
        int(item.schedule["block_threads"])
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
        and item.schedule.get("leader_finalize") is True
        and item.schedule.get("approximate_div") is True
        and item.schedule.get("materialize_shared_row_output") is True
    }

    assert set(schedule_threads[:3]) <= approximate_threads
    assert set(schedule_threads[:3]) <= leader_threads
    assert set(schedule_threads[:3]) <= combined_threads
    assert all(
        item.schedule.get("materialize_shared_row_output") is not False
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
    )
    assert all(
        item.schedule.get("reduction_parts", 1) == 1
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
    )
    assert len({item.candidate_id for item in candidates}) == len(candidates)


def test_materialization_axis_requires_a_visible_shared_row_output() -> None:
    state = _fresh_fused_add_rms_norm_state()
    axes = next(
        item
        for item in standard_ablation_plan(32)
        if item.arm is ExperimentArm.E3_STATE_SEARCH
    )
    candidates = enumerate_cuda_candidates(state, axes=axes)

    assert any(
        item.schedule.get("materialize_shared_row_output") is False
        for item in candidates
        if item.lowering_path_kind == "row_reduction"
    )


def test_fresh_splitk_search_enumerates_real_row_and_packet_schedules() -> None:
    state = _fresh_splitk_state()
    canonical = canonical_cuda_candidate(state)
    candidates = [
        item
        for item in enumerate_cuda_candidates(state)
        if item.fast_path_id == "cuda.splitk.reduce-varargs.v1"
    ]

    assert canonical.lowering_path_kind == "splitk_weighted_merge"
    assert canonical.pass_argument == (
        "-convert-intent-splitk-reduce-varargs-to-gpu"
    )
    assert {item.schedule["rows_per_block"] for item in candidates} == {
        1,
        2,
        4,
        8,
    }
    assert canonical.candidate_id in {item.candidate_id for item in candidates}
    assert {item.schedule["block_threads"] for item in candidates} == {
        32,
        64,
        128,
        256,
    }
    assert {item.schedule["vector_width"] for item in candidates} == {1, 2, 4}
    assert {item.schedule["broadcast_row_scalars"] for item in candidates} == {
        False,
        True,
    }
    assert {item.schedule["warps_per_row"] for item in candidates} == {
        1,
        2,
        4,
        8,
    }
    assert {item.schedule["d_partitions"] for item in candidates} == {
        1,
        2,
        4,
        8,
    }
    assert all(
        item.schedule["rows_per_block"] == 1
        for item in candidates
        if item.schedule["warps_per_row"] > 1
    )
    assert all(
        item.schedule["rows_per_block"] == 1
        and item.schedule["warps_per_row"] == 1
        for item in candidates
        if item.schedule["d_partitions"] > 1
    )
    assert len(candidates) == 60

    axes = next(
        item
        for item in standard_ablation_plan(96)
        if item.arm is ExperimentArm.E2_SCHEDULE
    )
    schedule_only = enumerate_cuda_candidates(
        state,
        axes=axes,
        fixed_fast_path_id=canonical.fast_path_id,
    )
    assert all(item.schedule["warps_per_row"] == 1 for item in schedule_only)
    assert all(item.schedule["d_partitions"] == 1 for item in schedule_only)


def test_c_fixed_is_mechanical_authored_tile_projection() -> None:
    state = _fresh_softmax_state()
    axes = next(
        item
        for item in standard_ablation_plan(96)
        if item.arm is ExperimentArm.E2_SCHEDULE
    )
    canonical = canonical_cuda_candidate(state)
    assert canonical.fast_path_id is not None
    schedules = enumerate_cuda_candidates(
        state,
        axes=axes,
        fixed_fast_path_id=canonical.fast_path_id,
    )

    selected, projection = project_cuda_c_fixed_candidate(state, schedules)

    assert selected in schedules
    # This concrete N=127 workload collapses larger auto-tile requests to one
    # register state. Of the remaining legal 16/32 coordinates, the mechanical
    # projection chooses 32 as the nearest representation of authored C=128.
    assert selected.schedule["tile_cols"] == 32
    # C does not author CUDA occupancy.  The missing coordinate stays at the
    # fixed compiler default instead of being inferred from the workload shape.
    assert selected.schedule["block_threads"] == 64
    assert projection["workload_or_cost_model_used"] is False
    assert projection["performance_measurement_used"] is False
    assert any(
        item["target_schedule_coordinate"] == "tile_cols"
        for item in projection["consumed_priors"]
    )
    assert any(
        item.schedule["tile_cols"] != selected.schedule["tile_cols"]
        or item.schedule["block_threads"] != selected.schedule["block_threads"]
        for item in schedules
    )


def test_flash_linear_has_real_retile_spine_and_distinct_full_mma_state() -> None:
    state = _stage2_flash_linear_state()
    arms = {item.arm: item for item in standard_ablation_plan(96)}

    path_kind, fast_path_id = schedule_only_cuda_path_identity(state)
    assert path_kind == "cuda.contraction.tiled.v1"
    assert fast_path_id == "cuda.contraction.tiled.v1"
    schedule_only = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_fast_path_id=fast_path_id,
    )
    assert len(schedule_only) == 60
    assert {item.fast_path_id for item in schedule_only} == {
        "cuda.contraction.tiled.v1"
    }
    assert len({item.schedule["tile_rows"] for item in schedule_only}) == 4
    assert len({item.schedule["tile_cols"] for item in schedule_only}) == 3
    assert len({item.schedule["pipeline_stages"] for item in schedule_only}) == 5

    c_fixed, projection = project_cuda_c_fixed_candidate(state, schedule_only)
    assert c_fixed.schedule["tile_rows"] == 32
    assert c_fixed.schedule["tile_cols"] == 32
    assert projection["workload_or_cost_model_used"] is False

    full = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    assert [item.candidate_id for item in full[: len(schedule_only)]] == [
        item.candidate_id for item in schedule_only
    ]
    mma = [
        item
        for item in full
        if item.fast_path_id == "cuda.contraction.mma-sync.v1"
    ]
    assert len(mma) >= 12
    assert all(item.search_partition == SEARCH_PARTITION_EXPANDED for item in mma)
    warp_n = [item for item in mma if item.schedule.get("warps_n", 1) > 1]
    assert {item.schedule["warps_n"] for item in warp_n} >= {2, 4}
    assert all(
        "cuda.contraction.mma-sync.warp-n-partition.v1"
        in item.requested_trace
        for item in warp_n
    )
    assert {item.schedule.get("group_m", 8) for item in mma} >= {2, 8, 16}
    splitk = [
        item
        for item in full
        if item.fast_path_id == "cuda.contraction.mma-sync.v1"
        and "split_k_parts" in item.schedule
    ]
    assert {item.schedule["split_k_parts"] for item in splitk} == {2, 4}
    assert {item.schedule["warps_n"] for item in splitk} == {4}
    assert min(full.index(item) for item in splitk) < 96
    assert all(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        and item.requested_observation_class
        is ObservationClass.OBSERVATION_CHANGING
        and "cuda.contraction.split_k_partial_finalize.v1"
        in item.requested_trace
        for item in splitk
    )
    assert {
        item["target_schedule_coordinate"]
        for item in projection["consumed_priors"]
    } >= {"tile_rows", "tile_cols", "tile_k"}


@pytest.mark.parametrize(
    "name", ("jsd", "xformers_rms_norm", "cross_entropy", "group_norm")
)
def test_stage2_complete_row_consumer_is_the_strong_schedule_spine(
    name: str,
) -> None:
    state = _stage2_state(name)
    arms = {item.arm: item for item in standard_ablation_plan(96)}

    path_kind, fast_path_id = schedule_only_cuda_path_identity(state)
    assert path_kind == "row_reduction"
    assert fast_path_id == ROW_FAST_PATH_ID

    schedule_only = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_fast_path_id=fast_path_id,
    )
    assert schedule_only
    assert {item.fast_path_id for item in schedule_only} == {ROW_FAST_PATH_ID}
    assert all(not item.b_transformations for item in schedule_only)
    assert all(
        item.schedule["compact_reduction_topology"] is False
        and item.schedule["accumulator_chains"] == 1
        and item.schedule["rows_per_block"] == 1
        and item.schedule["reduction_parts"] == 1
        and item.schedule["promote_rank0_scalars"] is False
        and item.schedule["defer_epilogue_col_loads"] is False
        for item in schedule_only
    )

    full = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    full_spine = [
        item
        for item in full
        if item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
    ]
    assert [item.candidate_id for item in full_spine] == [
        item.candidate_id for item in schedule_only
    ]
    assert any(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        and item.fast_path_id == ROW_FAST_PATH_ID
        and (
            item.schedule["compact_reduction_topology"] is True
            or item.schedule["accumulator_chains"] > 1
            or item.schedule["rows_per_block"] > 1
            or item.schedule["reduction_parts"] > 1
        )
        for item in full
    )


def test_stage2_jsd_full_search_reaches_shifted_exp_cancellation() -> None:
    state = _stage2_state("jsd")
    arms = {item.arm: item for item in standard_ablation_plan(96)}

    full = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    shifted = [
        (index, item)
        for index, item in enumerate(full)
        if item.fast_path_id
        == cuda_search.SHIFTED_EXP_CANCELLATION_FAST_PATH_ID
    ]

    assert len(shifted) == 4
    assert shifted[0][0] < 96
    assert all(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        and item.requested_observation_class
        == ObservationClass.OBSERVATION_CHANGING
        and "cancel-shifted-exp=true" in item.pass_argument
        for _index, item in shifted
    )


@pytest.mark.parametrize("name", ("liger_rope", "qwen2vl_mrope"))
def test_stage2_paired_rotation_is_canonical_j_consumption(name: str) -> None:
    state = _stage2_state(name)
    arms = {item.arm: item for item in standard_ablation_plan(96)}

    path_kind, fast_path_id = schedule_only_cuda_path_identity(state)
    assert path_kind == "elementwise"
    assert fast_path_id == ELEMENTWISE_FAST_PATH_ID

    schedule_only = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_fast_path_id=fast_path_id,
    )
    assert schedule_only
    assert {item.fast_path_id for item in schedule_only} == {
        ELEMENTWISE_FAST_PATH_ID
    }

    full = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    full_spine = [
        item
        for item in full
        if item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
    ]
    assert [item.candidate_id for item in full_spine] == [
        item.candidate_id for item in schedule_only
    ]
    expanded_elementwise = [
        item
        for item in full
        if item.search_partition == SEARCH_PARTITION_EXPANDED
        and item.fast_path_id == ELEMENTWISE_FAST_PATH_ID
    ]
    assert expanded_elementwise
    head_reuse = [
        item for item in expanded_elementwise
        if item.schedule["paired_head_reuse"] is True
    ]
    assert head_reuse
    assert {
        item.schedule["paired_head_group_size"] for item in head_reuse
    } == {2, 4, 8, 16}
    assert {item.schedule["vector_width"] for item in head_reuse} == (
        {1, 2} if name == "liger_rope" else {1}
    )
    assert all(
        item.requested_observation_class is ObservationClass.HARD_PRESERVING
        and "cuda.elementwise.paired_head_angle_reuse.request.v1"
        in item.requested_trace
        for item in head_reuse
    )
    direct_plane = [
        item for item in expanded_elementwise
        if item.schedule["paired_direct_plane"] is True
    ]
    assert bool(direct_plane) is (name == "qwen2vl_mrope")
    assert all(
        item.requested_observation_class is ObservationClass.HARD_PRESERVING
        and "cuda.elementwise.paired_direct_plane.request.v1"
        in item.requested_trace
        for item in direct_plane
    )
    assert all(
        item.requested_observation_class
        is ObservationClass.OBSERVATION_CHANGING
        and (
            item.schedule["tanh_approximation"] == "native-ptx"
            or item.b_transformations
        )
        for item in expanded_elementwise
        if item.schedule["paired_head_reuse"] is False
        and item.schedule["paired_direct_plane"] is False
    )


def test_elementwise_c_fixed_preserves_logical_tile_not_thread_count() -> None:
    state = _fresh_dyt_state()
    axes = next(
        item
        for item in standard_ablation_plan(96)
        if item.arm is ExperimentArm.E2_SCHEDULE
    )
    canonical = canonical_cuda_candidate(state)
    assert canonical.fast_path_id is not None
    schedules = enumerate_cuda_candidates(
        state,
        axes=axes,
        fixed_fast_path_id=canonical.fast_path_id,
    )

    selected, projection = project_cuda_c_fixed_candidate(state, schedules)

    assert selected.schedule["logical_tile_elements"] == 128
    assert selected.schedule["block_threads"] == 64
    assert any(
        item["target_schedule_coordinate"] == "logical_tile_elements"
        for item in projection["consumed_priors"]
    )


def test_c_fixed_ignores_splitk_c_without_corresponding_schedule_axis() -> None:
    state = _fresh_splitk_state()
    axes = next(
        item
        for item in standard_ablation_plan(96)
        if item.arm is ExperimentArm.E2_SCHEDULE
    )
    canonical = canonical_cuda_candidate(state)
    assert canonical.fast_path_id is not None
    schedules = enumerate_cuda_candidates(
        state,
        axes=axes,
        fixed_fast_path_id=canonical.fast_path_id,
    )

    selected, projection = project_cuda_c_fixed_candidate(state, schedules)

    assert selected.schedule["rows_per_block"] == 4
    assert selected.schedule["block_threads"] == 128
    assert selected.schedule["vector_width"] == 1
    assert selected.schedule["warps_per_row"] == 1
    assert projection["consumed_priors"] == []
    assert any(
        item["reason"] == "no_corresponding_schedule_coordinate"
        for item in projection["ignored_priors"]
    )


def test_c_schedule_score_consumes_roles_without_changing_states() -> None:
    warps = (
        CPrior("num_warps", "execution_warps", 4, None, None),
    )
    constexpr = (
        CPrior("block_size", "constexpr", 1024, None, None),
    )
    mixed = (
        CPrior("tile", "tile_extent", 32, "n0", "k0"),
        CPrior("reduction", "reduction_tile_extent", 16, "n1", "k1"),
    )

    assert cuda_search._c_score({"num_warps": 4}, warps) == 0.0
    assert cuda_search._c_score({"num_warps": 2}, warps) == 1.0
    assert cuda_search._c_score({"block_threads": 1024}, constexpr) == 0.0
    assert cuda_search._c_score({"block_threads": 512}, constexpr) == 1.0
    assert cuda_search._c_score(
        {"block_threads": 512, "tile_cols": 32},
        (CPrior("reduction", "reduction_tile_extent", 4096, "n1", "k1"),),
    ) == 0.0
    assert cuda_search._c_score(
        {"block_threads": 128, "tile_cols": 32},
        (CPrior("reduction", "reduction_tile_extent", 4096, "n1", "k1"),),
    ) == 2.0
    assert cuda_search._c_score(
        {"block_threads": 512, "tile_cols": 32},
        (CPrior("tile", "tile_extent", 4096, "n1", "k1"),),
    ) == 0.0
    assert cuda_search._c_score({"tile_cols": 32}, mixed) == 0.5
    assert cuda_search._c_score({"tile_rows": 16}, mixed) == 0.5
    assert cuda_search._c_score({"unrelated": 16}, mixed) == float("inf")


def test_contract_view_consumes_b_and_is_independent_of_c() -> None:
    with_c, without_c = _states()

    seeded = derive_cuda_contract_view(with_c)
    unseeded = derive_cuda_contract_view(without_c)

    assert seeded.supported
    assert seeded.consumed
    assert seeded.to_dict() == unseeded.to_dict()


def test_b_relation_can_reject_search_but_c_cannot_authorize_it() -> None:
    with_c, without_c = _states()
    for state in (with_c, without_c):
        state.base.j.boundary.aliases.append(
            AliasSpec("t0", "t1", AliasKind.MAY_ALIAS)
        )
        try:
            with pytest.raises(CudaContractError, match="alias_relation"):
                enumerate_cuda_candidates(state)
        finally:
            state.base.j.boundary.aliases.pop()


def test_unused_b_capability_does_not_create_a_candidate_state() -> None:
    corpus = Path(__file__).parents[2] / "corpus" / "production-jc-v1"
    snapshot, _path = load_production_snapshot(
        corpus, "flash_bwd_preprocess"
    )
    base = materialize_j(snapshot.j, {"x0": 2})
    state = attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )
    view = derive_cuda_contract_view(state)
    full = enumerate_cuda_candidates(state)
    schedule_only = enumerate_cuda_candidates(
        state,
        enable_b_transformations=False,
    )

    assert B_CAP_RANK0_HOST_CAPTURE in view.capabilities
    assert {item.candidate_id for item in schedule_only} == {
        item.candidate_id for item in full
    }
    assert all(not item.b_transformations for item in schedule_only)
    assert all(not item.b_transformations for item in full)
    assert any(
        item.schedule.get("reduction_topology") == "subgroup-feature"
        for item in full
        if item.lowering_path_kind == "padded_row_reduction"
    )
    assert all(
        set(item.required_b_capabilities) <= set(view.capabilities)
        for item in full
    )


def test_padded_row_schedule_names_concrete_reduction_topology() -> None:
    serial = PaddedRowSchedule(64, False, True, "serial-row")
    subgroup = PaddedRowSchedule(128, False, True, "subgroup-feature", 4, 8)

    assert "reduction-topology=serial-row" in serial.pass_argument
    assert "reduction-topology=subgroup-feature" in subgroup.pass_argument
    assert "rows-per-subgroup=4" in subgroup.pass_argument
    assert "lanes-per-row=8" in subgroup.pass_argument
    assert subgroup.to_dict()["reduction_topology"] == "subgroup-feature"
    assert subgroup.to_dict()["rows_per_subgroup"] == 4
    assert subgroup.to_dict()["lanes_per_row"] == 8
    with pytest.raises(ValueError, match="reduction_topology"):
        PaddedRowSchedule(64, False, True, "invented")
    with pytest.raises(ValueError, match="rows_per_subgroup"):
        PaddedRowSchedule(64, False, True, "serial-row", 2)


def test_elementwise_schedule_names_tanh_evaluation_state() -> None:
    stable = ElementwiseSchedule(128, False, 1, True, "stable-ex2")
    native = ElementwiseSchedule(128, False, 1, True, "native-ptx")
    unrolled = ElementwiseSchedule(
        256,
        False,
        1,
        grid_stride_unroll=4,
    )
    tiled = ElementwiseSchedule(
        128,
        False,
        1,
        logical_tile_elements=4096,
    )
    head_reuse = ElementwiseSchedule(
        128,
        False,
        1,
        paired_head_reuse=True,
    )

    assert "tanh-approximation=stable-ex2" in stable.pass_argument
    assert "tanh-approximation=native-ptx" in native.pass_argument
    assert "row-vector-width=1" in stable.pass_argument
    assert "row-vector-width=4" in ElementwiseSchedule(128, False, 4).pass_argument
    assert native.to_dict()["tanh_approximation"] == "native-ptx"
    assert "grid-stride-unroll=4" in unrolled.pass_argument
    assert unrolled.to_dict()["grid_stride_unroll"] == 4
    assert "logical-tile-elements=4096" in tiled.pass_argument
    assert tiled.to_dict()["logical_tile_elements"] == 4096
    assert "paired-head-reuse=true" in head_reuse.pass_argument
    assert head_reuse.to_dict()["paired_head_reuse"] is True
    direct_plane = ElementwiseSchedule(
        128,
        False,
        1,
        paired_direct_plane=True,
    )
    assert "paired-direct-plane=true" in direct_plane.pass_argument
    assert direct_plane.to_dict()["paired_direct_plane"] is True
    with pytest.raises(ValueError, match="tanh_approximation"):
        ElementwiseSchedule(128, False, 1, True, "invented")
    with pytest.raises(ValueError, match="grid_stride_unroll"):
        ElementwiseSchedule(128, False, 1, grid_stride_unroll=3)
    with pytest.raises(ValueError, match="requires unroll=1"):
        ElementwiseSchedule(128, True, 4, grid_stride_unroll=2)
    with pytest.raises(ValueError, match="logical_tile_elements"):
        ElementwiseSchedule(128, False, 1, logical_tile_elements=3000)
    with pytest.raises(ValueError, match="logical tile=0"):
        ElementwiseSchedule(128, True, 4, logical_tile_elements=4096)


def test_row_schedule_names_multi_row_warp_mapping() -> None:
    multi = RowSchedule(
        32,
        32,
        approximate_div=True,
        rows_per_block=2,
    )

    assert "rows-per-block=2" in multi.pass_argument
    assert "approximate-div=true" in multi.pass_argument
    assert multi.to_dict()["approximate_div"] is True
    assert multi.to_dict()["rows_per_block"] == 2
    assert RowSchedule(32, 32, rows_per_block=32).rows_per_block == 32
    with pytest.raises(ValueError, match="32 logical threads"):
        RowSchedule(64, 32, rows_per_block=2)
    with pytest.raises(ValueError, match="rows_per_block"):
        RowSchedule(32, 32, rows_per_block=64)


def test_row_schedule_names_canonical_and_compact_reduction_topologies() -> None:
    canonical = RowSchedule(128, 32, compact_reduction_topology=False)
    compact = RowSchedule(128, 32, compact_reduction_topology=True)

    assert "compact-reduction-topology=false" in canonical.pass_argument
    assert "compact-reduction-topology=true" in compact.pass_argument
    assert canonical.to_dict()["compact_reduction_topology"] is False
    assert compact.to_dict()["compact_reduction_topology"] is True


def test_row_schedule_names_multi_stage_reduction_mapping() -> None:
    multi_stage = RowSchedule(
        256,
        32,
        vector_width=1,
        reduction_parts=16,
    )

    assert "reduction-parts=16" in multi_stage.pass_argument
    assert multi_stage.to_dict()["reduction_parts"] == 16
    with pytest.raises(ValueError, match="power of two"):
        RowSchedule(256, 32, reduction_parts=3)
    with pytest.raises(ValueError, match="cannot also assign"):
        RowSchedule(32, 32, rows_per_block=2, reduction_parts=2)


def test_row_schedule_names_per_thread_multi_accumulator_tree() -> None:
    multi_accumulator = RowSchedule(256, 32, accumulator_chains=4)

    assert "accumulator-chains=4" in multi_accumulator.pass_argument
    assert multi_accumulator.to_dict()["accumulator_chains"] == 4
    with pytest.raises(ValueError, match="accumulator_chains"):
        RowSchedule(256, 32, accumulator_chains=3)
    with pytest.raises(ValueError, match="one row and one stage"):
        RowSchedule(32, 32, rows_per_block=2, accumulator_chains=2)


def test_row_schedule_names_leader_finalization_mapping() -> None:
    leader = RowSchedule(256, 32, leader_finalize=True)

    assert "leader-finalize=true" in leader.pass_argument
    assert leader.to_dict()["leader_finalize"] is True


def test_row_schedule_names_intermediate_materialization_choice() -> None:
    keep_in_ssa = RowSchedule(
        128,
        32,
        materialize_shared_row_output=False,
    )

    assert "materialize-shared-row-output=false" in keep_in_ssa.pass_argument
    assert keep_in_ssa.to_dict()["materialize_shared_row_output"] is False


def test_sparsemax_contract_view_reuses_guard_projection_and_scan_relation() -> None:
    corpus = Path(__file__).parents[2] / "corpus" / "production-jc-v1"
    snapshot, _path = load_production_snapshot(corpus, "sparsemax")
    base = materialize_j(snapshot.j, {"x0": 2, "x1": 127})
    state = attach_c(
        base,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )

    view = derive_cuda_contract_view(state)

    assert view.supported
    assert "reduction:n4" in view.consumed
    assert any(item.startswith("access_guard:a0:") for item in view.consumed)
    assert any(item.startswith("access_guard:a1:") for item in view.consumed)


def test_ablation_axes_expand_concrete_reduction_and_fma_states() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(96)}
    e2 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_lowering_path_kind="row_reduction",
    )
    n1 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.N1_REDUCTION_TREE],
    )
    n2 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.N2_FMA],
    )
    n3 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.N3_PRECISION_APPROX],
    )

    assert e2
    assert {item.lowering_path_kind for item in e2} == {"row_reduction"}
    assert all(not item.generic for item in e2)
    assert all(
        not item.schedule.get("fuse_fma", False) for item in e2
    )
    assert all(
        not item.schedule.get("approximate_div", False) for item in e2
    )
    assert all(
        not item.schedule.get("leader_finalize", False) for item in e2
    )
    assert all(
        item.schedule.get("materialize_shared_row_output", True) is True
        for item in e2
    )
    assert all(
        item.schedule.get("compact_reduction_topology") is False
        for item in e2
    )
    assert all(item.schedule.get("accumulator_chains") == 1 for item in e2)
    assert all(
        item.requested_observation_class.value == "hard_preserving"
        for item in e2
    )
    assert all(not item.b_transformations for item in e2)
    row_vector_requests = {
        item.schedule.get("vector_width")
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    }
    assert 0 in row_vector_requests  # compiler-resolved auto vectorization
    assert 1 in row_vector_requests  # forced-scalar ablation
    assert 4 not in row_vector_requests  # auto already owns the same concrete state
    assert any(
        item.schedule.get("block_threads") == 32
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("rows_per_block", 1) > 1
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("reduction_parts", 1) > 1
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("compact_reduction_topology") is True
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("accumulator_chains", 1) > 1
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert all(
        item.schedule.get("materialize_shared_row_output") is not False
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("fuse_fma") is True
        for item in n2
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("tanh_approximation") == "native-ptx"
        for item in n3
        if item.lowering_path_kind == "elementwise"
    )
    assert not any(
        item.schedule.get("approximate_div") is True
        for item in n3
        if item.lowering_path_kind == "row_reduction"
    )
    record = n2[1].to_dict()
    assert "lowering_path_kind" in record
    assert "family" not in record


def test_expanded_search_preserves_fixed_structure_schedule_spine() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(32)}
    canonical = canonical_cuda_candidate(state)
    assert canonical.fast_path_id is not None
    e2 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_fast_path_id=canonical.fast_path_id,
    )
    e3 = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )

    assert [item.candidate_id for item in e3[: len(e2)]] == [
        item.candidate_id for item in e2
    ]
    assert {item.candidate_id for item in e2} < {
        item.candidate_id for item in e3
    }
    assert all(
        item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
        for item in e3[: len(e2)]
    )
    assert all(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        for item in e3[len(e2) :]
    )


def test_expanded_search_visits_packetized_padded_states_within_first_lanes() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(32)}
    expanded = [
        item
        for item in enumerate_cuda_candidates(
            state,
            axes=arms[ExperimentArm.E3_STATE_SEARCH],
        )
        if item.search_partition == SEARCH_PARTITION_EXPANDED
    ][:32]

    observed = {
        int(item.schedule.get("lanes_per_row", 0))
        for item in expanded
        if item.lowering_path_kind == "padded_row_reduction"
        and item.schedule.get("reduction_topology") == "subgroup-feature"
        and item.schedule.get("vectorize_full_features") is True
    }
    assert {8, 16} <= observed


def test_full_search_covers_padded_multirow_schedule_domain() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(32)}
    full = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    multirow = [
        item
        for item in full
        if item.lowering_path_kind == "padded_row_reduction"
        and item.schedule.get("reduction_topology") == "subgroup-feature"
        and int(item.schedule.get("rows_per_subgroup", 1)) > 1
    ]

    assert {2, 4, 8} == {
        int(item.schedule["rows_per_subgroup"]) for item in multirow
    }
    assert all(
        "cuda.reduction.padded_subgroup_multirow.v1" in item.requested_trace
        for item in multirow
    )


def test_registry_paths_are_enumerated_without_kernel_allowlists() -> None:
    state, _without_c = _states()
    candidates = enumerate_cuda_candidates(state)
    ids = {item.fast_path_id for item in candidates}

    assert "cuda.row_scan.sparsemax.sorted_prefix.v1" in ids
    assert "cuda.gather-scatter.block.v1" in ids
    assert "cuda.layout.block.v1" in ids
    assert all("kernel" not in item.to_dict() for item in candidates)


def test_full_enumerates_multirow_warp_sorted_prefix_as_structural_state() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(32)}
    candidates = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E3_STATE_SEARCH],
    )
    prefix = [
        item
        for item in candidates
        if item.fast_path_id
        == "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        and int(item.schedule.get("rows_per_block", 1)) > 1
        and int(item.schedule.get("prefix_threads", 0)) == 0
        and item.schedule.get("frontier_ballot_support") is not True
        and item.schedule.get("prefetch_epilogue_input_before_summary")
        is not True
    ]

    assert {int(item.schedule["rows_per_block"]) for item in prefix} == {
        2,
        4,
        8,
        16,
    }
    assert all(item.schedule["threads_per_row"] == 32 for item in prefix)
    assert all(item.schedule["vector_width"] == 4 for item in prefix)
    assert all(
        item.requested_trace
        == ("cuda.reduction.sorted_prefix_multirow_warp_early_stop.v1",)
        for item in prefix
    )
    assert all(
        item.requested_observation_class
        is ObservationClass.OBSERVATION_CHANGING
        for item in prefix
    )

    wide = [
        item
        for item in candidates
        if item.fast_path_id
        == "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        and int(item.schedule.get("prefix_threads", 0)) == 32
        and int(item.schedule.get("threads_per_row", 0)) > 32
    ]
    assert {int(item.schedule["threads_per_row"]) for item in wide} >= {
        64,
        128,
        256,
        512,
        1024,
    }
    assert any(
        item.schedule.get("cross_multiply_support_predicate") is True
        for item in wide
    )
    assert any(
        int(item.schedule["rows_per_block"]) == 2
        and int(item.schedule["threads_per_row"]) == 512
        for item in wide
    )

    replicated = [
        item
        for item in candidates
        if item.fast_path_id
        == "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        and item.schedule.get("replicate_prefix_per_warp") is True
    ]
    assert {int(item.schedule["threads_per_row"]) for item in replicated} == {
        64,
        128,
        256,
        512,
        1024,
    }
    assert all(int(item.schedule["prefix_threads"]) == 32 for item in replicated)
    assert all(int(item.schedule["rows_per_block"]) == 1 for item in replicated)
    assert {int(item.schedule["vector_width"]) for item in replicated} == {4}
    assert any(
        item.requested_trace
        == ("cuda.reduction.sorted_prefix_replicated_warp_cross_multiply.v1",)
        for item in replicated
    )

    prefetched = [
        item
        for item in candidates
        if item.fast_path_id
        == "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        and item.schedule.get("prefetch_epilogue_input_before_summary") is True
    ]
    assert prefetched
    assert any(
        item.schedule.get("replicate_prefix_per_warp") is True
        for item in prefetched
    )
    assert any(
        item.schedule.get("replicate_prefix_per_warp") is not True
        for item in prefetched
    )
    assert all(
        "cuda.memory.prefetch_epilogue_input_before_summary.v1"
        in item.requested_trace
        for item in prefetched
    )

    ballot = [
        item
        for item in candidates
        if item.fast_path_id
        == "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        and item.schedule.get("frontier_ballot_support") is True
    ]
    assert ballot
    assert any(
        item.schedule.get("replicate_prefix_per_warp") is True
        and item.schedule.get("prefetch_epilogue_input_before_summary") is True
        for item in ballot
    )
    assert all(
        "frontier-ballot-support=true" in (item.pass_argument or "")
        for item in ballot
    )
    assert all(
        any(trace.endswith("_frontier_ballot.v1") for trace in item.requested_trace)
        for item in ballot
    )


def test_fixed_structure_uses_exact_compiler_path_identity() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(24)}
    fast_path_id = derive_default_fast_path_id(state)
    assert fast_path_id is not None

    candidates = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_fast_path_id=fast_path_id,
    )
    assert candidates
    assert all(not item.generic for item in candidates)
    assert {item.fast_path_id for item in candidates} == {fast_path_id}
    assert canonical_cuda_candidate(state).fast_path_id == fast_path_id


def test_fixed_structure_fails_closed_without_or_with_unknown_identity() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(24)}
    e2 = arms[ExperimentArm.E2_SCHEDULE]

    with pytest.raises(ValueError, match="concrete path identity"):
        enumerate_cuda_candidates(state, axes=e2)
    with pytest.raises(CudaContractError, match="not registered"):
        enumerate_cuda_candidates(
            state,
            axes=e2,
            fixed_fast_path_id="cuda.nonexistent.path.v1",
        )


def test_fixed_generic_path_excludes_all_specialized_states() -> None:
    state, _without_c = _states()
    arms = {item.arm: item for item in standard_ablation_plan(24)}

    candidates = enumerate_cuda_candidates(
        state,
        axes=arms[ExperimentArm.E2_SCHEDULE],
        fixed_lowering_path_kind="generic",
    )

    assert len(candidates) == 1
    assert candidates[0].generic
    assert candidates[0].fast_path_id is None
