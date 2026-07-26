from __future__ import annotations

from pathlib import Path

import pytest

import intentir.backend.cuda.search as cuda_search
from direct_fixtures import make_reduce_artifact
from intentir.backend.cuda.search import (
    B_CAP_RANK0_HOST_CAPTURE,
    B_TRANSFORM_RANK0_HOST_CAPTURE,
    CudaContractError,
    PerformanceCache,
    canonical_cuda_candidate,
    derive_cuda_contract_view,
    derive_default_fast_path_id,
    enumerate_cuda_candidates,
)
from intentir.backend.cuda.schedules import (
    ElementwiseSchedule,
    PaddedRowSchedule,
    RowSchedule,
)
from intentir.backend.materialize import CPrior, attach_c, materialize_j
from intentir.backend.ablation import ExperimentArm, standard_ablation_plan
from intentir.backend.protocol import BackendTarget
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import AliasKind, AliasSpec


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
    assert any("approximate-div=true" in argument for argument in row_arguments)


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
    assert cuda_search._c_score({"tile_cols": 32}, mixed) == 0.5
    assert cuda_search._c_score({"tile_rows": 16}, mixed) == 0.5
    assert cuda_search._c_score({"unrelated": 16}, mixed) == float("inf")


def test_contract_view_consumes_b_and_is_independent_of_c() -> None:
    with_c, without_c = _states()

    seeded = derive_cuda_contract_view(with_c)
    unseeded = derive_cuda_contract_view(without_c)

    assert seeded.supported
    assert seeded.consumed
    assert seeded.digest == unseeded.digest
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


def test_b_conditioned_search_is_a_strict_extension_of_schedule_only() -> None:
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
    assert {item.candidate_id for item in schedule_only} < {
        item.candidate_id for item in full
    }
    assert all(not item.b_transformations for item in schedule_only)
    assert any(
        B_TRANSFORM_RANK0_HOST_CAPTURE in item.b_transformations
        for item in full
    )
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
    subgroup = PaddedRowSchedule(128, False, True, "subgroup-feature")

    assert "reduction-topology=serial-row" in serial.pass_argument
    assert "reduction-topology=subgroup-feature" in subgroup.pass_argument
    assert subgroup.to_dict()["reduction_topology"] == "subgroup-feature"
    with pytest.raises(ValueError, match="reduction_topology"):
        PaddedRowSchedule(64, False, True, "invented")


def test_elementwise_schedule_names_tanh_evaluation_state() -> None:
    stable = ElementwiseSchedule(128, False, 1, True, "stable-ex2")
    native = ElementwiseSchedule(128, False, 1, True, "native-ptx")

    assert "tanh-approximation=stable-ex2" in stable.pass_argument
    assert "tanh-approximation=native-ptx" in native.pass_argument
    assert native.to_dict()["tanh_approximation"] == "native-ptx"
    with pytest.raises(ValueError, match="tanh_approximation"):
        ElementwiseSchedule(128, False, 1, True, "invented")


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
    with pytest.raises(ValueError, match="32 logical threads"):
        RowSchedule(64, 32, rows_per_block=2)


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


def test_performance_cache_identity_does_not_depend_on_c(tmp_path: Path) -> None:
    with_c, without_c = _states()
    candidate_c = enumerate_cuda_candidates(with_c)[1]
    candidate_none = next(
        item
        for item in enumerate_cuda_candidates(without_c)
        if item.candidate_id == candidate_c.candidate_id
    )
    cache = PerformanceCache(tmp_path / "cache.json")
    left = cache.key(
        state=with_c,
        candidate=candidate_c,
        toolchain_digest="a" * 64,
        benchmark_key="shape-2x128",
    )
    right = cache.key(
        state=without_c,
        candidate=candidate_none,
        toolchain_digest="a" * 64,
        benchmark_key="shape-2x128",
    )

    assert left == right
    cache.store(left, 3.25, candidate=candidate_c)
    assert PerformanceCache(tmp_path / "cache.json").lookup(right) == 3.25


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
    assert all(not item.b_transformations for item in e2)
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
        item.schedule.get("fuse_fma") is True
        for item in n2
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("tanh_approximation") == "native-ptx"
        for item in n3
        if item.lowering_path_kind == "elementwise"
    )
    assert any(
        item.schedule.get("approximate_div") is True
        for item in n3
        if item.lowering_path_kind == "row_reduction"
    )
    record = n2[1].to_dict()
    assert "lowering_path_kind" in record
    assert "family" not in record


def test_registry_paths_are_enumerated_without_kernel_allowlists() -> None:
    state, _without_c = _states()
    candidates = enumerate_cuda_candidates(state)
    ids = {item.fast_path_id for item in candidates}

    assert "cuda.row_scan.sparsemax.sorted_prefix.v1" in ids
    assert "cuda.gather-scatter.block.v1" in ids
    assert "cuda.layout.block.v1" in ids
    assert all("kernel" not in item.to_dict() for item in candidates)


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
