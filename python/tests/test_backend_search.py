from __future__ import annotations

from pathlib import Path

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.backend.cuda.search import (
    B_CAP_RANK0_HOST_CAPTURE,
    B_TRANSFORM_RANK0_HOST_CAPTURE,
    CudaContractError,
    PerformanceCache,
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
)
from intentir.backend.materialize import attach_c, materialize_j
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

    assert ordered[0].family == "generic"
    assert unseeded[0].family == "generic"
    assert {item.candidate_id for item in ordered} == {
        item.candidate_id for item in unseeded
    }
    assert [item.family for item in ordered[1:4]] == [
        "elementwise",
        "row_reduction",
        "padded_row_reduction",
    ]
    assert all(
        "fuse-fma=false" in item.pass_argument
        for item in ordered
        if item.family == "row_reduction"
    )


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
    assert all(
        set(item.required_b_capabilities) <= set(view.capabilities)
        for item in full
    )


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

    assert all(
        item.lowering_path_kind in {"generic", "row_reduction"}
        for item in e2
    )
    assert all(
        not item.schedule.get("fuse_fma", False) for item in e2
    )
    assert all(not item.b_transformations for item in e2)
    assert any(
        item.schedule.get("block_threads") == 32
        for item in n1
        if item.lowering_path_kind == "row_reduction"
    )
    assert any(
        item.schedule.get("fuse_fma") is True
        for item in n2
        if item.lowering_path_kind == "row_reduction"
    )
    record = n2[1].to_dict()
    assert "lowering_path_kind" in record
    assert "family" not in record
