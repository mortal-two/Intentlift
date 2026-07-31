from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from intentir.backend.ablation import ExperimentArm, SearchAxes
from intentir.backend.cuda.search import (
    GATHER_COMPOSED_FAST_PATH_ID,
    GATHER_FAST_PATH_ID,
    ROWMASK_ZERO_FILL_FAST_PATH_ID,
    enumerate_cuda_candidates,
    project_cuda_c_fixed_candidate,
    schedule_only_cuda_path_identity,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.search import plan_riscv_search
from intentir.ir.production_snapshot import load_production_snapshot


_ROOT = Path(__file__).resolve().parents[2]
_CATALOG = _ROOT / "corpus" / "production-jc-v4"
_INTENT_OPT = _ROOT / "build" / "tools" / "intent-opt" / "intent-opt"
_MLIR_OPT = Path("/usr/lib/llvm-20/bin/mlir-opt")


def _state(name: str, dims: dict[str, int]):
    snapshot, _path = load_production_snapshot(_CATALOG, name)
    assert snapshot.c is not None
    materialized = materialize_j(snapshot.j, dims)
    return snapshot, materialized, attach_c(
        materialized,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )


def _axes(arm: ExperimentArm, *, structural: bool) -> SearchAxes:
    return SearchAxes(
        arm=arm,
        candidate_budget=256,
        use_c_prior=True,
        search_structural_states=structural,
        enable_local_b_transformations=structural,
        search_reduction_topology=structural,
        search_fma=structural,
        search_precision=structural,
        search_approximation=structural,
    )


def test_stage3_catalog_contains_both_new_accepted_j() -> None:
    gather, _ = load_production_snapshot(_CATALOG, "gather2d")
    rowmask, _ = load_production_snapshot(_CATALOG, "rowmask_where2d")

    assert [op.op for op in gather.j.intent.body] == ["gather", "gather"]
    assert [op.op for op in rowmask.j.intent.body] == ["cast", "map_binary"]
    assert gather.c is not None and gather.c.to_json()["values"] == {"c0": 128}
    assert rowmask.c is not None and rowmask.c.to_json()["values"] == {"c0": 64}


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_gather_chain_has_materialized_and_composed_real_mlir() -> None:
    _snapshot, materialized, _state_value = _state("gather2d", {"x0": 64})
    outputs = []
    for argument in (
        "-convert-intent-gather-to-gpu=block-threads=128",
        (
            "-convert-intent-gather-to-gpu=block-threads=128 "
            "compose-gather-chain=true"
        ),
    ):
        result = subprocess.run(
            [str(_INTENT_OPT), argument],
            input=materialized.intent_mlir,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        assert "intent.gather" not in result
        assert result.count("gpu.launch") == 1
        outputs.append(result)

    assert "cuda.gather-chain.materialized-row.v1" in outputs[0]
    assert "gpu.dynamic_shared_memory" in outputs[0]
    assert "gpu.barrier" in outputs[0]
    assert "cuda.gather-chain.composed-index.v1" in outputs[1]
    assert "gpu.dynamic_shared_memory" not in outputs[1]
    assert "gpu.barrier" not in outputs[1]


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_gather_chain_has_direct_rvv_indexed_vector_mlir() -> None:
    _snapshot, materialized, _state_value = _state("gather2d", {"x0": 64})
    result = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-lower-gather-chain-to-loops=require-lowering=true "
                "vector-width=4"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "intent.gather" not in result
    assert "vector.gather" in result
    assert "memref<64x64xf32>" in result
    assert "intent.rvv.composed_gather_count = 1 : i64" in result
    assert "intent.rvv.composed_gather_vector_width = 4 : i64" in result


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_rowmask_rank1_broadcast_and_full_zero_fill_lower_to_real_mlir() -> None:
    _snapshot, materialized, _state_value = _state(
        "rowmask_where2d", {"x0": 16}
    )
    canonical = subprocess.run(
        [
            str(_INTENT_OPT),
            "-convert-intent-elementwise-to-gpu=block-threads=64 row-vector-width=1",
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    full = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-elementwise-to-gpu=block-threads=64 "
                "row-mask-zero-fill=true flatten-contiguous=true "
                "flat-vector-width=4"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "intent.contraction" not in canonical
    assert "arith.mulf" in canonical
    assert "memref<16xi1>" in canonical
    assert "cuda.elementwise.rowmask.zero-fill.v1" in full
    assert "scf.if" in full
    assert "vector<4xf32>" in full
    assert "arith.mulf" not in full


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_softmax_approximate_exp_is_a_real_vector_mlir_state() -> None:
    snapshot, _path = load_production_snapshot(_CATALOG, "softmax")
    assert snapshot.c is not None
    materialized = materialize_j(snapshot.j, {"x0": 4, "x1": 16})
    converted = subprocess.run(
        [
            str(_INTENT_OPT),
            "-convert-intent-to-linalg=reassociate-float-reductions=true",
            "-expand-rsqrt-for-libm",
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    approximate = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-fuse-linalg-softmax=vector-width=4 scalable-vector=false "
                "approximate-exp=true require-fusion=true"
            ),
            "-canonicalize",
            "-cse",
        ],
        input=converted,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "math.exp" not in approximate
    assert "math.roundeven" not in approximate
    assert "arith.fptosi" in approximate
    assert "arith.sitofp" in approximate
    assert "vector.reduction" in approximate
    assert "vector<4xf32>" in approximate
    assert "intent.rvv.fused_softmax_approximate_exp_count = 1 : i64" in approximate
    assert "intent.rvv.fused_softmax_vector_reduction_count = 2 : i64" in approximate

    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    candidates = tuple(
        item for item in plan.universe if item.transformation.approximate_exp
    )
    assert candidates
    assert not set(item.candidate_id for item in candidates) & set(
        plan.schedule_only_c_prior_order
    )
    assert set(item.candidate_id for item in candidates) <= set(plan.c_prior_order)


@pytest.mark.skipif(
    not _INTENT_OPT.exists() or not _MLIR_OPT.exists(),
    reason="Intent/MLIR tools are not built",
)
def test_mixed_precision_dot_is_a_real_widening_vector_mlir_state() -> None:
    snapshot, _path = load_production_snapshot(
        _CATALOG, "flash_bwd_preprocess"
    )
    assert snapshot.c is not None
    materialized = materialize_j(
        snapshot.j,
        {"x0": 1, "x1": 32, "x2": 33, "x3": 16, "x4": 64},
    )
    converted = subprocess.run(
        [
            str(_INTENT_OPT),
            "-convert-intent-to-linalg=reassociate-float-reductions=true",
            "-expand-rsqrt-for-libm",
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    fused = subprocess.run(
        [str(_MLIR_OPT), "--linalg-fuse-elementwise-ops", "--canonicalize"],
        input=converted,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    vectorized = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-vectorize-linalg-mixed-precision-dot-reduction="
                "vector-width=4 scalable-vector=true require-vectorization=true"
            ),
            "-canonicalize",
            "-cse",
        ],
        input=fused,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "vector<[4]xf16>" in vectorized
    assert "vector<[4]xf32>" in vectorized
    assert "arith.extf" in vectorized
    assert "vector.reduction <add>" in vectorized
    assert "intent.rvv.mixed_dot_reduction_count = 1 : i64" in vectorized

    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh_zvfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    candidates = tuple(
        item
        for item in plan.universe
        if item.transformation.mixed_precision_dot_reduction
    )
    assert len(candidates) == 3
    assert {
        (item.schedule.scalable_vectorization, item.schedule.vector_lanes)
        for item in candidates
    } == {(False, 16), (True, 4)}
    assert not set(item.candidate_id for item in candidates) & set(
        plan.schedule_only_c_prior_order
    )
    assert set(item.candidate_id for item in candidates) <= set(plan.c_prior_order)


@pytest.mark.parametrize(
    ("name", "dims", "canonical_id", "full_only_id", "c_threads"),
    (
        ("gather2d", {"x0": 64}, GATHER_FAST_PATH_ID,
         GATHER_COMPOSED_FAST_PATH_ID, 128),
        ("rowmask_where2d", {"x0": 16},
         "cuda.elementwise.row.vectorized.v1",
         ROWMASK_ZERO_FILL_FAST_PATH_ID, 64),
    ),
)
def test_new_paths_obey_three_arm_boundary(
    name: str,
    dims: dict[str, int],
    canonical_id: str,
    full_only_id: str,
    c_threads: int,
) -> None:
    _snapshot, _materialized, state = _state(name, dims)
    kind, fast_path_id = schedule_only_cuda_path_identity(state)
    assert fast_path_id == canonical_id

    schedule = enumerate_cuda_candidates(
        state,
        axes=_axes(ExperimentArm.E2_SCHEDULE, structural=False),
        fixed_lowering_path_kind=kind,
        fixed_fast_path_id=fast_path_id,
    )
    full = enumerate_cuda_candidates(
        state,
        axes=_axes(ExperimentArm.E3_STATE_SEARCH, structural=True),
    )

    assert schedule
    assert {item.fast_path_id for item in schedule} == {canonical_id}
    assert full_only_id not in {item.fast_path_id for item in schedule}
    assert full_only_id in {item.fast_path_id for item in full}
    c_fixed, _record = project_cuda_c_fixed_candidate(state, schedule)
    assert c_fixed.schedule["block_threads"] == c_threads


def test_rowmask_rvv_zero_fill_is_full_only_structural_state() -> None:
    snapshot, materialized, _cuda_state = _state(
        "rowmask_where2d", {"x0": 16}
    )
    assert snapshot.c is not None
    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    expanded = tuple(
        item for item in plan.universe if item.transformation.rowmask_zero_fill
    )

    assert expanded
    assert all(
        item.transformation.lowering_path_kind == "rowmask_zero_fill"
        for item in expanded
    )
    assert not set(item.candidate_id for item in expanded) & set(
        plan.schedule_only_c_prior_order
    )
    assert set(item.candidate_id for item in expanded) <= set(plan.c_prior_order)


def test_gather_rvv_composition_is_full_only_structural_state() -> None:
    snapshot, materialized, _cuda_state = _state("gather2d", {"x0": 64})
    assert snapshot.c is not None
    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    expanded = tuple(
        item
        for item in plan.universe
        if item.transformation.composed_indexed_gather
    )

    assert expanded
    assert all(
        item.transformation.lowering_path_kind == "composed_indexed_gather"
        for item in expanded
    )
    assert all(item.schedule.vectorize for item in expanded)
    assert all(not item.schedule.scalable_vectorization for item in expanded)
    assert not set(item.candidate_id for item in expanded) & set(
        plan.schedule_only_c_prior_order
    )
    assert set(item.candidate_id for item in expanded) <= set(plan.c_prior_order)


@pytest.mark.skipif(
    not _INTENT_OPT.exists() or not _MLIR_OPT.exists(),
    reason="Intent/MLIR tools are not built",
)
def test_tvd_sibling_outputs_fuse_into_one_multi_output_linalg() -> None:
    snapshot, _path = load_production_snapshot(_CATALOG, "tvd")
    materialized = materialize_j(snapshot.j, {"x0": 16, "x1": 16})
    converted = subprocess.run(
        [
            str(_INTENT_OPT),
            "-convert-intent-to-linalg=reassociate-float-reductions=false",
            "-expand-rsqrt-for-libm",
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    producer_fused = subprocess.run(
        [str(_MLIR_OPT), "--linalg-fuse-elementwise-ops", "--canonicalize", "--cse"],
        input=converted,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    sibling_fused = subprocess.run(
        [
            str(_INTENT_OPT),
            "-fuse-linalg-sibling-outputs=require-fusion=true",
            "-canonicalize",
            "-cse",
        ],
        input=producer_fused,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert producer_fused.count("linalg.generic") == 2
    assert sibling_fused.count("linalg.generic") == 1
    assert "linalg.yield" in sibling_fused
    assert "intent.rvv.sibling_output_fusion_count = 1 : i64" in sibling_fused
    assert sibling_fused.count("intent.sibling_output_slot") == 2


def test_tvd_rvv_sibling_fusion_is_full_only_structural_state() -> None:
    snapshot, _path = load_production_snapshot(_CATALOG, "tvd")
    assert snapshot.c is not None
    materialized = materialize_j(snapshot.j, {"x0": 16, "x1": 16})
    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    expanded = tuple(
        item for item in plan.universe if item.transformation.fuse_sibling_outputs
    )

    assert expanded
    assert all(item.transformation.fuse_producers for item in expanded)
    assert not set(item.candidate_id for item in expanded) & set(
        plan.schedule_only_c_prior_order
    )
    assert set(item.candidate_id for item in expanded) <= set(plan.c_prior_order)
