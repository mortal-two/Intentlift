from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from intentir.backend.ablation import ExperimentArm, SearchAxes
from intentir.backend.cuda.production_cases import (
    FORMAL_PRODUCTION_CASE_IDS,
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    constexpr_input_bindings,
    prepare_production_workload,
    profile_production_snapshot,
)
from intentir.backend.cuda.search import (
    enumerate_cuda_candidates,
    schedule_only_cuda_path_identity,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.search import plan_riscv_search
from intentir.ir.paired_rotation import derive_paired_rotation_view
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.scrc_workload_run import build_workload_jc_suite


_ROOT = Path(__file__).resolve().parents[2]
_CATALOG = _ROOT / "corpus" / "production-jc-v4"
_INTENT_OPT = _ROOT / "build" / "tools" / "intent-opt" / "intent-opt"
_MLIR_OPT = Path("/usr/lib/llvm-20/bin/mlir-opt")


def test_v4_formal_set_is_exactly_twenty_four_triton_j() -> None:
    assert len(FORMAL_PRODUCTION_CASE_IDS) == 24
    assert all(
        case_id.startswith("high_performance:")
        for case_id in FORMAL_PRODUCTION_CASE_IDS.values()
    )
    assert set(PRODUCTION_CASE_IDS) - set(FORMAL_PRODUCTION_CASE_IDS) == {
        "masked_copy",
        "masked_row_square_sum",
    }
    for name in FORMAL_PRODUCTION_CASE_IDS:
        snapshot, _path = load_production_snapshot(_CATALOG, name)
        assert snapshot.j is not None


def _profiled_materialization(name: str):
    snapshot, _path = load_production_snapshot(_CATALOG, name)
    assert snapshot.c is not None
    prepared = prepare_production_workload(
        name, profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE
    )
    snapshot, _valuation = profile_production_snapshot(
        snapshot,
        prepared,
        PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    assert snapshot.c is not None
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    materialized = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return snapshot, case, materialized


def _full_axes() -> SearchAxes:
    return SearchAxes(
        arm=ExperimentArm.E3_STATE_SEARCH,
        candidate_budget=256,
        use_c_prior=True,
        search_structural_states=True,
        enable_local_b_transformations=True,
        search_reduction_topology=True,
        search_fma=True,
        search_precision=True,
        search_approximation=True,
    )


def test_llama4_j_derives_interleaved_paired_rotation_view() -> None:
    snapshot, case, materialized = _profiled_materialization("llama4_rope")
    view = derive_paired_rotation_view(
        snapshot.j,
        case.refeval.dims_map,
        dict(materialized.constexpr_bindings),
    )

    assert view is not None
    assert view.layout == "interleaved"
    assert view.data_inputs == ("t0", "t1")
    assert view.cosine_input == "t2"
    assert view.sine_input == "t3"
    assert 'intent.paired_rotation = "interleaved"' in materialized.intent_mlir


@pytest.mark.skipif(
    not _INTENT_OPT.exists() or not _MLIR_OPT.exists(),
    reason="Intent/MLIR tools are not built",
)
def test_llama4_rvv_full_state_is_real_vector_mlir() -> None:
    _snapshot, _case, materialized = _profiled_materialization("llama4_rope")
    linalg = subprocess.run(
        [str(_INTENT_OPT), "-convert-intent-to-linalg"],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    bufferized = subprocess.run(
        [
            str(_MLIR_OPT),
            (
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map"
            ),
            "--buffer-results-to-out-params",
            "--canonicalize",
            "--cse",
        ],
        input=linalg,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-lower-paired-rotation-to-loops=require-lowering=true "
                "vector-width=4 scalable-vector=true"
            ),
            "--canonicalize",
            "--cse",
        ],
        input=bufferized,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "linalg." not in lowered
    assert "memref.collapse_shape" in lowered
    assert "vector.gather" in lowered
    assert "vector.scatter" in lowered
    assert "vector<[4]xf16>" in lowered
    assert "intent.rvv.paired_rotation_lowering_count = 1" in lowered
    assert "intent.rvv.paired_rotation_scalable_count = 1" in lowered


@pytest.mark.skipif(
    not _INTENT_OPT.exists() or not _MLIR_OPT.exists(),
    reason="Intent/MLIR tools are not built",
)
def test_llama4_rvv_fixed_vector_state_uses_contiguous_pair_packets() -> None:
    _snapshot, _case, materialized = _profiled_materialization("llama4_rope")
    linalg = subprocess.run(
        [str(_INTENT_OPT), "-convert-intent-to-linalg"],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    bufferized = subprocess.run(
        [
            str(_MLIR_OPT),
            (
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map"
            ),
            "--buffer-results-to-out-params",
            "--canonicalize",
            "--cse",
        ],
        input=linalg,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-lower-paired-rotation-to-loops=require-lowering=true "
                "vector-width=4 scalable-vector=false"
            ),
            "--canonicalize",
            "--cse",
        ],
        input=bufferized,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert "vector.gather" not in lowered
    assert "vector.scatter" not in lowered
    assert "vector.load" in lowered
    assert "vector.store" in lowered
    assert "vector.shuffle" in lowered
    assert "vector<8xf16>" in lowered


def test_llama4_rvv_complete_consumer_is_the_canonical_schedule_spine() -> None:
    snapshot, _case, materialized = _profiled_materialization("llama4_rope")
    assert snapshot.c is not None
    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("riscv", "rv64gcv_zfh_zvfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    by_id = {item.candidate_id: item for item in plan.universe}
    contiguous = {
        item.candidate_id
        for item in plan.universe
        if item.transformation.paired_rotation
        and not item.transformation.paired_rotation_head_angle_reuse
    }
    head_reuse = {
        item.candidate_id
        for item in plan.universe
        if item.transformation.paired_rotation_head_angle_reuse
    }

    assert contiguous
    assert head_reuse
    assert contiguous == set(plan.schedule_only_c_prior_order)
    assert head_reuse.isdisjoint(plan.schedule_only_c_prior_order)
    assert contiguous | head_reuse == set(plan.c_prior_order)
    assert plan.c_fixed_candidate_id in contiguous
    assert {
        by_id[candidate_id].transformation.lowering_path_kind
        for candidate_id in contiguous | head_reuse
    } == {
        "paired_rotation_contiguous",
        "paired_rotation_head_angle_reuse",
    }


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_structured_j_reuses_flash_mma_lowering() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    assert "intent.loop" in materialized.intent_mlir
    assert "tile_hint = 128" in materialized.intent_mlir
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "direct-v-matrix-load=true direct-v-row-padding=8 "
                "paired-b-matrix-loads=true stream-p-fragments=true "
                "score-fragment-columns=64"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("gpu.launch") == 1
    assert "intent.loop" not in lowered
    assert "nvgpu.mma.sync" in lowered
    assert "nvgpu.ldmatrix" in lowered


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_structured_j_has_hopper_warpgroup_lowering() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("gpu.launch") == 1
    assert "intent.loop" not in lowered
    # QK uses two independently verified m64 fragments for the authored
    # 128-key tile; PV chains two K64 warpgroup contractions in one accumulator.
    assert lowered.count("nvgpu.warpgroup.mma ") == 4
    assert lowered.count("nvgpu.warpgroup.mma.store") == 3


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_structured_j_has_hopper_hybrid_pv_lowering() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("gpu.launch") == 1
    assert "intent.loop" not in lowered
    # QK keeps the two independently verified 64x64 WGMMA fragments.  The
    # probabilities and output recurrence stay in registers and PV consumes
    # natural-layout V through ldmatrix + mma.sync instead of another shared
    # WGMMA round trip.
    assert lowered.count("nvgpu.warpgroup.mma ") == 2
    assert lowered.count("nvgpu.warpgroup.mma.store") == 2
    assert "nvgpu.ldmatrix" in lowered
    assert "nvgpu.mma.sync" in lowered


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_hopper_qk_accumulator_stays_in_registers() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true "
                "hopper-register-qk=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("nvvm.wgmma.mma_async") == 8
    assert "nvgpu.warpgroup.mma.store" not in lowered
    assert "llvm.extractvalue" in lowered
    assert "nvgpu.mma.sync" in lowered


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_hopper_register_qk_has_async_kv_state() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true "
                "hopper-register-qk=true hopper-async-kv=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("nvgpu.device_async_copy") == 16
    assert "nvgpu.device_async_create_group" in lowered
    assert "nvgpu.device_async_wait" in lowered
    assert lowered.count("nvvm.wgmma.mma_async") == 8


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_hopper_normalized_carry_rebuilds_observables() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true "
                "hopper-register-qk=true hopper-async-kv=true "
                "normalized-softmax-carry=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("nvvm.wgmma.mma_async") == 8
    assert lowered.count("nvgpu.device_async_copy") == 16
    assert "arith.divf" in lowered
    assert "math.log" in lowered
    assert "scf.if" in lowered


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_hopper_qk_and_pv_stay_in_registers() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true "
                "hopper-register-qk=true hopper-register-pv=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    # Eight m64n64k16 instructions form QK and another eight form PV per
    # authored key tile. Neither contraction stores its accumulator to shared.
    assert lowered.count("nvvm.wgmma.mma_async") == 16
    assert "nvgpu.warpgroup.mma.store" not in lowered
    assert "nvgpu.mma.sync" not in lowered
    assert "llvm.extractvalue" in lowered


@pytest.mark.skipif(not _INTENT_OPT.exists(), reason="intent-opt is not built")
def test_flash_forward_hopper_register_pv_combines_with_async_kv() -> None:
    _snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    lowered = subprocess.run(
        [
            str(_INTENT_OPT),
            (
                "-convert-intent-flash-mma-to-gpu=num-stages=1 block-m=64 "
                "block-n=128 certified-full-axis-only=true "
                "hopper-warpgroup=true hopper-hybrid-pv=true "
                "hopper-register-qk=true hopper-register-pv=true "
                "hopper-async-kv=true normalized-softmax-carry=true"
            ),
        ],
        input=materialized.intent_mlir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert lowered.count("nvvm.wgmma.mma_async") == 16
    assert lowered.count("nvgpu.device_async_copy") == 16
    assert "nvgpu.mma.sync" not in lowered
    assert "arith.divf" in lowered


def test_flash_forward_direct_and_full_paths_obey_three_column_boundary() -> None:
    snapshot, _case, materialized = _profiled_materialization(
        "flash_attention_forward"
    )
    assert snapshot.c is not None
    state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("cuda", "sm_120a"),
    )
    kind, direct_path = schedule_only_cuda_path_identity(state)
    assert direct_path == "cuda.flash_forward.structured_scan.direct.v1"
    schedule = enumerate_cuda_candidates(
        state,
        fixed_lowering_path_kind=kind,
        fixed_fast_path_id=direct_path,
    )
    full = enumerate_cuda_candidates(state, axes=_full_axes())

    assert schedule
    assert {item.fast_path_id for item in schedule} == {direct_path}
    assert {item.schedule["direct_v_row_padding"] for item in schedule} == {0}
    assert "cuda.flash_forward.certified_full_axis.mma.v1" in {
        item.fast_path_id for item in full
    }
    assert {
        item.schedule["direct_v_row_padding"]
        for item in full
        if item.schedule.get("direct_v_matrix_load")
    } == {0, 8}
    assert {
        item.schedule["stream_p_fragments"]
        for item in full
        if item.fast_path_id == "cuda.flash_forward.certified_full_axis.mma.v1"
    } == {False, True}
    assert {
        item.schedule["tile_cols"]
        for item in full
        if item.fast_path_id == "cuda.flash_forward.certified_full_axis.mma.v1"
    } == {128}
    assert {
        item.schedule["score_fragment_columns"]
        for item in full
        if item.fast_path_id == "cuda.flash_forward.certified_full_axis.mma.v1"
    } == {0, 64}
    assert all(
        item.schedule["stream_p_fragments"]
        for item in full
        if item.fast_path_id == "cuda.flash_forward.certified_full_axis.mma.v1"
        and item.schedule["score_fragment_columns"]
    )
    hopper_state = attach_c(
        materialized,
        snapshot.c,
        BackendTarget("cuda", "sm_90a"),
    )
    hopper_full = enumerate_cuda_candidates(hopper_state, axes=_full_axes())
    hopper_paths = {
        item.fast_path_id for item in hopper_full
    }
    assert "cuda.flash_forward.hopper_warpgroup.v1" in hopper_paths
    assert "cuda.flash_forward.hopper_hybrid_pv.v1" in hopper_paths
    assert "cuda.flash_forward.hopper_register_qk.v1" in hopper_paths
    assert "cuda.flash_forward.hopper_register_qk_async_kv.v1" in hopper_paths
    assert (
        "cuda.flash_forward.hopper_register_qk_async_kv_normalized.v1"
        in hopper_paths
    )
    assert "cuda.flash_forward.hopper_register_qk_pv.v1" in hopper_paths
    assert (
        "cuda.flash_forward.hopper_register_qk_pv_async_kv.v1" in hopper_paths
    )
    assert (
        "cuda.flash_forward.hopper_register_qk_pv_async_kv_normalized.v1"
        in hopper_paths
    )
    assert "cuda.flash_forward.normalized_softmax_carry.mma.v1" in hopper_paths
    assert "cuda.flash_forward.compact_query_warps.mma.v1" in hopper_paths
