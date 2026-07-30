from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import intentir.backend.riscv.compile as rvv
import intentir.backend.riscv.search as rvv_search
from intentir.backend.contract_capabilities import B_CAP_READONLY_INPUTS
from intentir.backend.cuda.production_cases import (
    constexpr_input_bindings,
    prepare_production_workload,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.contract import (
    RiscvContractError,
    derive_riscv_contract_view,
)
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.scrc_workload_run import build_workload_jc_suite


_ROOT = Path(__file__).resolve().parents[2]


def _candidate(fused: bool) -> rvv.RiscvLoweringCandidate:
    return rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(fused),
        rvv.RiscvSchedule(3, True, 128),
    )


def test_rvv_transformation_and_schedule_are_separate_search_axes() -> None:
    materialized = _candidate(False)
    fused = _candidate(True)
    reassociated = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(False, True),
        materialized.schedule,
    )

    assert materialized.schedule == fused.schedule
    assert materialized.transformation.lowering_path_kind == "materialized"
    assert fused.transformation.lowering_path_kind == "producer_consumer_fused"
    assert not materialized.transformation.b_transformations
    assert fused.transformation.required_b_capabilities == (
        B_CAP_READONLY_INPUTS,
    )
    assert materialized.candidate_id != fused.candidate_id
    assert reassociated.transformation.required_b_capabilities == ()
    assert "explicit_reduction_reassociation" in " ".join(
        reassociated.transformation.transformation_trace
    )
    assert reassociated.candidate_id != materialized.candidate_id
    explicit = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(False, True, True),
        rvv.RiscvSchedule(3, True, 0, scalable_vectorization=True),
    )
    assert explicit.transformation.lowering_path_kind == "explicit_vector_reduction"
    assert "explicit_row_reduction" in " ".join(
        explicit.transformation.transformation_trace
    )
    assert explicit.schedule.scalable_vectorization
    joint = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(True, True, True, True),
        explicit.schedule,
    )
    assert (
        joint.transformation.lowering_path_kind
        == "producer_consumer_fused_joint_vector_reduction"
    )
    assert "joint_row_statistics" in " ".join(
        joint.transformation.transformation_trace
    )
    assert joint.candidate_id != explicit.candidate_id
    fused_softmax = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(False, True, False, False, True),
        explicit.schedule,
    )
    assert (
        fused_softmax.transformation.lowering_path_kind
        == "fused_softmax_hybrid"
    )
    assert "fused_softmax_hybrid" in " ".join(
        fused_softmax.transformation.transformation_trace
    )
    assert fused_softmax.transformation.required_b_capabilities == (
        B_CAP_READONLY_INPUTS,
    )
    assert fused_softmax.candidate_id != explicit.candidate_id
    with pytest.raises(ValueError, match="requires producer fusion"):
        rvv.RiscvTransformationState(False, True, True, True)
    with pytest.raises(ValueError, match="requires the admitted reduction"):
        rvv.RiscvTransformationState(False, False, False, False, True)


def test_rvv_explicit_reduction_only_enumerates_scalable_schedules() -> None:
    universe = rvv_search.enumerate_riscv_candidates()
    explicit = [
        item for item in universe if item.transformation.explicit_vector_reduction
    ]

    assert explicit
    assert all(item.schedule.scalable_vectorization for item in explicit)
    assert all(item.schedule.vectorize for item in explicit)
    joint = [item for item in explicit if item.transformation.joint_vector_reduction]
    assert len(joint) == 13
    fused_softmax = [item for item in universe if item.transformation.fused_softmax]
    assert len(fused_softmax) == 9
    assert sum(item.schedule.scalable_vectorization for item in fused_softmax) == 1
    assert {
        item.schedule.vector_lanes
        for item in fused_softmax
        if not item.schedule.scalable_vectorization
    } == {2, 4, 8, 16}
    assert all(item.schedule.vectorize for item in fused_softmax)
    assert all(item.schedule.parallel_tile == 0 for item in fused_softmax)
    assert all(item.schedule.reduction_tile == 0 for item in fused_softmax)
    assert len(universe) == 204


def test_rvv_retile_schedule_is_an_explicit_search_axis() -> None:
    untiled = _candidate(False)
    tiled = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(False),
        rvv.RiscvSchedule(3, True, 128, parallel_tile=4, reduction_tile=8),
    )

    assert tiled.schedule.parallel_tile == 4
    assert tiled.schedule.reduction_tile == 8
    assert tiled.candidate_id != untiled.candidate_id
    with pytest.raises(ValueError, match="power of two"):
        rvv.RiscvSchedule(3, True, 128, parallel_tile=3)
    with pytest.raises(ValueError, match="single-thread only"):
        rvv.RiscvSchedule(3, True, 128, thread_count=16)
    scalable = rvv.RiscvSchedule(
        3, True, 0, parallel_tile=16, scalable_vectorization=True
    )
    assert scalable.to_dict()["vector_policy"] == "scalable"
    assert scalable.to_dict()["vector_lanes"] == 4
    with pytest.raises(ValueError, match="vector_lanes"):
        rvv.RiscvSchedule(3, True, 128, vector_lanes=3)
    with pytest.raises(ValueError, match="reserved for scalable"):
        rvv.RiscvSchedule(3, True, 0)
    with pytest.raises(ValueError, match="requires vectorize=true"):
        rvv.RiscvSchedule(3, False, 0, scalable_vectorization=True)


def test_rvv_fusion_fails_before_lowering_without_b_capability(monkeypatch) -> None:
    state = SimpleNamespace(target=SimpleNamespace(family="riscv"))
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("lowering must not run")

    monkeypatch.setattr(rvv, "_lower_to_llvm_ir", forbidden)

    with pytest.raises(ValueError, match="lacks required B capabilities"):
        rvv.compile_riscv_object(state, _candidate(True), capabilities=())
    assert not called


def test_rvv_fusion_request_fails_when_mlir_graph_did_not_change(monkeypatch) -> None:
    state = SimpleNamespace(target=SimpleNamespace(family="riscv"))
    monkeypatch.setattr(
        rvv,
        "_lower_to_llvm_ir",
        lambda state, candidate: (
            "lowered",
            "raw llvm",
            rvv.RiscvTransformationRecord(7, 7, fused_producers=0),
        ),
    )

    with pytest.raises(RuntimeError, match="produced no fused Linalg"):
        rvv.compile_riscv_object(
            state,
            _candidate(True),
            capabilities=(B_CAP_READONLY_INPUTS,),
        )


def test_rvv_joint_request_fails_without_actual_same_root_group(monkeypatch) -> None:
    state = SimpleNamespace(target=SimpleNamespace(family="riscv"))
    candidate = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(True, True, True, True),
        rvv.RiscvSchedule(3, True, 0, scalable_vectorization=True),
    )
    monkeypatch.setattr(
        rvv,
        "_lower_to_llvm_ir",
        lambda state, candidate: (
            "lowered",
            "raw llvm",
            rvv.RiscvTransformationRecord(
                7,
                3,
                fused_producers=1,
                reassociated_float_combines=3,
                explicit_vector_reductions=3,
                joint_vector_reduction_groups=0,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="same-root statistics group"):
        rvv.compile_riscv_object(
            state,
            candidate,
            capabilities=(B_CAP_READONLY_INPUTS,),
        )


def test_rvv_record_exposes_real_mlir_topology(monkeypatch) -> None:
    state = SimpleNamespace(
        target=SimpleNamespace(family="riscv"),
        base=SimpleNamespace(
            intent_mlir="module { func.func @kernel() { return } }",
        ),
    )
    transformation = rvv.RiscvTransformationRecord(7, 3, fused_producers=1)
    monkeypatch.setattr(
        rvv,
        "_lower_to_llvm_ir",
        lambda state, candidate: ("lowered", "raw llvm", transformation),
    )
    monkeypatch.setattr(
        rvv,
        "_run_text",
        lambda command, stdin: "  vsetvli a0, a1, e32\n",
    )
    monkeypatch.setattr(rvv, "_run_bytes", lambda command, stdin: b"\x7fELF-rvv")

    compiled = rvv.compile_riscv_object(
        state,
        _candidate(True),
        capabilities=(B_CAP_READONLY_INPUTS,),
    )
    record = compiled.to_record()

    assert record["lowering_path_kind"] == "producer_consumer_fused"
    assert record["transformation_record"]["linalg_ops_before"] == 7
    assert record["transformation_record"]["linalg_ops_after"] == 3
    assert record["true_mlir_pipeline"] is True
    assert record["python_target_kernel_emission"] is False


def test_rvv_scalable_policy_reaches_llvm_and_must_materialize(monkeypatch) -> None:
    state = SimpleNamespace(
        target=SimpleNamespace(family="riscv"),
        base=SimpleNamespace(
            intent_mlir="module { func.func @kernel() { return } }",
        ),
    )
    candidate = rvv.RiscvLoweringCandidate(
        rvv.RiscvTransformationState(False),
        rvv.RiscvSchedule(3, True, 0, scalable_vectorization=True),
    )
    monkeypatch.setattr(
        rvv,
        "_lower_to_llvm_ir",
        lambda state, candidate: (
            "lowered",
            "raw llvm",
            rvv.RiscvTransformationRecord(1, 1),
        ),
    )
    commands: list[list[str]] = []

    def run_text(command, *, stdin):
        commands.append(command)
        if command[0] == rvv.OPT:
            return "define void @kernel() { %v = insertelement <vscale x 4 x float> poison, float 0.0, i64 0 ret void }"
        return "  vsetvli a0, a1, e32, m2\n  vle32.v v8, (a0)\n"

    monkeypatch.setattr(rvv, "_run_text", run_text)
    monkeypatch.setattr(rvv, "_run_bytes", lambda command, stdin: b"\x7fELF-rvv")

    compiled = rvv.compile_riscv_object(state, candidate, capabilities=())

    assert compiled.scalable_vector_ir_count == 1
    assert compiled.rvv_vsetvl_modes == ("e32,m2",)
    assert any("--riscv-v-vector-bits-min=-1" in item for item in commands[0])
    assert any("--scalable-vectorization=preferred" in item for item in commands[0])


def test_rvv_bufferization_exposes_dense_identity_layout(monkeypatch) -> None:
    commands: list[list[str]] = []

    def record_command(command, *, stdin):
        commands.append(command)
        if command[0] == rvv.MLIR_TRANSLATE:
            return "raw llvm"
        return stdin

    monkeypatch.setattr(rvv, "_run_text", record_command)
    state = SimpleNamespace(
        base=SimpleNamespace(
            intent_mlir="module { func.func @kernel() { return } }",
        ),
    )

    rvv._lower_to_llvm_ir(state, _candidate(False))

    bufferize = next(
        command
        for command in commands
        if any("one-shot-bufferize" in item for item in command)
    )
    assert any(
        "function-boundary-type-conversion=identity-layout-map" in item
        for item in bufferize
    )


def test_rvv_contract_consumes_only_materialized_typed_access_guards() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "sparsemax",
    )
    assert snapshot.c is not None
    prepared = prepare_production_workload("sparsemax")
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
    target = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))

    view = derive_riscv_contract_view(attach_c(materialized, snapshot.c, target))

    assert view.supported
    assert "access_guard:a0:v22:0:positive" in view.consumed
    assert "access_guard:a1:v22:0:positive" in view.consumed

    stripped_projection = replace(
        materialized.boundary_projection,
        consumed=tuple(
            item
            for item in materialized.boundary_projection.consumed
            if not item.startswith("access_guard:")
        ),
    )
    stripped = replace(materialized, boundary_projection=stripped_projection)
    with pytest.raises(
        RiscvContractError,
        match="access_guard_materialization_not_implemented",
    ):
        derive_riscv_contract_view(attach_c(stripped, snapshot.c, target))


def test_rvv_contract_consumes_typed_materialized_internal_storage() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v3",
        "cross_entropy",
    )
    assert snapshot.c is not None
    prepared = prepare_production_workload(
        "cross_entropy", profile="natural-headline-v1"
    )
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
    target = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))

    view = derive_riscv_contract_view(
        attach_c(materialized, snapshot.c, target)
    )

    assert view.supported
    assert "storage:b0" in view.consumed
    assert "storage:b1" in view.consumed

    stripped_logical_projection = replace(
        materialized.boundary_projection.logical_projection,
        consumed=tuple(
            item
            for item in materialized.boundary_projection.consumed
            if item != "storage:b0"
        ),
    )
    stripped_projection = replace(
        materialized.boundary_projection,
        logical_projection=stripped_logical_projection,
    )
    stripped = replace(materialized, boundary_projection=stripped_projection)
    with pytest.raises(
        RiscvContractError,
        match="general_mutable_storage:b0",
    ):
        derive_riscv_contract_view(attach_c(stripped, snapshot.c, target))


def test_unused_abi_inputs_do_not_suppress_readonly_graph_fusion() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "flash_layer_norm_1pass",
    )
    assert snapshot.c is not None
    prepared = prepare_production_workload("flash_layer_norm_1pass")
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
    target = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))

    view = derive_riscv_contract_view(
        attach_c(materialized, snapshot.c, target)
    )

    assert not snapshot.j.boundary.aliases
    assert B_CAP_READONLY_INPUTS in view.capabilities


def test_rvv_c_reorders_but_does_not_change_retile_universe() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "softmax",
    )
    assert snapshot.c is not None
    prepared = prepare_production_workload("softmax")
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
    target = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))
    with_c = rvv_search.plan_riscv_search(
        attach_c(materialized, snapshot.c, target)
    )
    erased = rvv_search.plan_riscv_search(attach_c(materialized, None, target))

    assert {item.candidate_id for item in with_c.universe} == {
        item.candidate_id for item in erased.universe
    }
    assert with_c.c_prior_order != with_c.erase_c_order
    assert erased.c_prior_order == erased.erase_c_order
    by_id = {item.candidate_id: item for item in with_c.universe}
    c_fixed = by_id[with_c.c_fixed_candidate_id]
    assert not c_fixed.transformation.fuse_producers
    assert c_fixed.schedule.thread_count == 1
    assert c_fixed.schedule.parallel_tile == 0
    assert c_fixed.schedule.reduction_tile == 8
    assert with_c.schedule_only_c_prior_order[0] == c_fixed.candidate_id
    c_schedule_t1 = [
        item
        for item in with_c.schedule_only_c_prior_order
        if by_id[item].schedule.thread_count == 1
    ]
    c_anchor_schedules = [
        by_id[item].schedule
        for item in c_schedule_t1[1:4]
    ]
    assert [item.vectorize for item in c_anchor_schedules] == [True, True, False]
    assert [item.scalable_vectorization for item in c_anchor_schedules] == [
        False,
        True,
        False,
    ]
    assert all(item.opt_level == 3 for item in c_anchor_schedules)
    assert all(item.parallel_tile == 0 for item in c_anchor_schedules)
    assert all(item.reduction_tile == 0 for item in c_anchor_schedules)
    c_order_t1 = [
        item
        for item in with_c.c_prior_order
        if by_id[item].schedule.thread_count == 1
    ]
    first_seven = [by_id[item] for item in c_order_t1[:7]]
    assert [
        (
            item.transformation.fuse_producers,
            item.transformation.reassociate_reductions,
            item.transformation.explicit_vector_reduction,
            item.transformation.joint_vector_reduction,
        )
        for item in first_seven
    ] == [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ]
    assert first_seven[0].schedule == c_fixed.schedule
    assert all(
        item.schedule.scalable_vectorization
        == item.transformation.explicit_vector_reduction
        for item in first_seven
    )
    c_anchor_states = [
        by_id[item] for item in c_order_t1[7:14]
    ]
    assert all(item.schedule.vectorize for item in c_anchor_states)
    assert all(item.schedule.parallel_tile == 0 for item in c_anchor_states)
    assert all(item.schedule.reduction_tile == 0 for item in c_anchor_states)
    erase_first_seven = [by_id[item] for item in with_c.erase_c_order[:7]]
    assert {
        (
            item.transformation.fuse_producers,
            item.transformation.reassociate_reductions,
            item.transformation.explicit_vector_reduction,
            item.transformation.joint_vector_reduction,
        )
        for item in erase_first_seven
    } == {
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    }
    assert with_c.c_projection["raw_c_value_used_as_target_tile"] is False
    assert with_c.c_projection["projected_priors"][0][
        "schedule_dimension"
    ] == "reduction_tile"
    assert with_c.c_projection["projected_priors"][0]["target_value"] == 8
    record = with_c.to_record()
    assert record["same_universe_for_erase_c_and_c_prior"] is True
    assert record["c_controls_admission"] is False
    assert record["candidate_universe_count"] == 196
    assert with_c.c_projection["minimum_reduction_tile"] == 4
    assert with_c.c_projection["subvector_reduction_schedules_rejected"] == 8
    assert all(
        item.schedule.reduction_tile == 0
        or item.schedule.reduction_tile >= 4
        for item in with_c.universe
    )
    assert {
        item.schedule.thread_count for item in with_c.universe
    } == {1}


def test_rvv_c_fixed_does_not_activate_an_unmentioned_tile_axis() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v3",
        "tvd",
    )
    assert snapshot.c is not None
    prepared = prepare_production_workload("tvd", profile="natural-headline-v1")
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
    target = BackendTarget("riscv", "rv64gcv_zfh", ("vlen128",))

    plan = rvv_search.plan_riscv_search(
        attach_c(materialized, snapshot.c, target)
    )
    by_id = {item.candidate_id: item for item in plan.universe}
    c_fixed = by_id[plan.c_fixed_candidate_id]

    assert c_fixed.schedule.parallel_tile == 4
    assert c_fixed.schedule.reduction_tile == 0
    assert {
        item["schedule_dimension"]
        for item in plan.c_projection["projected_priors"]
    } == {"parallel_tile"}
