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

    assert materialized.schedule == fused.schedule
    assert materialized.transformation.lowering_path_kind == "materialized"
    assert fused.transformation.lowering_path_kind == "producer_consumer_fused"
    assert not materialized.transformation.b_transformations
    assert fused.transformation.required_b_capabilities == (
        B_CAP_READONLY_INPUTS,
    )
    assert materialized.candidate_id != fused.candidate_id


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


def test_rvv_record_exposes_real_mlir_topology(monkeypatch) -> None:
    state = SimpleNamespace(
        target=SimpleNamespace(family="riscv"),
        base=SimpleNamespace(
            intent_mlir="module { func.func @kernel() { return } }",
        ),
    )
    transformation = rvv.RiscvTransformationRecord(7, 3)
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
    assert c_fixed.schedule.parallel_tile == 2
    assert c_fixed.schedule.reduction_tile == 8
    assert with_c.schedule_only_c_prior_order[0] == c_fixed.candidate_id
    assert with_c.c_projection["raw_c_value_used_as_target_tile"] is False
    assert with_c.c_projection["projected_priors"][0][
        "schedule_dimension"
    ] == "reduction_tile"
    assert with_c.c_projection["projected_priors"][0]["target_value"] == 8
    record = with_c.to_record()
    assert record["same_universe_for_erase_c_and_c_prior"] is True
    assert record["c_controls_admission"] is False
    assert record["candidate_universe_count"] == 52
