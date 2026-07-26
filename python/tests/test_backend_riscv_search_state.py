from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import intentir.backend.riscv.compile as rvv
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
    assert fused.transformation.lowering_path_kind == "whole_graph_fused"
    assert not materialized.transformation.b_transformations
    assert fused.transformation.required_b_capabilities == (
        B_CAP_READONLY_INPUTS,
    )
    assert materialized.candidate_id != fused.candidate_id


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


def test_rvv_record_exposes_real_mlir_topology_receipt(monkeypatch) -> None:
    state = SimpleNamespace(
        target=SimpleNamespace(family="riscv"),
        base=SimpleNamespace(
            intent_mlir="module { func.func @kernel() { return } }",
            mlir_digest="1" * 64,
        ),
    )
    receipt = rvv.RiscvTransformationReceipt(
        "2" * 64,
        "3" * 64,
        7,
        3,
    )
    monkeypatch.setattr(
        rvv,
        "_lower_to_llvm_ir",
        lambda state, candidate: ("lowered", "raw llvm", receipt),
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

    assert record["lowering_path_kind"] == "whole_graph_fused"
    assert record["transformation_receipt"]["linalg_ops_before"] == 7
    assert record["transformation_receipt"]["linalg_ops_after"] == 3
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
