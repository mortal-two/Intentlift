from __future__ import annotations

from types import SimpleNamespace

import pytest

import intentir.backend.riscv.compile as rvv
from intentir.backend.contract_capabilities import B_CAP_READONLY_INPUTS


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
