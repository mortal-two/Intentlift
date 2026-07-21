"""Typed IntentProgram -> CUDA boundary tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import intentir.backend.cuda.program as cuda_program
from intentir.backend.protocol import (
    BackendProtocolError,
    BackendTarget,
    ComparisonEvidence,
    ComparisonKind,
    LoweringRequest,
)
from intentir.ir import (
    BoundaryContract,
    IntentFunction,
    IntentProgram,
    LogicalArgument,
    LogicalInterface,
)
from intentir.ir.contracts import (
    AccessContract,
    AccessMode,
    BroadcastProjection,
    ConflictKind,
    ContractExpr,
    DestinationBinding,
    EffectContract,
    EffectMode,
    ScopeContract,
    StorageContract,
    WriteCoverage,
)


class _FakeCudaArtifact:
    def __init__(self, source: str) -> None:
        self.source_mlir_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {"source_mlir_sha256": self.source_mlir_sha256, "kernel_count": 1}


def _program(*, conflict: ConflictKind = ConflictKind.UNIQUE_WRITER) -> IntentProgram:
    function = IntentFunction.from_json(
        {
            "name": "scaled",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "id": "n0", "op": "const", "result": "scale",
                    "attrs": {"value": 0.5, "dtype": "f32", "shape": []},
                },
                {
                    "id": "n1", "op": "map_binary", "result": "y",
                    "operands": ["x", "scale"], "attrs": {"fn": "mul"},
                },
            ],
            "returns": [{"output": "out", "value": "y"}],
        }
    )
    boundary = BoundaryContract(
        accesses=(
            AccessContract("a0", AccessMode.READ, "x", node_ref="n1"),
            AccessContract("a1", AccessMode.WRITE, "out", node_ref="n1"),
        ),
        broadcasts=(BroadcastProjection("p0", "n1", "scale", 0, 1, (0,)),),
        destinations=(
            DestinationBinding(
                "d0", "out", "out_ptr", "y", WriteCoverage.FULL,
                conflict, access_ref="a1",
            ),
        ),
        effects=(
            EffectContract("e0", EffectMode.READ, "x", "n1", "a0"),
            EffectContract("e1", EffectMode.WRITE, "out", "n1", "a1"),
        ),
        scopes=(
            ScopeContract(
                "q0", "function", None, (), ("x", "scale", "y"), ("e0", "e1")
            ),
        ),
        storage=(
            StorageContract(
                "b0", ("out",), "q0", "write_once",
                "scope_entry:q0", "scope_exit:q0", True, False,
            ),
        ),
    )
    interface = LogicalInterface(
        "scaled",
        (
            LogicalArgument(
                "x_ptr", "tensor", "input", "f32", rank=1,
                symbolic_shape=("N",), tensor_ref="x",
            ),
            LogicalArgument(
                "out_ptr", "tensor", "output", "f32", rank=1,
                symbolic_shape=("N",), tensor_ref="out",
            ),
        ),
        ("out",),
    )
    return IntentProgram(function, boundary, interface)


def test_typed_cuda_lowering_consumes_broadcast_and_binds_all_digests(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_lower(source, *, target, allow_specialized):
        seen.update(
            source=source, target=target, allow_specialized=allow_specialized
        )
        return _FakeCudaArtifact(source)

    monkeypatch.setattr(cuda_program, "lower_to_cuda", fake_lower)
    program = _program()
    request = LoweringRequest(
        program, cuda_program.cuda_target("sm_120a"), "generic"
    )

    lowered = cuda_program.lower_program_to_cuda(request, {"N": 8})

    assert not lowered.obligations.unsupported
    broadcast = next(
        item for item in lowered.obligations.results
        if item.obligation_id == "p0:projection"
    )
    assert broadcast.status.value == "consumed"
    assert seen["target"].chip == "sm_120a"
    assert seen["allow_specialized"] is False
    assert "intent.map_binary" in seen["source"]
    with pytest.raises(BackendProtocolError, match="comparison evidence"):
        lowered.conformance_record(test_domain={"N": 8})
    record = lowered.conformance_record(
        test_domain={"N": 8},
        comparisons=(
            ComparisonEvidence(
                ComparisonKind.INTENT_TO_LOWERING,
                True,
                "1" * 64,
            ),
        ),
    )
    assert record.j_digest == program.j_digest
    assert record.contract_digest == program.contract_digest
    assert record.program_digest == program.program_digest
    record.assert_conformant()


def test_unknown_conflict_needs_structural_proof_before_codegen(monkeypatch) -> None:
    program = _program(conflict=ConflictKind.UNKNOWN)
    called = False

    def forbidden_lower(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("codegen must not run")

    monkeypatch.setattr(cuda_program, "lower_to_cuda", forbidden_lower)
    request = LoweringRequest(
        program, cuda_program.cuda_target("sm_120a"), "generic"
    )

    with pytest.raises(
        cuda_program.CudaProgramLoweringError,
        match="d0:conflict",
    ):
        cuda_program.lower_program_to_cuda(request, {"N": 8})
    assert called is False


def test_predicated_read_and_fill_are_not_erased_without_domain_proof(
    monkeypatch,
) -> None:
    program = _program()
    accesses = list(program.boundary.accesses)
    accesses[0] = replace(
        accesses[0], predicate=ContractExpr.constant(False), fill_value=0
    )
    program = replace(
        program,
        boundary=replace(program.boundary, accesses=tuple(accesses)),
    )
    called = False

    def forbidden_lower(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("codegen must not run")

    monkeypatch.setattr(cuda_program, "lower_to_cuda", forbidden_lower)
    request = LoweringRequest(
        program, cuda_program.cuda_target("sm_120a"), "generic"
    )

    with pytest.raises(
        cuda_program.CudaProgramLoweringError,
        match="a0:predicate",
    ):
        cuda_program.lower_program_to_cuda(request, {"N": 8})
    assert called is False


def test_logical_return_dtype_mismatch_fails_before_codegen(monkeypatch) -> None:
    program = _program()
    function = IntentFunction.from_json(
        {
            "name": "scaled",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f16", "shape": ["N"]}],
            "body": [
                {
                    "id": "n0", "op": "const", "result": "scale",
                    "attrs": {"value": 0.5, "dtype": "f32", "shape": []},
                },
                {
                    "id": "n1", "op": "map_binary", "result": "y",
                    "operands": ["x", "scale"], "attrs": {"fn": "mul"},
                },
            ],
            "returns": [{"output": "out", "value": "y"}],
        }
    )
    arguments = list(program.interface.arguments)
    arguments[1] = replace(arguments[1], dtype="f16")
    program = replace(
        program,
        function=function,
        interface=replace(program.interface, arguments=tuple(arguments)),
    )
    called = False

    def forbidden_lower(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("codegen must not run")

    monkeypatch.setattr(cuda_program, "lower_to_cuda", forbidden_lower)

    with pytest.raises(ValueError, match="return value 'y' dtype f32"):
        cuda_program.lower_program_to_cuda(
            LoweringRequest(
                program, cuda_program.cuda_target("sm_120a"), "generic"
            ),
            {"N": 8},
        )
    assert called is False


def test_cuda_entry_rejects_another_target_family_before_codegen() -> None:
    request = LoweringRequest(
        _program(), BackendTarget("cpu", "x86_64"), "generic"
    )
    with pytest.raises(
        cuda_program.CudaProgramLoweringError,
        match="cannot lower target family",
    ):
        cuda_program.lower_program_to_cuda(request, {"N": 8})
