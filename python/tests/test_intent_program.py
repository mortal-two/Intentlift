"""Canonical IntentProgram A/B/ABI/C boundary tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intentir.ir import (
    AccessContract,
    AccessMode,
    BoundaryContract,
    ConflictKind,
    ContractError,
    DestinationBinding,
    IntentFunction,
    IntentProgram,
    IntentProgramError,
    LogicalArgument,
    LogicalInterface,
    ObligationLedger,
    ObligationResult,
    ObligationStatus,
    ScheduleHint,
    WriteCoverage,
)
from intentir.ir.contracts import parse_contract_expression
from intentir.verify.gate1 import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    Frontend,
    Gate1Artifact,
    KernelInterface,
    KernelParameter,
    LocationMap,
    OutputMapping,
    ParameterKind,
    ParameterMode,
    ReductionBoundary,
    ReplayPlan,
    ScheduleSpec,
    SemanticArchive,
)
from intentir.verify.gate1.program_adapter import intent_program_from_gate1


def _function(*, perm: list[int] | None = None) -> IntentFunction:
    op = (
        {
            "id": "n0",
            "op": "transpose",
            "result": "y",
            "operands": ["x"],
            "attrs": {"perm": perm or [0, 1]},
        }
    )
    return IntentFunction.from_json(
        {
            "name": "logical_permutation",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["M", "N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["M", "N"]}],
            "body": [op],
            "returns": [{"output": "out", "value": "y"}],
        }
    )


def _interface() -> LogicalInterface:
    return LogicalInterface(
        "logical_permutation",
        (
            LogicalArgument(
                "x", "tensor", "input", "f32", rank=2,
                symbolic_shape=("M", "N"), tensor_ref="x",
            ),
        ),
        ("out",),
    )


def test_program_roundtrip_is_strict_and_schedule_retuning_preserves_j() -> None:
    first = IntentProgram(
        _function(), BoundaryContract(), _interface(),
        ScheduleHint(values=(("BLOCK", 128),), grid=("M",)),
    )
    restored = IntentProgram.from_json_str(first.to_json_str())
    assert restored.to_dict() == first.to_dict()
    assert restored.j_digest == first.j_digest
    assert restored.contract_digest == first.contract_digest

    retuned = first.with_schedule_hint(
        ScheduleHint(values=(("BLOCK", 256),), grid=("ceil_div(M, 2)",))
    )
    assert retuned.j_digest == first.j_digest
    assert retuned.contract_digest == first.contract_digest
    assert retuned.program_digest != first.program_digest

    malformed = first.to_dict()
    malformed["intent_program"]["gate1_status"] = "replay_closed"
    with pytest.raises(IntentProgramError, match="unknown fields"):
        IntentProgram.from_dict(malformed)


def test_logical_index_and_output_destination_contracts_are_hard_digest_fields() -> None:
    identity = IntentProgram(_function(perm=[0, 1]), BoundaryContract(), _interface())
    transpose = IntentProgram(_function(perm=[1, 0]), BoundaryContract(), _interface())
    assert identity.j_digest != transpose.j_digest

    predicate = parse_contract_expression("row < M")
    boundary = BoundaryContract(
        accesses=(
            AccessContract(
                "a0", AccessMode.WRITE, "out", node_ref="n0", predicate=predicate,
            ),
        ),
        destinations=(
            DestinationBinding(
                "d0", "out", "out_ptr", "y", WriteCoverage.PREDICATED,
                ConflictKind.UNIQUE_WRITER, access_ref="a0", predicate=predicate,
            ),
        ),
    )
    interface = LogicalInterface(
        "logical_permutation",
        (
            *_interface().arguments,
            LogicalArgument(
                "out_ptr", "tensor", "output", "f32", rank=2,
                symbolic_shape=("M", "N"), tensor_ref="out",
            ),
        ),
        ("out",),
    )
    with_destination = IntentProgram(_function(), boundary, interface)
    assert with_destination.function.returns[0].value == "y"
    assert with_destination.boundary.destinations[0].destination == "out_ptr"
    assert with_destination.j_digest != identity.j_digest


def test_predicate_parser_and_obligation_ledger_fail_closed() -> None:
    expression = parse_contract_expression("(row < M) & (col < N)")
    assert expression.symbols == {"row", "M", "col", "N"}
    with pytest.raises(ContractError, match="outside portable expression"):
        parse_contract_expression("helper(row)")

    program = IntentProgram(
        _function(),
        BoundaryContract(
            accesses=(AccessContract("a0", AccessMode.WRITE, "out", node_ref="n0"),)
        ),
        _interface(),
    )
    required = program.obligations
    with pytest.raises(ContractError, match="ledger mismatch"):
        ObligationLedger(program.contract_digest, required, ())
    results = tuple(
        ObligationResult(item.obligation_id, ObligationStatus.UNSUPPORTED)
        for item in required
    )
    ledger = ObligationLedger(program.contract_digest, required, results)
    with pytest.raises(ContractError, match="does not support"):
        ledger.assert_supported()


def _gate_artifact() -> Gate1Artifact:
    function = IntentFunction.from_json(
        {
            "name": "row_sum",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["M", "N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["M"]}],
            "body": [
                {
                    "op": "reduce", "result": "y", "operands": ["x"],
                    "attrs": {"fn": "sum", "axes": [1]},
                }
            ],
        }
    )
    return Gate1Artifact(
        j=SemanticArchive(
            "private_kernel_id",
            function,
            {"n0": "body[0]"},
            Boundary(
                domains={"d0": "row < M"},
                accesses=[
                    BoundaryAccess(
                        "a0", AccessKind.LOAD, "x", "col < N", 0,
                        "n0", "row * N + col",
                    ),
                    BoundaryAccess(
                        "a1", AccessKind.STORE, "out", "row < M", None,
                        "n0", "row",
                    ),
                ],
                output_mappings=[OutputMapping("out", "y", "n0")],
                reductions=[ReductionBoundary("n0", "N", 0, 0)],
            ),
        ),
        interface=KernelInterface(
            "private_kernel_id",
            Frontend.TRITON,
            [
                KernelParameter(
                    "x_ptr", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, tensor_ref="x",
                ),
                KernelParameter(
                    "out_ptr", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 1, tensor_ref="out",
                ),
                KernelParameter("M", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
                KernelParameter("N", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
            ],
            ["M"],
            function_name="source_only_symbol",
        ),
        schedule=ScheduleSpec(),
        plan=ReplayPlan(),
        location_map=LocationMap("private.py", []),
        source="PRIVATE SOURCE MUST NOT CROSS THE HANDOFF",
    )


def test_gate_adapter_drops_privileged_fields_and_keeps_typed_ab_abi() -> None:
    program = intent_program_from_gate1(_gate_artifact())
    payload = program.to_dict()
    serialized = program.to_json_str()

    assert payload["intent_program"]["function"]["body"][0]["id"] == "n0"
    assert payload["intent_program"]["function"]["returns"] == [
        {"output": "out", "value": "y"}
    ]
    assert [item.name for item in program.interface.arguments] == [
        "x_ptr", "out_ptr", "M", "N"
    ]
    assert program.boundary.reductions[0].semantic_axes == (1,)
    assert program.boundary.destinations[0].destination == "out_ptr"
    for forbidden in (
        "PRIVATE SOURCE", "private_kernel_id", "source_only_symbol",
        "replay_closed", "provider", "location_map", "provenance",
    ):
        assert forbidden not in serialized


def test_legacy_op_layer_b_cannot_silently_cross_program_handoff() -> None:
    artifact = _gate_artifact()
    artifact.j.intent.body[0].layer_b = {"mystery_correctness_fact": True}
    with pytest.raises(ValueError, match="unconvertible legacy Layer-B"):
        intent_program_from_gate1(artifact)
