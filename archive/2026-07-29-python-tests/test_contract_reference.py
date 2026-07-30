from __future__ import annotations

import numpy as np
import pytest

from intentir.backend.cpu.contract_reference import (
    CONTRACT_REFERENCE_PROTOCOL,
    ContractReferenceUnsupportedError,
    execute_contract_reference,
    project_contract_reference,
)
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate2.scalar import (
    SCALAR_LOWERING_PROTOCOL,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
)


def _split_half_j() -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "split_half",
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["x0"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": ["x0"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "map_unary",
                    "result": "v0",
                    "operands": ["X"],
                    "attrs": {"fn": "identity"},
                    "type": {"dtype": "f32", "shape": ["x0"]},
                },
                {
                    "id": "n1",
                    "op": "map_unary",
                    "result": "v1",
                    "operands": ["X"],
                    "attrs": {"fn": "identity"},
                    "type": {"dtype": "f32", "shape": ["x0"]},
                },
            ],
        }
    )
    accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "X",
            node_ref="n0",
            index_expression="[k0 + (x0 // 2)]",
            predicate="k0 < (x0 // 2)",
            fill_value=0,
        ),
        BoundaryAccess(
            "a1",
            AccessKind.LOAD,
            "X",
            node_ref="n1",
            index_expression="[k0]",
            predicate="k0 < (x0 // 2)",
            fill_value=0,
        ),
        BoundaryAccess(
            "a2",
            AccessKind.STORE,
            "Y",
            node_ref="n0",
            index_expression="[k0]",
            predicate="k0 < (x0 // 2)",
        ),
        BoundaryAccess(
            "a3",
            AccessKind.STORE,
            "Y",
            node_ref="n1",
            index_expression="[k0 + (x0 // 2)]",
            predicate="k0 < (x0 // 2)",
        ),
    ]
    effects = [
        EffectSpec(
            EffectKind.READ if access.kind is AccessKind.LOAD else EffectKind.WRITE,
            access.tensor,
            access.node_ref,
            f"e{index}",
            access.access_id,
        )
        for index, access in enumerate(accesses)
    ]
    return SemanticArchive(
        "split_half",
        function,
        {"n0": "body[0]", "n1": "body[1]"},
        Boundary(
            domains={"d0": "(0 <= k0) and (k0 < x0)"},
            accesses=accesses,
            effects=effects,
        ),
        LogicalSignature(
            [
                SignatureParameter(
                    "p0",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    1,
                    logical_ref="X",
                ),
                SignatureParameter(
                    "p1",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "f32",
                    1,
                    logical_ref="Y",
                ),
                SignatureParameter(
                    "p2",
                    ParameterKind.SCALAR,
                    ParameterMode.VALUE,
                    "i32",
                    logical_ref="x0",
                ),
            ]
        ),
    )


def test_contract_reference_executes_authored_index_predicate_fill_and_stores() -> None:
    j = _split_half_j()
    plan = project_contract_reference(j)

    assert plan.to_json()["protocol"] == CONTRACT_REFERENCE_PROTOCOL
    assert [item.access.access_id for item in plan.loads] == ["a0", "a1"]
    assert [item.access.access_id for item in plan.stores] == ["a2", "a3"]

    value = np.arange(8, dtype=np.float32)
    result = execute_contract_reference(j, {"x0": 8}, [value], plan=plan)

    np.testing.assert_array_equal(
        result.outputs["Y"],
        np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.float32),
    )
    assert result.metadata["authority"] == "canonical_J_contract_ir_only"
    assert result.metadata["indexed_load_count"] == 2
    assert result.metadata["executed_store_element_count"] == 8


def test_scalar_runner_selects_contract_ir_reference_without_source_or_gate() -> None:
    j = _split_half_j()
    spec = ScalarLoweringSpec.create(
        j,
        dims={"x0": 8},
        input_bindings={"X": "input"},
        output_bindings={"Y": "output"},
    )
    value = np.arange(8, dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    assert result.metadata["protocol"] == SCALAR_LOWERING_PROTOCOL
    assert result.metadata["boundary_profile"] == "relation_indexed_pointwise_v1"
    assert result.metadata["path"] == (
        "canonical J -> ContractIR -> indexed CPU reference"
    )
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.float32),
    )


def test_contract_reference_does_not_invent_a_missing_consumer_access() -> None:
    j = _split_half_j()
    j.boundary.accesses = [
        access for access in j.boundary.accesses if access.access_id != "a0"
    ]
    j.boundary.effects = [
        effect for effect in j.boundary.effects if effect.access_ref != "a0"
    ]
    value = np.arange(8, dtype=np.float32)

    result = execute_contract_reference(j, {"x0": 8}, [value])

    # n0 retains ordinary Layer-A identity semantics.  The consumer neither
    # copies a sibling access relation nor guesses the intended shifted slice.
    np.testing.assert_array_equal(
        result.outputs["Y"],
        np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32),
    )
    assert result.metadata["indexed_load_count"] == 1


def test_contract_reference_executes_unpredicated_load_without_inventing_fill() -> None:
    j = _split_half_j()
    for access in j.boundary.accesses[:2]:
        access.index_expression = "[k0]"
        access.predicate = None
        access.fill_value = None

    value = np.arange(8, dtype=np.float32)
    result = execute_contract_reference(j, {"x0": 8}, [value])

    np.testing.assert_array_equal(
        result.outputs["Y"],
        np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32),
    )
    assert result.metadata["indexed_load_count"] == 2


def test_contract_reference_rejects_an_unbound_authored_index_reference() -> None:
    j = _split_half_j()
    j.boundary.accesses[0].index_expression = "[bogus]"

    with pytest.raises(
        ContractReferenceUnsupportedError,
        match="index\\[0\\] has no closed runtime binding",
    ):
        execute_contract_reference(j, {"x0": 8}, [np.arange(8, dtype=np.float32)])


@pytest.mark.parametrize("mutation", ["overlap", "gap"])
def test_contract_reference_rejects_nonexhaustive_store_partition(mutation: str) -> None:
    j = _split_half_j()
    if mutation == "overlap":
        j.boundary.accesses[3].index_expression = "[k0]"
        expected = "overlap"
    else:
        j.boundary.accesses = [
            access for access in j.boundary.accesses if access.access_id != "a3"
        ]
        j.boundary.effects = [
            effect for effect in j.boundary.effects if effect.access_ref != "a3"
        ]
        expected = "do not exhaust"

    with pytest.raises((ValueError, ContractReferenceUnsupportedError), match=expected):
        execute_contract_reference(j, {"x0": 8}, [np.arange(8, dtype=np.float32)])


def test_contract_reference_rejects_nonpointwise_graph_instead_of_erasing_it() -> None:
    j = _split_half_j()
    j.intent.body[0].op = "transpose"
    j.intent.body[0].attrs = {"perm": [0]}

    with pytest.raises(
        ContractReferenceUnsupportedError,
        match="flat pointwise",
    ):
        project_contract_reference(j)
