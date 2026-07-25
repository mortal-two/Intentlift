from __future__ import annotations

import json

import pytest

from direct_fixtures import make_copy_artifact
from intentir.backend.protocol import BackendTarget, LoweringRequest
from intentir.frontend.canonical_j import emit_j_module
from intentir.ir.model import IntentOp, Tensor
from intentir.ir.canonical_j import (
    CanonicalJError,
    elaborate_j,
    j_elaboration,
    parse_index_expression,
)
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    OutputMapping,
    SemanticArchive,
)
from intentir.verify.gate2.scalar import ScalarLoweringSpec


def _surface_j(index_expression: str) -> SemanticArchive:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = ["x0", "x1"]
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[0].rank = 2
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.intent.body[0].attrs["fn"] = "tanh"
    artifact.j.intent.body[0].type = None
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="i1 < x1",
            fill_value=0.0,
            node_ref="n0",
            index_expression=index_expression,
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t1",
            predicate="i1 < x1",
            node_ref="n0",
            index_expression=index_expression,
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "t1", "n0")
    ]
    return SemanticArchive.from_json(artifact.j.to_json())


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _value_dimension_j() -> SemanticArchive:
    payload = make_copy_artifact().j.to_json()
    payload["intent"]["intent_function"]["inputs"].append(
        {"name": "t2", "dtype": "i32", "shape": []}
    )
    extent_parameter = payload["signature"]["parameters"][2]
    extent_parameter["rank"] = 0
    extent_parameter["logical_ref"] = "t2"
    payload["value_dimension_relations"] = [
        {
            "value_ref": "t2",
            "dimension_ref": "x0",
            "relation": "eq",
        }
    ]
    return SemanticArchive.from_json(payload)


def test_equivalent_index_surface_forms_construct_one_j() -> None:
    comma = elaborate_j(_surface_j("i0,i1"))
    bracketed = elaborate_j(_surface_j(" [ i0, i1 ] "))
    parsed = j_elaboration(comma)

    assert parsed.accesses[0].index.direct_axes == ("i0", "i1")
    assert parsed.accesses[0].index.to_json() == [
        {"op": "ref", "sort": "unknown_ref", "value": "i0"},
        {"op": "ref", "sort": "unknown_ref", "value": "i1"},
    ]
    assert comma.to_json() == bracketed.to_json()
    assert comma.boundary.accesses[0].index_expression == "[i0, i1]"
    assert comma.intent.body[0].type == {
        "dtype": "f32",
        "shape": ["x0", "x1"],
    }
    assert comma.boundary.output_mappings == []


def test_j_elaboration_is_idempotent_and_byte_stable() -> None:
    first = elaborate_j(_surface_j("i0,i1"))
    second = elaborate_j(SemanticArchive.from_json(first.to_json()))

    assert first.to_json() == second.to_json()
    assert _stable_json(first.to_json()) == _stable_json(second.to_json())
    assert elaborate_j(first) is first


def test_affine_contract_is_typed_but_not_misread_as_dense_axes() -> None:
    expression = parse_index_expression("row * stride + col")

    assert expression.direct_axes is None
    assert expression.direct_coordinates is None
    assert expression.to_text() == "[((row * stride) + col)]"


def test_direct_coordinates_keep_integer_literals_distinct_from_axis_refs() -> None:
    expression = parse_index_expression("[row, 0, 2]")

    assert expression.direct_coordinates == ("row", 0, 2)
    assert expression.direct_axes is None
    assert expression.to_text() == "[row, 0, 2]"


@pytest.mark.parametrize(
    "expression",
    ["i0,,i1", "[i0,i1", "obj.attr", "system(i0)", "i0 < i1"],
)
def test_invalid_or_wrong_sort_index_fails_before_consumers(expression: str) -> None:
    with pytest.raises(CanonicalJError):
        elaborate_j(_surface_j(expression))


def test_explicit_result_type_conflict_fails_closed() -> None:
    j = _surface_j("i0,i1")
    j.intent.body[0].type = {"dtype": "f16", "shape": ["x0", "x1"]}

    with pytest.raises(CanonicalJError, match="conflicting types"):
        elaborate_j(j)


def _concat_j(*, operand_shape: list[int], result_shape: list[int], axis: int):
    j = make_copy_artifact().j
    j.intent.inputs[0].shape = list(operand_shape)
    j.intent.inputs.append(Tensor("t2", "f32", list(operand_shape)))
    j.intent.outputs[0].shape = list(result_shape)
    j.intent.body = [
        IntentOp.from_json(
            {
                "id": "n0",
                "op": "concat",
                "operands": ["t0", "t2"],
                "result": "t1",
                "attrs": {"axis": axis},
                "type": {"dtype": "f32", "shape": list(result_shape)},
            }
        )
    ]
    j.node_ids = {"n0": "body[0]"}
    j.boundary.domains = {}
    j.boundary.accesses = []
    j.boundary.effects = []
    j.boundary.output_mappings = []
    return j


def test_concat_axis_must_exist_in_every_operand_rank() -> None:
    j = _concat_j(operand_shape=[2], result_shape=[2, 2], axis=1)

    with pytest.raises(CanonicalJError, match="out of range for rank-1"):
        elaborate_j(j)


def test_concat_static_extent_must_match_declared_result() -> None:
    j = _concat_j(operand_shape=[2, 1], result_shape=[2, 3], axis=1)

    with pytest.raises(CanonicalJError, match="conflicting types"):
        elaborate_j(j)


def test_concat_valid_static_shape_elaborates() -> None:
    j = _concat_j(operand_shape=[2, 1], result_shape=[2, 2], axis=1)

    elaborated = elaborate_j(j)

    assert elaborated.intent.body[0].type == {
        "dtype": "f32",
        "shape": [2, 2],
    }


def test_rank_zero_closed_signature_is_materialized() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = []
    artifact.j.intent.outputs[0].shape = []
    artifact.j.signature.parameters[0].rank = 0
    artifact.j.signature.parameters[1].rank = 0
    artifact.j.signature.parameters = artifact.j.signature.parameters[:2]
    artifact.j.intent.body = [
        IntentOp.from_json(
            {
                "id": "n0",
                "op": "cast",
                "result": "v0",
                "operands": ["t0"],
                "attrs": {"to_dtype": "f32"},
            }
        ),
        IntentOp.from_json(
            {
                "id": "n1",
                "op": "map_unary",
                "result": "t1",
                "operands": ["v0"],
                "attrs": {"fn": "identity"},
            }
        ),
    ]
    artifact.j.node_ids = {"n0": "body[0]", "n1": "body[1]"}

    j = elaborate_j(artifact.j)

    assert j.intent.body[0].type == {"dtype": "f32", "shape": []}
    assert j_elaboration(j).value_type_by_name["v0"].shape == ()


def test_reference_and_production_requests_share_canonical_mlir_front() -> None:
    j = _surface_j("i0,i1")
    reference = ScalarLoweringSpec.create(
        j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    production = LoweringRequest(
        j,
        BackendTarget("cuda", "sm_120a"),
        "generic",
    )

    assert reference.j is production.j
    expected = emit_j_module(reference.j, {"x0": 2, "x1": 4})
    assert expected == emit_j_module(production.j, {"x0": 2, "x1": 4})


def test_canonical_emitter_projects_value_dimension_by_intent_input_order() -> None:
    j = _value_dimension_j()

    mlir = emit_j_module(j, {"x0": 8})

    assert "intent.scalar_equalities = array<i64: 1, 8>" in mlir
    assert mlir == emit_j_module(j, {"x0": 8})
