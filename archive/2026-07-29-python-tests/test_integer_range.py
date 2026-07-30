from __future__ import annotations

import numpy as np

from intentir.ir.integer_range import (
    IntegerInterval,
    infer_integer_value_intervals,
    proves_index_bounds,
)
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    IndexBoundsSpec,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate2.compare import validate_j_inputs
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


def _internal_index_j(offset: int = 8) -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "internal_index",
            "inputs": [{"name": "t0", "dtype": "f32", "shape": [16]}],
            "outputs": [{"name": "t1", "dtype": "f32", "shape": [8]}],
            "body": [
                {
                    "id": "n0",
                    "op": "iota",
                    "result": "v0",
                    "attrs": {"n": 8, "dtype": "i32"},
                    "type": {"dtype": "i32", "shape": [8]},
                },
                {
                    "id": "n1",
                    "op": "const",
                    "result": "v1",
                    "attrs": {"value": offset, "dtype": "i32", "shape": []},
                    "type": {"dtype": "i32", "shape": []},
                },
                {
                    "id": "n2",
                    "op": "map_binary",
                    "result": "v2",
                    "operands": ["v0", "v1"],
                    "attrs": {"fn": "add"},
                    "type": {"dtype": "i32", "shape": [8]},
                },
                {
                    "id": "n3",
                    "op": "gather",
                    "result": "t1",
                    "operands": ["t0", "v2"],
                    "attrs": {"axis": 0, "mode": "take"},
                    "type": {"dtype": "f32", "shape": [8]},
                },
            ],
        }
    )
    return SemanticArchive(
        kernel_id="internal_index",
        intent=function,
        node_ids={f"n{index}": f"body[{index}]" for index in range(4)},
        boundary=Boundary(
            index_bounds=[IndexBoundsSpec("n3", "t0", "v2", 0)],
        ),
        signature=LogicalSignature(
            [
                SignatureParameter(
                    "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 1, logical_ref="t0",
                ),
                SignatureParameter(
                    "p1", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 1, logical_ref="t1",
                ),
            ]
        ),
    )


def _input_derived_index_j() -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "input_derived_index",
            "inputs": [
                {"name": "t0", "dtype": "f32", "shape": ["x0", "x1"]},
                {"name": "t1", "dtype": "i64", "shape": ["x0"]},
            ],
            "outputs": [{"name": "t2", "dtype": "f32", "shape": ["x0"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "const",
                    "result": "v0",
                    "attrs": {"value": 0, "dtype": "i64", "shape": []},
                    "type": {"dtype": "i64", "shape": []},
                },
                {
                    "id": "n1",
                    "op": "gather",
                    "result": "v1",
                    "operands": ["t1", "v0"],
                    "attrs": {"axis": 0, "mode": "take"},
                    "type": {"dtype": "i64", "shape": []},
                },
                {
                    "id": "n2",
                    "op": "gather",
                    "result": "t2",
                    "operands": ["t0", "v1"],
                    "attrs": {"axis": 1, "mode": "take"},
                    "type": {"dtype": "f32", "shape": ["x0"]},
                },
            ],
        }
    )
    return SemanticArchive(
        kernel_id="input_derived_index",
        intent=function,
        node_ids={f"n{index}": f"body[{index}]" for index in range(3)},
        boundary=Boundary(
            index_bounds=[
                IndexBoundsSpec("n1", "t1", "v0", 0),
                IndexBoundsSpec("n2", "t0", "v1", 1),
            ]
        ),
        signature=LogicalSignature(
            [
                SignatureParameter(
                    "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, logical_ref="t0",
                ),
                SignatureParameter(
                    "p1", ParameterKind.POINTER, ParameterMode.INPUT,
                    "i64", 1, logical_ref="t1",
                ),
                SignatureParameter(
                    "p2", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 1, logical_ref="t2",
                ),
            ]
        ),
    )


def _dynamic_iota_div_j() -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "dynamic_iota_div",
            "inputs": [{"name": "t0", "dtype": "i32", "shape": []}],
            "outputs": [{"name": "t1", "dtype": "i32", "shape": ["x0"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "iota",
                    "result": "v0",
                    "attrs": {"n": "x0", "dtype": "i32"},
                    "type": {"dtype": "i32", "shape": ["x0"]},
                },
                {
                    "id": "n1",
                    "op": "map_binary",
                    "result": "t1",
                    "operands": ["v0", "t0"],
                    "attrs": {"fn": "div"},
                    "type": {"dtype": "i32", "shape": ["x0"]},
                },
            ],
        }
    )
    return SemanticArchive(
        kernel_id="dynamic_iota_div",
        intent=function,
        node_ids={"n0": "body[0]", "n1": "body[1]"},
        boundary=Boundary(),
        signature=LogicalSignature(
            [
                SignatureParameter(
                    "p0", ParameterKind.SCALAR, ParameterMode.VALUE,
                    "i32", 0, logical_ref="t0",
                ),
                SignatureParameter(
                    "p1", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "i32", 1, logical_ref="t1",
                ),
            ]
        ),
    )


def test_integer_range_follows_iota_const_and_typed_add() -> None:
    intervals = infer_integer_value_intervals(_internal_index_j())

    assert intervals["v0"] == IntegerInterval(0, 7)
    assert intervals["v1"] == IntegerInterval(8, 8)
    assert intervals["v2"] == IntegerInterval(8, 15)


def test_internal_index_bounds_are_proved_from_j_dataflow() -> None:
    j = _internal_index_j()
    relation = j.boundary.index_bounds[0]

    assert proves_index_bounds(j, relation)
    validate_j_inputs(
        j,
        {"t0": "input"},
        {"input": np.arange(16, dtype=np.float32)},
    )


def test_integer_range_refuses_finite_width_overflow() -> None:
    j = _internal_index_j(offset=(1 << 31) - 4)
    intervals = infer_integer_value_intervals(j)

    assert "v2" not in intervals
    assert not proves_index_bounds(j, j.boundary.index_bounds[0])


def test_integer_range_uses_dynamic_iota_extent_and_signed_trunc_division() -> None:
    j = _dynamic_iota_div_j()

    intervals = infer_integer_value_intervals(
        j,
        logical_inputs={"t0": np.asarray(3, dtype=np.int32)},
        dimension_extents={"x0": 10},
    )

    assert intervals["v0"] == IntegerInterval(0, 9)
    assert intervals["t1"] == IntegerInterval(0, 3)
    unsafe = infer_integer_value_intervals(
        j,
        logical_inputs={"t0": np.asarray(0, dtype=np.int32)},
        dimension_extents={"x0": 10},
    )
    assert "t1" not in unsafe


def test_input_derived_gather_index_is_proved_without_external_intermediate() -> None:
    j = _input_derived_index_j()
    logits = np.zeros((2, 4), dtype=np.float32)
    labels = np.array([3, 1], dtype=np.int64)
    logical_inputs = {"t0": logits, "t1": labels}
    dimensions = {"x0": 2, "x1": 4}

    intervals = infer_integer_value_intervals(
        j,
        logical_inputs=logical_inputs,
        dimension_extents=dimensions,
    )

    assert intervals["v0"] == IntegerInterval(0, 0)
    assert intervals["v1"] == IntegerInterval(1, 3)
    assert all(
        proves_index_bounds(
            j,
            relation,
            logical_inputs=logical_inputs,
            dimension_extents=dimensions,
        )
        for relation in j.boundary.index_bounds
    )
    validate_j_inputs(
        j,
        {"t0": "logits", "t1": "labels"},
        {"logits": logits, "labels": labels},
    )


def test_input_derived_gather_index_stays_fail_closed_when_range_is_unsafe() -> None:
    j = _input_derived_index_j()
    logits = np.zeros((2, 4), dtype=np.float32)
    labels = np.array([7, 1], dtype=np.int64)

    assert not proves_index_bounds(
        j,
        j.boundary.index_bounds[1],
        logical_inputs={"t0": logits, "t1": labels},
        dimension_extents={"x0": 2, "x1": 4},
    )


def test_refeval_uses_contract_ir_for_closed_internal_index_graph() -> None:
    j = _internal_index_j()
    j.boundary = Boundary(
        domains={"d0": "0 <= k0 and k0 < (8 // 1)"},
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", predicate="k0 < (8 // 1)",
                fill_value=0, node_ref="n3", index_expression="[k0 + 8]",
            ),
            BoundaryAccess(
                "a1", AccessKind.STORE, "t1", predicate="k0 < (8 // 1)",
                node_ref="n3", index_expression="[k0]",
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n3", "e0", "a0"),
            EffectSpec(EffectKind.WRITE, "t1", "n3", "e1", "a1"),
        ],
        index_bounds=[IndexBoundsSpec("n3", "t0", "v2", 0)],
    )
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(16, dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["output"], value[8:])
    assert result.metadata["boundary_profile"] == (
        "canonical_contract_ir_a_graph_v1"
    )
