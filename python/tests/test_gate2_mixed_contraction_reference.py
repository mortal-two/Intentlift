"""Gate2-only J denotation for scalar-CPU-incomplete contractions."""

from __future__ import annotations

import numpy as np
import pytest

from intentir.backend.cpu.structured_reference import (
    STRUCTURED_CPU_REFERENCE_PROTOCOL,
    StructuredReferenceUnsupportedError,
    execute_structured_reference,
    requires_structured_reference,
)
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    Boundary,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


def _archive(
    *,
    inputs: list[dict],
    output: dict,
    body: list[dict],
) -> SemanticArchive:
    parameters = [
        SignatureParameter(
            f"p{index}",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            tensor["dtype"],
            len(tensor["shape"]),
            logical_ref=tensor["name"],
        )
        for index, tensor in enumerate(inputs)
    ]
    parameters.append(
        SignatureParameter(
            f"p{len(parameters)}",
            ParameterKind.POINTER,
            ParameterMode.OUTPUT,
            output["dtype"],
            len(output["shape"]),
            logical_ref=output["name"],
        )
    )
    return SemanticArchive(
        kernel_id="gate2_mixed_contraction_reference",
        intent=IntentFunction.from_json(
            {
                "name": "gate2_mixed_contraction_reference",
                "inputs": inputs,
                "outputs": [output],
                "body": body,
            }
        ),
        node_ids={
            str(operation["id"]): f"body[{index}]"
            for index, operation in enumerate(body)
        },
        boundary=Boundary(),
        signature=LogicalSignature(parameters),
    )


def _direct_contraction_archive(
    *,
    lhs_dtype: str = "f32",
    rhs_dtype: str = "i1",
    result_dtype: str = "f32",
    result_shape: list[int] | None = None,
    subscripts: str = "i,j->ij",
) -> SemanticArchive:
    shape = [2, 3] if result_shape is None else result_shape
    return _archive(
        inputs=[
            {"name": "A", "dtype": lhs_dtype, "shape": [2]},
            {"name": "B", "dtype": rhs_dtype, "shape": [3]},
        ],
        output={"name": "Y", "dtype": result_dtype, "shape": shape},
        body=[
            {
                "id": "n0",
                "op": "contraction",
                "result": "Y",
                "operands": ["A", "B"],
                "attrs": {"subscripts": subscripts},
                "type": {"dtype": result_dtype, "shape": shape},
            }
        ],
    )


def test_mixed_bool_float_outer_product_uses_j_only_reference() -> None:
    archive = _archive(
        inputs=[{"name": "A", "dtype": "f32", "shape": [2]}],
        output={"name": "Y", "dtype": "f32", "shape": [2, 3]},
        body=[
            {
                "id": "n0",
                "op": "iota",
                "result": "v0",
                "operands": [],
                "attrs": {"n": 3, "dtype": "i64"},
                "type": {"dtype": "i64", "shape": [3]},
            },
            {
                "id": "n1",
                "op": "const",
                "result": "v1",
                "operands": [],
                "attrs": {"value": 1, "dtype": "i64", "shape": []},
                "type": {"dtype": "i64", "shape": []},
            },
            {
                "id": "n2",
                "op": "map_binary",
                "result": "v2",
                "operands": ["v0", "v1"],
                "attrs": {"fn": "eq"},
                "type": {"dtype": "i1", "shape": [3]},
            },
            {
                "id": "n3",
                "op": "contraction",
                "result": "Y",
                "operands": ["A", "v2"],
                "attrs": {"subscripts": "i,j->ij"},
                "type": {"dtype": "f32", "shape": [2, 3]},
            },
        ],
    )
    value = np.asarray([2.5, -4.0], dtype=np.float32)
    spec = ScalarLoweringSpec.create(
        archive,
        dims={},
        input_bindings={"A": "input"},
        output_bindings={"Y": "output"},
    )

    assert requires_structured_reference(archive)
    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"],
        value[:, None]
        * np.asarray([False, True, False], dtype=np.float32)[None, :],
    )
    assert result.outputs["output"].dtype == np.dtype(np.float32)
    assert result.metadata["execution_evidence"]["protocol"] == (
        STRUCTURED_CPU_REFERENCE_PROTOCOL
    )
    assert result.metadata["path"] == (
        "canonical J -> typed structured CPU reference"
    )


@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "result_dtype"),
    [
        ("f32", "f32", "f32"),
        ("f16", "f16", "f32"),
        ("i32", "i32", "i32"),
        ("bool", "i1", "i1"),
    ],
)
def test_cpu_lowerable_contraction_stays_on_scalar_path(
    lhs_dtype: str,
    rhs_dtype: str,
    result_dtype: str,
) -> None:
    archive = _direct_contraction_archive(
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        result_dtype=result_dtype,
    )

    assert not requires_structured_reference(archive)


def test_mixed_contraction_rejects_declared_result_shape_mismatch() -> None:
    archive = _direct_contraction_archive(result_shape=[2, 4])

    assert requires_structured_reference(archive)
    with pytest.raises(ValueError, match="declared result shape differs"):
        execute_structured_reference(
            archive,
            {},
            [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([True, False, True], dtype=np.bool_),
            ],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("arity", "exactly two operands"),
        ("subscripts", "closed binary einsum grammar"),
    ],
)
def test_mixed_contraction_rejects_nonbinary_contract(
    mutation: str,
    message: str,
) -> None:
    archive = _direct_contraction_archive()
    operation = archive.intent.body[0]
    if mutation == "arity":
        operation.operands = ["A"]
    else:
        operation.attrs = {"subscripts": "i,...->i"}

    assert requires_structured_reference(archive)
    with pytest.raises(StructuredReferenceUnsupportedError, match=message):
        execute_structured_reference(
            archive,
            {},
            [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([True, False, True], dtype=np.bool_),
            ],
        )


def test_mixed_contraction_rejects_float_to_integer_result() -> None:
    archive = _direct_contraction_archive(result_dtype="i32")

    assert requires_structured_reference(archive)
    with pytest.raises(
        StructuredReferenceUnsupportedError,
        match="integer contraction cannot consume floating operands",
    ):
        execute_structured_reference(
            archive,
            {},
            [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([True, False, True], dtype=np.bool_),
            ],
        )


def test_mixed_contraction_rejects_graph_outside_closed_operation_table() -> None:
    archive = _archive(
        inputs=[
            {"name": "A", "dtype": "f32", "shape": [2]},
            {"name": "B", "dtype": "i1", "shape": [3]},
        ],
        output={"name": "Y", "dtype": "f32", "shape": [2, 3]},
        body=[
            {
                "id": "n0",
                "op": "contraction",
                "result": "v0",
                "operands": ["A", "B"],
                "attrs": {"subscripts": "i,j->ij"},
                "type": {"dtype": "f32", "shape": [2, 3]},
            },
            {
                "id": "n1",
                "op": "flip",
                "result": "Y",
                "operands": ["v0"],
                "attrs": {"axes": [1]},
                "type": {"dtype": "f32", "shape": [2, 3]},
            },
        ],
    )

    assert requires_structured_reference(archive)
    with pytest.raises(
        StructuredReferenceUnsupportedError,
        match="outside its closed operation table: flip",
    ):
        execute_structured_reference(
            archive,
            {},
            [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([True, False, True], dtype=np.bool_),
            ],
        )
