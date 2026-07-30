"""Typed J-only Gate2 denotation for structural ``tile`` graphs."""

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
    IndexBoundsSpec,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


def _archive(
    *,
    body: list[dict],
    output_shape: list[int],
    output_dtype: str = "f32",
) -> SemanticArchive:
    return SemanticArchive(
        kernel_id="gate2_tile_reference",
        intent=IntentFunction.from_json(
            {
                "name": "gate2_tile_reference",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [2, 3]}],
                "outputs": [
                    {"name": "Y", "dtype": output_dtype, "shape": output_shape}
                ],
                "body": body,
            }
        ),
        node_ids={
            str(operation["id"]): f"body[{index}]"
            for index, operation in enumerate(body)
        },
        boundary=Boundary(
            index_bounds=[
                IndexBoundsSpec(
                    str(operation["id"]),
                    str(operation["operands"][0]),
                    str(operation["operands"][1]),
                    int(operation["attrs"]["axis"]),
                )
                for operation in body
                if operation["op"] == "gather"
            ]
        ),
        signature=LogicalSignature(
            [
                SignatureParameter(
                    "p0",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    2,
                    logical_ref="X",
                ),
                SignatureParameter(
                    "p1",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    output_dtype,
                    len(output_shape),
                    logical_ref="Y",
                ),
            ]
        ),
    )


def _single_tile_archive(
    *,
    reps: object,
    output_shape: list[int],
    output_dtype: str = "f32",
) -> SemanticArchive:
    return _archive(
        body=[
            {
                "id": "n0",
                "op": "tile",
                "result": "Y",
                "operands": ["X"],
                "attrs": {"reps": reps},
                "type": {"dtype": output_dtype, "shape": output_shape},
            }
        ],
        output_shape=output_shape,
        output_dtype=output_dtype,
    )


def test_gate2_tile_selects_j_only_reference_and_executes_closed_graph() -> None:
    archive = _archive(
        body=[
            {
                "id": "n0",
                "op": "tile",
                "result": "v0",
                "operands": ["X"],
                "attrs": {"reps": [2, 1]},
                "type": {"dtype": "f32", "shape": [4, 3]},
            },
            {
                "id": "n1",
                "op": "concat",
                "result": "v1",
                "operands": ["v0", "v0"],
                "attrs": {"axis": 1},
                "type": {"dtype": "f32", "shape": [4, 6]},
            },
            {
                "id": "n2",
                "op": "const",
                "result": "v2",
                "operands": [],
                "attrs": {"value": 2, "dtype": "i32", "shape": []},
                "type": {"dtype": "i32", "shape": []},
            },
            {
                "id": "n3",
                "op": "gather",
                "result": "Y",
                "operands": ["v1", "v2"],
                "attrs": {"axis": 0, "mode": "take"},
                "type": {"dtype": "f32", "shape": [6]},
            },
        ],
        output_shape=[6],
    )
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    spec = ScalarLoweringSpec.create(
        archive,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "output"},
    )

    assert requires_structured_reference(archive)
    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    tiled = np.tile(value, (2, 1))
    expected = np.take(np.concatenate((tiled, tiled), axis=1), 2, axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.metadata["execution_evidence"]["protocol"] == (
        STRUCTURED_CPU_REFERENCE_PROTOCOL
    )
    assert result.metadata["path"] == "canonical J -> typed structured CPU reference"


@pytest.mark.parametrize(
    ("reps", "output_shape", "message"),
    [
        ([2], [4, 3], "one repetition for every input dimension"),
        ([2, -1], [4, 0], "reps must be non-negative"),
        ([2, True], [4, 3], "reps must be closed integers"),
        ([2, 1], [5, 3], "result shape must equal input shape times reps"),
    ],
)
def test_typed_tile_rejects_invalid_reps_rank_or_shape(
    reps: object,
    output_shape: list[int],
    message: str,
) -> None:
    archive = _single_tile_archive(reps=reps, output_shape=output_shape)

    with pytest.raises(
        (StructuredReferenceUnsupportedError, ValueError), match=message
    ):
        execute_structured_reference(
            archive,
            {},
            [np.arange(6, dtype=np.float32).reshape(2, 3)],
        )


def test_typed_tile_rejects_result_dtype_change() -> None:
    archive = _single_tile_archive(
        reps=[2, 1],
        output_shape=[4, 3],
        output_dtype="f64",
    )

    with pytest.raises(ValueError, match="declares f64 but its semantics infer f32"):
        execute_structured_reference(
            archive,
            {},
            [np.arange(6, dtype=np.float32).reshape(2, 3)],
        )


def test_structured_gather_rejects_out_of_bounds_index() -> None:
    archive = _archive(
        body=[
            {
                "id": "n0",
                "op": "tile",
                "result": "v0",
                "operands": ["X"],
                "attrs": {"reps": [1, 1]},
                "type": {"dtype": "f32", "shape": [2, 3]},
            },
            {
                "id": "n1",
                "op": "const",
                "result": "v1",
                "operands": [],
                "attrs": {"value": -1, "dtype": "i32", "shape": []},
                "type": {"dtype": "i32", "shape": []},
            },
            {
                "id": "n2",
                "op": "gather",
                "result": "Y",
                "operands": ["v0", "v1"],
                "attrs": {"axis": 0, "mode": "take"},
                "type": {"dtype": "f32", "shape": [3]},
            },
        ],
        output_shape=[3],
    )

    with pytest.raises(ValueError, match="index is outside the data extent"):
        execute_structured_reference(
            archive,
            {},
            [np.arange(6, dtype=np.float32).reshape(2, 3)],
        )


def _compressed_along_axis_gather_archive(
    *, index_shape: list[int], output_shape: list[int]
) -> SemanticArchive:
    body = [
        {
            "id": "n0",
            "op": "tile",
            "result": "v0",
            "operands": ["X"],
            "attrs": {"reps": [1, 1]},
            "type": {"dtype": "f32", "shape": [2, 3]},
        },
        {
            "id": "n1",
            "op": "gather",
            "result": "Y",
            "operands": ["v0", "I"],
            "attrs": {"axis": 1, "mode": "along_axis"},
            "type": {"dtype": "f32", "shape": output_shape},
        }
    ]
    return SemanticArchive(
        kernel_id="gate2_compressed_gather",
        intent=IntentFunction.from_json(
            {
                "name": "gate2_compressed_gather",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [2, 3]},
                    {"name": "I", "dtype": "i64", "shape": index_shape},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": output_shape}],
                "body": body,
            }
        ),
        node_ids={"n0": "body[0]", "n1": "body[1]"},
        boundary=Boundary(index_bounds=[IndexBoundsSpec("n1", "v0", "I", 1)]),
        signature=LogicalSignature(
            [
                SignatureParameter(
                    "p0",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    2,
                    logical_ref="X",
                ),
                SignatureParameter(
                    "p1",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "i64",
                    len(index_shape),
                    logical_ref="I",
                ),
                SignatureParameter(
                    "p2",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "f32",
                    len(output_shape),
                    logical_ref="Y",
                ),
            ]
        ),
    )


def test_structured_gather_executes_compressed_along_axis_index() -> None:
    archive = _compressed_along_axis_gather_archive(
        index_shape=[2], output_shape=[2]
    )
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    index = np.asarray([2, 0], dtype=np.int64)

    result = execute_structured_reference(archive, {}, [data, index])

    np.testing.assert_array_equal(
        result.outputs["Y"], np.asarray([2.0, 3.0], dtype=np.float32)
    )


def test_structured_gather_rejects_compressed_non_axis_shape_mismatch() -> None:
    archive = _compressed_along_axis_gather_archive(
        index_shape=[1], output_shape=[1]
    )

    with pytest.raises(ValueError, match="match every non-axis data extent"):
        execute_structured_reference(
            archive,
            {},
            [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.asarray([1], dtype=np.int64),
            ],
        )
