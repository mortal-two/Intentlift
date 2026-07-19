"""Gate2 three-way orchestration and one real add-kernel prototype."""

from __future__ import annotations

import json
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import Boundary, BroadcastSpec, SemanticArchive
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Gate2Suite,
    InputCaseTemplate,
    InputDistribution,
    Lower2CRunner,
    Lower2CSpec,
    TensorInputSpec,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
    boundary_1d_extents,
    compare_outputs,
    generate_input_matrix,
    generate_tensor_input,
)


ROOT = Path(__file__).resolve().parents[2]
HAVE_CPU_LOWERING = (
    (ROOT / "build/tools/intent-opt/intent-opt").exists()
    and Path("/usr/lib/llvm-20/bin/mlir-runner").exists()
)


def test_gate2_integer_and_boolean_outputs_use_exact_comparison() -> None:
    large = np.asarray([2**60], dtype=np.int64)
    adjacent = np.asarray([2**60 + 1], dtype=np.int64)
    integer = compare_outputs(
        "source",
        {"Y": large},
        "lowering",
        {"Y": adjacent},
        ("Y",),
        {"Y": Tolerance(rtol=1.0, atol=1.0)},
    )
    assert not integer.passed
    assert integer.outputs[0].max_abs_error == 1.0

    boolean = compare_outputs(
        "source",
        {"Y": np.asarray([True, False])},
        "lowering",
        {"Y": np.asarray([False, False])},
        ("Y",),
    )
    assert not boolean.passed
    assert boolean.outputs[0].max_abs_error == 1.0


def _intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "identity",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [1]}],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [1]}],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["X", "X"],
                        "attrs": {"fn": "add"},
                    }
                ],
            }
        }
    )


def _eye_archive() -> SemanticArchive:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "eye_boundary_broadcast_gate2",
                "inputs": [],
                "outputs": [
                    {"name": "out", "dtype": "f32", "shape": ["N", "M"]}
                ],
                "body": [
                    {
                        "op": "iota",
                        "result": "idx_i",
                        "attrs": {"n": "N", "dtype": "i32"},
                        "type": {"dtype": "i32", "shape": ["N"]},
                    },
                    {
                        "op": "iota",
                        "result": "idx_j",
                        "attrs": {"n": "M", "dtype": "i32"},
                        "type": {"dtype": "i32", "shape": ["M"]},
                    },
                    {
                        "op": "map_binary",
                        "result": "is_diag",
                        "operands": ["idx_i", "idx_j"],
                        "attrs": {"fn": "eq"},
                        "type": {"dtype": "i1", "shape": ["N", "M"]},
                    },
                    {
                        "op": "const",
                        "result": "one",
                        "attrs": {"value": 1.0, "dtype": "f32", "shape": []},
                        "type": {"dtype": "f32", "shape": []},
                    },
                    {
                        "op": "const",
                        "result": "zero",
                        "attrs": {"value": 0.0, "dtype": "f32", "shape": []},
                        "type": {"dtype": "f32", "shape": []},
                    },
                    {
                        "op": "select",
                        "result": "out",
                        "operands": ["is_diag", "one", "zero"],
                        "type": {"dtype": "f32", "shape": ["N", "M"]},
                    },
                ],
            }
        }
    )
    return SemanticArchive(
        kernel_id="eye_boundary_broadcast_gate2",
        intent=intent,
        node_ids={
            "n0": "body[0]",
            "n1": "body[1]",
            "n2": "body[2]",
            "n3": "body[3]",
            "n4": "body[4]",
            "n5": "body[5]",
        },
        boundary=Boundary(
            broadcasts=[
                BroadcastSpec("n2", "idx_i", [1]),
                BroadcastSpec("n2", "idx_j", [0]),
                BroadcastSpec("n5", "one", [0, 1]),
                BroadcastSpec("n5", "zero", [0, 1]),
            ]
        ),
    )


def _outer_sub_archive() -> SemanticArchive:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "outer_sub_boundary_broadcast_gate2",
                "inputs": [
                    {"name": "lhs", "dtype": "f32", "shape": ["N"]},
                    {"name": "rhs", "dtype": "f32", "shape": ["M"]},
                ],
                "outputs": [
                    {"name": "out", "dtype": "f32", "shape": ["N", "M"]}
                ],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "out",
                        "operands": ["lhs", "rhs"],
                        "attrs": {"fn": "sub"},
                        "type": {"dtype": "f32", "shape": ["N", "M"]},
                    }
                ],
            }
        }
    )
    return SemanticArchive(
        kernel_id="outer_sub_boundary_broadcast_gate2",
        intent=intent,
        node_ids={"n0": "body[0]"},
        boundary=Boundary(
            broadcasts=[
                BroadcastSpec("n0", "lhs", [1]),
                BroadcastSpec("n0", "rhs", [0]),
            ]
        ),
    )


class _FakeTritonRunner:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, leg, program, spec, values):
        return ExecutionResult(leg, True, {"Y": self.output})


class _FakeLowerRunner:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, spec, values):
        return ExecutionResult("lowering", True, {"Y": self.output})


class _FailedLowerRunner:
    def __init__(self, *, unsupported: bool) -> None:
        self.unsupported = unsupported

    def run(self, spec, values):
        return ExecutionResult(
            "lowering",
            False,
            error="synthetic lowering failure",
            unsupported=self.unsupported,
        )


def _case() -> Gate2Case:
    return Gate2Case(
        "classification",
        {"X": np.array([0.0], dtype=np.float32)},
        TritonExecutionSpec(
            (TritonArgument("out", ArgumentRole.OUTPUT, dtype="f32", shape=(1,), output_key="Y"),),
            (1,),
        ),
        Lower2CSpec(_intent(), {"N": 1}, {"X": "X"}, "Y"),
        ("Y",),
    )


@pytest.mark.parametrize(
    "source,replay,lowering,status",
    [
        (0.0, 0.0, 0.0, Gate2Status.PASS),
        (0.0, 0.0, 1.0, Gate2Status.LOWERING_DIVERGED),
        (0.0, 1.0, 0.0, Gate2Status.REPLAY_DIVERGED),
        (1.0, 0.0, 0.0, Gate2Status.LIFT_DIVERGED),
        (0.0, 1.0, 2.0, Gate2Status.ALL_DIVERGED),
    ],
)
def test_gate2_mechanical_attribution(source, replay, lowering, status) -> None:
    array = lambda value: np.array([value], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner(array(source)),
        replay_runner=_FakeTritonRunner(array(replay)),
        lowering_runner=_FakeLowerRunner(array(lowering)),
    )
    dummy = KernelSource.from_code("", "kernel")
    result = runner.run(dummy, dummy, _case())
    assert result.status is status


@pytest.mark.parametrize(
    "unsupported,status",
    [
        (True, Gate2Status.LOWERING_UNSUPPORTED),
        (False, Gate2Status.EXECUTION_FAILED),
    ],
)
def test_gate2_retains_available_pairwise_evidence_when_lowering_fails(
    unsupported, status
) -> None:
    output = np.array([3.0], dtype=np.float32)
    runner = Gate2Runner(
        source_runner=_FakeTritonRunner(output),
        replay_runner=_FakeTritonRunner(output.copy()),
        lowering_runner=_FailedLowerRunner(unsupported=unsupported),
    )
    dummy = KernelSource.from_code("", "kernel")

    result = runner.run(dummy, dummy, _case())

    assert result.status is status
    assert result.source_replay is not None
    assert result.source_replay.passed
    assert result.source_lowering is None
    assert result.replay_lowering is None


class _EchoTritonRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, leg, program, spec, values):
        self.calls.append(str(values["case_name"]))
        return ExecutionResult(
            leg, True, {"Y": np.asarray(values["X"], dtype=np.float32)}
        )


class _ConditionalLowerRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, spec, values):
        self.calls.append(str(values["case_name"]))
        output = np.asarray(values["X"], dtype=np.float32) + float(
            values["lower_delta"]
        )
        return ExecutionResult("lowering", True, {"Y": output})


def test_gate2_suite_runs_every_case_and_aggregates_strictly() -> None:
    first = _case()
    first = Gate2Case(
        "matrix.pass",
        {"X": np.array([2.0], dtype=np.float32), "lower_delta": 0.0,
         "case_name": "pass"},
        first.triton,
        first.lowering,
        first.output_keys,
        first.tolerances,
        {"shape": "block_minus_one", "seed": 7},
    )
    second = Gate2Case(
        "matrix.lowering_fail",
        {"X": np.array([2.0], dtype=np.float32), "lower_delta": 1.0,
         "case_name": "fail"},
        first.triton,
        first.lowering,
        first.output_keys,
        first.tolerances,
        {"shape": "block", "seed": 11},
    )
    source_runner = _EchoTritonRunner()
    replay_runner = _EchoTritonRunner()
    lowering_runner = _ConditionalLowerRunner()
    runner = Gate2Runner(
        source_runner=source_runner,
        replay_runner=replay_runner,
        lowering_runner=lowering_runner,
    )
    dummy = KernelSource.from_code("", "kernel")

    result = runner.run_suite(
        dummy, dummy, Gate2Suite("strict.matrix", (first, second))
    )

    assert not result.passed
    assert result.failed_case_ids == ("matrix.lowering_fail",)
    assert result.status_counts == {
        "gate2_pass": 1,
        "lowering_diverged": 1,
    }
    assert [case.metadata for case in result.cases] == [
        {"shape": "block_minus_one", "seed": 7},
        {"shape": "block", "seed": 11},
    ]
    assert source_runner.calls == ["pass", "fail"]
    assert replay_runner.calls == ["pass", "fail"]
    assert lowering_runner.calls == ["pass", "fail"]


def test_gate2_suite_rejects_duplicate_case_ids() -> None:
    case = _case()
    with pytest.raises(ValueError, match="case_id values must be unique"):
        Gate2Suite("duplicates", (case, case))


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_uses_real_intent_lowering_not_legacy_interpreter() -> None:
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/add/intent.json").read_text())
    )
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    result = Lower2CRunner().run(
        Lower2CSpec(fn, {"M": 2, "N": 3}, {"A": "A", "B": "B"}, "Y"),
        {"A": a, "B": b},
    )
    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["Y"], a + b)
    assert "mlir-runner" in result.metadata["path"]
    assert result.metadata["protocol"] == (
        "intent_linalg_cpu_symbolic_iota_integer_readback_"
        "boundary_broadcast_compressed_gather_v5"
    )


@pytest.mark.parametrize(("n", "m"), ((1, 1), (4, 4), (3, 5), (5, 3)))
@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_materializes_boundary_broadcast_axes_for_eye(n, m) -> None:
    archive = _eye_archive()
    result = Lower2CRunner().run(
        Lower2CSpec(
            archive.intent,
            {"N": n, "M": m},
            {},
            "out",
            semantic_archive=archive,
        ),
        {},
    )

    assert result.ok, result.error
    expected = np.equal(
        np.arange(n, dtype=np.int32)[:, None],
        np.arange(m, dtype=np.int32)[None, :],
    ).astype(np.float32)
    assert result.outputs["out"].shape == (n, m)
    np.testing.assert_array_equal(result.outputs["out"], expected)
    assert result.metadata["boundary_broadcast_consumer_count"] == 2


@pytest.mark.parametrize(("n", "m"), ((4, 4), (3, 5)))
@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_boundary_broadcast_preserves_operand_axis_direction(n, m) -> None:
    archive = _outer_sub_archive()
    lhs = (np.arange(n, dtype=np.float32) + 1.0) * 100.0
    rhs = np.arange(m, dtype=np.float32) + 1.0
    result = Lower2CRunner().run(
        Lower2CSpec(
            archive.intent,
            {"N": n, "M": m},
            {"lhs": "lhs", "rhs": "rhs"},
            "out",
            semantic_archive=archive,
        ),
        {"lhs": lhs, "rhs": rhs},
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["out"], lhs[:, None] - rhs[None, :]
    )


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_does_not_guess_outer_broadcast_without_boundary_facts() -> None:
    archive = _eye_archive()
    legacy = Lower2CRunner().run(
        Lower2CSpec(archive.intent, {"N": 3, "M": 5}, {}, "out"),
        {},
    )

    assert not legacy.ok


@pytest.mark.parametrize(
    "broadcasts",
    (
        [
            BroadcastSpec("n0", "lhs", [2]),
            BroadcastSpec("n0", "rhs", [0]),
        ],
        [
            BroadcastSpec("n0", "lhs", ["axis"]),
            BroadcastSpec("n0", "rhs", [0]),
        ],
        [
            BroadcastSpec("n0", "lhs", [1]),
            BroadcastSpec("n0", "lhs", [0]),
        ],
        [
            BroadcastSpec("n0", "lhs", [1]),
            BroadcastSpec("n0", "lhs", [1]),
        ],
        [
            BroadcastSpec("n0", "lhs", [0]),
            BroadcastSpec("n0", "rhs", [0]),
        ],
    ),
    ids=[
        "axis-out-of-range",
        "symbolic-axis",
        "conflicting-axes",
        "duplicate-axes",
        "incompatible-projection",
    ],
)
@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_boundary_broadcast_invalid_contracts_fail_closed(
    broadcasts,
) -> None:
    base = _outer_sub_archive()
    archive = replace(base, boundary=Boundary(broadcasts=broadcasts))
    result = Lower2CRunner().run(
        Lower2CSpec(
            archive.intent,
            {"N": 3, "M": 5},
            {"lhs": "lhs", "rhs": "rhs"},
            "out",
            semantic_archive=archive,
        ),
        {
            "lhs": np.arange(3, dtype=np.float32),
            "rhs": np.arange(5, dtype=np.float32),
        },
    )

    assert not result.ok
    assert result.unsupported


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_executes_embedding_take_gather() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "embedding_take_gate2",
                "inputs": [
                    {"name": "W", "dtype": "f32", "shape": [5, 3]},
                    {"name": "I", "dtype": "i64", "shape": [4]},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 3]}
                ],
                "body": [
                    {
                        "op": "gather",
                        "result": "Y",
                        "operands": ["W", "I"],
                        "attrs": {"axis": 0, "mode": "take"},
                    }
                ],
            }
        }
    )
    weight = np.arange(15, dtype=np.float32).reshape(5, 3)
    indices = np.array([4, 0, 2, 4], dtype=np.int64)

    result = Lower2CRunner().run(
        Lower2CSpec(fn, {}, {"W": "W", "I": "I"}, "Y"),
        {"W": weight, "I": indices},
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["Y"], weight[indices])


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_executes_compressed_along_axis_gather(axis) -> None:
    data_shape = (4, 3) if axis == 0 else (3, 5)
    index_shape = (data_shape[1],) if axis == 0 else (data_shape[0],)
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": f"compressed_along_axis_{axis}_gate2",
                "inputs": [
                    {"name": "D", "dtype": "f32", "shape": list(data_shape)},
                    {"name": "I", "dtype": "i64", "shape": list(index_shape)},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": list(index_shape)}
                ],
                "body": [
                    {
                        "op": "gather",
                        "result": "Y",
                        "operands": ["D", "I"],
                        "attrs": {"axis": axis, "mode": "along_axis"},
                    }
                ],
            }
        }
    )
    data = np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape)
    indices = (
        np.asarray([3, 1, 0], dtype=np.int64)
        if axis == 0
        else np.asarray([4, 2, 0], dtype=np.int64)
    )
    expected = (
        data[indices, np.arange(data_shape[1])]
        if axis == 0
        else data[np.arange(data_shape[0]), indices]
    )

    result = Lower2CRunner().run(
        Lower2CSpec(fn, {}, {"D": "D", "I": "I"}, "Y"),
        {"D": data, "I": indices},
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["Y"], expected)


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_compressed_along_axis_rank0_fails_closed() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "compressed_along_axis_rank0_gate2",
                "inputs": [
                    {"name": "D", "dtype": "f32", "shape": [5]},
                    {"name": "I", "dtype": "i64", "shape": []},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
                "body": [
                    {
                        "op": "gather",
                        "result": "Y",
                        "operands": ["D", "I"],
                        "attrs": {"axis": 0, "mode": "along_axis"},
                    }
                ],
            }
        }
    )
    result = Lower2CRunner().run(
        Lower2CSpec(fn, {}, {"D": "D", "I": "I"}, "Y"),
        {
            "D": np.arange(5, dtype=np.float32),
            "I": np.asarray(2, dtype=np.int64),
        },
    )

    assert not result.ok
    assert result.unsupported


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_supports_no_input_symbolic_const_output() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "ones_gate2",
                "inputs": [],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": ["N"]}
                ],
                "body": [
                    {
                        "op": "const",
                        "result": "Y",
                        "attrs": {
                            "value": 1.0,
                            "dtype": "f32",
                            "shape": ["N"],
                        },
                    }
                ],
            }
        }
    )
    result = Lower2CRunner().run(
        Lower2CSpec(fn, {"N": 1003}, {}, "Y"), {}
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["Y"], np.ones(1003, dtype=np.float32)
    )


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (
        ("i8", np.int8),
        ("i16", np.int16),
        ("i32", np.int32),
        ("i64", np.int64),
    ),
)
@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_executes_iota_and_reads_back_integer_output(
    dtype, numpy_dtype
) -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": f"iota_{dtype}_gate2",
                "inputs": [],
                "outputs": [{"name": "Y", "dtype": dtype, "shape": [7]}],
                "body": [
                    {
                        "op": "iota",
                        "result": "Y",
                        "attrs": {"n": 7, "dtype": dtype},
                    }
                ],
            }
        }
    )

    result = Lower2CRunner().run(Lower2CSpec(fn, {}, {}, "Y"), {})

    assert result.ok, result.error
    assert result.outputs["Y"].dtype == np.dtype(numpy_dtype)
    np.testing.assert_array_equal(
        result.outputs["Y"], np.arange(7, dtype=numpy_dtype)
    )


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_reads_back_boolean_output_without_signed_i1_corruption() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "iota_eq_gate2",
                "inputs": [],
                "outputs": [{"name": "Y", "dtype": "i1", "shape": [7]}],
                "body": [
                    {
                        "op": "iota",
                        "result": "I",
                        "attrs": {"n": 7, "dtype": "i32"},
                    },
                    {
                        "op": "const",
                        "result": "Three",
                        "attrs": {"value": 3, "dtype": "i32", "shape": []},
                    },
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["I", "Three"],
                        "attrs": {"fn": "eq"},
                    },
                ],
            }
        }
    )

    result = Lower2CRunner().run(Lower2CSpec(fn, {}, {}, "Y"), {})

    assert result.ok, result.error
    assert result.outputs["Y"].dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(
        result.outputs["Y"],
        np.asarray([False, False, False, True, False, False, False]),
    )


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_integer_readback_preserves_signed_i64_range() -> None:
    expected = np.asarray([-(2**40), -1, 2**40], dtype=np.int64)
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "signed_i64_gate2",
                "inputs": [],
                "outputs": [{"name": "Y", "dtype": "i64", "shape": [3]}],
                "body": [
                    {
                        "op": "const",
                        "result": "Y",
                        "attrs": {
                            "value": expected.tolist(),
                            "dtype": "i64",
                            "shape": [3],
                        },
                    }
                ],
            }
        }
    )

    result = Lower2CRunner().run(Lower2CSpec(fn, {}, {}, "Y"), {})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["Y"], expected)


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_executes_affine_arange_composed_from_iota() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "affine_arange_gate2",
                "inputs": [],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": ["N"]}],
                "body": [
                    {
                        "op": "iota",
                        "result": "I",
                        "attrs": {"n": "N", "dtype": "i32"},
                    },
                    {
                        "op": "cast",
                        "result": "F",
                        "operands": ["I"],
                        "attrs": {"to_dtype": "f32"},
                    },
                    {
                        "op": "const",
                        "result": "Step",
                        "attrs": {"value": 0.125, "dtype": "f32", "shape": []},
                    },
                    {
                        "op": "map_binary",
                        "result": "Scaled",
                        "operands": ["F", "Step"],
                        "attrs": {"fn": "mul"},
                    },
                    {
                        "op": "const",
                        "result": "Start",
                        "attrs": {"value": -3.25, "dtype": "f32", "shape": []},
                    },
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["Scaled", "Start"],
                        "attrs": {"fn": "add"},
                    },
                ],
            }
        }
    )

    result = Lower2CRunner().run(
        Lower2CSpec(fn, {"N": 17}, {}, "Y"), {}
    )

    assert result.ok, result.error
    expected = -3.25 + 0.125 * np.arange(17, dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["Y"], expected)


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_casts_rank_zero_i32_semantic_scalar_to_f32() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "partial_mean_gate2",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["K"]},
                    {"name": "M", "dtype": "i32", "shape": []},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
                "body": [
                    {
                        "op": "reduce",
                        "result": "S",
                        "operands": ["X"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [0],
                            "assoc": "float_nondeterministic",
                        },
                    },
                    {
                        "op": "cast",
                        "result": "M_f32",
                        "operands": ["M"],
                        "attrs": {"to_dtype": "f32"},
                    },
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["S", "M_f32"],
                        "attrs": {"fn": "div"},
                    },
                ],
            }
        }
    )
    x = np.linspace(-1.0, 1.0, 37, dtype=np.float32)
    result = Lower2CRunner().run(
        Lower2CSpec(fn, {"K": 37}, {"X": "X", "M": "M"}, "Y"),
        {"X": x, "M": np.int32(1003)},
    )

    assert result.ok, result.error
    assert result.outputs["Y"].shape == ()
    np.testing.assert_allclose(
        result.outputs["Y"],
        np.asarray(x.sum() / 1003, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def _have_cuda_triton() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_add_source_replay_lowering_three_way() -> None:
    # This proves the Gate2 wiring.  The saved Intent JSON is LLM-extracted;
    # the temporary five-kernel replay remains probe material, so this test is
    # not claimed as a full new lifter->Gate1->Gate2 corpus result.
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/add/intent.json").read_text())
    )
    n = 65
    rng = np.random.default_rng(7)
    values = {
        "A": rng.standard_normal((1, n)).astype(np.float32),
        "B": rng.standard_normal((1, n)).astype(np.float32),
        "n": n,
    }
    triton_spec = TritonExecutionSpec(
        (
            TritonArgument("a_ptr", ArgumentRole.INPUT, "A", "f32"),
            TritonArgument("b_ptr", ArgumentRole.INPUT, "B", "f32"),
            TritonArgument(
                "out_ptr",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(1, n),
                output_key="Y",
            ),
            TritonArgument("n_elements", ArgumentRole.SCALAR, "n"),
        ),
        ((n + 63) // 64,),
        {"BLOCK": 64, "num_warps": 4, "num_stages": 2},
    )
    case = Gate2Case(
        "add.random.tail65",
        values,
        triton_spec,
        Lower2CSpec(fn, {"M": 1, "N": n}, {"A": "A", "B": "B"}, "Y"),
        ("Y",),
        {"Y": Tolerance(rtol=1.0e-5, atol=1.0e-6)},
    )
    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py", "add_kernel", trusted=True
        ),
        KernelSource.from_file(
            ROOT / "test/back/scrc_probe_replays_v0.py", "add_kernel"
        ),
        case,
    )
    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_collapsed_row_sum_three_way() -> None:
    fn = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "row_sum_gate2",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["M", "N"]}
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": ["M"]}
                ],
                "body": [
                    {
                        "op": "reduce",
                        "result": "Y",
                        "operands": ["X"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [1],
                            "assoc": "float_nondeterministic",
                        },
                    }
                ],
            }
        }
    )
    m, n = 3, 31
    rng = np.random.default_rng(17)
    values = {
        "X": (rng.standard_normal((m, n)) * 0.25).astype(np.float32),
        "M": m,
        "N": n,
    }
    triton_spec = TritonExecutionSpec(
        (
            TritonArgument("x_ptr", ArgumentRole.INPUT, "X", "f32"),
            TritonArgument(
                "out_ptr",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(m,),
                output_key="Y",
            ),
            TritonArgument("M", ArgumentRole.SCALAR, "M"),
            TritonArgument("N", ArgumentRole.SCALAR, "N"),
        ),
        (m,),
        {"BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
    )
    case = Gate2Case(
        "row_sum.fixed.tail31",
        values,
        triton_spec,
        Lower2CSpec(fn, {"M": m, "N": n}, {"X": "X"}, "Y"),
        ("Y",),
        {"Y": Tolerance(rtol=2.0e-4, atol=2.0e-4)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py",
            "row_sum_kernel",
            trusted=True,
        ),
        KernelSource.from_file(
            ROOT / "test/back/scrc_probe_replays_v0.py", "row_sum_kernel"
        ),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_softmax_source_replay_lowering_three_way() -> None:
    # Production-lowering preflight only.  The live SCRC path still requires
    # the artifact to come from the source-visible LLM lifter.
    fn = IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/softmax/intent.json").read_text())
    )
    m, n, block = 4, 37, 64
    rng = np.random.default_rng(23)
    values = {
        "X": (rng.standard_normal((m, n)) * 2.0).astype(np.float32),
        "M": m,
        "N": n,
    }
    triton_spec = TritonExecutionSpec(
        (
            TritonArgument("x_ptr", ArgumentRole.INPUT, "X", "f32"),
            TritonArgument(
                "out_ptr",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(m, n),
                output_key="Y",
            ),
            TritonArgument("M", ArgumentRole.SCALAR, "M"),
            TritonArgument("N", ArgumentRole.SCALAR, "N"),
        ),
        (m,),
        {"BLOCK_N": block, "num_warps": 4, "num_stages": 2},
    )
    case = Gate2Case(
        "softmax.fixed.tail37",
        values,
        triton_spec,
        Lower2CSpec(fn, {"M": m, "N": n}, {"X": "X"}, "Y"),
        ("Y",),
        {"Y": Tolerance(rtol=2.0e-4, atol=2.0e-5)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py",
            "softmax_kernel",
            trusted=True,
        ),
        KernelSource.from_file(
            ROOT / "test/back/scrc_probe_replays_v0.py", "softmax_kernel"
        ),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


def _rmsnorm_intent() -> IntentFunction:
    return IntentFunction.from_json(
        json.loads((ROOT / "corpus/kernels/rms_norm/intent.json").read_text())
    )


def _rmsnorm_gate2_case(
    x: np.ndarray,
    w: np.ndarray,
    eps: np.float32,
    *,
    block: int = 64,
) -> Gate2Case:
    m, n = x.shape
    return Gate2Case(
        f"rmsnorm.fixed.tail{n}.eps-{float(eps):g}",
        {"X": x, "W": w, "eps": eps, "M": m, "N": n},
        TritonExecutionSpec(
            (
                TritonArgument("x_ptr", ArgumentRole.INPUT, "X", "f32"),
                TritonArgument("w_ptr", ArgumentRole.INPUT, "W", "f32"),
                TritonArgument(
                    "out_ptr",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(m, n),
                    output_key="Y",
                ),
                TritonArgument("M", ArgumentRole.SCALAR, "M"),
                TritonArgument("N", ArgumentRole.SCALAR, "N"),
                TritonArgument("eps", ArgumentRole.SCALAR, "eps"),
            ),
            (m,),
            {"BLOCK_N": block, "num_warps": 4, "num_stages": 2},
        ),
        Lower2CSpec(
            _rmsnorm_intent(),
            {"M": m, "N": n},
            {"X": "X", "W": "W", "eps": "eps"},
            "Y",
        ),
        ("Y",),
        {"Y": Tolerance(rtol=3.0e-4, atol=3.0e-5)},
    )


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_rmsnorm_consumes_runtime_eps() -> None:
    fn = _rmsnorm_intent()
    m, n = 4, 37
    rng = np.random.default_rng(29)
    x = (rng.standard_normal((m, n)) * 0.25).astype(np.float32)
    w = (0.75 + rng.random(n) * 0.5).astype(np.float32)
    spec = Lower2CSpec(
        fn,
        {"M": m, "N": n},
        {"X": "X", "W": "W", "eps": "eps"},
        "Y",
    )
    outputs = []

    for eps in (np.float32(1.0e-5), np.float32(1.0e-2)):
        result = Lower2CRunner().run(spec, {"X": x, "W": w, "eps": eps})
        assert result.ok, result.error
        reference = x * (
            1.0
            / np.sqrt(
                np.mean(x * x, axis=1, keepdims=True, dtype=np.float32) + eps
            )
        ) * w
        np.testing.assert_allclose(
            result.outputs["Y"], reference, rtol=3.0e-4, atol=3.0e-5
        )
        outputs.append(result.outputs["Y"])

    assert not np.allclose(outputs[0], outputs[1], rtol=1.0e-7, atol=1.0e-8)


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_rmsnorm_runtime_eps_three_way() -> None:
    m, n = 4, 37
    rng = np.random.default_rng(31)
    x = (rng.standard_normal((m, n)) * 0.25).astype(np.float32)
    w = (0.75 + rng.random(n) * 0.5).astype(np.float32)
    source_outputs = []

    for eps in (np.float32(1.0e-5), np.float32(1.0e-2)):
        case = _rmsnorm_gate2_case(x, w, eps)

        result = Gate2Runner().run(
            KernelSource.from_file(
                ROOT / "test/scrc_probe_sources.py",
                "rmsnorm_kernel",
                trusted=True,
            ),
            KernelSource.from_file(
                ROOT / "test/back/scrc_probe_replays_v0.py", "rmsnorm_kernel"
            ),
            case,
        )

        assert result.status is Gate2Status.PASS, (
            result.source.error,
            result.replay.error,
            result.lowering.error,
        )
        source_outputs.append(result.source.outputs["Y"])

    assert not np.allclose(
        source_outputs[0], source_outputs[1], rtol=1.0e-7, atol=1.0e-8
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_rmsnorm_attributes_baked_eps_to_replay() -> None:
    m, n = 4, 37
    rng = np.random.default_rng(37)
    x = (rng.standard_normal((m, n)) * 0.25).astype(np.float32)
    w = (0.75 + rng.random(n) * 0.5).astype(np.float32)
    eps = np.float32(1.0e-2)
    replay_text = (ROOT / "test/back/scrc_probe_replays_v0.py").read_text()
    baked = replay_text.replace(
        "    shifted = mean_square + eps",
        "    shifted = mean_square + 1.0e-5",
        1,
    )
    assert baked != replay_text

    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py",
            "rmsnorm_kernel",
            trusted=True,
        ),
        KernelSource.from_code(baked, "rmsnorm_kernel"),
        _rmsnorm_gate2_case(x, w, eps),
    )

    assert result.status is Gate2Status.REPLAY_DIVERGED, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )
    assert result.source_lowering is not None
    assert result.source_lowering.passed
    assert result.source_replay is not None
    assert not result.source_replay.passed


def _matmul_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "matmul_gate2",
                "inputs": [
                    {"name": "A", "dtype": "f16", "shape": ["M", "K"]},
                    {"name": "B", "dtype": "f16", "shape": ["K", "N"]},
                ],
                "outputs": [
                    {"name": "O", "dtype": "f32", "shape": ["M", "N"]}
                ],
                "body": [
                    {
                        "op": "contraction",
                        "result": "O",
                        "operands": ["A", "B"],
                        "attrs": {"subscripts": "mk,kn->mn"},
                    }
                ],
            }
        }
    )


def _matmul_gate2_case(a: np.ndarray, b: np.ndarray) -> Gate2Case:
    m, k = a.shape
    other_k, n = b.shape
    assert k == other_k
    return Gate2Case(
        f"matmul.f16.tail-{m}x{n}x{k}",
        {"A": a, "B": b, "M": m, "N": n, "K": k},
        TritonExecutionSpec(
            (
                TritonArgument("a_ptr", ArgumentRole.INPUT, "A", "f16"),
                TritonArgument("b_ptr", ArgumentRole.INPUT, "B", "f16"),
                TritonArgument(
                    "out_ptr",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(m, n),
                    output_key="O",
                ),
                TritonArgument("M", ArgumentRole.SCALAR, "M"),
                TritonArgument("N", ArgumentRole.SCALAR, "N"),
                TritonArgument("K", ArgumentRole.SCALAR, "K"),
            ),
            ((m + 15) // 16, (n + 15) // 16),
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 16,
                "num_warps": 4,
                "num_stages": 2,
            },
        ),
        Lower2CSpec(
            _matmul_intent(),
            {"M": m, "N": n, "K": k},
            {"A": "A", "B": "B"},
            "O",
        ),
        ("O",),
        {"O": Tolerance(rtol=2.0e-3, atol=2.0e-3)},
    )


@pytest.mark.skipif(not HAVE_CPU_LOWERING, reason="needs intent-opt + mlir-runner")
def test_lower2c_matmul_uses_f16_quantized_values_with_f32_output() -> None:
    rng = np.random.default_rng(41)
    a = (rng.standard_normal((17, 19)) * 0.25).astype(np.float16)
    b = (rng.standard_normal((19, 13)) * 0.25).astype(np.float16)
    result = Lower2CRunner().run(
        Lower2CSpec(
            _matmul_intent(),
            {"M": 17, "N": 13, "K": 19},
            {"A": "A", "B": "B"},
            "O",
        ),
        {"A": a, "B": b},
    )

    assert result.ok, result.error
    assert result.outputs["O"].dtype == np.float32
    np.testing.assert_allclose(
        result.outputs["O"],
        a.astype(np.float32) @ b.astype(np.float32),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_matmul_f16_tail_three_way() -> None:
    max_source_lowering_error = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        a = (rng.standard_normal((17, 19)) * 0.25).astype(np.float16)
        b = (rng.standard_normal((19, 13)) * 0.25).astype(np.float16)
        result = Gate2Runner().run(
            KernelSource.from_file(
                ROOT / "test/scrc_probe_sources.py",
                "matmul_kernel",
                trusted=True,
            ),
            KernelSource.from_file(
                ROOT / "test/back/scrc_probe_replays_v0.py", "matmul_kernel"
            ),
            _matmul_gate2_case(a, b),
        )

        assert result.status is Gate2Status.PASS, (
            seed,
            result.source.error,
            result.replay.error,
            result.lowering.error,
        )
        assert result.source_lowering is not None
        error = result.source_lowering.outputs[0].max_abs_error
        assert error is not None
        max_source_lowering_error.append(error)

    assert max(max_source_lowering_error) < 2.0e-3


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_matmul_attributes_wrong_b_address_to_replay() -> None:
    rng = np.random.default_rng(43)
    a = (rng.standard_normal((17, 19)) * 0.25).astype(np.float16)
    b = (rng.standard_normal((19, 13)) * 0.25).astype(np.float16)
    replay_text = (ROOT / "test/back/scrc_probe_replays_v0.py").read_text()
    wrong = replay_text.replace(
        "            b_ptr + k[:, None] * N + n[None, :],",
        "            b_ptr + n[None, :] * K + k[:, None],",
        1,
    )
    assert wrong != replay_text

    result = Gate2Runner().run(
        KernelSource.from_file(
            ROOT / "test/scrc_probe_sources.py",
            "matmul_kernel",
            trusted=True,
        ),
        KernelSource.from_code(wrong, "matmul_kernel"),
        _matmul_gate2_case(a, b),
    )

    assert result.status is Gate2Status.REPLAY_DIVERGED, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )
    assert result.source_lowering is not None
    assert result.source_lowering.passed
    assert result.source_replay is not None
    assert not result.source_replay.passed


_FLAG_GEMS_DOT_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = tl.sum(x * y)
    tl.store(out_ptr, out)
"""


def _flag_gems_dot_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "flag_gems_dot",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["N"]},
                    {"name": "Y", "dtype": "f32", "shape": ["N"]},
                ],
                "outputs": [{"name": "O", "dtype": "f32", "shape": []}],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "product",
                        "operands": ["X", "Y"],
                        "attrs": {"fn": "mul"},
                    },
                    {
                        "op": "reduce",
                        "result": "O",
                        "operands": ["product"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [0],
                            "assoc": "float_nondeterministic",
                        },
                    },
                ],
            }
        }
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_libentry_dot_rank_zero_three_way() -> None:
    module = pytest.importorskip("flag_gems.ops.dot")
    python_function = module.dot_kernel
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    n = 1003
    rng = np.random.default_rng(47)
    x = (rng.standard_normal(n) * 0.25).astype(np.float32)
    y = (rng.standard_normal(n) * 0.25).astype(np.float32)
    case = Gate2Case(
        "flag_gems.dot.rank_zero",
        {"X": x, "Y": y, "N": n},
        TritonExecutionSpec(
            (
                TritonArgument("x_ptr", ArgumentRole.INPUT, "X", "f32"),
                TritonArgument("y_ptr", ArgumentRole.INPUT, "Y", "f32"),
                TritonArgument(
                    "out_ptr",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(),
                    output_key="O",
                ),
                TritonArgument("N", ArgumentRole.SCALAR, "N"),
            ),
            (1,),
            {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
        ),
        Lower2CSpec(
            _flag_gems_dot_intent(),
            {"N": n},
            {"X": "X", "Y": "Y"},
            "O",
        ),
        ("O",),
        {"O": Tolerance(rtol=2.0e-4, atol=2.0e-4)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(source_path, "dot_kernel", trusted=True),
        KernelSource.from_code(_FLAG_GEMS_DOT_REPLAY, "dot_kernel"),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


_FLAG_GEMS_ONES_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lane
    mask = offsets < n_elements
    output = 1.0
    tl.store(output_ptr + offsets, output, mask=mask)
"""


def _flag_gems_ones_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "flag_gems_ones",
                "inputs": [],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": ["N"]}
                ],
                "body": [
                    {
                        "op": "const",
                        "result": "Y",
                        "attrs": {
                            "value": 1.0,
                            "dtype": "f32",
                            "shape": ["N"],
                        },
                    }
                ],
            }
        }
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_ones_no_input_multi_program_tail() -> None:
    module = pytest.importorskip("flag_gems.ops.ones")
    python_function = module.ones_kernel
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    n = 2051
    case = Gate2Case(
        "flag_gems.ones.no_input_tail",
        {"N": n},
        TritonExecutionSpec(
            (
                TritonArgument(
                    "output_ptr",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(n,),
                    output_key="Y",
                ),
                TritonArgument("n_elements", ArgumentRole.SCALAR, "N"),
            ),
            (3,),
            {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
        ),
        Lower2CSpec(_flag_gems_ones_intent(), {"N": n}, {}, "Y"),
        ("Y",),
        {"Y": Tolerance(rtol=0.0, atol=0.0)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(source_path, "ones_kernel", trusted=True),
        KernelSource.from_code(_FLAG_GEMS_ONES_REPLAY, "ones_kernel"),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


_FLAG_GEMS_MEAN_FINAL_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_tile = tl.load(mid + offset, mask=offset < MID_SIZE, other=0.0)
    mid_sum = tl.sum(mid_tile)
    m_as_f32 = M.to(tl.float32)
    out_value = mid_sum / m_as_f32
    tl.store(out + 0, out_value)
"""


def _flag_gems_mean_final_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "flag_gems_mean_final",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": ["K"]},
                    {"name": "M", "dtype": "i32", "shape": []},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
                "body": [
                    {
                        "op": "reduce",
                        "result": "S",
                        "operands": ["X"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [0],
                            "assoc": "float_nondeterministic",
                        },
                    },
                    {
                        "op": "cast",
                        "result": "M_f32",
                        "operands": ["M"],
                        "attrs": {"to_dtype": "f32"},
                    },
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["S", "M_f32"],
                        "attrs": {"fn": "div"},
                    },
                ],
            }
        }
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_mean_final_runtime_i32_divisor() -> None:
    module = pytest.importorskip("flag_gems.ops.mean")
    python_function = module.mean_kernel_2
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    total, mid_size = 1003, 37
    rng = np.random.default_rng(53)
    x = (rng.standard_normal(mid_size) * 0.25).astype(np.float32)
    case = Gate2Case(
        "flag_gems.mean.final_runtime_divisor",
        {"X": x, "M": np.int32(total), "K": mid_size},
        TritonExecutionSpec(
            (
                TritonArgument("mid", ArgumentRole.INPUT, "X", "f32"),
                TritonArgument(
                    "out",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(),
                    output_key="Y",
                ),
                TritonArgument("M", ArgumentRole.SCALAR, "M"),
                TritonArgument("MID_SIZE", ArgumentRole.SCALAR, "K"),
            ),
            (1, 1, 1),
            {"BLOCK_MID": 64, "num_warps": 4, "num_stages": 1},
        ),
        Lower2CSpec(
            _flag_gems_mean_final_intent(),
            {"K": mid_size},
            {"X": "X", "M": "M"},
            "Y",
        ),
        ("Y",),
        {"Y": Tolerance(rtol=2.0e-4, atol=2.0e-6)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(source_path, "mean_kernel_2", trusted=True),
        KernelSource.from_code(
            _FLAG_GEMS_MEAN_FINAL_REPLAY, "mean_kernel_2"
        ),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


_FLAG_GEMS_MAX_FINAL_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_tile = tl.load(mid + offset, mask=offset < mid_size, other=-float("inf"))
    out_value = tl.max(mid_tile)
    tl.store(out + 0, out_value)
"""

_FLAG_GEMS_MIN_FINAL_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_tile = tl.load(mid + offset, mask=offset < mid_size, other=float("inf"))
    out_value = tl.min(mid_tile)
    tl.store(out + 0, out_value)
"""

_FLAG_GEMS_PROD_FINAL_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def _multiply_combine(left, right):
    return left * right

@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_tile = tl.load(mid + offset, mask=offset < mid_size, other=1.0)
    out_value = tl.reduce(mid_tile, axis=0, combine_fn=_multiply_combine)
    tl.store(out + 0, out_value)
"""

_FLAG_GEMS_L2_FINAL_REPLAY = """\
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_tile = tl.load(Mid + offset, mask=offset < MID_SIZE, other=0.0)
    out_value = tl.sqrt(tl.sum(mid_tile))
    tl.store(Out, out_value)
"""


def _flag_gems_final_reduce_intent(
    name: str, reduce_fn: str, *, sqrt: bool = False
) -> IntentFunction:
    body = [
        {
            "op": "reduce",
            "result": "S" if sqrt else "Y",
            "operands": ["X"],
            "attrs": {
                "fn": reduce_fn,
                "axes": [0],
                "assoc": "float_nondeterministic",
            },
        }
    ]
    if sqrt:
        body.append(
            {
                "op": "map_unary",
                "result": "Y",
                "operands": ["S"],
                "attrs": {"fn": "sqrt"},
            }
        )
    return IntentFunction.from_json(
        {
            "name": name,
            "inputs": [{"name": "X", "dtype": "f32", "shape": ["K"]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": []}],
            "body": body,
        }
    )


@pytest.mark.parametrize(
    (
        "module_name",
        "function_name",
        "replay_code",
        "reduce_fn",
        "mid_size",
        "block",
        "grid",
        "distribution",
        "scale",
        "sqrt",
        "tolerance",
    ),
    [
        (
            "flag_gems.ops.max",
            "max_kernel_2",
            _FLAG_GEMS_MAX_FINAL_REPLAY,
            "max",
            63,
            64,
            (1, 1, 1),
            InputDistribution.NORMAL,
            32.0,
            False,
            Tolerance(rtol=0.0, atol=0.0),
        ),
        (
            "flag_gems.ops.min",
            "min_kernel_2",
            _FLAG_GEMS_MIN_FINAL_REPLAY,
            "min",
            64,
            64,
            (1, 1, 1),
            InputDistribution.NORMAL,
            32.0,
            False,
            Tolerance(rtol=0.0, atol=0.0),
        ),
        (
            "flag_gems.ops.prod",
            "prod_kernel_result",
            _FLAG_GEMS_PROD_FINAL_REPLAY,
            "prod",
            17,
            32,
            (1, 1, 1),
            InputDistribution.NEAR_ONE,
            0.125,
            False,
            Tolerance(rtol=2.0e-4, atol=2.0e-6),
        ),
        (
            "flag_gems.ops.vector_norm",
            "l2_norm_kernel_2",
            _FLAG_GEMS_L2_FINAL_REPLAY,
            "sum",
            37,
            64,
            (1,),
            InputDistribution.POSITIVE,
            4.0,
            True,
            Tolerance(rtol=2.0e-4, atol=2.0e-6),
        ),
    ],
)
@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_final_reduction_family(
    module_name: str,
    function_name: str,
    replay_code: str,
    reduce_fn: str,
    mid_size: int,
    block: int,
    grid: tuple[int, ...],
    distribution: InputDistribution,
    scale: float,
    sqrt: bool,
    tolerance: Tolerance,
) -> None:
    module = pytest.importorskip(module_name)
    python_function = getattr(module, function_name)
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    x = generate_tensor_input(
        TensorInputSpec((mid_size,), "f32", distribution, scale),
        seed=20260715,
    )
    is_l2 = function_name == "l2_norm_kernel_2"
    input_name, output_name, extent_name = (
        ("Mid", "Out", "MID_SIZE")
        if is_l2
        else ("mid", "out", "mid_size")
    )
    case = Gate2Case(
        f"flag_gems.{function_name}.final",
        {"X": x, "K": mid_size},
        TritonExecutionSpec(
            (
                TritonArgument(input_name, ArgumentRole.INPUT, "X", "f32"),
                TritonArgument(
                    output_name,
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(),
                    output_key="Y",
                ),
                TritonArgument(extent_name, ArgumentRole.SCALAR, "K"),
            ),
            grid,
            {"BLOCK_MID": block, "num_warps": 4, "num_stages": 1},
        ),
        Lower2CSpec(
            _flag_gems_final_reduce_intent(
                f"lower_{function_name}", reduce_fn, sqrt=sqrt
            ),
            {"K": mid_size},
            {"X": "X"},
            "Y",
        ),
        ("Y",),
        {"Y": tolerance},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(source_path, function_name, trusted=True),
        KernelSource.from_code(replay_code, function_name),
        case,
    )

    assert result.status is Gate2Status.PASS, (
        result.source.error,
        result.replay.error,
        result.lowering.error,
    )


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_ones_boundary_shape_suite() -> None:
    module = pytest.importorskip("flag_gems.ops.ones")
    python_function = module.ones_kernel
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    block = 1024
    cases = []
    for boundary in boundary_1d_extents(block):
        n = boundary.extent
        cases.append(
            Gate2Case(
                f"flag_gems.ones.{boundary.label}",
                {"N": n},
                TritonExecutionSpec(
                    (
                        TritonArgument(
                            "output_ptr",
                            ArgumentRole.OUTPUT,
                            dtype="f32",
                            shape=(n,),
                            output_key="Y",
                        ),
                        TritonArgument(
                            "n_elements", ArgumentRole.SCALAR, "N"
                        ),
                    ),
                    ((n + block - 1) // block,),
                    {
                        "BLOCK_SIZE": block,
                        "num_warps": 4,
                        "num_stages": 1,
                    },
                ),
                Lower2CSpec(_flag_gems_ones_intent(), {"N": n}, {}, "Y"),
                ("Y",),
                {"Y": Tolerance(rtol=0.0, atol=0.0)},
                {
                    "shape_profile": boundary.label,
                    "N": n,
                    "BLOCK_SIZE": block,
                },
            )
        )

    result = Gate2Runner().run_suite(
        KernelSource.from_file(source_path, "ones_kernel", trusted=True),
        KernelSource.from_code(_FLAG_GEMS_ONES_REPLAY, "ones_kernel"),
        Gate2Suite("flag_gems.ones.boundary_shapes", tuple(cases)),
    )

    assert result.passed, [
        (case.case_id, case.status.value) for case in result.cases
    ]
    assert result.status_counts == {"gate2_pass": 6}


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_max_value_seed_matrix() -> None:
    module = pytest.importorskip("flag_gems.ops.max")
    python_function = module.max_kernel_2
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    mid_size, block = 63, 64
    templates = tuple(
        InputCaseTemplate(
            f"max.{distribution.value}",
            {
                "X": TensorInputSpec(
                    (mid_size,), "f32", distribution, scale
                )
            },
            {"K": mid_size},
            {"distribution": distribution.value},
        )
        for distribution, scale in (
            (InputDistribution.NORMAL, 32.0),
            (InputDistribution.UNIFORM, 32.0),
            (InputDistribution.LARGE_DYNAMIC, 1.0),
            (InputDistribution.IEEE_SPECIAL, 1.0),
        )
    )
    generated = generate_input_matrix(templates, (7, 11, 13))
    cases = tuple(
        Gate2Case(
            f"flag_gems.max.{item.case_id}",
            item.values,
            TritonExecutionSpec(
                (
                    TritonArgument("mid", ArgumentRole.INPUT, "X", "f32"),
                    TritonArgument(
                        "out",
                        ArgumentRole.OUTPUT,
                        dtype="f32",
                        shape=(),
                        output_key="Y",
                    ),
                    TritonArgument(
                        "mid_size", ArgumentRole.SCALAR, "K"
                    ),
                ),
                (1, 1, 1),
                {
                    "BLOCK_MID": block,
                    "num_warps": 4,
                    "num_stages": 1,
                },
            ),
            Lower2CSpec(
                _flag_gems_final_reduce_intent(
                    "lower_max_value_matrix", "max"
                ),
                {"K": mid_size},
                {"X": "X"},
                "Y",
            ),
            ("Y",),
            {"Y": Tolerance(rtol=0.0, atol=0.0)},
            item.metadata,
        )
        for item in generated
    )

    result = Gate2Runner().run_suite(
        KernelSource.from_file(source_path, "max_kernel_2", trusted=True),
        KernelSource.from_code(
            _FLAG_GEMS_MAX_FINAL_REPLAY, "max_kernel_2"
        ),
        Gate2Suite("flag_gems.max.value_seed_matrix", cases),
    )

    assert result.passed, [
        (
            case.case_id,
            case.status.value,
            None
            if case.source_lowering is None
            else [item.max_abs_error for item in case.source_lowering.outputs],
        )
        for case in result.cases
    ]
    assert result.status_counts == {"gate2_pass": 12}


@pytest.mark.skipif(
    not HAVE_CPU_LOWERING or not _have_cuda_triton(),
    reason="needs CUDA/Triton + intent-opt + mlir-runner",
)
def test_gate2_real_flag_gems_min_ieee_special_semantics() -> None:
    module = pytest.importorskip("flag_gems.ops.min")
    python_function = module.min_kernel_2
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    mid_size, block = 64, 64
    x = generate_tensor_input(
        TensorInputSpec(
            (mid_size,), "f32", InputDistribution.IEEE_SPECIAL
        ),
        seed=17,
    )
    case = Gate2Case(
        "flag_gems.min.ieee_special",
        {"X": x, "K": mid_size},
        TritonExecutionSpec(
            (
                TritonArgument("mid", ArgumentRole.INPUT, "X", "f32"),
                TritonArgument(
                    "out",
                    ArgumentRole.OUTPUT,
                    dtype="f32",
                    shape=(),
                    output_key="Y",
                ),
                TritonArgument("mid_size", ArgumentRole.SCALAR, "K"),
            ),
            (1, 1, 1),
            {"BLOCK_MID": block, "num_warps": 4, "num_stages": 1},
        ),
        Lower2CSpec(
            _flag_gems_final_reduce_intent("lower_min_ieee", "min"),
            {"K": mid_size},
            {"X": "X"},
            "Y",
        ),
        ("Y",),
        {"Y": Tolerance(rtol=0.0, atol=0.0)},
    )

    result = Gate2Runner().run(
        KernelSource.from_file(source_path, "min_kernel_2", trusted=True),
        KernelSource.from_code(
            _FLAG_GEMS_MIN_FINAL_REPLAY, "min_kernel_2"
        ),
        case,
    )

    assert result.status is Gate2Status.PASS
    assert np.isneginf(result.source.outputs["Y"])
    assert np.isneginf(result.lowering.outputs["Y"])
