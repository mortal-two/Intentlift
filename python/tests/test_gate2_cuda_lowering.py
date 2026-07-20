from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import intentir.verify.gate2.cuda_lowering as cuda_gate
import intentir.backend.cuda.compile as backend_compile
from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import (
    Boundary,
    BroadcastSpec,
    Frontend,
    Gate1Artifact,
    KernelInterface,
    KernelParameter,
    LocationMap,
    OutputMapping,
    ParameterKind,
    ParameterMode,
    ReplayPlan,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    CudaLoweringOptions,
    CudaLoweringRunner,
    CudaLoweringSpec,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    GateIntentBundle,
    TritonArgument,
    TritonExecutionSpec,
)


torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[2]
_HAVE_CUDA_BACKEND = (
    Path(backend_compile.INTENT_OPT).exists()
    and (Path(backend_compile.CUDA_BIN) / "ptxas").exists()
    and torch.cuda.is_available()
)


def _artifact(
    intent: IntentFunction,
    *,
    boundary: Boundary | None = None,
    num_warps: int | None = None,
    source: str = "private source must not reach lowering",
) -> Gate1Artifact:
    node_ids = {
        f"n{index}": f"body[{index}]" for index, _op in enumerate(intent.body)
    }
    if boundary is None:
        boundary = Boundary(
            output_mappings=[
                OutputMapping(
                    tensor.name,
                    tensor.name,
                    f"n{len(intent.body) - 1}",
                )
                for tensor in intent.outputs
            ]
        )
    parameters = [
        KernelParameter(
            tensor.name,
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            tensor.dtype,
            len(tensor.shape),
            tensor_ref=tensor.name,
        )
        for tensor in intent.inputs
    ] + [
        KernelParameter(
            tensor.name,
            ParameterKind.POINTER,
            ParameterMode.OUTPUT,
            tensor.dtype,
            len(tensor.shape),
            tensor_ref=tensor.name,
        )
        for tensor in intent.outputs
    ]
    schedule = (
        ScheduleSpec(
            [ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER)],
            {"num_warps": num_warps},
        )
        if num_warps is not None
        else ScheduleSpec([], {})
    )
    return Gate1Artifact(
        j=SemanticArchive(intent.name, intent, node_ids, boundary),
        interface=KernelInterface(
            kernel_id=intent.name,
            frontend=Frontend.TRITON,
            parameters=parameters,
            grid=["1"],
            compiler_slots=(["num_warps"] if num_warps is not None else []),
            function_name=intent.name,
        ),
        schedule=schedule,
        plan=ReplayPlan([]),
        location_map=LocationMap(f"{intent.name}.py", []),
        source=source,
    )


def _two_output_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_two_outputs",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [8, 128]}],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [8, 128]},
                    {"name": "Z", "dtype": "f32", "shape": [8, 128]},
                ],
                "body": [
                    {
                        "op": "map_unary",
                        "result": "Y",
                        "operands": ["X"],
                        "attrs": {"fn": "neg"},
                    },
                    {
                        "op": "map_unary",
                        "result": "Z",
                        "operands": ["X"],
                        "attrs": {"fn": "abs"},
                    },
                ],
            }
        }
    )


def _bf16_contraction_intent() -> IntentFunction:
    return IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_bf16_mma",
                "inputs": [
                    {"name": "A", "dtype": "bf16", "shape": [64, 32]},
                    {"name": "W", "dtype": "bf16", "shape": [64, 32]},
                ],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [64, 64]}],
                "body": [
                    {
                        "op": "contraction",
                        "result": "Y",
                        "operands": ["A", "W"],
                        "attrs": {
                            "subscripts": "mk,nk->mn",
                            "accumulator_dtype": "f32",
                        },
                        "type": {"dtype": "f32", "shape": [64, 64]},
                    }
                ],
            }
        }
    )


def test_gate_bundle_excludes_source_replay_and_verdict_surfaces() -> None:
    artifact = _artifact(_two_output_intent(), num_warps=4)
    changed_source = replace(artifact, source="different private implementation")

    first = GateIntentBundle.from_artifact(
        artifact,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "neg", "Z": "abs"},
    )
    second = GateIntentBundle.from_artifact(
        changed_source,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "neg", "Z": "abs"},
    )

    assert first == second
    assert first.intent is artifact.j.intent
    assert first.declared_schedule == {"num_warps": 4}
    assert not hasattr(first, "source")
    assert not hasattr(first, "verdict")


def test_gate_bundle_requires_exact_multi_output_bindings() -> None:
    intent = _two_output_intent()
    with pytest.raises(ValueError, match="output bindings mismatch"):
        GateIntentBundle(
            intent,
            {},
            {"X": "input"},
            {"Y": "only_one_output"},
        )


def test_gate_bundle_derives_bindings_from_live_interface() -> None:
    artifact = _artifact(_two_output_intent(), num_warps=4)

    bundle = GateIntentBundle.from_artifact_interface(artifact, dims={})

    assert bundle.input_bindings == {"X": "X"}
    assert bundle.output_bindings == {"Y": "Y", "Z": "Z"}
    assert bundle.declared_schedule == {"num_warps": 4}


def test_gate_bundle_live_interface_fails_closed_without_tensor_ref() -> None:
    artifact = _artifact(_two_output_intent())
    artifact.interface.parameters[0].tensor_ref = None

    with pytest.raises(ValueError, match="no input interface binding"):
        GateIntentBundle.from_artifact_interface(artifact, dims={})


def test_cuda_lower_materializes_boundary_broadcast_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_outer_sub",
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
    archive = SemanticArchive(
        intent.name,
        intent,
        {"n0": "body[0]"},
        Boundary(
            broadcasts=[
                BroadcastSpec("n0", "lhs", [1]),
                BroadcastSpec("n0", "rhs", [0]),
            ],
            output_mappings=[OutputMapping("out", "out", "n0")],
        ),
    )
    observed: dict[str, str] = {}

    def fake_lower(source: str, **_kwargs):
        observed["source"] = source
        return object()

    monkeypatch.setattr(cuda_gate, "lower_to_cuda", fake_lower)
    artifact = CudaLoweringRunner().lower(
        CudaLoweringSpec(
            GateIntentBundle(
                intent,
                {"N": 3, "M": 5},
                {"lhs": "lhs", "rhs": "rhs"},
                {"out": "out"},
                semantic_archive=archive,
            )
        )
    )

    assert artifact.broadcast_consumer_count == 1
    assert "tensor.expand_shape %lhs" in observed["source"]
    assert "output_shape [3, 1]" in observed["source"]
    assert "tensor.expand_shape %rhs" not in observed["source"]
    assert "tensor<5xf32>) -> tensor<3x5xf32>" in observed["source"]


def test_cuda_lower_reuses_existing_singleton_broadcast_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_existing_singleton",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [4, 8]},
                    {"name": "row", "dtype": "f32", "shape": [4, 1]},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 8]}
                ],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["X", "row"],
                        "attrs": {"fn": "mul"},
                        "type": {"dtype": "f32", "shape": [4, 8]},
                    }
                ],
            }
        }
    )
    archive = SemanticArchive(
        intent.name,
        intent,
        {"n0": "body[0]"},
        Boundary(
            broadcasts=[BroadcastSpec("n0", "row", [1])],
            output_mappings=[OutputMapping("Y", "Y", "n0")],
        ),
    )
    observed: dict[str, str] = {}

    def fake_lower(source: str, **_kwargs):
        observed["source"] = source
        return object()

    monkeypatch.setattr(cuda_gate, "lower_to_cuda", fake_lower)
    lowered = CudaLoweringRunner().lower(
        CudaLoweringSpec(
            GateIntentBundle(
                intent,
                {},
                {"X": "X", "row": "row"},
                {"Y": "Y"},
                semantic_archive=archive,
            )
        )
    )

    assert lowered.broadcast_consumer_count == 1
    assert "tensor.expand_shape %row" not in observed["source"]
    assert "tensor<4x1xf32>) -> tensor<4x8xf32>" in observed["source"]


def test_cuda_lower_rejects_non_singleton_declared_broadcast_axis() -> None:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_invalid_existing_broadcast",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [4, 8]},
                    {"name": "row", "dtype": "f32", "shape": [4, 8]},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 8]}
                ],
                "body": [
                    {
                        "op": "map_binary",
                        "result": "Y",
                        "operands": ["X", "row"],
                        "attrs": {"fn": "mul"},
                        "type": {"dtype": "f32", "shape": [4, 8]},
                    }
                ],
            }
        }
    )
    archive = SemanticArchive(
        intent.name,
        intent,
        {"n0": "body[0]"},
        Boundary(
            broadcasts=[BroadcastSpec("n0", "row", [1])],
            output_mappings=[OutputMapping("Y", "Y", "n0")],
        ),
    )

    with pytest.raises(
        ValueError,
        match="existing broadcast dimensions must be size one",
    ):
        CudaLoweringRunner().lower(
            CudaLoweringSpec(
                GateIntentBundle(
                    intent,
                    {},
                    {"X": "X", "row": "row"},
                    {"Y": "Y"},
                    semantic_archive=archive,
                )
            )
        )


def test_cuda_bundle_accepts_contraction_broadcast_encoded_by_subscripts() -> None:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_contraction_broadcast",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [4, 8]},
                    {"name": "scale", "dtype": "f32", "shape": [4]},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 8]}
                ],
                "body": [
                    {
                        "op": "contraction",
                        "result": "Y",
                        "operands": ["X", "scale"],
                        "attrs": {"subscripts": "mn,m->mn"},
                    }
                ],
            }
        }
    )
    archive = SemanticArchive(
        intent.name,
        intent,
        {"n0": "body[0]"},
        Boundary(
            broadcasts=[BroadcastSpec("n0", "scale", [1])],
            output_mappings=[OutputMapping("Y", "Y", "n0")],
        ),
    )

    bundle = GateIntentBundle(
        intent,
        {},
        {"X": "X", "scale": "scale"},
        {"Y": "Y"},
        semantic_archive=archive,
    )

    assert bundle.semantic_archive is archive


def test_cuda_bundle_rejects_contraction_broadcast_conflicting_with_subscripts() -> None:
    intent = IntentFunction.from_json(
        {
            "intent_function": {
                "name": "gate_cuda_bad_contraction_broadcast",
                "inputs": [
                    {"name": "X", "dtype": "f32", "shape": [4, 8]},
                    {"name": "scale", "dtype": "f32", "shape": [4]},
                ],
                "outputs": [
                    {"name": "Y", "dtype": "f32", "shape": [4, 8]}
                ],
                "body": [
                    {
                        "op": "contraction",
                        "result": "Y",
                        "operands": ["X", "scale"],
                        "attrs": {"subscripts": "mn,m->mn"},
                    }
                ],
            }
        }
    )
    archive = SemanticArchive(
        intent.name,
        intent,
        {"n0": "body[0]"},
        Boundary(
            broadcasts=[BroadcastSpec("n0", "scale", [0])],
            output_mappings=[OutputMapping("Y", "Y", "n0")],
        ),
    )

    with pytest.raises(ValueError, match="disagree with subscripts"):
        GateIntentBundle(
            intent,
            {},
            {"X": "X", "scale": "scale"},
            {"Y": "Y"},
            semantic_archive=archive,
        )


class _ExpectedRunner:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def run(self, leg, _program, _spec, _values):
        return ExecutionResult(leg, True, self.outputs)


@pytest.mark.skipif(not _HAVE_CUDA_BACKEND, reason="needs built CUDA backend")
def test_gate2_consumes_multi_output_bundle_through_real_cuda() -> None:
    intent = _two_output_intent()
    artifact = _artifact(intent, num_warps=4)
    bundle = GateIntentBundle.from_artifact(
        artifact,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "neg", "Z": "abs"},
    )
    spec = CudaLoweringSpec(
        bundle,
        CudaLoweringOptions(
            only_fast_path="cuda.elementwise.row.vectorized.v1"
        ),
    )
    values = np.linspace(-3.0, 3.0, 8 * 128, dtype=np.float32).reshape(8, 128)
    expected = {"neg": -values, "abs": np.abs(values)}
    case = Gate2Case(
        "cuda.multi_output",
        {"input": values},
        TritonExecutionSpec(
            (
                TritonArgument(
                    "unused", ArgumentRole.SCALAR, value=1
                ),
            ),
            (1,),
        ),
        spec,
        ("neg", "abs"),
    )
    source_runner = _ExpectedRunner(expected)
    replay_runner = _ExpectedRunner(expected)
    result = Gate2Runner(
        source_runner=source_runner,
        replay_runner=replay_runner,
        lowering_runner=CudaLoweringRunner(),
    ).run(
        KernelSource.from_code("", "unused"),
        KernelSource.from_code("", "unused"),
        case,
    )

    assert result.status is Gate2Status.PASS
    assert result.lowering.metadata["protocol"] == (
        "canonical_intent_abc_cuda_cubin_v1"
    )
    selection = result.lowering.metadata["lowering"]["program"]["selection"]
    assert selection["pipeline_kind"] == "specialized"
    assert selection["fast_path_id"] == "cuda.elementwise.row.vectorized.v1"


@pytest.mark.skipif(not _HAVE_CUDA_BACKEND, reason="needs built CUDA backend")
def test_gate2_automatic_elementwise_selection_consumes_layer_c_schedule() -> None:
    intent = _two_output_intent()
    bundle = GateIntentBundle.from_artifact(
        _artifact(intent, num_warps=4),
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "neg", "Z": "abs"},
    )

    lowered = CudaLoweringRunner().lower(CudaLoweringSpec(bundle))

    assert lowered.program.selection.pipeline_kind == "specialized"
    assert lowered.program.selection.fast_path_id == (
        "cuda.elementwise.row.vectorized.v1"
    )
    assert lowered.schedule_decision is not None
    assert lowered.schedule_decision.observed_block_threads == 128
    assert lowered.schedule_decision.schedule.block_threads == 128
    assert "block-threads=128" in lowered.program.selection.pass_argument


@pytest.mark.skipif(not _HAVE_CUDA_BACKEND, reason="needs built CUDA backend")
def test_gate1_bf16_contraction_reaches_native_mma_and_executes() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device has no BF16 tensor-core support")
    intent = _bf16_contraction_intent()
    bundle = GateIntentBundle.from_artifact(
        _artifact(intent),
        dims={},
        input_bindings={"A": "A", "W": "W"},
        output_bindings={"Y": "Y"},
    )
    spec = CudaLoweringSpec(
        bundle,
        CudaLoweringOptions(
            only_fast_path="cuda.linear.f16_f32.bias_saved.mma_sync.v1"
        ),
    )
    runner = CudaLoweringRunner()
    lowered = runner.lower(spec)
    binary_mlir = "\n".join(kernel.binary_mlir for kernel in lowered.program.kernels)

    assert lowered.program.selection.pipeline_kind == "specialized"
    assert lowered.program.selection.fast_path_id == (
        "cuda.linear.f16_f32.bias_saved.mma_sync.v1"
    )
    assert (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        in binary_mlir
    )

    torch.manual_seed(19)
    a = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    result = runner.execute(spec, lowered, {"A": a, "W": w})
    assert result.ok, result.error
    expected = torch.mm(a, w.t(), out_dtype=torch.float32).cpu().numpy()
    np.testing.assert_allclose(result.outputs["Y"], expected, rtol=1.0e-3, atol=1.0e-3)
