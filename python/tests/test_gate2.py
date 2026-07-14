"""Gate2 three-way orchestration and one real add-kernel prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2 import (
    ArgumentRole,
    ExecutionResult,
    Gate2Case,
    Gate2Runner,
    Gate2Status,
    Lower2CRunner,
    Lower2CSpec,
    Tolerance,
    TritonArgument,
    TritonExecutionSpec,
)


ROOT = Path(__file__).resolve().parents[2]
HAVE_CPU_LOWERING = (
    (ROOT / "build/tools/intent-opt/intent-opt").exists()
    and Path("/usr/lib/llvm-20/bin/mlir-runner").exists()
)


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
            ROOT / "test/scrc_probe_replays.py", "add_kernel"
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
            ROOT / "test/scrc_probe_replays.py", "row_sum_kernel"
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
            ROOT / "test/scrc_probe_replays.py", "softmax_kernel"
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
                ROOT / "test/scrc_probe_replays.py", "rmsnorm_kernel"
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
    replay_text = (ROOT / "test/scrc_probe_replays.py").read_text()
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
                ROOT / "test/scrc_probe_replays.py", "matmul_kernel"
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
    replay_text = (ROOT / "test/scrc_probe_replays.py").read_text()
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
