from __future__ import annotations

from pathlib import Path

import pytest

import intentir.backend.compile as backend_compile
from intentir.backend.pipeline import execute_cuda_artifact, lower_to_cuda


torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[2]
_ADD = _ROOT / "test" / "add.mlir"
_MATMUL = _ROOT / "test" / "matmul.mlir"
_SOFTMAX = _ROOT / "test" / "softmax.mlir"

pytestmark = [
    pytest.mark.skipif(
        not Path(backend_compile.INTENT_OPT).exists(), reason="intent-opt not built"
    ),
    pytest.mark.skipif(
        not (Path(backend_compile.CUDA_BIN) / "ptxas").exists(),
        reason="ptxas not available",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def test_generic_artifact_executes_without_recompiling() -> None:
    torch.manual_seed(1)
    a = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    program = lower_to_cuda(_ADD, allow_specialized=False)

    output = execute_cuda_artifact(program, _ADD, {"a": a, "b": b})["out0"]

    assert program.selection.pipeline_kind == "generic"
    assert len(program.kernels) == 1
    torch.testing.assert_close(output, a + b, rtol=0.0, atol=0.0)


def test_direct_elementwise_artifact_executes_the_persisted_cubin() -> None:
    torch.manual_seed(2)
    a = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        _ADD,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=16"
        },
    )

    output = execute_cuda_artifact(program, _ADD, {"a": a, "b": b})["out0"]

    assert program.selection.pipeline_kind == "specialized"
    assert program.selection.fast_path_id == fast_path_id
    torch.testing.assert_close(output, a + b, rtol=0.0, atol=0.0)


def test_generic_multikernel_artifact_preserves_launch_order_and_scratch() -> None:
    torch.manual_seed(3)
    a = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    program = lower_to_cuda(_MATMUL, allow_specialized=False)

    output = execute_cuda_artifact(
        program,
        _MATMUL,
        {"a": a, "b": b},
    )["out0"]

    assert program.selection.pipeline_kind == "generic"
    assert len(program.kernels) == 2
    torch.testing.assert_close(output, a @ b.T, rtol=1.0e-5, atol=1.0e-5)


def test_direct_row_reduction_executes_with_real_collectives() -> None:
    torch.manual_seed(4)
    x = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"
    program = lower_to_cuda(
        _SOFTMAX,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-row-reduction-to-gpu=block-threads=4"
        },
    )

    output = execute_cuda_artifact(program, _SOFTMAX, {"x": x})["out0"]

    assert program.selection.pipeline_kind == "specialized"
    torch.testing.assert_close(
        output,
        torch.softmax(x, dim=1),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
