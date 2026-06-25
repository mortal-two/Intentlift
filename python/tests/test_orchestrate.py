"""M9: end-to-end self-orchestrating execution of a whole intent op-graph.

Unlike ``test_backend_execute.py`` (which hand-specifies each kernel's
grid/block/args), this drives ``orchestrate.run_program`` which parses the lowered
host module and launches every ``gpu.launch_func`` in order with grid/block/args
resolved from the IR. Two cases exercise the single- and multi-kernel paths:

* ``add.mlir``   -> one kernel; output == a + b (bit-exact).
* ``matmul.mlir`` -> two kernels (linalg.fill then matmul) with an intermediate
  zeroed buffer, an f32 scalar fill arg, and an index (K) scalar arg;
  output ~= a @ b.T within 1e-4.

Skips cleanly without a GPU / CUDA / build (mirrors test_backend_execute.py).
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT

_TEST = pathlib.Path(__file__).resolve().parents[2] / "test"
_ADD = _TEST / "add.mlir"
_MATMUL = _TEST / "matmul.mlir"
_SOFTMAX = _TEST / "softmax.mlir"

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def test_run_program_add() -> None:
    """Single-kernel program: run_program(add, {a, b}) == a + b bit-exact."""
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()

    out = run_program(_ADD, {"a": a, "b": b})
    assert set(out) == {"out0"}
    got = out["out0"]
    assert got.shape == (8, 16)
    max_abs = (got - (a + b)).abs().max().item()
    assert max_abs == 0.0, f"add max abs err {max_abs}"


def test_run_program_matmul() -> None:
    """Multi-kernel program: fill + matmul with intermediate buffer, f32 + index
    scalar args. run_program(matmul, {a, b}) ~= a @ b.T within 1e-4."""
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()

    out = run_program(_MATMUL, {"a": a, "b": b})
    assert set(out) == {"out0"}
    got = out["out0"]
    assert got.shape == (8, 8)
    max_abs = (got - a @ b.T).abs().max().item()
    assert max_abs < 1e-4, f"matmul max abs err {max_abs}"


def test_run_program_softmax() -> None:
    """Six-kernel program: row softmax (max-reduce, sub, exp, sum-reduce, div)
    with four loose intermediate buffers (the reduce/elementwise temporaries are
    passed to kernels as bare aligned malloc pointers, never struct-packed) plus
    a struct-packed returned output, an f32 -inf identity (hex literal), and an
    f32 0.0 identity. run_program(softmax, {x}) ~= torch.softmax(x, 1)."""
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device="cuda").contiguous()

    out = run_program(_SOFTMAX, {"x": x})
    assert set(out) == {"out0"}
    got = out["out0"]
    assert got.shape == (4, 8)
    ref = torch.softmax(x, 1)
    rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    assert rel < 1e-3, f"softmax rel err {rel}"
