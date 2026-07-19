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

_INPLACE_EXP = """
module {
  func.func @inplace_exp(%x: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {intent.inplace = array<i64: 0>} {
    %y = intent.map_unary %x {fn = "exp"}
      : (tensor<8x16xf32>) -> tensor<8x16xf32>
    return %y : tensor<8x16xf32>
  }
}
"""

_LARGE_SOFTMAX = """
module {
  func.func @softmax_large(%x: tensor<4x1024xf32>) -> tensor<4x1024xf32> {
    %m = intent.reduce %x {fn = "max", axes = [1], assoc = "associative"}
      : (tensor<4x1024xf32>) -> tensor<4x1xf32>
    %s = intent.map_binary %x, %m {fn = "sub"}
      : (tensor<4x1024xf32>, tensor<4x1xf32>) -> tensor<4x1024xf32>
    %e = intent.map_unary %s {fn = "exp"}
      : (tensor<4x1024xf32>) -> tensor<4x1024xf32>
    %l = intent.reduce %e {fn = "sum", axes = [1], assoc = "float_nondeterministic"}
      : (tensor<4x1024xf32>) -> tensor<4x1xf32>
    %o = intent.map_binary %e, %l {fn = "div"}
      : (tensor<4x1024xf32>, tensor<4x1xf32>) -> tensor<4x1024xf32>
    return %o : tensor<4x1024xf32>
  }
}
"""

_ODD_TANH = """
module {
  func.func @odd_tanh(%x: tensor<7x33xf32>) -> tensor<7x33xf32> {
    %y = intent.map_unary %x {fn = "tanh"}
      : (tensor<7x33xf32>) -> tensor<7x33xf32>
    return %y : tensor<7x33xf32>
  }
}
"""

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


def test_direct_elementwise_preserves_real_inplace_aliasing() -> None:
    torch.manual_seed(5)
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    expected = torch.exp(x.clone())
    original_pointer = x.data_ptr()
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        _INPLACE_EXP,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=16"
        },
    )

    output = execute_cuda_artifact(program, _INPLACE_EXP, {"x": x})["out0"]

    assert output.data_ptr() == original_pointer
    torch.testing.assert_close(output, expected, rtol=2.0e-6, atol=2.0e-6)


def test_large_direct_row_reduction_executes_shared_collectives() -> None:
    torch.manual_seed(6)
    x = torch.randn((4, 1024), device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"
    program = lower_to_cuda(
        _LARGE_SOFTMAX,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-row-reduction-to-gpu=block-threads=128"
        },
    )

    output = execute_cuda_artifact(program, _LARGE_SOFTMAX, {"x": x})["out0"]

    torch.testing.assert_close(
        output,
        torch.softmax(x, dim=1),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_generic_odd_tanh_is_self_contained_and_executes() -> None:
    x = torch.linspace(
        -10.0,
        10.0,
        7 * 33,
        device="cuda",
        dtype=torch.float32,
    ).reshape(7, 33)
    program = lower_to_cuda(_ODD_TANH, allow_specialized=False)

    output = execute_cuda_artifact(program, _ODD_TANH, {"x": x})["out0"]

    assert program.selection.pipeline_kind == "generic"
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.tanh(x), rtol=3.0e-6, atol=3.0e-6)


@pytest.mark.parametrize(
    ("operation", "atol"),
    (("abs", 0.0), ("atan", 5.0e-7), ("erf", 7.0e-7)),
)
def test_generic_unary_math_avoids_libdevice_and_matches_torch(
    operation: str,
    atol: float,
) -> None:
    source = f"""
module {{
  func.func @{operation}(%x: tensor<4x32xf32>) -> tensor<4x32xf32> {{
    %y = intent.map_unary %x {{fn = "{operation}"}}
      : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %y : tensor<4x32xf32>
  }}
}}
"""
    x = torch.linspace(
        -8.0,
        8.0,
        4 * 32,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, 32)
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]

    assert program.selection.pipeline_kind == "generic"
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        getattr(torch, operation)(x),
        rtol=1.0e-6,
        atol=atol,
    )


def test_generic_atan2_avoids_libdevice_across_all_quadrants() -> None:
    source = """
module {
  func.func @atan2(
      %y: tensor<4x32xf32>, %x: tensor<4x32xf32>)
      -> tensor<4x32xf32> {
    %z = intent.map_binary %y, %x {fn = "atan2"}
      : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %z : tensor<4x32xf32>
  }
}
"""
    positive = torch.linspace(
        0.125,
        5.0,
        32,
        device="cuda",
        dtype=torch.float32,
    )
    y = torch.stack((positive, positive.flip(0), -positive, -positive.flip(0)))
    x = torch.stack((positive.flip(0), -positive, positive.flip(0), -positive))
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(program, source, {"y": y, "x": x})["out0"]

    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        torch.atan2(y, x),
        rtol=1.0e-6,
        atol=7.0e-7,
    )


@pytest.mark.parametrize(
    ("operation", "atol"),
    (("abs", 0.0), ("atan", 5.0e-7), ("erf", 7.0e-7)),
)
def test_generic_unary_math_preserves_special_values(
    operation: str,
    atol: float,
) -> None:
    source = f"""
module {{
  func.func @{operation}(%x: tensor<1x8xf32>) -> tensor<1x8xf32> {{
    %y = intent.map_unary %x {{fn = "{operation}"}}
      : (tensor<1x8xf32>) -> tensor<1x8xf32>
    return %y : tensor<1x8xf32>
  }}
}}
"""
    x = torch.tensor(
        [[-torch.inf, -8.0, -0.0, 0.0, 8.0, torch.inf, torch.nan, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]
    expected = getattr(torch, operation)(x)

    torch.testing.assert_close(
        output,
        expected,
        rtol=1.0e-6,
        atol=atol,
        equal_nan=True,
    )
    assert torch.equal(
        torch.signbit(output[:, 2:4]),
        torch.signbit(expected[:, 2:4]),
    )


def test_generic_atan2_preserves_zero_infinity_and_nan_semantics() -> None:
    source = """
module {
  func.func @atan2_special(
      %y: tensor<2x8xf32>, %x: tensor<2x8xf32>)
      -> tensor<2x8xf32> {
    %z = intent.map_binary %y, %x {fn = "atan2"}
      : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %z : tensor<2x8xf32>
  }
}
"""
    y = torch.tensor(
        [
            [0.0, -0.0, 0.0, -0.0, 1.0, -1.0, torch.inf, torch.inf],
            [-torch.inf, -torch.inf, torch.inf, -torch.inf, 1.0, -1.0, torch.nan, 1.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, torch.inf, -torch.inf],
            [torch.inf, -torch.inf, 1.0, 1.0, torch.inf, -torch.inf, 1.0, torch.nan],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(program, source, {"y": y, "x": x})["out0"]
    expected = torch.atan2(y, x)

    torch.testing.assert_close(
        output,
        expected,
        rtol=1.0e-6,
        atol=7.0e-7,
        equal_nan=True,
    )
    zero = expected == 0.0
    assert torch.equal(torch.signbit(output[zero]), torch.signbit(expected[zero]))
