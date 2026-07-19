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


def _half_softmax_source(rows: int, columns: int, mlir_type: str) -> str:
    row = f"tensor<{rows}x{columns}x{mlir_type}>"
    scalar = f"tensor<{rows}x1x{mlir_type}>"
    return f"""
module {{
  func.func @half_softmax(%x: {row}) -> {row} {{
    %max = intent.reduce %x {{fn = "max", axes = [1], assoc = "associative"}}
      : ({row}) -> {scalar}
    %shifted = intent.map_binary %x, %max {{fn = "sub"}}
      : ({row}, {scalar}) -> {row}
    %exp = intent.map_unary %shifted {{fn = "exp"}}
      : ({row}) -> {row}
    %sum = intent.reduce %exp
      {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}}
      : ({row}) -> {scalar}
    %out = intent.map_binary %exp, %sum {{fn = "div"}}
      : ({row}, {scalar}) -> {row}
    return %out : {row}
  }}
}}
"""


def _half_rms_norm_source(rows: int, columns: int, mlir_type: str) -> str:
    storage_row = f"tensor<{rows}x{columns}x{mlir_type}>"
    storage_column = f"tensor<{columns}x{mlir_type}>"
    compute_row = f"tensor<{rows}x{columns}xf32>"
    compute_column = f"tensor<{columns}xf32>"
    statistic = f"tensor<{rows}x1xf32>"
    return f"""
module {{
  func.func @half_rms_norm(%X: {storage_row}, %W: {storage_column},
                           %eps: tensor<f32>) -> {storage_row} {{
    %Xf = intent.cast %X {{to_dtype = "f32"}}
      : ({storage_row}) -> {compute_row}
    %Wf = intent.cast %W {{to_dtype = "f32"}}
      : ({storage_column}) -> {compute_column}
    %x2 = intent.map_binary %Xf, %Xf {{fn = "mul"}}
      : ({compute_row}, {compute_row}) -> {compute_row}
    %mean = intent.reduce %x2
      {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}}
      : ({compute_row}) -> {statistic}
    %me = intent.map_binary %mean, %eps {{fn = "add"}}
      : ({statistic}, tensor<f32>) -> {statistic}
    %rms = intent.map_unary %me {{fn = "sqrt"}}
      : ({statistic}) -> {statistic}
    %inv = intent.map_unary %rms {{fn = "recip"}}
      : ({statistic}) -> {statistic}
    %xs = intent.map_binary %Xf, %inv {{fn = "mul"}}
      : ({compute_row}, {statistic}) -> {compute_row}
    %Yf = intent.map_binary %xs, %Wf {{fn = "mul"}}
      : ({compute_row}, {compute_column}) -> {compute_row}
    %Y = intent.cast %Yf {{to_dtype = "{mlir_type}"}}
      : ({compute_row}) -> {storage_row}
    return %Y : {storage_row}
  }}
}}
"""


def _half_layer_norm_source(rows: int, columns: int, mlir_type: str) -> str:
    storage_row = f"tensor<{rows}x{columns}x{mlir_type}>"
    storage_column = f"tensor<{columns}x{mlir_type}>"
    compute_row = f"tensor<{rows}x{columns}xf32>"
    compute_column = f"tensor<{columns}xf32>"
    statistic = f"tensor<{rows}x1xf32>"
    return f"""
module {{
  func.func @half_layer_norm(%X: {storage_row}, %W: {storage_column},
                             %B: {storage_column}, %eps: tensor<f32>)
      -> ({storage_row}, {statistic}, {statistic}) {{
    %Xf = intent.cast %X {{to_dtype = "f32"}}
      : ({storage_row}) -> {compute_row}
    %Wf = intent.cast %W {{to_dtype = "f32"}}
      : ({storage_column}) -> {compute_column}
    %Bf = intent.cast %B {{to_dtype = "f32"}}
      : ({storage_column}) -> {compute_column}
    %mean = intent.reduce %Xf
      {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}}
      : ({compute_row}) -> {statistic}
    %xc = intent.map_binary %Xf, %mean {{fn = "sub"}}
      : ({compute_row}, {statistic}) -> {compute_row}
    %xc2 = intent.map_binary %xc, %xc {{fn = "mul"}}
      : ({compute_row}, {compute_row}) -> {compute_row}
    %var = intent.reduce %xc2
      {{fn = "mean", axes = [1], assoc = "float_nondeterministic"}}
      : ({compute_row}) -> {statistic}
    %vare = intent.map_binary %var, %eps {{fn = "add"}}
      : ({statistic}, tensor<f32>) -> {statistic}
    %rstd = intent.map_unary %vare {{fn = "rsqrt"}}
      : ({statistic}) -> {statistic}
    %xn = intent.map_binary %xc, %rstd {{fn = "mul"}}
      : ({compute_row}, {statistic}) -> {compute_row}
    %xnw = intent.map_binary %xn, %Wf {{fn = "mul"}}
      : ({compute_row}, {compute_column}) -> {compute_row}
    %Yf = intent.map_binary %xnw, %Bf {{fn = "add"}}
      : ({compute_row}, {compute_column}) -> {compute_row}
    %Y = intent.cast %Yf {{to_dtype = "{mlir_type}"}}
      : ({compute_row}) -> {storage_row}
    return %Y, %mean, %rstd : {storage_row}, {statistic}, {statistic}
  }}
}}
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


@pytest.mark.parametrize(
    ("operation", "atol"),
    (("abs", 0.0), ("atan", 5.0e-7), ("erf", 7.0e-7)),
)
def test_direct_unary_math_reuses_target_special_value_semantics(
    operation: str,
    atol: float,
) -> None:
    source = f"""
module {{
  func.func @{operation}_direct(
      %x: tensor<4x32xf32>) -> tensor<4x32xf32> {{
    %y = intent.map_unary %x {{fn = "{operation}"}}
      : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %y : tensor<4x32xf32>
  }}
}}
"""
    row = torch.tensor(
        [-torch.inf, -8.0, -0.0, 0.0, 8.0, torch.inf, torch.nan, 1.0],
        device="cuda",
        dtype=torch.float32,
    )
    x = row.repeat(4, 4)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
        },
    )

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]
    expected = getattr(torch, operation)(x)

    assert program.selection.pipeline_kind == "specialized"
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


def test_direct_atan2_reuses_target_special_value_semantics() -> None:
    source = """
module {
  func.func @atan2_direct(
      %y: tensor<4x32xf32>, %x: tensor<4x32xf32>)
      -> tensor<4x32xf32> {
    %z = intent.map_binary %y, %x {fn = "atan2"}
      : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %z : tensor<4x32xf32>
  }
}
"""
    special_y = torch.tensor(
        [
            0.0,
            -0.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            torch.inf,
            torch.inf,
            -torch.inf,
            -torch.inf,
            torch.inf,
            -torch.inf,
            1.0,
            -1.0,
            torch.nan,
            1.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    special_x = torch.tensor(
        [
            0.0,
            0.0,
            -0.0,
            -0.0,
            0.0,
            0.0,
            torch.inf,
            -torch.inf,
            torch.inf,
            -torch.inf,
            1.0,
            1.0,
            torch.inf,
            -torch.inf,
            1.0,
            torch.nan,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    y = special_y.repeat(8).reshape(4, 32)
    x = special_x.repeat(8).reshape(4, 32)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
        },
    )

    output = execute_cuda_artifact(program, source, {"y": y, "x": x})["out0"]
    expected = torch.atan2(y, x)

    assert program.selection.pipeline_kind == "specialized"
    torch.testing.assert_close(
        output,
        expected,
        rtol=1.0e-6,
        atol=7.0e-7,
        equal_nan=True,
    )
    zero = expected == 0.0
    assert torch.equal(torch.signbit(output[zero]), torch.signbit(expected[zero]))


@pytest.mark.parametrize("pipeline_kind", ("generic", "specialized"))
def test_pow_is_libdevice_free_and_executes_on_cuda(pipeline_kind: str) -> None:
    source = """
module {
  func.func @pow(
      %base: tensor<4x32xf32>, %exponent: tensor<4x32xf32>)
      -> tensor<4x32xf32> {
    %z = intent.map_binary %base, %exponent {fn = "pow"}
      : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %z : tensor<4x32xf32>
  }
}
"""
    base = torch.linspace(
        0.125,
        8.0,
        4 * 32,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, 32)
    exponent = torch.linspace(
        -3.0,
        3.0,
        4 * 32,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, 32)
    if pipeline_kind == "generic":
        program = lower_to_cuda(source, allow_specialized=False)
    else:
        fast_path_id = "cuda.elementwise.row.vectorized.v1"
        program = lower_to_cuda(
            source,
            only_fast_path=fast_path_id,
            pass_overrides={
                fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
            },
        )

    output = execute_cuda_artifact(
        program,
        source,
        {"base": base, "exponent": exponent},
    )["out0"]

    assert program.selection.pipeline_kind == pipeline_kind
    torch.testing.assert_close(
        output,
        torch.pow(base, exponent),
        rtol=3.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.parametrize("pipeline_kind", ("generic", "specialized"))
def test_pow_matches_cuda_special_value_and_integer_parity_semantics(
    pipeline_kind: str,
) -> None:
    source = """
module {
  func.func @pow_special(
      %base: tensor<4x32xf32>, %exponent: tensor<4x32xf32>)
      -> tensor<4x32xf32> {
    %z = intent.map_binary %base, %exponent {fn = "pow"}
      : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %z : tensor<4x32xf32>
  }
}
"""
    base_values = torch.tensor(
        [torch.nan, -torch.inf, -2.0, -1.0, -0.0, 0.0, 1.0, torch.inf],
        device="cuda",
        dtype=torch.float32,
    )
    exponent_values = torch.tensor(
        [
            torch.nan,
            -torch.inf,
            -3.0,
            -2.0,
            -1.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.5,
            2.0,
            3.0,
            torch.inf,
            8388607.0,
            8388608.0,
            8388609.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    base = base_values.repeat_interleave(exponent_values.numel()).reshape(4, 32)
    exponent = exponent_values.repeat(base_values.numel()).reshape(4, 32)
    if pipeline_kind == "generic":
        program = lower_to_cuda(source, allow_specialized=False)
    else:
        fast_path_id = "cuda.elementwise.row.vectorized.v1"
        program = lower_to_cuda(
            source,
            only_fast_path=fast_path_id,
            pass_overrides={
                fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
            },
        )

    output = execute_cuda_artifact(
        program,
        source,
        {"base": base, "exponent": exponent},
    )["out0"]
    expected = torch.pow(base, exponent)

    assert program.selection.pipeline_kind == pipeline_kind
    torch.testing.assert_close(
        output,
        expected,
        rtol=3.0e-5,
        atol=2.0e-6,
        equal_nan=True,
    )
    signed_special = (expected == 0.0) | torch.isinf(expected)
    assert torch.equal(
        torch.signbit(output[signed_special]),
        torch.signbit(expected[signed_special]),
    )


def test_pow_handles_subnormals_near_one_and_f32_parity_boundaries() -> None:
    source = """
module {
  func.func @pow_edges(
      %base: tensor<3x8xf32>, %exponent: tensor<3x8xf32>)
      -> tensor<3x8xf32> {
    %z = intent.map_binary %base, %exponent {fn = "pow"}
      : (tensor<3x8xf32>, tensor<3x8xf32>) -> tensor<3x8xf32>
    return %z : tensor<3x8xf32>
  }
}
"""
    zero = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    one = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    min_subnormal = torch.nextafter(zero, one)
    min_normal = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device="cuda",
        dtype=torch.float32,
    )
    below_one = torch.nextafter(one, zero)
    above_one = torch.nextafter(one, torch.tensor(torch.inf, device="cuda"))
    base = torch.stack(
        (
            min_subnormal,
            min_subnormal,
            min_subnormal,
            min_normal,
            min_normal,
            above_one,
            below_one,
            torch.tensor(1.0001, device="cuda"),
            *[torch.tensor(-1.0, device="cuda") for _ in range(16)],
        )
    ).reshape(3, 8)
    exponent = torch.stack(
        (
            torch.tensor(0.01, device="cuda"),
            torch.tensor(0.5, device="cuda"),
            one,
            torch.tensor(0.01, device="cuda"),
            torch.tensor(0.5, device="cuda"),
            torch.tensor(100000.0, device="cuda"),
            torch.tensor(100000.0, device="cuda"),
            torch.tensor(10000.0, device="cuda"),
            torch.tensor(-16777218.0, device="cuda"),
            torch.tensor(-16777216.0, device="cuda"),
            torch.tensor(-16777215.0, device="cuda"),
            torch.tensor(-8388609.0, device="cuda"),
            torch.tensor(-8388608.0, device="cuda"),
            torch.tensor(-8388607.0, device="cuda"),
            torch.tensor(8388607.0, device="cuda"),
            torch.tensor(8388608.0, device="cuda"),
            torch.tensor(8388609.0, device="cuda"),
            torch.tensor(16777215.0, device="cuda"),
            torch.tensor(16777216.0, device="cuda"),
            torch.tensor(16777218.0, device="cuda"),
            torch.tensor(torch.finfo(torch.float32).max, device="cuda"),
            below_one,
            above_one,
            min_subnormal,
        )
    ).reshape(3, 8)
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(
        program,
        source,
        {"base": base, "exponent": exponent},
    )["out0"]
    expected = torch.pow(base, exponent)

    torch.testing.assert_close(
        output,
        expected,
        rtol=3.0e-6,
        atol=0.0,
        equal_nan=True,
    )
    assert output[0, 2].item() == min_subnormal.item()


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    ((torch.float16, "f16"), (torch.bfloat16, "bf16")),
)
@pytest.mark.parametrize("pipeline_kind", ("generic", "specialized"))
def test_half_add_executes_with_the_declared_storage_dtype(
    dtype: torch.dtype,
    mlir_type: str,
    pipeline_kind: str,
) -> None:
    source = f"""
module {{
  func.func @half_add(
      %a: tensor<7x33x{mlir_type}>, %b: tensor<7x33x{mlir_type}>)
      -> tensor<7x33x{mlir_type}> {{
    %y = intent.map_binary %a, %b {{fn = "add"}}
      : (tensor<7x33x{mlir_type}>, tensor<7x33x{mlir_type}>)
        -> tensor<7x33x{mlir_type}>
    return %y : tensor<7x33x{mlir_type}>
  }}
}}
"""
    torch.manual_seed(7)
    a = torch.randn((7, 33), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    if pipeline_kind == "generic":
        program = lower_to_cuda(source, allow_specialized=False)
    else:
        fast_path_id = "cuda.elementwise.row.vectorized.v1"
        program = lower_to_cuda(
            source,
            only_fast_path=fast_path_id,
            pass_overrides={
                fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=33"
            },
        )

    output = execute_cuda_artifact(program, source, {"a": a, "b": b})["out0"]

    assert program.selection.pipeline_kind == pipeline_kind
    assert output.dtype == dtype
    assert torch.equal(output, a + b)


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    ((torch.float16, "f16"), (torch.bfloat16, "bf16")),
)
def test_direct_half_vector_dag_rounds_each_typed_map_result(
    dtype: torch.dtype,
    mlir_type: str,
) -> None:
    source = f"""
module {{
  func.func @half_chain(
      %a: tensor<4x128x{mlir_type}>,
      %b: tensor<4x128x{mlir_type}>,
      %c: tensor<4x128x{mlir_type}>) -> tensor<4x128x{mlir_type}> {{
    %sum = intent.map_binary %a, %b {{fn = "add"}}
      : (tensor<4x128x{mlir_type}>, tensor<4x128x{mlir_type}>)
        -> tensor<4x128x{mlir_type}>
    %out = intent.map_binary %sum, %c {{fn = "mul"}}
      : (tensor<4x128x{mlir_type}>, tensor<4x128x{mlir_type}>)
        -> tensor<4x128x{mlir_type}>
    return %out : tensor<4x128x{mlir_type}>
  }}
}}
"""
    torch.manual_seed(8)
    a = torch.randn((4, 128), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    c = torch.randn_like(a)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
        },
    )

    output = execute_cuda_artifact(
        program,
        source,
        {"a": a, "b": b, "c": c},
    )["out0"]

    assert program.selection.pipeline_kind == "specialized"
    assert torch.equal(output, (a + b) * c)


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol"),
    (
        (torch.float16, "f16", 1.0e-3),
        (torch.bfloat16, "bf16", 8.0e-3),
    ),
)
def test_direct_half_math_accepts_an_f32_broadcast_scalar(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
) -> None:
    source = f"""
module {{
  func.func @half_math(%x: tensor<4x128x{mlir_type}>, %scale: tensor<f32>)
      -> (tensor<4x128x{mlir_type}>, tensor<4x128x{mlir_type}>) {{
    %tanh = intent.map_unary %x {{fn = "tanh"}}
      : (tensor<4x128x{mlir_type}>) -> tensor<4x128x{mlir_type}>
    %scaled = intent.map_binary %x, %scale {{fn = "mul"}}
      : (tensor<4x128x{mlir_type}>, tensor<f32>)
        -> tensor<4x128x{mlir_type}>
    return %tanh, %scaled
      : tensor<4x128x{mlir_type}>, tensor<4x128x{mlir_type}>
  }}
}}
"""
    torch.manual_seed(9)
    x = torch.randn((4, 128), device="cuda", dtype=dtype)
    scale = torch.tensor(1.2345, device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
        },
    )

    outputs = execute_cuda_artifact(program, source, {"x": x, "scale": scale})

    assert outputs["out0"].dtype == dtype
    assert outputs["out1"].dtype == dtype
    torch.testing.assert_close(
        outputs["out0"],
        torch.tanh(x),
        rtol=0.0,
        atol=atol,
    )
    assert torch.equal(outputs["out1"], (x.float() * scale).to(dtype))


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol"),
    (
        (torch.float16, "f16", 2.0e-3),
        (torch.bfloat16, "bf16", 1.0e-2),
    ),
)
def test_direct_half_inplace_alias_preserves_pointer_and_dtype(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
) -> None:
    source = f"""
module {{
  func.func @half_inplace_exp(%x: tensor<4x128x{mlir_type}>)
      -> tensor<4x128x{mlir_type}>
      attributes {{intent.inplace = array<i64: 0>}} {{
    %y = intent.map_unary %x {{fn = "exp"}}
      : (tensor<4x128x{mlir_type}>) -> tensor<4x128x{mlir_type}>
    return %y : tensor<4x128x{mlir_type}>
  }}
}}
"""
    torch.manual_seed(10)
    x = torch.randn((4, 128), device="cuda", dtype=dtype)
    expected = torch.exp(x.clone())
    pointer = x.data_ptr()
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=32"
        },
    )

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]

    assert output.data_ptr() == pointer
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol"),
    (
        (torch.float16, "f16", 5.0e-4),
        (torch.bfloat16, "bf16", 4.0e-3),
    ),
)
def test_generic_half_softmax_executes_promoted_math_and_half_scalar_abi(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
) -> None:
    source = _half_softmax_source(4, 32, mlir_type)
    torch.manual_seed(11)
    x = torch.randn((4, 32), device="cuda", dtype=dtype)
    program = lower_to_cuda(source, allow_specialized=False)

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]

    assert program.selection.pipeline_kind == "generic"
    assert len(program.kernels) == 7
    assert output.dtype == dtype
    torch.testing.assert_close(
        output,
        torch.softmax(x, dim=1),
        rtol=0.0,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol", "sum_atol"),
    (
        (torch.float16, "f16", 1.0e-3, 2.0e-3),
        (torch.bfloat16, "bf16", 4.0e-3, 8.0e-3),
    ),
)
@pytest.mark.parametrize(
    ("columns", "block_threads"),
    ((32, 32), (37, 32), (1000, 128), (1024, 128)),
)
def test_direct_half_softmax_uses_one_f32_accumulation_kernel(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
    sum_atol: float,
    columns: int,
    block_threads: int,
) -> None:
    source = _half_softmax_source(4, columns, mlir_type)
    torch.manual_seed(12 + columns)
    x = 4.0 * torch.randn((4, columns), device="cuda", dtype=dtype)
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: (
                "-convert-intent-row-reduction-to-gpu="
                f"block-threads={block_threads}"
            )
        },
    )

    output = execute_cuda_artifact(program, source, {"x": x})["out0"]
    reference = torch.softmax(x.float(), dim=1).to(dtype)

    assert program.selection.pipeline_kind == "specialized"
    assert program.selection.fast_path_id == fast_path_id
    assert len(program.kernels) == 1
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)
    torch.testing.assert_close(output, reference, rtol=0.0, atol=atol)
    torch.testing.assert_close(
        output.float().sum(dim=1),
        torch.ones(4, device="cuda"),
        rtol=0.0,
        atol=sum_atol,
    )


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol", "rtol"),
    (
        (torch.float16, "f16", 2.0e-3, 2.0e-3),
        (torch.bfloat16, "bf16", 1.0e-2, 1.0e-2),
    ),
)
@pytest.mark.parametrize(
    ("columns", "block_threads"),
    ((32, 32), (37, 32), (1000, 128), (1024, 128)),
)
def test_direct_half_rms_norm_uses_explicit_f32_compute_graph(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
    rtol: float,
    columns: int,
    block_threads: int,
) -> None:
    source = _half_rms_norm_source(4, columns, mlir_type)
    torch.manual_seed(20 + columns)
    # Squaring this in f16 would overflow; the explicit cast must move the
    # complete norm graph into f32 before x*x is formed.
    x = 300.0 * torch.randn((4, columns), device="cuda", dtype=dtype)
    weight = torch.randn((columns,), device="cuda", dtype=dtype)
    eps = torch.tensor(1.0e-6, device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: (
                "-convert-intent-row-reduction-to-gpu="
                f"block-threads={block_threads}"
            )
        },
    )

    output = execute_cuda_artifact(
        program,
        source,
        {"X": x, "W": weight, "eps": eps},
    )["out0"]
    xf = x.float()
    reference = (
        xf
        * torch.rsqrt((xf * xf).mean(dim=1, keepdim=True) + eps)
        * weight.float()
    ).to(dtype)

    assert program.selection.pipeline_kind == "specialized"
    assert program.selection.fast_path_id == fast_path_id
    assert len(program.kernels) == 1
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "mlir_type", "atol", "rtol"),
    (
        (torch.float16, "f16", 2.0e-3, 2.0e-3),
        (torch.bfloat16, "bf16", 1.0e-2, 1.0e-2),
    ),
)
@pytest.mark.parametrize(
    ("columns", "block_threads"),
    ((32, 32), (37, 32), (1000, 128), (1024, 128)),
)
def test_direct_half_layer_norm_keeps_statistics_in_f32(
    dtype: torch.dtype,
    mlir_type: str,
    atol: float,
    rtol: float,
    columns: int,
    block_threads: int,
) -> None:
    source = _half_layer_norm_source(4, columns, mlir_type)
    torch.manual_seed(30 + columns)
    # At N=1024, a half-precision row sum near 64*N would overflow.  The f32
    # mean/variance path must remain finite and retain the small centered signal.
    x = 64.0 + torch.randn((4, columns), device="cuda", dtype=dtype)
    weight = torch.randn((columns,), device="cuda", dtype=dtype)
    bias = torch.randn((columns,), device="cuda", dtype=dtype)
    eps = torch.tensor(1.0e-5, device="cuda", dtype=torch.float32)
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"
    program = lower_to_cuda(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: (
                "-convert-intent-row-reduction-to-gpu="
                f"block-threads={block_threads}"
            )
        },
    )

    outputs = execute_cuda_artifact(
        program,
        source,
        {"X": x, "W": weight, "B": bias, "eps": eps},
    )
    xf = x.float()
    mean = xf.mean(dim=1, keepdim=True)
    centered = xf - mean
    rstd = torch.rsqrt((centered * centered).mean(dim=1, keepdim=True) + eps)
    reference = (centered * rstd * weight.float() + bias.float()).to(dtype)

    assert program.selection.pipeline_kind == "specialized"
    assert program.selection.fast_path_id == fast_path_id
    assert len(program.kernels) == 1
    assert outputs["out0"].dtype == dtype
    assert outputs["out1"].dtype == torch.float32
    assert outputs["out2"].dtype == torch.float32
    torch.testing.assert_close(outputs["out0"], reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(outputs["out1"], mean, rtol=0.0, atol=2.0e-5)
    torch.testing.assert_close(outputs["out2"], rstd, rtol=0.0, atol=2.0e-5)
