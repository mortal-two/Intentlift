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


def test_parse_multi_result_p4_signature() -> None:
    from intentir.backend.orchestrate import _parse_intent_results

    source = """
    module {
      func.func @paper_flash(%x: tensor<1x128x1x64xf16>)
          -> (tensor<1x128x1x64xf16>, tensor<1x1x128xf32>, tensor<f32>) {
        return
      }
    }
    """

    assert _parse_intent_results(source) == [
        ((1, 128, 1, 64), "f16"),
        ((1, 1, 128), "f32"),
        ((), "f32"),
    ]


def test_parse_and_marshal_f64_launch_constants() -> None:
    import ctypes

    from intentir.backend.orchestrate import _F64Arg, _marshal, _parse_f64

    value = _parse_f64("0x3FE62E42FEFA39EF")
    assert value == pytest.approx(0.6931471805599453)
    keep, pointers = _marshal([_F64Arg(value)])
    assert len(keep) == 1
    assert isinstance(keep[0], ctypes.c_double)
    assert keep[0].value == pytest.approx(value)
    assert len(pointers) == 1


def test_parse_and_marshal_half_launch_constants_as_raw_bits() -> None:
    import ctypes

    from intentir.backend.orchestrate import (
        _BF16Arg,
        _F16Arg,
        _marshal,
        _parse_bf16_bits,
        _parse_f16_bits,
    )

    assert _parse_f16_bits("0xFC00") == 0xFC00
    assert _parse_f16_bits("-2.0") == 0xC000
    assert _parse_bf16_bits("0xFF80") == 0xFF80
    assert _parse_bf16_bits("1.0") == 0x3F80

    keep, pointers = _marshal([_F16Arg(0x3C00), _BF16Arg(0xBF80)])
    assert [type(item) for item in keep] == [ctypes.c_uint16, ctypes.c_uint16]
    assert [item.value for item in keep] == [0x3C00, 0xBF80]
    assert len(pointers) == 2


def test_parse_typed_inplace_result_contract() -> None:
    from intentir.backend.orchestrate import _parse_intent_inplace

    source = """
    func.func @loss(%x: tensor<2x8xf32>, %loss: tensor<2xf32>)
        -> (tensor<2x8xf32>, tensor<2xf32>)
        attributes {intent.inplace = array<i64: 0, 1>} {
      return
    }
    """

    assert _parse_intent_inplace(source, 2) == [0, 1]


def test_parse_returned_malloc_uses_its_own_element_type() -> None:
    from intentir.backend.orchestrate import _parse_malloc_buffers

    host = """
      %c4 = llvm.mlir.constant(4 : i64) : i64
      %c64 = llvm.mlir.constant(64 : i64) : i64
      %null = llvm.mlir.zero : !llvm.ptr
      %g = llvm.getelementptr %null[4] : (!llvm.ptr) -> !llvm.ptr, f16
      %bytes = llvm.ptrtoint %g : !llvm.ptr to i64
      %size = llvm.add %bytes, %c64 : i64
      %malloc = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
      %raw = llvm.ptrtoint %malloc : !llvm.ptr to i64
      %aligned_i = llvm.add %raw, %c64 : i64
      %aligned = llvm.inttoptr %aligned_i : i64 to !llvm.ptr
      %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %desc0 = llvm.insertvalue %aligned, %undef[1]
      %desc1 = llvm.insertvalue %c4, %desc0[3, 0]
    """

    buffers = _parse_malloc_buffers(host)

    assert buffers["%aligned"].shape == (4,)
    assert buffers["%aligned"].elem == "f16"


def test_parse_scalar_and_splat_constant_globals() -> None:
    from intentir.backend.orchestrate import _parse_const_globals

    module = """
      llvm.mlir.global private constant @zero(0 : i64) : i64
      llvm.mlir.global private constant @ones(dense<1.000000e+00> :
          tensor<4xf32>) : !llvm.array<4 x f32>
    """
    host = """
      %z = llvm.mlir.addressof @zero : !llvm.ptr
      %o = llvm.mlir.addressof @ones : !llvm.ptr
      %op = llvm.getelementptr %o[0, 0] : (!llvm.ptr) -> !llvm.ptr,
          !llvm.array<4 x f32>
    """

    constants = _parse_const_globals(module, host)

    assert constants["%z"].shape == ()
    assert constants["%z"].elem == "i64"
    assert constants["%z"].value == 0
    assert constants["%op"].shape == (4,)
    assert constants["%op"].elem == "f32"
    assert constants["%op"].value == 1.0
    assert constants["%op"] is constants["%o"]


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


def test_run_program_binds_layer_b_output_alias_without_copy() -> None:
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    original = a.clone()
    pointer = a.data_ptr()

    out = run_program(_ADD, {"a": a, "b": b}, output_aliases={"out0": "a"})

    assert out["out0"].data_ptr() == pointer
    assert a.data_ptr() == pointer
    torch.testing.assert_close(a, original + b, rtol=0, atol=0)


def test_run_program_accepts_destination_passing_row_kernel(tmp_path) -> None:
    """A direct row conversion appends result memrefs to the host ABI.

    Those destination-passing block arguments are outputs, not extra user
    inputs.  The generic orchestrator must allocate/bind them and recover the
    declared P4 results without a host malloc or compiler fallback.
    """

    import intentir.backend.compile as backend_compile
    from intentir.backend.orchestrate import run_program

    specialized = backend_compile._run(
        [
            backend_compile.INTENT_OPT,
            "-convert-intent-row-reduction-to-gpu=block-threads=4",
            str(_SOFTMAX),
        ]
    )
    specialized_path = tmp_path / "softmax-row-specialized.mlir"
    specialized_path.write_text(specialized, encoding="utf-8")
    kernels = backend_compile.lower_memref_module_to_kernels(specialized_path)
    assert len(kernels) == 1

    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float32, device="cuda").contiguous()
    out = run_program(
        _SOFTMAX,
        {"x": x},
        compile_all=lambda _path: kernels,
    )

    assert out["out0"].shape == (4, 8)
    torch.testing.assert_close(
        out["out0"], torch.softmax(x, dim=1), rtol=1.0e-5, atol=1.0e-6
    )


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


def test_run_program_executes_rank_zero_cast_host_prelude(tmp_path) -> None:
    """A rank-0 i32->f32 cast is bufferized to a host load/cast/store prelude.

    The runtime must initialize that one-element device buffer before launching
    the pointwise consumer kernel; otherwise the whole result is silently zero
    or garbage despite valid cubins.
    """
    from intentir.backend.orchestrate import run_program

    source = """
    module {
      func.func @scale_by_scalar(%x: tensor<2x3xf32>, %n: tensor<i32>)
          -> tensor<2x3xf32> {
        %nf = intent.cast %n {to_dtype = "f32"} : (tensor<i32>) -> tensor<f32>
        %y = intent.map_binary %x, %nf {fn = "mul"} :
            (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
        return %y : tensor<2x3xf32>
      }
    }
    """
    path = tmp_path / "rank-zero-cast.mlir"
    path.write_text(source, encoding="utf-8")
    x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    n = torch.tensor(3, dtype=torch.int32, device="cuda")

    out = run_program(path, {"x": x, "n": n})["out0"]

    torch.testing.assert_close(out, x * 3.0, rtol=0, atol=0)


def test_run_program_loads_vector_gather_scalars_through_literal_gep(
    tmp_path,
) -> None:
    from intentir.backend.orchestrate import run_program

    source = """
    module {
      func.func @sum_weights(%x: tensor<2x3xf32>, %w: tensor<3xf32>)
          -> tensor<2x3xf32> {
        %i0 = arith.constant dense<0> : tensor<i32>
        %i1 = arith.constant dense<1> : tensor<i32>
        %i2 = arith.constant dense<2> : tensor<i32>
        %w0 = intent.gather %w, %i0 {axis = 0 : i64, mode = "take", intent.index_bounds_proven = true} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
        %w1 = intent.gather %w, %i1 {axis = 0 : i64, mode = "take", intent.index_bounds_proven = true} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
        %w2 = intent.gather %w, %i2 {axis = 0 : i64, mode = "take", intent.index_bounds_proven = true} : (tensor<3xf32>, tensor<i32>) -> tensor<f32>
        %s01 = intent.map_binary %w0, %w1 {fn = "add"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %sum = intent.map_binary %s01, %w2 {fn = "add"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %y = intent.map_binary %x, %sum {fn = "mul"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
        return %y : tensor<2x3xf32>
      }
    }
    """
    path = tmp_path / "vector-gather-scalars.mlir"
    path.write_text(source, encoding="utf-8")
    x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    w = torch.tensor([1.25, -2.5, 3.75], dtype=torch.float32, device="cuda")

    out = run_program(path, {"x": x, "w": w})["out0"]

    torch.testing.assert_close(out, x * w.sum(), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (lambda: torch.zeros((3, 2), dtype=torch.float32, device="cuda"), "shape"),
        (lambda: torch.zeros((2, 3), dtype=torch.float64, device="cuda"), "dtype"),
    ],
)
def test_run_program_rejects_input_abi_mismatch(
    tmp_path, tensor, message: str
) -> None:
    from intentir.backend.orchestrate import run_program

    source = """
    module {
      func.func @neg(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %y = intent.map_unary %x {fn = "neg"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
        return %y : tensor<2x3xf32>
      }
    }
    """
    path = tmp_path / f"wrong-{message}.mlir"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_program(path, {"x": tensor()})


@pytest.mark.parametrize(("flag", "use_x"), [(0, False), (7, True)])
def test_run_program_evaluates_scalar_icmp_ne_with_const_global(
    tmp_path, flag: int, use_x: bool
) -> None:
    from intentir.backend.orchestrate import run_program

    source = """
    module {
      func.func @conditional_tensor(%x: tensor<2x3xf32>, %flag: tensor<i32>)
          -> tensor<2x3xf32> {
        %zero_i32 = arith.constant dense<0> : tensor<i32>
        %cond = intent.map_binary %flag, %zero_i32 {fn = "ne"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %zero = arith.constant dense<0.0> : tensor<f32>
        %z = intent.map_binary %x, %zero {fn = "mul"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
        %y = intent.select %cond, %x, %z : (tensor<i1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
        return %y : tensor<2x3xf32>
      }
    }
    """
    path = tmp_path / f"icmp-ne-{flag}.mlir"
    path.write_text(source, encoding="utf-8")
    x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    flag_tensor = torch.tensor(flag, dtype=torch.int32, device="cuda")

    out = run_program(path, {"x": x, "flag": flag_tensor})["out0"]

    expected = x if use_x else torch.zeros_like(x)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
