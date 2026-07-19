from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intentir.verify.gate1.codegen import (
    CodeExtractionError,
    UnsafeReplayCodeError,
    extract_python_module,
    validate_triton_module,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1 import _triton_worker


ROOT = Path(__file__).resolve().parents[2]
PROBE_SOURCES = ROOT / "test" / "scrc_probe_sources.py"

ADD_MODULE = """\
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)
"""


def test_extracts_one_markdown_python_module() -> None:
    response = f"Here is the replay:\n\n```python\n{ADD_MODULE}```\n"
    extracted = extract_python_module(response, required_function="add_kernel")
    assert extracted.startswith("import triton")
    assert "Here is" not in extracted


def test_rejects_ambiguous_model_code_blocks() -> None:
    response = f"```python\n{ADD_MODULE}```\n```python\n{ADD_MODULE}```"
    with pytest.raises(CodeExtractionError, match="2 Python code fences"):
        extract_python_module(response)


def test_accepts_probe_triton_subset() -> None:
    validation = validate_triton_module(PROBE_SOURCES.read_text())
    assert validation.function_names == (
        "add_kernel",
        "row_sum_kernel",
        "softmax_kernel",
        "rmsnorm_kernel",
        "matmul_kernel",
    )


def test_accepts_a_validated_sibling_jit_reduce_helper() -> None:
    module = """import triton
import triton.language as tl

@triton.jit
def product_combine(left, right):
    return left * right

@triton.jit
def product_kernel(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets)
    result = tl.reduce(values, axis=0, combine_fn=product_combine)
    tl.store(out, result)
"""

    validation = validate_triton_module(
        module, required_function="product_kernel"
    )

    assert validation.function_names == ("product_combine", "product_kernel")


def test_accepts_allowlisted_triton_libdevice_pow() -> None:
    module = """import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def pow_kernel(x, exponent, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets)
    powered = libdevice.pow(tl.abs(values), exponent)
    tl.store(out, tl.sum(powered))
"""

    validation = validate_triton_module(
        module, required_function="pow_kernel"
    )

    assert validation.function_names == ("pow_kernel",)


def test_rejects_non_allowlisted_libdevice_calls() -> None:
    module = """import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def bad(x):
    return libdevice.sin(x)
"""

    with pytest.raises(
        UnsafeReplayCodeError, match=r"libdevice\.sin.*forbidden"
    ):
        validate_triton_module(module)


def test_rejects_recursive_sibling_jit_helpers() -> None:
    module = """import triton
import triton.language as tl

@triton.jit
def first(x):
    return second(x)

@triton.jit
def second(x):
    return first(x)
"""

    with pytest.raises(UnsafeReplayCodeError, match="recursive Triton helper"):
        validate_triton_module(module)


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (
            "import os\nimport triton\nimport triton.language as tl\n",
            "imports are limited",
        ),
        (
            "import triton\nimport triton.language as tl\nPAYLOAD = 1\n",
            "top level",
        ),
        (
            "import triton\nimport triton.language as tl\nopen('/tmp/pwn', 'w')\n",
            "top level",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    open('/tmp/pwn', 'w')
""",
            "call to Python name 'open'",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    exec(x)
""",
            "call to Python name 'exec'",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    y = x.__class__
""",
            "dunder attribute",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    while x:
        x = x - 1
""",
            "While is forbidden",
        ),
        (
            """import triton
import triton.language as tl
def bad(x):
    return x
""",
            "@triton.jit",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    return x.item()
""",
            "calls must target",
        ),
        (
            """import triton
import triton.language as tl
@triton.jit
def bad(x):
    return tl.inline_asm_elementwise('ret;', '=r', [], dtype=tl.int32)
""",
            "inline_asm_elementwise",
        ),
    ],
)
def test_strict_ast_policy_rejects_host_effects_and_escape_hatches(
    module: str, message: str
) -> None:
    with pytest.raises(UnsafeReplayCodeError, match=message):
        validate_triton_module(module)


def test_compile_case_rejects_overlapping_schedule_slots() -> None:
    with pytest.raises(ValueError, match="both arguments and schedule"):
        CompileCase(
            arguments={"N": RuntimeArgument.scalar(16)},
            grid=(1,),
            schedule={"N": 16},
        )


def test_compile_case_preserves_none_arguments_and_string_constexprs() -> None:
    case = CompileCase(
        arguments={
            "x": RuntimeArgument.tensor((4,), "float32"),
            "bias": RuntimeArgument.none(),
        },
        grid=(1,),
        schedule={"BIAS_TYPE": "none", "BLOCK": 4},
    )

    assert case.to_payload()["arguments"]["bias"] == {
        "kind": "none",
        "dtype": "none",
        "shape": None,
        "strides": None,
        "value": None,
    }
    assert case.to_payload()["schedule"]["BIAS_TYPE"] == "none"
    assert _triton_worker._runtime_value(None, {"kind": "none"}, "cuda") is None


def test_worker_unwraps_kernel_interface_metadata_and_warmup_result() -> None:
    class Inner:
        arg_names = ["x", "BLOCK"]
        constexprs = [1]
        do_not_specialize = []

    class Wrapper:
        arg_names = ["x", "BLOCK"]
        fn = Inner()

    class Compiled:
        asm = {"ttir": "module", "ttgir": "module"}

    interface = _triton_worker._interface(Wrapper())
    assert interface["arg_names"] == ["x", "BLOCK"]
    assert interface["constexprs"] == ["BLOCK"]
    assert _triton_worker._compiled_kernel((Compiled(), {"BLOCK": 64})).asm


def test_worker_rejects_wrapper_and_inner_argument_order_drift() -> None:
    class Inner:
        arg_names = ["x", "BLOCK"]
        constexprs = [1]
        do_not_specialize = []

    class Wrapper:
        arg_names = ["BLOCK", "x"]
        fn = Inner()

    with pytest.raises(TypeError, match="argument order differs"):
        _triton_worker._interface(Wrapper())


def _cuda_triton_available() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not _cuda_triton_available(), reason="CUDA/Triton unavailable")
def test_compile_add_with_warmup_and_disk_cache(tmp_path: Path) -> None:
    source = KernelSource.from_file(PROBE_SOURCES, "add_kernel")
    case = CompileCase(
        arguments={
            "a_ptr": RuntimeArgument.tensor((1000,), "float32"),
            "b_ptr": RuntimeArgument.tensor((1000,), "float32"),
            "out_ptr": RuntimeArgument.tensor((1000,), "float32"),
            "n_elements": RuntimeArgument.scalar(1000),
        },
        grid=(4,),
        schedule={
            "BLOCK": 256,
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
        },
    )
    compiler = TritonCompiler(cache_dir=tmp_path / "cache")

    result = compiler.compile(source, case, cache_key="probe-add-v1")
    assert result.ok, result.diagnostic
    assert result.ttir and "tt.func" in result.ttir
    assert result.ttgir and "ttg" in result.ttgir
    assert result.python_interface["arg_names"] == [
        "a_ptr",
        "b_ptr",
        "out_ptr",
        "n_elements",
        "BLOCK",
    ]
    assert result.python_interface["constexprs"] == ["BLOCK"]
    assert result.compiler["triton"]
    assert result.target
    assert result.cache_key and result.cache_key != "probe-add-v1"
    assert not result.cache_hit

    cached = compiler.compile(source, case, cache_key="probe-add-v1")
    assert cached.ok
    assert cached.cache_hit
    assert cached.ttir == result.ttir


@pytest.mark.skipif(not _cuda_triton_available(), reason="CUDA/Triton unavailable")
def test_compile_real_flag_gems_libentry_dot(tmp_path: Path) -> None:
    module = pytest.importorskip("flag_gems.ops.dot")
    python_function = module.dot_kernel
    while not inspect.isfunction(python_function):
        python_function = python_function.fn
    source_path = inspect.getsourcefile(python_function)
    assert source_path is not None

    source = KernelSource.from_file(source_path, "dot_kernel", trusted=True)
    case = CompileCase(
        arguments={
            "x_ptr": RuntimeArgument.tensor((1003,), "float32"),
            "y_ptr": RuntimeArgument.tensor((1003,), "float32"),
            "out_ptr": RuntimeArgument.tensor((), "float32"),
            "N": RuntimeArgument.scalar(1003, "int32"),
        },
        grid=(1,),
        schedule={
            "BLOCK_SIZE": 1024,
            "num_warps": 4,
            "num_stages": 1,
        },
    )

    result = TritonCompiler(cache_dir=tmp_path / "cache").compile(source, case)
    assert result.ok, result.diagnostic
    assert result.python_interface["arg_names"] == [
        "x_ptr",
        "y_ptr",
        "out_ptr",
        "N",
        "BLOCK_SIZE",
    ]
    assert result.python_interface["constexprs"] == ["BLOCK_SIZE"]
    assert result.ttir and "tt.func" in result.ttir
