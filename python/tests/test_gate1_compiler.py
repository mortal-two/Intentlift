from __future__ import annotations

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
