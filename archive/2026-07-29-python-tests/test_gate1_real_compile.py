from __future__ import annotations

import pytest
import torch

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.model import Gate1Status
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import StaticReplayWriter


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real Gate1 compilation requires CUDA"
)


COPY_SOURCE = """import triton
import triton.language as tl

@triton.jit
def source_copy(X, Y, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""

COPY_REPLAY = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = v0 < p2
    v2 = tl.load(p0 + v0, mask=v1, other=0.0)
    tl.store(p1 + v0, v2, mask=v1)
"""

REDUCE_SOURCE = """import triton
import triton.language as tl

@triton.jit
def source_reduce(X, Y, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + row * N + offsets, mask=mask, other=0.0)
    squared = value * value
    total = tl.sum(squared, axis=0)
    tl.store(Y + row, total)
"""

REDUCE_REPLAY = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = tl.arange(0, c0)
    v2 = v1 < p2
    v3 = tl.load(p0 + v0 * p2 + v1, mask=v2, other=0.0)
    v4 = v3 * v3
    v5 = tl.sum(v4, axis=0)
    tl.store(p1 + v0, v5)
"""


@pytest.mark.parametrize(
    "expression",
    [
        "libdevice.tanh(value)",
        "libdevice.tan(value)",
        "libdevice.atan(value)",
        "libdevice.atan2(value, value + 1.0)",
    ],
)
def test_reviewed_triton36_libdevice_frontend_calls_compile(
    expression: str,
) -> None:
    code = f"""import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def scrc_math(p0, p1, p2, c0: tl.constexpr):
    offsets = tl.arange(0, c0)
    mask = offsets < p2
    value = tl.load(p0 + offsets, mask=mask, other=0.0)
    result = {expression}
    tl.store(p1 + offsets, result, mask=mask)
"""
    result = TritonCompiler(timeout=180).compile(
        KernelSource.from_code(code, "scrc_math", trusted=False),
        CompileCase(
            {
                "p0": RuntimeArgument.tensor((128,), "float32"),
                "p1": RuntimeArgument.tensor((128,), "float32"),
                "p2": RuntimeArgument.scalar(128, "int32"),
            },
            (1,),
            {"c0": 128},
        ),
    )

    assert result.ok, result.diagnostic


@pytest.mark.parametrize("kind", ["copy", "masked_reduce"])
def test_real_triton_gate1_strict_closure_with_anonymous_signature(kind: str) -> None:
    if kind == "copy":
        artifact = make_copy_artifact()
        source_code, source_name, replay_code = (
            COPY_SOURCE,
            "source_copy",
            COPY_REPLAY,
        )
        arguments = {
            "X": RuntimeArgument.tensor((1024,), "float32"),
            "Y": RuntimeArgument.tensor((1024,), "float32"),
            "N": RuntimeArgument.scalar(1000, "int32"),
        }
        grid = (1,)
    else:
        artifact = make_reduce_artifact()
        artifact.c.values["c0"] = 1024
        source_code, source_name, replay_code = (
            REDUCE_SOURCE,
            "source_reduce",
            REDUCE_REPLAY,
        )
        arguments = {
            "X": RuntimeArgument.tensor((2, 1024), "float32"),
            "Y": RuntimeArgument.tensor((2,), "float32"),
            "N": RuntimeArgument.scalar(1000, "int32"),
        }
        grid = (2,)

    result = Gate1Runner(
        compiler=TritonCompiler(timeout=180),
        writer=StaticReplayWriter(replay_code),
        config=Gate1Config(run_masked_j=False),
    ).run(
        artifact,
        source=KernelSource.from_code(source_code, source_name, trusted=True),
        source_case=CompileCase(
            arguments,
            grid,
            {"BLOCK_SIZE": artifact.c.values["c0"]},
        ),
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.normal.hard_closed
    assert result.normal.comparison is not None
    assert all(result.normal.comparison.hard_passes.values())
