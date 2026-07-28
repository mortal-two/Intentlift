from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaKernelSource,
    FrontendParameter,
    find_nvcc,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangKernelSource,
)
from intentir.verify.gate2.native import (
    CudaRunner,
    NativeArgument,
    NativeExecutionSpec,
    TileLangRunner,
)


_CUDA = """extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    int v0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v0 < p2) p1[v0] = p0[v0] + 1.0f;
}
"""

_TILELANG = """import tilelang.language as T

@T.prim_func
def scrc_k0(p0: T.Tensor((32,), "float32"), p1: T.Tensor((32,), "float32")):
    with T.Kernel(1, threads=c0) as (v0,):
        for v1 in T.Parallel(32):
            p1[v1] = p0[v1] + T.float32(1.0)
"""


def _cuda_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


@pytest.mark.skipif(find_nvcc() is None, reason="nvcc unavailable")
def test_cuda_native_runner_executes_compiled_ptx() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    case = CudaCompileCase(
        (
            FrontendParameter("pointer", "f32", 1, "input"),
            FrontendParameter("pointer", "f32", 1, "output"),
            FrontendParameter("scalar", "i32", 0, "value"),
        ),
        block=(32, 1, 1),
        arch=_cuda_arch(),
    )
    spec = NativeExecutionSpec(
        (
            NativeArgument("input", value_key="x", dtype="f32"),
            NativeArgument("output", dtype="f32", shape=(32,), output_key="y"),
            NativeArgument("scalar", dtype="i32", value=32),
        ),
        case,
        (1, 1, 1),
        (32, 1, 1),
    )
    result = CudaRunner().run(
        "cuda",
        CudaKernelSource.from_replay(_CUDA, "scrc_k0"),
        spec,
        {"x": np.arange(32, dtype=np.float32)},
    )
    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["y"], np.arange(32, dtype=np.float32) + 1
    )


@pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="TileLang unavailable",
)
def test_tilelang_native_runner_executes_public_primfunc() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    case = TileLangCompileCase(
        (
            FrontendParameter("pointer", "f32", 1, "input"),
            FrontendParameter("pointer", "f32", 1, "output"),
        ),
        block=(32, 1, 1),
        schedule={"c0": 32},
    )
    spec = NativeExecutionSpec(
        (
            NativeArgument("input", value_key="x", dtype="f32"),
            NativeArgument("output", dtype="f32", shape=(32,), output_key="y"),
        ),
        case,
        (1, 1, 1),
        (32, 1, 1),
    )
    result = TileLangRunner().run(
        "tilelang",
        TileLangKernelSource.from_replay(_TILELANG, "scrc_k0"),
        spec,
        {"x": np.arange(32, dtype=np.float32)},
    )
    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["y"], np.arange(32, dtype=np.float32) + 1
    )
