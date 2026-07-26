from __future__ import annotations

from types import SimpleNamespace

import pytest

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.cuda_compiler import CudaCompileCase, CudaKernelSource
from intentir.verify.gate1.model import Frontend
from intentir.verify.gate1.replay import (
    replay_source_ingress,
    source_blind_replay_compile_check,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangKernelSource,
)


CUDA_REPLAY = r'''
extern "C" __global__ void scrc_k0(
    const float* p0,
    float* p1,
    int p2) {
  int v0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v0 < p2) p1[v0] = p0[v0];
}
'''


TILELANG_REPLAY = r'''
import tilelang.language as T

@T.prim_func
def scrc_k0(
    p0: T.Tensor((32,), "float32"),
    p1: T.Tensor((32,), "float32"),
    p2: T.int32,
):
    with T.Kernel(1, threads=32) as (v0,):
        p1[v0] = p0[v0]
'''


@pytest.mark.parametrize(
    ("frontend", "source"),
    (
        (Frontend.CUDA, CUDA_REPLAY),
        (Frontend.TILELANG, TILELANG_REPLAY),
    ),
)
def test_native_replay_source_ingress_uses_audited_validator(
    frontend: Frontend,
    source: str,
) -> None:
    result = replay_source_ingress(
        source,
        artifact=make_copy_artifact(),
        frontend=frontend,
    )

    assert result.status == "pass"
    assert result.code is not None


@pytest.mark.parametrize(
    ("frontend", "source", "source_type", "case_type"),
    (
        (
            Frontend.CUDA,
            CUDA_REPLAY,
            CudaKernelSource,
            CudaCompileCase,
        ),
        (
            Frontend.TILELANG,
            TILELANG_REPLAY,
            TileLangKernelSource,
            TileLangCompileCase,
        ),
    ),
)
def test_native_synthetic_compile_probe_is_frontend_dispatched(
    frontend: Frontend,
    source: str,
    source_type: type,
    case_type: type,
) -> None:
    class Compiler:
        compiled_source = None
        compiled_case = None

        def compile(self, compiled_source, compiled_case):
            self.compiled_source = compiled_source
            self.compiled_case = compiled_case
            return SimpleNamespace(ok=True, diagnostic="")

    compiler = Compiler()
    result = source_blind_replay_compile_check(
        source,
        make_copy_artifact(),
        "scrc_k0",
        frontend=frontend,
        compiler=compiler,
    )

    assert result.status == "pass"
    assert isinstance(compiler.compiled_source, source_type)
    assert isinstance(compiler.compiled_case, case_type)
    assert [item.kind for item in compiler.compiled_case.parameters] == [
        "pointer",
        "pointer",
        "scalar",
    ]
