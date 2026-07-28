from __future__ import annotations

from types import SimpleNamespace

import pytest

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.cuda_compiler import CudaCompileCase, CudaKernelSource
from intentir.verify.gate1.model import Frontend, Gate1Artifact
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


UNSAFE_TILELANG_SCHEDULE_BINDING = r'''import tilelang.language as T

x0 = T.symbolic("x0")

@T.prim_func
def scrc_k0(
    p0: T.Tensor((x0,), "float32"),
    p1: T.Tensor((x0,), "float32"),
    p2: T.int32,
):
    with T.Kernel(x0) as (v0,):
        p1[v0] = p0[v0]
'''


def _launch_slot_artifact() -> Gate1Artifact:
    payload = make_copy_artifact().to_json()
    payload["c"]["slots"][0].update(
        kind="compiler",
        role="launch_block_x",
    )
    return Gate1Artifact.from_json(payload)


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


def test_tilelang_ingress_returns_actionable_source_blind_schedule_rule() -> None:
    result = replay_source_ingress(
        UNSAFE_TILELANG_SCHEDULE_BINDING,
        artifact=make_copy_artifact(),
        frontend=Frontend.TILELANG,
    )

    assert result.to_check_response()["diagnostics"] == [
        {
            "rule_code": (
                "tilelang_module_binding_requires_uppercase_literal_or_T_dynamic"
            ),
            "path": "<line:3>",
            "anonymous_handles": [],
            "line": 3,
        }
    ]


def test_tilelang_ingress_keeps_compiler_slots_out_of_runtime_abi() -> None:
    with_runtime_c = TILELANG_REPLAY.replace(
        "    p2: T.int32,\n",
        "    p2: T.int32,\n    c0: T.int32,\n",
    )

    result = replay_source_ingress(
        with_runtime_c,
        artifact=make_copy_artifact(),
        frontend=Frontend.TILELANG,
    )

    assert result.status == "fail"
    assert result.to_check_response()["diagnostics"][0]["rule_code"] == (
        "abi_parameter_count"
    )


def test_tilelang_ingress_binds_launch_slots_only_to_kernel_threads() -> None:
    artifact = _launch_slot_artifact()
    valid = TILELANG_REPLAY.replace("threads=32", "threads=c0")
    tuple_valid = valid.replace("threads=c0", "threads=(c0, 1, 1)")
    wrong_grid = TILELANG_REPLAY.replace(
        "T.Kernel(1, threads=32)", "T.Kernel(c0, threads=32)"
    )
    wrong_thread = TILELANG_REPLAY

    assert replay_source_ingress(
        valid, artifact=artifact, frontend=Frontend.TILELANG
    ).status == "pass"
    assert replay_source_ingress(
        tuple_valid, artifact=artifact, frontend=Frontend.TILELANG
    ).status == "pass"
    grid_result = replay_source_ingress(
        wrong_grid, artifact=artifact, frontend=Frontend.TILELANG
    )
    thread_result = replay_source_ingress(
        wrong_thread, artifact=artifact, frontend=Frontend.TILELANG
    )
    assert grid_result.to_check_response()["diagnostics"][0]["rule_code"] == (
        "tilelang_launch_slot_grid_forbidden"
    )
    assert thread_result.to_check_response()["diagnostics"][0]["rule_code"] == (
        "tilelang_launch_slot_thread_binding"
    )


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
    assert compiler.compiled_case.schedule == {"c0": 1}
