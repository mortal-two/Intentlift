from __future__ import annotations

from dataclasses import dataclass
from threading import Barrier

import pytest

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.cuda_codegen import (
    UnsafeCudaReplayCodeError,
    validate_cuda_module,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompileResult,
    CudaCompiler,
    CudaKernelSource,
    FrontendParameter,
    find_nvcc,
)
from intentir.verify.gate1.model import Frontend, Gate1Status
from intentir.verify.gate1.native_orchestrate import CudaGate1Runner
from intentir.verify.gate1.orchestrate import Gate1Config
from intentir.verify.gate1.ptx import compare_ptx_fingerprints, fingerprint_ptx
from intentir.verify.gate1.replay import StaticReplayWriter


def _ptx(entry: str, *, load_space: str = "global", names: tuple[str, str, str] = ("%rd1", "%f1", "$L0")) -> str:
    address, value, label = names
    return f"""
.version 8.0
.target sm_80
.address_size 64
.visible .entry {entry}(
    .param .u64 {entry}_param_0,
    .param .u64 {entry}_param_1,
    .param .u32 {entry}_param_2
)
{{
    .reg .pred %p<2>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<3>;
    .loc 1 2 0
    ld.param.u64 {address}, [{entry}_param_0];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 32;
    @%p1 bra {label};
    ld.{load_space}.f32 {value}, [{address}];
    bar.sync 0;
{label}:
    ret;
}}
"""


def test_ptx_fingerprint_alpha_normalizes_entry_registers_labels_and_debug() -> None:
    source = fingerprint_ptx(
        _ptx("source", names=("%rd1", "%f1", "$Lsrc")),
        entry="source",
        interface={"parameters": [{"kind": "pointer"}, {"kind": "pointer"}]},
        target={"arch": "sm_80"},
    )
    replay = fingerprint_ptx(
        _ptx("scrc_k0", names=("%rd3", "%f2", "$Lreplay")),
        entry="scrc_k0",
        interface={"parameters": [{"kind": "pointer"}, {"kind": "pointer"}]},
        target={"arch": "sm_80"},
    )
    assert source.digest == replay.digest
    assert compare_ptx_fingerprints(source, replay).hard_pass


def test_ptx_fingerprint_keeps_memory_space_predicate_and_barrier_semantic() -> None:
    source = fingerprint_ptx(_ptx("kernel"), entry="kernel")
    changed_space = fingerprint_ptx(
        _ptx("kernel", load_space="shared"), entry="kernel"
    )
    comparison = compare_ptx_fingerprints(source, changed_space)
    assert not comparison.hard_pass
    assert not comparison.hard_passes["memory_effects"]

    no_barrier = fingerprint_ptx(
        _ptx("kernel").replace("    bar.sync 0;\n", ""), entry="kernel"
    )
    comparison = compare_ptx_fingerprints(source, no_barrier)
    assert not comparison.hard_passes["collectives"]


def test_ptx_semantic_projection_relaxes_cache_and_address_spelling() -> None:
    source = fingerprint_ptx(
        """
.visible .entry source(.param .u64 source_param_0) {
  .reg .b64 %rd<4>;
  .reg .f32 %f<2>;
  ld.param.u64 %rd1, [source_param_0];
  mov.u64 %rd2, %tid.x;
  or.b64 %rd3, %rd1, %rd2;
  ld.global.nc.f32 %f1, [%rd3];
  st.global.f32 [%rd3], %f1;
  ret;
}
""",
        entry="source",
    )
    replay = fingerprint_ptx(
        """
.visible .entry replay(.param .u64 replay_param_0) {
  .reg .b64 %rd<4>;
  .reg .f32 %f<2>;
  ld.param.u64 %rd1, [replay_param_0];
  mov.u64 %rd2, %tid.x;
  add.s64 %rd3, %rd1, %rd2;
  ld.global.f32 %f1, [%rd3];
  st.global.f32 [%rd3], %f1;
  ret;
}
""",
        entry="replay",
    )

    comparison = compare_ptx_fingerprints(source, replay)

    assert comparison.hard_pass
    assert comparison.primary_difference_category is None
    assert not comparison.diagnostic_passes["raw_instruction_topology"]


def test_ptx_semantic_projection_keeps_value_operation_hard() -> None:
    source_ptx = _ptx("kernel").replace(
        "    bar.sync 0;",
        "    st.global.f32 [%rd1], %f1;\n    bar.sync 0;",
    )
    source = fingerprint_ptx(source_ptx, entry="kernel")
    changed = fingerprint_ptx(
        source_ptx.replace(
            "st.global.f32 [%rd1], %f1;",
            "neg.f32 %f2, %f1;\n    st.global.f32 [%rd1], %f2;",
        ),
        entry="kernel",
    )

    comparison = compare_ptx_fingerprints(source, changed)

    assert not comparison.hard_pass
    assert not comparison.hard_passes["value_operations"]


_CUDA_REPLAY = """extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
    int v0 = (int)threadIdx.x;
    if (v0 < p2) p1[v0] = p0[v0];
}
"""


def test_cuda_replay_policy_accepts_kernel_and_rejects_host_surface() -> None:
    result = validate_cuda_module(_CUDA_REPLAY, required_kernel="scrc_k0")
    assert result.global_functions == ("scrc_k0",)
    launch_bounded = _CUDA_REPLAY.replace(
        "void scrc_k0(", "void __launch_bounds__(16, 1) scrc_k0("
    )
    assert validate_cuda_module(
        launch_bounded, required_kernel="scrc_k0"
    ).global_functions == ("scrc_k0",)
    prefix_bounded = _CUDA_REPLAY.replace(
        "__global__ void", "__global__ __launch_bounds__(16, 1) void"
    )
    assert validate_cuda_module(
        prefix_bounded, required_kernel="scrc_k0"
    ).global_functions == ("scrc_k0",)
    with_helper_call = '''
__device__ int helper(int x) { return x + 1; }
extern "C" __global__ void scrc_k0(const float* p0, float* p1, int p2) {
  int index = helper(p2);
  p1[index] = p0[index];
}
'''
    validated = validate_cuda_module(
        with_helper_call, required_kernel="scrc_k0"
    )
    assert validated.device_helpers == ("helper",)
    with pytest.raises(UnsafeCudaReplayCodeError, match="positive integer"):
        validate_cuda_module(
            _CUDA_REPLAY.replace(
                "void scrc_k0(",
                "void __launch_bounds__(threads, 1) scrc_k0(",
            ),
            required_kernel="scrc_k0",
        )
    with pytest.raises(UnsafeCudaReplayCodeError, match="host-side"):
        validate_cuda_module(
            _CUDA_REPLAY + "\nint helper() { return 0; }\n",
            required_kernel="scrc_k0",
        )
    with pytest.raises(UnsafeCudaReplayCodeError, match="preprocessor"):
        validate_cuda_module(
            '#include "private.h"\n' + _CUDA_REPLAY,
            required_kernel="scrc_k0",
        )


def _case() -> CudaCompileCase:
    return CudaCompileCase(
        (
            FrontendParameter("pointer", "f32", 1, "input"),
            FrontendParameter("pointer", "f32", 1, "output"),
            FrontendParameter("scalar", "i32", 0, "value"),
        ),
        block=(32, 1, 1),
        schedule={"c0": 1024},
    )


def test_cuda_compile_case_records_normalized_include_directories(tmp_path) -> None:
    include = tmp_path / "headers"
    include.mkdir()
    case = CudaCompileCase(
        (FrontendParameter("pointer", "f32", 1, "input"),),
        include_dirs=(str(include),),
    )
    assert case.include_dirs == (str(include.resolve()),)
    assert case.to_payload()["include_dirs"] == [str(include.resolve())]


@dataclass
class _Compiler:
    def compile(self, source: CudaKernelSource, case: CudaCompileCase) -> CudaCompileResult:
        return CudaCompileResult(
            True,
            ptx=_ptx(source.function),
            entry=source.function,
            interface=case.interface,
            target={"arch": case.arch},
        )


def test_cuda_gate1_uses_source_blind_replay_and_native_profile() -> None:
    result = CudaGate1Runner(
        compiler=_Compiler(),  # type: ignore[arg-type]
        writer=StaticReplayWriter(_CUDA_REPLAY, frontend=Frontend.CUDA),
        config=Gate1Config(run_masked_j=False),
    ).run(
        make_copy_artifact(),
        source=CudaKernelSource.from_code(
            'extern "C" __global__ void source(const float* x, float* y, int n) {}',
            "source",
            trusted=True,
        ),
        source_case=_case(),
    )
    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.normal.hard_closed
    assert result.observer_profile.frontend is Frontend.CUDA
    assert result.to_json()["observer_profile"]["compiler_stage"] == "ptx"


def test_cuda_gate1_runs_independent_replay_workers_and_selects_the_closure() -> None:
    barrier = Barrier(2, timeout=2.0)

    class Writer:
        def __init__(self, *, fails: bool) -> None:
            self.fails = fails
            self.delegate = StaticReplayWriter(_CUDA_REPLAY, frontend=Frontend.CUDA)

        def generate(self, artifact, *, masked=False):
            barrier.wait()
            if self.fails:
                raise RuntimeError("independent candidate rejected")
            return self.delegate.generate(artifact, masked=masked)

    result = CudaGate1Runner(
        compiler=_Compiler(),  # type: ignore[arg-type]
        writer=(Writer(fails=True), Writer(fails=False)),  # type: ignore[arg-type]
        config=Gate1Config(run_masked_j=False),
    ).run(
        make_copy_artifact(),
        source=CudaKernelSource.from_code(
            'extern "C" __global__ void source(const float* x, float* y, int n) {}',
            "source",
            trusted=True,
        ),
        source_case=_case(),
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.normal_calls == 2
    assert result.selected_normal_index == 1
    assert len(result.normal_candidates) == 2
    assert "independent candidate rejected" in (
        result.normal_candidates[0].diagnostic or ""
    )
    assert result.normal_candidates[1].hard_closed


@pytest.mark.skipif(find_nvcc() is None, reason="nvcc unavailable")
def test_cuda_real_compile_emits_ptx_without_launching() -> None:
    compiler = CudaCompiler(timeout=120)
    result = compiler.compile(
        CudaKernelSource.from_code(_CUDA_REPLAY, "scrc_k0"),
        _case(),
    )
    assert result.ok, result.diagnostic
    assert result.ptx is not None and ".entry" in result.ptx
    fingerprint = fingerprint_ptx(
        result.ptx,
        entry="scrc_k0",
        interface=_case().interface,
        target={"arch": _case().arch},
    )
    assert fingerprint.memory_effects

    renamed_code = _CUDA_REPLAY.replace("scrc_k0", "source_kernel")
    renamed_code = renamed_code.replace("p0", "input_ptr")
    renamed_code = renamed_code.replace("p1", "output_ptr")
    renamed_code = renamed_code.replace("p2", "extent")
    renamed = compiler.compile(
        CudaKernelSource.from_code(
            renamed_code,
            "source_kernel",
            trusted=True,
        ),
        _case(),
    )
    assert renamed.ok, renamed.diagnostic
    renamed_fingerprint = fingerprint_ptx(
        renamed.ptx or "",
        entry="source_kernel",
        interface=_case().interface,
        target={"arch": _case().arch},
    )
    assert compare_ptx_fingerprints(fingerprint, renamed_fingerprint).hard_pass
