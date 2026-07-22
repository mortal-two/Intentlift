from __future__ import annotations

from dataclasses import dataclass

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.compiler import RuntimeArgument
from intentir.verify.gate1.cute_compiler import (
    CuteCompileCase,
    CuteCompileResult,
    CuteKernelSource,
)
from intentir.verify.gate1.cute_orchestrate import CuteGate1Runner
from intentir.verify.gate1.model import (
    Frontend,
    Gate1Artifact,
    Gate1Status,
    KernelInterface,
    KernelParameter,
    ParameterKind,
    ParameterMode,
    ScheduleSpec,
)
from intentir.verify.gate1.replay import StaticReplayWriter


_GOOD_MLIR = """
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @copy(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {gpu.kernel} {
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      nvvm.mbarrier.init.shared %arg0, %tid : !llvm.ptr<1>, i32
      cuda.return
    }
  }
}
"""
_BAD_MLIR = _GOOD_MLIR.replace(
    "%tid = nvvm.read.ptx.sreg.tid.x : i32",
    "%tid = nvvm.read.ptx.sreg.tid.x : i32\n      %bad = arith.addi %tid, %tid : i32",
)
_GOOD_PTX = """
    mbarrier.init.shared.b64 [%r1], 1;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r2], [%rd1], 16, [%r1];
    bar.sync 0;
"""
_REPLAY = """import cutlass
import cutlass.cute as cute

@cute.kernel
def scrc_k0(p0: cute.Tensor, p1: cute.Tensor):
    v0, _, _ = cute.arch.thread_idx()
    p1[v0] = p0[v0]

@cute.jit
def scrc_entry(t0, t1, stream):
    scrc_k0(t0, t1).launch(
        grid=[1, 1, 1], block=[64, 1, 1], smem=4096, stream=stream
    )
"""


def _artifact() -> Gate1Artifact:
    base = make_copy_artifact()
    return Gate1Artifact(
        base.j,
        KernelInterface(
            "scrc_k0",
            Frontend.CUTE,
            [
                KernelParameter(
                    "p0",
                    ParameterKind.TENSOR,
                    ParameterMode.INPUT,
                    "f32",
                    1,
                    tensor_ref="t0",
                ),
                KernelParameter(
                    "p1",
                    ParameterKind.TENSOR,
                    ParameterMode.OUTPUT,
                    "f32",
                    1,
                    tensor_ref="t1",
                ),
            ],
            ["1"],
            function_name="scrc_k0",
        ),
        ScheduleSpec([], {}),
        base.pi,
        base.location_map,
        source=base.source,
    )


def _case() -> CuteCompileCase:
    return CuteCompileCase(
        [
            RuntimeArgument.tensor((512,), "float32"),
            RuntimeArgument.tensor((512,), "float32"),
        ],
        (1, 1, 1),
        (64, 1, 1),
        4096,
        {"pipeline_stages": 2},
    )


@dataclass
class _Compiler:
    masked_closes: bool = False

    def __post_init__(self) -> None:
        self.untrusted_calls = 0

    def compile(self, source: CuteKernelSource, case: CuteCompileCase):
        del case
        if source.trusted:
            mlir = _GOOD_MLIR
            device_names = ["src", "dst"]
            entry_names = ["src", "dst", "stream"]
        else:
            self.untrusted_calls += 1
            mlir = (
                _GOOD_MLIR
                if self.untrusted_calls == 1 or self.masked_closes
                else _BAD_MLIR
            )
            device_names = ["p0", "p1"]
            entry_names = ["t0", "t1", "stream"]
        return CuteCompileResult(
            True,
            mlir=mlir,
            ptx=_GOOD_PTX,
            python_interface={
                "device": {
                    "arg_names": device_names,
                    "annotations": ["cute.Tensor", "cute.Tensor"],
                },
                "entry": {
                    "arg_names": entry_names,
                    "annotations": [None, None, None],
                },
            },
        )


def test_cute_normal_closure_reports_masked_j_mismatch_separately() -> None:
    writer = StaticReplayWriter(_REPLAY)
    result = CuteGate1Runner(compiler=_Compiler(), writer=writer).run(
        _artifact(),
        source=CuteKernelSource.from_code(
            "source", "source_kernel", "source_entry", trusted=True
        ),
        source_case=_case(),
    )
    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.j_necessity_status.value == "proved"
    assert result.normal.hard_closed
    assert result.masked_j is not None and not result.masked_j.hard_closed
    assert writer.calls == [False, True]


def test_cute_masked_j_closure_does_not_reject_main_gate1() -> None:
    result = CuteGate1Runner(
        compiler=_Compiler(masked_closes=True),
        writer=StaticReplayWriter(_REPLAY),
    ).run(
        _artifact(),
        source=CuteKernelSource.from_code(
            "source", "source_kernel", "source_entry", trusted=True
        ),
        source_case=_case(),
    )
    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.j_necessity_status.value == "not_proved"
    assert result.passed
