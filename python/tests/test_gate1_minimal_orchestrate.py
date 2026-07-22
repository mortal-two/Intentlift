from __future__ import annotations

from dataclasses import dataclass

from direct_fixtures import make_reduce_artifact
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.model import Gate1Artifact, Gate1Status
from intentir.verify.gate1.orchestrate import Gate1Runner, _replay_case
from intentir.verify.gate1.replay import StaticReplayWriter


GOOD_TTIR = """
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""
BAD_TTIR = GOOD_TTIR.replace(
    "%value = tt.load %x : !tt.ptr<f32>",
    "%one = arith.constant 1.0 : f32\n    %value = tt.load %x : !tt.ptr<f32>\n    %bad = arith.addf %value, %one : f32",
).replace("tt.store %out, %value", "tt.store %out, %bad")

REPLAY = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = v0 < p2
    v2 = tl.load(p0 + v0, mask=v1, other=0.0)
    tl.store(p1 + v0, v2, mask=v1)
"""


@dataclass
class FakeCompiler:
    masked_closes: bool = False

    def __post_init__(self) -> None:
        self.untrusted_calls = 0

    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        del case
        if source.trusted:
            text = GOOD_TTIR
            interface = {
                "arg_names": ["X", "Y", "N", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
            }
        else:
            self.untrusted_calls += 1
            text = GOOD_TTIR if self.untrusted_calls == 1 or self.masked_closes else BAD_TTIR
            interface = {
                "arg_names": ["p0", "p1", "p2", "c0"],
                "constexprs": ["c0"],
                "constexpr_indices": [3],
            }
        return CompileResult(
            True, ttir=text, python_interface=interface, compiler={"test": "1"}
        )


def _case() -> CompileCase:
    return CompileCase(
        arguments={
            "X": RuntimeArgument.tensor((4, 8), "float32"),
            "Y": RuntimeArgument.tensor((4,), "float32"),
            "N": RuntimeArgument.scalar(8, "int32"),
        },
        grid=(1,),
        schedule={"BLOCK": 128},
    )


def test_normal_closure_reports_masked_j_mismatch_separately() -> None:
    writer = StaticReplayWriter(REPLAY)
    compiler = FakeCompiler(masked_closes=False)
    result = Gate1Runner(compiler=compiler, writer=writer).run(
        make_reduce_artifact(),
        source=KernelSource.from_code("source", "kernel", trusted=True),
        source_case=_case(),
    )
    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.j_necessity_status.value == "proved"
    assert result.to_json()["j_necessity_status"] == "proved"
    assert result.normal.hard_closed
    assert result.masked_j is not None and not result.masked_j.hard_closed
    assert result.normal_calls == 1
    assert result.masked_j_calls == 1
    assert writer.calls == [False, True]


def test_masked_j_hard_closure_does_not_reject_main_gate1() -> None:
    writer = StaticReplayWriter(REPLAY)
    result = Gate1Runner(
        compiler=FakeCompiler(masked_closes=True), writer=writer
    ).run(
        make_reduce_artifact(),
        source=KernelSource.from_code("source", "kernel", trusted=True),
        source_case=_case(),
    )
    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.j_necessity_status.value == "not_proved"
    assert result.passed
    assert result.masked_j is not None and result.masked_j.hard_closed


def test_source_parameter_spelling_is_not_a_gate1_semantic() -> None:
    result = Gate1Runner(
        compiler=FakeCompiler(masked_closes=False),
        writer=StaticReplayWriter(REPLAY),
    ).run(
        make_reduce_artifact(),
        source=KernelSource.from_code("source", "kernel", trusted=True),
        source_case=_case(),
    )
    assert result.normal.hard_closed
    assert result.normal.comparison is not None
    assert result.normal.comparison.hard_passes["interface"]


def test_source_schedule_value_must_match_c_before_replay() -> None:
    artifact = make_reduce_artifact()
    artifact.schedule.values["c0"] = 256
    writer = StaticReplayWriter(REPLAY)
    result = Gate1Runner(compiler=FakeCompiler(), writer=writer).run(
        artifact,
        source=KernelSource.from_code("source", "kernel", trusted=True),
        source_case=_case(),
    )
    assert result.status is Gate1Status.NO_CLOSURE
    assert result.normal_calls == 0
    assert writer.calls == []
    assert "differs from C" in (result.normal.diagnostic or "")


def test_bound_constexpr_binding_is_copied_from_harness_not_c() -> None:
    payload = make_reduce_artifact().to_json()
    payload["interface"]["parameters"].insert(
        3,
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "i32",
            "rank": 0,
        },
    )
    artifact = Gate1Artifact.from_json(payload)
    source_case = CompileCase(
        arguments={
            "X": RuntimeArgument.tensor((4, 8), "float32"),
            "Y": RuntimeArgument.tensor((4,), "float32"),
            "N": RuntimeArgument.scalar(8, "int32"),
        },
        grid=(1,),
        schedule={"VALID": 8, "BLOCK": 128},
    )
    source_compile = CompileResult(
        True,
        ttir=GOOD_TTIR,
        python_interface={
            "arg_names": ["X", "Y", "N", "VALID", "BLOCK"],
            "constexprs": ["VALID", "BLOCK"],
            "constexpr_indices": [3, 4],
        },
    )
    replay_case = _replay_case(artifact, source_case, source_compile)
    assert replay_case.schedule == {"p3": 8, "c0": 128}
    assert artifact.schedule.values == {"c0": 128}
