from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.codegen import UnsafeReplayCodeError
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.diagnose import sanitize_replay_diagnostic
from intentir.verify.gate1.interventions import InterventionStatus
from intentir.verify.gate1.compare import compare_ttir
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    Frontend,
    Gate1Artifact,
    Gate1Status,
    KernelInterface,
    KernelParameter,
    LocationMap,
    ParameterKind,
    ParameterMode,
    ReplayPlan,
    ReplayStep,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
)
from intentir.verify.gate1.orchestrate import (
    CandidateRecord,
    Gate1Config,
    Gate1Runner,
    ReliftTrigger,
    _validate_compiled_interface,
)
from intentir.verify.gate1.pi_ablations import _case_with_artifact_schedule
from intentir.verify.gate1.replay import (
    ReplayGeneration,
    StaticReplayWriter,
    _augment_prompt,
    restore_masked_replay_module,
)
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate1.ttir import fingerprint_ttir


GOOD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

BAD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %one = arith.constant 1.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    %wrong = arith.addf %value, %one : f32
    tt.store %out, %wrong : !tt.ptr<f32>
    tt.return
  }
}
'''

ADD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    %sum = arith.addf %value, %value : f32
    tt.store %out, %sum : !tt.ptr<f32>
    tt.return
  }
}
'''

SUB_TTIR = ADD_TTIR.replace("arith.addf", "arith.subf")

SPECIALIZED_SCALAR_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

GOOD_CODE = '''import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n, BLOCK: tl.constexpr):
    return
'''

REAL_COPY_CODE = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["n"])
def kernel(x, out, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    value = tl.load(x + offset, mask=mask, other=0.0)
    tl.store(out + offset, value, mask=mask)
'''

REAL_MASKED_FAILURE_CODE = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    offset = tl.program_id(0) * scrc_p3 + tl.arange(0, scrc_p3)
    mask = offset < scrc_p2
    value = tl.gate1_masked_ablation_unknown(scrc_p0 + offset)
    tl.store(scrc_p1 + offset, value, mask=mask)
'''

OPAQUE_MASKED_COMPILE_FAIL = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    "MASKED_FAIL"
    return
'''

OPAQUE_MASKED_SEMANTIC_DRIFT = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    "BAD"
    return
'''


def _artifact() -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "identity_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "out",
                    "operands": ["x"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    return Gate1Artifact(
        j=SemanticArchive(
            "kernel",
            intent,
            {"identity_node": "body[0]"},
            Boundary(
                domains={"lane_domain": "lane < N"},
                accesses=[
                    BoundaryAccess(
                        "load_x",
                        AccessKind.LOAD,
                        "x",
                        predicate="lane < N",
                        fill_value=0.0,
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                    BoundaryAccess(
                        "store_out",
                        AccessKind.STORE,
                        "out",
                        predicate="lane < N",
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                ],
            ),
        ),
        interface=KernelInterface(
            kernel_id="kernel",
            function_name="kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "x",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    1,
                    tensor_ref="x",
                ),
                KernelParameter(
                    "out",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "f32",
                    1,
                    tensor_ref="out",
                ),
                KernelParameter(
                    "n", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
                ),
                KernelParameter("BLOCK", ParameterKind.CONSTEXPR),
            ],
            grid=["ceil_div(n, BLOCK)"],
            compiler_slots=["num_warps"],
            do_not_specialize=["n"],
        ),
        schedule=ScheduleSpec(
            slots=[
                ScheduleSlot("BLOCK", ScheduleSlotKind.CONSTEXPR),
                ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
            ],
            values={"BLOCK": 64, "num_warps": 4},
        ),
        plan=ReplayPlan(
            [
                ReplayStep("bind_program_domain", refs=["lane_domain"]),
                ReplayStep("bind_runtime_scalar", refs=["n"]),
                ReplayStep("declare_schedule_slot", refs=["BLOCK"]),
                ReplayStep("declare_schedule_slot", refs=["num_warps"]),
                ReplayStep("materialize_index_tile", refs=["load_x"]),
                ReplayStep("realize_masked_access", refs=["load_x"]),
                ReplayStep("realize_cast", node_ref="identity_node"),
                ReplayStep("materialize_index_tile", refs=["store_out"]),
                ReplayStep("realize_masked_access", refs=["store_out"]),
            ]
        ),
        location_map=LocationMap("private.py", []),
        source="SOURCE_CANARY_91F7",
    )


def _case() -> CompileCase:
    return CompileCase(
        arguments={
            "x": RuntimeArgument.tensor((65,), "float32"),
            "out": RuntimeArgument.tensor((65,), "float32"),
            "n": RuntimeArgument.scalar(65, "int32"),
        },
        grid=(2,),
        schedule={"BLOCK": 64, "num_warps": 4},
    )


class FakeCompiler:
    def __init__(self, *, target: str = "sm_test") -> None:
        self.target = target
        self.calls: list[str] = []

    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        self.calls.append(code)
        if "COMPILE_FAIL" in code or "MASKED_FAIL" in code:
            return CompileResult(ok=False, diagnostic='File "/secret/worker.py", line 7: bad replay')
        if "REPLAY_SUB" in code:
            ttir = SUB_TTIR
        elif "SOURCE_ADD" in code or "REPLAY_ADD" in code:
            ttir = ADD_TTIR
        else:
            ttir = BAD_TTIR if "BAD" in code else GOOD_TTIR
        return CompileResult(
            ok=True,
            ttir=ttir,
            ttgir="ttgir",
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": self.target},
        )


class CaptureWriter:
    def __init__(
        self,
        normal: list[str],
        *,
        masked: str = OPAQUE_MASKED_COMPILE_FAIL,
        repeated_id: bool = False,
    ) -> None:
        self.normal = normal
        self.masked = masked
        self.repeated_id = repeated_id
        self.normal_index = 0
        self.masked_index = 0
        self.calls: list[tuple[ReplayGeneration | None, bool]] = []
        self.artifacts: list[Gate1Artifact] = []

    def generate(
        self,
        artifact: Gate1Artifact,
        registry,
        *,
        base_candidate: ReplayGeneration | None = None,
        masked: bool = False,
        plan_override: ReplayPlan | None = None,
        ablation_kind: str | None = None,
    ) -> ReplayGeneration:
        self.calls.append((base_candidate, masked))
        self.artifacts.append(artifact)
        if masked:
            code = self.masked
            index = self.masked_index
            self.masked_index += 1
            generation_id = f"masked:{index}"
        else:
            index = self.normal_index
            code = self.normal[min(index, len(self.normal) - 1)]
            self.normal_index += 1
            generation_id = "reused" if self.repeated_id else f"normal:{index}"
        digest = hashlib.sha256(code.encode()).hexdigest()
        return ReplayGeneration(
            code=code,
            prompt_sha256="prompt",
            requested_model="capture",
            generation_id=generation_id,
            code_sha256=digest,
            response_model="capture",
            masked=masked,
            base_generation_id=(
                None if base_candidate is None else base_candidate.generation_id
            ),
            ablation_kind=ablation_kind,
        )


def test_controlled_replay_prompt_preserves_unchanged_strict_pi_steps() -> None:
    baseline = ReplayGeneration(
        code="MODEL_GENERATED_BASELINE",
        prompt_sha256="baseline-prompt",
        requested_model="capture",
        generation_id="normal:0",
        code_sha256="baseline-code",
    )
    prompt = _augment_prompt(
        "CURRENT_PACKET",
        base_candidate=baseline,
    )

    assert "MODEL_GENERATED_BASELINE" in prompt
    assert "complete pi is the construction contract" in prompt
    assert "do not introduce redundant predicates" in prompt
    assert "ALLOWLISTED_REPAIR_FEEDBACK" not in prompt


def test_pi_ablation_compile_case_uses_the_frozen_artifact_schedule() -> None:
    unscheduled = replace(_case(), schedule={})
    scheduled = _case_with_artifact_schedule(unscheduled, _artifact())
    assert scheduled.schedule == {"BLOCK": 64, "num_warps": 4}

    conflicting = replace(_case(), schedule={"BLOCK": 32, "num_warps": 4})
    with pytest.raises(ValueError, match="differs from the frozen artifact"):
        _case_with_artifact_schedule(conflicting, _artifact())


def _runner(**config) -> Gate1Runner:
    config.setdefault("run_semantic_interventions", False)
    return Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(**config),
    )


def _source(function: str = "kernel") -> KernelSource:
    return KernelSource.from_code("SOURCE", function, trusted=True)


def _binary_artifact() -> Gate1Artifact:
    artifact = _artifact()
    intent = IntentFunction.from_json(
        {
            "name": "binary_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "map_binary",
                    "result": "out",
                    "operands": ["x", "x"],
                    "attrs": {"fn": "add"},
                }
            ],
        }
    )
    return replace(
        artifact,
        j=replace(artifact.j, intent=intent),
        plan=ReplayPlan(
            [
                replace(step, rule="realize_elementwise")
                if step.rule == "realize_cast"
                else step
                for step in artifact.plan.steps
            ]
        ),
    )


def test_static_masked_writer_preserves_opaque_evidence_for_runner() -> None:
    writer = StaticReplayWriter(GOOD_CODE, masked=REAL_MASKED_FAILURE_CODE)
    generation = writer.generate(
        _artifact(), default_triton_rule_registry(), masked=True
    )

    assert generation.masked
    assert "def scrc_k0(scrc_p0, scrc_p1, scrc_p2" in generation.code
    restored = restore_masked_replay_module(generation.code, _artifact())
    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in restored
    assert 'do_not_specialize=[\'n\']' in restored


def test_paired_intervention_is_the_deterministic_anti_leakage_basis() -> None:
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"])
    result = _runner(
        stable_closures=1, run_semantic_interventions=True
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.anti_leakage_basis == "paired_intervention"
    assert len(result.interventions) == 1
    assert result.interventions[0].status is InterventionStatus.PASS
    assert [item.j.intent.body[0].attrs["fn"] for item in writer.artifacts[:2]] == [
        "add",
        "sub",
    ]
    assert writer.artifacts[0].plan.to_json() == writer.artifacts[1].plan.to_json()
    assert writer.artifacts[0].schedule.to_json() == (
        writer.artifacts[1].schedule.to_json()
    )
    assert [masked for _, masked in writer.calls] == [False, False, True]
    assert writer.calls[1][0] is not None
    assert writer.calls[1][0].generation_id == "normal:0"
    assert result.interventions[0].generation is not None
    assert result.interventions[0].generation.base_generation_id == "normal:0"


def test_paired_intervention_rejects_a_writer_that_ignores_changed_j() -> None:
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_ADD"])
    result = _runner(
        stable_closures=1, run_semantic_interventions=True
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert result.anti_leakage_basis == "paired_intervention"
    assert result.interventions[0].status is InterventionStatus.IGNORED_J
    assert [masked for _, masked in writer.calls] == [False, False]


def test_masked_abi_restore_renames_model_locals_that_collide() -> None:
    opaque = '''import triton
import triton.language as tl

@triton.jit
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    x = tl.load(scrc_p0)
    tl.static_assert(True, "x", scrc_p0=1)
    tl.store(scrc_p1, x)
'''
    restored = restore_masked_replay_module(opaque, _artifact())

    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in restored
    assert "scrc_local_0 = tl.load(x)" in restored
    assert "tl.static_assert(True, 'x', scrc_p0=1)" in restored
    assert "tl.store(out, scrc_local_0)" in restored


def test_masked_abi_restore_rejects_a_wrong_public_signature() -> None:
    wrong_signature = REAL_MASKED_FAILURE_CODE.replace(
        "scrc_p2, scrc_p3: tl.constexpr",
        "scrc_p3: tl.constexpr, scrc_p2",
    )

    with pytest.raises(UnsafeReplayCodeError, match="opaque ABI"):
        restore_masked_replay_module(wrong_signature, _artifact())


def test_failed_replay_relifts_then_restarts_stability_and_ablation() -> None:
    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(["BAD", "GOOD", "GOOD"])
    relifter = RecordingRelifter()
    runner = _runner(stable_closures=2, max_relifts=2)
    result = runner.run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert len(result.candidates) == 3
    assert [item.comparison is not None and item.comparison.hard_pass for item in result.candidates] == [False, True, True]
    assert relifter.requests[0].trigger is ReliftTrigger.NORMAL_NO_CLOSURE
    assert [item.relift_round for item in result.candidates] == [0, 1, 1]
    assert writer.calls[:3] == [(None, False), (None, False), (None, False)]
    assert writer.calls[3][1]
    assert runner.compiler.calls[:5] == [
        "SOURCE",
        "BAD",
        "SOURCE",
        "GOOD",
        "GOOD",
    ]
    assert result.masked_compile is not None and not result.masked_compile.ok
    assert result.masked_generation is not None
    assert result.masked_generation.masked
    assert result.to_json()["masked_generation"]["generation_id"].startswith(
        "masked:"
    )
    assert result.leakage_ratio == 0.0


def test_runner_restores_opaque_masked_code_and_rejects_real_leakage() -> None:
    class ForbiddenLeakageRelifter:
        def relift(self, *args, **kwargs):
            raise AssertionError(
                "masked-J leakage is not a normal source/replay relift trigger"
            )

    compiler = FakeCompiler()
    writer = CaptureWriter(["GOOD"], masked=REAL_MASKED_FAILURE_CODE)
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            max_relifts=2,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=ForbiddenLeakageRelifter(),
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert result.leakage_ratio == 1.0
    assert result.masked_generation is not None
    assert "scrc_p0" in result.masked_generation.code
    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in compiler.calls[-1]


def test_masked_fallback_uses_exact_semantic_degradation_not_score_threshold() -> None:
    writer = CaptureWriter(
        ["GOOD"], masked=OPAQUE_MASKED_SEMANTIC_DRIFT
    )
    result = Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
            leakage_threshold=0.0,
        ),
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.anti_leakage_basis == "masked_j_fallback"
    assert result.masked_comparison is not None
    assert not result.masked_comparison.hard_pass
    assert result.leakage_ratio is not None and result.leakage_ratio > 0.0
    assert any(
        "deterministically lost semantic hard component" in diagnostic
        for diagnostic in result.diagnostics
    )
    assert any(
        "score ratio=" in diagnostic and "diagnostic only" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_compile_failure_has_no_replay_side_retry() -> None:
    writer = CaptureWriter(["COMPILE_FAIL"])
    result = _runner(stable_closures=1, run_masked_ablation=False).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert len(result.candidates) == 1
    assert len(writer.calls) == 1
    assert result.best_candidate is not None
    assert result.best_candidate.diagnosis is not None
    assert result.best_candidate.diagnosis.mismatches == ("bad replay",)


def test_compile_diagnostic_guides_lift_without_replay_source() -> None:
    summary = sanitize_replay_diagnostic(
        '''Traceback (most recent call last):
  File "/secret/worker.py", line 7
    output_value = tl.load(error_ptr)
                   ^
NameError: output_value is not defined'''
    )

    assert summary == "NameError: output_value is not defined"
    assert "/secret/worker.py" not in summary
    assert "tl.load" not in summary


def test_semantic_relift_failure_is_audited_as_no_closure() -> None:
    class FailingRelifter:
        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            assert source_fingerprint is not None
            assert history
            assert request.trigger is ReliftTrigger.NORMAL_NO_CLOSURE
            assert request.hard_difference_categories
            raise RuntimeError("relift provider unavailable")

    writer = CaptureWriter(["BAD"])
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=FailingRelifter(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert len(result.candidates) == 1
    assert any(
        "semantic relift failed: RuntimeError: relift provider unavailable"
        in diagnostic
        for diagnostic in result.diagnostics
    )


def test_replay_policy_failure_triggers_fresh_source_aware_relift() -> None:
    class PolicyRejectingWriter:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            assert kwargs.get("base_candidate") is None
            if self.calls == 1:
                raise UnsafeReplayCodeError("libdevice call is not allowlisted")
            code = GOOD_CODE
            digest = hashlib.sha256(code.encode()).hexdigest()
            return ReplayGeneration(
                code=code,
                prompt_sha256="prompt",
                requested_model="capture",
                generation_id=f"normal:{self.calls}",
                code_sha256=digest,
            )

    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.requests.append(request)
            assert history[-1].generation is None
            return artifact

    writer = PolicyRejectingWriter()
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert len(result.candidates) == 2
    assert result.candidates[0].comparison is None
    assert relifter.requests[0].trigger is ReliftTrigger.REPLAY_GENERATION_FAILED
    assert writer.calls == 2


def test_outer_relift_round_replays_the_revised_artifact() -> None:
    class RecordingRelifter:
        calls = 0
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            assert source_fingerprint is not None
            assert [item.relift_round for item in history] == [0]
            assert history[0].comparison is not None
            assert history[0].comparison.differences
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(["BAD", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert relifter.calls == 1
    assert len(relifter.requests) == 1
    request = relifter.requests[0]
    assert request.trigger is ReliftTrigger.NORMAL_NO_CLOSURE
    assert request.evaluation_round == 0
    assert request.candidate_count == 1
    assert request.compiled_candidate_count == 1
    assert request.best_generation_round == 0
    assert request.hard_difference_categories == (
        "op_multiset",
        "dataflow",
    )
    assert [item.relift_round for item in result.candidates] == [0, 1]


def test_compile_failure_relifts_and_never_reuses_failed_replay_as_base() -> None:
    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(["COMPILE_FAIL", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert relifter.requests[0].trigger is ReliftTrigger.REPLAY_COMPILE_FAILED
    assert writer.calls == [(None, False), (None, False)]


def test_reused_generation_id_cannot_satisfy_stability() -> None:
    writer = CaptureWriter(["GOOD"], repeated_id=True)
    result = _runner(stable_closures=2).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert any(
        candidate.diagnosis
        and "generation_id" in " ".join(candidate.diagnosis.mismatches)
        for candidate in result.candidates
    )


def test_missing_masked_generation_fails_closed() -> None:
    writer = StaticReplayWriter(GOOD_CODE)
    result = _runner().run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert any("ablation was unavailable" in item for item in result.diagnostics)


def test_declared_function_and_alias_contracts_are_rejected_before_writer() -> None:
    writer = CaptureWriter(["GOOD"])
    function_mismatch = _runner().run(
        _source("other"),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )
    assert function_mismatch.status is Gate1Status.LIFT_REJECTED
    assert writer.calls == []

    artifact = _artifact()
    aliased = replace(
        artifact,
        interface=replace(
            artifact.interface,
            parameters=[
                artifact.interface.parameters[0],
                replace(artifact.interface.parameters[1], alias_of="x"),
                *artifact.interface.parameters[2:],
            ],
        ),
    )
    writer2 = CaptureWriter(["GOOD"])
    alias_result = _runner().run(
        _source(), aliased, _case(), default_triton_rule_registry(), writer2
    )
    assert alias_result.status is Gate1Status.LIFT_REJECTED
    assert writer2.calls == []


def test_rule_registry_is_versioned_and_has_a_stable_digest() -> None:
    first = default_triton_rule_registry()
    second = default_triton_rule_registry()
    assert first.digest == second.digest
    assert all(card.version for card in first.cards)
    assert {
        "bind_program_domain",
        "bind_wide_program_domain",
        "materialize_index_tile",
        "materialize_pointer_tile",
        "realize_masked_access",
        "materialize_mask_tile",
        "realize_elementwise",
        "realize_reduce",
        "realize_flattened_reduce",
        "realize_k_tiled_dot",
        "realize_outer_product",
    } <= {card.id for card in first.cards}
    cards = {card.id: card for card in first.cards}
    assert {
        card.id for card in first.cards if card.selection_mode.value == "override"
    } == {
        "bind_wide_program_domain",
        "materialize_pointer_tile",
        "materialize_mask_tile",
        "realize_flattened_reduce",
    }
    assert all(
        card.override_family is not None
        for card in first.cards
        if card.selection_mode.value == "override"
    )
    assert cards["realize_k_tiled_dot"].selection_mode.value == "derived"
    assert cards["materialize_pointer_tile"].version == "2"
    assert any(
        "coordinate components" in item
        for item in cards["materialize_pointer_tile"].pitfalls
    )
    assert cards["realize_masked_access"].version == "3"
    assert cards["realize_masked_access"].api_mapping["vector_conjunction"] == "&"
    assert cards["realize_elementwise"].version == "4"
    assert cards["realize_elementwise"].api_mapping["recip"] == "1.0 / operand"
    assert cards["realize_elementwise"].api_mapping["abs"] == "tl.abs"
    assert "libdevice.pow" in cards["realize_elementwise"].api_mapping["pow"]
    assert cards["realize_reduce"].version == "4"
    assert "tile_axis" in cards["realize_reduce"].reads
    assert "semantic_extent" in cards["realize_reduce"].api_mapping["mean"]
    assert "tl.reduce" in cards["realize_reduce"].api_mapping["prod"]
    assert "multiply combiner" in cards["realize_reduce"].api_mapping["prod"]
    assert cards["bind_wide_program_domain"].api_mapping[
        "program_coordinate_i64"
    ].endswith("to(tl.int64)")
    assert cards["realize_flattened_reduce"].api_mapping["sum"] == "tl.sum(tile)"
    assert cards["bind_runtime_scalar"].version == "2"
    assert "tensor_ref" in cards["bind_runtime_scalar"].reads
    assert cards["declare_schedule_slot"].version == "2"
    assert any(
        "compiler slot is not a kernel parameter" in item
        for item in cards["declare_schedule_slot"].pitfalls
    )
    assert "tl.sum" in first.api_tokens(Frontend.TRITON)
    assert "+" not in first.api_tokens(Frontend.TRITON)
    assert "*" not in first.api_tokens(Frontend.TRITON)
    assert cards["realize_k_tiled_dot"].version == "3"
    assert "result dtype" in cards["realize_k_tiled_dot"].reads
    assert cards["realize_k_tiled_dot"].api_mapping["float32"] == "tl.float32"
    assert cards["realize_outer_product"].api_mapping["combine"] == "*"
    assert "subscripts" in cards["realize_outer_product"].reads
    assert cards["materialize_mask_tile"].api_mapping[
        "vector_conjunction"
    ] == "&"
    assert any(
        "Do not recompute component comparisons" in item
        for item in cards["materialize_mask_tile"].pitfalls
    )
    assert cards["realize_broadcast"].api_mapping["scalar_to_tile"] == (
        "use the scalar operand directly"
    )
    assert any(
        "redundant domain facts" in item
        for item in cards["realize_masked_access"].pitfalls
    )


def test_compiled_interface_allows_specialized_nonbinding_scalars_only() -> None:
    artifact = _artifact()
    python_interface = {
        "arg_names": ["x", "out", "n", "BLOCK"],
        "constexprs": ["BLOCK"],
        "constexpr_indices": [3],
    }
    result = CompileResult(
        ok=True,
        ttir=SPECIALIZED_SCALAR_TTIR,
        python_interface=python_interface,
    )
    fingerprint = fingerprint_ttir(
        SPECIALIZED_SCALAR_TTIR,
        python_interface=python_interface,
    )
    _validate_compiled_interface(result, fingerprint, artifact)

    semantic_intent = IntentFunction.from_json(
        {
            "name": "semantic_scalar",
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": ["N"]},
                {"name": "eps", "dtype": "f32", "shape": []},
            ],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "out",
                    "operands": ["x"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    semantic = replace(
        artifact,
        j=replace(artifact.j, intent=semantic_intent),
        interface=replace(
            artifact.interface,
            parameters=[
                *artifact.interface.parameters[:2],
                replace(
                    artifact.interface.parameters[2],
                    dtype="f32",
                    tensor_ref="eps",
                ),
                artifact.interface.parameters[3],
            ],
        ),
    )
    with pytest.raises(ValueError, match="omitted binding runtime argument.*n"):
        _validate_compiled_interface(result, fingerprint, semantic)


def test_best_candidate_rank_keeps_interface_ahead_of_semantic_score() -> None:
    interface_mismatch = compare_ttir(
        GOOD_TTIR,
        GOOD_TTIR.replace("%x: !tt.ptr<f32>", "%x: !tt.ptr<f16>"),
    )
    semantic_mismatch = compare_ttir(GOOD_TTIR, BAD_TTIR)
    compiled = CompileResult(ok=True, ttir=GOOD_TTIR)
    interface_record = CandidateRecord(
        0, 0, None, compiled, interface_mismatch, None
    )
    semantic_record = CandidateRecord(
        0, 1, None, compiled, semantic_mismatch, None
    )

    assert interface_mismatch.score > semantic_mismatch.score
    assert semantic_record.rank_tuple > interface_record.rank_tuple


def test_real_triton_runner_closes_without_launching(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Gate1 compile integration")

    writer = StaticReplayWriter(
        REAL_COPY_CODE,
        masked=REAL_MASKED_FAILURE_CODE,
    )
    runner = Gate1Runner(
        compiler=TritonCompiler(cache_dir=tmp_path, timeout=180.0),
            config=Gate1Config(run_semantic_interventions=False),
    )
    result = runner.run(
        KernelSource.from_code(REAL_COPY_CODE, "kernel", trusted=True),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.source_compile is not None and result.source_compile.ok
    assert result.best_candidate is not None
    assert result.best_candidate.comparison is not None
    assert result.best_candidate.comparison.hard_pass
    assert result.masked_compile is not None and not result.masked_compile.ok
