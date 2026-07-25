from __future__ import annotations

from dataclasses import dataclass
import re
from threading import Event

import numpy as np
import pytest

from direct_fixtures import make_reduce_artifact
from intentir.extraction.lifters import scrc
from intentir.extraction.lifters.scrc import (
    ResponsesSCRCLiftAdapter,
    SCRCExtraction,
    SCRCHarnessBinding,
    SCRCLiftAttempt,
    SCRCLiftError,
    SCRCProvenance,
)
from intentir.verify.adapters import (
    BoundGate1Adapter,
    TritonGate2ExecutionAdapter,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.model import Gate1Status, JNecessityStatus
from intentir.verify.gate1.orchestrate import (
    Gate1ArmEvidence,
    Gate1Config,
    Gate1ExecutionResult,
    Gate1Runner,
)
from intentir.verify.gate1.pi import RealizationWitness
from intentir.verify.gate1.replay import ReplayGeneration, StaticReplayWriter
from intentir.verify.gate2.model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
)
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from intentir.verify.gate2.scalar import ScalarLoweringSpec
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)
from intentir.verify.scrc_runner import (
    CandidateLiftError,
    FeedbackKind,
    Gate1SearchStatus,
    HarnessBindingInstance,
    LiftCallResult,
    LiftFeedback,
    MainStatus,
    SCRCGateRunner,
    SCRCInfrastructureError,
    SCRCRunDependencies,
    SCRCRunRequest,
    SourceProgram,
    _compact_gate1_observer_diff,
)
from intentir.verify.typecheck import TypeCheckStatus


@dataclass(frozen=True)
class _Comparison:
    hard_pass: bool

    def to_dict(self):
        return {
            "hard_pass": self.hard_pass,
            "differences": [] if self.hard_pass else [{"kind": "test_diff"}],
        }


def _gate1_result(
    *,
    closed: bool,
    masked_closed: bool | None = None,
    compile_failed: bool = False,
) -> Gate1ExecutionResult:
    compile_result = CompileResult(
        not compile_failed,
        ttir="module {}" if not compile_failed else None,
        diagnostic="source compiler unavailable" if compile_failed else "",
    )
    generation = ReplayGeneration(
        "def scrc_k0():\n    pass\n",
        "scrc_k0",
        "deterministic-test",
        None,
        "trace-only",
        "trace-only",
    )
    normal = Gate1ArmEvidence(
        "normal",
        generation,
        compile_result,
        None,
        _Comparison(closed),  # type: ignore[arg-type]
        None if closed else "observer mismatch",
    )
    masked = None
    if masked_closed is not None:
        masked = Gate1ArmEvidence(
            "masked_j",
            generation,
            compile_result,
            None,
            _Comparison(masked_closed),  # type: ignore[arg-type]
        )
    status = (
        Gate1Status.COMPILE_FAILED
        if compile_failed
        else Gate1Status.REPLAY_CLOSED
        if closed
        else Gate1Status.NO_CLOSURE
    )
    return Gate1ExecutionResult(
        status,
        "trace-only",
        compile_result,
        None,
        normal,
        masked,
        0 if compile_failed else 1,
        int(masked is not None),
    )


def _gate2_suite(*, passed: bool) -> Gate2SuiteResult:
    status = Gate2Status.PASS if passed else Gate2Status.PAIRWISE_DIVERGED
    output = ExecutionResult("source", True, {})
    case = Gate2Result(
        "case0",
        status,
        output,
        ExecutionResult("replay", True, {}),
        ExecutionResult("j", True, {}),
    )
    return Gate2SuiteResult("suite0", (case,), (("c0", 128),))


def test_gate1_relift_diff_is_bounded_actionable_and_has_no_node_ids() -> None:
    class VerboseComparison:
        def to_dict(self):
            return {
                "hard_pass": False,
                "hard_passes": {"dataflow": False, "op_multiset": False},
                "primary_difference_category": "op_multiset",
                "score": 0.25,
                "components": [{"duplicated": True}],
                "differences": [
                    {
                        "category": "dataflow",
                        "kind": "missing",
                        "item": [
                            "edge",
                            (
                                f"region[scf.for:{index:020x}]"
                                if index == 0
                                else f"tt.addptr[{index:020x}]"
                            ),
                            (
                                f"effect.trace[{index + 1:020x}]"
                                if index == 0
                                else "operand[0]"
                            ),
                        ],
                        "expected_count": 1,
                        "actual_count": 0,
                        "message": "not forwarded",
                    }
                    for index in range(12)
                ]
                + [
                    {
                        "category": "op_multiset",
                        "kind": "missing",
                        "item": "arith.extsi",
                        "expected_count": 3,
                        "actual_count": 0,
                    }
                ],
            }

    compact = _compact_gate1_observer_diff(VerboseComparison())

    assert set(compact) == {
        "hard_pass",
        "hard_passes",
        "primary_difference_category",
        "categories",
    }
    dataflow = next(
        item for item in compact["categories"] if item["category"] == "dataflow"
    )
    assert dataflow["difference_count"] == 12
    assert len(dataflow["examples"]) == 8
    assert dataflow["truncated_count"] == 4
    assert "00000000000000000000" not in str(compact)
    assert dataflow["examples"][0]["item"][1] == "region[scf.for:node]"
    assert dataflow["examples"][0]["item"][2] == "effect.trace[node]"
    assert re.search(r"[0-9a-f]{12,}", str(compact)) is None
    assert "score" not in compact
    assert "components" not in compact


class _Lift:
    def __init__(self, artifacts, events=None):
        self.artifacts = list(artifacts)
        self.feedbacks = []
        self.events = events

    def lift(self, request, feedback):
        del request
        self.feedbacks.append(feedback)
        if self.events is not None:
            self.events.append("lift")
        item = self.artifacts.pop(0)
        if isinstance(item, Exception):
            raise item
        return LiftCallResult(item)


class _Gate1Adapter:
    def __init__(self, result, events=None):
        self.result = result
        self.events = events

    def run(self, artifact, request):
        del artifact, request
        if self.events is not None:
            self.events.append("gate1")
        return self.result


class _Gate1Factory:
    def __init__(self, results, events=None):
        self.results = list(results)
        self.rounds = []
        self.instances = []
        self.events = events

    def __call__(self, formal_round):
        self.rounds.append(formal_round)
        adapter = _Gate1Adapter(self.results.pop(0), self.events)
        self.instances.append(adapter)
        return adapter


class _Gate2:
    def __init__(self, *, passed=True, events=None):
        self.passed = passed
        self.calls = 0
        self.events = events

    def execute(self, artifact, gate1, request):
        del artifact, gate1, request
        self.calls += 1
        if self.events is not None:
            self.events.append("gate2_inputs_and_execution")
        return _gate2_suite(passed=self.passed)


def _request(*, max_pre_replay_rejections=3) -> SCRCRunRequest:
    return SCRCRunRequest(
        source=SourceProgram("trusted-source", "trusted-gate-source"),
        binding=HarnessBindingInstance.create(
            {"c0": 128}, gate1_case="same-c0-case"
        ),
        frontend="triton",
        max_pre_replay_rejections=max_pre_replay_rejections,
    )


def _runner(lift, factory, gate2):
    return SCRCGateRunner(SCRCRunDependencies(lift, factory, gate2))


def test_typecheck_failure_relifts_without_entering_a_formal_round() -> None:
    invalid = make_reduce_artifact()
    payload = invalid.pi.to_json()
    payload["instances"][0]["semantic_ref"] = "n99"
    invalid.pi = RealizationWitness.from_json(payload)
    valid = make_reduce_artifact()
    lift = _Lift([invalid, valid])
    factory = _Gate1Factory([_gate1_result(closed=True)])
    gate2 = _Gate2()

    report = _runner(lift, factory, gate2).run(_request())

    assert report.main_status is MainStatus.SCRC_CERTIFIED
    assert report.call_budget.lift_calls == 2
    assert report.call_budget.formal_gate1_rounds == 1
    assert factory.rounds == [1]
    assert gate2.calls == 1
    assert lift.feedbacks[0] is None
    assert lift.feedbacks[1].kind is FeedbackKind.TYPECHECK
    assert lift.feedbacks[1].previous_artifact is report.attempts[0].artifact
    assert report.attempts[0].formal_round is None
    assert report.attempts[0].artifact.j.to_json() != invalid.j.to_json()


def test_unparseable_lift_is_recorded_without_entering_typecheck_or_replay() -> None:
    lift = _Lift(
        [
            CandidateLiftError({"diagnostic": "closed schema parse failed"}),
            make_reduce_artifact(),
        ]
    )
    factory = _Gate1Factory([_gate1_result(closed=True)])
    gate2 = _Gate2()

    report = _runner(lift, factory, gate2).run(_request())

    assert report.main_status is MainStatus.SCRC_CERTIFIED
    assert report.attempts[0].typecheck_status.value == "not_run"
    assert report.call_budget.typecheck_calls == 1
    assert lift.feedbacks[1].kind is FeedbackKind.LIFT
    assert factory.rounds == [1]


def test_responses_lift_adapter_forwards_structured_feedback_only_to_relift(
    monkeypatch,
) -> None:
    artifact = make_reduce_artifact()
    contexts = []

    def fake_extract(source, **kwargs):
        assert source == "trusted-source"
        contexts.append(kwargs["relift_context"])
        return SCRCExtraction(
            artifact,
            SCRCProvenance("test-model", "trace", attempts=2),
            SCRCHarnessBinding((), "kernel"),
            source,
        )

    monkeypatch.setattr(scrc, "extract_scrc_artifact", fake_extract)
    adapter = ResponsesSCRCLiftAdapter()
    request = _request()
    initial = adapter.lift(request, None)
    feedback = LiftFeedback(
        FeedbackKind.GATE1,
        {"observer_diff": {"category": "dataflow"}},
        1,
        artifact,
    )
    relifted = adapter.lift(request, feedback)

    assert initial.physical_calls == relifted.physical_calls == 2
    assert initial.metadata["artifact_schema"] == "intentir.scrc.artifact.v6"
    assert initial.metadata["decode_protocol"] == "intentir.scrc.decode.v7"
    assert initial.metadata["typecheck_protocol"] == "intentir.scrc.typecheck.v15"
    assert contexts[0] is None
    assert contexts[1].failure_kind == "gate1"
    assert contexts[1].diagnostics == feedback.diagnostics
    assert contexts[1].previous_artifact == artifact.to_json()


def test_responses_lift_adapter_preserves_rejected_provider_evidence(
    monkeypatch,
) -> None:
    rejected = SCRCLiftAttempt(
        model="test-model",
        diagnostic="ValueError: invalid anonymous handle",
        response_text='{"invalid": true}',
        response_id="response-1",
        response_model="test-model-version",
        usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        request_sha256="request-trace",
        response_sha256="response-trace",
        raw_response_sha256="raw-response-trace",
        attempts=1,
        elapsed_ms=12.5,
    )

    def fake_extract(source, **kwargs):
        del source, kwargs
        raise SCRCLiftError(rejected)

    monkeypatch.setattr(scrc, "extract_scrc_artifact", fake_extract)
    adapter = ResponsesSCRCLiftAdapter()

    with pytest.raises(CandidateLiftError):
        adapter.lift(_request(), None)

    assert len(adapter.records) == 1
    record = adapter.records[0]
    assert record.extraction is None
    assert record.rejected_attempts == (rejected,)
    assert "invalid anonymous handle" in (record.diagnostic or "")


def test_each_gate1_retry_uses_a_fresh_factory_and_only_lift_gets_diff() -> None:
    artifacts = [make_reduce_artifact() for _ in range(3)]
    lift = _Lift(artifacts)
    factory = _Gate1Factory(
        [
            _gate1_result(closed=False),
            _gate1_result(closed=False),
            _gate1_result(closed=True),
        ]
    )
    gate2 = _Gate2()

    report = _runner(lift, factory, gate2).run(_request())

    assert report.main_status is MainStatus.SCRC_CERTIFIED
    assert factory.rounds == [1, 2, 3]
    assert len({id(item) for item in factory.instances}) == 3
    assert [item.kind if item is not None else None for item in lift.feedbacks] == [
        None,
        FeedbackKind.GATE1,
        FeedbackKind.GATE1,
    ]
    assert "observer_diff" in lift.feedbacks[1].diagnostics
    assert lift.feedbacks[1].previous_artifact is report.attempts[0].artifact
    assert gate2.calls == 1


def test_three_gate1_failures_end_as_no_closure_without_gate2() -> None:
    lift = _Lift([make_reduce_artifact() for _ in range(3)])
    factory = _Gate1Factory([_gate1_result(closed=False) for _ in range(3)])
    gate2 = _Gate2()

    report = _runner(lift, factory, gate2).run(_request())

    assert report.main_status is MainStatus.NO_CLOSURE
    assert report.call_budget.formal_gate1_rounds == 3
    assert report.accepted_attempt is None
    assert gate2.calls == 0


def test_source_observer_gap_stops_branch_without_misleading_relift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lift = _Lift([make_reduce_artifact(), make_reduce_artifact()])
    source_compile = CompileResult(True, ttir="module {}")
    observer_failure = Gate1ExecutionResult(
        Gate1Status.NO_CLOSURE,
        "trace-only",
        source_compile,
        None,
        Gate1ArmEvidence(
            "normal",
            None,
            None,
            None,
            None,
            (
                "observer unsupported: source TTIR: TTIRParseError: "
                "multi-block CFG is not supported"
            ),
        ),
        None,
        0,
        0,
    )
    factory = _Gate1Factory(
        [
            observer_failure,
            _gate1_result(closed=True),
        ]
    )
    gate2 = _Gate2()

    def forbidden_feedback(*args, **kwargs):
        del args, kwargs
        pytest.fail("source observer unsupported must not construct relift feedback")

    monkeypatch.setattr(SCRCGateRunner, "_gate1_feedback", forbidden_feedback)

    search = _runner(lift, factory, gate2).search_gate1(_request())

    assert search.status is Gate1SearchStatus.NO_CLOSURE
    assert search.stop_reason == "observer_unsupported"
    assert search.stop_observed_at == "after_gate1"
    assert len(search.attempts) == 1
    attempt = search.attempts[0]
    assert attempt.artifact is not None
    assert attempt.typecheck is not None and attempt.typecheck.passed
    assert attempt.gate1 is observer_failure
    assert attempt.feedback is None
    assert attempt.typecheck_status is TypeCheckStatus.PASS
    assert search.call_budget.lift_calls == 1
    assert search.call_budget.typecheck_calls == 1
    assert search.call_budget.formal_gate1_rounds == 1
    assert search.call_budget.replay_normal_calls == 0
    assert search.call_budget.replay_physical_calls == 0
    assert lift.feedbacks == [None]
    assert factory.rounds == [1]
    assert gate2.calls == 0


def test_gate2_is_terminal_and_hidden_cases_are_created_after_gate1() -> None:
    events = []
    lift = _Lift([make_reduce_artifact()], events)
    factory = _Gate1Factory([_gate1_result(closed=True)], events)
    gate2 = _Gate2(passed=False, events=events)

    report = _runner(lift, factory, gate2).run(_request())

    assert report.main_status is MainStatus.GATE2_FAILED
    assert report.call_budget.lift_calls == 1
    assert events == ["lift", "gate1", "gate2_inputs_and_execution"]
    assert gate2.calls == 1


def test_gate1_search_defers_hidden_gate2_until_explicit_completion() -> None:
    events = []
    lift = _Lift([make_reduce_artifact()], events)
    factory = _Gate1Factory([_gate1_result(closed=True)], events)
    gate2 = _Gate2(events=events)
    runner = _runner(lift, factory, gate2)

    search = runner.search_gate1(_request())

    assert search.status is Gate1SearchStatus.GATE1_CLOSED
    assert search.closed
    assert search.accepted_attempt == 1
    assert search.call_budget.gate2_cases == 0
    assert gate2.calls == 0
    assert events == ["lift", "gate1"]

    report = runner.complete_gate2(_request(), search)

    assert report.main_status is MainStatus.SCRC_CERTIFIED
    assert report.call_budget.gate2_cases == 1
    assert gate2.calls == 1
    assert events == ["lift", "gate1", "gate2_inputs_and_execution"]


def test_gate1_search_reports_source_free_stage_transitions() -> None:
    transitions = []
    runner = _runner(
        _Lift([make_reduce_artifact()]),
        _Gate1Factory([_gate1_result(closed=True)]),
        _Gate2(),
    )

    search = runner.search_gate1(
        _request(),
        progress_callback=lambda stage, attempt, formal_round, status: (
            transitions.append((stage, attempt, formal_round, status))
        ),
    )

    assert search.closed
    assert transitions == [
        ("lift", 1, None, "started"),
        ("lift", 1, None, "finished"),
        ("typecheck", 1, None, "started"),
        ("typecheck", 1, None, "pass"),
        ("gate1", None, 1, "started"),
        ("gate1", None, 1, "replay_closed"),
    ]


def test_progress_callback_failure_cannot_change_gate1_search() -> None:
    runner = _runner(
        _Lift([make_reduce_artifact()]),
        _Gate1Factory([_gate1_result(closed=True)]),
        _Gate2(),
    )

    def broken_progress(*_args) -> None:
        raise RuntimeError("progress sink is unavailable")

    search = runner.search_gate1(
        _request(),
        progress_callback=broken_progress,
    )

    assert search.closed
    assert search.accepted_attempt == 1


def test_gate1_search_cooperatively_stops_before_relift() -> None:
    stop = Event()
    lift = _Lift([make_reduce_artifact(), make_reduce_artifact()])

    class StopAfterGate1:
        def run(self, artifact, request):
            del artifact, request
            stop.set()
            return _gate1_result(closed=False)

    rounds = []

    def factory(formal_round):
        rounds.append(formal_round)
        return StopAfterGate1()

    search = _runner(lift, factory, _Gate2()).search_gate1(
        _request(),
        stop_requested=lambda: 7 if stop.is_set() else None,
    )

    assert search.status is Gate1SearchStatus.SUPERSEDED
    assert search.superseded and not search.closed
    assert search.stop_reason == "first_gate1_closure"
    assert search.winner_branch_id == 7
    assert search.stop_observed_at == "after_gate1"
    assert search.call_budget.lift_calls == 1
    assert search.call_budget.formal_gate1_rounds == 1
    assert rounds == [1]
    assert len(lift.feedbacks) == 1


def test_masked_j_status_is_independent_of_main_certification() -> None:
    for masked_closed, expected in (
        (True, JNecessityStatus.NOT_PROVED),
        (False, JNecessityStatus.PROVED),
    ):
        report = _runner(
            _Lift([make_reduce_artifact()]),
            _Gate1Factory(
                [_gate1_result(closed=True, masked_closed=masked_closed)]
            ),
            _Gate2(),
        ).run(_request())
        assert report.main_status is MainStatus.SCRC_CERTIFIED
        assert report.j_necessity_status is expected


def test_typecheck_rejection_cap_never_calls_replay_or_gate2() -> None:
    invalid = []
    for _ in range(2):
        artifact = make_reduce_artifact()
        payload = artifact.pi.to_json()
        payload["instances"][0]["semantic_ref"] = "n99"
        artifact.pi = RealizationWitness.from_json(payload)
        invalid.append(artifact)
    lift = _Lift(invalid)
    factory = _Gate1Factory([])
    gate2 = _Gate2()

    report = _runner(lift, factory, gate2).run(
        _request(max_pre_replay_rejections=2)
    )

    assert report.main_status is MainStatus.NO_CLOSURE
    assert report.call_budget.typecheck_calls == 2
    assert factory.rounds == []
    assert gate2.calls == 0


def test_source_compile_failure_is_not_reported_as_no_closure() -> None:
    runner = _runner(
        _Lift([make_reduce_artifact()]),
        _Gate1Factory([_gate1_result(closed=False, compile_failed=True)]),
        _Gate2(),
    )
    with pytest.raises(SCRCInfrastructureError, match="source compilation") as raised:
        runner.run(_request())
    assert raised.value.evidence["formal_round"] == 1
    assert raised.value.evidence["gate1"]["status"] == "compile_failed"
    assert raised.value.evidence["artifact"]["schema"]


def test_gate1_factory_cannot_reuse_one_replay_session_across_rounds() -> None:
    shared = _Gate1Adapter(_gate1_result(closed=False))
    runner = _runner(
        _Lift([make_reduce_artifact(), make_reduce_artifact()]),
        lambda formal_round: shared,
        _Gate2(),
    )
    with pytest.raises(SCRCInfrastructureError, match="Replay must be fresh"):
        runner.run(_request())


_SMOKE_TTIR = """
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

_SMOKE_REPLAY = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    v0 = tl.arange(0, c0)
    v1 = v0 < p2
    v2 = tl.load(p0 + v0, mask=v1, other=0.0)
    tl.store(p1 + v0, v2, mask=v1)
"""


class _SmokeCompiler:
    def compile(self, source, case):
        del case
        names = ["X", "Y", "N", "BLOCK"] if source.trusted else [
            "p0", "p1", "p2", "c0"
        ]
        return CompileResult(
            True,
            ttir=_SMOKE_TTIR,
            python_interface={
                "arg_names": names,
                "constexprs": [names[3]],
                "constexpr_indices": [3],
            },
        )


class _SmokeExecutionRunner:
    def __init__(self, output):
        self.output = output

    def run(self, leg, source, spec, values):
        del source, spec, values
        return ExecutionResult(leg, True, {"output": self.output.copy()})


class _SmokeRefEvalRunner:
    def __init__(self, output):
        self.output = output

    def run(self, spec, values):
        del spec, values
        return ExecutionResult("j", True, {"output": self.output.copy()})


def test_deterministic_end_to_end_smoke_uses_active_adapters_and_same_c0() -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    gate_case = CompileCase(
        arguments={
            "X": RuntimeArgument.tensor((2, 4), "float32"),
            "Y": RuntimeArgument.tensor((2,), "float32"),
            "N": RuntimeArgument.scalar(4, "int32"),
        },
        grid=(1,),
        schedule={"BLOCK": 128},
    )
    binding = HarnessBindingInstance.create(
        artifact.c.values,
        gate1_case=gate_case,
        gate2_context={"seeds": (7,)},
    )
    source = KernelSource.from_code(
        "def kernel(X, Y, N, BLOCK):\n    return None\n",
        "kernel",
        trusted=True,
    )
    request = SCRCRunRequest(
        SourceProgram(source.code, source), binding, "triton"
    )
    writers = []

    def gate1_factory(formal_round):
        assert formal_round == 1
        writer = StaticReplayWriter(_SMOKE_REPLAY)
        writers.append(writer)
        return BoundGate1Adapter(
            Gate1Runner(
                compiler=_SmokeCompiler(),
                writer=writer,
                config=Gate1Config(run_masked_j=False),
            )
        )

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    expected = np.sum(x * x, axis=1)
    refeval = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    execution_spec = TritonExecutionSpec(
        (
            TritonArgument("p0", ArgumentRole.INPUT, value_key="input"),
            TritonArgument(
                "p1",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=(2,),
                output_key="output",
            ),
        ),
        (1,),
    )
    suite_calls = []

    def suite_factory(accepted, gate1, actual_binding):
        del gate1
        suite_calls.append(actual_binding)
        assert actual_binding.values == accepted.c.values == {"c0": 128}
        return Gate2Suite(
            "smoke",
            (
                Gate2Case(
                    "case0",
                    accepted.j,
                    {"input": x},
                    execution_spec,
                    refeval,
                    ("output",),
                ),
            ),
            actual_binding.c_values,
        )

    gate2 = TritonGate2ExecutionAdapter(
        Gate2Runner(
            source_runner=_SmokeExecutionRunner(expected),
            replay_runner=_SmokeExecutionRunner(expected),
            refeval_runner=_SmokeRefEvalRunner(expected),
        ),
        suite_factory,
    )
    report = SCRCGateRunner(
        SCRCRunDependencies(_Lift([artifact]), gate1_factory, gate2)
    ).run(request)

    assert report.main_status is MainStatus.SCRC_CERTIFIED
    assert report.call_budget.formal_gate1_rounds == 1
    assert report.call_budget.gate2_cases == 1
    assert writers[0].calls == [False]
    assert suite_calls == [binding]
    assert report.gate2.cases[0].replay_j.passed
