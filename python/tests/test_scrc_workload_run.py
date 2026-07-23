from __future__ import annotations

from dataclasses import replace

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate2.model import (
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
)
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import (
    CallBudget,
    CandidateAttemptReport,
    FeedbackKind,
    LiftFeedback,
    MainStatus,
    SCRCRunReport,
)
from intentir.verify.scrc_workload_run import (
    build_workload_gate2_suite,
    classify_scrc_report,
    scrc_attempt_stage_counts,
    scrc_report_diagnostic,
)
from intentir.verify.typecheck import TypeCheckStatus
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferRole,
    PreparedTritonWorkload,
    ScalarBinding,
    TRITON_WORKLOAD_ADAPTERS,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)


def _budget() -> CallBudget:
    return CallBudget(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)


def test_gate2_suite_binds_adapter_roles_to_accepted_j() -> None:
    base = prepare_triton_workload(
        next(iter(TRITON_WORKLOAD_ADAPTERS.values()))
    )
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding("X", BufferRole.INPUT, "f32", (1000,)),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=17,
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(
                base.request.binding,
                c_values=(("c0", 1024),),
            ),
        ),
        source_to_opaque={"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"},
    )
    artifact = make_copy_artifact()

    suite = build_workload_gate2_suite(prepared, artifact)

    case = suite.cases[0]
    assert suite.c_values == (("c0", 1024),)
    assert case.output_keys == ("output",)
    assert case.refeval.input_binding_map == {"t0": "input_p0"}
    assert case.refeval.output_binding_map == {"t1": "output"}
    assert case.triton.schedule == {"c0": 1024}
    assert case.triton.arguments[0].name == "p0"
    assert case.triton.arguments[1].name == "p1"
    assert case.triton.arguments[2].name == "p2"


def test_report_classification_preserves_reference_unsupported() -> None:
    execution = ExecutionResult("source", True, {"out": __import__("numpy").zeros(1)})
    gate2 = Gate2SuiteResult(
        "suite",
        (
            Gate2Result(
                "case",
                Gate2Status.REFERENCE_UNSUPPORTED,
                execution,
                ExecutionResult("replay", True, execution.outputs),
                ExecutionResult("j", False, unsupported=True),
            ),
        ),
        (),
    )
    report = SCRCRunReport(
        MainStatus.GATE2_FAILED,
        (),
        1,
        gate2,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.REFERENCE_UNSUPPORTED


def test_gate2_binds_semantic_constexpr_as_rank_zero_j_input() -> None:
    base = prepare_triton_workload(next(iter(TRITON_WORKLOAD_ADAPTERS.values())))
    adapter = TritonWorkloadAdapter(
        base.adapter.case_id,
        buffers=(
            BufferBinding("X", BufferRole.INPUT, "f32", (1000,)),
            BufferBinding("Y", BufferRole.OUTPUT, "f32", (1000,), "output"),
        ),
        scalars=(ScalarBinding("N", "i32", 1000),),
        semantic_constexprs=(("FLAG", True),),
        c_values=(("BLOCK", 1024),),
        grid=(1,),
        grid_contract=("1",),
        seed=19,
    )
    prepared = replace(
        base,
        adapter=adapter,
        request=replace(
            base.request,
            binding=replace(base.request.binding, c_values=(("c0", 1024),)),
        ),
        source_to_opaque={
            "X": "p0",
            "Y": "p1",
            "N": "p2",
            "FLAG": "p3",
            "BLOCK": "c0",
        },
    )
    payload = make_copy_artifact().to_json()
    function = payload["j"]["intent"]["intent_function"]
    function["inputs"].append({"name": "t1", "dtype": "i1", "shape": []})
    function["outputs"][0]["name"] = "t2"
    function["body"][0]["result"] = "t2"
    payload["j"]["signature"]["parameters"][1]["logical_ref"] = "t2"
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "i1",
            "rank": 0,
            "logical_ref": "t1",
        }
    )
    artifact = Gate1Artifact.from_json(payload)

    suite = build_workload_gate2_suite(prepared, artifact)

    case = suite.cases[0]
    assert case.refeval.input_binding_map == {
        "t0": "input_p0",
        "t1": "logical_p3",
    }
    assert case.values["logical_p3"].shape == ()
    assert bool(case.values["logical_p3"])


def test_certified_report_classifies_directly() -> None:
    report = SCRCRunReport(
        MainStatus.SCRC_CERTIFIED,
        (),
        1,
        None,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.SCRC_CERTIFIED


def test_terminal_pre_replay_rejection_keeps_stage_and_diagnostic() -> None:
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        {"diagnostic": "artifact schema rejected"},
    )
    attempt = CandidateAttemptReport(
        lift_call=4,
        formal_round=None,
        typecheck_status=TypeCheckStatus.NOT_RUN,
        feedback=feedback,
    )
    report = SCRCRunReport(
        MainStatus.NO_CLOSURE,
        (attempt,),
        None,
        None,
        None,
        _budget(),
    )

    assert classify_scrc_report(report) is BatchClassification.LIFT_RESPONSE_REJECTED
    assert scrc_attempt_stage_counts(report) == {"lift_response_rejected": 1}
    assert scrc_report_diagnostic(report) == "artifact schema rejected"
