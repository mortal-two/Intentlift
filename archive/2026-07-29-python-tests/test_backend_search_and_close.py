from __future__ import annotations

import numpy as np
import pytest

from direct_fixtures import make_reduce_artifact
from intentir.backend.search_and_close import (
    BackendClosureError,
    ConcreteLoweringState,
    ExecutionDomain,
    ExecutionDomainCase,
    ObservationClass,
    OpaqueExecutionRunners,
    SearchAndCloseController,
)
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate2.model import ExecutionResult
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY


def _case(j, case_id: str, value: float) -> ExecutionDomainCase:
    return ExecutionDomainCase(
        case_id,
        j,
        {"input": np.asarray([[value]], dtype=np.float32)},
        (("t0", "input"),),
        (("t1", "output"),),
    )


def _output(leg: str, values, *, bias: float = 0.0) -> ExecutionResult:
    value = np.sum(
        np.asarray(values["input"], dtype=np.float32), axis=1
    ) + np.float32(bias)
    return ExecutionResult(leg, True, {"output": value})


def _state() -> ConcreteLoweringState:
    return ConcreteLoweringState(
        "cuda.state.abc",
        "row_reduction",
        (
            "cuda.row_reduction.read_once.v1",
            "cuda.reduction.workgroup_hierarchical.v1",
        ),
        {"block_threads": 128},
        ObservationClass.OBSERVATION_CHANGING,
    )


def _controller(*, source_bias: float = 0.0, replay: bool = True):
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    j = artifact.j
    d_search = ExecutionDomain("search", (_case(j, "search-1", 1.0),))
    d_cert = ExecutionDomain("cert", (_case(j, "cert-1", 2.0),))
    runners = OpaqueExecutionRunners(
        ref_j=lambda case, values: _output("ref_j", values),
        target=lambda executable, case, values: _output(
            "target", values, bias=float(executable["bias"])
        ),
        source=lambda case, values: _output(
            "source", values, bias=source_bias
        ),
        replay=(
            (lambda case, values: _output("replay", values))
            if replay
            else None
        ),
    )
    return SearchAndCloseController(
        j=j,
        target={"family": "cuda", "architecture": "sm_120a"},
        policy=DEFAULT_GATE2_POLICY,
        d_search=d_search,
        d_cert=d_cert,
        runners=runners,
    )


def test_search_filter_then_three_direct_certification_edges() -> None:
    controller = _controller()
    state = _state()

    filtered = controller.filter_candidate({"bias": 0.0}, state)
    result = controller.close_winner({"bias": 0.0}, state)

    assert filtered.passed
    assert {item.edge for item in filtered.edges} == {"target_ref_j"}
    assert result.passed
    assert {item.edge for item in result.certification_edges} == {
        "target_source",
        "target_replay",
        "target_ref_j",
    }
    assert result.d_search["domain_id"] == "search"
    assert result.d_cert["domain_id"] == "cert"
    assert result.d_search["cases"][0]["case_id"] == "search-1"
    assert result.d_cert["cases"][0]["case_id"] == "cert-1"


def test_final_result_does_not_infer_target_source_by_tolerance_transitivity() -> None:
    controller = _controller(source_bias=1.0e-3)
    state = _state()
    assert controller.filter_candidate({"bias": 0.0}, state).passed

    result = controller.close_winner({"bias": 0.0}, state)

    assert not result.passed
    edges = {item.edge: item for item in result.certification_edges}
    assert edges["target_ref_j"].passed
    assert edges["target_replay"].passed
    assert not edges["target_source"].passed


def test_search_and_cert_domains_must_be_exactly_disjoint() -> None:
    artifact = canonicalize_gate1_artifact(make_reduce_artifact())
    case = _case(artifact.j, "same", 1.0)
    runners = OpaqueExecutionRunners(
        ref_j=lambda case, values: _output("ref_j", values),
        target=lambda executable, case, values: _output("target", values),
    )

    with pytest.raises(BackendClosureError, match="disjoint"):
        SearchAndCloseController(
            j=artifact.j,
            target={"family": "cuda", "architecture": "sm_120a"},
            policy=DEFAULT_GATE2_POLICY,
            d_search=ExecutionDomain("search", (case,)),
            d_cert=ExecutionDomain("cert", (case,)),
            runners=runners,
        )


def test_formal_closure_fails_closed_without_replay_runner() -> None:
    controller = _controller(replay=False)
    state = _state()
    assert controller.filter_candidate({"bias": 0.0}, state).passed

    with pytest.raises(BackendClosureError, match="Source and Replay"):
        controller.close_winner({"bias": 0.0}, state)
