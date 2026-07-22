from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from direct_fixtures import make_reduce_artifact
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate2.model import ExecutionResult, Gate2Status
from intentir.verify.gate2.orchestrate import Gate2Case, Gate2Runner
from intentir.verify.gate2.scalar import (
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    _validate_complete_boundary,
)
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)


class FakeTritonRunner:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, leg, source, spec, values):
        del source, spec, values
        return ExecutionResult(leg, True, self.outputs)


def test_gate2_checks_all_three_pairwise_edges(monkeypatch) -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    expected = np.sum(x * x, axis=1)
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )

    def fake_execute(function, dims, inputs, **kwargs):
        del function, dims, kwargs
        return SimpleNamespace(
            outputs={"t1": np.sum(inputs[0] * inputs[0], axis=1)},
            metadata={"backend": "test"},
        )

    monkeypatch.setattr(
        "intentir.verify.gate2.scalar.execute_on_cpu_outputs", fake_execute
    )
    triton_spec = TritonExecutionSpec(
        (
            TritonArgument("p0", ArgumentRole.INPUT, value_key="input"),
            TritonArgument(
                "p1", ArgumentRole.OUTPUT, dtype="f32", shape=(2,),
                output_key="output",
            ),
        ),
        (1,),
    )
    case = Gate2Case(
        "case0",
        artifact.j,
        {"input": x},
        triton_spec,
        spec,
        ("output",),
    )
    assert case.refeval.j is case.j
    assert not hasattr(case, "replay_artifact_digest")
    assert not hasattr(spec, "final_artifact_digest")
    runner = Gate2Runner(
        source_runner=FakeTritonRunner({"output": expected}),
        replay_runner=FakeTritonRunner({"output": expected}),
        refeval_runner=ScalarLoweringRunner(),
    )
    result = runner.run(
        KernelSource.from_code("source", "source", trusted=True),
        KernelSource.from_code("replay", "replay", trusted=True),
        case,
    )
    assert result.status is Gate2Status.PASS
    assert result.source_replay is not None and result.source_replay.passed
    assert result.source_j is not None and result.source_j.passed
    assert result.replay_j is not None and result.replay_j.passed
    assert spec.j is artifact.j
    assert result.reference.metadata["boundary_digest"]
    report = result.to_json()
    assert report["source_replay"]["passed"]
    assert report["source_j"]["passed"]
    assert report["replay_j"]["passed"]
    assert report["metadata"]["comparison_policy"]["policy_id"]


def test_extrema_boundary_uses_json_safe_infinity_sentinel() -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.intent.body[1].attrs["fn"] = "max"
    reduction = artifact.j.boundary.reductions[0]
    reduction.identity = "NEG_INF"
    assert _validate_complete_boundary(artifact.j)

    reduction.identity = -1.0
    with pytest.raises(NotImplementedError, match="NEG_INF"):
        _validate_complete_boundary(artifact.j)


def test_scalar_e_j_fails_closed_on_unimplemented_access_relations() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.domains.clear()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    result = ScalarLoweringRunner().run(
        spec, {"input": np.ones((2, 4), dtype=np.float32)}
    )
    assert not result.ok
    assert result.unsupported
    assert "explicit access relations" in (result.error or "")
