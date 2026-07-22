from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    ParameterMode,
)
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


def _attach_dense_copy_boundary(artifact, *, domains=None) -> None:
    artifact.j.boundary.domains = dict(domains or {})
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0", node_ref="n0",
            index_expression="[i0]",
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1", node_ref="n0",
            index_expression="[i0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]


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


def test_scalar_e_j_fails_closed_on_predicated_access_relations() -> None:
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
    assert "predicated access or fill" in (result.error or "")


@pytest.mark.parametrize(
    ("domains", "expected_domain_count"),
    [
        ({}, 0),
        ({"d0": "0 <= i0 and i0 < x0"}, 1),
    ],
)
def test_scalar_e_j_consumes_dense_functional_boundary(
    monkeypatch, domains, expected_domain_count
) -> None:
    artifact = make_copy_artifact()
    _attach_dense_copy_boundary(artifact, domains=domains)
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )

    def fake_execute(function, dims, inputs, **kwargs):
        del function, dims, kwargs
        return SimpleNamespace(
            outputs={"t1": np.asarray(inputs[0]).copy()},
            metadata={"backend": "test"},
        )

    monkeypatch.setattr(
        "intentir.verify.gate2.scalar.execute_on_cpu_outputs", fake_execute
    )
    value = np.arange(8, dtype=np.float32)
    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["output"], value)
    assert result.metadata["boundary_profile"] == "dense_functional_v1"
    assert result.metadata["domain_relation_count"] == expected_domain_count
    assert result.metadata["access_relation_count"] == 2
    assert result.metadata["effect_relation_count"] == 2


def test_scalar_e_j_rejects_dense_boundary_extent_mismatch() -> None:
    artifact = make_copy_artifact()
    _attach_dense_copy_boundary(
        artifact, domains={"d0": "0 <= i0 and i0 < 7"}
    )
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    result = ScalarLoweringRunner().run(
        spec, {"input": np.arange(8, dtype=np.float32)}
    )

    assert not result.ok
    assert not result.unsupported
    assert "differs from J tensor extent" in (result.error or "")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("malicious_index", "requires index_expression"),
        ("effect_target", "disagrees with its effect"),
        ("duplicate_store", "exactly one store per logical output"),
        ("inout", "does not implement inout/alias ABI"),
    ],
)
def test_scalar_e_j_dense_boundary_mutations_fail_closed(mutation, expected) -> None:
    artifact = make_copy_artifact()
    _attach_dense_copy_boundary(artifact)
    if mutation == "malicious_index":
        artifact.j.boundary.accesses[0].index_expression = "[i0 + 1]"
    elif mutation == "effect_target":
        artifact.j.boundary.effects[1].target = "t0"
    elif mutation == "duplicate_store":
        artifact.j.boundary.accesses.append(
            BoundaryAccess(
                "a2", AccessKind.STORE, "t1", node_ref="n0",
                index_expression="[i0]",
            )
        )
        artifact.j.boundary.effects.append(
            EffectSpec(EffectKind.WRITE, "t1", "n0", "e2", "a2")
        )
    elif mutation == "inout":
        artifact.j.signature.parameters[0].mode = ParameterMode.INOUT
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    result = ScalarLoweringRunner().run(
        spec, {"input": np.arange(8, dtype=np.float32)}
    )

    assert not result.ok
    assert expected in (result.error or "")
