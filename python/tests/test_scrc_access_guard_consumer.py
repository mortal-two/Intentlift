from __future__ import annotations

import copy

import numpy as np
import pytest

import intentir.backend.cuda.program as cuda_program
from intentir.backend.protocol import LoweringRequest
from intentir.ir.functional_boundary import (
    FUNCTIONAL_BOUNDARY_PROTOCOL,
    project_functional_boundary,
)
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    AccessGuardPolarity,
    AccessGuardRelation,
    AccessKind,
    Boundary,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    ImplementationConfig,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    OutputMapping,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


def _guarded_select_j(
    polarity: AccessGuardPolarity = AccessGuardPolarity.POSITIVE,
) -> SemanticArchive:
    active = "Candidate" if polarity is AccessGuardPolarity.POSITIVE else "Y"
    inactive = "Y" if polarity is AccessGuardPolarity.POSITIVE else "Candidate"
    function = IntentFunction.from_json(
        {
            "name": "guarded_select",
            "inputs": [
                {"name": "Candidate", "dtype": "f32", "shape": [2, 3]},
                {"name": "Y", "dtype": "f32", "shape": [2, 3]},
                {"name": "Guard", "dtype": "i1", "shape": [2]},
            ],
            "outputs": [
                {"name": "Y", "dtype": "f32", "shape": [2, 3]},
            ],
            "returns": [{"output": "Y", "value": "v0"}],
            "body": [
                {
                    "id": "n0",
                    "op": "select",
                    "result": "v0",
                    "operands": ["Guard", active, inactive],
                    "type": {"dtype": "f32", "shape": [2, 3]},
                }
            ],
        }
    )
    accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "Candidate", node_ref="n0",
            index_expression="[k0, k1]",
        ),
        BoundaryAccess(
            "a1", AccessKind.LOAD, "Y", node_ref="n0",
            index_expression="[k0, k1]",
        ),
        BoundaryAccess(
            "a2", AccessKind.LOAD, "Guard", node_ref="n0",
            index_expression="[k0]",
        ),
        BoundaryAccess(
            "a3", AccessKind.STORE, "Y", node_ref="n0",
            index_expression="[k0, k1]",
        ),
    ]
    effects = [
        EffectSpec(EffectKind.READ, "Candidate", "n0", "e0", "a0"),
        EffectSpec(EffectKind.READ, "Y", "n0", "e1", "a1"),
        EffectSpec(EffectKind.READ, "Guard", "n0", "e2", "a2"),
        EffectSpec(EffectKind.WRITE, "Y", "n0", "e3", "a3"),
    ]
    return SemanticArchive(
        "guarded_select",
        function,
        {"n0": "body[0]"},
        Boundary(
            domains={
                "d0": "0 <= k0 and k0 < 2",
                "d1": "0 <= k1 and k1 < 3",
            },
            accesses=accesses,
            effects=effects,
            output_mappings=[OutputMapping("Y", "v0", "n0")],
            access_guard_relations=[
                AccessGuardRelation("a3", "Guard", [1], polarity),
            ],
        ),
        LogicalSignature(
            [
                SignatureParameter(
                    "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, logical_ref="Candidate",
                ),
                SignatureParameter(
                    "p1", ParameterKind.POINTER, ParameterMode.INOUT,
                    "f32", 2, logical_ref="Y",
                ),
                SignatureParameter(
                    "p2", ParameterKind.POINTER, ParameterMode.INPUT,
                    "i1", 1, logical_ref="Guard",
                ),
            ]
        ),
    )


def test_access_guard_projects_authored_relation_and_unique_broadcast() -> None:
    projection = project_functional_boundary(_guarded_select_j())

    assert FUNCTIONAL_BOUNDARY_PROTOCOL == "intentir.j-functional-boundary.v19"
    assert projection.emitter_broadcast_axes() == {"v0": {"Guard": (1,)}}
    assert projection.to_json()["access_guard_relations"] == [
        {
            "proof": "authored_access_guard_relation",
            "access_ref": "a3",
            "value_ref": "Guard",
            "node_ref": "n0",
            "consumer_result": "v0",
            "axes": [1],
            "polarity": "positive",
        }
    ]
    assert "access_guard:a3:Guard:1:positive" in projection.consumed


@pytest.mark.parametrize(
    ("polarity", "guard_value"),
    [
        (AccessGuardPolarity.POSITIVE, True),
        (AccessGuardPolarity.POSITIVE, False),
        (AccessGuardPolarity.NEGATIVE, True),
        (AccessGuardPolarity.NEGATIVE, False),
    ],
)
def test_access_guard_refeval_uses_only_authored_select(
    polarity: AccessGuardPolarity,
    guard_value: bool,
) -> None:
    j = _guarded_select_j(polarity)
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={
            "Candidate": "candidate",
            "Y": "current",
            "Guard": "guard",
        },
        output_bindings={"Y": "output"},
    )
    candidate = np.arange(6, dtype=np.float32).reshape(2, 3)
    current = np.full((2, 3), -7.0, dtype=np.float32)
    guard = np.full((2,), guard_value, dtype=np.bool_)

    result = ScalarLoweringRunner().run(
        spec,
        {"candidate": candidate, "current": current, "guard": guard},
    )

    assert result.ok, result.error
    true_value, false_value = (
        (candidate, current)
        if polarity is AccessGuardPolarity.POSITIVE
        else (current, candidate)
    )
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.where(guard[:, None], true_value, false_value),
    )
    receipt = result.metadata["boundary_projection"][
        "access_guard_relations"
    ][0]
    assert receipt["polarity"] == polarity.value
    assert receipt["axes"] == [1]


def test_access_guard_broadcast_is_shared_with_structured_reference() -> None:
    j = _guarded_select_j()
    payload = j.intent.to_json()["intent_function"]
    payload["body"].append(
        {
            "id": "n1",
            "op": "tile",
            "result": "Dead",
            "operands": ["Candidate"],
            "attrs": {"reps": [1, 1]},
            "type": {"dtype": "f32", "shape": [2, 3]},
        }
    )
    j.intent = IntentFunction.from_json(payload)
    j.node_ids["n1"] = "body[1]"
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={
            "Candidate": "candidate",
            "Y": "current",
            "Guard": "guard",
        },
        output_bindings={"Y": "output"},
    )
    candidate = np.arange(6, dtype=np.float32).reshape(2, 3)
    current = np.full((2, 3), -3.0, dtype=np.float32)
    guard = np.asarray([True, False], dtype=np.bool_)

    result = ScalarLoweringRunner().run(
        spec,
        {"candidate": candidate, "current": current, "guard": guard},
    )

    assert result.ok, result.error
    assert result.metadata["path"] == "canonical J -> typed structured CPU reference"
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.where(guard[:, None], candidate, current),
    )


def test_access_guard_load_materializes_one_authored_consumer_edge() -> None:
    j = copy.deepcopy(_guarded_select_j())
    relation = j.boundary.access_guard_relations[0]
    relation.access_ref = "a0"
    relation.polarity = AccessGuardPolarity.NEGATIVE
    j.boundary.accesses[0].fill_value = 5.0

    projection = project_functional_boundary(j)
    receipt = projection.access_guard_relations[0]

    assert receipt.access_kind == "load"
    assert receipt.tensor == "Candidate"
    assert receipt.operand_index == 1
    assert receipt.fill_value == 5.0
    assert receipt.to_json()["materialization"] == (
        "access_local_pre_execution_select"
    )

    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={
            "Candidate": "candidate",
            "Y": "current",
            "Guard": "guard",
        },
        output_bindings={"Y": "output"},
    )
    candidate = np.arange(6, dtype=np.float32).reshape(2, 3)
    current = np.full((2, 3), -7.0, dtype=np.float32)
    guard = np.asarray([True, False], dtype=np.bool_)

    result = ScalarLoweringRunner().run(
        spec,
        {"candidate": candidate, "current": current, "guard": guard},
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.where(guard[:, None], np.float32(5.0), current),
    )
    assert result.metadata["execution_evidence"][
        "access_guard_load_materialization_count"
    ] == 1


def test_access_guard_update_is_explicitly_unsupported() -> None:
    j = copy.deepcopy(_guarded_select_j())
    relation = j.boundary.access_guard_relations[0]
    j.boundary.accesses[3].kind = AccessKind.UPDATE
    j.boundary.effects[3].kind = EffectKind.UPDATE

    with pytest.raises(
        NotImplementedError,
        match="guarded update access",
    ):
        project_functional_boundary(j)


def test_access_guard_refeval_api_has_no_c_or_pi_authority() -> None:
    j = _guarded_select_j()
    projection = project_functional_boundary(j)
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={
            "Candidate": "candidate",
            "Y": "current",
            "Guard": "guard",
        },
        output_bindings={"Y": "output"},
    )

    assert projection.access_guard_relations
    assert not hasattr(spec, "c")
    assert not hasattr(spec, "pi")


def test_cuda_fails_closed_before_codegen_for_access_guard(monkeypatch) -> None:
    j = _guarded_select_j()
    analysis = cuda_program.analyze_j_for_cuda(j)
    called = False

    def forbidden_lower(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("production codegen must not run")

    monkeypatch.setattr(cuda_program, "lower_to_cuda", forbidden_lower)
    request = LoweringRequest(
        j,
        cuda_program.cuda_target("sm_120a"),
        "generic",
        ImplementationConfig(),
    )

    assert analysis["supported"] is False
    assert "access_guard_materialization_not_implemented" in analysis[
        "unsupported"
    ]
    assert analysis["boundary_projection"]["access_guard_relations"]
    with pytest.raises(
        cuda_program.CudaJLoweringError,
        match="access_guard_materialization_not_implemented",
    ):
        cuda_program.lower_j_to_cuda(request, {})
    assert called is False
