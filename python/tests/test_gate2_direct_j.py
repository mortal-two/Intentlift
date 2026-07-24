from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.canonical_j import CanonicalJError, j_elaboration
from intentir.ir.model import IntentReturn
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    OutputMapping,
    ParameterMode,
)
from intentir.verify.gate2.model import ExecutionResult, Gate2Status
from intentir.verify.gate2.orchestrate import Gate2Case, Gate2Runner
from intentir.verify.gate2.scalar import (
    SCALAR_LOWERING_PROTOCOL,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    _validate_complete_boundary,
    _parse_domain_predicate,
    _parse_direct_index,
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
        spec.j,
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
    assert j_elaboration(spec.j).value_type_by_name["t1"].dtype == "f32"
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
    reduction.masked_contribution = "NEG_INF"
    assert _validate_complete_boundary(artifact.j)

    reduction.identity = -1.0
    with pytest.raises(NotImplementedError, match="NEG_INF"):
        _validate_complete_boundary(artifact.j)


def _make_masked_sum_reduction_artifact():
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.intent.body[0].type = {
        "dtype": "f32",
        "shape": ["x0", "x1"],
    }
    artifact.j.intent.body[1].result = "v1"
    artifact.j.intent.body[1].type = {
        "dtype": "f32",
        "shape": ["x0"],
    }
    artifact.j.intent.returns = [IntentReturn("t1", "v1")]
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="i1 < x1",
            fill_value=0.0,
            node_ref="n0",
            index_expression="[i0, i1]",
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t1",
            node_ref="n1",
            index_expression="[i0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n1", "e1", "a1"),
    ]
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "v1", "n1")
    ]
    artifact.j.boundary.reductions[0].tile_axis = 0
    return artifact


def test_scalar_e_j_consumes_masked_sum_reduction_padding() -> None:
    artifact = _make_masked_sum_reduction_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_allclose(
        result.outputs["output"], np.sum(value * value, axis=1)
    )
    assert (
        result.metadata["boundary_profile"]
        == "logical_domain_reduction_padding_v1"
    )
    assert result.metadata["predicated_access_count"] == 1
    assert result.metadata["erased_fill_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_outer_domain", "full domain for every input axis"),
        ("nonzero_fill", "explicit zero masked-load fill"),
        ("wrong_predicate_axis", "does not bind the reduced axis"),
        ("predicated_store", "unpredicated output store"),
        ("wrong_store_axis", "differs from J tensor extent"),
        ("nonzero_identity", "explicit neutral identity=0"),
        ("nonzero_masked_contribution", "explicit neutral masked_contribution=0"),
        ("wrong_tile_axis", "single physical tile axis 0"),
        ("non_sum", "currently supports sum only"),
        ("dead_prefix", "every prefix node to feed reduction"),
        ("non_zero_stable_prefix", "zero-stable identity/mul prefix"),
        ("missing_output_mapping", "explicit reduction output mapping"),
    ],
)
def test_scalar_e_j_masked_sum_mutations_fail_closed(mutation, expected) -> None:
    artifact = _make_masked_sum_reduction_artifact()
    if mutation == "missing_outer_domain":
        del artifact.j.boundary.domains["d0"]
    elif mutation == "nonzero_fill":
        artifact.j.boundary.accesses[0].fill_value = 1.0
    elif mutation == "wrong_predicate_axis":
        artifact.j.boundary.accesses[0].predicate = "i0 < x0"
    elif mutation == "predicated_store":
        artifact.j.boundary.accesses[1].predicate = "i0 < x0"
    elif mutation == "wrong_store_axis":
        artifact.j.boundary.accesses[1].index_expression = "[i1]"
    elif mutation == "nonzero_identity":
        artifact.j.boundary.reductions[0].identity = 1.0
    elif mutation == "nonzero_masked_contribution":
        artifact.j.boundary.reductions[0].masked_contribution = 1.0
    elif mutation == "wrong_tile_axis":
        artifact.j.boundary.reductions[0].tile_axis = 1
    elif mutation == "non_sum":
        artifact.j.intent.body[1].attrs["fn"] = "max"
    elif mutation == "dead_prefix":
        artifact.j.intent.body[1].operands = ["t0"]
    elif mutation == "non_zero_stable_prefix":
        artifact.j.intent.body[0].attrs["fn"] = "add"
    elif mutation == "missing_output_mapping":
        artifact.j.boundary.output_mappings = []
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
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
    assert expected in (result.error or "")


def _make_masked_identity_artifact():
    artifact = make_copy_artifact()
    artifact.j.intent.body = []
    artifact.j.intent.returns = [IntentReturn("t1", "t0")]
    artifact.j.node_ids = {}
    artifact.j.boundary.domains = {"d0": "0 <= i0 and i0 < x0"}
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0", predicate="i0 < x0",
            fill_value=0.0, index_expression="i0",
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1", predicate="i0 < x0",
            index_expression="i0",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", effect_id="e0", access_ref="a0"),
        EffectSpec(EffectKind.WRITE, "t1", effect_id="e1", access_ref="a1"),
    ]
    artifact.j.boundary.output_mappings = [OutputMapping("t1", "t0")]
    return artifact


def test_scalar_e_j_erases_full_logical_domain_mask_for_identity() -> None:
    artifact = _make_masked_identity_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["output"], value)
    assert result.metadata["boundary_profile"] == "logical_domain_mask_erasure_v1"
    assert result.metadata["predicated_access_count"] == 2
    assert result.metadata["erased_fill_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("predicate_extent", "differs from the full logical domain"),
        ("unmasked_store", "same predicate on every access"),
        ("missing_fill", "requires an explicit masked-load fill"),
        ("affine_index", "requires a direct ordered axis vector"),
        ("dtype_change", "infers conflicting types"),
        ("destination_change", "references unknown Layer-A value"),
    ],
)
def test_scalar_e_j_mask_erasure_mutations_fail_closed(mutation, expected) -> None:
    artifact = _make_masked_identity_artifact()
    if mutation == "predicate_extent":
        artifact.j.boundary.accesses[0].predicate = "i0 < 7"
    elif mutation == "unmasked_store":
        artifact.j.boundary.accesses[1].predicate = None
    elif mutation == "missing_fill":
        artifact.j.boundary.accesses[0].fill_value = None
    elif mutation == "affine_index":
        artifact.j.boundary.accesses[0].index_expression = "i0 + 1"
    elif mutation == "dtype_change":
        artifact.j.intent.outputs[0].dtype = "f16"
    elif mutation == "destination_change":
        artifact.j.intent.returns[0] = IntentReturn("t1", "t1")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    if mutation in {"dtype_change", "destination_change"}:
        with pytest.raises((CanonicalJError, ValueError), match=expected):
            ScalarLoweringSpec.create(
                artifact.j,
                dims={"x0": 8},
                input_bindings={"t0": "input"},
                output_bindings={"t1": "output"},
            )
        return
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


def test_scalar_e_j_erases_tail_mask_for_general_pointwise_j() -> None:
    artifact = _make_masked_identity_artifact()
    operation = make_copy_artifact().j.intent.body[0]
    operation.result = "v0"
    operation.attrs["fn"] = "tanh"
    operation.type = {"dtype": "f32", "shape": ["x0"]}
    artifact.j.intent.body = [operation]
    artifact.j.intent.returns = [IntentReturn("t1", "v0")]
    artifact.j.node_ids = {"n0": "body[0]"}
    for access in artifact.j.boundary.accesses:
        access.node_ref = "n0"
    for effect in artifact.j.boundary.effects:
        effect.node_ref = "n0"
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "v0", "n0")
    ]
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    assert result.metadata["protocol"] == SCALAR_LOWERING_PROTOCOL
    assert SCALAR_LOWERING_PROTOCOL == "j_scalar_cpu_mlir_v8"
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))


def _make_node_less_input_load_artifact():
    artifact = _make_masked_identity_artifact()
    operation = make_copy_artifact().j.intent.body[0]
    operation.result = "v0"
    operation.attrs["fn"] = "tanh"
    operation.type = {"dtype": "f32", "shape": ["x0"]}
    artifact.j.intent.body = [operation]
    artifact.j.intent.returns = [IntentReturn("t1", "v0")]
    artifact.j.node_ids = {"n0": "body[0]"}
    artifact.j.boundary.accesses[1].node_ref = "n0"
    artifact.j.boundary.effects[1].node_ref = "n0"
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "v0", "n0")
    ]
    return artifact


def test_scalar_e_j_consumes_node_less_input_load_in_nonidentity_graph() -> None:
    artifact = _make_node_less_input_load_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))


def test_scalar_e_j_rejects_ambiguous_node_less_input_load() -> None:
    artifact = _make_node_less_input_load_artifact()
    artifact.j.boundary.accesses.insert(
        1,
        BoundaryAccess(
            "a2", AccessKind.LOAD, "t0", predicate="i0 < x0",
            fill_value=0.0, index_expression="i0",
        ),
    )
    artifact.j.boundary.effects.insert(
        1,
        EffectSpec(EffectKind.READ, "t0", effect_id="e2", access_ref="a2"),
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
    assert "node-less input load must be the sole load" in (result.error or "")


def test_mask_predicate_parser_separates_logical_guard_from_domain_bound() -> None:
    guards, lower, upper = _parse_domain_predicate(
        "flag and i0 < x0", logical_guards=frozenset({"flag"})
    )

    assert guards == {"flag"}
    assert lower == set()
    assert upper == (("i0", "x0"),)


def test_direct_index_parser_canonicalizes_equivalent_axis_vectors() -> None:
    assert _parse_direct_index("[0]") == ("0",)
    assert _parse_direct_index("[i0, i1]") == ("i0", "i1")
    assert _parse_direct_index("i0,i1") == ("i0", "i1")
    with pytest.raises(NotImplementedError, match="direct ordered axis vector"):
        _parse_direct_index("[1]")


def _make_masked_2d_pointwise_artifact():
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = ["x0", "x1"]
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[0].rank = 2
    artifact.j.signature.parameters[1].rank = 2
    operation = artifact.j.intent.body[0]
    operation.attrs["fn"] = "tanh"
    operation.type = {"dtype": "f32", "shape": ["x0", "x1"]}
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0",
            predicate="0 <= i1 and i1 < x1",
            fill_value=0.0,
            node_ref="n0",
            index_expression="[i0, i1]",
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1",
            predicate="0 <= i1 and i1 < x1",
            node_ref="n0",
            index_expression="[i0, i1]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]
    artifact.j.boundary.output_mappings = []
    return artifact


def test_scalar_e_j_erases_2d_pointwise_full_domain_mask() -> None:
    artifact = _make_masked_2d_pointwise_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))
    assert result.metadata["boundary_profile"] == "logical_domain_mask_erasure_v1"


@pytest.mark.parametrize("mutation", ["unbracketed", "missing_type", "redundant_mapping"])
def test_scalar_e_j_2d_pointwise_elaborates_unique_surface_forms(
    mutation: str,
) -> None:
    artifact = _make_masked_2d_pointwise_artifact()
    if mutation == "unbracketed":
        artifact.j.boundary.accesses[0].index_expression = "i0,i1"
    elif mutation == "missing_type":
        artifact.j.intent.body[0].type = None
    elif mutation == "redundant_mapping":
        artifact.j.boundary.output_mappings = [
            OutputMapping("t1", "t1", "n0")
        ]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec,
        {"input": np.arange(8, dtype=np.float32).reshape(2, 4)},
    )

    assert result.ok, result.error
    if mutation == "unbracketed":
        assert spec.j.boundary.accesses[0].index_expression == "[i0, i1]"
        assert j_elaboration(spec.j).accesses[0].index.direct_axes == ("i0", "i1")
    elif mutation == "missing_type":
        assert spec.j.intent.body[0].type == {
            "dtype": "f32", "shape": ["x0", "x1"]
        }
    else:
        assert spec.j.boundary.output_mappings == []


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
        ("malicious_index", "requires a direct ordered axis vector"),
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
