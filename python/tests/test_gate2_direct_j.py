from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.canonical_j import CanonicalJError, j_elaboration
from intentir.ir.functional_boundary import (
    FUNCTIONAL_BOUNDARY_PROTOCOL,
    RUNTIME_PREDICATE_PROOF_PROTOCOL,
    functional_reduction_padding_issues,
    predicate_dominated_pointwise_load,
    project_functional_boundary,
    project_functional_boundary_for_refeval,
)
from intentir.ir.model import IntentFunction, IntentReturn
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    BroadcastSpec,
    ConsumerExtension,
    EffectKind,
    EffectSpec,
    IndexBoundsSpec,
    LogicalSignature,
    OutputMapping,
    ParameterKind,
    ParameterMode,
    ReductionBoundary,
    SemanticArchive,
    SignatureParameter,
    TypedOutsideValue,
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


def _mixed_radix_copy_j() -> SemanticArchive:
    artifact = make_copy_artifact()
    # ``identity`` is sufficient for Gate1 fixtures but is intentionally not a
    # production Intent lowering recipe.  Use a supported unary while this test
    # exercises the real scalar/MLIR consumer end to end.
    artifact.j.intent.body[0].attrs["fn"] = "abs"
    for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]:
        tensor.shape = ["x0", "x1", "x2"]
    artifact.j.signature.parameters[0].rank = 3
    artifact.j.signature.parameters[1].rank = 3
    artifact.j.signature.parameters.extend(
        [
            SignatureParameter(
                "p3", ParameterKind.SCALAR, ParameterMode.VALUE,
                "i32", logical_ref="x1",
            ),
            SignatureParameter(
                "p4", ParameterKind.SCALAR, ParameterMode.VALUE,
                "i32", logical_ref="x2",
            ),
        ]
    )
    coordinate = (
        "[k0 // (x1 * x2), (k0 // x2) % x1, k0 % x2]"
    )
    artifact.j.boundary = Boundary(
        domains={"d0": "0 <= k0 and k0 < ((x0 * x1) * x2)"},
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", node_ref="n0",
                index_expression=coordinate,
            ),
            BoundaryAccess(
                "a1", AccessKind.STORE, "t1", node_ref="n0",
                index_expression=coordinate,
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
        ],
    )
    return artifact.j


def test_functional_boundary_allows_semantically_unused_tensor_abi_input() -> None:
    artifact = make_copy_artifact()
    _attach_dense_copy_boundary(
        artifact,
        domains={"d0": "0 <= i0 and i0 < x0"},
    )
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"].append(
        {"name": "unused", "dtype": "f32", "shape": ["x0"]}
    )
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.signature.parameters.append(
        SignatureParameter(
            f"p{len(artifact.j.signature.parameters)}",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            "f32",
            1,
            logical_ref="unused",
        )
    )

    projection = project_functional_boundary(artifact.j)

    assert "access:a0" in projection.consumed
    assert not any("unused" in item for item in projection.consumed)


def test_functional_boundary_consumes_complete_mixed_radix_bijection() -> None:
    j = _mixed_radix_copy_j()

    projection = project_functional_boundary(j)
    receipts = projection.to_json()["mixed_radix_accesses"]

    assert projection.domains == (("k0", "((x0 * x1) * x2)"),)
    assert len(receipts) == 2
    assert receipts[0]["proof"] == "complete_mixed_radix_bijection"
    assert receipts[0]["flat_axis"] == "k0"
    assert receipts[0]["domain_factors"] == ["x0", "x1", "x2"]
    assert [item["stride_factors"] for item in receipts[0]["digits"]] == [
        ["x1", "x2"],
        ["x2"],
        [],
    ]

    spec = ScalarLoweringSpec.create(
        j,
        dims={"x0": 2, "x1": 3, "x2": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["output"], value)


def test_functional_boundary_rejects_incomplete_mixed_radix_digits() -> None:
    j = _mixed_radix_copy_j()
    j.boundary.accesses[0].index_expression = (
        "[k0 // (x1 * x2), k0 % x1, k0 % x2]"
    )

    with pytest.raises(NotImplementedError, match="mixed-radix bijection"):
        project_functional_boundary(j)


def _constant_gather_boundary_j() -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "constant_gather_boundary",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [3]}],
            "outputs": [
                {"name": "Y0", "dtype": "f32", "shape": []},
                {"name": "Y1", "dtype": "f32", "shape": []},
                {"name": "Y2", "dtype": "f32", "shape": []},
            ],
            "body": [
                {
                    "id": "n0", "op": "const", "result": "I0",
                    "attrs": {"value": 0, "dtype": "i32", "shape": []},
                    "type": {"dtype": "i32", "shape": []},
                },
                {
                    "id": "n1", "op": "gather", "result": "Y0",
                    "operands": ["X", "I0"],
                    "attrs": {"axis": 0, "mode": "take"},
                    "type": {"dtype": "f32", "shape": []},
                },
                {
                    "id": "n2", "op": "const", "result": "I1",
                    "attrs": {"value": 1, "dtype": "i32", "shape": []},
                    "type": {"dtype": "i32", "shape": []},
                },
                {
                    "id": "n3", "op": "gather", "result": "Y1",
                    "operands": ["X", "I1"],
                    "attrs": {"axis": 0, "mode": "take"},
                    "type": {"dtype": "f32", "shape": []},
                },
                {
                    "id": "n4", "op": "const", "result": "I2",
                    "attrs": {"value": 2, "dtype": "i32", "shape": []},
                    "type": {"dtype": "i32", "shape": []},
                },
                {
                    "id": "n5", "op": "gather", "result": "Y2",
                    "operands": ["X", "I2"],
                    "attrs": {"axis": 0, "mode": "take"},
                    "type": {"dtype": "f32", "shape": []},
                },
            ],
        }
    )
    accesses = [
        BoundaryAccess(
            f"a{index}", AccessKind.LOAD, "X", node_ref=f"n{2 * index + 1}",
            index_expression=f"[{index}]",
        )
        for index in range(3)
    ] + [
        BoundaryAccess(
            f"a{index + 3}", AccessKind.STORE, f"Y{index}",
            node_ref=f"n{2 * index + 1}", index_expression="[]",
        )
        for index in range(3)
    ]
    effects = [
        EffectSpec(
            EffectKind.READ, "X", f"n{2 * index + 1}", f"e{index}",
            f"a{index}",
        )
        for index in range(3)
    ] + [
        EffectSpec(
            EffectKind.WRITE, f"Y{index}", f"n{2 * index + 1}",
            f"e{index + 3}", f"a{index + 3}",
        )
        for index in range(3)
    ]
    return SemanticArchive(
        "constant_gather_boundary",
        function,
        {f"n{index}": f"body[{index}]" for index in range(6)},
        Boundary(
            accesses=accesses,
            effects=effects,
            index_bounds=[
                IndexBoundsSpec(f"n{2 * index + 1}", "X", f"I{index}", 0)
                for index in range(3)
            ],
        ),
        LogicalSignature(
            [
                SignatureParameter(
                    "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 1, logical_ref="X",
                ),
                *[
                    SignatureParameter(
                        f"p{index + 1}", ParameterKind.POINTER,
                        ParameterMode.OUTPUT, "f32", 0,
                        logical_ref=f"Y{index}",
                    )
                    for index in range(3)
                ],
            ]
        ),
    )


def _unit_concat_partition_j(*, width: int = 1) -> SemanticArchive:
    result_width = 3 * width
    function = IntentFunction.from_json(
        {
            "name": "unit_concat_partition",
            "inputs": [
                {"name": name, "dtype": "f32", "shape": [2, width]}
                for name in ("A", "B", "C")
            ],
            "outputs": [
                {"name": "Y", "dtype": "f32", "shape": [2, result_width]}
            ],
            "body": [
                {
                    "id": "n0", "op": "concat", "result": "Y",
                    "operands": ["A", "B", "C"],
                    "attrs": {"axis": 1},
                    "type": {"dtype": "f32", "shape": [2, result_width]},
                }
            ],
        }
    )
    loads = [
        BoundaryAccess(
            f"a{index}", AccessKind.LOAD, tensor, node_ref="n0",
            index_expression="[k0, k1]",
        )
        for index, tensor in enumerate(("A", "B", "C"))
    ]
    stores = [
        BoundaryAccess(
            f"a{index + 3}", AccessKind.STORE, "Y", node_ref="n0",
            index_expression=f"[k0, {index * width}]",
        )
        for index in range(3)
    ]
    effects = [
        EffectSpec(
            EffectKind.READ, tensor, "n0", f"e{index}", f"a{index}",
        )
        for index, tensor in enumerate(("A", "B", "C"))
    ] + [
        EffectSpec(
            EffectKind.WRITE, "Y", "n0", f"e{index + 3}", f"a{index + 3}",
        )
        for index in range(3)
    ]
    return SemanticArchive(
        "unit_concat_partition",
        function,
        {"n0": "body[0]"},
        Boundary(
            domains={
                "d0": "0 <= k0 and k0 < 2",
                "d1": f"0 <= k1 and k1 < {width}",
            },
            accesses=[*loads, *stores],
            effects=effects,
        ),
        LogicalSignature(
            [
                *[
                    SignatureParameter(
                        f"p{index}", ParameterKind.POINTER,
                        ParameterMode.INPUT, "f32", 2, logical_ref=tensor,
                    )
                    for index, tensor in enumerate(("A", "B", "C"))
                ],
                SignatureParameter(
                    "p3", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 2, logical_ref="Y",
                ),
            ]
        ),
    )


def _unit_direct_store_partition_j() -> SemanticArchive:
    function = IntentFunction.from_json(
        {
            "name": "unit_direct_store_partition",
            "inputs": [
                {"name": name, "dtype": "f32", "shape": [2, 3]}
                for name in ("A", "B")
            ],
            "outputs": [
                {"name": "Y", "dtype": "f32", "shape": [2, 3]}
            ],
            "body": [
                {
                    "id": "n0", "op": "map_binary", "result": "Y",
                    "operands": ["A", "B"], "attrs": {"fn": "add"},
                    "type": {"dtype": "f32", "shape": [2, 3]},
                }
            ],
        }
    )
    loads = [
        BoundaryAccess(
            f"a{index}", AccessKind.LOAD, tensor, node_ref="n0",
            index_expression="[k0, k1]",
        )
        for index, tensor in enumerate(("A", "B"))
    ]
    stores = [
        BoundaryAccess(
            f"a{index + 2}", AccessKind.STORE, "Y", node_ref="n0",
            index_expression=f"[k0, {index}]",
        )
        for index in range(3)
    ]
    effects = [
        EffectSpec(
            EffectKind.READ, tensor, "n0", f"e{index}", f"a{index}",
        )
        for index, tensor in enumerate(("A", "B"))
    ] + [
        EffectSpec(
            EffectKind.WRITE, "Y", "n0", f"e{index + 2}", f"a{index + 2}",
        )
        for index in range(3)
    ]
    return SemanticArchive(
        "unit_direct_store_partition",
        function,
        {"n0": "body[0]"},
        Boundary(
            domains={
                "d0": "0 <= k0 and k0 < 2",
                "d1": "0 <= k1 and k1 < 3",
            },
            accesses=[*loads, *stores],
            effects=effects,
        ),
        LogicalSignature(
            [
                *[
                    SignatureParameter(
                        f"p{index}", ParameterKind.POINTER,
                        ParameterMode.INPUT, "f32", 2, logical_ref=tensor,
                    )
                    for index, tensor in enumerate(("A", "B"))
                ],
                SignatureParameter(
                    "p2", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 2, logical_ref="Y",
                ),
            ]
        ),
    )


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
    with pytest.raises(NotImplementedError, match="neutral 'max' identity"):
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
    artifact.j.boundary.consumer_extensions = [
        ConsumerExtension(
            "n1",
            0,
            "a0",
            0,
            TypedOutsideValue("f32", 0),
            "two_sided",
        )
    ]
    return artifact


def _make_shared_access_sum_max_artifact():
    artifact = _make_masked_sum_reduction_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["outputs"].append(
        {"name": "t2", "dtype": "f32", "shape": ["x0"]}
    )
    payload["body"].append(
        {
            "id": "n2",
            "op": "reduce",
            "result": "v2",
            "operands": ["v0"],
            "attrs": {
                "fn": "max",
                "axes": [1],
                "assoc": "float_nondeterministic",
            },
            "type": {"dtype": "f32", "shape": ["x0"]},
        }
    )
    payload["returns"].append({"output": "t2", "value": "v2"})
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.node_ids["n2"] = "body[2]"
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.POINTER,
            ParameterMode.OUTPUT,
            "f32",
            1,
            logical_ref="t2",
        )
    )
    artifact.j.boundary.accesses.append(
        BoundaryAccess(
            "a2",
            AccessKind.STORE,
            "t2",
            node_ref="n2",
            index_expression="[i0]",
        )
    )
    artifact.j.boundary.effects.append(
        EffectSpec(EffectKind.WRITE, "t2", "n2", "e2", "a2")
    )
    artifact.j.boundary.output_mappings.append(
        OutputMapping("t2", "v2", "n2")
    )
    artifact.j.boundary.reductions.append(
        ReductionBoundary("n2", 0, "NEG_INF")
    )
    artifact.j.boundary.consumer_extensions.append(
        ConsumerExtension(
            "n2",
            0,
            "a0",
            0,
            TypedOutsideValue("f32", "NEG_INF"),
            "two_sided",
        )
    )
    return artifact


def _make_masked_forward_scan_artifact():
    artifact = _make_masked_sum_reduction_artifact()
    scan = artifact.j.intent.body[1]
    scan.op = "scan"
    scan.attrs = {"fn": "sum", "axis": 1}
    scan.type = {"dtype": "f32", "shape": ["x0", "x1"]}
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.boundary.accesses[1].index_expression = "[i0, i1]"
    artifact.j.boundary.accesses[1].predicate = "i1 < x1"
    return artifact


def test_functional_boundary_consumes_closed_forward_scan_extension() -> None:
    projection = project_functional_boundary(
        _make_masked_forward_scan_artifact().j
    )

    assert projection.consumer_extensions[0].consumer_ref == "n1"
    assert projection.consumer_extensions[0].tile_axis == 0
    assert projection.profile == "logical_domain_reduction_padding_v3"


@pytest.mark.parametrize("mutation", ["wrong_local_axis", "reverse_control"])
def test_functional_boundary_rejects_unclosed_scan_padding(mutation: str) -> None:
    artifact = _make_masked_forward_scan_artifact()
    if mutation == "wrong_local_axis":
        artifact.j.boundary.consumer_extensions[0].tile_axis = 1
    elif mutation == "reverse_control":
        artifact.j.intent.body[1].attrs["reverse"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(
        ValueError,
        match="closed local axis|closed forward inclusive",
    ):
        project_functional_boundary(artifact.j)


def test_scan_padded_suffix_cannot_flow_unmasked_into_later_reduce() -> None:
    artifact = _make_masked_forward_scan_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["outputs"][0]["shape"] = ["x0"]
    payload["body"].append(
        {
            "id": "n2",
            "op": "reduce",
            "result": "v2",
            "operands": ["v1"],
            "attrs": {
                "fn": "sum",
                "axes": [1],
                "assoc": "float_nondeterministic",
            },
            "type": {"dtype": "f32", "shape": ["x0"]},
        }
    )
    payload["returns"] = [{"output": "t1", "value": "v2"}]
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.node_ids["n2"] = "body[2]"
    artifact.j.signature.parameters[1].rank = 1
    artifact.j.boundary.accesses[1].node_ref = "n2"
    artifact.j.boundary.accesses[1].index_expression = "[i0]"
    artifact.j.boundary.accesses[1].predicate = None
    artifact.j.boundary.effects[1].node_ref = "n2"
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "v2", "n2")
    ]
    artifact.j.boundary.reductions.append(
        ReductionBoundary("n2", 0, 0)
    )

    with pytest.raises(
        NotImplementedError,
        match="n2:0:a0",
    ):
        project_functional_boundary(artifact.j)


def test_functional_boundary_consumes_same_access_per_consumer_identity() -> None:
    projection = project_functional_boundary(
        _make_shared_access_sum_max_artifact().j
    )

    assert [item.to_json() for item in projection.consumer_extensions] == [
        {
            "consumer_ref": "n1",
            "operand_index": 0,
            "access_ref": "a0",
            "tile_axis": 0,
            "outside": {"dtype": "f32", "value": 0},
            "identity_law": "two_sided",
        },
        {
            "consumer_ref": "n2",
            "operand_index": 0,
            "access_ref": "a0",
            "tile_axis": 0,
            "outside": {"dtype": "f32", "value": "NEG_INF"},
            "identity_law": "two_sided",
        },
    ]
    assert "consumer_extension:n1:0:a0" in projection.consumed
    assert "consumer_extension:n2:0:a0" in projection.consumed


def test_refeval_consumes_same_access_with_distinct_consumer_identities() -> None:
    artifact = _make_shared_access_sum_max_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "sum_output", "t2": "max_output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    squared = value * value
    np.testing.assert_allclose(
        result.outputs["sum_output"], np.sum(squared, axis=1)
    )
    np.testing.assert_allclose(
        result.outputs["max_output"], np.max(squared, axis=1)
    )
    relations = result.metadata["boundary_projection"]["consumer_extensions"]
    assert [item["outside"]["value"] for item in relations] == [0, "NEG_INF"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_consumer", "wrong_operand"])
def test_functional_boundary_has_no_global_consumer_identity_fallback(
    mutation: str,
) -> None:
    artifact = _make_shared_access_sum_max_artifact()
    extension = artifact.j.boundary.consumer_extensions[1]
    if mutation == "missing":
        artifact.j.boundary.consumer_extensions.pop()
    elif mutation == "wrong_consumer":
        extension.consumer_ref = "n0"
    elif mutation == "wrong_operand":
        extension.operand_index = 1
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(
        (NotImplementedError, ValueError),
        match="outside-domain identity relation|operand ordinal|sum/max/min",
    ):
        project_functional_boundary(artifact.j)


def _add_logical_guard(artifact, *, predicate: str, live: bool) -> None:
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"].append({"name": "flag", "dtype": "i1", "shape": []})
    if live:
        payload["body"].append(
            {
                "id": "n2",
                "op": "select",
                "result": "v2",
                "operands": ["flag", "v1", "v1"],
                "attrs": {},
                "type": {"dtype": "f32", "shape": ["x0"]},
            }
        )
        payload["returns"] = [{"output": "t1", "value": "v2"}]
        artifact.j.boundary.accesses[1].node_ref = "n2"
        artifact.j.boundary.effects[1].node_ref = "n2"
        artifact.j.boundary.output_mappings = [
            OutputMapping("t1", "v2", "n2")
        ]
    else:
        payload["body"].append(
            {
                "id": "n2",
                "op": "map_unary",
                "result": "dead_guard",
                "operands": ["flag"],
                "attrs": {"fn": "identity"},
                "type": {"dtype": "i1", "shape": []},
            }
        )
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.node_ids["n2"] = "body[2]"
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i1",
            0,
            logical_ref="flag",
        )
    )
    artifact.j.boundary.accesses[0].predicate = predicate


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
        == "logical_domain_reduction_padding_v3"
    )
    assert result.metadata["predicated_access_count"] == 1
    assert result.metadata["erased_fill_count"] == 1


def test_refeval_runtime_logical_bound_preserves_reduction_padding_contract() -> None:
    artifact = _make_masked_sum_reduction_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"].append({"name": "limit", "dtype": "i32", "shape": []})
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.signature.parameters.append(
        SignatureParameter(
            f"p{len(artifact.j.signature.parameters)}",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i32",
            0,
            logical_ref="limit",
        )
    )
    artifact.j.boundary.accesses[0].predicate = "i1 < limit"
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input", "limit": "hidden_limit"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": value,
            "hidden_limit": np.asarray(4, dtype=np.int32),
        },
    )

    assert result.ok, result.error
    np.testing.assert_allclose(
        result.outputs["output"],
        np.sum(value * value, axis=1),
    )
    assert result.metadata["boundary_profile"] == (
        "logical_domain_runtime_reduction_padding_v2"
    )
    assert result.metadata["runtime_predicate_proof_count"] == 1
    assert result.metadata["boundary_projection"]["runtime_predicate_proofs"][
        0
    ]["access_id"] == "a0"


@pytest.mark.parametrize(
    "predicate",
    ["flag and i1 < x1", "not flag and i1 < x1"],
)
def test_scalar_e_j_rejects_guard_bearing_access_predicate(predicate) -> None:
    artifact = _make_masked_sum_reduction_artifact()
    _add_logical_guard(artifact, predicate=predicate, live=True)

    with pytest.raises(
        NotImplementedError,
        match="does not implement guard-bearing access predicates",
    ):
        _validate_complete_boundary(artifact.j)


def test_scalar_e_j_rejects_guard_with_only_dead_dataflow_membership() -> None:
    artifact = _make_masked_sum_reduction_artifact()
    _add_logical_guard(
        artifact,
        predicate="flag and i1 < x1",
        live=False,
    )

    with pytest.raises(
        NotImplementedError,
        match="does not implement guard-bearing access predicates",
    ):
        _validate_complete_boundary(artifact.j)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_outer_domain", "complete logical domain"),
        ("missing_fill", "explicit masked-load fill"),
        ("wrong_predicate_axis", "same predicate"),
        ("predicated_store", "same predicate"),
        ("wrong_store_axis", "differs from J tensor extent"),
        ("nonzero_identity", "neutral 'sum' identity"),
        (
            "nonneutral_consumer_extension",
            "two-sided identity",
        ),
        ("non_sum", "neutral 'max' identity"),
        ("padded_mean", "physical denominator"),
        ("missing_reduction_contract", "contract for every affected consumer"),
    ],
)
def test_scalar_e_j_masked_sum_mutations_fail_closed(mutation, expected) -> None:
    artifact = _make_masked_sum_reduction_artifact()
    if mutation == "missing_outer_domain":
        del artifact.j.boundary.domains["d0"]
    elif mutation == "missing_fill":
        artifact.j.boundary.accesses[0].fill_value = None
    elif mutation == "wrong_predicate_axis":
        artifact.j.boundary.accesses[0].predicate = "i0 < x0"
    elif mutation == "predicated_store":
        artifact.j.boundary.accesses[1].predicate = "i0 < x0"
    elif mutation == "wrong_store_axis":
        artifact.j.boundary.accesses[1].index_expression = "[i1]"
    elif mutation == "nonzero_identity":
        artifact.j.boundary.reductions[0].identity = 1.0
    elif mutation == "nonneutral_consumer_extension":
        artifact.j.boundary.consumer_extensions[0].outside.value = 1.0
    elif mutation == "non_sum":
        artifact.j.intent.body[1].attrs["fn"] = "max"
    elif mutation == "padded_mean":
        artifact.j.intent.body[1].attrs["fn"] = "mean"
    elif mutation == "missing_reduction_contract":
        artifact.j.boundary.reductions = []
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


@pytest.mark.parametrize("binding_source", ["return_only", "mapping_only"])
def test_scalar_e_j_accepts_one_complete_output_binding_source(
    binding_source,
) -> None:
    artifact = _make_masked_sum_reduction_artifact()
    if binding_source == "return_only":
        artifact.j.boundary.output_mappings = []
    else:
        artifact.j.intent.returns = []
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"], np.sum(value * value, axis=1)
    )


def test_padded_mean_issue_is_j_only_and_does_not_ban_dense_mean() -> None:
    padded = _make_masked_sum_reduction_artifact()
    padded.j.intent.body[1].attrs["fn"] = "mean"

    issues = functional_reduction_padding_issues(padded.j)

    assert len(issues) == 1
    assert issues[0].path == "j.intent.intent_function.body[1].attrs.fn"
    assert issues[0].ref == "n1"
    assert "logical valid count" in issues[0].detail

    dense = make_reduce_artifact(explicit_accesses=False)
    dense.j.intent.body[1].attrs["fn"] = "mean"
    assert functional_reduction_padding_issues(dense.j) == ()


def test_scalar_e_j_rejects_masked_provenance_at_unmodeled_contraction() -> None:
    artifact = _make_masked_sum_reduction_artifact()
    contraction = artifact.j.intent.body[0]
    contraction.op = "contraction"
    contraction.operands = ["t0", "t0"]
    contraction.attrs = {"subscripts": "mk,nk->mn"}
    contraction.type = {"dtype": "f32", "shape": ["x0", "x0"]}
    mean = artifact.j.intent.body[1]
    mean.attrs["fn"] = "mean"

    # TypeCheck's narrow necessary condition does not guess through an
    # unmodeled op, while the functional consumer must fail at that boundary.
    assert functional_reduction_padding_issues(artifact.j) == ()
    with pytest.raises(
        NotImplementedError,
        match="masked padding provenance through unmodeled op 'contraction'",
    ):
        _validate_complete_boundary(artifact.j)


def _make_masked_softmax_artifact():
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.intent = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [
                {"name": "t0", "dtype": "f32", "shape": [2, "x0"]},
            ],
            "outputs": [
                {"name": "t1", "dtype": "f32", "shape": [2, "x0"]},
            ],
            "body": [
                {
                    "id": "n0",
                    "op": "reduce",
                    "result": "v0",
                    "operands": ["t0"],
                    "attrs": {
                        "fn": "max",
                        "axes": [1],
                        "assoc": "float_nondeterministic",
                    },
                    "type": {"dtype": "f32", "shape": [2]},
                },
                {
                    "id": "n1",
                    "op": "map_binary",
                    "result": "v1",
                    "operands": ["t0", "v0"],
                    "attrs": {"fn": "sub"},
                    "type": {"dtype": "f32", "shape": [2, "x0"]},
                },
                {
                    "id": "n2",
                    "op": "map_unary",
                    "result": "v2",
                    "operands": ["v1"],
                    "attrs": {"fn": "exp"},
                    "type": {"dtype": "f32", "shape": [2, "x0"]},
                },
                {
                    "id": "n3",
                    "op": "reduce",
                    "result": "v3",
                    "operands": ["v2"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [1],
                        "assoc": "float_nondeterministic",
                    },
                    "type": {"dtype": "f32", "shape": [2]},
                },
                {
                    "id": "n4",
                    "op": "map_binary",
                    "result": "t1",
                    "operands": ["v2", "v3"],
                    "attrs": {"fn": "div"},
                    "type": {"dtype": "f32", "shape": [2, "x0"]},
                },
            ],
        }
    )
    artifact.j.node_ids = {
        "n0": "body[0]",
        "n1": "body[1]",
        "n2": "body[2]",
        "n3": "body[3]",
        "n4": "body[4]",
    }
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.signature.parameters[2].logical_ref = "x0"
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < 2",
        "d1": "0 <= i1 and i1 < x0",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="i1 < x0",
            fill_value="-inf",
            index_expression="[i0, i1]",
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t1",
            predicate="i1 < x0",
            node_ref="n4",
            index_expression="[i0, i1]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", effect_id="e0", access_ref="a0"),
        EffectSpec(
            EffectKind.WRITE,
            "t1",
            "n4",
            "e1",
            "a1",
        ),
    ]
    artifact.j.boundary.broadcasts = [
        BroadcastSpec("n1", "v0", [1]),
        BroadcastSpec("n4", "v3", [1]),
    ]
    artifact.j.boundary.reductions = [
        ReductionBoundary("n0", 0, "-inf", "-inf"),
        ReductionBoundary("n3", 0, 0, 0),
    ]
    artifact.j.boundary.output_mappings = []
    return artifact


def test_scalar_e_j_consumes_general_masked_softmax_graph() -> None:
    artifact = _make_masked_softmax_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.asarray(
        [[1.0, 2.0, -1.0, 0.5], [0.0, 3.0, 1.5, -2.0]],
        dtype=np.float32,
    )

    result = ScalarLoweringRunner().run(spec, {"input": value})

    shifted = value - np.max(value, axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= np.sum(expected, axis=1, keepdims=True)
    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=1e-6)
    assert result.metadata["protocol"] == "j_scalar_cpu_mlir_v16"
    assert (
        result.metadata["boundary_profile"]
        == "logical_domain_reduction_padding_v2"
    )
    assert result.metadata["predicated_access_count"] == 2
    assert result.metadata["erased_fill_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_domain", "differs from the full logical domain"),
        ("missing_fill", "explicit masked-load fill"),
        ("missing_broadcast", "incompatible NumPy broadcast shapes"),
        ("wrong_broadcast", "cannot align with consumer shape"),
        ("missing_reduction", "contract for every affected consumer"),
        ("nonneutral_max", "neutral 'max' masked_contribution"),
    ],
)
def test_scalar_e_j_masked_softmax_graph_mutations_fail_closed(
    mutation: str,
    expected: str,
) -> None:
    artifact = _make_masked_softmax_artifact()
    if mutation == "missing_domain":
        del artifact.j.boundary.domains["d1"]
    elif mutation == "missing_fill":
        artifact.j.boundary.accesses[0].fill_value = None
    elif mutation == "missing_broadcast":
        artifact.j.boundary.broadcasts.pop(0)
    elif mutation == "wrong_broadcast":
        artifact.j.boundary.broadcasts[0].axes = [0]
    elif mutation == "missing_reduction":
        artifact.j.boundary.reductions.pop()
    elif mutation == "nonneutral_max":
        artifact.j.boundary.reductions[0].masked_contribution = 0
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    if mutation in {"missing_broadcast", "wrong_broadcast"}:
        with pytest.raises(CanonicalJError, match=expected):
            ScalarLoweringSpec.create(
                artifact.j,
                dims={"x0": 4},
                input_bindings={"t0": "input"},
                output_bindings={"t1": "output"},
            )
        return
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 4},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec,
        {"input": np.ones((2, 4), dtype=np.float32)},
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


def _make_runtime_logical_mask_artifact(
    *,
    predicate: str = "i0 < limit",
    limit_dtype: str = "i32",
    guard: bool = False,
):
    artifact = _make_masked_identity_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"].append(
        {"name": "limit", "dtype": limit_dtype, "shape": []}
    )
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            limit_dtype,
            0,
            logical_ref="limit",
        )
    )
    if guard:
        payload["inputs"].append({"name": "flag", "dtype": "i1", "shape": []})
        artifact.j.signature.parameters.append(
            SignatureParameter(
                "p4",
                ParameterKind.SCALAR,
                ParameterMode.VALUE,
                "i1",
                0,
                logical_ref="flag",
            )
        )
    artifact.j.intent = IntentFunction.from_json(payload)
    for access in artifact.j.boundary.accesses:
        access.predicate = predicate
    return artifact


def _runtime_logical_mask_spec(artifact) -> ScalarLoweringSpec:
    input_bindings = {"t0": "input", "limit": "hidden_limit"}
    if any(item.name == "flag" for item in artifact.j.intent.inputs):
        input_bindings["flag"] = "hidden_flag"
    return ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings=input_bindings,
        output_bindings={"t1": "output"},
    )


def _make_value_dimension_mask_artifact():
    artifact = _make_masked_identity_artifact()
    payload = artifact.j.to_json()
    function = payload["intent"]["intent_function"]
    function["inputs"].append(
        {"name": "t2", "dtype": "i32", "shape": []}
    )
    extent_parameter = payload["signature"]["parameters"][2]
    extent_parameter["rank"] = 0
    extent_parameter["logical_ref"] = "t2"
    for access in payload["boundary"]["accesses"]:
        access["predicate"] = "i0 < t2"
    payload["value_dimension_relations"] = [
        {
            "value_ref": "t2",
            "dimension_ref": "x0",
            "relation": "eq",
        }
    ]
    artifact.j = SemanticArchive.from_json(payload)
    return artifact


def test_value_dimension_relation_closes_value_free_production_projection() -> None:
    artifact = _make_value_dimension_mask_artifact()

    projection = project_functional_boundary(artifact.j)

    assert projection.runtime_predicate_proofs == ()
    assert projection.value_dimension_relations == (("t2", "x0"),)
    assert (
        "value_dimension:t2:x0:eq" in projection.consumed
    )
    assert dict(projection.predicates)["a0"].upper_bounds == (("i0", "x0"),)
    assert projection.to_json()["value_dimension_relations"] == [
        {
            "value_ref": "t2",
            "dimension_ref": "x0",
            "relation": "eq",
        }
    ]


def test_refeval_rejects_value_dimension_runtime_mismatch() -> None:
    artifact = _make_value_dimension_mask_artifact()
    inputs = {
        "t0": np.arange(8, dtype=np.float32),
        "t2": np.asarray(8, dtype=np.int32),
    }

    project_functional_boundary_for_refeval(artifact.j, inputs, {"x0": 8})
    inputs["t2"] = np.asarray(13, dtype=np.int32)
    with pytest.raises(ValueError, match="value.*dimension|dimension.*value"):
        project_functional_boundary_for_refeval(
            artifact.j, inputs, {"x0": 8}
        )


@pytest.mark.parametrize("limit", [8, 13])
def test_refeval_runtime_proves_rank_zero_logical_bound_tautology(limit: int) -> None:
    artifact = _make_runtime_logical_mask_artifact()
    spec = _runtime_logical_mask_spec(artifact)
    value = np.arange(8, dtype=np.float32)

    with pytest.raises(
        NotImplementedError,
        match="cross-axis logical access predicates",
    ):
        project_functional_boundary(spec.j)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": value,
            "hidden_limit": np.asarray(limit, dtype=np.int32),
        },
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["output"], value)
    projection = result.metadata["boundary_projection"]
    assert projection["protocol"] == "intentir.j-functional-boundary.v15"
    assert FUNCTIONAL_BOUNDARY_PROTOCOL == projection["protocol"]
    assert result.metadata["boundary_profile"] == (
        "logical_domain_runtime_mask_erasure_v1"
    )
    assert result.metadata["runtime_predicate_proof_count"] == 2
    receipts = projection["runtime_predicate_proofs"]
    assert [item["access_id"] for item in receipts] == ["a0", "a1"]
    assert all(
        item["protocol"] == RUNTIME_PREDICATE_PROOF_PROTOCOL
        and item["proof"] == "closed_domain_tautology"
        and item["logical_input_refs"] == ["limit"]
        and item["domain_axes"] == ["i0"]
        and "value" not in item
        for item in receipts
    )


def test_refeval_runtime_rejects_logical_bound_short_of_closed_domain() -> None:
    artifact = _make_runtime_logical_mask_artifact()
    spec = _runtime_logical_mask_spec(artifact)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(7, dtype=np.int32),
        },
    )

    assert not result.ok
    assert result.unsupported
    assert "cross-axis logical access predicates" in (result.error or "")
    assert "7" not in (result.error or "")


@pytest.mark.parametrize(
    ("predicate", "flag", "accepted"),
    [
        ("flag and i0 < limit", True, True),
        ("flag and i0 < limit", False, False),
        ("not flag and i0 < limit", False, True),
        ("not flag and i0 < limit", True, False),
    ],
)
def test_refeval_runtime_guard_polarity_is_proved_not_erased(
    predicate: str,
    flag: bool,
    accepted: bool,
) -> None:
    artifact = _make_runtime_logical_mask_artifact(
        predicate=predicate,
        guard=True,
    )
    spec = _runtime_logical_mask_spec(artifact)
    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(8, dtype=np.int32),
            "hidden_flag": np.asarray(flag, dtype=np.bool_),
        },
    )

    assert result.ok is accepted
    if accepted:
        receipts = result.metadata["boundary_projection"][
            "runtime_predicate_proofs"
        ]
        assert all(
            item["logical_input_refs"] == ["flag", "limit"]
            for item in receipts
        )
    else:
        assert result.unsupported
        assert "guard-bearing access predicates" in (result.error or "")
        assert str(flag).lower() not in (result.error or "").lower()


def test_refeval_runtime_boolean_guard_accepts_canonical_bool_alias() -> None:
    artifact = _make_runtime_logical_mask_artifact(
        predicate="flag and i0 < limit",
        guard=True,
    )
    payload = artifact.j.to_json()
    function = payload["intent"]["intent_function"]
    next(item for item in function["inputs"] if item["name"] == "flag")[
        "dtype"
    ] = "bool"
    next(
        item
        for item in payload["signature"]["parameters"]
        if item.get("logical_ref") == "flag"
    )["dtype"] = "bool"
    artifact.j = SemanticArchive.from_json(payload)
    spec = _runtime_logical_mask_spec(artifact)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(8, dtype=np.int32),
            "hidden_flag": np.asarray(True, dtype=np.bool_),
        },
    )

    assert result.ok, result.error
    assert result.metadata["runtime_predicate_proof_count"] == 2


def test_runtime_tautology_does_not_require_unobservable_load_fill() -> None:
    artifact = _make_runtime_logical_mask_artifact(
        predicate="flag and i0 < limit",
        guard=True,
    )
    for access in artifact.j.boundary.accesses:
        if access.kind.value == "load":
            access.fill_value = None
    spec = _runtime_logical_mask_spec(artifact)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(8, dtype=np.int32),
            "hidden_flag": np.asarray(True, dtype=np.bool_),
        },
    )

    assert result.ok, result.error
    assert result.metadata["runtime_predicate_proof_count"] == 2


@pytest.mark.parametrize(
    ("limit_dtype", "runtime_value"),
    [
        ("f32", np.asarray(8.0, dtype=np.float32)),
        ("f32", np.asarray(np.nan, dtype=np.float32)),
        ("i1", np.asarray(True, dtype=np.bool_)),
        ("i32", np.asarray([8], dtype=np.int32)),
    ],
)
def test_refeval_runtime_bound_requires_exact_rank_zero_integer_logical_input(
    limit_dtype: str,
    runtime_value: np.ndarray,
) -> None:
    artifact = _make_runtime_logical_mask_artifact(limit_dtype=limit_dtype)
    spec = _runtime_logical_mask_spec(artifact)
    result = ScalarLoweringRunner().run(
        spec,
        {"input": np.arange(8, dtype=np.float32), "hidden_limit": runtime_value},
    )

    assert not result.ok
    assert result.unsupported
    assert "cross-axis logical access predicates" in (result.error or "")
    assert "nan" not in (result.error or "").lower()


def test_refeval_runtime_proves_each_store_predicate_independently() -> None:
    artifact = _make_runtime_logical_mask_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"].append({"name": "short", "dtype": "i32", "shape": []})
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p4",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i32",
            0,
            logical_ref="short",
        )
    )
    artifact.j.boundary.accesses[1].predicate = "i0 < short"
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={
            "t0": "input",
            "limit": "hidden_limit",
            "short": "hidden_short",
        },
        output_bindings={"t1": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(8, dtype=np.int32),
            "hidden_short": np.asarray(7, dtype=np.int32),
        },
    )

    assert not result.ok
    assert result.unsupported
    assert "cross-axis logical access predicates" in (result.error or "")


def test_refeval_runtime_rejects_predicate_axis_outside_direct_access() -> None:
    artifact = _make_runtime_logical_mask_artifact(
        predicate="i1 < limit",
    )
    spec = _runtime_logical_mask_spec(artifact)
    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(8, dtype=np.int32),
        },
    )

    assert not result.ok
    assert result.unsupported
    assert "cross-axis logical access predicates" in (result.error or "")


def test_runtime_predicate_receipt_does_not_disclose_hidden_value() -> None:
    artifact = _make_runtime_logical_mask_artifact()
    spec = _runtime_logical_mask_spec(artifact)
    hidden_value = 13
    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32),
            "hidden_limit": np.asarray(hidden_value, dtype=np.int32),
        },
    )

    assert result.ok, result.error
    receipts = result.metadata["boundary_projection"][
        "runtime_predicate_proofs"
    ]
    assert receipts
    assert all("value" not in receipt for receipt in receipts)
    assert all(
        hidden_value not in receipt.values()
        for receipt in receipts
    )


def test_runtime_predicate_receipt_and_compile_identity_are_value_independent() -> None:
    artifact = _make_runtime_logical_mask_artifact()
    spec = _runtime_logical_mask_spec(artifact)
    runner = ScalarLoweringRunner()
    results = [
        runner.run(
            spec,
            {
                "input": np.arange(8, dtype=np.float32),
                "hidden_limit": np.asarray(limit, dtype=np.int32),
            },
        )
        for limit in (8, 13)
    ]

    assert all(result.ok for result in results)
    assert results[0].metadata["scalar_spec_digest"] == results[1].metadata[
        "scalar_spec_digest"
    ]
    assert results[0].metadata["boundary_digest"] == results[1].metadata[
        "boundary_digest"
    ]
    assert results[0].metadata["boundary_projection"][
        "runtime_predicate_proofs"
    ] == results[1].metadata["boundary_projection"][
        "runtime_predicate_proofs"
    ]


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
    assert SCALAR_LOWERING_PROTOCOL == "j_scalar_cpu_mlir_v16"
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))


def test_scalar_e_j_erases_unobservable_pointwise_masked_load_fill() -> None:
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
    artifact.j.boundary.accesses[0].fill_value = None
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
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))
    assert "predicate_dominated_load:a0:i0 < x0" in result.metadata[
        "boundary_projection"
    ]["consumed"]

    artifact.j.boundary.accesses[1].predicate = "i0 < 7"
    assert not predicate_dominated_pointwise_load(artifact.j, "a0")
    rejected = ScalarLoweringRunner().run(
        ScalarLoweringSpec.create(
            artifact.j,
            dims={"x0": 8},
            input_bindings={"t0": "input"},
            output_bindings={"t1": "output"},
        ),
        {"input": value},
    )
    assert not rejected.ok
    assert "predicate differs from the full logical domain" in (
        rejected.error or ""
    )


def test_scalar_e_j_erases_mask_through_exact_terminal_transpose() -> None:
    artifact = make_copy_artifact()
    payload = artifact.j.intent.to_json()["intent_function"]
    payload["inputs"][0]["shape"] = ["x0", "x1"]
    payload["outputs"][0]["shape"] = ["x1", "x0"]
    payload["body"] = [
        {
            "id": "n0",
            "op": "map_unary",
            "result": "v0",
            "operands": ["t0"],
            "attrs": {"fn": "tanh"},
            "type": {"dtype": "f32", "shape": ["x0", "x1"]},
        },
        {
            "id": "n1",
            "op": "transpose",
            "result": "t1",
            "operands": ["v0"],
            "attrs": {"perm": [1, 0]},
            "type": {"dtype": "f32", "shape": ["x1", "x0"]},
        },
    ]
    artifact.j.intent = IntentFunction.from_json(payload)
    artifact.j.node_ids = {"n0": "body[0]", "n1": "body[1]"}
    artifact.j.signature.parameters[0].rank = 2
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3", ParameterKind.SCALAR, ParameterMode.VALUE,
            "i32", logical_ref="x1",
        )
    )
    artifact.j.boundary = Boundary(
        domains={
            "d0": "0 <= i0 and i0 < x0",
            "d1": "0 <= i1 and i1 < x1",
        },
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", node_ref="n0",
                predicate="i1 < x1", index_expression="[i0, i1]",
            ),
            BoundaryAccess(
                "a1", AccessKind.STORE, "t1", node_ref="n1",
                predicate="i1 < x1", index_expression="[i1, i0]",
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.WRITE, "t1", "n1", "e1", "a1"),
        ],
    )

    assert predicate_dominated_pointwise_load(artifact.j, "a0")
    projection = project_functional_boundary(artifact.j)
    assert "predicate_dominated_load:a0:i1 < x1" in projection.consumed

    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = ScalarLoweringRunner().run(
        ScalarLoweringSpec.create(
            artifact.j,
            dims={"x0": 2, "x1": 3},
            input_bindings={"t0": "input"},
            output_bindings={"t1": "output"},
        ),
        {"input": value},
    )
    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value).T)

    artifact.j.boundary.accesses[1].index_expression = "[i0, i1]"
    assert not predicate_dominated_pointwise_load(artifact.j, "a0")


def test_scalar_e_j_erases_mask_through_closed_pointwise_broadcast() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [
                {"name": "t0", "dtype": "f32", "shape": ["x1"]},
                {"name": "t2", "dtype": "f32", "shape": ["x0", "x1"]},
            ],
            "outputs": [
                {"name": "t1", "dtype": "f32", "shape": ["x0", "x1"]}
            ],
            "body": [
                {
                    "id": "n0",
                    "op": "map_unary",
                    "result": "v0",
                    "operands": ["t0"],
                    "attrs": {"fn": "abs"},
                    "type": {"dtype": "f32", "shape": ["x1"]},
                },
                {
                    "id": "n1",
                    "op": "map_binary",
                    "result": "t1",
                    "operands": ["t2", "v0"],
                    "attrs": {"fn": "mul"},
                    "type": {"dtype": "f32", "shape": ["x0", "x1"]},
                },
            ],
        }
    )
    artifact.j.node_ids = {"n0": "body[0]", "n1": "body[1]"}
    artifact.j.signature = LogicalSignature(
        [
            SignatureParameter(
                "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                "f32", 1, logical_ref="t0",
            ),
            SignatureParameter(
                "p1", ParameterKind.POINTER, ParameterMode.OUTPUT,
                "f32", 2, logical_ref="t1",
            ),
            SignatureParameter(
                "p2", ParameterKind.POINTER, ParameterMode.INPUT,
                "f32", 2, logical_ref="t2",
            ),
            SignatureParameter(
                "p3", ParameterKind.SCALAR, ParameterMode.VALUE,
                "i32", logical_ref="x0",
            ),
            SignatureParameter(
                "p4", ParameterKind.SCALAR, ParameterMode.VALUE,
                "i32", logical_ref="x1",
            ),
        ]
    )
    artifact.j.boundary = Boundary(
        domains={
            "d0": "0 <= i0 and i0 < x0",
            "d1": "0 <= i1 and i1 < x1",
        },
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", node_ref="n0",
                predicate="i1 < x1", index_expression="[i1]",
            ),
            BoundaryAccess(
                "a1", AccessKind.LOAD, "t2", node_ref="n1",
                predicate="i1 < x1", fill_value=0.0,
                index_expression="[i0, i1]",
            ),
            BoundaryAccess(
                "a2", AccessKind.STORE, "t1", node_ref="n1",
                predicate="i1 < x1", index_expression="[i0, i1]",
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.READ, "t2", "n1", "e1", "a1"),
            EffectSpec(EffectKind.WRITE, "t1", "n1", "e2", "a2"),
        ],
    )

    assert predicate_dominated_pointwise_load(artifact.j, "a0")
    projection = project_functional_boundary(artifact.j)
    assert "predicate_dominated_load:a0:i1 < x1" in projection.consumed

    first = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    second = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = ScalarLoweringRunner().run(
        ScalarLoweringSpec.create(
            artifact.j,
            dims={"x0": 2, "x1": 3},
            input_bindings={"t0": "input0", "t2": "input1"},
            output_bindings={"t1": "output"},
        ),
        {"input0": first, "input1": second},
    )
    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], second * np.abs(first))


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


def _make_value_proved_row_major_artifact():
    artifact = _make_masked_2d_pointwise_artifact()
    payload = artifact.j.to_json()
    function = payload["intent"]["intent_function"]
    function["inputs"].append(
        {"name": "t2", "dtype": "i64", "shape": []}
    )
    payload["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "scalar",
            "mode": "value",
            "dtype": "i64",
            "rank": 0,
            "logical_ref": "t2",
        }
    )
    for access in payload["boundary"]["accesses"]:
        access["index_expression"] = "[(i0 * t2) + i1]"
    payload["value_dimension_relations"] = [
        {
            "value_ref": "t2",
            "dimension_ref": "x1",
            "relation": "eq",
        }
    ]
    artifact.j = SemanticArchive.from_json(payload)
    return artifact


def test_scalar_e_j_projects_j_proved_row_major_stride() -> None:
    artifact = _make_value_proved_row_major_artifact()
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input", "t2": "row_stride"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(8, dtype=np.float32).reshape(2, 4)

    result = ScalarLoweringRunner().run(
        spec,
        {"input": value, "row_stride": np.asarray(4, dtype=np.int64)},
    )

    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["output"], np.tanh(value))
    assert dict(result.metadata["boundary_projection"]["access_coordinates"])[
        "a0"
    ] == ["i0", "i1"]


def test_scalar_e_j_rejects_unproved_row_major_stride() -> None:
    artifact = _make_value_proved_row_major_artifact()
    artifact.j.value_dimension_relations = []
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 4},
        input_bindings={"t0": "input", "t2": "row_stride"},
        output_bindings={"t1": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": np.arange(8, dtype=np.float32).reshape(2, 4),
            "row_stride": np.asarray(4, dtype=np.int64),
        },
    )

    assert not result.ok
    assert "direct ordered axis vector" in (result.error or "")


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


def test_constant_gather_loads_match_boundary_coordinates_bidirectionally() -> None:
    j = _constant_gather_boundary_j()

    projection = project_functional_boundary(j)
    receipt = projection.to_json()

    assert projection.access_coordinates_map == {
        "a0": (0,), "a1": (1,), "a2": (2,),
        "a3": (), "a4": (), "a5": (),
    }
    assert projection.access_axes_map == {
        "a0": ("0",), "a1": (), "a2": (),
        "a3": (), "a4": (), "a5": (),
    }
    assert [
        (
            item["access_id"], item["node_ref"], item["index_ref"],
            item["coordinate"],
        )
        for item in receipt["constant_gathers"]
    ] == [
        ("a0", "n1", "I0", 0),
        ("a1", "n3", "I1", 1),
        ("a2", "n5", "I2", 2),
    ]


def test_singleton_zero_keeps_legacy_axis_receipt_and_typed_coordinate() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = [1]
    artifact.j.intent.outputs[0].shape = [1]
    artifact.j.intent.body[0].type = {"dtype": "f32", "shape": [1]}
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0", node_ref="n0",
            index_expression="[0]",
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1", node_ref="n0",
            index_expression="[0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]

    projection = project_functional_boundary(artifact.j)

    assert projection.access_axes_map == {"a0": ("0",), "a1": ("0",)}
    assert projection.access_coordinates_map == {"a0": (0,), "a1": (0,)}


def test_constant_gather_boundary_coordinate_mismatch_fails_closed() -> None:
    j = _constant_gather_boundary_j()
    j.boundary.accesses[0].index_expression = "[1]"

    with pytest.raises(ValueError, match="differs from its J gather index"):
        project_functional_boundary(j)


def test_constant_gather_requires_reverse_coverage_of_direct_j_uses() -> None:
    j = _constant_gather_boundary_j()
    j.boundary.accesses = [
        access for access in j.boundary.accesses if access.access_id != "a2"
    ]
    j.boundary.effects = [
        effect for effect in j.boundary.effects if effect.access_ref != "a2"
    ]

    with pytest.raises(ValueError, match="do not cover.*direct J uses"):
        project_functional_boundary(j)


def test_unit_concat_stores_form_auditable_complete_partition() -> None:
    j = _unit_concat_partition_j()
    projection = project_functional_boundary(j)

    assert len(projection.concat_store_partitions) == 1
    receipt = projection.to_json()["concat_store_partitions"][0]
    assert receipt == {
        "proof": "unit_width_concat_store_partition",
        "output": "Y",
        "producer_ref": "n0",
        "axis": 1,
        "slices": [
            {"access_id": "a3", "operand_ref": "A", "start": 0, "width": 1},
            {"access_id": "a4", "operand_ref": "B", "start": 1, "width": 1},
            {"access_id": "a5", "operand_ref": "C", "start": 2, "width": 1},
        ],
    }

    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"A": "a", "B": "b", "C": "c"},
        output_bindings={"Y": "out"},
    )
    values = {
        "a": np.asarray([[1.0], [2.0]], dtype=np.float32),
        "b": np.asarray([[3.0], [4.0]], dtype=np.float32),
        "c": np.asarray([[5.0], [6.0]], dtype=np.float32),
    }
    result = ScalarLoweringRunner().run(spec, values)

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["out"],
        np.concatenate([values["a"], values["b"], values["c"]], axis=1),
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_unit_concat_partition_rejects_duplicate_or_missing_store(mutation) -> None:
    j = _unit_concat_partition_j()
    if mutation == "duplicate":
        j.boundary.accesses[4].index_expression = "[k0, 0]"
        expected = "duplicate/nonconstant slices"
    else:
        j.boundary.accesses = [
            access for access in j.boundary.accesses if access.access_id != "a5"
        ]
        j.boundary.effects = [
            effect for effect in j.boundary.effects if effect.access_ref != "a5"
        ]
        expected = "missing or extra store"

    with pytest.raises(ValueError, match=expected):
        project_functional_boundary(j)


def test_wide_concat_point_stores_remain_unsupported() -> None:
    j = _unit_concat_partition_j(width=2)

    with pytest.raises(NotImplementedError, match="unit-width operands"):
        project_functional_boundary(j)


def test_direct_stores_form_auditable_complete_output_partition() -> None:
    j = _unit_direct_store_partition_j()
    projection = project_functional_boundary(j)

    assert projection.concat_store_partitions == ()
    assert projection.to_json()["direct_store_partitions"] == [
        {
            "proof": "unit_width_direct_store_partition",
            "output": "Y",
            "value_ref": "Y",
            "producer_ref": "n0",
            "axis": 1,
            "slices": [
                {"access_id": "a2", "start": 0, "width": 1},
                {"access_id": "a3", "start": 1, "width": 1},
                {"access_id": "a4", "start": 2, "width": 1},
            ],
        }
    ]

    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"A": "a", "B": "b"},
        output_bindings={"Y": "out"},
    )
    values = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.full((2, 3), 2.0, dtype=np.float32),
    }
    result = ScalarLoweringRunner().run(spec, values)

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["out"], values["a"] + values["b"])


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nonpartition"])
def test_direct_store_partition_rejects_closed_contradictions(mutation) -> None:
    j = _unit_direct_store_partition_j()
    if mutation == "duplicate":
        j.boundary.accesses[3].index_expression = "[k0, 0]"
        expected = "duplicate slices"
    elif mutation == "missing":
        j.boundary.accesses.pop()
        j.boundary.effects.pop()
        expected = "missing or extra store"
    else:
        for access in j.boundary.accesses[2:]:
            access.index_expression = access.index_expression.replace("k0", "0")
        expected = "do not cover a non-partition extent"

    with pytest.raises(ValueError, match=expected):
        project_functional_boundary(j)


def test_multi_axis_direct_stores_remain_consumer_unsupported() -> None:
    j = _unit_direct_store_partition_j()
    for tensor in j.intent.inputs:
        tensor.shape = [2, 2]
    j.intent.outputs[0].shape = [2, 2]
    j.intent.body[0].type = {"dtype": "f32", "shape": [2, 2]}
    j.boundary.domains["d1"] = "0 <= k1 and k1 < 2"
    j.boundary.accesses = j.boundary.accesses[:2] + [
        BoundaryAccess(
            f"a{index + 2}", AccessKind.STORE, "Y", node_ref="n0",
            index_expression=f"[{row}, {column}]",
        )
        for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
    ]
    j.boundary.effects = j.boundary.effects[:2] + [
        EffectSpec(
            EffectKind.WRITE, "Y", "n0", f"e{index + 2}", f"a{index + 2}",
        )
        for index in range(4)
    ]

    with pytest.raises(NotImplementedError, match="exactly one partition axis"):
        project_functional_boundary(j)


def test_predicated_direct_stores_remain_consumer_unsupported() -> None:
    j = _unit_direct_store_partition_j()
    for access in j.boundary.accesses[2:]:
        access.predicate = "0 <= k0 and k0 < 2"

    with pytest.raises(NotImplementedError, match="requires unpredicated stores"):
        project_functional_boundary(j)


def test_symbolic_axis_direct_stores_remain_consumer_unsupported() -> None:
    j = _unit_direct_store_partition_j()
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        tensor.shape[1] = "x0"
    j.intent.body[0].type = {"dtype": "f32", "shape": [2, "x0"]}
    j.boundary.domains["d1"] = "0 <= k1 and k1 < x0"

    with pytest.raises(NotImplementedError, match="static (tensor|output) extent"):
        project_functional_boundary(j)


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
        ("alias", "does not implement physical alias ABI"),
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
    elif mutation == "alias":
        artifact.j.signature.parameters[1].alias_of = "p0"
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


def test_scalar_e_j_dense_inout_consumes_prestate_and_returns_poststate() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [
                {"name": "t0", "dtype": "f32", "shape": ["x0"]},
                {"name": "t1", "dtype": "f32", "shape": ["x0"]},
            ],
            "outputs": [
                {"name": "t0", "dtype": "f32", "shape": ["x0"]},
                {"name": "t2", "dtype": "f32", "shape": ["x0"]},
            ],
            "returns": [
                {"output": "t0", "value": "v0"},
                {"output": "t2", "value": "v1"},
            ],
            "body": [
                {
                    "id": "n0",
                    "op": "map_binary",
                    "result": "v0",
                    "operands": ["t0", "t1"],
                    "attrs": {"fn": "add"},
                },
                {
                    "id": "n1",
                    "op": "map_unary",
                    "result": "v1",
                    "operands": ["v0"],
                    "attrs": {"fn": "neg"},
                },
            ],
        }
    )
    artifact.j.node_ids = {"n0": "body[0]", "n1": "body[1]"}
    artifact.j.signature = LogicalSignature(
        [
            SignatureParameter(
                "p0", ParameterKind.POINTER, ParameterMode.INOUT,
                "f32", 1, logical_ref="t0",
            ),
            SignatureParameter(
                "p1", ParameterKind.POINTER, ParameterMode.INPUT,
                "f32", 1, logical_ref="t1",
            ),
            SignatureParameter(
                "p2", ParameterKind.POINTER, ParameterMode.OUTPUT,
                "f32", 1, logical_ref="t2",
            ),
            SignatureParameter(
                "p3", ParameterKind.SCALAR, ParameterMode.VALUE, "i32",
                logical_ref="x0",
            ),
        ]
    )
    artifact.j.boundary = Boundary(
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", node_ref="n0",
                index_expression="[i0]",
            ),
            BoundaryAccess(
                "a1", AccessKind.STORE, "t0", node_ref="n0",
                index_expression="[i0]",
            ),
            BoundaryAccess(
                "a2", AccessKind.LOAD, "t1", node_ref="n0",
                index_expression="[i0]",
            ),
            BoundaryAccess(
                "a3", AccessKind.LOAD, "t0", node_ref="n1",
                index_expression="[i0]",
            ),
            BoundaryAccess(
                "a4", AccessKind.STORE, "t2", node_ref="n1",
                index_expression="[i0]",
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.READ, "t1", "n0", "e1", "a2"),
            EffectSpec(EffectKind.WRITE, "t0", "n0", "e2", "a1"),
            EffectSpec(EffectKind.READ, "t0", "n1", "e3", "a3"),
            EffectSpec(EffectKind.WRITE, "t2", "n1", "e4", "a4"),
        ],
        output_mappings=[
            OutputMapping("t0", "v0", "n0"),
            OutputMapping("t2", "v1", "n1"),
        ],
    )
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 8},
        input_bindings={"t0": "prestate", "t1": "delta"},
        output_bindings={"t0": "poststate", "t2": "derived"},
    )
    prestate = np.arange(8, dtype=np.float32)
    delta = np.full(8, 0.5, dtype=np.float32)

    result = ScalarLoweringRunner().run(
        spec, {"prestate": prestate, "delta": delta}
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(result.outputs["poststate"], prestate + delta)
    np.testing.assert_array_equal(result.outputs["derived"], -(prestate + delta))
    assert "inout_poststate_reload:a3:t0:v0" in result.metadata[
        "boundary_projection"
    ]["consumed"]

    artifact.j.boundary.effects[2], artifact.j.boundary.effects[3] = (
        artifact.j.boundary.effects[3],
        artifact.j.boundary.effects[2],
    )
    rejected = ScalarLoweringRunner().run(
        ScalarLoweringSpec.create(
            artifact.j,
            dims={"x0": 8},
            input_bindings={"t0": "prestate", "t1": "delta"},
            output_bindings={"t0": "poststate", "t2": "derived"},
        ),
        {"prestate": prestate, "delta": delta},
    )
    assert not rejected.ok
    assert "not a direct J operand" in (rejected.error or "")
