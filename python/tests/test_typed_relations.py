from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from intentir.backend.cuda.program import analyze_j_for_cuda
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import (
    Boundary,
    DuplicatePolicy,
    Gate1Artifact,
    ImplementationConfig,
    IndexBoundsSpec,
    LogicalSignature,
    LogicalUpdateSpec,
    ParameterKind,
    ParameterMode,
    ResultRelationKind,
    ResultRelationSpec,
    SchemaError,
    SemanticArchive,
    SignatureParameter,
    UpdateCombiner,
    UpdateOrder,
)


def _signature(fn: IntentFunction) -> LogicalSignature:
    parameters: list[SignatureParameter] = []
    for tensor in fn.inputs:
        parameters.append(
            SignatureParameter(
                f"p{len(parameters)}",
                ParameterKind.POINTER,
                ParameterMode.INPUT,
                tensor.dtype,
                len(tensor.shape),
                logical_ref=tensor.name,
            )
        )
    for tensor in fn.outputs:
        parameters.append(
            SignatureParameter(
                f"p{len(parameters)}",
                ParameterKind.POINTER,
                ParameterMode.OUTPUT,
                tensor.dtype,
                len(tensor.shape),
                logical_ref=tensor.name,
            )
        )
    return LogicalSignature(parameters)
from intentir.verify.gate1.pi import RealizationWitness
from intentir.verify.gate2.model import ExecutionResult, Gate2Status
from intentir.verify.gate2.orchestrate import Gate2Case, Gate2Runner
from intentir.verify.gate2.policy import ComparisonPolicy, NumericTolerance
from intentir.verify.gate2.scalar import (
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    _validate_complete_boundary,
    scalar_index_bounds,
    validate_index_bounds_inputs,
)
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)


def _update_j(
    *,
    combiner: str = "overwrite",
    duplicate_policy: str = "forbid",
    order: str = "input_order",
    relation: str = "exact",
    rank2: bool = False,
) -> SemanticArchive:
    base_shape = [1, 3] if rank2 else [3]
    update_shape = [1, 2] if rank2 else [2]
    reduce = "none" if combiner == "overwrite" else "add"
    fn = IntentFunction.from_json(
        {
            "name": "typed_update",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": base_shape},
                {"name": "I", "dtype": "i32", "shape": update_shape},
                {"name": "V", "dtype": "f32", "shape": update_shape},
            ],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": base_shape}],
            "body": [
                {
                    "id": "n0",
                    "op": "scatter",
                    "result": "Y",
                    "operands": ["X", "I", "V"],
                    "attrs": {
                        "axis": len(base_shape) - 1,
                        "mode": "along_axis",
                        "reduce": reduce,
                    },
                }
            ],
        }
    )
    if relation == "tolerance":
        result_relation = ResultRelationSpec(
            "Y", ResultRelationKind.TOLERANCE
        )
    elif relation == "one_of_updates":
        result_relation = ResultRelationSpec(
            "Y",
            ResultRelationKind.ONE_OF_UPDATES,
            update_node_ref="n0",
        )
    else:
        result_relation = ResultRelationSpec("Y", ResultRelationKind.EXACT)
    axis = len(base_shape) - 1
    return SemanticArchive(
        "typed_update",
        fn,
        {"n0": "body[0]"},
        Boundary(
            index_bounds=[IndexBoundsSpec("n0", "X", "I", axis)],
            updates=[
                LogicalUpdateSpec(
                    "n0",
                    "X",
                    "I",
                    "V",
                    axis,
                    combiner,
                    duplicate_policy,
                    order,
                )
            ],
            result_relations=[result_relation],
        ),
        _signature(fn),
    )


def _gather_j() -> SemanticArchive:
    fn = IntentFunction.from_json(
        {
            "name": "typed_gather",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": [4]},
                {"name": "I", "dtype": "i32", "shape": [2]},
            ],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [2]}],
            "body": [
                {
                    "id": "n0",
                    "op": "gather",
                    "result": "Y",
                    "operands": ["X", "I"],
                    "attrs": {"axis": 0, "mode": "take"},
                }
            ],
        }
    )
    return SemanticArchive(
        "typed_gather",
        fn,
        {"n0": "body[0]"},
        Boundary(index_bounds=[IndexBoundsSpec("n0", "X", "I", 0)]),
        _signature(fn),
    )


class _FakeRunner:
    def __init__(self, output: np.ndarray):
        self.output = np.asarray(output)

    def run(self, leg, source, spec, values):
        del source, spec, values
        return ExecutionResult(leg, True, {"out": self.output.copy()})


class _FakeScalarRunner:
    def __init__(self, output: np.ndarray):
        self.output = np.asarray(output)

    def run(self, spec, values):
        del spec, values
        return ExecutionResult("j", True, {"out": self.output.copy()})


def _gate2_case(j: SemanticArchive, values: dict[str, np.ndarray]) -> Gate2Case:
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"X": "base", "I": "indices", "V": "values"},
        output_bindings={"Y": "out"},
    )
    triton = TritonExecutionSpec(
        (
            TritonArgument("p0", ArgumentRole.INPUT, value_key="base"),
            TritonArgument("p1", ArgumentRole.INPUT, value_key="indices"),
            TritonArgument("p2", ArgumentRole.INPUT, value_key="values"),
            TritonArgument(
                "p3",
                ArgumentRole.OUTPUT,
                dtype="f32",
                shape=tuple(j.intent.outputs[0].shape),
                output_key="out",
            ),
        ),
        (1,),
    )
    return Gate2Case("case0", j, values, triton, spec, ("out",))


def _run_fake_gate2(
    case: Gate2Case,
    source_output: np.ndarray,
    replay_output: np.ndarray,
    scalar_output: np.ndarray,
    comparison_policy: ComparisonPolicy | None = None,
):
    runner = Gate2Runner(
        source_runner=_FakeRunner(source_output),
        replay_runner=_FakeRunner(replay_output),
        refeval_runner=_FakeScalarRunner(scalar_output),
        comparison_policy=comparison_policy,
    )
    source = KernelSource.from_code("source", "source", trusted=True)
    replay = KernelSource.from_code("replay", "replay", trusted=True)
    return runner.run(source, replay, case)


def test_lift_schema_exposes_only_typed_relation_arrays() -> None:
    boundary = (
        scrc_lift_core_json_schema()["properties"]["j"]["properties"]
        ["boundary"]
    )
    assert {"index_bounds", "updates", "result_relations"} <= set(
        boundary["properties"]
    )
    assert {"index_bounds", "updates", "result_relations"} <= set(
        boundary["required"]
    )
    rendered = repr(boundary).lower()
    assert "proof" in rendered  # description explicitly forbids proof prose
    assert "atomic" in rendered  # description explicitly assigns it to realization


def test_typed_relation_roundtrip_is_closed_and_deterministic() -> None:
    j = _update_j(
        combiner="add",
        duplicate_policy="combine",
        order="unordered",
        relation="tolerance",
    )
    encoded = j.to_json()
    assert SemanticArchive.from_json(deepcopy(encoded)) == j
    assert SemanticArchive.from_json(deepcopy(encoded)).to_json() == encoded
    boundary = encoded["boundary"]
    assert boundary["updates"][0]["combiner"] == "add"
    assert boundary["result_relations"][0]["kind"] == "tolerance"


def test_typed_relations_are_opaque_and_masked_with_the_rest_of_j() -> None:
    j = _update_j(
        duplicate_policy="one_of",
        order="unordered",
        relation="one_of_updates",
    )
    artifact = Gate1Artifact(
        j,
        ImplementationConfig([], {}),
        RealizationWitness.from_json(
            {
                "regions": [{"id": "r0", "parent": None, "placement": "root"}],
                "instances": [
                    {
                        "id": "i0",
                        "semantic_ref": "n0",
                        "region": "r0",
                        "mechanism": "update_atomic",
                    }
                ],
                "routes": [],
                "order": [],
                "sharing": [],
            }
        ),
    )
    normal = artifact.rewriter_view()
    masked = artifact.rewriter_view(masked=True)
    assert normal["j"]["boundary"]["updates"][0]["base"] == "t0"
    assert normal["j"]["boundary"]["result_relations"][0]["kind"] == "one_of_updates"
    assert masked["j"]["boundary"]["updates"][0]["combiner"] == "<masked>"
    assert masked["j"]["boundary"]["result_relations"][0]["kind"] == "<masked>"
    assert normal["pi"] == masked["pi"]
    assert "interface" not in normal


def test_bounds_and_update_must_exactly_match_target_node() -> None:
    j = _gather_j()
    bad = j.to_json()
    bad["boundary"]["index_bounds"][0]["indices"] = "X"
    with pytest.raises(SchemaError, match="data/indices do not match"):
        SemanticArchive.from_json(bad)

    update = _update_j().to_json()
    update["boundary"]["updates"][0]["axis"] = 1
    with pytest.raises(SchemaError, match="axis does not match"):
        SemanticArchive.from_json(update)

    missing_gather_bounds = j.to_json()
    missing_gather_bounds["boundary"]["index_bounds"] = []
    with pytest.raises(SchemaError, match="gather requires a typed"):
        SemanticArchive.from_json(missing_gather_bounds)


def test_free_form_proof_and_illegal_relation_combinations_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown key.*proven"):
        IndexBoundsSpec.from_json(
            {
                "node_ref": "n0",
                "data": "X",
                "indices": "I",
                "axis": 0,
                "proven": True,
            }
        )
    with pytest.raises(SchemaError, match="overwrite.*combine"):
        LogicalUpdateSpec(
            "n0", "X", "I", "V", 0, "overwrite", "combine", "unordered"
        )
    with pytest.raises(SchemaError, match="one_of.*unordered"):
        LogicalUpdateSpec(
            "n0", "X", "I", "V", 0, "overwrite", "one_of", "input_order"
        )
    with pytest.raises(SchemaError, match="unknown key"):
        ResultRelationSpec.from_json(
            {"output": "Y", "kind": "tolerance", "rtol": 1.0}
        )


def test_canonical_scatter_requires_bounds_update_and_result_relation() -> None:
    valid = _update_j().to_json()
    missing_bounds = deepcopy(valid)
    missing_bounds["boundary"]["index_bounds"] = []
    with pytest.raises(SchemaError, match="requires one matching IndexBoundsSpec"):
        SemanticArchive.from_json(missing_bounds)

    missing_update = deepcopy(valid)
    missing_update["boundary"]["updates"] = []
    with pytest.raises(SchemaError, match="requires a typed LogicalUpdateSpec"):
        SemanticArchive.from_json(missing_update)

    missing_relation = deepcopy(valid)
    missing_relation["boundary"]["result_relations"] = []
    with pytest.raises(SchemaError, match="explicit result relation"):
        SemanticArchive.from_json(missing_relation)


def test_typed_gather_bounds_drive_emitter_and_runtime_precondition() -> None:
    j = _gather_j()
    bare = emit_module(j.intent, {})
    typed = emit_module(
        j.intent, {}, proven_index_bounds=scalar_index_bounds(j)
    )
    assert "intent.index_bounds_proven" not in bare
    assert "intent.index_bounds_proven = true" in typed
    assert _validate_complete_boundary(j)
    validate_index_bounds_inputs(
        j,
        {
            "X": np.arange(4, dtype=np.float32),
            "I": np.asarray([3, 0], dtype=np.int32),
        },
    )
    with pytest.raises(ValueError, match="precondition failed"):
        validate_index_bounds_inputs(
            j,
            {
                "X": np.arange(4, dtype=np.float32),
                "I": np.asarray([4, 0], dtype=np.int32),
            },
        )


def test_bounds_authorization_is_node_specific_not_signature_wide() -> None:
    fn = IntentFunction.from_json(
        {
            "name": "two_gathers",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": [4]},
                {"name": "I", "dtype": "i32", "shape": [2]},
            ],
            "outputs": [
                {"name": "Y0", "dtype": "f32", "shape": [2]},
                {"name": "Y1", "dtype": "f32", "shape": [2]},
            ],
            "body": [
                {
                    "id": node,
                    "op": "gather",
                    "result": result,
                    "operands": ["X", "I"],
                    "attrs": {"axis": 0, "mode": "take"},
                }
                for node, result in (("n0", "Y0"), ("n1", "Y1"))
            ],
        }
    )
    j = SemanticArchive(
        "two_gathers",
        fn,
        {"n0": "body[0]", "n1": "body[1]"},
        Boundary(
            index_bounds=[
                IndexBoundsSpec("n0", "X", "I", 0),
                IndexBoundsSpec("n1", "X", "I", 0),
            ]
        ),
        _signature(fn),
    )
    mlir = emit_module(
        j.intent, {}, proven_index_bounds=frozenset({"n0"})
    )
    assert mlir.count("intent.index_bounds_proven = true") == 1


def test_cpu_e_j_recognizes_typed_update_then_fails_on_real_capability() -> None:
    j = _update_j()
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"X": "base", "I": "indices", "V": "values"},
        output_bindings={"Y": "out"},
    )
    result = ScalarLoweringRunner().run(
        spec,
        {
            "base": np.zeros(3, dtype=np.float32),
            "indices": np.asarray([0, 2], dtype=np.int32),
            "values": np.asarray([1, 2], dtype=np.float32),
        },
    )
    assert result.unsupported
    assert "logical update" in result.error


def test_gate2_tolerance_threshold_is_owned_only_by_external_q() -> None:
    fn = IntentFunction.from_json(
        {
            "name": "close",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [1]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [1]}],
            "returns": [{"output": "Y", "value": "X"}],
            "body": [],
        }
    )
    j = SemanticArchive(
        "close",
        fn,
        {},
        Boundary(
            result_relations=[
                ResultRelationSpec("Y", "tolerance")
            ]
        ),
        _signature(fn),
    )
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "out"},
    )
    triton = TritonExecutionSpec(
        (
            TritonArgument("p0", ArgumentRole.INPUT, value_key="input"),
            TritonArgument(
                "p1", ArgumentRole.OUTPUT, dtype="f32", shape=(1,), output_key="out"
            ),
        ),
        (1,),
    )
    case = Gate2Case(
        "close", j, {"input": np.asarray([1.0], dtype=np.float32)},
        triton, spec, ("out",)
    )
    q = ComparisonPolicy.create(
        "test.gate2.q.v1",
        (("f32", NumericTolerance(0.0, 0.11)),),
    )
    result = _run_fake_gate2(
        case,
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.05], dtype=np.float32),
        np.asarray([0.95], dtype=np.float32),
        q,
    )
    assert result.status is Gate2Status.PASS
    assert result.source_replay.outputs[0].relation_kind == "tolerance"
    assert result.replay_j is not None and result.replay_j.passed
    assert result.source_replay.outputs[0].policy_id == "test.gate2.q.v1"

    wrong_dtype = _run_fake_gate2(
        case,
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float32),
        q,
    )
    assert wrong_dtype.status is Gate2Status.PAIRWISE_DIVERGED
    assert "dtype differs from canonical J" in wrong_dtype.source_replay.outputs[0].detail

    exact_j = SemanticArchive("close", fn, {}, Boundary(), _signature(fn))
    exact_spec = ScalarLoweringSpec.create(
        exact_j,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "out"},
    )
    exact_case = Gate2Case(
        "exact", exact_j, {"input": np.asarray([1.0], dtype=np.float32)},
        triton, exact_spec, ("out",)
    )
    exact_result = _run_fake_gate2(
        exact_case,
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.05], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    assert exact_result.status is Gate2Status.PAIRWISE_DIVERGED


def test_gate2_replay_j_edge_catches_nontransitive_approximation() -> None:
    fn = IntentFunction.from_json(
        {
            "name": "close",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [1]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [1]}],
            "returns": [{"output": "Y", "value": "X"}],
            "body": [],
        }
    )
    j = SemanticArchive(
        "close",
        fn,
        {},
        Boundary(result_relations=[ResultRelationSpec("Y", "tolerance")]),
        _signature(fn),
    )
    spec = ScalarLoweringSpec.create(
        j,
        dims={},
        input_bindings={"X": "input"},
        output_bindings={"Y": "out"},
    )
    triton = TritonExecutionSpec(
        (
            TritonArgument("p0", ArgumentRole.INPUT, value_key="input"),
            TritonArgument(
                "p1", ArgumentRole.OUTPUT, dtype="f32", shape=(1,),
                output_key="out",
            ),
        ),
        (1,),
    )
    case = Gate2Case(
        "nontransitive",
        j,
        {"input": np.asarray([0.0], dtype=np.float32)},
        triton,
        spec,
        ("out",),
    )
    q = ComparisonPolicy.create(
        "test.nontransitive.q.v1",
        (("f32", NumericTolerance(0.0, 1.0)),),
    )
    result = _run_fake_gate2(
        case,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        np.asarray([-0.9], dtype=np.float32),
        q,
    )
    assert result.status is Gate2Status.PAIRWISE_DIVERGED
    assert result.source_replay is not None and result.source_replay.passed
    assert result.source_j is not None and result.source_j.passed
    assert result.replay_j is not None and not result.replay_j.passed


def test_gate2_main_path_rejects_set_valued_result_relations() -> None:
    j = _update_j(
        duplicate_policy="one_of",
        order="unordered",
        relation="one_of_updates",
    )
    values = {
        "base": np.asarray([0, 0, 0], dtype=np.float32),
        "indices": np.asarray([1, 1], dtype=np.int32),
        "values": np.asarray([3, 7], dtype=np.float32),
    }
    case = _gate2_case(j, values)
    output = np.asarray([0, 3, 0], dtype=np.float32)
    result = _run_fake_gate2(case, output, output, output)
    assert result.status is Gate2Status.COMPARISON_UNSUPPORTED
    assert "set-valued" in result.source.error
    assert result.source_replay is None
    assert result.source_j is None
    assert result.replay_j is None


def test_gate2_rejects_invalid_unique_or_bounds_challenge_before_execution() -> None:
    j = _update_j()
    duplicate_case = _gate2_case(
        j,
        {
            "base": np.zeros(3, dtype=np.float32),
            "indices": np.asarray([1, 1], dtype=np.int32),
            "values": np.asarray([3, 7], dtype=np.float32),
        },
    )
    result = _run_fake_gate2(
        duplicate_case,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    assert result.status is Gate2Status.EXECUTION_FAILED
    assert "unique-index precondition" in result.source.error


def test_gate2_legs_receive_isolated_committed_prestate() -> None:
    j = _update_j()
    values = {
        "base": np.asarray([0, 0, 0], dtype=np.float32),
        "indices": np.asarray([1, 2], dtype=np.int32),
        "values": np.asarray([3, 7], dtype=np.float32),
    }
    case = _gate2_case(j, values)
    snapshots: list[np.ndarray] = []

    class MutatingRunner:
        def __init__(self, leg: str):
            self.leg = leg

        def run(self, *args):
            runtime_values = args[-1]
            snapshots.append(runtime_values["base"].copy())
            runtime_values["base"][0] = 99
            return ExecutionResult(
                self.leg,
                True,
                {"out": np.asarray([0, 3, 7], dtype=np.float32)},
            )

    class MutatingScalar:
        def run(self, spec, runtime_values):
            del spec
            snapshots.append(runtime_values["base"].copy())
            runtime_values["base"][0] = 88
            return ExecutionResult(
                "j",
                True,
                {"out": np.asarray([0, 3, 7], dtype=np.float32)},
            )

    runner = Gate2Runner(
        source_runner=MutatingRunner("source"),
        replay_runner=MutatingRunner("replay"),
        refeval_runner=MutatingScalar(),
    )
    result = runner.run(
        KernelSource.from_code("source", "source", trusted=True),
        KernelSource.from_code("replay", "replay", trusted=True),
        case,
    )
    assert result.status is Gate2Status.PASS
    assert all(np.array_equal(item, np.zeros(3, dtype=np.float32)) for item in snapshots)
    np.testing.assert_array_equal(values["base"], np.zeros(3, dtype=np.float32))


def test_cuda_analyzer_consumes_supported_update_contracts_and_rejects_one_of() -> None:
    unique = analyze_j_for_cuda(_update_j(rank2=True))
    assert unique["supported"]
    assert any(item.startswith("update:n0:overwrite:forbid") for item in unique["consumed"])

    combined = analyze_j_for_cuda(
        _update_j(
            combiner="add",
            duplicate_policy="combine",
            order="unordered",
            relation="tolerance",
            rank2=True,
        )
    )
    assert combined["supported"]
    assert any(item.startswith("update:n0:add:combine") for item in combined["consumed"])

    racing = analyze_j_for_cuda(
        _update_j(
            duplicate_policy="one_of",
            order="unordered",
            relation="one_of_updates",
            rank2=True,
        )
    )
    assert not racing["supported"]
    assert any("racing_overwrite_not_cuda_defined" in item for item in racing["unsupported"])
