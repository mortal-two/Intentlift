"""Closed SCRC loop-state schema and scope tests."""

from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from direct_fixtures import make_reduce_artifact
from intentir.backend.cpu.structured_reference import (
    STRUCTURED_CPU_REFERENCE_PROTOCOL,
    execute_structured_reference,
)
from intentir.ir import IntentFunction, LoopStateBinding
from intentir.verify.gate1 import Gate1Artifact, SchemaError, SemanticArchive
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.typecheck import TypeCheckErrorCode, type_check


def _loop_archive() -> dict:
    return {
        "kernel_id": "semantic_scan",
        "intent": {
            "intent_function": {
                "name": "semantic_scan",
                "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
                "outputs": [{"name": "y", "dtype": "f32", "shape": []}],
                "body": [
                    {
                        "op": "loop",
                        "scope_id": "scope0",
                        "result": ["acc"],
                        "carry": [
                            {
                                "state_id": "state0",
                                "init_value": "ZERO",
                                "body_argument": "acc_iter",
                                "next_value": "acc_next",
                                "result_value": "acc",
                                "type": {"dtype": "f32", "shape": []},
                            }
                        ],
                        "trip": {"axis": "N"},
                        "scan": [{"name": "x_tile", "from": "x", "axis": 0}],
                        "body": [
                            {
                                "op": "reduce",
                                "result": "tile_sum",
                                "operands": ["x_tile"],
                                "attrs": {"fn": "sum", "axes": [0]},
                            },
                            {
                                "op": "map_binary",
                                "result": "acc_next",
                                "operands": ["acc_iter", "tile_sum"],
                                "attrs": {"fn": "add"},
                                "type": {"dtype": "f32", "shape": []},
                            },
                        ],
                        "yield": ["acc_next"],
                    },
                    {
                        "op": "map_unary",
                        "result": "y",
                        "operands": ["acc"],
                        "attrs": {"fn": "identity"},
                    },
                ],
            }
        },
        "node_ids": {"n0": "body[0]", "n1": "body[1]"},
        "boundary": {
            "domains": {},
            "accesses": [
                {
                    "access_id": "a0",
                    "kind": "store",
                    "tensor": "y",
                    "node_ref": "n1",
                    "index_expression": "0",
                }
            ],
            "broadcasts": [],
            "aliases": [],
            "effects": [
                {
                    "effect_id": "e0",
                    "kind": "write",
                    "target": "y",
                    "node_ref": "n1",
                    "access_ref": "a0",
                }
            ],
            "output_mappings": [],
            "reductions": [],
            "index_bounds": [],
            "updates": [],
            "result_relations": [],
            "realization_authorizations": [],
        },
        "signature": {
            "parameters": [
                {
                    "name": "p0",
                    "kind": "pointer",
                    "mode": "input",
                    "dtype": "f32",
                    "rank": 1,
                    "logical_ref": "x",
                },
                {
                    "name": "p1",
                    "kind": "pointer",
                    "mode": "output",
                    "dtype": "f32",
                    "rank": 0,
                    "logical_ref": "y",
                },
            ]
        },
        "scopes": [
            {
                "scope_id": "function0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": ["x", "acc", "y"],
                "effects": ["e0"],
            },
            {
                "scope_id": "scope0",
                "kind": "loop_region",
                "parent_scope": "function0",
                "captures": ["x"],
                "definitions": [
                    "x_tile", "acc_iter", "tile_sum", "acc_next",
                ],
                "effects": [],
            },
        ],
        "storage": [
            {
                "storage_id": "storage0",
                "members": ["state0", "acc_iter", "acc_next", "acc"],
                "scope_id": "scope0",
                "mutability": "mutable",
                "init_event": {"kind": "state", "ref": "state0"},
                "last_use_event": {"kind": "node", "ref": "n1"},
                "escapes": True,
                "crosses_iteration": True,
            },
            {
                "storage_id": "storage1",
                "members": ["y"],
                "scope_id": "function0",
                "mutability": "write_once",
                "init_event": {"kind": "node", "ref": "n1"},
                "last_use_event": {
                    "kind": "scope_exit",
                    "ref": "function0",
                },
                "escapes": True,
                "crosses_iteration": False,
            },
        ],
    }


def _refeval_loop_archive() -> SemanticArchive:
    payload = _loop_archive()
    payload["boundary"]["accesses"] = []
    payload["boundary"]["effects"] = []
    payload["scopes"][0]["effects"] = []
    payload["storage"] = payload["storage"][:1]
    return SemanticArchive.from_json(payload)


def test_structured_refeval_executes_one_full_logical_axis_step() -> None:
    archive = _refeval_loop_archive()
    spec = ScalarLoweringSpec.create(
        archive,
        dims={"N": 5},
        input_bindings={"x": "input"},
        output_bindings={"y": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec, {"input": np.arange(5, dtype=np.float32)}
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"], np.asarray(10, dtype=np.float32)
    )
    evidence = result.metadata["execution_evidence"]
    assert evidence["protocol"] == STRUCTURED_CPU_REFERENCE_PROTOCOL
    assert evidence["authority"] == "canonical_J_only"
    assert evidence["logical_loop_step_count"] == 1
    assert result.metadata["boundary_profile"] == "structured_value_v1"
    assert "storage:storage0" in result.metadata["boundary_projection"]["consumed"]
    assert set(inspect.signature(execute_structured_reference).parameters) == {
        "j", "dims", "inputs", "broadcast_axes",
    }


def test_structured_refeval_zero_trip_returns_typed_initializer() -> None:
    archive = _refeval_loop_archive()
    spec = ScalarLoweringSpec.create(
        archive,
        dims={"N": 0},
        input_bindings={"x": "input"},
        output_bindings={"y": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec, {"input": np.empty((0,), dtype=np.float32)}
    )

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"], np.asarray(0, dtype=np.float32)
    )
    evidence = result.metadata["execution_evidence"]
    assert evidence["logical_loop_step_count"] == 0
    assert evidence["zero_trip_loop_count"] == 1


def test_structured_refeval_executes_closed_flat_scan() -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    scan = artifact.j.intent.body[1]
    scan.op = "scan"
    scan.attrs = {"fn": "sum", "axis": 1}
    scan.type = {"dtype": "f32", "shape": ["x0", "x1"]}
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[1].rank = 2
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={"x0": 2, "x1": 3},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    value = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = ScalarLoweringRunner().run(spec, {"input": value})

    assert result.ok, result.error
    np.testing.assert_array_equal(
        result.outputs["output"], np.cumsum(value * value, axis=1)
    )
    assert result.metadata["execution_evidence"]["scan_count"] == 1


def test_structured_refeval_rejects_unsupported_body_op_explicitly() -> None:
    payload = _loop_archive()
    payload["boundary"]["accesses"] = []
    payload["boundary"]["effects"] = []
    payload["scopes"][0]["effects"] = []
    payload["storage"] = payload["storage"][:1]
    payload["intent"]["intent_function"]["body"][0]["body"][1]["op"] = (
        "future_recurrence"
    )
    payload["intent"]["intent_function"]["body"][0]["body"][1]["attrs"] = {}
    archive = SemanticArchive.from_json(payload)
    spec = ScalarLoweringSpec.create(
        archive,
        dims={"N": 3},
        input_bindings={"x": "input"},
        output_bindings={"y": "output"},
    )

    result = ScalarLoweringRunner().run(
        spec, {"input": np.arange(3, dtype=np.float32)}
    )

    assert not result.ok
    assert result.unsupported
    assert "future_recurrence" in (result.error or "")


def _tiled_loop_artifact(*, basis: str = "extent_step") -> Gate1Artifact:
    archive = _loop_archive()
    archive["intent"]["intent_function"]["inputs"][0]["shape"] = ["x0"]
    archive["intent"]["intent_function"]["body"][0]["trip"] = {
        "axis": "x0"
    }
    archive["intent"]["intent_function"]["body"][0]["body"][0]["attrs"][
        "assoc"
    ] = "float_nondeterministic"
    archive["signature"]["parameters"].append(
        {
            "name": "p2",
            "kind": "scalar",
            "mode": "value",
            "dtype": "i32",
            "logical_ref": "x0",
        }
    )
    archive["boundary"]["result_relations"] = [
        {"output": "y", "kind": "tolerance"}
    ]
    archive["boundary"]["consumer_extensions"] = []
    archive["dimension_constraints"] = []
    return Gate1Artifact.from_json(
        {
            "schema": Gate1Artifact.SCHEMA,
            "j": archive,
            "c": {
                "slots": [
                    {
                        "name": "c0",
                        "kind": "constexpr",
                        "role": "tile_extent",
                        "value_type": "int",
                        "required": True,
                        "subject_ref": "n0",
                        "axis_ref": "x0",
                    }
                ],
                "values": {"c0": 32},
            },
            "pi": {
                "regions": [
                    {"id": "r0", "parent": None, "placement": "root"},
                    {"id": "r1", "parent": "r0", "placement": "program"},
                    {"id": "r2", "parent": "r1", "placement": "loop"},
                ],
                "instances": [
                    {
                        "id": "i0",
                        "semantic_ref": "n0",
                        "region": "r0",
                        "mechanism": "realize",
                    },
                    {
                        "id": "i1",
                        "semantic_ref": "c0",
                        "region": "r1",
                        "mechanism": "realize",
                    },
                ],
                "routes": [],
                "order": [],
                "sharing": [],
                "refinements": [],
                "iterations": [
                    {
                        "region": "r2",
                        "loop_instance": "i0",
                        "tile_instance": "i1",
                        "induction_basis": basis,
                    }
                ],
                "loop_carriers": [],
            },
        }
    )


def test_closed_loop_state_round_trips_as_typed_intentir() -> None:
    archive = SemanticArchive.from_json(_loop_archive())
    loop = archive.intent.body[0]

    assert loop.scope_id == "scope0"
    assert isinstance(loop.carry[0], LoopStateBinding)
    assert loop.carry[0].body_argument == "acc_iter"
    assert loop.carry[0].next_value == "acc_next"
    assert loop.carry[0].result_value == "acc"
    assert loop.carry[0].type is not None
    assert loop.carry[0].type.dtype == "f32"
    assert SemanticArchive.from_json(archive.to_json()).to_json() == archive.to_json()


@pytest.mark.parametrize("axis_ref", [0, "x0"])
@pytest.mark.parametrize("basis", ["extent_step", "tile_ordinal"])
def test_tiled_loop_iteration_basis_is_typed_and_round_trips(
    basis: str,
    axis_ref: int | str,
) -> None:
    artifact = _tiled_loop_artifact(basis=basis)
    artifact.c.slots[0].axis_ref = axis_ref

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not {
        TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
        TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
    }.intersection(item.code for item in report.diagnostics)
    assert artifact.to_json()["pi"]["iterations"][0]["induction_basis"] == basis
    assert Gate1Artifact.from_json(artifact.to_json()) == artifact


def test_explicit_iteration_relation_does_not_depend_on_free_text_role() -> None:
    artifact = _tiled_loop_artifact()
    artifact.c.slots[0].role = "chunk_extent"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not {
        TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
        TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
    }.intersection(item.code for item in report.diagnostics)


def test_unreserved_role_does_not_claim_iteration_totality() -> None:
    artifact = _tiled_loop_artifact()
    artifact.c.slots[0].role = "chunk_extent"
    artifact.pi.iterations = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.ITERATION_WITNESS_MISSING not in {
        item.code for item in report.diagnostics
    }


def test_tiled_loop_without_iteration_basis_is_fail_closed() -> None:
    artifact = _tiled_loop_artifact()
    artifact.pi.iterations = []
    artifact.wire_schema = "intentir.scrc.artifact.v2"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.ITERATION_WITNESS_MISSING
    ]
    assert [item.path for item in matching] == ["pi.iterations"]
    assert matching[0].ref == "n0:c0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "r0"),
        ("loop_instance", "i1"),
        ("tile_instance", "i0"),
    ],
)
def test_tiled_loop_iteration_relation_rejects_wrong_dependent_anchor(
    field: str,
    value: str,
) -> None:
    artifact = _tiled_loop_artifact()
    artifact.pi.iterations[0][field] = value

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.ITERATION_WITNESS_INVALID in {
        item.code for item in report.diagnostics
    }


def test_tiled_loop_iteration_relation_requires_positive_integer_tile() -> None:
    artifact = _tiled_loop_artifact()
    artifact.c.values["c0"] = 0

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.ITERATION_WITNESS_INVALID in {
        item.code for item in report.diagnostics
    }


def test_iteration_relation_rejects_an_anchor_from_a_sibling_region() -> None:
    artifact = _tiled_loop_artifact()
    artifact.pi.regions.append(
        {"id": "r3", "parent": "r0", "placement": "program"}
    )
    artifact.pi.instances[1]["region"] = "r3"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.ITERATION_WITNESS_INVALID in {
        item.code for item in report.diagnostics
    }


def test_one_tiled_j_loop_can_have_distinct_physical_loop_relations() -> None:
    artifact = _tiled_loop_artifact()
    artifact.pi.regions.append(
        {"id": "r3", "parent": "r1", "placement": "loop"}
    )
    artifact.pi.iterations.append(
        {
            "region": "r3",
            "loop_instance": "i0",
            "tile_instance": "i1",
            "induction_basis": "tile_ordinal",
        }
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not {
        TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
        TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
    }.intersection(item.code for item in report.diagnostics)


def test_untiled_loop_does_not_require_iteration_relation() -> None:
    artifact = _tiled_loop_artifact()
    artifact.c.slots = []
    artifact.c.values = {}
    artifact.pi.instances = artifact.pi.instances[:1]
    artifact.pi.iterations = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not {
        TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
        TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
    }.intersection(item.code for item in report.diagnostics)


def test_artifact_v3_requires_explicit_iterations_field() -> None:
    payload = _tiled_loop_artifact().to_json()
    del payload["pi"]["iterations"]

    with pytest.raises(SchemaError, match="missing required field: iterations"):
        Gate1Artifact.from_json(payload)


def test_iteration_basis_is_a_closed_enum() -> None:
    payload = _tiled_loop_artifact().to_json()
    payload["pi"]["iterations"][0]["induction_basis"] = "source_range"

    with pytest.raises(SchemaError, match="unsupported pi induction basis"):
        Gate1Artifact.from_json(payload)


@pytest.mark.parametrize("invalid_init", [{"value": 0.0}, [0.0]])
def test_loop_init_value_rejects_structured_json_with_schema_error(
    invalid_init,
) -> None:
    payload = copy.deepcopy(_loop_archive())
    payload["intent"]["intent_function"]["body"][0]["carry"][0][
        "init_value"
    ] = invalid_init

    with pytest.raises(
        SchemaError,
        match="init_value must be a scalar literal or a visible value reference",
    ):
        SemanticArchive.from_json(payload)


def test_loop_lowering_rejects_an_unsupported_carry_dtype_instead_of_silently_f32() -> None:
    payload = _loop_archive()
    loop = payload["intent"]["intent_function"]["body"][0]
    loop["carry"][0]["type"] = {"dtype": "i32", "shape": []}
    loop["body"][1]["type"] = {"dtype": "i32", "shape": []}
    archive = SemanticArchive.from_json(payload)

    from intentir.frontend import emit_loop_module

    with pytest.raises(NotImplementedError, match="supports f32 carry"):
        emit_loop_module(archive.intent, {"N": 8}, fold=0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda loop: loop["carry"][0].pop("type"),
            "missing required key.*type",
        ),
        (
            lambda loop: loop.update({"result": ["wrong"]}),
            "result must exactly match carry.result_value",
        ),
        (
            lambda loop: loop.update({"yield": ["wrong"]}),
            "yield must exactly match carry.next_value",
        ),
        (
            lambda loop: (
                loop["carry"][0].update({"next_value": "never_defined"}),
                loop.update({"yield": ["never_defined"]}),
            ),
            "next_value must be defined in the loop body",
        ),
        (
            lambda loop: loop["carry"][0].update({"body_argument": "x"}),
            "body_argument must be local",
        ),
        (
            lambda loop: loop["trip"].update({"tile_hint": "BLOCK_N"}),
            "trip has unknown key.*tile_hint",
        ),
        (
            lambda loop: loop["scan"][0].update({"axis": 1}),
            "axis is outside source tensor rank",
        ),
    ],
)
def test_closed_loop_state_rejects_incomplete_or_inconsistent_contracts(
    mutate, message: str,
) -> None:
    payload = _loop_archive()
    mutate(payload["intent"]["intent_function"]["body"][0])

    with pytest.raises(SchemaError, match=message):
        SemanticArchive.from_json(payload)


def test_loop_body_argument_cannot_escape_its_scope() -> None:
    payload = _loop_archive()
    payload["intent"]["intent_function"]["body"][1]["operands"] = ["acc_iter"]

    with pytest.raises(SchemaError, match="outside the current scope"):
        SemanticArchive.from_json(payload)


def test_scope_captures_definitions_and_effects_are_exact() -> None:
    extra_capture = _loop_archive()
    extra_capture["scopes"][1]["captures"].append("y")
    with pytest.raises(SchemaError, match="captures must exactly match"):
        SemanticArchive.from_json(extra_capture)

    missing_definition = _loop_archive()
    missing_definition["scopes"][1]["definitions"].remove("acc_next")
    with pytest.raises(SchemaError, match="definitions must exactly match"):
        SemanticArchive.from_json(missing_definition)

    orphan_branch = _loop_archive()
    orphan_branch["scopes"].append(
        {
            "scope_id": "branch0",
            "kind": "branch_region",
            "parent_scope": "function0",
            "captures": [],
            "definitions": [],
            "effects": [],
        }
    )
    with pytest.raises(SchemaError, match="first-class IntentIR branch-region op"):
        SemanticArchive.from_json(orphan_branch)

    nested_program = _loop_archive()
    nested_program["scopes"].append(
        {
            "scope_id": "program0",
            "kind": "program_instance",
            "parent_scope": "scope0",
            "captures": [],
            "definitions": [],
            "effects": [],
        }
    )
    with pytest.raises(SchemaError, match="unique direct child"):
        SemanticArchive.from_json(nested_program)


def _loop_archive_with_store_effect() -> dict:
    return _loop_archive()


def test_access_effect_identity_scope_and_write_once_are_cross_checked() -> None:
    archive = SemanticArchive.from_json(_loop_archive_with_store_effect())
    assert archive.boundary.effects[0].effect_id == "e0"

    missing_effect = _loop_archive_with_store_effect()
    missing_effect["boundary"]["effects"] = []
    missing_effect["scopes"][0]["effects"] = []
    with pytest.raises(SchemaError, match="requires exactly one matching EffectSpec"):
        SemanticArchive.from_json(missing_effect)

    wrong_scope = _loop_archive_with_store_effect()
    wrong_scope["scopes"][0]["effects"] = []
    wrong_scope["scopes"][1]["effects"] = ["e0"]
    with pytest.raises(SchemaError, match="effects must exactly match"):
        SemanticArchive.from_json(wrong_scope)

    immutable_write = _loop_archive_with_store_effect()
    immutable_write["storage"][1]["mutability"] = "immutable"
    with pytest.raises(SchemaError, match="immutable storage.*write event"):
        SemanticArchive.from_json(immutable_write)


def test_storage_lifetime_and_loop_state_membership_are_fail_closed() -> None:
    missing_member = _loop_archive()
    missing_member["storage"][0]["members"].remove("acc_next")
    with pytest.raises(SchemaError, match="must bind init, body argument"):
        SemanticArchive.from_json(missing_member)

    wrong_event_namespace = _loop_archive()
    wrong_event_namespace["storage"][0]["init_event"] = {
        "kind": "state",
        "ref": "n0",
    }
    with pytest.raises(SchemaError, match="state event references unknown ref"):
        SemanticArchive.from_json(wrong_event_namespace)

    truncated_lifetime = _loop_archive()
    truncated_lifetime["storage"][0]["last_use_event"] = {
        "kind": "state",
        "ref": "state0",
    }
    with pytest.raises(SchemaError, match="use outside declared lifetime"):
        SemanticArchive.from_json(truncated_lifetime)

    unrelated_last = _loop_archive_with_store_effect()
    unrelated_last["storage"][0]["last_use_event"] = {
        "kind": "effect",
        "ref": "e0",
    }
    with pytest.raises(SchemaError, match="last_use_event is unrelated"):
        SemanticArchive.from_json(unrelated_last)

    unrelated_init = _loop_archive_with_store_effect()
    unrelated_init["storage"][1]["init_event"] = {
        "kind": "state",
        "ref": "state0",
    }
    with pytest.raises(SchemaError, match="init_event is unrelated"):
        SemanticArchive.from_json(unrelated_init)


def test_scope_state_effect_and_storage_ids_are_opaque_in_writer_payload() -> None:
    archive = SemanticArchive.from_json(_loop_archive_with_store_effect())
    artifact = Gate1Artifact.from_json(
        {
            "schema": Gate1Artifact.SCHEMA,
            "j": archive.to_json(),
            "c": {"slots": [], "values": {}},
            "pi": {
                "regions": [
                    {"id": "r0", "parent": None, "placement": "root"}
                ],
                "instances": [
                    {
                        "id": "i0",
                        "semantic_ref": "n0",
                        "region": "r0",
                        "mechanism": "realize",
                    },
                    {
                        "id": "i1",
                        "semantic_ref": "n1",
                        "region": "r0",
                        "mechanism": "realize",
                    },
                ],
                "routes": [
                    {"source": "i0", "target": "i1", "slots": [0]}
                ],
                "order": [{"before": "i0", "after": "i1"}],
                "sharing": [],
                    "refinements": [],
                    "iterations": [],
                    "loop_carriers": [],
                },
        }
    )

    view = artifact.rewriter_view()
    rendered = json.dumps(view, sort_keys=True)
    for leaked in (
        "scope0", "function0", "state0", "storage0", "storage1",
        "acc_iter", "acc_next",
    ):
        assert leaked not in rendered
    assert view["j"]["scopes"][0]["scope_id"].startswith("q")
    assert view["j"]["storage"][0]["storage_id"].startswith("b")
    assert view["j"]["boundary"]["effects"][0]["effect_id"].startswith("e")

def _aliased_destination_archive(alias_kind: str = "must_alias") -> dict:
    return {
        "kernel_id": "aliased_destination",
        "intent": {
            "intent_function": {
                "name": "aliased_destination",
                "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
                "outputs": [{"name": "y", "dtype": "f32", "shape": ["N"]}],
                "body": [
                    {
                        "op": "cast",
                        "result": "y",
                        "operands": ["x"],
                        "attrs": {"to_dtype": "f32"},
                    }
                ],
            }
        },
        "node_ids": {"n0": "body[0]"},
        "boundary": {
            "domains": {},
            "accesses": [
                {
                    "access_id": "a0",
                    "kind": "store",
                    "tensor": "y",
                    "node_ref": "n0",
                    "index_expression": "0",
                }
            ],
            "broadcasts": [],
            "aliases": [{"left": "x", "right": "y", "kind": alias_kind}],
            "effects": [
                {
                    "effect_id": "e0",
                    "kind": "write",
                    "target": "y",
                    "node_ref": "n0",
                    "access_ref": "a0",
                }
            ],
            "output_mappings": [],
            "reductions": [],
            "index_bounds": [],
            "updates": [],
            "result_relations": [],
            "realization_authorizations": [],
        },
        "signature": {
            "parameters": [
                {
                    "name": "p0",
                    "kind": "pointer",
                    "mode": "input",
                    "dtype": "f32",
                    "rank": 1,
                    "logical_ref": "x",
                },
                {
                    "name": "p1",
                    "kind": "pointer",
                    "mode": "output",
                    "dtype": "f32",
                    "rank": 1,
                    "alias_of": "p0",
                    "logical_ref": "y",
                },
            ]
        },
        "scopes": [
            {
                "scope_id": "function0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": ["x", "y"],
                "effects": ["e0"],
            }
        ],
        "storage": [
            {
                "storage_id": "storage0",
                "members": ["x", "y"],
                "scope_id": "function0",
                "mutability": "write_once",
                "init_event": {"kind": "scope_entry", "ref": "function0"},
                "last_use_event": {"kind": "scope_exit", "ref": "function0"},
                "escapes": True,
                "crosses_iteration": False,
            }
        ],
    }


def test_storage_object_is_the_alias_identity() -> None:
    parsed = SemanticArchive.from_json(_aliased_destination_archive())
    assert parsed.storage[0].members == ["x", "y"]

    no_alias_same_storage = _aliased_destination_archive("no_alias")
    with pytest.raises(SchemaError, match="no_alias endpoints"):
        SemanticArchive.from_json(no_alias_same_storage)

    split_must_alias = _aliased_destination_archive()
    split_must_alias["storage"] = [
        {
            **split_must_alias["storage"][0],
            "members": ["x"],
            "storage_id": "storage0",
            "mutability": "immutable",
            "last_use_event": {"kind": "node", "ref": "n0"},
            "escapes": False,
        },
        {
            **split_must_alias["storage"][0],
            "members": ["y"],
            "storage_id": "storage1",
            "init_event": {"kind": "node", "ref": "n0"},
        },
    ]
    with pytest.raises(SchemaError, match="must_alias endpoints"):
        SemanticArchive.from_json(split_must_alias)


def test_write_once_counts_distinct_access_events_even_at_one_node() -> None:
    payload = _aliased_destination_archive()
    payload["boundary"]["accesses"].append(
        {
            **payload["boundary"]["accesses"][0],
            "access_id": "a1",
        }
    )
    payload["boundary"]["effects"].append(
        {
            **payload["boundary"]["effects"][0],
            "effect_id": "e1",
            "access_ref": "a1",
        }
    )
    payload["scopes"][0]["effects"] = ["e0", "e1"]

    with pytest.raises(SchemaError, match="write_once storage.*exactly one"):
        SemanticArchive.from_json(payload)


def test_node_ids_are_canonical_json_paths_not_free_text_channels() -> None:
    payload = _loop_archive()
    payload["node_ids"]["n0"] = "SECRET/body[0]/helper"

    with pytest.raises(SchemaError, match="canonical.*JSON path"):
        SemanticArchive.from_json(payload)


def test_direct_intent_model_rejects_legacy_loop_shape() -> None:
    payload = _loop_archive()["intent"]
    loop = payload["intent_function"]["body"][0]
    loop["trip"] = {"axis": "N", "tile_hint": "BLOCK_N"}

    with pytest.raises(ValueError, match="tile_hint"):
        IntentFunction.from_json(copy.deepcopy(payload))

    with pytest.raises(SchemaError, match="tile_hint"):
        SemanticArchive.from_json({**_loop_archive(), "intent": payload})
