"""Closed SCRC loop-state schema and scope tests."""

from __future__ import annotations

import copy
import json

import pytest

from intentir.ir import IntentFunction, LoopStateBinding
from intentir.verify.gate1 import Gate1Artifact, SchemaError, SemanticArchive


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
