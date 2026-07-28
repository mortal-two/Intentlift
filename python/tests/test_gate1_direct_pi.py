from __future__ import annotations

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.model import IntentFunction
from intentir.extraction.lifters import scrc
from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCReliftContext,
    anonymize_source,
    prepare_scrc_lift,
)
from intentir.extraction.providers.client import LLMReply
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import (
    Frontend,
    Gate1Artifact,
    ParameterKind,
    ScheduleSlot,
    SchemaError,
)
from intentir.verify.gate1.orchestrate import canonical_artifact_digest
from intentir.verify.gate1.pi import (
    REALIZATION_MECHANISMS,
    RealizationWitness,
    RealizationWitnessError,
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)
from intentir.verify.gate1.prompt import (
    PUBLIC_FRONTEND_CONTEXT,
    PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY,
    PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY,
    ReplayPacket,
    assert_source_isolated_payload,
    audit_public_frontend_context_payload,
    audit_public_frontend_context_receipt,
    audit_replay_packet_digest,
    build_public_frontend_context_receipt,
    build_normal_rewriter_prompt,
    build_masked_rewriter_view,
    build_normal_rewriter_view,
    build_rewriter_packet,
    public_frontend_context_digest,
    replay_packet_digest,
)
from intentir.verify.typecheck import type_check




def test_direct_pi_artifact_roundtrip_and_legacy_interface_break() -> None:
    artifact = make_reduce_artifact()
    encoded = artifact.to_json()
    assert Gate1Artifact.from_json(encoded) == artifact
    assert encoded["schema"] == Gate1Artifact.SCHEMA

    legacy = copy.deepcopy(encoded)
    legacy["interface"] = {"parameters": []}
    with pytest.raises(SchemaError, match=r"unknown key\(s\): interface"):
        Gate1Artifact.from_json(legacy)


def test_artifact_v10_preserves_v9_through_v1_legacy_contracts() -> None:
    current = make_reduce_artifact().to_json()
    assert current["schema"] == "intentir.scrc.artifact.v10"

    legacy_v9 = copy.deepcopy(current)
    legacy_v9["schema"] = "intentir.scrc.artifact.v9"
    del legacy_v9["pi"]["unit_iterations"]
    parsed_v9 = Gate1Artifact.from_json(legacy_v9)
    assert parsed_v9.to_json() == legacy_v9

    legacy_v8 = copy.deepcopy(legacy_v9)
    legacy_v8["schema"] = "intentir.scrc.artifact.v8"
    del legacy_v8["pi"]["terminal_controls"]
    parsed_v8 = Gate1Artifact.from_json(legacy_v8)
    assert parsed_v8.to_json() == legacy_v8

    legacy_v7 = copy.deepcopy(legacy_v8)
    legacy_v7["schema"] = "intentir.scrc.artifact.v7"
    del legacy_v7["pi"]["physical_iterations"]
    del legacy_v7["pi"]["physical_loop_carriers"]
    parsed_v7 = Gate1Artifact.from_json(legacy_v7)
    assert parsed_v7.to_json() == legacy_v7

    legacy_v6 = copy.deepcopy(legacy_v7)
    legacy_v6["schema"] = "intentir.scrc.artifact.v6"
    parsed_v6 = Gate1Artifact.from_json(legacy_v6)
    assert parsed_v6.to_json() == legacy_v6

    legacy_v5 = copy.deepcopy(legacy_v6)
    legacy_v5["schema"] = "intentir.scrc.artifact.v5"
    parsed_v5 = Gate1Artifact.from_json(legacy_v5)
    assert parsed_v5.to_json() == legacy_v5

    legacy_v4 = copy.deepcopy(legacy_v5)
    legacy_v4["schema"] = "intentir.scrc.artifact.v4"
    del legacy_v4["j"]["boundary"]["access_guard_relations"]
    parsed_v4 = Gate1Artifact.from_json(legacy_v4)
    assert parsed_v4.to_json() == legacy_v4

    legacy_v3 = copy.deepcopy(legacy_v4)
    legacy_v3["schema"] = "intentir.scrc.artifact.v3"
    del legacy_v3["j"]["dimension_constraints"]
    del legacy_v3["j"]["value_dimension_relations"]
    del legacy_v3["j"]["boundary"]["consumer_extensions"]
    del legacy_v3["pi"]["loop_carriers"]
    parsed_v3 = Gate1Artifact.from_json(legacy_v3)
    assert parsed_v3.to_json() == legacy_v3

    legacy_v2 = copy.deepcopy(legacy_v3)
    legacy_v2["schema"] = "intentir.scrc.artifact.v2"
    del legacy_v2["pi"]["iterations"]
    parsed_v2 = Gate1Artifact.from_json(legacy_v2)
    assert parsed_v2.to_json() == legacy_v2
    expected_payload = {
        **legacy_v2,
        "observer_level": 2,
        "protocol": "llm_direct_pi_gate1_level2_v1",
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert canonical_artifact_digest(parsed_v2) == expected_digest

    for legacy_schema in (
        "intentir.scrc.artifact.v1",
        "intentir.scrc.artifact.v2",
    ):
        mislabeled_legacy = copy.deepcopy(legacy_v2)
        mislabeled_legacy["schema"] = legacy_schema
        mislabeled_legacy["pi"]["iterations"] = []
        with pytest.raises(SchemaError, match="do not contain pi iterations"):
            Gate1Artifact.from_json(mislabeled_legacy)

    legacy = copy.deepcopy(legacy_v2)
    legacy["schema"] = "intentir.scrc.artifact.v1"
    parsed = Gate1Artifact.from_json(legacy)
    assert parsed.to_json() == legacy
    expected_legacy_payload = {
        **legacy,
        "observer_level": 2,
        "protocol": "llm_direct_pi_gate1_level2_v1",
    }
    expected_legacy_digest = hashlib.sha256(
        json.dumps(
            expected_legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert canonical_artifact_digest(parsed) == expected_legacy_digest
    assert canonical_artifact_digest(parsed) != canonical_artifact_digest(
        make_reduce_artifact()
    )

    legacy["c"]["slots"][0]["subject_ref"] = "n0"
    with pytest.raises(SchemaError, match="do not contain C attachments"):
        Gate1Artifact.from_json(legacy)


@pytest.mark.parametrize(
    "wire_schema",
    [
        "intentir.scrc.artifact.v1",
        "intentir.scrc.artifact.v2",
        "intentir.scrc.artifact.v3",
        "intentir.scrc.artifact.v4",
        "intentir.scrc.artifact.v5",
        "intentir.scrc.artifact.v6",
        "intentir.scrc.artifact.v7",
    ],
)
def test_every_artifact_wire_version_requires_explicit_pi_refinements(
    wire_schema: str,
) -> None:
    payload = make_reduce_artifact().to_json()
    payload["schema"] = wire_schema
    del payload["pi"]["unit_iterations"]
    del payload["pi"]["terminal_controls"]
    del payload["pi"]["physical_iterations"]
    del payload["pi"]["physical_loop_carriers"]
    if wire_schema in {
        "intentir.scrc.artifact.v1",
        "intentir.scrc.artifact.v2",
        "intentir.scrc.artifact.v3",
    }:
        del payload["j"]["dimension_constraints"]
        del payload["j"]["value_dimension_relations"]
        del payload["j"]["boundary"]["consumer_extensions"]
        del payload["pi"]["loop_carriers"]
    if wire_schema in {
        "intentir.scrc.artifact.v1",
        "intentir.scrc.artifact.v2",
        "intentir.scrc.artifact.v3",
        "intentir.scrc.artifact.v4",
    }:
        del payload["j"]["boundary"]["access_guard_relations"]
    if wire_schema in {
        "intentir.scrc.artifact.v1",
        "intentir.scrc.artifact.v2",
    }:
        del payload["pi"]["iterations"]
    del payload["pi"]["refinements"]

    with pytest.raises(
        SchemaError,
        match="Gate1Artifact.pi is missing required field: refinements",
    ):
        Gate1Artifact.from_json(payload)


def test_standalone_pi_decoder_requires_explicit_refinements() -> None:
    payload = make_reduce_artifact().pi.to_json()
    del payload["refinements"]

    with pytest.raises(
        RealizationWitnessError,
        match=r"pi is missing fields: \['refinements'\]",
    ):
        RealizationWitness.from_json(payload)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"operator": "mul"},
        {"predicate": "p0 < p1"},
        {"value": 0},
        {"api": "tl.load"},
        {"text": "emit source here"},
    ],
)
def test_pi_rejects_semantic_atoms_and_source_text(forbidden: dict) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["instances"][0].update(forbidden)
    with pytest.raises(RealizationWitnessError, match="forbidden fields"):
        RealizationWitness.from_json(payload)


def test_pi_parser_is_separate_from_dependent_refs_and_rejects_cycles() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = "n99"
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.pi.instances[0]["semantic_ref"] == "n99"

    pi = artifact.pi.to_json()
    pi["order"].append({"before": "i3", "after": "i0"})
    with pytest.raises(RealizationWitnessError, match="acyclic"):
        RealizationWitness.from_json(pi)


@pytest.mark.parametrize(
    ("relation", "field", "bad_ref", "expected_path"),
    [
        ("routes", "source", "p0", r"pi\.routes\[0\]\.source"),
        ("routes", "target", "c0", r"pi\.routes\[0\]\.target"),
        ("order", "before", "n0", r"pi\.order\[0\]\.before"),
        ("order", "after", "p0", r"pi\.order\[0\]\.after"),
        ("sharing", "source", "c0", r"pi\.sharing\[0\]\.source"),
    ],
)
def test_pi_graph_endpoints_reject_raw_anchor_namespaces(
    relation: str,
    field: str,
    bad_ref: str,
    expected_path: str,
) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload[relation][0][field] = bad_ref
    with pytest.raises(
        RealizationWitnessError,
        match=expected_path + r" must reference a declared pi instance id i\*",
    ):
        RealizationWitness.from_json(payload)


def test_pi_sharing_consumer_rejects_raw_anchor_namespace() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["sharing"][0]["consumers"][0] = "p0"
    with pytest.raises(
        RealizationWitnessError,
        match=(
            r"pi\.sharing\[0\]\.consumers\[0\] must reference a declared "
            r"pi instance id i\*"
        ),
    ):
        RealizationWitness.from_json(payload)


@pytest.mark.parametrize(
    ("relation", "field", "expected_path"),
    [
        ("routes", "source", r"pi\.routes\[0\]\.source"),
        ("routes", "target", r"pi\.routes\[0\]\.target"),
        ("order", "before", r"pi\.order\[0\]\.before"),
        ("order", "after", r"pi\.order\[0\]\.after"),
        ("sharing", "source", r"pi\.sharing\[0\]\.source"),
    ],
)
def test_pi_graph_endpoints_reject_undeclared_instance_ids(
    relation: str,
    field: str,
    expected_path: str,
) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload[relation][0][field] = "i99"
    with pytest.raises(
        RealizationWitnessError,
        match=expected_path + r" references undeclared pi instance 'i99'",
    ):
        RealizationWitness.from_json(payload)


def test_pi_sharing_consumer_rejects_undeclared_instance_id() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["sharing"][0]["consumers"][0] = "i99"
    with pytest.raises(
        RealizationWitnessError,
        match=(
            r"pi\.sharing\[0\]\.consumers\[0\] references undeclared "
            r"pi instance 'i99'"
        ),
    ):
        RealizationWitness.from_json(payload)


def test_pi_rejects_graph_instance_id_in_anchor_field() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["instances"][0]["semantic_ref"] = "i0"
    with pytest.raises(
        RealizationWitnessError,
        match=r"pi\.instances\[0\]\.semantic_ref must be an anonymous",
    ):
        RealizationWitness.from_json(payload)


def test_pi_rejects_duplicate_instance_and_sharing_consumer_ids() -> None:
    duplicate_instance = make_reduce_artifact().pi.to_json()
    duplicate_instance["instances"][1]["id"] = "i0"
    with pytest.raises(
        RealizationWitnessError,
        match="declaration-order handles",
    ):
        RealizationWitness.from_json(duplicate_instance)

    duplicate_consumer = make_reduce_artifact().pi.to_json()
    duplicate_consumer["sharing"][0]["consumers"] = ["i1", "i1"]
    with pytest.raises(
        RealizationWitnessError,
        match=r"pi\.sharing\[0\]\.consumers must not contain duplicates",
    ):
        RealizationWitness.from_json(duplicate_consumer)


@pytest.mark.parametrize("semantic_ref", ["v0", "x0", "c0", "p0"])
def test_pi_may_reference_any_declared_anonymous_j_or_c_handle(
    semantic_ref: str,
) -> None:
    payload = make_reduce_artifact().to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = semantic_ref
    assert Gate1Artifact.from_json(payload).pi.instances[0]["semantic_ref"] == semantic_ref


def _row_stride_pi() -> RealizationWitness:
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": [
                {"id": "i0", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i1", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i2", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i3", "semantic_ref": "a0", "region": "r1", "mechanism": "realize"},
                {"id": "i4", "semantic_ref": "a0", "region": "r1", "mechanism": "transfer_direct"},
                {"id": "i5", "semantic_ref": "a1", "region": "r1", "mechanism": "realize"},
                {"id": "i6", "semantic_ref": "a1", "region": "r1", "mechanism": "transfer_direct"},
                {"id": "i7", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
                {"id": "i8", "semantic_ref": "p1", "region": "r1", "mechanism": "realize"},
            ],
            "routes": [
                {"source": "i0", "target": "i1", "slots": [0]},
                {"source": "i1", "target": "i2", "slots": [0]},
                {"source": "i2", "target": "i3", "slots": [0]},
                {"source": "i3", "target": "i4", "slots": [0]},
                {"source": "i2", "target": "i5", "slots": [0]},
                {"source": "i5", "target": "i6", "slots": [0]},
                {"source": "i7", "target": "i4", "slots": [1]},
                {"source": "i8", "target": "i6", "slots": [1]},
            ],
            "order": [],
            "sharing": [{"source": "i2", "consumers": ["i3", "i5"]}],
            "refinements": [],
        }
    )


def _flat_pi() -> RealizationWitness:
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": [
                {"id": "i0", "semantic_ref": "a0", "region": "r1", "mechanism": "transfer_direct"},
                {"id": "i1", "semantic_ref": "a1", "region": "r1", "mechanism": "transfer_direct"},
            ],
            "routes": [],
            "order": [],
            "sharing": [],
            "refinements": [],
        }
    )


def _address_pi(*, staged: bool) -> RealizationWitness:
    instances = [
        {"id": "i0", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
    ]
    if staged:
        instances.extend(
            [
                {
                    "id": "i1",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_base",
                },
                {
                    "id": "i2",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_lane",
                },
                {
                    "id": "i3",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "transfer_direct",
                },
            ]
        )
        routes = [
            {"source": "i0", "target": "i1", "slots": [0]},
            {"source": "i1", "target": "i2", "slots": [0]},
            {"source": "i2", "target": "i3", "slots": [0]},
        ]
    else:
        instances.extend(
            [
                {
                    "id": "i1",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_lane",
                },
                {
                    "id": "i2",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "transfer_direct",
                },
            ]
        )
        routes = [
            {"source": "i0", "target": "i1", "slots": [0]},
            {"source": "i1", "target": "i2", "slots": [0]},
        ]
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": instances,
            "routes": routes,
            "order": [],
            "sharing": [],
            "refinements": [],
        }
    )


def _integer_merge_address_pi() -> RealizationWitness:
    """Pointer bypasses integer base and enters the lane/final address stage."""

    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": [
                {
                    "id": "i0",
                    "semantic_ref": "p0",
                    "region": "r1",
                    "mechanism": "realize",
                },
                {
                    "id": "i1",
                    "semantic_ref": "d0",
                    "region": "r1",
                    "mechanism": "realize",
                },
                {
                    "id": "i2",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_base",
                },
                {
                    "id": "i3",
                    "semantic_ref": "c0",
                    "region": "r1",
                    "mechanism": "realize",
                },
                {
                    "id": "i4",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_lane",
                },
                {
                    "id": "i5",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "transfer_direct",
                },
            ],
            "routes": [
                {"source": "i1", "target": "i2", "slots": [0]},
                {"source": "i2", "target": "i4", "slots": [0]},
                {"source": "i3", "target": "i4", "slots": [1]},
                {"source": "i0", "target": "i4", "slots": [2]},
                {"source": "i4", "target": "i5", "slots": [0]},
            ],
            "order": [],
            "sharing": [],
            "refinements": [],
        }
    )


def test_existing_pi_distinguishes_runtime_row_base_from_flat_access() -> None:
    row = _row_stride_pi()
    flat = _flat_pi()
    assert row.to_json() != flat.to_json()
    assert [
        item["semantic_ref"] for item in row.instances
    ].count("p2") == 3
    assert all(item["semantic_ref"] != "p2" for item in flat.instances)
    assert row.sharing == [{"source": "i2", "consumers": ["i3", "i5"]}]


def test_address_mechanisms_distinguish_staged_base_from_flat_lane() -> None:
    staged = _address_pi(staged=True)
    flat = _address_pi(staged=False)
    assert staged.to_json() != flat.to_json()
    assert [item["mechanism"] for item in staged.instances] == [
        "realize", "address_base", "address_lane", "transfer_direct"
    ]
    assert [item["mechanism"] for item in flat.instances] == [
        "realize", "address_lane", "transfer_direct"
    ]
    assert staged.routes[1] == {
        "source": "i1", "target": "i2", "slots": [0]
    }
    assert RealizationWitness.from_json(staged.to_json()) == staged


def test_pointer_base_and_integer_merge_have_distinct_carrier_routes() -> None:
    pointer_base = _address_pi(staged=True)
    integer_merge = _integer_merge_address_pi()

    assert pointer_base.routes[0] == {
        "source": "i0",
        "target": "i1",
        "slots": [0],
    }
    assert {"source": "i0", "target": "i2", "slots": [0]} not in (
        integer_merge.routes
    )
    assert {"source": "i0", "target": "i4", "slots": [2]} in (
        integer_merge.routes
    )
    assert pointer_base.to_json() != integer_merge.to_json()
    for witness in (pointer_base, integer_merge):
        artifact = make_reduce_artifact()
        artifact.pi = witness
        assert build_normal_rewriter_view(artifact)["pi"] == witness.to_json()


@pytest.mark.parametrize(
    "downstream",
    [
        "transfer_direct",
        "transfer_async",
        "update_atomic",
        "update_privatized",
    ],
)
def test_address_mechanisms_are_orthogonal_to_transfer_and_update(
    downstream: str,
) -> None:
    payload = _address_pi(staged=True).to_json()
    payload["instances"][-1]["mechanism"] = downstream
    parsed = RealizationWitness.from_json(payload)
    assert parsed.instances[-2]["mechanism"] == "address_lane"
    assert parsed.instances[-1]["mechanism"] == downstream


def test_address_mechanisms_survive_replay_without_semantic_payload() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _address_pi(staged=True)
    view = build_normal_rewriter_view(artifact)
    assert view["pi"] == artifact.pi.to_json()
    encoded = json.dumps(view["pi"], sort_keys=True)
    for forbidden in ("formula", "dtype", "shape", "predicate", "tl.load"):
        assert forbidden not in encoded


def test_address_mechanism_deletion_or_swap_changes_witness_identity() -> None:
    staged = make_reduce_artifact()
    staged.pi = _address_pi(staged=True)
    flat = make_reduce_artifact()
    flat.pi = _address_pi(staged=False)
    swapped = make_reduce_artifact()
    swapped_payload = _address_pi(staged=True).to_json()
    swapped_payload["instances"][1]["mechanism"] = "address_lane"
    swapped_payload["instances"][2]["mechanism"] = "address_base"
    swapped.pi = RealizationWitness.from_json(swapped_payload)

    assert len(
        {
            canonical_artifact_digest(staged),
            canonical_artifact_digest(flat),
            canonical_artifact_digest(swapped),
        }
    ) == 3


def test_address_pi_is_unchanged_by_masked_j_and_cannot_supply_value_dag() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _address_pi(staged=True)
    normal = build_normal_rewriter_view(artifact)
    masked = build_masked_rewriter_view(artifact)
    assert normal["pi"] == masked["pi"]
    assert normal["j"] != masked["j"]
    encoded = json.dumps(masked["pi"], sort_keys=True)
    for semantic_atom in ("map_binary", "reduce", "sum", "mul", "output_mappings"):
        assert semantic_atom not in encoded


def test_pi_repeated_anchor_and_runtime_routes_survive_replay_packet() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _row_stride_pi()
    view = build_normal_rewriter_view(artifact)
    assert view["pi"] == artifact.pi.to_json()
    assert view["pi"]["routes"][2] == {
        "source": "i2", "target": "i3", "slots": [0]
    }
    instructions, _user = build_normal_rewriter_prompt(artifact)
    assert "do not collapse their chain" in instructions
    assert "discard J/C dependencies" in instructions
    assert "representation refinement" in instructions
    assert "materialize it at the refined instance" in instructions


def test_layernorm_mask_motion_changes_j_not_pi_semantics() -> None:
    common = {
        "name": "scrc_j0",
        "inputs": [
            {"name": "t0", "dtype": "f32", "shape": [8]},
            {"name": "t1", "dtype": "i1", "shape": [8]},
        ],
        "outputs": [{"name": "t2", "dtype": "f32", "shape": []}],
    }
    pre = IntentFunction.from_json(
        {
            **common,
            "body": [
                {"id": "n0", "op": "select", "operands": ["t1", "t0", "ZERO"], "result": "v0"},
                {"id": "n1", "op": "map_binary", "operands": ["v0", "v0"], "attrs": {"fn": "mul"}, "result": "v1"},
                {"id": "n2", "op": "reduce", "operands": ["v1"], "attrs": {"fn": "sum", "axes": [0]}, "result": "t2"},
            ],
        }
    )
    post = IntentFunction.from_json(
        {
            **common,
            "body": [
                {"id": "n0", "op": "map_binary", "operands": ["t0", "t0"], "attrs": {"fn": "mul"}, "result": "v0"},
                {"id": "n1", "op": "select", "operands": ["t1", "v0", "ZERO"], "result": "v1"},
                {"id": "n2", "op": "reduce", "operands": ["v1"], "attrs": {"fn": "sum", "axes": [0]}, "result": "t2"},
            ],
        }
    )
    shared_pi = _flat_pi().to_json()
    assert pre.to_json() != post.to_json()
    assert shared_pi == _flat_pi().to_json()
    assert "select" not in json.dumps(shared_pi, sort_keys=True)


def test_masked_j_changes_only_j() -> None:
    artifact = make_reduce_artifact()
    normal = build_normal_rewriter_view(artifact)
    masked = build_masked_rewriter_view(artifact)
    assert normal["j"] != masked["j"]
    for key in ("c", "pi", "u_f"):
        assert normal[key] == masked[key]
    assert set(normal) == {"j", "c", "pi", "u_f"}


def test_replay_packet_is_source_isolated() -> None:
    packet = build_rewriter_packet(make_reduce_artifact())
    assert isinstance(packet, ReplayPacket)
    payload = packet.to_json()
    encoded = json.dumps(payload, sort_keys=True)
    assert "PRIVATE_SOURCE_CANARY" not in encoded
    assert "location_map" not in encoded
    assert "provenance" not in encoded
    assert set(payload["c"]) == {"slots"}
    assert "values" not in payload["c"]
    assert payload["u_f"]["protocol"] == "intentir.scrc.replay-input.v1"
    assert_source_isolated_payload(payload)


def _historical_triton_v6_payload() -> dict[str, object]:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    u_f = payload["u_f"]
    u_f["context_policy"] = "public_frontend_context_v6"
    contract = u_f["compile_entry_contract"]
    contract = contract.replace(
        " Every pi.unit_iterations entry is mandatory physical topology over "
        "one already-authored non-loop J subject and axis. Materialize exactly "
        "the logical points [0, extent) with unit step, or the same points in "
        "descending order for direction=reverse. It has no C tile and never "
        "authorizes a new semantic J loop, rounded bound, launch-derived bound, "
        "formula, or operator.",
        "",
    )
    contract = contract.replace(
        " A reduction_linearization refinement is the exact source-neutral "
        "full-axis materialization immediately before its uniquely routed J "
        "reduce. For Triton, emit `tl.reshape(value, (value.numel,), "
        "can_reorder=True)` at that stage, then reduce on local axis 0. Never "
        "infer it from reduction kind or shape, apply it to another consumer, "
        "or omit it because the current carrier is already rank one.",
        "",
    )
    contract = contract.replace(
        " When pi.terminal_controls relates the two authored arm instances of "
        "that effect_if, emit the terminal arm as the sole body of a runtime "
        "`if` followed by a bare return, then emit the continuation arm after "
        "the `if` without an `else`. Use the authored predicate unchanged when "
        "the positive arm is terminal and its logical negation when the negative "
        "arm is terminal. Resolve both roles only through the typed pi relation; "
        "never infer termination from empty arms, effects, values, names, or "
        "frontend conventions.",
        "",
    )
    contract = contract.replace(
        " Every pi.physical_iterations entry is mandatory realization "
        "topology for a Source-visible physical loop around an existing "
        "non-loop J reduce/contraction or boundary access/domain; it never "
        "creates a semantic J loop. Resolve body_instance to that authored "
        "subject and tile_instance to the required positive C slot attached "
        "to the same closed dependent axis. The "
        "axis attachment determines the logical extent, and induction_basis "
        "has the same extent_step, rounded_extent_step, or tile_ordinal "
        "denotation as pi.iterations. direction=forward visits that closed "
        "basis in ascending order; direction=reverse visits exactly the same "
        "basis points in descending order. Never infer the extent or direction "
        "from Source names, current shapes, C values, grid conventions, or "
        "frontend APIs. Every pi.physical_loop_carriers entry preserves the "
        "declared init/state/next/result temporal stages for its physical "
        "iteration region; do not collapse the loop-carried accumulator into "
        "one stateless reduction or invent an update formula absent from J.",
        "",
    )
    contract = contract.replace(
        " A J const with value '-inf', 'inf', or 'nan' is the exact typed IEEE "
        "special authored by J. Materialize it directly at its declared dtype; "
        "never approximate it with a finite literal or promote it through an "
        "undeclared wider floating type."
        " A J.signature constexpr p* with logical_ref=x* is the callable "
        "carrier of that exact semantic dimension. Use that same p* directly "
        "where the authored pi lane/iteration realization requires x*; never "
        "replace it with a current-shape literal or an arbitrary unit extent.",
        "",
    )
    contract = contract.replace(
        " A J.signature constexpr p* with logical_ref=null is only a "
        "Harness-fixed JIT-specialization carrier: preserve its constexpr "
        "ABI, emit the already-authored active graph, and never read it, "
        "reconstruct inactive control, or invent a dummy use. Its concrete "
        "value is absent from the Replay packet.",
        "",
    )
    contract = contract.replace(
        " Every J value_if is mandatory pure value control. Emit one real "
        "runtime `if` with explicit positive and negative arms in authored "
        "polarity and lexical order. Each arm may compute pure local "
        "temporaries, but its final statement must assign the authored yield "
        "to the same merge local in both arms. Never replace value_if with "
        "tl.where/select, effect_if, predicated unconditional computation, or "
        "compile-time control. effect_if, value_if, and select are "
        "hard-distinct; read the predicate, arms, yields, and result only "
        "from J.",
        "",
    )
    contract = contract.replace(
        " For induction_basis=rounded_extent_step, use the same zero lower "
        "and matching C tile step as extent_step, but use exactly "
        "ceil_div(the declared J trip extent, that C tile) multiplied by "
        "that C tile as the upper bound. Do not use the current tensor shape, "
        "a launch value, or an invented rounding formula."
        " Every J effect_if is mandatory effect control. Emit one real "
        "runtime `if` with explicit positive and negative arms, preserve "
        "predicate polarity, lexical arm scope, nested operation/effect order, "
        "and an explicitly empty arm. Never replace effect_if with tl.where, "
        "a value select, predicated unconditional effects, or compile-time "
        "control. Read the predicate and both arms only from J.",
        "",
    )
    contract = contract.replace(
        " It is the local tile axis after program/grid axes select a program "
        "instance, not the whole-tensor logical ordinal in J reduce.attrs.axes; "
        "never copy or offset that logical ordinal to guess it.",
        "",
    )
    contract = contract.replace(
        "the explicitly authorized target_dtype; preserve distinct refined "
        "stages and do not algebraically remove or move the conversion. Never "
        "propagate one refinement to a sibling parameter/instance that merely "
        "shares a dtype or address expression. Signature dtypes are otherwise "
        "authoritative: do not guess a stride/index widening or emit `.to` for "
        "a p* carrier without its own routed refinement or an explicit J cast."
        " A scalar J.signature p* with logical_ref=null has no logical denotation. It "
        "may affect the body only through an instance anchored to that p* and "
        "a declared route to its realization consumer; if no such route exists, "
        "keep the ABI parameter but do not read it in the kernel body.",
        "the authorized target_dtype; preserve distinct refined stages and do "
        "not algebraically remove or move the conversion.",
    )
    u_f["compile_entry_contract"] = contract
    return payload


def test_public_frontend_context_history_is_read_only_and_exact() -> None:
    history = PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY["triton"]
    receipt_history = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY["triton"]
    current_digest = public_frontend_context_digest(
        PUBLIC_FRONTEND_CONTEXT["triton"]
    )

    assert current_digest == history["public_frontend_context_v15"]
    current_receipt_digest = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY[
        "triton"
    ]["public_frontend_context_v15"]["context_sha256"]
    current_receipt = audit_public_frontend_context_receipt(
        frontend="triton",
        context_policy="public_frontend_context_v15",
        context_sha256=current_receipt_digest,
    )
    assert current_receipt["status"] == "current"
    assert current_receipt["receipt_hash_basis"] == "full_u_f"
    prior_receipt = audit_public_frontend_context_receipt(
        frontend="triton",
        context_policy="public_frontend_context_v8",
        context_sha256=receipt_history["public_frontend_context_v8"][
            "context_sha256"
        ],
    )
    assert prior_receipt["status"] == "historical"
    assert audit_public_frontend_context_payload(
        frontend="triton",
        context_policy="public_frontend_context_v8",
        public_context_sha256=history["public_frontend_context_v8"],
    )["status"] == "historical"
    historical_receipt = audit_public_frontend_context_receipt(
        frontend="triton",
        context_policy="public_frontend_context_v6",
        context_sha256=receipt_history["public_frontend_context_v6"][
            "context_sha256"
        ],
    )
    assert historical_receipt["status"] == "historical"
    assert historical_receipt["receipt_hash_basis"] == "full_u_f"
    assert audit_public_frontend_context_payload(
        frontend="triton",
        context_policy="public_frontend_context_v6",
        public_context_sha256=history["public_frontend_context_v6"],
    )["status"] == "historical"
    with pytest.raises(TypeError):
        history["public_frontend_context_v7"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY["triton"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt_history["public_frontend_context_v6"][
            "receipt_hash_basis"
        ] = "public_context"  # type: ignore[index]
    with pytest.raises(SchemaError, match="unknown context_policy"):
        audit_public_frontend_context_receipt(
            frontend="triton",
            context_policy="public_frontend_context_v5",
            context_sha256="0" * 64,
        )
    with pytest.raises(SchemaError, match="digest does not match"):
        audit_public_frontend_context_receipt(
            frontend="triton",
            context_policy="public_frontend_context_v7",
            context_sha256=current_digest,
        )
    with pytest.raises(SchemaError, match="digest does not match"):
        audit_public_frontend_context_receipt(
            frontend="triton",
            context_policy="public_frontend_context_v6",
            context_sha256=history["public_frontend_context_v6"],
        )


def test_historical_packet_has_read_only_audit_but_cannot_be_fresh_replay() -> None:
    payload = _historical_triton_v6_payload()
    public_context = {
        key: value
        for key, value in payload["u_f"].items()
        if key not in {"protocol", "frontend", "replay_function"}
    }
    assert public_frontend_context_digest(public_context) == (
        PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY["triton"][
            "public_frontend_context_v6"
        ]
    )
    stored_digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt = audit_replay_packet_digest(
        payload,
        expected_digest=stored_digest,
    )
    assert receipt["status"] == "historical"
    assert receipt["public_context_sha256"] == (
        PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY["triton"][
            "public_frontend_context_v6"
        ]
    )
    assert receipt["packet_sha256"] == stored_digest
    with pytest.raises(SchemaError, match="fixed public frontend context"):
        assert_source_isolated_payload(payload)
    with pytest.raises(SchemaError, match="fixed public frontend context"):
        replay_packet_digest(payload)
    with pytest.raises(SchemaError, match="packet digest mismatch"):
        audit_replay_packet_digest(payload, expected_digest="0" * 64)
    assert build_rewriter_packet(make_reduce_artifact()).u_f[
        "context_policy"
    ] == "public_frontend_context_v15"


def test_real_v6_u_f_uses_the_historical_full_u_f_receipt_basis() -> None:
    u_f = _historical_triton_v6_payload()["u_f"]
    receipt = build_public_frontend_context_receipt(u_f)

    assert receipt == {
        "frontend": "triton",
        "context_policy": "public_frontend_context_v6",
        "context_sha256": (
            "33f1d58a4c1e762fae6b8ff27c100adfef21bcc1d8a149409f85f8b4c7a281ec"
        ),
        "receipt_hash_basis": "full_u_f",
        "status": "historical",
    }
    assert receipt["context_sha256"] != (
        PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY["triton"][
            "public_frontend_context_v6"
        ]
    )


@pytest.mark.parametrize(
    ("frontend", "expected_digest"),
    [
        (
            Frontend.TRITON,
            "c7706c0d986a883cb30f703f67238885d9395b5fb9fa0251c443989edfce4b7e",
        ),
        (
            Frontend.CUTE,
            "13c3f240eb4bd19bfbbc6982e3058ebb4ecc37c139a1e483cff0cff6a811bb37",
        ),
        (
            Frontend.TILELANG,
            "b600cb80e9c1496252b024b3a2257332a74a64ac77dc1c75ca70b7c8f5ea7140",
        ),
        (
            Frontend.CUDA,
            "1d5bee977f82a852fbde57428bb4b4bdca784c837ede9ae91a5f18d8b75e8e65",
        ),
    ],
)
def test_existing_frontend_receipts_keep_the_full_u_f_hash_basis(
    frontend: Frontend,
    expected_digest: str,
) -> None:
    u_f = build_normal_rewriter_view(
        make_reduce_artifact(), frontend=frontend
    )["u_f"]
    independently_hashed = hashlib.sha256(
        json.dumps(u_f, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = build_public_frontend_context_receipt(u_f)

    assert independently_hashed == expected_digest
    assert receipt["context_sha256"] == expected_digest
    assert receipt["receipt_hash_basis"] == "full_u_f"


def test_one_physical_scalar_cannot_bind_rank0_value_and_dimension() -> None:
    payload = make_reduce_artifact().to_json()
    payload["j"]["signature"]["parameters"][2]["logical_ref"] = ["t1", "x0"]

    with pytest.raises(SchemaError, match="logical_ref"):
        Gate1Artifact.from_json(payload)


def _artifact_with_typed_representation_widen(
    *,
    source_dtype: str | None,
    target_dtype: str | None,
) -> Gate1Artifact:
    payload = make_reduce_artifact().to_json()
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "scalar",
            "mode": "value",
            "dtype": source_dtype or "i32",
            "rank": 0,
            "logical_ref": None,
        }
    )
    payload["pi"]["instances"].append(
        {
            "id": "i4",
            "semantic_ref": "p3",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    payload["pi"]["routes"].append(
        {"source": "i4", "target": "i0", "slots": [0]}
    )
    authorization = {
        "auth_id": "u0",
        "kind": "signed_widen",
        "subject_ref": "p3",
    }
    if source_dtype is not None:
        authorization.update(
            source_dtype=source_dtype,
            target_dtype=target_dtype,
        )
    payload["j"]["boundary"]["realization_authorizations"].append(
        authorization
    )
    payload["pi"]["refinements"].append(
        {"instance": "i4", "kind": "signed_widen", "auth_ref": "u0"}
    )
    return Gate1Artifact.from_json(payload)


def test_replay_contract_accepts_an_explicit_strict_integer_widen() -> None:
    artifact = _artifact_with_typed_representation_widen(
        source_dtype="i32",
        target_dtype="i64",
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed
    assert report.protocol == "intentir.scrc.typecheck.v26"
    assert build_rewriter_packet(artifact).u_f["context_policy"] == (
        "public_frontend_context_v15"
    )


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [("i32", "i32"), (None, None), ("u32", "u64")],
)
def test_typecheck_rejects_non_widening_or_untyped_signed_widen(
    source_dtype: str | None,
    target_dtype: str | None,
) -> None:
    artifact = _artifact_with_typed_representation_widen(
        source_dtype=source_dtype,
        target_dtype=target_dtype,
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert not report.passed
    assert report.protocol == "intentir.scrc.typecheck.v26"
    assert "refinement_auth_invalid_widen" in {
        item.code.value for item in report.diagnostics
    }
    with pytest.raises(SchemaError, match="refinement_auth_invalid_widen"):
        build_rewriter_packet(artifact)


def test_triton_u_f_preserves_constexpr_and_projects_coordinate_roles() -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    context = payload["u_f"]
    contract = context["compile_entry_contract"]

    assert context["context_policy"] == "public_frontend_context_v15"
    assert "pass a c* handle directly to tl.arange" in contract
    assert "never copy it to a local first" in contract
    assert "satisfied by the parameter binding" in contract
    assert "J.boundary.reductions.tile_axis" in contract
    assert "physical frontend reduction axis" in contract
    assert "local tile axis after program/grid axes" in contract
    assert "not the whole-tensor logical ordinal" in contract
    assert "when predicate is absent" in contract
    assert "never synthesize a defensive or redundant mask" in contract
    assert "placement='program' alone never authorizes tl.program_id" in contract
    assert "path enters address_base" in contract
    assert "path enters address_lane" in contract
    assert "first declaration of each distinct J domain semantic_ref" in contract
    assert "same d* remain" in contract
    assert "never allocate another axis" in contract
    assert "Only a distinct domain anchor may allocate the next" in contract
    assert "even if its grid extent is one" in contract
    assert "pointer/tensor p* carrier directly enters address_base" in contract
    assert "requires a second pointer/address operation" in contract
    assert "A reduction_linearization refinement" in contract
    assert "tl.reshape(value, (value.numel,), can_reorder=True)" in contract
    assert "carrier bypasses address_base" in contract
    assert "before exactly one pointer/address operation" in contract
    assert "edge alone" in contract
    assert "never infer from Source names, paths, or text" in contract
    assert "canonical binary J contraction" in contract
    assert "one pi realize instance" in contract
    assert "maps to one direct tl.dot realization" in contract
    assert "pi.terminal_controls relates the two authored arm instances" in contract
    assert "continuation arm after the `if` without an `else`" in contract
    assert "Do not replace that single stage" in contract
    assert "broadcasted outer-product multiplication plus tl.sum" in contract
    assert "does not apply to outer products" in contract
    assert "unsupported tile/dtype contracts" in contract
    assert "pi chain that explicitly declares multiple" in contract
    assert "Every pi.iterations entry is mandatory" in contract
    assert "induction_basis=extent_step" in contract
    assert "induction_basis=tile_ordinal" in contract
    assert "never insert or remove the trip-count ceiling division" in contract
    assert "from triton.language.extra.cuda import libdevice" in contract
    assert "Directly importing a libdevice function is rejected" in contract
    assert "tanh/tan/atan/atan2 as libdevice calls" in contract
    assert "elementwise power as the public `x ** y` operator" in contract
    assert "isnan as `x != x`" in contract
    assert "Every pi.refinements entry is mandatory" in contract
    assert "explicitly authorized target_dtype" in contract
    assert "Never propagate one refinement to a sibling" in contract
    assert "do not guess a stride/index widening" in contract
    assert "logical_ref=null has no logical denotation" in contract
    assert "keep the ABI parameter but do not read it" in contract
    assert_source_isolated_payload(payload)


@pytest.mark.parametrize("frontend", list(Frontend))
def test_every_frontend_receives_the_same_closed_iteration_denotation(
    frontend: Frontend,
) -> None:
    instructions, _ = build_normal_rewriter_prompt(
        make_reduce_artifact(), frontend=frontend
    )

    assert "extent_step means lower zero" in instructions
    assert "upper equal to the J loop trip extent" in instructions
    assert "step equal to the matching C tile" in instructions
    assert "tile_ordinal means lower zero" in instructions
    assert "ceil_div of that J trip extent by that matching C tile" in instructions




def test_replay_prompt_distinguishes_pointer_base_from_integer_merge() -> None:
    instructions, _ = build_normal_rewriter_prompt(make_reduce_artifact())

    assert "A boundary access with no predicate is unconditional" in instructions
    assert "never derive a redundant mask from domains" in instructions
    assert "pointer/tensor J.signature carrier directly enters address_base" in instructions
    assert "a second lane pointer/address operation" in instructions
    assert "carrier bypasses address_base" in instructions
    assert "before exactly one pointer/address operation" in instructions
    assert "base-to-lane edge alone" in instructions
    assert "never from Source names, paths, text" in instructions
    assert "axis identity belongs to the J domain anchor" in instructions
    assert "same axis, never as a new axis" in instructions
    assert "use one direct tl.dot" in instructions
    assert "do not expand it into outer-product multiply plus tl.sum" in instructions
    assert "Do not apply this spelling outside those support conditions" in instructions
    assert "pi explicitly declares multiple contraction stages" in instructions


@pytest.mark.parametrize("frontend", ["cuda", "cute", "tilelang"])
def test_non_triton_replay_prompt_excludes_triton_contraction_spelling(
    frontend: str,
) -> None:
    instructions, _ = build_normal_rewriter_prompt(
        make_reduce_artifact(), frontend=frontend
    )

    assert "tl.dot" not in instructions
    assert "tl.sum" not in instructions
    assert "pointer/tensor J.signature carrier directly enters" not in instructions
    assert "axis identity belongs to the J domain anchor" not in instructions
    assert "base-to-lane edge alone" not in instructions
    assert "Consume J, pure C and pi directly" in instructions
    assert "pi is topology, not semantic authority" in instructions


def test_lift_instruction_assigns_only_comparison_class_not_thresholds() -> None:
    instructions = scrc.build_scrc_lift_instructions()
    normalized = " ".join(instructions.split())

    assert "supplied Responses schema is authoritative" in normalized
    assert len(instructions) < 18_500
    assert "Floating transcendentals" in normalized
    assert "must use `tolerance`" in normalized
    assert "external Gate2 policy Q supplies all atol" in normalized
    assert "Never place a Python comparator" in normalized
    assert "omit `update_node_ref` entirely" in normalized
    assert "empty string or null is invalid" in normalized
    assert "Every J.boundary access must have exactly one matching effect" in normalized
    assert "node_ref (including null)" in normalized
    assert "compiler-visible sensitive representation event" in normalized
    assert "distinct instances/refinements for distinct source stages" in normalized
    assert "reuse that same t* in the predicate and arithmetic dataflow" in normalized
    assert "never split one Source value" in normalized
    assert "an extent-only scalar binds x* directly" in normalized
    assert "constexprness alone never makes one a graph selector" in normalized
    assert "every semantic use is exclusively compile-time control" in normalized
    assert "fixed code checks only the authored wire/type contract" in normalized
    assert "explicit Source `reduce(sum)` followed by division" in normalized
    assert "generated `reduce(sum)` over ones" in normalized
    assert "never propagate one conversion to sibling parameters" in normalized
    assert "strictly wider target_dtype" in normalized
    assert "local tile axis after program/grid axes" in normalized
    assert "not the whole-tensor logical axis number" in normalized
    assert "effect-controlling binary Source branch is one Layer-A `effect_if`" in normalized
    assert "value-yielding Source `scf.if` is one Layer-A `value_if`" in normalized
    assert "scope.effects plus matching EffectSpec/access_ref is sole membership" in normalized
    assert "`rounded_extent_step` uses the same lower/step" in normalized


def test_lift_preserves_degenerate_program_coordinate_topology() -> None:
    normalized = " ".join(scrc.build_scrc_lift_instructions().split())

    assert "program-coordinate occurrence" in normalized
    assert "even when the current concrete grid extent on that axis is one" in normalized
    assert "both a program-coordinate path into address_base" in normalized
    assert "route its p* instance directly into address_base" in normalized
    assert "two source pointer/address operations" in normalized
    assert "pointer p* bypasses address_base" in normalized
    assert "base-to-lane edge alone" in normalized
    assert "defined once and reused by several accesses" in normalized
    assert "one producer instance with routed consumers" in normalized
    assert "Multiple syntactic uses never justify" in normalized
    assert "not once per consumer" in normalized


def test_lift_requires_complete_semantic_recurrence_in_j() -> None:
    normalized = " ".join(scrc.build_scrc_lift_instructions().split())

    assert "compiler-visible semantic recurrence" in normalized
    assert "existing structured `loop` operation" in normalized
    assert "every carried state and initial value" in normalized
    assert "matching next/yield/result values" in normalized
    assert "contraction's canonical subscripts" in normalized
    assert "result-axis orientation" in normalized
    assert "transpose only when it is a real logical transformation" in normalized
    assert "ask Replay to rediscover the missing update algorithm" in normalized
    assert "cannot carry, repair, or infer" in normalized


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("j",), "interface"),
        (("c",), "values"),
        (("pi",), "binding_map"),
        (("u_f",), "compiler_fingerprint"),
    ],
)
def test_replay_packet_nested_allowlists_fail_closed(
    path: tuple[str, ...], key: str
) -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    target = payload
    for component in path:
        target = target[component]
    target[key] = "forbidden"
    with pytest.raises(SchemaError):
        assert_source_isolated_payload(payload)


def test_replay_packet_rejects_candidate_selected_frontend_context() -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    payload["u_f"]["custom_context"] = "source-derived hint"
    with pytest.raises(SchemaError, match="unknown field"):
        ReplayPacket(
            payload["j"], payload["c"], payload["pi"], payload["u_f"]
        )


@pytest.mark.parametrize(
    "stale_policy",
    ["public_frontend_context_v3", "public_frontend_context_v4"],
)
def test_replay_packet_rejects_stale_or_modified_public_context(
    stale_policy,
) -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    payload["u_f"]["context_policy"] = stale_policy

    with pytest.raises(SchemaError, match="fixed public frontend context"):
        ReplayPacket(
            payload["j"], payload["c"], payload["pi"], payload["u_f"]
        )

    payload = build_normal_rewriter_view(make_reduce_artifact())
    payload["u_f"]["compile_entry_contract"] += " source-derived suffix"
    with pytest.raises(SchemaError, match="fixed public frontend context"):
        ReplayPacket(
            payload["j"], payload["c"], payload["pi"], payload["u_f"]
        )


def test_structured_gate1_diff_returns_only_to_source_aware_relift() -> None:
    source = """
import triton
import triton.language as tl

@triton.jit
def source_kernel(X, Y, N: tl.constexpr):
    tl.store(Y, tl.load(X))
"""
    context = SCRCReliftContext(
        1,
        "gate1",
        {"observer_diff": {"category": "dataflow", "kind": "missing"}},
    )
    request = prepare_scrc_lift(source, relift_context=context)
    lift_payload = json.loads(request.user_input)
    assert lift_payload["relift"] == context.to_json()

    replay_payload = build_normal_rewriter_view(make_reduce_artifact())
    assert "relift" not in json.dumps(replay_payload, sort_keys=True)
    assert "observer_diff" not in json.dumps(replay_payload, sort_keys=True)

    with pytest.raises(ValueError, match="unsupported relift"):
        SCRCReliftContext(1, "normal_no_closure")


def test_lift_schema_requires_direct_pi_and_no_embedded_c_channel() -> None:
    schema = scrc_lift_core_json_schema()
    properties = schema["properties"]
    assert set(properties) == {"schema", "j", "c", "pi"}
    assert set(properties["pi"]["required"]) == {
        "regions",
        "instances",
        "routes",
        "order",
        "sharing",
        "refinements",
        "iterations",
        "loop_carriers",
            "physical_iterations",
            "unit_iterations",
            "physical_loop_carriers",
        "terminal_controls",
    }
    assert "consumer_extensions" in (
        properties["j"]["properties"]["boundary"]["required"]
    )
    function_properties = (
        properties["j"]["properties"]["intent"]["properties"]
        ["intent_function"]["properties"]
    )
    assert "layer_c" not in function_properties


def test_lift_wire_schema_encodes_parser_critical_anonymous_shapes() -> None:
    schema = scrc_lift_core_json_schema()
    j = schema["properties"]["j"]
    function = j["properties"]["intent"]["properties"]["intent_function"]
    assert function["properties"]["name"]["enum"] == ["scrc_j0"]
    assert j["properties"]["kernel_id"]["enum"] == ["scrc_k0"]
    assert j["properties"]["node_ids"]["propertyNames"]["pattern"] == "^n[0-9]+$"
    domains = j["properties"]["boundary"]["properties"]["domains"]
    assert domains["propertyNames"]["pattern"] == "^d[0-9]+$"
    assert domains["additionalProperties"]["type"] == "string"
    signature = j["properties"]["signature"]
    parameter_schema = signature["properties"]["parameters"]["items"]
    assert parameter_schema["properties"]["name"]["pattern"] == "^p[0-9]+$"
    assert "representation/address realization" in parameter_schema["description"]
    assert "may be referenced only by pi" in parameter_schema["description"]
    c_schema = schema["properties"]["c"]
    slot = c_schema["properties"]["slots"]["items"]["properties"]
    assert slot["name"]["pattern"] == "^c[0-9]+$"
    assert slot["subject_ref"]["pattern"] == "^[ndaqb][0-9]+$"
    assert slot["axis_ref"]["anyOf"] == [
        {"type": "integer", "minimum": 0},
        {"type": "string", "pattern": "^x[0-9]+$"},
    ]


def test_lift_wire_schema_rejects_orphan_and_non_load_fill() -> None:
    schema = scrc_lift_core_json_schema()
    access_schema = (
        schema["properties"]["j"]["properties"]["boundary"]["properties"]
        ["accesses"]["items"]
    )
    validator = Draft202012Validator(access_schema)
    accesses = make_reduce_artifact().to_json()["j"]["boundary"]["accesses"]
    assert not list(validator.iter_errors(accesses[0]))

    orphan = copy.deepcopy(accesses[0])
    orphan.pop("predicate")
    errors = list(validator.iter_errors(orphan))
    assert errors
    assert any("predicate" in error.message for error in errors)

    non_load = copy.deepcopy(accesses[1])
    non_load["fill_value"] = 0
    errors = list(validator.iter_errors(non_load))
    assert errors
    assert any("load" in error.message for error in errors)


def test_lift_prompt_keeps_runtime_extent_symbolic_and_out_of_ssa_scopes() -> None:
    instructions = " ".join(scrc.build_scrc_lift_instructions().split())

    assert "only the current Harness binding" in instructions
    assert "never permission to specialize J" in instructions
    assert "use it in every affected J tensor shape" in instructions
    assert "not SSA values" in instructions
    assert "never list x* in portable scope definitions/captures" in instructions
    assert "Each physical p* has exactly one logical_ref" in instructions
    assert "value(t*)=x*" in instructions
    assert "J.value_dimension_relations" in instructions
    assert "Author it only from exact Source identity" in instructions
    assert "fixed code and Replay never add or repair it" in instructions


def test_lift_prompt_and_schema_admit_harness_specialization_constexpr() -> None:
    instructions = " ".join(scrc.build_scrc_lift_instructions().split())

    assert "Harness-owned, non-C JIT bindings" in instructions
    assert "its value stays only in Harness" in instructions
    assert "never selects or repairs the graph" in instructions
    assert "every semantic use is exclusively compile-time control" in instructions

    payload = make_reduce_artifact().to_json()
    parameter = payload["j"]["signature"]["parameters"][2]
    parameter["kind"] = "constexpr"
    parameter["rank"] = 0
    parameter.pop("logical_ref")
    assert Gate1Artifact.from_json(payload).to_json() == payload


def test_one_pi_graph_grammar_is_projected_to_schema_lift_and_replay() -> None:
    grammar = render_realization_witness_wire_grammar()
    mechanism_grammar = render_realization_mechanism_grammar()
    schema = scrc_lift_core_json_schema()
    pi_schema = schema["properties"]["pi"]
    lift_instructions = scrc.build_scrc_lift_instructions()
    replay_instructions, _ = build_normal_rewriter_prompt(make_reduce_artifact())

    assert grammar in pi_schema["description"]
    assert grammar in replay_instructions
    assert mechanism_grammar in pi_schema["description"]
    assert mechanism_grammar in replay_instructions
    assert grammar not in lift_instructions
    assert mechanism_grammar not in lift_instructions
    assert "supplied Responses schema is authoritative" in lift_instructions
    assert "routing p0 to i1 is invalid" in grammar
    assert "placement='program' only scopes an instance" in mechanism_grammar
    assert "does not require an identity assignment" in mechanism_grammar
    assert "corresponding J domain" in mechanism_grammar
    assert "path enters address_base" in mechanism_grammar
    assert "entering address_lane is lane-varying" in mechanism_grammar
    assert "must not invent program_id" in mechanism_grammar
    assert "Integer coordinate arithmetic is a realize stage" in mechanism_grammar
    assert "source-neutral carrier routing" in mechanism_grammar
    assert "requires a second pointer/address operation" in mechanism_grammar
    assert "carrier bypasses address_base" in mechanism_grammar
    assert "edge alone never decides" in mechanism_grammar
    assert "Program-axis identity follows the J domain semantic_ref" in mechanism_grammar
    assert "distinct stages of the same axis" in mechanism_grammar
    assert "only a distinct domain anchor may allocate another axis" in mechanism_grammar
    assert "no separable program-uniform contribution" in mechanism_grammar
    assert pi_schema["properties"]["routes"]["items"]["properties"][
        "source"
    ]["pattern"] == "^i[0-9]+$"
    mechanism_schema = pi_schema["properties"]["instances"]["items"][
        "properties"
    ]["mechanism"]
    assert mechanism_schema["enum"] == list(REALIZATION_MECHANISMS)
    assert mechanism_schema["description"] == mechanism_grammar


def test_c_role_rejects_semantic_or_realization_payload() -> None:
    with pytest.raises(SchemaError, match="pure scheduling"):
        ScheduleSlot.from_json(
            {
                "name": "c0",
                "kind": "constexpr",
                "role": "reduction_" + "mask_placement",
            }
        )


def test_source_is_anonymized_before_lift() -> None:
    source = """@triton.jit
def layer_norm_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    # private X and BLOCK_SIZE names must not survive in comments
    mask = N > 0
    centered = tl.load(X, mask=mask)
    tl.store(Y, centered)
"""
    anonymous, binding, _mapping = anonymize_source(source)
    assert "layer_norm_kernel" not in anonymous
    assert "BLOCK_SIZE" not in anonymous
    assert "centered" not in anonymous
    assert "def scrc_k0(p0, p1, p2, c0: tl.constexpr)" in anonymous
    assert "tl.load(p0, mask=v0)" in anonymous
    assert "private X" not in anonymous
    assert dict(binding.source_to_opaque)["X"] == "p0"
    assert dict(binding.source_to_opaque)["BLOCK_SIZE"] == "c0"


def test_source_privacy_does_not_confuse_closed_fn_with_private_name() -> None:
    source = """@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    sum = tl.sum(tl.load(X), axis=0)
    tl.store(Y, sum)
"""
    _anonymous, binding, _mapping = anonymize_source(source)
    artifact = make_reduce_artifact()

    # ``sum`` is both a private Source local and a closed Intent op function.
    # The latter is schema vocabulary, not a retained source identifier.
    scrc._validate_source_identifier_privacy(artifact, binding)


def test_source_privacy_does_not_confuse_canonical_index_with_private_name() -> None:
    source = """@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    k0 = tl.arange(0, BLOCK_SIZE)
    tl.store(Y + k0, tl.load(X + k0))
"""
    _anonymous, binding, _mapping = anonymize_source(source)
    artifact = make_reduce_artifact()
    artifact.j.boundary.accesses[0].index_expression = "[k0, k1]"

    scrc._validate_source_identifier_privacy(artifact, binding)


def test_source_privacy_does_not_confuse_einsum_label_with_private_name() -> None:
    source = """@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    k = tl.arange(0, BLOCK_SIZE)
    tl.store(Y + k, tl.load(X + k))
"""
    _anonymous, binding, _mapping = anonymize_source(source)
    artifact = make_reduce_artifact()
    artifact.j.intent.body[0].attrs["subscripts"] = "k,abhsq->abhsqk"

    scrc._validate_source_identifier_privacy(artifact, binding)


def test_source_privacy_still_rejects_private_name_in_open_expression() -> None:
    source = """@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    secret = tl.arange(0, BLOCK_SIZE)
    tl.store(Y, tl.load(X) + secret)
"""
    _anonymous, binding, _mapping = anonymize_source(source)
    artifact = make_reduce_artifact()
    artifact.j.boundary.domains["d0"] = "secret < x0"

    with pytest.raises(ValueError, match="retained a source identifier"):
        scrc._validate_source_identifier_privacy(artifact, binding)


def test_exact_source_symbol_is_selected_before_anonymization() -> None:
    source = """import triton
import triton.language as tl

def helper(first):
    return first

@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""
    anonymous, binding, mapping = anonymize_source(
        source, source_symbol="selected_kernel"
    )
    assert binding.source_function == "selected_kernel"
    assert mapping["selected_kernel"] == "scrc_k0"
    assert mapping["helper"] == "scrc_k1"
    assert mapping["X"] == "p0"
    assert mapping["Y"] == "p1"
    assert mapping["N"] == "p2"
    assert mapping["BLOCK_SIZE"] == "c0"
    assert "def scrc_k0(p0, p1, p2, c0: tl.constexpr)" in anonymous
    assert "selected_kernel" not in anonymous


def test_anonymization_renames_only_statically_local_call_keywords() -> None:
    source = """import triton
import triton.language as tl

@triton.jit
def helper_left(value, axis: tl.constexpr):
    return tl.sum(value, axis=axis)

@triton.jit
def helper_right(value, axis: tl.constexpr):
    return tl.max(value, axis=axis)

@triton.jit
def selected_kernel(X, Y, LEFT: tl.constexpr, RIGHT: tl.constexpr):
    left = helper_left(X, axis=LEFT)
    right = helper_right(X, axis=RIGHT)
    library = tl.sum(X, axis=0)
    attribute = external.module_call(X, axis=0)
    dynamic = callback(X, axis=0)
    tl.store(Y, left + right + library + attribute + dynamic)
"""

    anonymous, _binding, mapping = anonymize_source(
        source, source_symbol="selected_kernel"
    )

    local_axis = mapping["axis"]
    assert f"scrc_k1(p0, {local_axis}=c0)" in anonymous
    assert f"scrc_k2(p0, {local_axis}=c1)" in anonymous
    # The two helpers share the same private formal spelling, but keyword
    # renaming is selected independently at each statically local call.
    assert "tl.sum(p0, axis=0)" in anonymous
    assert ".module_call(p0, axis=0)" in anonymous
    assert "axis=0" in anonymous
    assert anonymous.count("axis=0") == 3


def test_anonymization_preserves_string_data_for_plain_local_helper() -> None:
    source = """import triton
import triton.language as tl

def resolve_public_api(name):
    return getattr(tl.extra.libdevice, name)

resolved_op = resolve_public_api("resolved_op")

@triton.jit
def selected_kernel(X, Y):
    tl.store(Y, resolved_op(tl.load(X)))
"""

    anonymous, _binding, mapping = anonymize_source(
        source, source_symbol="selected_kernel"
    )

    assert mapping["resolved_op"].startswith("v")
    assert "resolve_public_api" not in anonymous
    assert f"scrc_k1('resolved_op')" in anonymous
    assert f"scrc_k1('{mapping['resolved_op']}')" not in anonymous


def test_bound_constexpr_is_anonymous_p_and_never_pure_c() -> None:
    source = """import triton
import triton.language as tl

@triton.jit
def selected_kernel(X, Y, N, VALID: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < VALID
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""
    view = SCRCCompilerView(
        schedule_values={"BLOCK": 128, "num_warps": 4},
        bound_constexpr_values={"VALID": 17},
        compiler_slots=("num_warps",),
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (17,)),
            SCRCCompilerArgument("Y", "tensor", "f32", (17,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=17),
        ),
        grid=(1,),
        grid_contract=("1",),
        parameter_order=("X", "Y", "N", "VALID", "BLOCK"),
        constexpr_parameters=("VALID", "BLOCK"),
    )
    request = scrc.prepare_scrc_lift(
        source, compiler_view=view, source_symbol="selected_kernel"
    )
    payload = json.loads(request.user_input)
    compiler_view = payload["harness_compiler_view"]
    assert "def scrc_k0(p0, p1, p2, p3: tl.constexpr, c0: tl.constexpr)" in (
        payload["anonymous_source"]
    )
    assert compiler_view["bound_constexpr_values"] == {"p3": 17}
    assert compiler_view["schedule_values"] == {"c0": 128, "c1": 4}
    assert compiler_view["parameter_order"] == ["p0", "p1", "p2", "p3", "c0"]
    assert compiler_view["constexpr_parameters"] == ["p3", "c0"]


def test_compiler_view_rejects_bound_schedule_overlap() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        SCRCCompilerView(
            schedule_values={"N": 8},
            bound_constexpr_values={"N": 8},
            parameter_order=("N",),
            constexpr_parameters=("N",),
        )


def test_compiler_view_mechanically_checks_harness_parameter_modes() -> None:
    artifact = make_copy_artifact()
    view = SCRCCompilerView(
        schedule_values={"BLOCK": 1024},
        arguments=(
            SCRCCompilerArgument(
                "X", "tensor", "f32", (1000,), mode="input"
            ),
            SCRCCompilerArgument(
                "Y", "tensor", "f32", (1000,), mode="input"
            ),
            SCRCCompilerArgument(
                "N", "scalar", "i32", value=1000, mode="value"
            ),
        ),
        parameter_order=("X", "Y", "N", "BLOCK"),
        constexpr_parameters=("BLOCK",),
    )
    mapping = {"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"}
    anonymous_view = scrc._anonymous_compiler_view(view, mapping)

    with pytest.raises(ValueError, match="argument mode"):
        scrc._validate_compiler_view(artifact, anonymous_view)


def test_compiler_view_none_requires_exact_opaque_signature_handle() -> None:
    payload = make_copy_artifact().to_json()
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "opaque",
            "mode": "value",
            "dtype": "none",
            "rank": None,
            "alias_of": None,
            "logical_ref": None,
        }
    )
    artifact = Gate1Artifact.from_json(payload)
    view = SCRCCompilerView(
        schedule_values={"BLOCK": 1024},
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (1000,), mode="input"),
            SCRCCompilerArgument("Y", "tensor", "f32", (1000,), mode="output"),
            SCRCCompilerArgument("N", "scalar", "i32", value=1000, mode="value"),
            SCRCCompilerArgument("OPTIONAL", "none", "none", mode="value"),
        ),
        parameter_order=("X", "Y", "N", "OPTIONAL", "BLOCK"),
        constexpr_parameters=("BLOCK",),
    )
    mapping = {
        "X": "p0",
        "Y": "p1",
        "N": "p2",
        "OPTIONAL": "p3",
        "BLOCK": "c0",
    }
    anonymous_view = scrc._anonymous_compiler_view(view, mapping)

    scrc._validate_compiler_view(artifact, anonymous_view)
    artifact.j.signature.parameters[-1].kind = ParameterKind.SCALAR
    with pytest.raises(ValueError, match="argument kind"):
        scrc._validate_compiler_view(artifact, anonymous_view)


def test_lift_keeps_bound_constexpr_outside_c(monkeypatch) -> None:
    response = make_reduce_artifact().to_json()
    response["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "i32",
            "rank": 0,
            "logical_ref": "x1",
        },
    )
    response["j"]["boundary"]["domains"]["d0"] = "v1 < p3"

    def fake_call(instructions, user, **kwargs):
        del instructions, kwargs
        view = json.loads(user)["harness_compiler_view"]
        assert view["bound_constexpr_values"] == {"p3": 4}
        assert view["schedule_values"] == {"c0": 128}
        return LLMReply(json.dumps(response), "test-model", {"id": "resp-sem"})

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    compiler_view = SCRCCompilerView(
        schedule_values={"BLOCK": 128},
        bound_constexpr_values={"VALID": 4},
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (2, 4)),
            SCRCCompilerArgument("Y", "tensor", "f32", (2,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=4),
        ),
        grid=(1,),
        grid_contract=("1",),
        parameter_order=("X", "Y", "N", "VALID", "BLOCK"),
        constexpr_parameters=("VALID", "BLOCK"),
    )
    source = """@triton.jit
def private_reduce(X, Y, N, VALID: tl.constexpr, BLOCK: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(
        source, compiler_view=compiler_view
    )
    assert extraction.artifact.c.values == {"c0": 128}
    assert [
        item.name for item in extraction.artifact.j.signature.parameters
    ] == ["p0", "p1", "p2", "p3"]
    assert extraction.artifact.j.signature.parameters[3].kind.value == "constexpr"


def test_source_symbol_selection_fails_closed() -> None:
    source = "def present(x):\n    return x\n"
    with pytest.raises(ValueError, match="must resolve exactly once"):
        anonymize_source(source, source_symbol="missing")


def test_lift_llm_directly_returns_complete_pi_in_one_call(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
    calls = []

    def fake_call(instructions, user, **kwargs):
        calls.append((instructions, user, kwargs))
        assert "private_reduce" not in user
        assert "BLOCK_SIZE" not in user
        assert "anonymous_source" in user
        return LLMReply(
            json.dumps(response),
            "test-model",
            {"id": "resp0", "model": "test-model", "usage": {"total_tokens": 1}},
        )

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    source = """@triton.jit
def private_reduce(X, Y, N, BLOCK_SIZE: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(source)
    assert extraction.artifact.pi == artifact.pi
    assert extraction.source == source
    assert len(calls) == 1
    assert extraction.provenance.response_id == "resp0"


def test_lift_preserves_trusted_c_and_public_compiler_role(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
    response["c"]["slots"].append(
        {
            "name": "c1",
            "kind": "compiler",
            "role": "num_warps",
            "value_type": "int",
            "required": True,
        }
    )
    response["c"]["values"]["c1"] = 4

    def fake_call(instructions, user, **kwargs):
        del instructions, kwargs
        assert "BLOCK_SIZE" not in user
        assert '"compiler_roles": {"c1": "num_warps"}' in user
        return LLMReply(json.dumps(response), "test-model", {"id": "resp1"})

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    compiler_view = SCRCCompilerView(
        schedule_values={"BLOCK_SIZE": 128, "num_warps": 4},
        compiler_slots=("num_warps",),
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (2, 4)),
            SCRCCompilerArgument("Y", "tensor", "f32", (2,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=4),
        ),
        grid=(1,),
        grid_contract=("ceil_div(N, BLOCK_SIZE)",),
        parameter_order=("X", "Y", "N", "BLOCK_SIZE"),
        constexpr_parameters=("BLOCK_SIZE",),
    )
    source = """@triton.jit
def private_reduce(X, Y, N, BLOCK_SIZE: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(source, compiler_view=compiler_view)
    assert extraction.artifact.c.values == {"c0": 128, "c1": 4}
    assert extraction.artifact.c.slots[1].role == "num_warps"
