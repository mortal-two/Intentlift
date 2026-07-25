from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.extraction.lifters.scrc import (
    SCRC_DECODE_PROTOCOL,
    SCRCCandidateCheckContext,
    check_scrc_lift_candidate,
)
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    ConsumerExtension,
    EffectKind,
    EffectSpec,
    Gate1Artifact,
    SchemaError,
    TypedOutsideValue,
)
from intentir.verify.gate1.pi import RealizationWitness, RealizationWitnessError
from intentir.verify.typecheck import (
    TYPECHECK_PROTOCOL,
    TypeCheckErrorCode,
    type_check,
)
from test_gate1_loop_schema import _tiled_loop_artifact


def _codes(artifact: Gate1Artifact) -> set[TypeCheckErrorCode]:
    return {
        item.code
        for item in type_check(artifact.j, artifact.c, artifact.pi).diagnostics
    }


def _legacy_payload(version: int) -> dict:
    payload = make_copy_artifact().to_json()
    payload["schema"] = f"intentir.scrc.artifact.v{version}"
    payload["j"]["boundary"].pop("consumer_extensions")
    payload["pi"].pop("loop_carriers")
    if version < 3:
        payload["pi"].pop("iterations")
    return payload


def test_artifact_v4_requires_both_new_arrays_and_round_trips_them() -> None:
    payload = make_copy_artifact().to_json()
    assert payload["schema"] == "intentir.scrc.artifact.v4"
    assert payload["j"]["boundary"]["consumer_extensions"] == []
    assert payload["pi"]["loop_carriers"] == []
    assert Gate1Artifact.from_json(payload).to_json() == payload

    missing_consumer = copy.deepcopy(payload)
    missing_consumer["j"]["boundary"].pop("consumer_extensions")
    with pytest.raises(
        SchemaError, match="missing required field: consumer_extensions"
    ):
        Gate1Artifact.from_json(missing_consumer)

    missing_carriers = copy.deepcopy(payload)
    missing_carriers["pi"].pop("loop_carriers")
    with pytest.raises(SchemaError, match="missing required field: loop_carriers"):
        Gate1Artifact.from_json(missing_carriers)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_artifact_v1_v3_round_trip_exactly_without_v4_fields(version: int) -> None:
    payload = _legacy_payload(version)
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.to_json() == payload
    assert parsed.j.boundary.consumer_extensions == []
    assert parsed.pi.loop_carriers == []

    with_consumer = copy.deepcopy(payload)
    with_consumer["j"]["boundary"]["consumer_extensions"] = []
    with pytest.raises(SchemaError, match="v1-v3 do not contain J consumer_extensions"):
        Gate1Artifact.from_json(with_consumer)

    with_carrier = copy.deepcopy(payload)
    with_carrier["pi"]["loop_carriers"] = []
    with pytest.raises(SchemaError, match="v1-v3 do not contain pi loop_carriers"):
        Gate1Artifact.from_json(with_carrier)


def test_artifact_v4_rejects_the_duplicate_masked_contribution_authority() -> None:
    payload = make_reduce_artifact().to_json()
    payload["j"]["boundary"]["reductions"][0]["masked_contribution"] = 0

    with pytest.raises(SchemaError, match="unique physical outside-domain authority"):
        Gate1Artifact.from_json(payload)


def test_static_v4_schema_requires_new_arrays_and_forbids_old_reduction_field() -> None:
    schema = scrc_lift_core_json_schema()
    payload = make_reduce_artifact().to_json()
    boundary_schema = schema["properties"]["j"]["properties"]["boundary"]
    consumer_validator = Draft202012Validator(
        boundary_schema["properties"]["consumer_extensions"]["items"]
    )
    assert not list(
        consumer_validator.iter_errors(
            payload["j"]["boundary"]["consumer_extensions"][0]
        )
    )
    assert "consumer_extensions" in boundary_schema["required"]
    assert "loop_carriers" in schema["properties"]["pi"]["required"]

    old_reduction = copy.deepcopy(payload["j"]["boundary"]["reductions"][0])
    old_reduction["masked_contribution"] = 0
    reduction_validator = Draft202012Validator(
        boundary_schema["properties"]["reductions"]["items"]
    )
    assert any(
        "masked_contribution" in error.message
        for error in reduction_validator.iter_errors(old_reduction)
    )


def test_consumer_extension_wire_is_closed_and_relation_key_is_unique() -> None:
    payload = make_reduce_artifact().to_json()
    payload["j"]["boundary"]["consumer_extensions"][0]["predicate"] = "k0 < x0"
    with pytest.raises(SchemaError, match="unknown key"):
        Gate1Artifact.from_json(payload)

    payload = make_reduce_artifact().to_json()
    extension = payload["j"]["boundary"]["consumer_extensions"][0]
    payload["j"]["boundary"]["consumer_extensions"].append(copy.deepcopy(extension))
    with pytest.raises(
        SchemaError, match="consumer/operand/access keys must be unique"
    ):
        Gate1Artifact.from_json(payload)


@pytest.mark.parametrize(
    ("fn", "outside"),
    [("sum", 0), ("max", "NEG_INF"), ("min", "POS_INF")],
)
def test_consumer_extension_accepts_closed_reduce_identities(fn: str, outside) -> None:
    artifact = make_reduce_artifact()
    artifact.j.intent.body[1].attrs["fn"] = fn
    artifact.j.boundary.reductions[0].identity = outside
    artifact.j.boundary.consumer_extensions[0].outside.value = outside

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_consumer_extension_accepts_scan_and_keeps_predicate_in_layer_b() -> None:
    artifact = make_reduce_artifact()
    consumer = artifact.j.intent.body[1]
    consumer.op = "scan"
    consumer.attrs = {"fn": "sum", "axis": 1}
    consumer.type = {"dtype": "f32", "shape": ["x0", "x1"]}
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.boundary.reductions = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert artifact.j.boundary.accesses[0].predicate == "v1 < x1"
    assert all(op.op != "select" for op in artifact.j.intent.body)


def _insert_pad_before_reduce(artifact: Gate1Artifact, *, op_name: str = "pad") -> None:
    pad = IntentOp.from_json(
        {
            "id": "n2",
            "op": op_name,
            "result": "v1",
            "operands": ["v0"],
            "attrs": {"lo": [0, 0], "hi": [0, 1], "value": 0}
            if op_name == "pad"
            else {},
            "type": {"dtype": "f32", "shape": ["x0", "x1"]},
        }
    )
    artifact.j.intent.body.insert(1, pad)
    artifact.j.intent.body[2].operands = ["v1"]
    artifact.j.node_ids = {"n0": "body[0]", "n2": "body[1]", "n1": "body[2]"}


def test_consumer_extension_provenance_crosses_closed_pad() -> None:
    artifact = make_reduce_artifact()
    _insert_pad_before_reduce(artifact)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_consumer_extension_provenance_stops_at_unknown_op() -> None:
    artifact = make_reduce_artifact()
    _insert_pad_before_reduce(artifact, op_name="future_transform")

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID in {
        item.code for item in report.diagnostics
    }
    assert any("unsupported J op" in item.detail for item in report.diagnostics)


@pytest.mark.parametrize(
    ("mutation", "expected_path"),
    [
        ("wrong_dtype", ".outside.dtype"),
        ("non_identity", ".outside.value"),
        ("wrong_operand", ".operand_index"),
        ("unpredicated", ".access_ref"),
        ("unrelated_node", ".access_ref"),
    ],
)
def test_consumer_extension_dependent_errors_fail_closed(
    mutation: str, expected_path: str
) -> None:
    artifact = make_reduce_artifact()
    extension = artifact.j.boundary.consumer_extensions[0]
    if mutation == "wrong_dtype":
        extension.outside.dtype = "f16"
    elif mutation == "non_identity":
        extension.outside.value = 1
    elif mutation == "wrong_operand":
        extension.operand_index = 7
    elif mutation == "unpredicated":
        artifact.j.boundary.accesses[0].predicate = None
        artifact.j.boundary.accesses[0].fill_value = None
    elif mutation == "unrelated_node":
        artifact.j.boundary.accesses[0].node_ref = "n1"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    report = type_check(artifact.j, artifact.c, artifact.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID
    ]

    assert matching
    assert any(item.path.endswith(expected_path) for item in matching)


def test_consumer_extension_missing_and_node_less_ambiguity_are_diagnosed() -> None:
    missing = make_reduce_artifact()
    missing.j.boundary.consumer_extensions = []
    assert TypeCheckErrorCode.CONSUMER_EXTENSION_MISSING in _codes(missing)

    ambiguous = make_reduce_artifact()
    ambiguous.j.boundary.accesses[0].node_ref = None
    ambiguous.j.boundary.accesses.append(
        BoundaryAccess(
            "a2",
            AccessKind.LOAD,
            "t0",
            predicate="v1 < x1",
            fill_value=0,
            index_expression="v2",
        )
    )
    ambiguous.j.boundary.effects.append(
        EffectSpec(EffectKind.READ, "t0", None, "e2", "a2")
    )
    assert TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID in _codes(ambiguous)


def test_direct_typecheck_rejects_two_outside_domain_authorities() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.reductions[0].masked_contribution = 0

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID
        and "duplicate authorities" in item.detail
        for item in report.diagnostics
    )


def _add_pointer_loop_carrier(
    artifact: Gate1Artifact, *, identity: bool = False
) -> None:
    payload = artifact.pi.to_json()
    payload["instances"].extend(
        [
            {"id": "i2", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
            {"id": "i3", "semantic_ref": "p0", "region": "r2", "mechanism": "realize"},
            {"id": "i4", "semantic_ref": "p0", "region": "r2", "mechanism": "realize"},
            {"id": "i5", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
            {
                "id": "i6",
                "semantic_ref": "a0",
                "region": "r2",
                "mechanism": "transfer_direct",
            },
        ]
    )
    payload["routes"].extend(
        [
            {"source": "i2", "target": "i3", "slots": []},
            {"source": "i3", "target": "i6", "slots": []},
        ]
    )
    if not identity:
        payload["routes"].append(
            {"source": "i3", "target": "i4", "slots": []}
        )
    payload["loop_carriers"] = [
        {
            "region": "r2",
            "loop_instance": "i0",
            "init": "i2",
            "state": "i3",
            "next": "i3" if identity else "i4",
            "result": "i5",
        }
    ]
    artifact.pi = RealizationWitness.from_json(payload)


@pytest.mark.parametrize("identity", [False, True])
def test_loop_carrier_closes_temporal_and_zero_trip_exit_without_dag_cycle(
    identity: bool,
) -> None:
    artifact = _tiled_loop_artifact()
    _add_pointer_loop_carrier(artifact, identity=identity)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.LOOP_CARRIER_INVALID not in {
        item.code for item in report.diagnostics
    }, report.to_json()
    # next[k] -> state[k+1] and result=state[T] are denoted only by the
    # loop_carriers record; adding the temporal backedge to ordinary routes
    # would make this otherwise acyclic graph cyclic.
    assert not any(
        route["source"] == ("i3" if identity else "i4")
        and route["target"] == "i3"
        for route in artifact.pi.routes
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_region",
        "wrong_loop",
        "init_inside",
        "state_outside",
        "result_inside",
        "missing_init_route",
        "missing_next_route",
        "no_observable_consumer",
        "carrier_type_mismatch",
        "iteration_mismatch",
    ],
)
def test_loop_carrier_dependent_contract_fails_closed(mutation: str) -> None:
    artifact = _tiled_loop_artifact()
    _add_pointer_loop_carrier(artifact)
    relation = artifact.pi.loop_carriers[0]
    instances = {item["id"]: item for item in artifact.pi.instances}
    if mutation == "wrong_region":
        relation["region"] = "r1"
    elif mutation == "wrong_loop":
        instances["i0"]["semantic_ref"] = "n1"
    elif mutation == "init_inside":
        instances["i2"]["region"] = "r2"
    elif mutation == "state_outside":
        instances["i3"]["region"] = "r1"
    elif mutation == "result_inside":
        instances["i5"]["region"] = "r2"
    elif mutation == "missing_init_route":
        artifact.pi.routes = [
            item for item in artifact.pi.routes
            if (item["source"], item["target"]) != ("i2", "i3")
        ]
    elif mutation == "missing_next_route":
        artifact.pi.routes = [
            item for item in artifact.pi.routes
            if (item["source"], item["target"]) != ("i3", "i4")
        ]
    elif mutation == "no_observable_consumer":
        artifact.pi.routes = [
            item for item in artifact.pi.routes if item["target"] != "i6"
        ]
    elif mutation == "carrier_type_mismatch":
        instances["i4"]["semantic_ref"] = "p2"
    elif mutation == "iteration_mismatch":
        artifact.pi.iterations[0]["loop_instance"] = "i1"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    assert TypeCheckErrorCode.LOOP_CARRIER_INVALID in _codes(artifact)


@pytest.mark.parametrize("forbidden_ref", ["c0", "x0", "q0", "b0", "e0"])
def test_loop_carrier_roles_reject_noncarrier_anchor_classes(
    forbidden_ref: str,
) -> None:
    artifact = _canonical_j_carry_artifact()
    state_id = artifact.pi.loop_carriers[0]["state"]
    state = next(item for item in artifact.pi.instances if item["id"] == state_id)
    state["semantic_ref"] = forbidden_ref

    assert TypeCheckErrorCode.LOOP_CARRIER_INVALID in _codes(artifact)


def test_loop_carrier_allows_multiple_ordinals_on_one_j_loop_and_same_p_handle(
) -> None:
    artifact = _tiled_loop_artifact()
    _add_pointer_loop_carrier(artifact)
    payload = artifact.pi.to_json()
    payload["instances"].extend(
        [
            {"id": "i7", "semantic_ref": "p0", "region": "r2", "mechanism": "realize"},
            {"id": "i8", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
        ]
    )
    payload["routes"].extend(
        [
            {"source": "i2", "target": "i7", "slots": []},
            {"source": "i7", "target": "i4", "slots": []},
            {"source": "i7", "target": "i6", "slots": []},
        ]
    )
    payload["loop_carriers"].append(
        {
            "region": "r2",
            "loop_instance": "i0",
            "init": "i2",
            "state": "i7",
            "next": "i4",
            "result": "i8",
        }
    )
    artifact.pi = RealizationWitness.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.LOOP_CARRIER_INVALID not in {
        item.code for item in report.diagnostics
    }, report.to_json()
    assert [item["state"] for item in artifact.pi.loop_carriers] == ["i3", "i7"]


def _canonical_j_carry_artifact(*, referenced_init: bool = False) -> Gate1Artifact:
    payload = _tiled_loop_artifact().to_json()
    mapping = {
        "semantic_scan": "scrc_j0",
        "x": "t0",
        "y": "t1",
        "acc": "v0",
        "acc_iter": "v1",
        "x_tile": "v2",
        "tile_sum": "v3",
        "acc_next": "v4",
        "state0": "s0",
        "function0": "q0",
        "scope0": "q1",
        "storage0": "b0",
        "storage1": "b1",
    }

    def rename(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    payload["j"] = rename(payload["j"])
    payload["j"]["kernel_id"] = "scrc_k0"
    payload["j"]["boundary"]["accesses"][0]["index_expression"] = "[]"
    if referenced_init:
        payload["j"]["intent"]["intent_function"]["inputs"].append(
            {"name": "t2", "dtype": "f32", "shape": []}
        )
        carry = payload["j"]["intent"]["intent_function"]["body"][0]["carry"][0]
        carry["init_value"] = "t2"
        payload["j"]["signature"]["parameters"].append(
            {
                "name": "p3",
                "kind": "pointer",
                "mode": "input",
                "dtype": "f32",
                "rank": 0,
                "logical_ref": "t2",
            }
        )
        payload["j"]["scopes"][0]["definitions"].insert(1, "t2")
        payload["j"]["scopes"][1]["captures"].append("t2")
        payload["j"]["storage"][0]["members"].insert(0, "t2")

    payload["pi"]["instances"].extend(
        [
            {
                "id": "i2",
                "semantic_ref": "t2" if referenced_init else "s0",
                "region": "r1",
                "mechanism": "realize",
            },
            {"id": "i3", "semantic_ref": "s0", "region": "r2", "mechanism": "realize"},
            {"id": "i4", "semantic_ref": "v4", "region": "r2", "mechanism": "realize"},
            {"id": "i5", "semantic_ref": "v0", "region": "r1", "mechanism": "realize"},
            {
                "id": "i6",
                "semantic_ref": "a0",
                "region": "r2",
                "mechanism": "transfer_direct",
            },
        ]
    )
    payload["pi"]["routes"].extend(
        [
            {"source": "i2", "target": "i3", "slots": []},
            {"source": "i3", "target": "i4", "slots": []},
            {"source": "i3", "target": "i6", "slots": []},
        ]
    )
    payload["pi"]["loop_carriers"] = [
        {
            "region": "r2",
            "loop_instance": "i0",
            "init": "i2",
            "state": "i3",
            "next": "i4",
            "result": "i5",
        }
    ]
    return Gate1Artifact.from_json(payload)


@pytest.mark.parametrize("referenced_init", [False, True])
def test_loop_carrier_reuses_one_j_carry_for_literal_or_referenced_init(
    referenced_init: bool,
) -> None:
    artifact = _canonical_j_carry_artifact(referenced_init=referenced_init)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.LOOP_CARRIER_INVALID not in {
        item.code for item in report.diagnostics
    }, report.to_json()


def test_loop_carrier_rejects_a_second_accumulator_for_one_j_carry_ordinal() -> None:
    artifact = _canonical_j_carry_artifact()
    payload = artifact.pi.to_json()
    payload["instances"].extend(
        [
            {"id": "i7", "semantic_ref": "s0", "region": "r2", "mechanism": "realize"},
            {"id": "i8", "semantic_ref": "v0", "region": "r1", "mechanism": "realize"},
        ]
    )
    payload["routes"].extend(
        [
            {"source": "i2", "target": "i7", "slots": []},
            {"source": "i7", "target": "i4", "slots": []},
            {"source": "i7", "target": "i6", "slots": []},
        ]
    )
    payload["loop_carriers"].append(
        {
            "region": "r2",
            "loop_instance": "i0",
            "init": "i2",
            "state": "i7",
            "next": "i4",
            "result": "i8",
        }
    )
    artifact.pi = RealizationWitness.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.LOOP_CARRIER_INVALID
        and "second accumulator" in item.detail
        for item in report.diagnostics
    )


def test_loop_carrier_wire_rejects_unknown_fields_and_reused_state_result_slots(
) -> None:
    artifact = _tiled_loop_artifact()
    _add_pointer_loop_carrier(artifact)
    payload = artifact.pi.to_json()
    payload["loop_carriers"][0]["formula"] = "next[k]"
    with pytest.raises(RealizationWitnessError, match="forbidden fields"):
        RealizationWitness.from_json(payload)

    payload = artifact.pi.to_json()
    payload["loop_carriers"].append(copy.deepcopy(payload["loop_carriers"][0]))
    with pytest.raises(RealizationWitnessError, match="state slots must be unique"):
        RealizationWitness.from_json(payload)


def test_candidate_check_uses_v4_v8_and_reports_missing_consumer_relation() -> None:
    artifact = make_reduce_artifact()
    context = SCRCCandidateCheckContext(("p0", "p1", "p2"))

    accepted = check_scrc_lift_candidate(
        json.dumps(artifact.to_json(), sort_keys=True), context
    )
    assert accepted["status"] == "pass", accepted

    artifact.j.boundary.consumer_extensions = []

    result = check_scrc_lift_candidate(
        json.dumps(artifact.to_json(), sort_keys=True), context
    )

    assert SCRC_DECODE_PROTOCOL == "intentir.scrc.decode.v4"
    assert TYPECHECK_PROTOCOL == "intentir.scrc.typecheck.v8"
    assert result["status"] == "fail"
    assert any(
        item["rule_code"] == "consumer_extension_missing"
        for item in result["diagnostics"]
    )
