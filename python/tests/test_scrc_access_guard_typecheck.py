from __future__ import annotations

import copy
import json

import pytest

from intentir.extraction.lifters.scrc import (
    SCRCCandidateCheckContext,
    build_scrc_lift_instructions,
    check_scrc_lift_candidate,
)
from intentir.verify.gate1.model import Gate1Artifact, SchemaError
from intentir.verify.typecheck import (
    TYPECHECK_PROTOCOL,
    TypeCheckErrorCode,
    type_check,
)
from test_scrc_artifact_v4 import _canonical_j_carry_artifact
from test_scrc_artifact_v5_access_guard import _guarded_payload


def _guarded_carry_payload() -> dict:
    """One generic conditional carry with an explicit inout pre-state."""

    payload = _canonical_j_carry_artifact().to_json()

    def rename_output(value):
        if isinstance(value, str):
            return "t3" if value == "t1" else value
        if isinstance(value, list):
            return [rename_output(item) for item in value]
        if isinstance(value, dict):
            return {key: rename_output(item) for key, item in value.items()}
        return value

    payload = rename_output(payload)
    function = payload["j"]["intent"]["intent_function"]
    loop = function["body"][0]
    function["inputs"].extend(
        [
            {"name": "t1", "dtype": "f32", "shape": []},
            {"name": "t2", "dtype": "i1", "shape": []},
        ]
    )
    function["body"] = [loop]
    function["returns"] = [{"output": "t3", "value": "v0"}]

    loop["carry"][0]["init_value"] = "t1"
    loop["carry"][0]["next_value"] = "v5"
    loop["yield"] = ["v5"]
    loop["body"][0]["type"] = {"dtype": "f32", "shape": []}
    loop["body"].append(
        {
            "op": "select",
            "result": "v5",
            "operands": ["t2", "v4", "v1"],
            "type": {"dtype": "f32", "shape": []},
        }
    )
    payload["j"]["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[0].body[0]",
        "n2": "body[0].body[1]",
        "n3": "body[0].body[2]",
    }

    boundary = payload["j"]["boundary"]
    boundary["accesses"][0]["node_ref"] = "n3"
    boundary["effects"][0]["node_ref"] = "n3"
    boundary["output_mappings"] = [
        {"output": "t3", "value": "v0", "node_ref": "n0"}
    ]
    boundary["access_guard_relations"] = [
        {
            "access_ref": "a0",
            "value_ref": "t2",
            "axes": [],
            "polarity": "positive",
        }
    ]

    parameters = payload["j"]["signature"]["parameters"]
    parameters[1]["mode"] = "inout"
    parameters.extend(
        [
            {
                "name": "p3",
                "kind": "pointer",
                "mode": "input",
                "dtype": "f32",
                "rank": 0,
                "logical_ref": "t1",
            },
            {
                "name": "p4",
                "kind": "scalar",
                "mode": "value",
                "dtype": "i1",
                "rank": 0,
                "logical_ref": "t2",
            },
        ]
    )

    root, loop_scope = payload["j"]["scopes"]
    root["definitions"] = ["t0", "t1", "t2", "v0"]
    root["effects"] = []
    loop_scope["captures"] = ["t0", "t1", "t2"]
    loop_scope["definitions"] = ["v2", "v1", "v3", "v4", "v5"]
    loop_scope["effects"] = ["e0"]

    carry_storage = payload["j"]["storage"][0]
    carry_storage["members"] = ["t1", "s0", "v1", "v5", "v0"]
    carry_storage["last_use_event"] = {"kind": "node", "ref": "n3"}
    payload["j"]["storage"] = [carry_storage]

    payload["pi"]["instances"][2]["semantic_ref"] = "t1"
    payload["pi"]["instances"][4]["semantic_ref"] = "v5"
    return payload


def _guarded_carry_artifact() -> Gate1Artifact:
    return Gate1Artifact.from_json(_guarded_carry_payload())


def test_access_guard_typecheck_accepts_generic_rank_projection_and_load_fill() -> None:
    artifact = Gate1Artifact.from_json(_guarded_payload())

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert report.protocol == TYPECHECK_PROTOCOL == "intentir.scrc.typecheck.v14"


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (
            lambda relation, _payload: relation.update({"axes": [1, 1]}),
            "strictly increasing",
        ),
        (
            lambda relation, _payload: relation.update({"axes": [0]}),
            "cannot project",
        ),
        (
            lambda relation, payload: payload["j"]["intent"]
            ["intent_function"]["inputs"][1].update({"dtype": "f32"})
            or payload["j"]["signature"]["parameters"][3].update(
                {"dtype": "f32"}
            ),
            "canonical dtype i1",
        ),
    ],
)
def test_access_guard_rejects_bad_axes_or_non_boolean_value(
    mutation,
    detail: str,
) -> None:
    payload = _guarded_payload()
    relation = payload["j"]["boundary"]["access_guard_relations"][0]
    mutation(relation, payload)
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID
        and detail in item.detail
        for item in report.diagnostics
    ), report.to_json()


def test_access_guard_rejects_duplicate_pair_and_predicate_authority() -> None:
    payload = _guarded_payload()
    relation = payload["j"]["boundary"]["access_guard_relations"][0]
    payload["j"]["boundary"]["access_guard_relations"].append(
        copy.deepcopy(relation)
    )
    payload["j"]["boundary"]["accesses"][0]["predicate"] = (
        "(k1 < x1) and t1"
    )
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    details = [item.detail for item in report.diagnostics]
    assert any("pair may be declared only once" in item for item in details)
    assert any("must not also appear" in item for item in details)
    assert any("exactly one guard per access" in item for item in details)


def test_relation_guarded_load_requires_fill_but_not_a_copied_predicate() -> None:
    payload = _guarded_payload()
    access = payload["j"]["boundary"]["accesses"][0]
    access.pop("fill_value")
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID
        and "requires an explicit fill_value" in item.detail
        for item in report.diagnostics
    ), report.to_json()


def test_access_guard_value_must_dominate_the_access_event() -> None:
    payload = _guarded_payload()
    function = payload["j"]["intent"]["intent_function"]
    function["body"].insert(
        1,
        {
            "id": "n1",
            "op": "map_binary",
            "result": "v1",
            "operands": ["t0", "t0"],
            "attrs": {"fn": "gt"},
            "type": {"dtype": "i1", "shape": ["x0", "x1"]},
        },
    )
    function["body"][2]["id"] = "n2"
    payload["j"]["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[1]",
        "n2": "body[2]",
    }
    boundary = payload["j"]["boundary"]
    boundary["access_guard_relations"][0].update(
        {"value_ref": "v1", "axes": []}
    )
    boundary["accesses"][1]["node_ref"] = "n2"
    boundary["reductions"][0]["node_ref"] = "n2"
    boundary["consumer_extensions"][0]["consumer_ref"] = "n2"
    for instance in payload["pi"]["instances"]:
        if instance["semantic_ref"] == "n1":
            instance["semantic_ref"] = "n2"
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.BOUNDARY_VALUE_NOT_DOMINATING
        and item.path.endswith("access_guard_relations[0].value_ref")
        for item in report.diagnostics
    ), report.to_json()


def test_output_only_name_and_unbacked_next_value_are_not_declarations() -> None:
    output_only = _guarded_payload()
    function = output_only["j"]["intent"]["intent_function"]
    function["outputs"].append(
        {"name": "t3", "dtype": "i1", "shape": ["x0"]}
    )
    output_only["j"]["signature"]["parameters"].append(
        {
            "name": "p4",
            "kind": "pointer",
            "mode": "output",
            "dtype": "i1",
            "rank": 1,
            "logical_ref": "t3",
        }
    )
    output_only["j"]["boundary"]["access_guard_relations"][0][
        "value_ref"
    ] = "t3"
    with pytest.raises(SchemaError, match="unknown Layer-A value"):
        Gate1Artifact.from_json(output_only)

    unbacked_next = _guarded_carry_payload()
    loop = unbacked_next["j"]["intent"]["intent_function"]["body"][0]
    loop["carry"][0]["next_value"] = "v6"
    loop["yield"] = ["v6"]
    unbacked_next["j"]["boundary"]["access_guard_relations"][0][
        "value_ref"
    ] = "v6"
    with pytest.raises(SchemaError, match="next_value must be defined"):
        Gate1Artifact.from_json(unbacked_next)


def test_guarded_carry_accepts_explicit_positive_select_merge() -> None:
    artifact = _guarded_carry_artifact()

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (
            lambda payload: payload["j"]["signature"]["parameters"][1].update(
                {"mode": "output"}
            ),
            "mode=inout",
        ),
        (
            lambda payload: payload["j"]["intent"]["intent_function"]
            ["body"][0]["carry"][0].update({"init_value": "ZERO"}),
            "explicit typed Layer-A input pre-state",
        ),
        (
            lambda payload: payload["j"]["intent"]["intent_function"]
            ["body"][0]["body"][2].update({"operands": ["t2", "v1", "v4"]}),
            "inactive under the authored polarity",
        ),
        (
            lambda payload: payload["j"]["intent"]["intent_function"]
            ["body"][0]["carry"][0].update({"next_value": "v4"}),
            "carry.next_value",
        ),
    ],
)
def test_guarded_carry_rejects_missing_prestate_or_wrong_merge(
    mutation,
    detail: str,
) -> None:
    payload = _guarded_carry_payload()
    mutation(payload)
    if detail == "explicit typed Layer-A input pre-state":
        payload["j"]["scopes"][1]["captures"].remove("t1")
        payload["j"]["storage"][0]["members"].remove("t1")
        payload["pi"]["instances"][2]["semantic_ref"] = "s0"
    if detail == "carry.next_value":
        payload["j"]["intent"]["intent_function"]["body"][0]["yield"] = [
            "v4"
        ]
        members = payload["j"]["storage"][0]["members"]
        members[members.index("v5")] = "v4"
        payload["pi"]["instances"][4]["semantic_ref"] = "v4"
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID
        and detail in item.detail
        for item in report.diagnostics
    ), report.to_json()


def test_guarded_carry_negative_polarity_reverses_inactive_branch() -> None:
    payload = _guarded_carry_payload()
    payload["j"]["boundary"]["access_guard_relations"][0][
        "polarity"
    ] = "negative"
    payload["j"]["intent"]["intent_function"]["body"][0]["body"][2][
        "operands"
    ] = ["t2", "v1", "v4"]
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_access_guard_v1_rejects_multiple_writes_to_one_carry() -> None:
    payload = _guarded_carry_payload()
    boundary = payload["j"]["boundary"]
    second_access = copy.deepcopy(boundary["accesses"][0])
    second_access["access_id"] = "a1"
    boundary["accesses"].append(second_access)
    second_effect = copy.deepcopy(boundary["effects"][0])
    second_effect["effect_id"] = "e1"
    second_effect["access_ref"] = "a1"
    boundary["effects"].append(second_effect)
    boundary["access_guard_relations"].append(
        {
            "access_ref": "a1",
            "value_ref": "t2",
            "axes": [],
            "polarity": "positive",
        }
    )
    payload["j"]["scopes"][1]["effects"].append("e1")
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID
        and "multiple guarded writes to one loop carry" in item.detail
        for item in report.diagnostics
    ), report.to_json()


def test_guarded_carry_rejects_duplicate_select_broadcast_authority() -> None:
    payload = _guarded_carry_payload()
    projection = {"node_ref": "n3", "operand": "t2", "axes": []}
    payload["j"]["boundary"]["broadcasts"] = [
        projection,
        copy.deepcopy(projection),
    ]
    artifact = Gate1Artifact.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert any(
        item.code is TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID
        and "multiple independent broadcast declarations" in item.detail
        for item in report.diagnostics
    ), report.to_json()


def test_candidate_check_reuses_authoritative_access_guard_typecheck() -> None:
    payload = _guarded_payload()
    payload["j"]["boundary"]["access_guard_relations"][0]["axes"] = [0]
    context = SCRCCandidateCheckContext(("p0", "p1", "p2", "p3"))

    result = check_scrc_lift_candidate(
        json.dumps(payload, sort_keys=True),
        context,
    )

    assert result["status"] == "fail"
    assert any(
        item["rule_code"] == "access_guard_relation_invalid"
        for item in result["diagnostics"]
    )


def test_lift_rules_keep_access_guard_authorship_explicit_and_generic() -> None:
    instructions = " ".join(build_scrc_lift_instructions().split())

    assert "exactly one boundary.access_guard_relations item" in instructions
    assert "select(guard,candidate,current) when positive" in instructions
    assert "One carry has one guarded write" in instructions
