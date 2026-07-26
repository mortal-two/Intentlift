"""Artifact.v7 typed pure value-conditional tests."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from intentir.extraction.lifters.scrc import (
    SCRC_DECODE_PROTOCOL,
    SCRCCandidateCheckContext,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
)
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.ir.functional_boundary import (
    predicate_dominated_pointwise_load,
    project_functional_boundary,
)
from intentir.verify.gate1.model import Frontend, Gate1Artifact, SchemaError
from intentir.verify.gate1.orchestrate import canonical_artifact_digest
from intentir.verify.gate1.prompt import (
    build_masked_rewriter_view,
    build_normal_rewriter_prompt,
    build_normal_rewriter_view,
)
from intentir.verify.gate1.replay import replay_source_ingress
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.typecheck import TYPECHECK_PROTOCOL, TypeCheckErrorCode, type_check
from test_scrc_typed_control_core import _effect_if_payload, _request
from test_gate1_loop_schema import _tiled_loop_artifact


def _value_if_payload() -> dict:
    payload = _effect_if_payload()
    payload["schema"] = Gate1Artifact.SCHEMA
    j = payload["j"]
    function = j["intent"]["intent_function"]
    function["inputs"] = [
        {"name": "t0", "dtype": "f32", "shape": [128]},
        {"name": "t1", "dtype": "i1", "shape": []},
    ]
    function["outputs"] = [{"name": "t2", "dtype": "f32", "shape": [128]}]
    function["returns"] = [{"output": "t2", "value": "t2"}]
    function["body"] = [
        {
            "id": "n0",
            "op": "value_if",
            "result": "v1",
            "type": {"dtype": "f32", "shape": [128]},
            "predicate": "t1",
            "positive": {
                "scope_id": "q1",
                "body": [
                    {
                        "id": "n1",
                        "op": "map_binary",
                        "result": "v0",
                        "operands": ["t0", "t0"],
                        "attrs": {"fn": "add"},
                        "type": {"dtype": "f32", "shape": [128]},
                    }
                ],
                "yield_value": "v0",
            },
            "negative": {
                "scope_id": "q2",
                "body": [],
                "yield_value": "t0",
            },
        },
        {
            "id": "n2",
            "op": "map_binary",
            "result": "t2",
            "operands": ["v1", "t0"],
            "attrs": {"fn": "mul"},
            "type": {"dtype": "f32", "shape": [128]},
        },
    ]
    j["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[0].positive.body[0]",
        "n2": "body[1]",
    }
    j["boundary"] = {
        "domains": {},
        "accesses": [],
        "broadcasts": [],
        "aliases": [],
        "effects": [],
        "output_mappings": [],
        "reductions": [],
        "index_bounds": [],
        "updates": [],
        "result_relations": [{"output": "t2", "kind": "tolerance"}],
        "realization_authorizations": [],
        "consumer_extensions": [],
        "access_guard_relations": [],
    }
    j["signature"]["parameters"][0].update(dtype="f32", rank=1)
    j["signature"]["parameters"][1].update(dtype="f32", rank=1)
    j["scopes"] = [
        {
            "scope_id": "q0",
            "kind": "function",
            "parent_scope": None,
            "captures": [],
            "definitions": ["t0", "t1", "v1", "t2"],
            "effects": [],
        },
        {
            "scope_id": "q1",
            "kind": "branch_region",
            "parent_scope": "q0",
            "captures": ["t0"],
            "definitions": ["v0"],
            "effects": [],
        },
        {
            "scope_id": "q2",
            "kind": "branch_region",
            "parent_scope": "q0",
            "captures": ["t0"],
            "definitions": [],
            "effects": [],
        },
    ]
    return payload


def test_value_if_v7_schema_decode_round_trip_and_digest() -> None:
    payload = _value_if_payload()
    assert Gate1Artifact.SCHEMA == "intentir.scrc.artifact.v7"
    assert SCRC_DECODE_PROTOCOL == "intentir.scrc.decode.v10"
    assert TYPECHECK_PROTOCOL == "intentir.scrc.typecheck.v16"
    Draft202012Validator(scrc_lift_core_json_schema()).validate(payload)
    artifact = decode_scrc_lift_artifact(json.dumps(payload), _request())
    assert artifact.to_json() == payload
    assert canonical_artifact_digest(artifact) == canonical_artifact_digest(
        Gate1Artifact.from_json(copy.deepcopy(payload))
    )
    changed = copy.deepcopy(payload)
    changed["j"]["intent"]["intent_function"]["body"][0]["positive"][
        "yield_value"
    ] = "t0"
    assert canonical_artifact_digest(artifact) != canonical_artifact_digest(
        Gate1Artifact.from_json(changed)
    )


def test_value_if_wire_is_closed_single_result_and_has_two_yields() -> None:
    payload = _value_if_payload()
    for mutation in (
        lambda op: op.pop("negative"),
        lambda op: op["positive"].pop("yield_value"),
        lambda op: op.update(result=["v1", "v2"]),
        lambda op: op.update(operands=["t1"]),
        lambda op: op["positive"].update(effect_ids=[]),
    ):
        malformed = copy.deepcopy(payload)
        mutation(malformed["j"]["intent"]["intent_function"]["body"][0])
        with pytest.raises(Exception):
            decode_scrc_lift_artifact(json.dumps(malformed), _request())


def test_value_if_typecheck_and_candidate_check_share_acceptance() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    type_report = type_check(artifact.j, artifact.c, artifact.pi)
    assert type_report.passed, [item.to_json() for item in type_report.diagnostics]
    report = check_scrc_lift_candidate(
        json.dumps(artifact.to_json()),
        SCRCCandidateCheckContext(expected_parameters=("p0", "p1", "p2")),
    )
    assert report["status"] == "pass", report


@pytest.mark.parametrize("flag", [False, True])
def test_structured_refeval_executes_typed_value_if(flag: bool) -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    spec = ScalarLoweringSpec.create(
        artifact.j,
        dims={},
        input_bindings={"t0": "input", "t1": "flag"},
        output_bindings={"t2": "output"},
    )
    value = np.arange(128, dtype=np.float32)

    result = ScalarLoweringRunner().run(
        spec,
        {
            "input": value,
            "flag": np.asarray(flag, dtype=np.bool_),
        },
    )

    assert result.ok, result.error
    expected = value * value * (2 if flag else 1)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    evidence = result.metadata["execution_evidence"]
    assert evidence["value_if_count"] == 1
    assert evidence["effect_if_count"] == 0


def test_value_if_pointwise_mask_dominance_uses_canonical_output_binding() -> None:
    payload = _value_if_payload()
    j = payload["j"]
    j["boundary"]["domains"] = {"d0": "0 <= k0 and k0 < 128"}
    j["boundary"]["accesses"] = [
        {
            "access_id": "a0",
            "kind": "load",
            "tensor": "t0",
            "node_ref": "n0",
            "predicate": "k0 < 128",
            "index_expression": "k0",
        },
        {
            "access_id": "a1",
            "kind": "store",
            "tensor": "t2",
            "node_ref": "n2",
            "predicate": "k0 < 128",
            "index_expression": "k0",
        },
    ]
    j["boundary"]["effects"] = [
        {
            "effect_id": "e0",
            "kind": "read",
            "target": "t0",
            "node_ref": "n0",
            "access_ref": "a0",
        },
        {
            "effect_id": "e1",
            "kind": "write",
            "target": "t2",
            "node_ref": "n2",
            "access_ref": "a1",
        },
    ]
    j["scopes"][0]["effects"] = ["e0", "e1"]

    artifact = Gate1Artifact.from_json(payload)

    assert artifact.j.boundary.output_mappings == []
    assert predicate_dominated_pointwise_load(artifact.j, "a0")
    projection = project_functional_boundary(artifact.j)
    assert "predicate_dominated_load:a0:k0 < 128" in projection.consumed


def test_value_if_normal_and_masked_views_preserve_control_and_yields() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    normal = build_normal_rewriter_view(artifact)
    operation = normal["j"]["intent"]["intent_function"]["body"][0]
    assert operation["op"] == "value_if"
    assert operation["result"]
    assert operation["positive"]["yield_value"]
    assert operation["negative"]["yield_value"]
    masked = build_masked_rewriter_view(artifact)
    masked_op = masked["j"]["intent"]["intent_function"]["body"][0]
    assert set(masked_op) >= {"result", "predicate", "positive", "negative"}
    assert "yield_value" in masked_op["positive"]
    assert "yield_value" in masked_op["negative"]


def _pure_value_if_replay_source() -> str:
    return """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    value = tl.load(p0)
    if p2:
        positive = value + value
        merged = positive
    else:
        negative = value
        merged = negative
    result = merged * value
    tl.store(p1, result)
"""


def test_value_if_replay_contract_is_hard_distinct_and_source_neutral() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    instructions, packet = build_normal_rewriter_prompt(artifact)

    assert "value_if as pure value control" in instructions
    assert "same merge local" in instructions
    assert "Never lower value_if to tl.where/select" in instructions
    assert "distinct from effect_if and select" in instructions
    for forbidden in ("source_kernel", "flash_attn", "Gate1 diff"):
        assert forbidden not in instructions + packet


def test_value_if_replay_check_accepts_runtime_if_and_rejects_select() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    accepted = _pure_value_if_replay_source()
    selected = accepted.replace(
        "    if p2:\n"
        "        positive = value + value\n"
        "        merged = positive\n"
        "    else:\n"
        "        negative = value\n"
        "        merged = negative\n",
        "    merged = tl.where(p2, value + value, value)\n",
    )

    assert replay_source_ingress(
        accepted, artifact=artifact, frontend=Frontend.TRITON
    ).status == "pass"
    result = replay_source_ingress(
        selected, artifact=artifact, frontend=Frontend.TRITON
    )
    assert result.to_check_response()["diagnostics"][0]["rule_code"] == (
        "value_if_runtime_count"
    )


def test_value_if_replay_check_requires_else_and_common_merge_target() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    accepted = _pure_value_if_replay_source()
    missing_else = accepted.replace(
        "    else:\n"
        "        negative = value\n"
        "        merged = negative\n",
        "",
    )
    split_merge = accepted.replace(
        "        merged = positive\n",
        "        merged1 = positive\n",
    ).replace(
        "        merged = negative\n",
        "        merged2 = negative\n",
    ).replace(
        "    result = merged * value\n",
        "    result = value * value\n",
    )

    missing_result = replay_source_ingress(
        missing_else, artifact=artifact, frontend=Frontend.TRITON
    )
    split_result = replay_source_ingress(
        split_merge, artifact=artifact, frontend=Frontend.TRITON
    )
    assert missing_result.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "value_if_explicit_negative_arm"
    assert split_result.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "value_if_merge_target"


def test_mixed_typed_controls_match_runtime_ifs_in_j_lexical_order() -> None:
    artifact = Gate1Artifact.from_json(_value_if_payload())
    packet = build_normal_rewriter_view(artifact)
    body = packet["j"]["intent"]["intent_function"]["body"]
    body.insert(
        1,
        {
            "op": "effect_if",
            "predicate": "t1",
            "positive": {"body": [{"op": "opaque_effect"}]},
            "negative": {"body": []},
        },
    )
    value_control = (
        "    if p2:\n"
        "        positive = value + value\n"
        "        merged = positive\n"
        "    else:\n"
        "        negative = value\n"
        "        merged = negative\n"
    )
    effect_control = (
        "    if p2:\n"
        "        tl.store(p1, merged)\n"
        "    else:\n"
        "        pass\n"
    )
    prefix = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    value = tl.load(p0)
"""
    ordered = prefix + value_control + effect_control
    swapped = prefix + effect_control.replace("merged", "value") + value_control

    assert replay_source_ingress(
        ordered,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    ).status == "pass"
    result = replay_source_ingress(
        swapped,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    )
    assert result.to_check_response()["diagnostics"][0]["rule_code"] == (
        "value_if_merge_target"
    )


def test_v6_effect_if_replay_contract_remains_compatible() -> None:
    payload = _effect_if_payload()
    payload["schema"] = "intentir.scrc.artifact.v6"
    artifact = Gate1Artifact.from_json(payload)
    source = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    if p2:
        value = tl.load(p0)
        tl.store(p1, value)
    else:
        pass
"""

    assert replay_source_ingress(
        source, artifact=artifact, frontend=Frontend.TRITON
    ).status == "pass"


def test_value_if_predicate_yields_purity_and_scope_anchors_fail_closed() -> None:
    cases: list[tuple[dict, TypeCheckErrorCode]] = []
    non_boolean = _value_if_payload()
    non_boolean["j"]["intent"]["intent_function"]["inputs"][1]["dtype"] = "f32"
    non_boolean["j"]["signature"]["parameters"][2]["dtype"] = "f32"
    cases.append((non_boolean, TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID))

    unknown_yield = _value_if_payload()
    unknown_yield["j"]["intent"]["intent_function"]["body"][0]["negative"][
        "yield_value"
    ] = "missing"
    with pytest.raises(SchemaError):
        Gate1Artifact.from_json(unknown_yield)

    wrong_type = _value_if_payload()
    wrong_type["j"]["intent"]["intent_function"]["body"][0]["type"] = {
        "dtype": "f16",
        "shape": [128],
    }
    cases.append((wrong_type, TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID))

    effect = _value_if_payload()
    effect["j"]["scopes"][1]["effects"] = ["e0"]
    effect["j"]["boundary"]["effects"] = [
        {"effect_id": "e0", "kind": "read", "target": "t0", "node_ref": "n1"}
    ]
    cases.append((effect, TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID))

    shared_scope = _value_if_payload()
    shared_scope["j"]["intent"]["intent_function"]["body"][0]["negative"][
        "scope_id"
    ] = "q1"
    with pytest.raises(SchemaError):
        Gate1Artifact.from_json(shared_scope)

    wrong_anchor = _value_if_payload()
    wrong_anchor["pi"]["regions"][2]["placement"] = "root"
    cases.append((wrong_anchor, TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID))

    for payload, code in cases:
        artifact = Gate1Artifact.from_json(payload)
        report = type_check(artifact.j, artifact.c, artifact.pi)
        assert not report.passed
        assert any(item.code is code for item in report.diagnostics)


def test_value_if_rejects_unknown_nested_access_and_non_yield_escape() -> None:
    unknown = _value_if_payload()
    unknown["j"]["intent"]["intent_function"]["body"][0]["positive"]["body"][
        0
    ]["op"] = "invented_pure_op"
    artifact = Gate1Artifact.from_json(unknown)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert any(
        item.code is TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID
        and "known pure ordinary" in item.detail
        for item in report.diagnostics
    )

    access = _value_if_payload()
    access["j"]["boundary"]["accesses"] = [
        {
            "access_id": "a0",
            "kind": "load",
            "tensor": "t0",
            "node_ref": "n1",
            "index_expression": "k0",
        }
    ]
    access["j"]["boundary"]["effects"] = [
        {
            "effect_id": "e0",
            "kind": "read",
            "target": "t0",
            "node_ref": "n1",
            "access_ref": "a0",
        }
    ]
    access["j"]["scopes"][1]["effects"] = ["e0"]
    artifact = Gate1Artifact.from_json(access)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert any(
        item.code is TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID
        and "logical accesses" in item.detail
        for item in report.diagnostics
    )

    escaped = _value_if_payload()
    escaped["j"]["intent"]["intent_function"]["body"][1]["operands"][0] = "v0"
    with pytest.raises(SchemaError, match="outside the current scope"):
        Gate1Artifact.from_json(escaped)

    sibling_yield = _value_if_payload()
    sibling_yield["j"]["intent"]["intent_function"]["body"][0]["negative"][
        "yield_value"
    ] = "v0"
    with pytest.raises(SchemaError, match="outside the authored arm"):
        Gate1Artifact.from_json(sibling_yield)

    self_yield = _value_if_payload()
    self_yield["j"]["intent"]["intent_function"]["body"][0]["positive"][
        "yield_value"
    ] = "v1"
    with pytest.raises(SchemaError, match="outside the authored arm"):
        Gate1Artifact.from_json(self_yield)

    nested = _value_if_payload()
    nested_op = {
        "id": "n1",
        "op": "effect_if",
        "predicate": "t1",
        "positive": {"scope_id": "q3", "body": []},
        "negative": {"scope_id": "q4", "body": []},
    }
    nested["j"]["intent"]["intent_function"]["body"][0]["positive"].update(
        body=[nested_op], yield_value="t0"
    )
    nested["j"]["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[0].positive.body[0]",
        "n2": "body[1]",
    }
    nested["j"]["scopes"][1].update(captures=["t0", "t1"], definitions=[])
    nested["j"]["scopes"].extend(
        [
            {
                "scope_id": "q3",
                "kind": "branch_region",
                "parent_scope": "q1",
                "captures": [],
                "definitions": [],
                "effects": [],
            },
            {
                "scope_id": "q4",
                "kind": "branch_region",
                "parent_scope": "q1",
                "captures": [],
                "definitions": [],
                "effects": [],
            },
        ]
    )
    nested["pi"]["regions"].extend(
        [
            {"id": "r3", "parent": "r1", "placement": "branch"},
            {"id": "r4", "parent": "r1", "placement": "branch"},
        ]
    )
    nested["pi"]["instances"].extend(
        [
            {"id": "i2", "semantic_ref": "q3", "region": "r3", "mechanism": "realize"},
            {"id": "i3", "semantic_ref": "q4", "region": "r4", "mechanism": "realize"},
        ]
    )
    artifact = Gate1Artifact.from_json(nested)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert any(
        item.code is TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID
        and "no nested regions" in item.detail
        for item in report.diagnostics
    )


def test_value_if_pi_anchor_multiplicity_and_sibling_topology_fail_closed() -> None:
    missing = _value_if_payload()
    missing["pi"]["instances"] = missing["pi"]["instances"][:1]

    duplicate = _value_if_payload()
    duplicate["pi"]["instances"].append(
        {"id": "i2", "semantic_ref": "q1", "region": "r1", "mechanism": "realize"}
    )

    reused = _value_if_payload()
    reused["pi"]["instances"][1]["region"] = "r1"

    nonsibling = _value_if_payload()
    nonsibling["pi"]["regions"].append(
        {"id": "r3", "parent": "r1", "placement": "root"}
    )
    nonsibling["pi"]["regions"][2]["parent"] = "r3"

    for payload in (missing, duplicate, reused, nonsibling):
        artifact = Gate1Artifact.from_json(payload)
        report = type_check(artifact.j, artifact.c, artifact.pi)
        assert not report.passed
        assert any(
            item.code is TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID
            for item in report.diagnostics
        )


def test_two_sequential_value_if_regions_are_typed_and_may_feed_continuation() -> None:
    payload = _value_if_payload()
    second = {
        "id": "n3",
        "op": "value_if",
        "result": "v2",
        "type": {"dtype": "f32", "shape": [128]},
        "predicate": "t1",
        "positive": {"scope_id": "q3", "body": [], "yield_value": "v1"},
        "negative": {"scope_id": "q4", "body": [], "yield_value": "t0"},
    }
    body = payload["j"]["intent"]["intent_function"]["body"]
    body.insert(1, second)
    body[2]["operands"][0] = "v2"
    payload["j"]["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[0].positive.body[0]",
        "n3": "body[1]",
        "n2": "body[2]",
    }
    payload["j"]["scopes"][0]["definitions"].insert(3, "v2")
    payload["j"]["scopes"].extend(
        [
            {
                "scope_id": "q3",
                "kind": "branch_region",
                "parent_scope": "q0",
                "captures": ["v1"],
                "definitions": [],
                "effects": [],
            },
            {
                "scope_id": "q4",
                "kind": "branch_region",
                "parent_scope": "q0",
                "captures": ["t0"],
                "definitions": [],
                "effects": [],
            },
        ]
    )
    payload["pi"]["regions"].extend(
        [
            {"id": "r3", "parent": "r0", "placement": "branch"},
            {"id": "r4", "parent": "r0", "placement": "branch"},
        ]
    )
    payload["pi"]["instances"].extend(
        [
            {"id": "i2", "semantic_ref": "q3", "region": "r3", "mechanism": "realize"},
            {"id": "i3", "semantic_ref": "q4", "region": "r4", "mechanism": "realize"},
        ]
    )
    artifact = Gate1Artifact.from_json(payload)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed, [item.to_json() for item in report.diagnostics]


def test_v1_through_v6_reject_value_if_but_v6_keeps_typed_effect_control() -> None:
    payload = _value_if_payload()
    for version in range(1, 7):
        legacy = copy.deepcopy(payload)
        legacy["schema"] = f"intentir.scrc.artifact.v{version}"
        with pytest.raises(SchemaError):
            Gate1Artifact.from_json(legacy)

    v6_value = copy.deepcopy(payload)
    v6_value["schema"] = "intentir.scrc.artifact.v6"
    with pytest.raises(SchemaError, match="v1-v6 do not contain J value_if"):
        Gate1Artifact.from_json(v6_value)

    v6 = _effect_if_payload()
    v6["schema"] = "intentir.scrc.artifact.v6"
    assert Gate1Artifact.from_json(v6).to_json() == v6

    rounded_v6 = _tiled_loop_artifact(basis="rounded_extent_step").to_json()
    rounded_v6["schema"] = "intentir.scrc.artifact.v6"
    assert Gate1Artifact.from_json(rounded_v6).to_json() == rounded_v6
