"""Artifact.v6 typed effect control and rounded induction-basis tests."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from direct_fixtures import make_reduce_artifact
from intentir.extraction.lifters.scrc import (
    SCRC_DECODE_PROTOCOL,
    SCRCCandidateCheckContext,
    SCRCHarnessBinding,
    SCRCLiftRequest,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
)
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import Frontend, Gate1Artifact, SchemaError
from intentir.verify.gate1.orchestrate import canonical_artifact_digest
from intentir.verify.gate1.prompt import (
    build_normal_rewriter_prompt,
    build_masked_rewriter_view,
    build_normal_rewriter_view,
)
from intentir.verify.gate1.replay import replay_source_ingress
from intentir.verify.typecheck import (
    TYPECHECK_PROTOCOL,
    TypeCheckErrorCode,
    type_check,
)
from test_gate1_loop_schema import _tiled_loop_artifact


def _effect_if_payload() -> dict:
    payload = make_reduce_artifact().to_json()
    payload["schema"] = Gate1Artifact.SCHEMA
    j = payload["j"]
    function = j["intent"]["intent_function"]
    function["inputs"] = [
        {"name": "t0", "dtype": "f32", "shape": ["x0", "x1"]},
        {"name": "t1", "dtype": "i1", "shape": []},
    ]
    function["outputs"] = [
        {"name": "t2", "dtype": "f32", "shape": ["x0"]},
    ]
    function["body"] = [
        {
            "id": "n0",
            "op": "effect_if",
            "predicate": "t1",
            "positive": {
                "scope_id": "q1",
                "body": [
                    {
                        "id": "n1",
                        "op": "map_binary",
                        "result": "v0",
                        "operands": ["t0", "t0"],
                        "attrs": {"fn": "mul"},
                        "type": {"dtype": "f32", "shape": ["x0", "x1"]},
                    },
                    {
                        "id": "n2",
                        "op": "reduce",
                        "result": "v1",
                        "operands": ["v0"],
                        "attrs": {
                            "fn": "sum",
                            "axes": [1],
                            "assoc": "float_nondeterministic",
                        },
                        "type": {"dtype": "f32", "shape": ["x0"]},
                    },
                ],
            },
            "negative": {"scope_id": "q2", "body": []},
        }
    ]
    j["node_ids"] = {
        "n0": "body[0]",
        "n1": "body[0].positive.body[0]",
        "n2": "body[0].positive.body[1]",
    }
    j["boundary"] = {
        "domains": {
            "d0": "0 <= k0 and k0 < x0",
            "d1": "0 <= k1 and k1 < x1",
        },
        "accesses": [
            {
                "access_id": "a0",
                "kind": "load",
                "tensor": "t0",
                "node_ref": "n1",
                "index_expression": "[k0, k1]",
            },
            {
                "access_id": "a1",
                "kind": "store",
                "tensor": "t2",
                "node_ref": "n2",
                "index_expression": "k0",
            },
        ],
        "broadcasts": [],
        "aliases": [],
        "effects": [
            {
                "effect_id": "e0",
                "kind": "read",
                "target": "t0",
                "node_ref": "n1",
                "access_ref": "a0",
            },
            {
                "effect_id": "e1",
                "kind": "write",
                "target": "t2",
                "node_ref": "n2",
                "access_ref": "a1",
            },
        ],
        "output_mappings": [],
        "reductions": [{"node_ref": "n2", "tile_axis": 0, "identity": 0}],
        "index_bounds": [],
        "updates": [],
        "result_relations": [{"output": "t2", "kind": "tolerance"}],
        "realization_authorizations": [],
        "consumer_extensions": [],
        "access_guard_relations": [],
    }
    j["signature"] = {
        "parameters": [
            {
                "name": "p0",
                "kind": "pointer",
                "mode": "input",
                "dtype": "f32",
                "rank": 2,
                "logical_ref": "t0",
            },
            {
                "name": "p1",
                "kind": "pointer",
                "mode": "output",
                "dtype": "f32",
                "rank": 1,
                "logical_ref": "t2",
            },
            {
                "name": "p2",
                "kind": "scalar",
                "mode": "value",
                "dtype": "i1",
                "rank": 0,
                "logical_ref": "t1",
            },
        ]
    }
    j["scopes"] = [
        {
            "scope_id": "q0",
            "kind": "function",
            "parent_scope": None,
            "captures": [],
            "definitions": ["t0", "t1"],
            "effects": [],
        },
        {
            "scope_id": "q1",
            "kind": "branch_region",
            "parent_scope": "q0",
            "captures": ["t0"],
            "definitions": ["v0", "v1"],
            "effects": ["e0", "e1"],
        },
        {
            "scope_id": "q2",
            "kind": "branch_region",
            "parent_scope": "q0",
            "captures": [],
            "definitions": [],
            "effects": [],
        },
    ]
    j["storage"] = []
    j["dimension_constraints"] = []
    j["value_dimension_relations"] = []
    payload["pi"] = {
        "regions": [
            {"id": "r0", "parent": None, "placement": "root"},
            {"id": "r1", "parent": "r0", "placement": "branch"},
            {"id": "r2", "parent": "r0", "placement": "branch"},
        ],
        "instances": [
            {
                "id": "i0",
                "semantic_ref": "q1",
                "region": "r1",
                "mechanism": "realize",
            },
            {
                "id": "i1",
                "semantic_ref": "q2",
                "region": "r2",
                "mechanism": "realize",
            },
        ],
        "routes": [],
        "order": [],
        "sharing": [],
        "refinements": [],
        "iterations": [],
        "loop_carriers": [],
    }
    return payload


def _request() -> SCRCLiftRequest:
    return SCRCLiftRequest(
        instructions="",
        user_input="",
        prompt_sha256="0" * 64,
        harness_binding=SCRCHarnessBinding(
            source_to_opaque=(
                ("source0", "p0"),
                ("source1", "p1"),
                ("source2", "p2"),
            ),
            source_function="source_kernel",
        ),
    )


def test_v6_effect_if_schema_decode_round_trip_and_digest() -> None:
    payload = _effect_if_payload()
    validator = Draft202012Validator(scrc_lift_core_json_schema())
    assert not list(validator.iter_errors(payload))

    artifact = Gate1Artifact.from_json(payload)
    assert artifact.to_json() == payload
    assert Gate1Artifact.SCHEMA == "intentir.scrc.artifact.v7"
    assert SCRC_DECODE_PROTOCOL == "intentir.scrc.decode.v8"
    assert TYPECHECK_PROTOCOL == "intentir.scrc.typecheck.v16"
    decoded = decode_scrc_lift_artifact(json.dumps(payload), _request())
    decoded_json = decoded.to_json()
    assert Gate1Artifact.from_json(decoded_json).to_json() == decoded_json
    assert decoded_json["j"]["intent"] == payload["j"]["intent"]

    changed = copy.deepcopy(payload)
    changed["j"]["intent"]["intent_function"]["body"][0]["negative"][
        "body"
    ] = [{"op": "const", "result": "v2", "attrs": {"value": 0}}]
    changed["j"]["scopes"][2]["definitions"] = ["v2"]
    changed["j"]["node_ids"]["n3"] = "body[0].negative.body[0]"
    assert canonical_artifact_digest(artifact) != canonical_artifact_digest(
        Gate1Artifact.from_json(changed)
    )


def test_effect_if_wire_is_closed_and_allows_nested_loop_with_empty_arm() -> None:
    schema = scrc_lift_core_json_schema()
    validator = Draft202012Validator(schema)
    payload = _effect_if_payload()
    operation = payload["j"]["intent"]["intent_function"]["body"][0]
    operation["positive"]["body"] = [
        _tiled_loop_artifact().to_json()["j"]["intent"]["intent_function"][
            "body"
        ][0]
    ]
    assert not list(validator.iter_errors(payload))
    assert operation["negative"]["body"] == []

    with_result = _effect_if_payload()
    with_result["j"]["intent"]["intent_function"]["body"][0]["result"] = "v9"
    assert list(validator.iter_errors(with_result))
    missing_arm = _effect_if_payload()
    del missing_arm["j"]["intent"]["intent_function"]["body"][0]["negative"]
    assert list(validator.iter_errors(missing_arm))
    duplicate_membership = _effect_if_payload()
    duplicate_membership["j"]["intent"]["intent_function"]["body"][0][
        "positive"
    ]["effects"] = ["e0"]
    assert list(validator.iter_errors(duplicate_membership))


def test_effect_if_typecheck_and_candidate_check_share_acceptance() -> None:
    payload = _effect_if_payload()
    artifact = Gate1Artifact.from_json(payload)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed, [item.to_json() for item in report.diagnostics]

    checked = check_scrc_lift_candidate(
        json.dumps(payload),
        SCRCCandidateCheckContext(expected_parameters=("p0", "p1", "p2")),
    )
    assert checked["status"] == "pass", checked


def test_effect_if_normal_and_masked_views_preserve_binary_structure() -> None:
    artifact = Gate1Artifact.from_json(_effect_if_payload())
    normal = build_normal_rewriter_view(artifact)
    operation = normal["j"]["intent"]["intent_function"]["body"][0]
    assert operation["op"] == "effect_if"
    assert operation["predicate"] == "t1"
    assert operation["positive"]["scope_id"] == "q1"
    assert len(operation["positive"]["body"]) == 2
    assert operation["negative"] == {"scope_id": "q2", "body": []}

    masked = build_masked_rewriter_view(artifact)
    operation = masked["j"]["intent"]["intent_function"]["body"][0]
    assert operation["predicate"] == "<masked>"
    assert set(operation) >= {"positive", "negative"}
    assert len(operation["positive"]["body"]) == 2
    assert operation["negative"]["body"] == []
    assert masked["pi"]["regions"][1]["placement"] == "branch"
    assert "predicate" not in masked["pi"]["regions"][1]


def test_effect_if_replay_contract_is_source_neutral_and_requires_runtime_control() -> None:
    artifact = Gate1Artifact.from_json(_effect_if_payload())
    instructions, packet = build_normal_rewriter_prompt(artifact)

    assert "one real runtime if" in instructions
    assert "explicit positive and negative arms" in instructions
    assert "Never lower effect_if to tl.where/select" in instructions
    encoded = instructions + packet
    for forbidden in (
        "source_kernel",
        "zero_centered_weight",
        "flash_attn",
        "Gate1 diff",
    ):
        assert forbidden not in encoded


def test_effect_if_replay_check_requires_runtime_dual_arm_polarity() -> None:
    artifact = Gate1Artifact.from_json(_effect_if_payload())
    accepted = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    if c0:
        marker = 0
    if p2:
        value = tl.load(p0)
        tl.store(p1, value)
    else:
        pass
"""
    folded = accepted.replace(
        "    if p2:\n"
        "        value = tl.load(p0)\n"
        "        tl.store(p1, value)\n"
        "    else:\n"
        "        pass\n",
        "    value = tl.load(p0)\n"
        "    selected = tl.where(p2, value, value)\n"
        "    tl.store(p1, selected)\n",
    )
    swapped = accepted.replace(
        "    if p2:\n"
        "        value = tl.load(p0)\n"
        "        tl.store(p1, value)\n"
        "    else:\n"
        "        pass\n",
        "    if p2:\n"
        "        pass\n"
        "    else:\n"
        "        value = tl.load(p0)\n"
        "        tl.store(p1, value)\n",
    )
    missing_negative = accepted.replace(
        "    else:\n"
        "        pass\n",
        "",
    )

    assert replay_source_ingress(
        accepted, artifact=artifact, frontend=Frontend.TRITON
    ).status == "pass"
    folded_result = replay_source_ingress(
        folded, artifact=artifact, frontend=Frontend.TRITON
    )
    swapped_result = replay_source_ingress(
        swapped, artifact=artifact, frontend=Frontend.TRITON
    )
    missing_negative_result = replay_source_ingress(
        missing_negative, artifact=artifact, frontend=Frontend.TRITON
    )
    assert folded_result.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "effect_if_runtime_count"
    assert swapped_result.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "effect_if_arm_polarity"
    assert missing_negative_result.to_check_response()["diagnostics"][0][
        "rule_code"
    ] == "effect_if_explicit_negative_arm"


def test_effect_if_replay_check_does_not_guess_control_kind_from_arm_shape() -> None:
    artifact = Gate1Artifact.from_json(_effect_if_payload())
    packet = build_normal_rewriter_view(artifact)
    operation = packet["j"]["intent"]["intent_function"]["body"][0]
    operation["op"] = "value_if"
    source_without_runtime_if = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    value = tl.load(p0)
    tl.store(p1, value)
"""

    assert replay_source_ingress(
        source_without_runtime_if,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    ).status == "pass"


def test_effect_if_requires_dominating_scalar_i1() -> None:
    non_boolean = _effect_if_payload()
    non_boolean["j"]["intent"]["intent_function"]["inputs"][1]["dtype"] = "i32"
    non_boolean["j"]["signature"]["parameters"][2]["dtype"] = "i32"
    artifact = Gate1Artifact.from_json(non_boolean)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID in {
        item.code for item in report.diagnostics
    }

    non_dominating = _effect_if_payload()
    operation = non_dominating["j"]["intent"]["intent_function"]["body"][0]
    operation["predicate"] = "v0"
    with pytest.raises(SchemaError, match="predicate references non-visible"):
        Gate1Artifact.from_json(non_dominating)


def test_effect_if_branch_local_ssa_cannot_escape() -> None:
    payload = _effect_if_payload()
    payload["j"]["intent"]["intent_function"]["body"].append(
        {
            "id": "n3",
            "op": "map_unary",
            "result": "v2",
            "operands": ["v1"],
            "attrs": {"fn": "neg"},
            "type": {"dtype": "f32", "shape": ["x0"]},
        }
    )
    payload["j"]["node_ids"]["n3"] = "body[1]"
    with pytest.raises(SchemaError, match="outside the current scope"):
        Gate1Artifact.from_json(payload)

    implicit_output = _effect_if_payload()
    body = implicit_output["j"]["intent"]["intent_function"]["body"][0]
    body["positive"]["body"][1]["result"] = "t2"
    implicit_output["j"]["scopes"][1]["definitions"][-1] = "t2"
    with pytest.raises(SchemaError, match="cannot implicitly define function outputs"):
        Gate1Artifact.from_json(implicit_output)


def test_effect_if_scope_effect_and_guard_authority_are_unique() -> None:
    shared_scope = _effect_if_payload()
    shared_scope["j"]["intent"]["intent_function"]["body"][0]["negative"][
        "scope_id"
    ] = "q1"
    with pytest.raises(SchemaError, match="scopes must be distinct"):
        Gate1Artifact.from_json(shared_scope)

    duplicate_effect = _effect_if_payload()
    duplicate_effect["j"]["scopes"][2]["effects"] = ["e0"]
    with pytest.raises(SchemaError, match="multiple portable scopes"):
        Gate1Artifact.from_json(duplicate_effect)

    duplicate_guard = _effect_if_payload()
    duplicate_guard["j"]["boundary"]["access_guard_relations"] = [
        {
            "access_ref": "a0",
            "value_ref": "t1",
            "axes": [0, 1],
            "polarity": "positive",
        }
    ]
    duplicate_guard["j"]["boundary"]["accesses"][0]["fill_value"] = 0
    artifact = Gate1Artifact.from_json(duplicate_guard)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert any(
        item.code is TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID
        and item.ref == "a0"
        for item in report.diagnostics
    )


def test_effect_if_pi_uses_only_distinct_sibling_branch_anchors() -> None:
    missing = _effect_if_payload()
    missing["pi"]["instances"] = missing["pi"]["instances"][:1]
    artifact = Gate1Artifact.from_json(missing)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID in {
        item.code for item in report.diagnostics
    }

    copied_predicate = _effect_if_payload()
    copied_predicate["pi"]["regions"][1]["predicate"] = "t1"
    with pytest.raises(SchemaError, match="forbidden fields"):
        Gate1Artifact.from_json(copied_predicate)

    escaped = _effect_if_payload()
    escaped["pi"]["instances"].append(
        {
            "id": "i2",
            "semantic_ref": "e0",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    artifact = Gate1Artifact.from_json(escaped)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert any(
        item.code is TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID
        and item.ref == "e0"
        for item in report.diagnostics
    )


def test_multiple_sequential_effect_if_regions_are_typed() -> None:
    payload = _effect_if_payload()
    function = payload["j"]["intent"]["intent_function"]
    function["body"].append(
        {
            "id": "n3",
            "op": "effect_if",
            "predicate": "t1",
            "positive": {"scope_id": "q3", "body": []},
            "negative": {"scope_id": "q4", "body": []},
        }
    )
    payload["j"]["node_ids"]["n3"] = "body[1]"
    payload["j"]["boundary"]["accesses"].append(
        {
            "access_id": "a2",
            "kind": "load",
            "tensor": "t0",
            "index_expression": "[k0, k1]",
        }
    )
    payload["j"]["boundary"]["effects"].append(
        {
            "effect_id": "e2",
            "kind": "read",
            "target": "t0",
            "access_ref": "a2",
        }
    )
    payload["j"]["scopes"].extend(
        [
            {
                "scope_id": "q3",
                "kind": "branch_region",
                "parent_scope": "q0",
                "captures": ["t0"],
                "definitions": [],
                "effects": ["e2"],
            },
            {
                "scope_id": "q4",
                "kind": "branch_region",
                "parent_scope": "q0",
                "captures": [],
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
            {
                "id": "i2",
                "semantic_ref": "q3",
                "region": "r3",
                "mechanism": "realize",
            },
            {
                "id": "i3",
                "semantic_ref": "q4",
                "region": "r4",
                "mechanism": "realize",
            },
        ]
    )
    artifact = Gate1Artifact.from_json(payload)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed, [item.to_json() for item in report.diagnostics]


def test_v1_through_v5_cannot_claim_v6_control_or_rounded_basis() -> None:
    conditional = _effect_if_payload()
    conditional["schema"] = "intentir.scrc.artifact.v5"
    with pytest.raises(SchemaError, match="v1-v5 do not contain J effect_if"):
        Gate1Artifact.from_json(conditional)

    rounded = _tiled_loop_artifact(basis="rounded_extent_step").to_json()
    rounded["schema"] = "intentir.scrc.artifact.v5"
    with pytest.raises(SchemaError, match="v1-v5 do not contain rounded"):
        Gate1Artifact.from_json(rounded)


def test_rounded_extent_step_is_closed_wire_only_and_typechecked() -> None:
    artifact = _tiled_loop_artifact(basis="rounded_extent_step")
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert not {
        TypeCheckErrorCode.ITERATION_WITNESS_INVALID,
        TypeCheckErrorCode.ITERATION_WITNESS_MISSING,
    }.intersection(item.code for item in report.diagnostics)
    relation = artifact.to_json()["pi"]["iterations"][0]
    assert relation == {
        "region": "r2",
        "loop_instance": "i0",
        "tile_instance": "i1",
        "induction_basis": "rounded_extent_step",
    }
    assert not ({"lower", "upper", "step", "formula"} & set(relation))

    unknown = artifact.to_json()
    unknown["pi"]["iterations"][0]["upper"] = "ceil_div(x0,c0)*c0"
    with pytest.raises(SchemaError, match="forbidden fields"):
        Gate1Artifact.from_json(unknown)


def test_rounded_extent_step_views_and_prompt_preserve_declared_basis_only() -> None:
    artifact = _tiled_loop_artifact(basis="rounded_extent_step")
    normal = artifact.rewriter_view(masked=False)
    masked = artifact.rewriter_view(masked=True)
    instructions, _packet = build_normal_rewriter_prompt(
        Gate1Artifact.from_json(_effect_if_payload())
    )

    for view in (normal, masked):
        assert view["pi"]["iterations"][0]["induction_basis"] == (
            "rounded_extent_step"
        )
        assert not ({"lower", "upper", "step", "formula"} & set(
            view["pi"]["iterations"][0]
        ))
    assert "upper exactly ceil_div(trip extent, C tile) * C tile" in instructions
    assert "current shapes/values" in instructions
    assert "ceil_div(x0,c0)*c0" not in json.dumps(normal, sort_keys=True)


def test_rounded_extent_step_replay_check_requires_declared_extent_tile_formula() -> None:
    artifact = _tiled_loop_artifact(basis="rounded_extent_step")
    view = artifact.rewriter_view(masked=False)
    packet = {
        **view,
        "u_f": build_normal_rewriter_view(make_reduce_artifact())["u_f"],
    }
    valid = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, c0: tl.constexpr):
    rounded = tl.cdiv(p2, c0) * c0
    for offset in range(0, rounded, c0):
        pass
"""
    extent_only = valid.replace(
        "    rounded = tl.cdiv(p2, c0) * c0\n", ""
    ).replace("rounded", "p2")
    guessed_constant = valid.replace(
        "rounded = tl.cdiv(p2, c0) * c0", "rounded = 64"
    )

    accepted = replay_source_ingress(
        valid,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    )
    extent_result = replay_source_ingress(
        extent_only,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    )
    guessed_result = replay_source_ingress(
        guessed_constant,
        artifact=artifact,
        frontend=Frontend.TRITON,
        packet=packet,
    )

    assert accepted.status == "pass", accepted.to_check_response()
    for result in (extent_result, guessed_result):
        diagnostic = result.to_check_response()["diagnostics"][0]
        assert diagnostic["rule_code"] == "rounded_extent_step_contract"
        assert diagnostic["anonymous_handles"] == ["p2", "c0"]
