"""Harness-owned compile-time specialization carrier tests."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from direct_fixtures import make_copy_artifact
from intentir.extraction.lifters import scrc
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.prompt import build_rewriter_payload
from intentir.verify.gate1.replay import (
    build_source_blind_replay_compile_case,
    replay_source_ingress,
)
from intentir.verify.typecheck import type_check


def _string_specialization_payload() -> dict:
    payload = make_copy_artifact().to_json()
    payload["j"]["scopes"] = [
        {
            "scope_id": "q0",
            "kind": "function",
            "parent_scope": None,
            "captures": [],
            "definitions": ["t0", "t1"],
            "effects": [],
        }
    ]
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "string",
            "rank": 0,
        }
    )
    return payload


def _anonymous_view(value: str = "none") -> dict:
    view = scrc.SCRCCompilerView(
        schedule_values={"BLOCK": 1024},
        bound_constexpr_values={"BIAS_TYPE": value},
        arguments=(
            scrc.SCRCCompilerArgument(
                "X", "tensor", "f32", (1000,), mode="input"
            ),
            scrc.SCRCCompilerArgument(
                "Y", "tensor", "f32", (1000,), mode="output"
            ),
            scrc.SCRCCompilerArgument(
                "N", "scalar", "i32", value=1000, mode="value"
            ),
        ),
        parameter_order=("X", "Y", "N", "BIAS_TYPE", "BLOCK"),
        constexpr_parameters=("BIAS_TYPE", "BLOCK"),
    )
    return scrc._anonymous_compiler_view(
        view,
        {
            "X": "p0",
            "Y": "p1",
            "N": "p2",
            "BIAS_TYPE": "p3",
            "BLOCK": "c0",
        },
    )


def test_string_specialization_carrier_is_v7_and_typechecks() -> None:
    payload = _string_specialization_payload()
    assert Gate1Artifact.SCHEMA == "intentir.scrc.artifact.v7"
    assert "semantic_bindings" not in payload["j"]
    Draft202012Validator(scrc_lift_core_json_schema()).validate(payload)

    artifact = Gate1Artifact.from_json(payload)
    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, [item.to_json() for item in report.diagnostics]
    assert artifact.to_json() == payload


def test_specialization_classification_is_llm_authored_and_harness_checked() -> None:
    artifact = Gate1Artifact.from_json(_string_specialization_payload())
    scrc._validate_compiler_view(artifact, _anonymous_view())

    fake_tensor = copy.deepcopy(_string_specialization_payload())
    fake_tensor["j"]["signature"]["parameters"][-1]["dtype"] = "i1"
    fake_tensor["j"]["signature"]["parameters"][-1]["logical_ref"] = "t2"
    fake_tensor["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "t2", "dtype": "i1", "shape": []}
    )
    fake_tensor["j"]["scopes"][0]["definitions"].append("t2")
    with pytest.raises(ValueError, match="string specialization carrier"):
        scrc._validate_compiler_view(
            Gate1Artifact.from_json(fake_tensor), _anonymous_view()
        )


def test_specialization_value_never_enters_j_c_pi_or_replay_packet() -> None:
    artifact = Gate1Artifact.from_json(_string_specialization_payload())
    packet = build_rewriter_payload(artifact)

    parameter = packet["j"]["signature"]["parameters"][-1]
    assert parameter == {
        "name": "p3",
        "kind": "constexpr",
        "mode": "value",
        "dtype": "string",
        "rank": 0,
    }
    assert "none" not in repr(packet)
    assert "semantic_bindings" not in packet["j"]

    compile_case = build_source_blind_replay_compile_case(artifact)
    assert compile_case.schedule["p3"] == "synthetic"


def test_pure_implementation_constexpr_cannot_replace_bound_carrier() -> None:
    payload = _string_specialization_payload()
    payload["j"]["signature"]["parameters"].pop()
    artifact = Gate1Artifact.from_json(payload)

    with pytest.raises(ValueError, match="Harness-declared logical/C handles"):
        scrc._validate_compiler_view(artifact, _anonymous_view())


def test_replay_must_preserve_but_not_read_specialization_carrier() -> None:
    artifact = Gate1Artifact.from_json(_string_specialization_payload())
    prefix = """import triton
import triton.language as tl

@triton.jit
def scrc_k0(p0, p1, p2, p3: tl.constexpr, c0: tl.constexpr):
"""
    accepted = replay_source_ingress(
        prefix + "    return\n",
        artifact=artifact,
        frontend=Frontend.TRITON,
    )
    rejected = replay_source_ingress(
        prefix + "    copied_selector = p3\n",
        artifact=artifact,
        frontend=Frontend.TRITON,
    )

    assert accepted.status == "pass"
    assert rejected.to_check_response()["diagnostics"] == [
        {
            "rule_code": "specialization_carrier_use_forbidden",
            "path": "<parameter:p3>",
            "anonymous_handles": ["p3"],
            "line": 6,
        }
    ]
