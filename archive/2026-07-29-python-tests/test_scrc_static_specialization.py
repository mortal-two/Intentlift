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


def _copy_anonymous_view(
    *,
    x_contract: tuple[int | str, ...] | None = ("h0",),
    y_contract: tuple[int | str, ...] | None = ("h0",),
    scalar_coordinate: str | None = "h0",
) -> dict:
    view = scrc.SCRCCompilerView(
        schedule_values={"BLOCK": 1024},
        arguments=(
            scrc.SCRCCompilerArgument(
                "X",
                "tensor",
                "f32",
                (8,),
                mode="input",
                shape_contract=x_contract,
            ),
            scrc.SCRCCompilerArgument(
                "Y",
                "tensor",
                "f32",
                (8,),
                mode="output",
                shape_contract=y_contract,
            ),
            scrc.SCRCCompilerArgument(
                "N",
                "scalar",
                "i32",
                value=8,
                mode="value",
                shape_coordinate=scalar_coordinate,
            ),
        ),
        parameter_order=("X", "Y", "N", "BLOCK"),
        constexpr_parameters=("BLOCK",),
    )
    return scrc._anonymous_compiler_view(
        view,
        {"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"},
    )


def _constexpr_copy_anonymous_view() -> dict:
    view = scrc.SCRCCompilerView(
        schedule_values={"BLOCK": 1024},
        bound_constexpr_values={"N": 8},
        bound_constexpr_shape_coordinates={"N": "h0"},
        arguments=(
            scrc.SCRCCompilerArgument(
                "X", "tensor", "f32", (8,), mode="input",
                shape_contract=("h0",),
            ),
            scrc.SCRCCompilerArgument(
                "Y", "tensor", "f32", (8,), mode="output",
                shape_contract=("h0",),
            ),
        ),
        parameter_order=("X", "Y", "N", "BLOCK"),
        constexpr_parameters=("N", "BLOCK"),
    )
    return scrc._anonymous_compiler_view(
        view,
        {"X": "p0", "Y": "p1", "N": "p2", "BLOCK": "c0"},
    )


def test_compiler_shape_coordinate_requires_integer_dtype_and_explicit_grid_binding() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        scrc.SCRCCompilerArgument(
            "N", "scalar", "f32", value=8, shape_coordinate="h0"
        )
    with pytest.raises(ValueError, match="concrete compiler grid"):
        scrc.SCRCCompilerView(
            schedule_values={}, grid=(), grid_contract=("rows",)
        )

    argument = scrc.SCRCCompilerArgument(
        "X",
        "tensor",
        "f32",
        (2,),
        mode="input",
        shape_contract=("h0",),
    )
    prose_only = scrc.SCRCCompilerView(
        schedule_values={},
        arguments=(argument,),
        grid=(2,),
        grid_contract=("human_rows_label",),
    )
    assert prose_only.grid_shape_coordinates == ()
    with pytest.raises(ValueError, match="grid extent differs"):
        scrc.SCRCCompilerView(
            schedule_values={},
            arguments=(argument,),
            grid=(3,),
            grid_contract=("human_rows_label",),
            grid_shape_coordinates=("h0",),
        )

    with pytest.raises(ValueError, match="requires a bound value"):
        scrc.SCRCCompilerView(
            schedule_values={},
            bound_constexpr_shape_coordinates={"N": "h0"},
            arguments=(argument,),
        )
    with pytest.raises(ValueError, match="valuations differ"):
        scrc.SCRCCompilerView(
            schedule_values={},
            bound_constexpr_values={"N": 3},
            bound_constexpr_shape_coordinates={"N": "h0"},
            arguments=(argument,),
            parameter_order=("X", "N"),
            constexpr_parameters=("N",),
        )


def test_shape_coordinate_contract_rejects_literal_specialization_and_collapse() -> None:
    artifact = make_copy_artifact()
    scrc._validate_compiler_view(artifact, _copy_anonymous_view())

    specialized = artifact.to_json()
    specialized["j"]["intent"]["intent_function"]["inputs"][0]["shape"] = [8]
    with pytest.raises(ValueError, match="specialized source-grounded"):
        scrc._validate_compiler_view(
            Gate1Artifact.from_json(specialized), _copy_anonymous_view()
        )

    with pytest.raises(ValueError, match="collapsed distinct"):
        scrc._validate_compiler_view(
            artifact,
            _copy_anonymous_view(y_contract=("h1",)),
        )

    with pytest.raises(ValueError, match="Harness-declared static shape"):
        scrc._validate_compiler_view(
            artifact,
            _copy_anonymous_view(
                x_contract=(8,),
                y_contract=(8,),
                scalar_coordinate=None,
            ),
        )


def test_scalar_shape_coordinate_allows_direct_x_or_authored_value_equality() -> None:
    direct = make_copy_artifact()
    scrc._validate_compiler_view(direct, _copy_anonymous_view())

    related = direct.to_json()
    related["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "t2", "dtype": "i32", "shape": []}
    )
    related["j"]["signature"]["parameters"][2]["logical_ref"] = "t2"
    related["j"]["signature"]["parameters"][2]["rank"] = 0
    related["j"]["value_dimension_relations"] = [
        {"value_ref": "t2", "dimension_ref": "x0", "relation": "eq"}
    ]
    scrc._validate_compiler_view(
        Gate1Artifact.from_json(related), _copy_anonymous_view()
    )

    missing_relation = copy.deepcopy(related)
    missing_relation["j"]["value_dimension_relations"] = []
    with pytest.raises(ValueError, match="scalar-to-shape-coordinate equality"):
        scrc._validate_compiler_view(
            Gate1Artifact.from_json(missing_relation), _copy_anonymous_view()
        )


def test_bound_constexpr_shape_coordinate_requires_direct_x_or_authored_equality() -> None:
    direct_payload = make_copy_artifact().to_json()
    direct_payload["j"]["signature"]["parameters"][2]["kind"] = "constexpr"
    direct = Gate1Artifact.from_json(direct_payload)
    view = _constexpr_copy_anonymous_view()

    assert view["bound_constexpr_shape_coordinates"] == {"p2": "h0"}
    scrc._validate_compiler_view(direct, view)

    related = direct.to_json()
    related["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "t2", "dtype": "i32", "shape": []}
    )
    related["j"]["signature"]["parameters"][2]["logical_ref"] = "t2"
    related["j"]["signature"]["parameters"][2]["rank"] = 0
    related["j"]["value_dimension_relations"] = [
        {"value_ref": "t2", "dimension_ref": "x0", "relation": "eq"}
    ]
    scrc._validate_compiler_view(Gate1Artifact.from_json(related), view)

    related["j"]["value_dimension_relations"] = []
    with pytest.raises(
        ValueError,
        match="bound-constexpr-to-shape-coordinate equality",
    ):
        scrc._validate_compiler_view(Gate1Artifact.from_json(related), view)


def test_string_specialization_carrier_is_v7_and_typechecks() -> None:
    payload = _string_specialization_payload()
    assert Gate1Artifact.SCHEMA == "intentir.scrc.artifact.v10"
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
