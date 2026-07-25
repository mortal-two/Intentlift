from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from direct_fixtures import make_reduce_artifact
from intentir.extraction.lifters.scrc import (
    SCRC_DECODE_PROTOCOL,
    SCRCHarnessBinding,
    SCRCLiftRequest,
    decode_scrc_lift_artifact,
)
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import Gate1Artifact, SchemaError
from intentir.verify.gate1.prompt import (
    build_masked_rewriter_view,
    build_normal_rewriter_view,
)


def _guarded_payload() -> dict:
    payload = make_reduce_artifact().to_json()

    def rename_output(value):
        if isinstance(value, str):
            return "t2" if value == "t1" else value
        if isinstance(value, list):
            return [rename_output(item) for item in value]
        if isinstance(value, dict):
            return {key: rename_output(item) for key, item in value.items()}
        return value

    payload = rename_output(payload)
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "t1", "dtype": "i1", "shape": ["x0"]}
    )
    payload["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "pointer",
            "mode": "input",
            "dtype": "i1",
            "rank": 1,
            "logical_ref": "t1",
        }
    )
    payload["j"]["boundary"]["access_guard_relations"] = [
        {
            "access_ref": "a0",
            "value_ref": "t1",
            "axes": [1],
            "polarity": "positive",
        }
    ]
    return payload


def _request() -> SCRCLiftRequest:
    return SCRCLiftRequest(
        instructions="",
        user_input="",
        prompt_sha256="0" * 64,
        harness_binding=SCRCHarnessBinding(
            source_to_opaque=(
                ("arg0", "p0"),
                ("arg1", "p1"),
                ("arg2", "p2"),
                ("arg3", "p3"),
            ),
            source_function="source_kernel",
        ),
    )


def test_v6_preserves_v5_access_guard_round_trip_and_decode_exactness() -> None:
    payload = _guarded_payload()
    assert payload["schema"] == "intentir.scrc.artifact.v7"
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.to_json() == payload
    assert SCRC_DECODE_PROTOCOL == "intentir.scrc.decode.v8"
    decoded = decode_scrc_lift_artifact(json.dumps(payload), _request())
    assert decoded.wire_schema == "intentir.scrc.artifact.v7"
    assert decoded.j.boundary.access_guard_relations[0].to_json() == {
        "access_ref": "a0",
        "value_ref": "t1",
        "axes": [1],
        "polarity": "positive",
    }


def test_v5_access_guard_artifact_remains_field_exact_but_is_legacy_for_lift() -> None:
    payload = _guarded_payload()
    payload["schema"] = "intentir.scrc.artifact.v5"
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.to_json() == payload
    with pytest.raises(ValueError, match="this round requires"):
        decode_scrc_lift_artifact(json.dumps(payload), _request())


def test_v5_static_schema_requires_closed_access_guard_shape() -> None:
    schema = scrc_lift_core_json_schema()
    boundary = schema["properties"]["j"]["properties"]["boundary"]
    assert "access_guard_relations" in boundary["required"]
    relation = boundary["properties"]["access_guard_relations"]["items"]
    validator = Draft202012Validator(relation)
    valid = _guarded_payload()["j"]["boundary"]["access_guard_relations"][0]
    assert not list(validator.iter_errors(valid))

    for field in ("access_ref", "value_ref", "axes", "polarity"):
        missing = copy.deepcopy(valid)
        del missing[field]
        assert list(validator.iter_errors(missing))

    bad_polarity = {**valid, "polarity": "unknown"}
    assert list(validator.iter_errors(bad_polarity))
    bad_axes = {**valid, "axes": ["0"]}
    assert list(validator.iter_errors(bad_axes))
    unknown = {**valid, "extra": True}
    assert list(validator.iter_errors(unknown))


def test_v5_model_rejects_non_integer_axes_and_unknown_refs() -> None:
    bad_axis = _guarded_payload()
    bad_axis["j"]["boundary"]["access_guard_relations"][0]["axes"] = [False]
    with pytest.raises(SchemaError, match=r"axes\[0\].*integer"):
        Gate1Artifact.from_json(bad_axis)

    unknown_access = _guarded_payload()
    unknown_access["j"]["boundary"]["access_guard_relations"][0][
        "access_ref"
    ] = "a99"
    with pytest.raises(SchemaError, match="unknown access"):
        Gate1Artifact.from_json(unknown_access)

    unknown_value = _guarded_payload()
    unknown_value["j"]["boundary"]["access_guard_relations"][0][
        "value_ref"
    ] = "v99"
    with pytest.raises(SchemaError, match="unknown Layer-A value"):
        Gate1Artifact.from_json(unknown_value)


def test_v4_round_trip_remains_field_exact_and_cannot_claim_v5_relation() -> None:
    payload = make_reduce_artifact().to_json()
    payload["schema"] = "intentir.scrc.artifact.v4"
    del payload["j"]["boundary"]["access_guard_relations"]
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.to_json() == payload
    assert "access_guard_relations" not in parsed.to_json()["j"]["boundary"]

    mislabeled = copy.deepcopy(payload)
    mislabeled["j"]["boundary"]["access_guard_relations"] = []
    with pytest.raises(
        SchemaError, match="v1-v4 do not contain J access_guard_relations"
    ):
        Gate1Artifact.from_json(mislabeled)

    with pytest.raises(ValueError, match="this round requires"):
        decode_scrc_lift_artifact(json.dumps(payload), _request())


def test_normal_replay_packet_preserves_public_access_guard_relation() -> None:
    artifact = Gate1Artifact.from_json(_guarded_payload())
    normal = build_normal_rewriter_view(artifact)
    assert normal["j"]["boundary"]["access_guard_relations"] == [
        {
            "access_ref": "a0",
            "value_ref": "t1",
            "axes": [1],
            "polarity": "positive",
        }
    ]
    assert "values" not in normal["c"]

    masked = build_masked_rewriter_view(artifact)
    assert masked["j"]["boundary"]["access_guard_relations"] == [
        {
            "access_ref": "a0",
            "value_ref": "t1",
                "axes": ["<masked>"],
                "polarity": "<masked>",
            }
    ]
