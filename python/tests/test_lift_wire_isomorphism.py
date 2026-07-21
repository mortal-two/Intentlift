from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from intentir.verify.gate1.model import Gate1Artifact, SchemaError, _strict_intent


_ROOT = Path(__file__).resolve().parents[2]
_CASE = (
    _ROOT
    / "artifacts"
    / "real_llm_batch_28_v2"
    / "recovery"
    / "liger_swiglu_forward"
)


def _evidence() -> tuple[dict, dict]:
    request = json.loads((_CASE / "calls/01_lift/request.json").read_text())
    response = json.loads((_CASE / "calls/01_lift/response.txt").read_text())
    return request, response


def test_v2_fixture_exposes_every_nonloop_scope_mismatch() -> None:
    request, response = _evidence()
    intent_op_schema = request["text"]["format"]["schema"]["$defs"]["intent_op"]
    assert intent_op_schema["required"] == ["op"]
    assert "scope_id" in intent_op_schema["properties"]

    operations = response["j"]["intent"]["intent_function"]["body"]
    assert [(index, item["op"], item.get("scope_id")) for index, item in enumerate(operations)] == [
        (0, "cast", "q0"),
        (1, "map_unary", "q0"),
        (2, "map_binary", "q0"),
        (3, "cast", "q0"),
        (4, "map_binary", "q0"),
    ]
    # Local IntentFunction shape checking admitted the response; the complete
    # semantic archive rejected the cross-scope ownership contract.
    _strict_intent(response["j"]["intent"])
    with pytest.raises(
        SchemaError, match="non-loop operation 'body\\[0\\]' must not declare scope_id"
    ):
        Gate1Artifact.from_json(response)


def test_v2_fixture_separates_structural_and_semantic_rejections() -> None:
    _request, response = _evidence()
    diagnostic = copy.deepcopy(response)
    for operation in diagnostic["j"]["intent"]["intent_function"]["body"]:
        operation.pop("scope_id", None)
    with pytest.raises(SchemaError, match="definitions must exactly match"):
        Gate1Artifact.from_json(diagnostic)

    diagnostic["j"]["scopes"][0]["definitions"].remove("t2")
    with pytest.raises(SchemaError, match="write_once storage 'b3'"):
        Gate1Artifact.from_json(diagnostic)

    diagnostic["j"]["storage"] = [
        item
        for item in diagnostic["j"]["storage"]
        if item["storage_id"] in {"b0", "b1", "b2"}
    ]
    assert Gate1Artifact.from_json(diagnostic).j.storage


def test_v2_schema_has_latent_loop_variant_drift() -> None:
    request, _response = _evidence()
    schema = request["text"]["format"]["schema"]["$defs"]["intent_op"]
    assert set(schema["properties"]) >= {
        "scope_id",
        "carry",
        "trip",
        "scan",
        "body",
        "yield",
    }
    assert schema["required"] == ["op"]
    assert "id" not in schema["properties"]
