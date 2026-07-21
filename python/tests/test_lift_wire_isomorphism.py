from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from intentir.extraction.lifters.scrc import build_scrc_lift_instructions
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.ir.intent_op_contract import (
    IntentOpContractError,
    LOOP_ONLY_FIELDS,
    LOOP_REQUIRED_FIELDS,
    intent_op_json_schema,
    render_intent_op_wire_grammar,
    validate_intent_op_structure,
)
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


def _op_validator() -> Draft7Validator:
    schema = {
        "$defs": {"intent_op": intent_op_json_schema()},
        "$ref": "#/$defs/intent_op",
    }
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def _schema_accepts(value: dict) -> bool:
    return not list(_op_validator().iter_errors(value))


def _contract_accepts(value: dict) -> bool:
    try:
        validate_intent_op_structure(value)
    except IntentOpContractError:
        return False
    return True


def _valid_loop(
    *,
    scope: str = "q1",
    init: str | float = 0.0,
    body_argument: str = "v_iter",
    next_value: str = "v_next",
    result_value: str = "v_out",
    body: list[dict] | None = None,
) -> dict:
    return {
        "op": "loop",
        "result": [result_value],
        "scope_id": scope,
        "carry": [
            {
                "state_id": f"s_{scope}",
                "init_value": init,
                "body_argument": body_argument,
                "next_value": next_value,
                "result_value": result_value,
                "type": {"dtype": "f32", "shape": []},
            }
        ],
        "trip": {"axis": "N"},
        "scan": [],
        "body": body
        or [
            {
                "op": "map_unary",
                "result": next_value,
                "operands": [body_argument],
                "attrs": {"fn": "identity"},
                "type": {"dtype": "f32", "shape": []},
            }
        ],
        "yield": [next_value],
    }


def _intent_with(operation: dict) -> dict:
    return {
        "intent_function": {
            "name": "scrc_j0",
            "inputs": [{"name": "t0", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "t1", "dtype": "f32", "shape": []}],
            "returns": [{"output": "t1", "value": "v_out"}],
            "body": [operation],
        }
    }


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
    # The active shared contract now rejects this at the local IntentFunction
    # boundary, before cross-scope/archive validation.
    with pytest.raises(SchemaError, match="loop-only key.*scope_id"):
        _strict_intent(response["j"]["intent"])
    with pytest.raises(
        SchemaError, match="loop-only key.*scope_id"
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


def test_shared_contract_projects_schema_and_prompt_from_one_declaration() -> None:
    schema = intent_op_json_schema()
    assert set(schema) == {"anyOf"}
    assert len(schema["anyOf"]) == 2
    grammar = render_intent_op_wire_grammar()
    assert "op is exactly 'loop'" in grammar
    assert ", ".join(sorted(LOOP_ONLY_FIELDS)) in grammar
    assert grammar in build_scrc_lift_instructions()


@pytest.mark.parametrize("field", sorted(LOOP_ONLY_FIELDS))
def test_nonloop_rejects_every_loop_only_field_in_schema_and_parser(field: str) -> None:
    values = {
        "scope_id": "q0",
        "carry": [],
        "trip": {"axis": "N"},
        "scan": [],
        "body": [],
        "yield": [],
    }
    operation = {"op": "future_atomic_fusion", field: values[field]}
    assert not _schema_accepts(operation)
    assert not _contract_accepts(operation)
    with pytest.raises(SchemaError, match="loop-only key"):
        _strict_intent(
            {
                "intent_function": {
                    "name": "scrc_j0",
                    "inputs": [],
                    "outputs": [],
                    "body": [operation],
                }
            }
        )


@pytest.mark.parametrize("field", sorted(LOOP_REQUIRED_FIELDS))
def test_loop_requires_every_structural_field_in_schema_and_parser(field: str) -> None:
    operation = _valid_loop()
    operation.pop(field)
    assert not _schema_accepts(operation)
    assert not _contract_accepts(operation)
    with pytest.raises(SchemaError, match="missing required key"):
        _strict_intent(_intent_with(operation))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope_id", None),
        ("result", "v_out"),
        ("result", []),
        ("carry", []),
        ("trip", None),
        ("body", []),
        ("yield", []),
    ],
)
def test_loop_null_and_empty_shapes_are_rejected_equally(field: str, value) -> None:
    operation = _valid_loop()
    operation[field] = value
    assert not _schema_accepts(operation)
    assert not _contract_accepts(operation)
    with pytest.raises(SchemaError):
        _strict_intent(_intent_with(operation))


def test_nested_loop_and_canonical_ids_are_schema_parser_complete() -> None:
    inner = _valid_loop(
        scope="q2",
        init="v_iter",
        body_argument="v_inner_iter",
        next_value="v_inner_next",
        result_value="v_next",
    )
    outer = _valid_loop(body=[inner])
    outer["id"] = "n0"
    assert _schema_accepts(outer)
    assert _contract_accepts(outer)
    parsed = _strict_intent(_intent_with(outer))
    canonical = parsed.to_json()["intent_function"]["body"][0]
    assert canonical["id"] == "n0"
    assert _schema_accepts(canonical)
    assert _contract_accepts(canonical)


def test_new_wire_schema_rejects_the_frozen_v2_response_before_full_parser() -> None:
    _request, response = _evidence()
    schema = scrc_lift_core_json_schema()
    Draft7Validator.check_schema(schema)
    errors = list(Draft7Validator(schema).iter_errors(response))
    assert errors
    for operation in response["j"]["intent"]["intent_function"]["body"]:
        assert not _schema_accepts(operation)
