from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from intentir.extraction.lifters.scrc import build_scrc_lift_instructions
from intentir.extraction.lifters.scrc_schema import (
    render_semantic_archive_wire_grammar,
    scrc_lift_core_json_schema,
)
from intentir.ir.intent_op_contract import (
    IntentOpContractError,
    LOOP_ONLY_FIELDS,
    LOOP_REQUIRED_FIELDS,
    intent_op_json_schema,
    render_intent_op_wire_grammar,
    validate_intent_op_structure,
)
from intentir.verify.gate1.model import Gate1Artifact, SchemaError, _strict_intent

from direct_fixtures import make_copy_artifact, make_reduce_artifact


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


_OP_SCHEMA = {
    "$defs": {"intent_op": intent_op_json_schema()},
    "$ref": "#/$defs/intent_op",
}
Draft7Validator.check_schema(_OP_SCHEMA)
_OP_VALIDATOR = Draft7Validator(_OP_SCHEMA)


def _op_validator() -> Draft7Validator:
    return _OP_VALIDATOR


def _schema_accepts(value: dict) -> bool:
    return not list(_op_validator().iter_errors(value))


def _wire_accepts(value: dict) -> bool:
    """Model the actual RFC-8259 wire before applying its JSON Schema."""

    try:
        encoded = json.dumps(value, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return False
    return _schema_accepts(decoded)


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


def _walk_operations(operations: list[dict]):
    for operation in operations:
        yield operation
        body = operation.get("body")
        if isinstance(body, list):
            yield from _walk_operations(body)


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


def test_global_scope_storage_grammar_is_shared_without_kernel_rules() -> None:
    grammar = render_semantic_archive_wire_grammar()
    schema = scrc_lift_core_json_schema()
    assert schema["properties"]["j"]["description"] == grammar
    assert grammar in build_scrc_lift_instructions()
    assert "ordinary SSA temporaries" in grammar
    assert "storage array may be empty" in grammar
    assert "Swiglu" not in grammar


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
    deepest = _valid_loop(
        scope="q3",
        init="v_inner_iter",
        body_argument="v_deep_iter",
        next_value="v_deep_next",
        result_value="v_inner_next",
    )
    inner = _valid_loop(
        scope="q2",
        init="v_iter",
        body_argument="v_inner_iter",
        next_value="v_inner_next",
        result_value="v_next",
        body=[deepest],
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


@pytest.mark.parametrize(
    ("relation", "field", "bad_ref"),
    [
        ("routes", "source", "p0"),
        ("routes", "target", "c0"),
        ("order", "before", "n0"),
        ("order", "after", "p0"),
        ("sharing", "source", "c0"),
    ],
)
def test_pi_raw_anchor_endpoint_is_rejected_by_schema_and_parser(
    relation: str,
    field: str,
    bad_ref: str,
) -> None:
    payload = make_reduce_artifact().to_json()
    payload.pop("source")
    payload.pop("provenance")
    payload["pi"][relation][0][field] = bad_ref
    frozen = copy.deepcopy(payload)

    schema = scrc_lift_core_json_schema()
    errors = list(Draft7Validator(schema).iter_errors(payload))
    assert errors
    with pytest.raises(SchemaError, match="declared pi instance id"):
        Gate1Artifact.from_json(payload)
    assert payload == frozen


def test_pi_graph_instance_is_rejected_from_anchor_by_schema_and_parser() -> None:
    payload = make_reduce_artifact().to_json()
    payload.pop("source")
    payload.pop("provenance")
    payload["pi"]["instances"][0]["semantic_ref"] = "i0"

    schema = scrc_lift_core_json_schema()
    assert list(Draft7Validator(schema).iter_errors(payload))
    with pytest.raises(SchemaError, match="anonymous declaration-order handle"):
        Gate1Artifact.from_json(payload)


def test_pi_dangling_i_endpoint_passes_shape_but_fails_local_closure() -> None:
    payload = make_reduce_artifact().to_json()
    payload.pop("source")
    payload.pop("provenance")
    payload["pi"]["routes"][0]["source"] = "i99"

    pi_schema = scrc_lift_core_json_schema()["properties"]["pi"]
    assert not list(Draft7Validator(pi_schema).iter_errors(payload["pi"]))
    with pytest.raises(SchemaError, match="references undeclared pi instance 'i99'"):
        Gate1Artifact.from_json(payload)


@pytest.mark.parametrize(
    "artifact_factory", [make_copy_artifact, make_reduce_artifact]
)
def test_canonical_typed_fixture_ops_are_wire_complete_and_round_trip(
    artifact_factory,
) -> None:
    canonical = artifact_factory().j.intent.to_json()
    parsed = _strict_intent(canonical)
    assert parsed.to_json() == canonical
    assert _strict_intent(parsed.to_json()).to_json() == canonical
    operations = canonical["intent_function"]["body"]
    assert operations
    for operation in _walk_operations(operations):
        assert _wire_accepts(operation)
        assert _contract_accepts(operation)


def test_field_presence_projection_is_exhaustively_isomorphic() -> None:
    values = {
        "id": "n0",
        "op": "future_atomic_fusion",
        "result": "v0",
        "operands": ["t0"],
        "attrs": {"future": {"nested": [1, True, None]}},
        "type": {"dtype": "f32", "shape": ["N"]},
        "scope_id": "q1",
        "carry": [
            {
                "state_id": "s0",
                "init_value": 0.0,
                "body_argument": "v_iter",
                "next_value": "v_next",
                "result_value": "v_out",
                "type": {"dtype": "f32", "shape": []},
            }
        ],
        "trip": {"axis": "N"},
        "scan": [],
        "body": [{"op": "const", "result": "v_next"}],
        "yield": ["v_next"],
    }
    fields = tuple(values)
    for discriminator in (None, "future_atomic_fusion", "loop"):
        for present in itertools.chain.from_iterable(
            itertools.combinations(fields, count)
            for count in range(len(fields) + 1)
        ):
            operation = {field: copy.deepcopy(values[field]) for field in present}
            if "op" in operation:
                operation["op"] = discriminator
            if operation.get("op") == "loop" and "result" in operation:
                operation["result"] = ["v_out"]
            assert _schema_accepts(operation) == _contract_accepts(operation), (
                discriminator,
                present,
            )


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("op",), ""),
        (("id",), None),
        (("result",), ["v_out", "v_out"]),
        (("operands",), [""]),
        (("attrs",), []),
        (("type",), []),
        (("scope_id",), None),
        (("carry",), []),
        (("carry", 0, "state_id"), ""),
        (("carry", 0, "init_value"), {}),
        (("carry", 0, "type", "shape"), [True]),
        (("trip", "axis"), ""),
        (("scan",), [{}]),
        (("scan",), [{"name": "s", "from": "t0", "axis": True}]),
        (("scan",), [{"name": "s", "from": "t0", "axis": 1.5}]),
        (("body",), []),
        (("body", 0, "scope_id"), "q9"),
        (("yield",), []),
    ],
)
def test_type_and_nested_mutations_are_rejected_isomorphically(
    path: tuple[str | int, ...], invalid,
) -> None:
    operation = _valid_loop()
    cursor = operation
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = invalid
    assert not _schema_accepts(operation)
    assert not _contract_accepts(operation)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_are_rejected_at_the_actual_json_wire(invalid: float) -> None:
    operation = {"op": "const", "attrs": {"value": invalid}}
    assert not _wire_accepts(operation)
    assert not _contract_accepts(operation)


def test_json_schema_integral_float_is_accepted_and_canonicalized() -> None:
    operation = _valid_loop()
    operation["scan"] = [{"name": "v_scan", "from": "t0", "axis": 0.0}]
    operation["carry"][0]["type"]["shape"] = [1.0]
    operation["body"][0]["type"] = {"dtype": "f32", "shape": [1.0]}
    assert _schema_accepts(operation)
    assert _contract_accepts(operation)
    parsed = _strict_intent(_intent_with(operation))
    canonical = parsed.to_json()["intent_function"]["body"][0]
    assert canonical["scan"][0]["axis"] == 0
    assert canonical["carry"][0]["type"]["shape"] == [1]


def test_unknown_fields_are_rejected_at_every_recursive_op_level() -> None:
    for operation in (
        {"op": "future_op", "unknown": 1},
        {**_valid_loop(), "unknown": 1},
        _valid_loop(body=[{"op": "future_op", "unknown": 1}]),
    ):
        assert not _schema_accepts(operation)
        assert not _contract_accepts(operation)
