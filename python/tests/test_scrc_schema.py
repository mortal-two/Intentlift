from __future__ import annotations

from intentir.extraction.lifters.scrc_schema import (
    scrc_artifact_json_schema,
    scrc_artifact_text_format,
    scrc_lift_core_json_schema,
    scrc_lift_core_text_format,
)


def _parameter_schema() -> dict:
    schema = scrc_artifact_json_schema()
    return schema["properties"]["interface"]["properties"]["parameters"][
        "items"
    ]


def test_scrc_wire_schema_closes_gate1_objects() -> None:
    schema = scrc_artifact_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "j",
        "interface",
        "schedule",
        "strategy",
        "location_map",
    }
    assert _parameter_schema()["additionalProperties"] is False
    assert "referenced_dimension" not in _parameter_schema()["properties"]
    assert "tensor_ref" in _parameter_schema()["properties"]
    assert "strategy" in schema["properties"]
    assert "strategy" in schema["required"]
    assert "plan" not in schema["properties"]

    boundary = schema["properties"]["j"]["properties"]["boundary"]
    domain_text = boundary["properties"]["domains"]["additionalProperties"]
    access_predicate = boundary["properties"]["accesses"]["items"][
        "properties"
    ]["predicate"]["anyOf"][0]
    assert "boolean ContractExpr v1" in domain_text["description"]
    assert access_predicate == domain_text


def test_scrc_strategy_wire_uses_closed_override_only_pairs() -> None:
    strategy = scrc_artifact_json_schema()["properties"]["strategy"]
    assert strategy["additionalProperties"] is False
    variants = strategy["properties"]["bindings"]["items"]["oneOf"]
    pairs = {
        (
            variant["properties"]["family"]["enum"][0],
            variant["properties"]["choice"]["enum"][0],
        )
        for variant in variants
    }
    assert pairs == {
        ("program_coordinate_width", "wide"),
        ("address_association", "pointer_staged"),
        ("mask_materialization", "staged_reuse"),
        ("reduction_call_shape", "flattened"),
        ("reduction_tiling", "multi_tile_accumulate"),
        (
            "reduction_preheader_staging",
            "stage_invariant_access_scaffolding",
        ),
    }
    assert all(variant["additionalProperties"] is False for variant in variants)
    tiled = next(
        variant
        for variant in variants
        if variant["properties"]["family"]["enum"] == ["reduction_tiling"]
    )
    assert set(tiled["required"]) == {
        "target_ref",
        "family",
        "choice",
        "tile_slot",
        "accumulator_dtype",
    }
    assert tiled["properties"]["accumulator_dtype"]["enum"] == [
        "f16",
        "bf16",
        "f32",
        "f64",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
    ]


def test_scrc_wire_schema_preserves_open_intent_payloads() -> None:
    schema = scrc_artifact_json_schema()
    op = schema["$defs"]["intent_op"]
    intent_function = schema["properties"]["j"]["properties"]["intent"][
        "properties"
    ]["intent_function"]

    assert op["additionalProperties"] is False
    assert op["properties"]["attrs"] == {"type": "object"}
    assert intent_function["properties"]["layer_c"] == {
        "type": "object",
        "maxProperties": 0,
    }
    assert op["properties"]["body"]["items"] == {
        "$ref": "#/$defs/intent_op"
    }


def test_scrc_text_format_uses_schema_guidance_plus_local_validation() -> None:
    text_format = scrc_lift_core_text_format()

    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "scrc_lift_core"
    assert text_format["strict"] is False
    assert text_format["schema"] == scrc_lift_core_json_schema()
    assert scrc_artifact_json_schema() == scrc_lift_core_json_schema()
    assert scrc_artifact_text_format() == scrc_lift_core_text_format()


def test_scrc_schema_calls_do_not_share_mutable_nodes() -> None:
    first = scrc_artifact_json_schema()
    first["properties"]["j"]["properties"]["kernel_id"]["type"] = "null"

    second = scrc_artifact_json_schema()
    assert second["properties"]["j"]["properties"]["kernel_id"][
        "type"
    ] == "string"
