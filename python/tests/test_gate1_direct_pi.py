from __future__ import annotations

import copy
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.extraction.lifters import scrc
from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.extraction.providers.client import LLMReply
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import Gate1Artifact, ScheduleSlot, SchemaError
from intentir.verify.gate1.pi import RealizationWitness, RealizationWitnessError
from intentir.verify.gate1.prompt import (
    assert_source_isolated_payload,
    build_masked_rewriter_view,
    build_normal_rewriter_view,
)


def test_direct_pi_artifact_roundtrip_and_epoch_break() -> None:
    artifact = make_reduce_artifact()
    encoded = artifact.to_json()
    assert Gate1Artifact.from_json(encoded) == artifact
    assert encoded["schema"] == Gate1Artifact.SCHEMA

    legacy = copy.deepcopy(encoded)
    legacy.pop("schema")
    legacy["plan"] = {"steps": []}
    with pytest.raises(SchemaError):
        Gate1Artifact.from_json(legacy)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"operator": "mul"},
        {"predicate": "p0 < p1"},
        {"value": 0},
        {"api": "tl.load"},
        {"text": "emit source here"},
    ],
)
def test_pi_rejects_semantic_atoms_and_source_text(forbidden: dict) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["instances"][0].update(forbidden)
    with pytest.raises(RealizationWitnessError, match="forbidden fields"):
        RealizationWitness.from_json(payload)


def test_pi_rejects_unknown_semantic_reference_and_cycles() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = "n99"
    with pytest.raises(SchemaError, match="unknown J/C"):
        Gate1Artifact.from_json(payload)

    pi = artifact.pi.to_json()
    pi["order"].append({"before": "i3", "after": "i0"})
    with pytest.raises(RealizationWitnessError, match="acyclic"):
        RealizationWitness.from_json(pi)


@pytest.mark.parametrize("semantic_ref", ["v0", "x0", "c0", "p0"])
def test_pi_may_reference_any_declared_anonymous_j_c_or_harness_handle(
    semantic_ref: str,
) -> None:
    payload = make_reduce_artifact().to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = semantic_ref
    assert Gate1Artifact.from_json(payload).pi.instances[0]["semantic_ref"] == semantic_ref


def test_masked_j_changes_only_j() -> None:
    artifact = make_reduce_artifact()
    normal = build_normal_rewriter_view(artifact)
    masked = build_masked_rewriter_view(artifact)
    assert normal["j"] != masked["j"]
    for key in ("c", "pi", "harness", "frontend_context"):
        assert normal[key] == masked[key]
    assert normal["mode"] == "normal"
    assert masked["mode"] == "masked_j"


def test_replay_packet_is_source_isolated() -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    encoded = json.dumps(payload, sort_keys=True)
    assert "PRIVATE_SOURCE_CANARY" not in encoded
    assert "location_map" not in encoded
    assert "provenance" not in encoded
    assert_source_isolated_payload(payload)


def test_lift_schema_requires_direct_pi_and_no_embedded_c_channel() -> None:
    schema = scrc_lift_core_json_schema()
    properties = schema["properties"]
    assert set(properties) == {
        "schema", "j", "interface", "schedule", "pi", "location_map"
    }
    function_properties = (
        properties["j"]["properties"]["intent"]["properties"]
        ["intent_function"]["properties"]
    )
    assert "layer_c" not in function_properties


def test_c_role_rejects_semantic_or_realization_payload() -> None:
    with pytest.raises(SchemaError, match="pure scheduling"):
        ScheduleSlot.from_json(
            {
                "name": "c0",
                "kind": "constexpr",
                "role": "reduction_" + "mask_placement",
            }
        )


def test_source_is_anonymized_before_lift() -> None:
    source = """@triton.jit
def layer_norm_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    # private X and BLOCK_SIZE names must not survive in comments
    mask = N > 0
    centered = tl.load(X, mask=mask)
    tl.store(Y, centered)
"""
    anonymous, binding, _mapping = anonymize_source(source)
    assert "layer_norm_kernel" not in anonymous
    assert "BLOCK_SIZE" not in anonymous
    assert "centered" not in anonymous
    assert "def scrc_k0(p0, p1, p2, c0: tl.constexpr)" in anonymous
    assert "tl.load(p0, mask=v0)" in anonymous
    assert "private X" not in anonymous
    assert dict(binding.source_to_opaque)["X"] == "p0"
    assert dict(binding.source_to_opaque)["BLOCK_SIZE"] == "c0"


def test_lift_llm_directly_returns_complete_pi_in_one_call(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
    response.pop("source")
    response.pop("provenance")
    calls = []

    def fake_call(instructions, user, **kwargs):
        calls.append((instructions, user, kwargs))
        assert "private_reduce" not in user
        assert "BLOCK_SIZE" not in user
        assert "anonymous_source" in user
        return LLMReply(
            json.dumps(response),
            "test-model",
            {"id": "resp0", "model": "test-model", "usage": {"total_tokens": 1}},
        )

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    source = """@triton.jit
def private_reduce(X, Y, N, BLOCK_SIZE: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(source)
    assert extraction.artifact.pi == artifact.pi
    assert extraction.artifact.source == source
    assert len(calls) == 1
    assert extraction.provenance.response_id == "resp0"


def test_lift_preserves_trusted_c_and_public_compiler_role(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
    response.pop("source")
    response.pop("provenance")
    response["interface"]["compiler_slots"] = ["c1"]
    response["schedule"]["slots"].append(
        {
            "name": "c1",
            "kind": "compiler",
            "role": "num_warps",
            "value_type": "int",
            "required": True,
        }
    )
    response["schedule"]["values"]["c1"] = 4

    def fake_call(instructions, user, **kwargs):
        del instructions, kwargs
        assert "BLOCK_SIZE" not in user
        assert '"compiler_roles": {"c1": "num_warps"}' in user
        return LLMReply(json.dumps(response), "test-model", {"id": "resp1"})

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    compiler_view = SCRCCompilerView(
        schedule_values={"BLOCK_SIZE": 128, "num_warps": 4},
        compiler_slots=("num_warps",),
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (2, 4)),
            SCRCCompilerArgument("Y", "tensor", "f32", (2,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=4),
        ),
        grid=(1,),
        grid_contract=("ceil_div(N, BLOCK_SIZE)",),
        parameter_order=("X", "Y", "N", "BLOCK_SIZE"),
        constexpr_parameters=("BLOCK_SIZE",),
    )
    source = """@triton.jit
def private_reduce(X, Y, N, BLOCK_SIZE: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(source, compiler_view=compiler_view)
    assert extraction.artifact.schedule.values == {"c0": 128, "c1": 4}
    assert extraction.artifact.schedule.slots[1].role == "num_warps"
