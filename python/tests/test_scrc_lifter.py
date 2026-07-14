from __future__ import annotations

import json

import pytest

from intentir.extraction.providers.client import LLMReply
from intentir.extraction.lifters import scrc
from intentir.verify.gate1.rules import default_triton_rule_registry


def _artifact_payload() -> dict:
    return {
        "j": {
            "kernel_id": "add_kernel",
            "intent": {
                "intent_function": {
                    "name": "add",
                    "inputs": [
                        {"name": "A", "dtype": "f32", "shape": ["N"]},
                        {"name": "B", "dtype": "f32", "shape": ["N"]},
                    ],
                    "outputs": [
                        {"name": "Y", "dtype": "f32", "shape": ["N"]}
                    ],
                    "body": [
                        {
                            "op": "map_binary",
                            "result": "Y",
                            "operands": ["A", "B"],
                            "attrs": {"fn": "add"},
                        }
                    ],
                }
            },
            "node_ids": {"n0": "body[0]"},
            "boundary": {
                "domains": {"d0": "0 <= i < N"},
                "accesses": [
                    {
                        "access_id": "a0",
                        "kind": "load",
                        "tensor": "A",
                        "predicate": "i < N",
                        "fill_value": 0.0,
                        "node_ref": "n0",
                        "index_expression": "i",
                    },
                    {
                        "access_id": "a1",
                        "kind": "load",
                        "tensor": "B",
                        "predicate": "i < N",
                        "fill_value": 0.0,
                        "node_ref": "n0",
                        "index_expression": "i",
                    },
                    {
                        "access_id": "a2",
                        "kind": "store",
                        "tensor": "Y",
                        "predicate": "i < N",
                        "node_ref": "n0",
                        "index_expression": "i",
                    },
                ],
                "effects": [
                    {"kind": "read", "target": "A", "node_ref": "n0"},
                    {"kind": "read", "target": "B", "node_ref": "n0"},
                    {"kind": "write", "target": "Y", "node_ref": "n0"},
                ],
                "output_mappings": [
                    {"output": "Y", "value": "Y", "node_ref": "n0"}
                ],
            },
        },
        "interface": {
            "kernel_id": "add_kernel",
            "frontend": "triton",
            "function_name": "add_kernel",
            "parameters": [
                {"name": "a_ptr", "kind": "pointer", "mode": "input", "dtype": "f32", "rank": 1, "tensor_ref": "A"},
                {"name": "b_ptr", "kind": "pointer", "mode": "input", "dtype": "f32", "rank": 1, "tensor_ref": "B"},
                {"name": "out_ptr", "kind": "pointer", "mode": "output", "dtype": "f32", "rank": 1, "tensor_ref": "Y"},
                {"name": "N", "kind": "scalar", "mode": "value", "dtype": "i32", "rank": 0},
                {"name": "BLOCK", "kind": "constexpr", "mode": "value", "dtype": "i32", "rank": 0},
            ],
            "grid": ["ceil_div(N, BLOCK)"],
            "compiler_slots": ["num_warps"],
        },
        "schedule": {
            "slots": [
                {"name": "BLOCK", "kind": "constexpr"},
                {"name": "num_warps", "kind": "compiler"},
            ],
            "values": {"BLOCK": 64, "num_warps": 4},
        },
        "plan": {
            "steps": [
                {"rule": "bind_program_domain", "refs": ["d0"]},
                {"rule": "materialize_index_tile", "refs": ["a0"]},
                {"rule": "realize_masked_access", "refs": ["a0", "a1"]},
                {"rule": "realize_elementwise", "node_ref": "n0"},
                {"rule": "realize_masked_access", "refs": ["a2"]},
            ]
        },
        "location_map": {
            "source_name": "add_kernel.py",
            "entries": [
                {
                    "span": {"start_line": 1, "end_line": 12},
                    "node_refs": ["n0"],
                    "access_refs": ["a0", "a1", "a2"],
                }
            ]
        },
    }


def test_scrc_lift_prompt_uses_rule_ids_without_frontend_api_spellings() -> None:
    prompt = scrc.build_scrc_lift_instructions(default_triton_rule_registry())
    assert "realize_reduce" in prompt
    assert "COLLAPSED" in prompt
    assert "at least one of" in prompt
    assert "global [M,N] row reduce" in prompt
    assert "tile_axis=0" in prompt
    assert "copy `col < N`" in prompt
    assert "materialize_pointer_tile" in prompt
    assert "staged pointer additions" in prompt
    assert "load(ptr + row*N + lane)" in prompt
    assert "offset = row*N + lane" in prompt
    assert "semantic index_expression string alone" in prompt
    assert "runtime scalar consumed by the computation" in prompt
    assert "shape=[]" in prompt
    assert "canonicalize reciprocal square root" in prompt
    assert "semantic extent" in prompt
    assert "global J axis" in prompt
    assert "re-roll a schedule-only tiled-K accumulator loop" in prompt
    assert "mk,kn->mn" in prompt
    assert "does not create" in prompt
    assert "boundary.reductions" in prompt
    assert "coordinate components are not a" in prompt
    assert "scalar-language `and` or `&&`" in prompt
    assert "Do not also select realize_reduce" in prompt
    assert "bind_wide_program_domain" in prompt
    assert "flag_gems.utils.triton_lang_extension as tle" in prompt
    assert "`get_dtype_min` returns negative" in prompt
    assert "`get_dtype_max` returns positive" in prompt
    assert "do not substitute the largest finite value" in prompt
    assert "tle.program_id" in prompt
    assert "select realize_flattened_reduce" in prompt
    assert "exactly one of realize_reduce/realize_flattened_reduce" in prompt
    assert "m,n->mn" in prompt
    assert "selects realize_outer_product" in prompt
    assert "select materialize_mask_tile" in prompt
    assert "mutually exclusive" in prompt
    assert "tl.load" not in prompt
    assert "tl.sum" not in prompt


def test_compiler_view_accepts_rank_zero_tensor_pointer() -> None:
    argument = scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", ())
    assert argument.shape == ()
    assert argument.to_json()["shape"] == []


def test_scrc_lift_driver_attaches_source_and_provenance(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["j"]["node_ids"] = [
        {"id": key, "path": value}
        for key, value in payload["j"]["node_ids"].items()
    ]
    first_access = payload["j"]["boundary"]["accesses"][0]
    first_access["fill"] = first_access.pop("fill_value")
    first_effect = payload["j"]["boundary"]["effects"][0]
    first_effect["tensor"] = first_effect.pop("target")
    first_effect.update(
        {
            "effect_id": "e0",
            "index_expression": "i",
            "predicate": "i < N",
            "value": "A[i]",
        }
    )
    response = LLMReply(
        json.dumps(payload),
        "gpt-test",
        {"id": "resp-test"},
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    source = "@triton.jit\ndef add_kernel(...): pass"
    result = scrc.extract_scrc_artifact(source, "add_kernel", max_repairs=0)
    assert result.artifact.source == source
    assert result.artifact.provenance[0].model == "gpt-test"
    assert result.provenance.response_id == "resp-test"
    assert result.provenance.prompt_sha256 == scrc.scrc_lift_prompt_sha256(
        source, "add_kernel"
    )


def test_scrc_lift_uses_and_checks_trusted_compiler_view(monkeypatch) -> None:
    captured: dict = {}
    response = LLMReply(
        json.dumps(_artifact_payload()), "gpt-test", {"id": "resp-view"}
    )

    def fake_call(instructions, user_input, **kwargs):
        captured["user_input"] = user_input
        captured["text_format"] = kwargs["text_format"]
        return response

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    view = scrc.SCRCCompilerView(
        {"BLOCK": 64, "num_warps": 4},
        ("num_warps",),
        (),
        (
            scrc.SCRCCompilerArgument("a_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("b_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("N", "scalar", "i32", value=65),
        ),
        (2,),
        ("ceil_div(N, BLOCK)",),
        ("a_ptr", "b_ptr", "out_ptr", "N", "BLOCK"),
        ("BLOCK",),
    )

    result = scrc.extract_scrc_artifact(
        "source", "add_kernel", compiler_view=view, max_repairs=0
    )

    assert result.artifact.schedule.values == {
        "BLOCK": 64,
        "num_warps": 4,
    }
    assert "TRUSTED_COMPILER_VIEW" in captured["user_input"]
    assert '"compiler_slots": ["num_warps"]' in captured["user_input"]
    assert '"shape": [65]' in captured["user_input"]
    assert '"grid": [2]' in captured["user_input"]
    assert '"grid_contract": ["ceil_div(N, BLOCK)"]' in captured[
        "user_input"
    ]
    assert captured["text_format"]["name"] == "scrc_gate1_artifact"
    rewriter_view = result.artifact.rewriter_view()
    assert rewriter_view["j"]["intent"]["intent_function"]["layer_c"] == {}
    assert "values" not in rewriter_view["schedule"]


def test_scrc_lift_rejects_compiler_view_drift(monkeypatch) -> None:
    response = LLMReply(
        json.dumps(_artifact_payload()), "gpt-test", {"id": "resp-drift"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    view = scrc.SCRCCompilerView({"BLOCK": 128}, (), ())

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            "source",
            "add_kernel",
            compiler_view=view,
            max_repairs=0,
        )

    assert "schedule.values differs from trusted compiler view" in str(
        caught.value
    )


def test_scrc_lift_accepts_runtime_scalar_tensor_ref(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "eps", "dtype": "f32", "shape": []}
    )
    payload["interface"]["parameters"].insert(
        -1,
        {
            "name": "eps",
            "kind": "scalar",
            "mode": "value",
            "dtype": "f32",
            "rank": 0,
            "tensor_ref": "eps",
        },
    )
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-scalar"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    view = scrc.SCRCCompilerView(
        {"BLOCK": 64, "num_warps": 4},
        ("num_warps",),
        (),
        (
            scrc.SCRCCompilerArgument("a_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("b_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("N", "scalar", "i32", value=65),
            scrc.SCRCCompilerArgument("eps", "scalar", "f32", value=1.0e-5),
        ),
        (2,),
        ("ceil_div(N, BLOCK)",),
        ("a_ptr", "b_ptr", "out_ptr", "N", "eps", "BLOCK"),
        ("BLOCK",),
    )

    result = scrc.extract_scrc_artifact(
        "source", "add_kernel", compiler_view=view, max_repairs=0
    )

    eps = next(
        item for item in result.artifact.interface.parameters if item.name == "eps"
    )
    assert eps.tensor_ref == "eps"
    assert result.artifact.j.intent.inputs[-1].shape == []


def test_scrc_lift_rejects_non_scalar_semantic_tensor_ref(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "eps", "dtype": "f32", "shape": [1]}
    )
    payload["interface"]["parameters"].insert(
        -1,
        {
            "name": "eps",
            "kind": "scalar",
            "mode": "value",
            "dtype": "f32",
            "rank": 0,
            "tensor_ref": "eps",
        },
    )
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-bad-scalar"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    view = scrc.SCRCCompilerView(
        {"BLOCK": 64, "num_warps": 4},
        ("num_warps",),
        (),
        (
            scrc.SCRCCompilerArgument("a_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("b_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("N", "scalar", "i32", value=65),
            scrc.SCRCCompilerArgument("eps", "scalar", "f32", value=1.0e-5),
        ),
        (2,),
        ("ceil_div(N, BLOCK)",),
        ("a_ptr", "b_ptr", "out_ptr", "N", "eps", "BLOCK"),
        ("BLOCK",),
    )

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            "source", "add_kernel", compiler_view=view, max_repairs=0
        )

    assert "tensor_ref must have shape=[]" in str(caught.value)


def test_scrc_lift_rejects_j_tensor_dtype_drift(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["inputs"][0]["dtype"] = "f16"
    payload["interface"]["parameters"][0]["dtype"] = "f16"
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-j-dtype"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    view = scrc.SCRCCompilerView(
        {"BLOCK": 64, "num_warps": 4},
        ("num_warps",),
        (),
        (
            scrc.SCRCCompilerArgument("a_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("b_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("N", "scalar", "i32", value=65),
        ),
        (2,),
        ("ceil_div(N, BLOCK)",),
        ("a_ptr", "b_ptr", "out_ptr", "N", "BLOCK"),
        ("BLOCK",),
    )

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            "source", "add_kernel", compiler_view=view, max_repairs=0
        )

    assert "interface parameter 'a_ptr' dtype differs" in str(caught.value)


def test_scrc_lift_rejects_concrete_c_inside_j(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["layer_c"] = {
        "tile_size": 64
    }
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-layer-c"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact("source", "add_kernel", max_repairs=0)

    assert "J.intent.layer_c must be empty" in str(caught.value)


def test_scrc_lift_error_retains_all_rejected_responses(monkeypatch) -> None:
    replies = iter(
        [
            LLMReply('{"bad": 1}', "gpt-test", {"id": "resp-1"}),
            LLMReply('{"still_bad": 2}', "gpt-test", {"id": "resp-2"}),
        ]
    )
    monkeypatch.setattr(
        scrc, "call_responses", lambda *args, **kwargs: next(replies)
    )

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact("source", "add_kernel", max_repairs=1)

    assert [item.response_id for item in caught.value.attempts] == [
        "resp-1",
        "resp-2",
    ]
    assert caught.value.attempts[0].response_text == '{"bad": 1}'
    assert "attempt 1" in str(caught.value)
    assert "attempt 2" in str(caught.value)
