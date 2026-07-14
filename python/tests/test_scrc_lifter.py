from __future__ import annotations

import json

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
    assert "tl.load" not in prompt
    assert "tl.sum" not in prompt


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
