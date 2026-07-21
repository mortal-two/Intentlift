from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from intentir.extraction.providers.client import LLMReply
from intentir.extraction.lifters import scrc
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.composer import materialize_replay_plan
from intentir.verify.gate1.model import Frontend, Gate1Artifact
from intentir.verify.gate1.orchestrate import (
    ReliftEvidence,
    ReliftGuidance,
    ReliftRequest,
    ReliftTrigger,
)
from intentir.verify.gate1.rules import (
    RuleCard,
    RuleRegistry,
    default_triton_rule_registry,
)


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
                    {"effect_id": "e0", "kind": "read", "target": "A", "node_ref": "n0", "access_ref": "a0"},
                    {"effect_id": "e1", "kind": "read", "target": "B", "node_ref": "n0", "access_ref": "a1"},
                    {"effect_id": "e2", "kind": "write", "target": "Y", "node_ref": "n0", "access_ref": "a2"},
                ],
                "output_mappings": [
                    {"output": "Y", "value": "Y", "node_ref": "n0"}
                ],
            },
            "scopes": [
                {
                    "scope_id": "q0",
                    "kind": "function",
                    "parent_scope": None,
                    "captures": [],
                    "definitions": ["A", "B", "Y"],
                    "effects": ["e0", "e1", "e2"],
                }
            ],
            "storage": [
                {
                    "storage_id": "b0",
                    "members": ["Y"],
                    "scope_id": "q0",
                    "mutability": "write_once",
                    "init_event": {"kind": "access", "ref": "a2"},
                    "last_use_event": {"kind": "scope_exit", "ref": "q0"},
                    "escapes": True,
                    "crosses_iteration": False,
                }
            ],
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
        "strategy": {"bindings": []},
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


def _trusted_artifact() -> Gate1Artifact:
    payload = _artifact_payload()
    payload["plan"] = {"steps": []}
    payload["source"] = "ORIGINAL_SOURCE_CANARY"
    payload["provenance"] = [
        {
            "role": "lifter",
            "producer": "initial-test-lifter",
            "model": "gpt-initial",
            "source_digest": "initial-digest",
        }
    ]
    artifact = Gate1Artifact.from_json(payload)
    return materialize_replay_plan(
        artifact, default_triton_rule_registry()
    )[0]


def test_scrc_lift_core_prompt_keeps_pi_inside_lift_finalization() -> None:
    registry = default_triton_rule_registry()
    prompt = scrc.build_scrc_lift_instructions(registry)
    assert "Return one JSON object for LiftCore" in prompt
    assert "LiftCore response schema forbids plan/pi" in prompt
    assert "same\nLift operation deterministically materializes pi" in prompt
    assert "complete LiftArtifact to Gate1" in prompt
    assert "Gate1 does not generate missing pi" in prompt
    assert "C.strategy rules" in prompt
    assert "program_coordinate_width=wide" in prompt
    assert "address_association=pointer_staged" in prompt
    assert "mask_materialization=staged_reuse" in prompt
    assert "reduction_call_shape=flattened" in prompt
    assert "reduction_tiling=multi_tile_accumulate" in prompt
    assert (
        "reduction_preheader_staging=stage_invariant_access_scaffolding"
        in prompt
    )
    assert "reduction_tile_fusion=shared_producer_outputs" in prompt
    assert "shared per-tile producer" in prompt
    assert "shared lane/range basis before the runtime loop" in prompt
    assert "Tile-start-plus-lane offsets" in prompt
    assert "complete ordered ABI capability" in prompt
    assert "compile-time inactive" in prompt
    assert "including such an\n  inactive argument" in prompt
    assert "complete shape exactly match the trusted" in prompt
    assert "static ABI typing, not a\n  semantic use" in prompt
    assert "tile_slot" in prompt
    assert "accumulator_dtype" in prompt
    assert "lower bound exactly zero" in prompt
    assert "non-identity seed" in prompt
    assert "Registered replay rule ids" not in prompt
    assert '"id": "bind_wide_program_domain"' not in prompt
    assert registry.digest in prompt
    assert '"target_kinds"' in prompt
    assert "COLLAPSED" in prompt
    assert "at least one of" in prompt
    assert "global [M,N] row reduce" in prompt
    assert "tile_axis=0" in prompt
    assert "Mean/RSTD [M]" in prompt
    assert "require no boundary-driven rank insertion" in prompt
    assert "local Triton insertion positions" in prompt
    assert "copy `col < N`" in prompt
    assert "staged pointer additions" in prompt
    assert "load(ptr + row*N + lane)" in prompt
    assert "offset = row*N + lane" in prompt
    assert "index_expression string alone" in prompt
    assert "one boolean ContractExpr" in prompt
    assert '`"d0":"0 <= row < n_rows"`' in prompt
    assert "`program_id`, `grid_axis`" in prompt
    assert "not portable Layer B" in prompt


def test_scrc_lift_prompt_identity_tracks_registry_capabilities_without_api_text() -> None:
    base = default_triton_rule_registry()
    added = RuleCard(
        "realize_reviewed_new_op",
        Frontend.TRITON,
        target_kinds=["intent_op:reviewed_new_op"],
        construction_phase=60,
        required_fields=["operands"],
        api_mapping={"unsafe_canary": "libdevice.private_canary"},
    )
    extended = RuleRegistry([*base.cards, added])

    assert scrc.scrc_lift_prompt_sha256("SOURCE", "kernel", registry=base) != (
        scrc.scrc_lift_prompt_sha256("SOURCE", "kernel", registry=extended)
    )
    prompt = scrc.build_scrc_lift_instructions(extended)
    assert "intent_op:reviewed_new_op" in prompt
    assert "private_canary" not in prompt
    assert "realize_reviewed_new_op" not in prompt

    empty = RuleRegistry([])
    assert scrc.scrc_lift_prompt_sha256("SOURCE", "kernel", registry=empty) != (
        scrc.scrc_lift_prompt_sha256("SOURCE", "kernel", registry=base)
    )
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
    assert "flag_gems.utils.triton_lang_extension as tle" in prompt
    assert "`get_dtype_min` returns negative" in prompt
    assert "`get_dtype_max` returns positive" in prompt
    assert "do not substitute the largest finite value" in prompt
    assert "tle.program_id" in prompt
    assert "reduction_call_shape=flattened" in prompt
    assert "reduction_tiling=multi_tile_accumulate" in prompt
    assert "reduction_tile_fusion=shared_producer_outputs" in prompt
    assert "only loop-carried value" in prompt
    assert "explicit source axis uses the" in prompt
    assert "m,n->mn" in prompt
    assert "encode a contraction rule choice" in prompt
    assert "Never emit plan/pi in LiftCore" in prompt
    assert "enclosing Lift finalizer adds canonical pi" in prompt
    assert "exact symbolic runtime extent name" in prompt
    assert "Never substitute a sampled" in prompt
    assert "J.intent Layer A must be invariant" in prompt
    assert "program ids, block ids, tile bases, lane ids" in prompt
    assert "one global semantic ordinal in J" in prompt
    assert "attrs.n is the exact runtime symbol" in prompt
    assert "quotient/block-base/lane arithmetic" in prompt
    assert "declaration-order opaque handles" in prompt
    assert "Scope.definitions contains J value handles" in prompt
    assert "`[n0,a0,a1,a2,d0]`" in prompt
    assert "only a semantic `op=\"loop\"` may carry IntentOp.scope_id" in prompt
    assert "scope ids are q0,q1" in prompt
    assert "access_id values" in prompt
    assert "Never encode" in prompt
    assert "operation, tensor, load/store" in prompt
    assert "reduction, source variable or source-line hints" in prompt
    assert "tl.load" not in prompt
    assert "tl.sum" not in prompt


def _iota_extent_artifact(n, output_extent):
    intent = IntentFunction.from_json(
        {
            "name": "iota_extent_contract",
            "inputs": [],
            "outputs": [
                {"name": "Y", "dtype": "i32", "shape": [output_extent]}
            ],
            "body": [
                {
                    "op": "iota",
                    "result": "I",
                    "attrs": {"n": n, "dtype": "i32"},
                },
                {
                    "op": "cast",
                    "result": "Y",
                    "operands": ["I"],
                    "attrs": {"to_dtype": "i32"},
                },
            ],
        }
    )
    mapping = SimpleNamespace(output="Y", value="Y")
    return SimpleNamespace(
        j=SimpleNamespace(
            intent=intent,
            boundary=SimpleNamespace(output_mappings=[mapping]),
        )
    )


@pytest.mark.parametrize(
    ("n", "output_extent"),
    ((0, 0), (1, 1), (127, 127), (128, 128), (129, 129), ("N", "N")),
)
def test_iota_extent_contract_accepts_static_boundaries_and_runtime_symbol(
    n, output_extent
) -> None:
    scrc._validate_scrc_iota_extents(
        _iota_extent_artifact(n, output_extent)
    )


def test_iota_extent_contract_rejects_sampled_and_random_mismatches() -> None:
    with pytest.raises(ValueError, match="keep runtime extents symbolic"):
        scrc._validate_scrc_iota_extents(
            _iota_extent_artifact(1003, "size")
        )

    rng = random.Random(20260717)
    for _ in range(25):
        extent = rng.randint(1, 4096)
        wrong = extent + rng.choice((-1, 1))
        with pytest.raises(ValueError, match="iota extent"):
            scrc._validate_scrc_iota_extents(
                _iota_extent_artifact(wrong, extent)
            )


def test_lift_driver_rejects_sampled_iota_extent_before_pi(
    monkeypatch,
) -> None:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["body"] = [
        {
            "op": "iota",
            "result": "Y",
            "attrs": {"n": 64, "dtype": "i32"},
        }
    ]
    response = LLMReply(json.dumps(payload), "gpt-test", {"id": "bad-iota"})
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            "def add_kernel(...): pass",
            "add_kernel",
            max_repairs=0,
        )

    assert "iota extent must equal" in str(caught.value)


def test_compiler_view_accepts_rank_zero_tensor_pointer() -> None:
    argument = scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", ())
    assert argument.shape == ()
    assert argument.to_json()["shape"] == []


def test_compiler_view_preserves_disabled_optional_none_argument() -> None:
    argument = scrc.SCRCCompilerArgument("Bias", "none", "none")
    view = scrc.SCRCCompilerView(
        {"BIAS_TYPE": "none"},
        arguments=(argument,),
        grid=(1,),
        grid_contract=("1",),
        parameter_order=("Bias",),
    )

    assert argument.to_json() == {
        "name": "Bias",
        "kind": "none",
        "dtype": "none",
    }
    assert view.to_json()["schedule_values"] == {"BIAS_TYPE": "none"}


def test_scrc_extraction_rejects_an_incomplete_lift_artifact() -> None:
    payload = _artifact_payload()
    payload["plan"] = {"steps": []}
    incomplete = Gate1Artifact.from_json(payload)
    provenance = scrc.SCRCProvenance("gpt-test", "prompt", 1)

    with pytest.raises(ValueError, match="complete LiftArtifact"):
        scrc.SCRCExtraction(incomplete, provenance)


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
        {
            "id": "resp-test",
            "model": "gpt-test-2026-07-19",
            "usage": {
                "input_tokens": 123,
                "output_tokens": 456,
                "total_tokens": 579,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    source = "@triton.jit\ndef add_kernel(...): pass"
    result = scrc.extract_scrc_artifact(source, "add_kernel", max_repairs=0)
    assert result.artifact.source == source
    assert result.artifact.plan.steps
    assert result.artifact.compose_provenance is not None
    assert result.artifact.provenance[0].model == "gpt-test"
    assert result.provenance.response_id == "resp-test"
    assert result.provenance.response_ids == ("resp-test",)
    assert result.provenance.response_model == "gpt-test-2026-07-19"
    assert result.provenance.usage == {
        "input_tokens": 123,
        "output_tokens": 456,
        "total_tokens": 579,
    }
    assert result.artifact.provenance[0].metadata["usage"] == {
        "input_tokens": 123,
        "output_tokens": 456,
        "total_tokens": 579,
    }
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
    assert captured["text_format"]["name"] == "scrc_lift_core"
    rewriter_view = result.artifact.rewriter_view()
    assert rewriter_view["j"]["intent"]["intent_function"]["layer_c"] == {}
    assert "values" not in rewriter_view["schedule"]


def test_scrc_lift_rejects_provider_supplied_pi(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["plan"] = {"steps": []}
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-illegal-pi"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact("source", "add_kernel", max_repairs=0)

    assert "must not contain plan/pi" in str(caught.value)


def test_semantic_relift_may_change_strategy_and_always_recomposes_pi(
    monkeypatch,
) -> None:
    payload = _artifact_payload()
    payload["strategy"] = {
        "bindings": [
            {
                "target_ref": "d0",
                "family": "program_coordinate_width",
                "choice": "wide",
            }
        ]
    }
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-strategy-relift"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    trusted = _trusted_artifact()
    context = scrc.SCRCReliftContext(1, {"schema_version": "test"}, ())

    result = scrc.extract_scrc_artifact(
        trusted.source or "",
        trusted.j.kernel_id,
        relift_context=context,
        trusted_artifact=trusted,
        max_repairs=0,
    ).artifact

    assert result.interface == trusted.interface
    assert result.schedule == trusted.schedule
    assert result.strategy.to_json() == payload["strategy"]
    assert any(step.rule == "bind_wide_program_domain" for step in result.plan.steps)
    assert result.compose_provenance is not None


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
    payload["j"]["scopes"][0]["definitions"].append("eps")
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
    assert any(
        step.rule == "bind_runtime_scalar" and step.refs == ["eps"]
        for step in result.artifact.plan.steps
    )


def _inactive_tensor_payload_and_view(
    *, tensor_shape: list[int],
) -> tuple[dict, scrc.SCRCCompilerView]:
    payload = _artifact_payload()
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "Optional", "dtype": "f32", "shape": tensor_shape}
    )
    payload["j"]["scopes"][0]["definitions"].insert(2, "Optional")
    payload["interface"]["parameters"].insert(
        2,
        {
            "name": "optional_ptr",
            "kind": "pointer",
            "mode": "input",
            "dtype": "f32",
            "rank": 1,
            "tensor_ref": "Optional",
        },
    )
    view = scrc.SCRCCompilerView(
        {"BLOCK": 64, "num_warps": 4},
        ("num_warps",),
        (),
        (
            scrc.SCRCCompilerArgument("a_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("b_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("optional_ptr", "tensor", "f32", (1,)),
            scrc.SCRCCompilerArgument("out_ptr", "tensor", "f32", (65,)),
            scrc.SCRCCompilerArgument("N", "scalar", "i32", value=65),
        ),
        (2,),
        ("ceil_div(N, BLOCK)",),
        ("a_ptr", "b_ptr", "optional_ptr", "out_ptr", "N", "BLOCK"),
        ("BLOCK",),
    )
    return payload, view


def test_scrc_lift_preserves_exact_typed_inactive_tensor_abi(monkeypatch) -> None:
    payload, view = _inactive_tensor_payload_and_view(tensor_shape=[1])
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-inactive-tensor"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    artifact = scrc.extract_scrc_artifact(
        "source", "add_kernel", compiler_view=view, max_repairs=0
    ).artifact

    parameter = artifact.interface.parameters[2]
    assert parameter.tensor_ref == "Optional"
    assert artifact.j.intent.inputs[2].shape == [1]
    assert not any(
        access.tensor == "Optional" for access in artifact.j.boundary.accesses
    )


def test_scrc_lift_reports_inactive_tensor_shape_rank_drift(monkeypatch) -> None:
    payload, view = _inactive_tensor_payload_and_view(tensor_shape=[1, 1])
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-inactive-rank-drift"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            "source", "add_kernel", compiler_view=view, max_repairs=0
        )

    diagnostic = str(caught.value)
    assert "J shape [1, 1] (rank 2)" in diagnostic
    assert "compiler-view shape [1] (rank 1)" in diagnostic


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
    payload["j"]["scopes"][0]["definitions"].append("eps")
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


def test_scrc_lift_rejects_semantic_names_in_opaque_reference_ids(
    monkeypatch,
) -> None:
    payload = _artifact_payload()
    payload["j"]["boundary"]["accesses"][0]["access_id"] = "a_load"
    payload["j"]["boundary"]["effects"][0]["access_ref"] = "a_load"
    payload["location_map"]["entries"][0]["access_refs"][0] = "a_load"
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-semantic-ref"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact("source", "add_kernel", max_repairs=0)

    assert "must use declaration-order opaque ids" in str(caught.value)
    assert "a_load" in str(caught.value)


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


def test_scrc_lift_retains_untyped_domain_for_gate2_scalar_handoff(
    monkeypatch,
) -> None:
    invalid_payload = _artifact_payload()
    invalid_domain = "b = grid_axis(0), 0 <= b < 2048"
    invalid_payload["j"]["boundary"]["domains"]["d0"] = invalid_domain
    replies = iter(
        [
            LLMReply(json.dumps(invalid_payload), "gpt-test", {"id": "resp-bad"}),
        ]
    )
    conversations: list[str] = []

    def fake_call(instructions, user_input, **kwargs):
        del instructions, kwargs
        conversations.append(user_input)
        return next(replies)

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    result = scrc.extract_scrc_artifact(
        "source", "add_kernel", max_repairs=1
    )

    assert result.artifact.j.boundary.domains == {"d0": invalid_domain}
    assert result.provenance.repairs == ()
    assert invalid_domain in json.dumps(invalid_payload)
    assert len(conversations) == 1


def test_real_relifter_sees_source_evidence_but_never_replay_source(
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    response = LLMReply(
        json.dumps(_artifact_payload()),
        "gpt-5.6-sol",
        {"id": "resp-relift"},
    )

    def fake_call(instructions, user_input, **kwargs):
        captured["instructions"] = instructions
        captured["user_input"] = user_input
        return response

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    evidence = ReliftEvidence(
        relift_round=0,
        generation_round=1,
        generation_available=True,
        compiled=True,
        comparison={
            "hard_pass": False,
            "primary_difference_category": "op_multiset",
            "differences": [
                {
                    "category": "op_multiset",
                    "kind": "missing",
                    "item": "RAW_HISTORY_ONLY_CANARY",
                    "expected_count": 1,
                    "actual_count": 0,
                }
            ],
        },
        diagnosis={
            "cause": "replay_compile",
            "responsibility": "lift_artifact",
            "mismatches": [
                "NameError: replay-local dependency is undefined"
            ],
            "confirmed_fields": [],
            "allowed_modifications": ["j", "strategy", "location_map"],
            "explanation": "return the compile diagnosis to Lift",
        },
        adopted=True,
        regressed=False,
    )
    fingerprint = SimpleNamespace(
        to_dict=lambda: {
            "schema_version": "test",
            "op_multiset": {"arith.extsi": 3},
        }
    )
    trusted = _trusted_artifact()
    relifter = scrc.ResponsesSCRCRelifter(max_repairs=0, effort="xhigh")

    result = relifter.relift(
        trusted,
        source_fingerprint=fingerprint,
        history=(evidence,),
        request=ReliftRequest(
            ReliftTrigger.NORMAL_NO_CLOSURE,
            evaluation_round=0,
            candidate_count=1,
            compiled_candidate_count=1,
            best_generation_round=1,
            hard_difference_categories=("op_multiset",),
            semantic_guidance=ReliftGuidance(
                hard_difference_digest="diff-digest-canary",
                failed_hard_categories=("op_multiset",),
                confirmed_hard_categories=("interface",),
                category_guidance=(
                    {
                        "category": "op_multiset",
                        "grouped_differences": [
                            {
                                "kind": "missing",
                                "item": "SOURCE_EXPECTED_OP_CANARY",
                                "expected_count": 3,
                                "actual_count": 0,
                            }
                        ],
                        "inspect_j_surfaces": ["j.intent.body[*].op/attrs"],
                        "reviewed_strategy_families": [],
                        "semantic_question": "re-derive whole tensor algebra",
                    },
                ),
                current_j_summary={"intent_op_multiset": {"cast": 1}},
                decision_contract=(
                    "return_unchanged_editable_fields_when_no_semantic_change_is_justified",
                ),
            ),
        ),
    )

    assert result.interface.to_json() == trusted.interface.to_json()
    assert result.schedule.to_json() == trusted.schedule.to_json()
    assert len(result.provenance) == 2
    assert result.provenance[-1].metadata["kind"] == "semantic_relift"
    assert result.provenance[-1].metadata["relift_round"] == 1
    assert "outer-loop semantic relift" in captured["instructions"]
    assert "A TTIR op-count/association difference alone is not proof" in (
        captured["instructions"]
    )
    assert "return the semantically correct J unchanged" in (
        captured["instructions"]
    )
    assert "Gate1 may legitimately end in" in captured["instructions"]
    assert "not an attempt to improve generic first-response Lift" in (
        captured["instructions"]
    )
    assert "required to preserve semantics for arbitrary runtime extents" in (
        captured["instructions"]
    )
    assert "bounded scoped-SSA witness" in captured["instructions"]
    assert "exact init-to-yield pairs" in captured["instructions"]
    assert "deterministic stagnation policy" in captured["instructions"]
    assert "never copy a schedule value into a J const/iota/shape" in (
        captured["instructions"]
    )
    assert "block = iota / BLOCK" in captured["instructions"]
    assert "ORIGINAL_SOURCE_CANARY" in captured["user_input"]
    assert '"arith.extsi": 3' in captured["user_input"]
    assert '"trigger": "normal_no_closure"' in captured["user_input"]
    assert '"policy": "deterministic_gate1_hard_diff_guidance_v2"' in (
        captured["user_input"]
    )
    assert "diff-digest-canary" in captured["user_input"]
    assert "SOURCE_EXPECTED_OP_CANARY" in captured["user_input"]
    assert "RAW_HISTORY_ONLY_CANARY" not in captured["user_input"]
    assert '"comparison_summary"' in captured["user_input"]
    assert '"dataflow_edge_count": 0' in captured["user_input"]
    assert "replay-local dependency is undefined" in captured["user_input"]
    assert not hasattr(evidence, "generation")
    assert '"strategy": {"bindings": []}' in captured["user_input"]
    assert '"plan"' not in captured["user_input"]
    assert len(relifter.records) == 1
    assert relifter.records[0].extraction is not None
    assert relifter.records[0].extraction.provenance.response_id == "resp-relift"
    assert relifter.records[0].request["trigger"] == "normal_no_closure"


def test_semantic_relifter_rejects_raw_replay_history_before_provider(
    monkeypatch,
) -> None:
    def forbidden_provider_call(*args, **kwargs):
        raise AssertionError("raw replay history must fail before provider use")

    monkeypatch.setattr(scrc, "call_responses", forbidden_provider_call)
    raw_candidate = SimpleNamespace(
        generation=SimpleNamespace(code="REPLAY_SOURCE_CANARY")
    )
    fingerprint = SimpleNamespace(to_dict=lambda: {"schema_version": "test"})
    relifter = scrc.ResponsesSCRCRelifter(max_repairs=0, effort="xhigh")

    with pytest.raises(TypeError, match="source-free ReliftEvidence"):
        relifter.relift(
            _trusted_artifact(),
            source_fingerprint=fingerprint,
            history=(raw_candidate,),
            request=ReliftRequest(
                ReliftTrigger.REPLAY_COMPILE_FAILED,
                evaluation_round=0,
                candidate_count=1,
                compiled_candidate_count=0,
                best_generation_round=0,
                hard_difference_categories=(),
            ),
        )


def test_semantic_relift_rejects_trusted_interface_drift(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["interface"]["grid"] = ["N"]
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-interface-drift"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    trusted = _trusted_artifact()
    context = scrc.SCRCReliftContext(1, {"schema_version": "test"}, ())

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            trusted.source or "",
            trusted.j.kernel_id,
            relift_context=context,
            trusted_artifact=trusted,
            max_repairs=0,
        )

    assert "semantic relift changed trusted interface" in str(caught.value)


def test_semantic_relift_rejects_trusted_schedule_drift(monkeypatch) -> None:
    payload = _artifact_payload()
    payload["schedule"]["values"]["BLOCK"] = 128
    response = LLMReply(
        json.dumps(payload), "gpt-test", {"id": "resp-schedule-drift"}
    )
    monkeypatch.setattr(scrc, "call_responses", lambda *args, **kwargs: response)
    trusted = _trusted_artifact()
    context = scrc.SCRCReliftContext(1, {"schema_version": "test"}, ())

    with pytest.raises(scrc.SCRCLiftError) as caught:
        scrc.extract_scrc_artifact(
            trusted.source or "",
            trusted.j.kernel_id,
            relift_context=context,
            trusted_artifact=trusted,
            max_repairs=0,
        )

    assert "semantic relift changed trusted schedule C" in str(caught.value)
