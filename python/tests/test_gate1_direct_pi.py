from __future__ import annotations

import copy
import json

import pytest

from direct_fixtures import make_reduce_artifact
from intentir.ir.model import IntentFunction
from intentir.extraction.lifters import scrc
from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCReliftContext,
    anonymize_source,
    prepare_scrc_lift,
)
from intentir.extraction.providers.client import LLMReply
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.verify.gate1.model import Gate1Artifact, ScheduleSlot, SchemaError
from intentir.verify.gate1.orchestrate import canonical_artifact_digest
from intentir.verify.gate1.pi import (
    REALIZATION_MECHANISMS,
    RealizationWitness,
    RealizationWitnessError,
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)
from intentir.verify.gate1.prompt import (
    ReplayPacket,
    assert_source_isolated_payload,
    build_normal_rewriter_prompt,
    build_masked_rewriter_view,
    build_normal_rewriter_view,
    build_rewriter_packet,
)


def test_direct_pi_artifact_roundtrip_and_legacy_interface_break() -> None:
    artifact = make_reduce_artifact()
    encoded = artifact.to_json()
    assert Gate1Artifact.from_json(encoded) == artifact
    assert encoded["schema"] == Gate1Artifact.SCHEMA

    legacy = copy.deepcopy(encoded)
    legacy["interface"] = {"parameters": []}
    with pytest.raises(SchemaError, match=r"unknown key\(s\): interface"):
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


def test_pi_parser_is_separate_from_dependent_refs_and_rejects_cycles() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = "n99"
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.pi.instances[0]["semantic_ref"] == "n99"

    pi = artifact.pi.to_json()
    pi["order"].append({"before": "i3", "after": "i0"})
    with pytest.raises(RealizationWitnessError, match="acyclic"):
        RealizationWitness.from_json(pi)


@pytest.mark.parametrize(
    ("relation", "field", "bad_ref", "expected_path"),
    [
        ("routes", "source", "p0", r"pi\.routes\[0\]\.source"),
        ("routes", "target", "c0", r"pi\.routes\[0\]\.target"),
        ("order", "before", "n0", r"pi\.order\[0\]\.before"),
        ("order", "after", "p0", r"pi\.order\[0\]\.after"),
        ("sharing", "source", "c0", r"pi\.sharing\[0\]\.source"),
    ],
)
def test_pi_graph_endpoints_reject_raw_anchor_namespaces(
    relation: str,
    field: str,
    bad_ref: str,
    expected_path: str,
) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload[relation][0][field] = bad_ref
    with pytest.raises(
        RealizationWitnessError,
        match=expected_path + r" must reference a declared pi instance id i\*",
    ):
        RealizationWitness.from_json(payload)


def test_pi_sharing_consumer_rejects_raw_anchor_namespace() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["sharing"][0]["consumers"][0] = "p0"
    with pytest.raises(
        RealizationWitnessError,
        match=(
            r"pi\.sharing\[0\]\.consumers\[0\] must reference a declared "
            r"pi instance id i\*"
        ),
    ):
        RealizationWitness.from_json(payload)


@pytest.mark.parametrize(
    ("relation", "field", "expected_path"),
    [
        ("routes", "source", r"pi\.routes\[0\]\.source"),
        ("routes", "target", r"pi\.routes\[0\]\.target"),
        ("order", "before", r"pi\.order\[0\]\.before"),
        ("order", "after", r"pi\.order\[0\]\.after"),
        ("sharing", "source", r"pi\.sharing\[0\]\.source"),
    ],
)
def test_pi_graph_endpoints_reject_undeclared_instance_ids(
    relation: str,
    field: str,
    expected_path: str,
) -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload[relation][0][field] = "i99"
    with pytest.raises(
        RealizationWitnessError,
        match=expected_path + r" references undeclared pi instance 'i99'",
    ):
        RealizationWitness.from_json(payload)


def test_pi_sharing_consumer_rejects_undeclared_instance_id() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["sharing"][0]["consumers"][0] = "i99"
    with pytest.raises(
        RealizationWitnessError,
        match=(
            r"pi\.sharing\[0\]\.consumers\[0\] references undeclared "
            r"pi instance 'i99'"
        ),
    ):
        RealizationWitness.from_json(payload)


def test_pi_rejects_graph_instance_id_in_anchor_field() -> None:
    payload = make_reduce_artifact().pi.to_json()
    payload["instances"][0]["semantic_ref"] = "i0"
    with pytest.raises(
        RealizationWitnessError,
        match=r"pi\.instances\[0\]\.semantic_ref must be an anonymous",
    ):
        RealizationWitness.from_json(payload)


def test_pi_rejects_duplicate_instance_and_sharing_consumer_ids() -> None:
    duplicate_instance = make_reduce_artifact().pi.to_json()
    duplicate_instance["instances"][1]["id"] = "i0"
    with pytest.raises(
        RealizationWitnessError,
        match="declaration-order handles",
    ):
        RealizationWitness.from_json(duplicate_instance)

    duplicate_consumer = make_reduce_artifact().pi.to_json()
    duplicate_consumer["sharing"][0]["consumers"] = ["i1", "i1"]
    with pytest.raises(
        RealizationWitnessError,
        match=r"pi\.sharing\[0\]\.consumers must not contain duplicates",
    ):
        RealizationWitness.from_json(duplicate_consumer)


@pytest.mark.parametrize("semantic_ref", ["v0", "x0", "c0", "p0"])
def test_pi_may_reference_any_declared_anonymous_j_or_c_handle(
    semantic_ref: str,
) -> None:
    payload = make_reduce_artifact().to_json()
    payload["pi"]["instances"][0]["semantic_ref"] = semantic_ref
    assert Gate1Artifact.from_json(payload).pi.instances[0]["semantic_ref"] == semantic_ref


def _row_stride_pi() -> RealizationWitness:
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": [
                {"id": "i0", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i1", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i2", "semantic_ref": "p2", "region": "r1", "mechanism": "realize"},
                {"id": "i3", "semantic_ref": "a0", "region": "r1", "mechanism": "realize"},
                {"id": "i4", "semantic_ref": "a0", "region": "r1", "mechanism": "transfer_direct"},
                {"id": "i5", "semantic_ref": "a1", "region": "r1", "mechanism": "realize"},
                {"id": "i6", "semantic_ref": "a1", "region": "r1", "mechanism": "transfer_direct"},
            ],
            "routes": [
                {"source": "i0", "target": "i1", "slots": [0]},
                {"source": "i1", "target": "i2", "slots": [0]},
                {"source": "i2", "target": "i3", "slots": [0]},
                {"source": "i3", "target": "i4", "slots": [0]},
                {"source": "i2", "target": "i5", "slots": [0]},
                {"source": "i5", "target": "i6", "slots": [0]},
            ],
            "order": [],
            "sharing": [{"source": "i2", "consumers": ["i3", "i5"]}],
        }
    )


def _flat_pi() -> RealizationWitness:
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": [
                {"id": "i0", "semantic_ref": "a0", "region": "r1", "mechanism": "transfer_direct"},
                {"id": "i1", "semantic_ref": "a1", "region": "r1", "mechanism": "transfer_direct"},
            ],
            "routes": [],
            "order": [],
            "sharing": [],
        }
    )


def _address_pi(*, staged: bool) -> RealizationWitness:
    instances = [
        {"id": "i0", "semantic_ref": "p0", "region": "r1", "mechanism": "realize"},
    ]
    if staged:
        instances.extend(
            [
                {
                    "id": "i1",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_base",
                },
                {
                    "id": "i2",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_lane",
                },
                {
                    "id": "i3",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "transfer_direct",
                },
            ]
        )
        routes = [
            {"source": "i0", "target": "i1", "slots": [0]},
            {"source": "i1", "target": "i2", "slots": [0]},
            {"source": "i2", "target": "i3", "slots": [0]},
        ]
    else:
        instances.extend(
            [
                {
                    "id": "i1",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "address_lane",
                },
                {
                    "id": "i2",
                    "semantic_ref": "a0",
                    "region": "r1",
                    "mechanism": "transfer_direct",
                },
            ]
        )
        routes = [
            {"source": "i0", "target": "i1", "slots": [0]},
            {"source": "i1", "target": "i2", "slots": [0]},
        ]
    return RealizationWitness.from_json(
        {
            "regions": [
                {"id": "r0", "parent": None, "placement": "root"},
                {"id": "r1", "parent": "r0", "placement": "program"},
            ],
            "instances": instances,
            "routes": routes,
            "order": [],
            "sharing": [],
        }
    )


def test_existing_pi_distinguishes_runtime_row_base_from_flat_access() -> None:
    row = _row_stride_pi()
    flat = _flat_pi()
    assert row.to_json() != flat.to_json()
    assert [
        item["semantic_ref"] for item in row.instances
    ].count("p2") == 3
    assert all(item["semantic_ref"] != "p2" for item in flat.instances)
    assert row.sharing == [{"source": "i2", "consumers": ["i3", "i5"]}]


def test_address_mechanisms_distinguish_staged_base_from_flat_lane() -> None:
    staged = _address_pi(staged=True)
    flat = _address_pi(staged=False)
    assert staged.to_json() != flat.to_json()
    assert [item["mechanism"] for item in staged.instances] == [
        "realize", "address_base", "address_lane", "transfer_direct"
    ]
    assert [item["mechanism"] for item in flat.instances] == [
        "realize", "address_lane", "transfer_direct"
    ]
    assert staged.routes[1] == {
        "source": "i1", "target": "i2", "slots": [0]
    }
    assert RealizationWitness.from_json(staged.to_json()) == staged


@pytest.mark.parametrize(
    "downstream",
    [
        "transfer_direct",
        "transfer_async",
        "update_atomic",
        "update_privatized",
    ],
)
def test_address_mechanisms_are_orthogonal_to_transfer_and_update(
    downstream: str,
) -> None:
    payload = _address_pi(staged=True).to_json()
    payload["instances"][-1]["mechanism"] = downstream
    parsed = RealizationWitness.from_json(payload)
    assert parsed.instances[-2]["mechanism"] == "address_lane"
    assert parsed.instances[-1]["mechanism"] == downstream


def test_address_mechanisms_survive_replay_without_semantic_payload() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _address_pi(staged=True)
    view = build_normal_rewriter_view(artifact)
    assert view["pi"] == artifact.pi.to_json()
    encoded = json.dumps(view["pi"], sort_keys=True)
    for forbidden in ("formula", "dtype", "shape", "predicate", "tl.load"):
        assert forbidden not in encoded


def test_address_mechanism_deletion_or_swap_changes_witness_identity() -> None:
    staged = make_reduce_artifact()
    staged.pi = _address_pi(staged=True)
    flat = make_reduce_artifact()
    flat.pi = _address_pi(staged=False)
    swapped = make_reduce_artifact()
    swapped_payload = _address_pi(staged=True).to_json()
    swapped_payload["instances"][1]["mechanism"] = "address_lane"
    swapped_payload["instances"][2]["mechanism"] = "address_base"
    swapped.pi = RealizationWitness.from_json(swapped_payload)

    assert len(
        {
            canonical_artifact_digest(staged),
            canonical_artifact_digest(flat),
            canonical_artifact_digest(swapped),
        }
    ) == 3


def test_address_pi_is_unchanged_by_masked_j_and_cannot_supply_value_dag() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _address_pi(staged=True)
    normal = build_normal_rewriter_view(artifact)
    masked = build_masked_rewriter_view(artifact)
    assert normal["pi"] == masked["pi"]
    assert normal["j"] != masked["j"]
    encoded = json.dumps(masked["pi"], sort_keys=True)
    for semantic_atom in ("map_binary", "reduce", "sum", "mul", "output_mappings"):
        assert semantic_atom not in encoded


def test_pi_repeated_anchor_and_runtime_routes_survive_replay_packet() -> None:
    artifact = make_reduce_artifact()
    artifact.pi = _row_stride_pi()
    view = build_normal_rewriter_view(artifact)
    assert view["pi"] == artifact.pi.to_json()
    assert view["pi"]["routes"][2] == {
        "source": "i2", "target": "i3", "slots": [0]
    }
    instructions, _user = build_normal_rewriter_prompt(artifact)
    assert "do not collapse their chain" in instructions
    assert "discard J/C dependencies" in instructions


def test_layernorm_mask_motion_changes_j_not_pi_semantics() -> None:
    common = {
        "name": "scrc_j0",
        "inputs": [
            {"name": "t0", "dtype": "f32", "shape": [8]},
            {"name": "t1", "dtype": "i1", "shape": [8]},
        ],
        "outputs": [{"name": "t2", "dtype": "f32", "shape": []}],
    }
    pre = IntentFunction.from_json(
        {
            **common,
            "body": [
                {"id": "n0", "op": "select", "operands": ["t1", "t0", "ZERO"], "result": "v0"},
                {"id": "n1", "op": "map_binary", "operands": ["v0", "v0"], "attrs": {"fn": "mul"}, "result": "v1"},
                {"id": "n2", "op": "reduce", "operands": ["v1"], "attrs": {"fn": "sum", "axes": [0]}, "result": "t2"},
            ],
        }
    )
    post = IntentFunction.from_json(
        {
            **common,
            "body": [
                {"id": "n0", "op": "map_binary", "operands": ["t0", "t0"], "attrs": {"fn": "mul"}, "result": "v0"},
                {"id": "n1", "op": "select", "operands": ["t1", "v0", "ZERO"], "result": "v1"},
                {"id": "n2", "op": "reduce", "operands": ["v1"], "attrs": {"fn": "sum", "axes": [0]}, "result": "t2"},
            ],
        }
    )
    shared_pi = _flat_pi().to_json()
    assert pre.to_json() != post.to_json()
    assert shared_pi == _flat_pi().to_json()
    assert "select" not in json.dumps(shared_pi, sort_keys=True)


def test_masked_j_changes_only_j() -> None:
    artifact = make_reduce_artifact()
    normal = build_normal_rewriter_view(artifact)
    masked = build_masked_rewriter_view(artifact)
    assert normal["j"] != masked["j"]
    for key in ("c", "pi", "u_f"):
        assert normal[key] == masked[key]
    assert set(normal) == {"j", "c", "pi", "u_f"}


def test_replay_packet_is_source_isolated() -> None:
    packet = build_rewriter_packet(make_reduce_artifact())
    assert isinstance(packet, ReplayPacket)
    payload = packet.to_json()
    encoded = json.dumps(payload, sort_keys=True)
    assert "PRIVATE_SOURCE_CANARY" not in encoded
    assert "location_map" not in encoded
    assert "provenance" not in encoded
    assert set(payload["c"]) == {"slots"}
    assert "values" not in payload["c"]
    assert_source_isolated_payload(payload)


def test_triton_u_f_preserves_constexpr_and_projects_coordinate_roles() -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    context = payload["u_f"]
    contract = context["compile_entry_contract"]

    assert context["context_policy"] == "public_frontend_context_v2"
    assert "pass a c* handle directly to tl.arange" in contract
    assert "never copy it to a local first" in contract
    assert "satisfied by the parameter binding" in contract
    assert "J.boundary.reductions.tile_axis" in contract
    assert "physical frontend reduction axis" in contract
    assert "placement='program' alone never authorizes tl.program_id" in contract
    assert "path enters address_base" in contract
    assert "path enters address_lane" in contract
    assert "map declaration-ordered stages" in contract
    assert_source_isolated_payload(payload)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("j",), "interface"),
        (("c",), "values"),
        (("pi",), "binding_map"),
        (("u_f",), "compiler_fingerprint"),
    ],
)
def test_replay_packet_nested_allowlists_fail_closed(
    path: tuple[str, ...], key: str
) -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    target = payload
    for component in path:
        target = target[component]
    target[key] = "forbidden"
    with pytest.raises(SchemaError):
        assert_source_isolated_payload(payload)


def test_replay_packet_rejects_candidate_selected_frontend_context() -> None:
    payload = build_normal_rewriter_view(make_reduce_artifact())
    payload["u_f"]["custom_context"] = "source-derived hint"
    with pytest.raises(SchemaError, match="unknown field"):
        ReplayPacket(
            payload["j"], payload["c"], payload["pi"], payload["u_f"]
        )


def test_structured_gate1_diff_returns_only_to_source_aware_relift() -> None:
    source = """
import triton
import triton.language as tl

@triton.jit
def source_kernel(X, Y, N: tl.constexpr):
    tl.store(Y, tl.load(X))
"""
    context = SCRCReliftContext(
        1,
        "gate1",
        {"observer_diff": {"category": "dataflow", "kind": "missing"}},
    )
    request = prepare_scrc_lift(source, relift_context=context)
    lift_payload = json.loads(request.user_input)
    assert lift_payload["relift"] == context.to_json()

    replay_payload = build_normal_rewriter_view(make_reduce_artifact())
    assert "relift" not in json.dumps(replay_payload, sort_keys=True)
    assert "observer_diff" not in json.dumps(replay_payload, sort_keys=True)

    with pytest.raises(ValueError, match="unsupported relift"):
        SCRCReliftContext(1, "normal_no_closure")


def test_lift_schema_requires_direct_pi_and_no_embedded_c_channel() -> None:
    schema = scrc_lift_core_json_schema()
    properties = schema["properties"]
    assert set(properties) == {"schema", "j", "c", "pi"}
    function_properties = (
        properties["j"]["properties"]["intent"]["properties"]
        ["intent_function"]["properties"]
    )
    assert "layer_c" not in function_properties


def test_lift_wire_schema_encodes_parser_critical_anonymous_shapes() -> None:
    schema = scrc_lift_core_json_schema()
    j = schema["properties"]["j"]
    function = j["properties"]["intent"]["properties"]["intent_function"]
    assert function["properties"]["name"]["enum"] == ["scrc_j0"]
    assert j["properties"]["kernel_id"]["enum"] == ["scrc_k0"]
    assert j["properties"]["node_ids"]["propertyNames"]["pattern"] == "^n[0-9]+$"
    domains = j["properties"]["boundary"]["properties"]["domains"]
    assert domains["propertyNames"]["pattern"] == "^d[0-9]+$"
    assert domains["additionalProperties"]["type"] == "string"
    signature = j["properties"]["signature"]
    assert signature["properties"]["parameters"]["items"]["properties"][
        "name"
    ]["pattern"] == "^p[0-9]+$"
    c_schema = schema["properties"]["c"]
    assert c_schema["properties"]["slots"]["items"]["properties"]["name"][
        "pattern"
    ] == "^c[0-9]+$"


def test_lift_prompt_keeps_runtime_extent_symbolic_and_out_of_ssa_scopes() -> None:
    instructions = " ".join(scrc.build_scrc_lift_instructions().split())

    assert "only the current Harness binding" in instructions
    assert "never permission to specialize J" in instructions
    assert "use it in every affected J tensor shape" in instructions
    assert "not SSA values" in instructions
    assert "never list x* in portable scope definitions/captures" in instructions


def test_one_pi_graph_grammar_is_projected_to_schema_lift_and_replay() -> None:
    grammar = render_realization_witness_wire_grammar()
    mechanism_grammar = render_realization_mechanism_grammar()
    schema = scrc_lift_core_json_schema()
    pi_schema = schema["properties"]["pi"]
    lift_instructions = scrc.build_scrc_lift_instructions()
    replay_instructions, _ = build_normal_rewriter_prompt(make_reduce_artifact())

    assert grammar in pi_schema["description"]
    assert grammar in lift_instructions
    assert grammar in replay_instructions
    assert mechanism_grammar in pi_schema["description"]
    assert mechanism_grammar in lift_instructions
    assert mechanism_grammar in replay_instructions
    assert "routing p0 to i1 is invalid" in grammar
    assert "placement='program' only scopes an instance" in mechanism_grammar
    assert "does not require an identity assignment" in mechanism_grammar
    assert "corresponding J domain" in mechanism_grammar
    assert "path enters address_base" in mechanism_grammar
    assert "entering address_lane is lane-varying" in mechanism_grammar
    assert "must not invent program_id" in mechanism_grammar
    assert pi_schema["properties"]["routes"]["items"]["properties"][
        "source"
    ]["pattern"] == "^i[0-9]+$"
    mechanism_schema = pi_schema["properties"]["instances"]["items"][
        "properties"
    ]["mechanism"]
    assert mechanism_schema["enum"] == list(REALIZATION_MECHANISMS)
    assert mechanism_schema["description"] == mechanism_grammar


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


def test_exact_source_symbol_is_selected_before_anonymization() -> None:
    source = """import triton
import triton.language as tl

def helper(first):
    return first

@triton.jit
def selected_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""
    anonymous, binding, mapping = anonymize_source(
        source, source_symbol="selected_kernel"
    )
    assert binding.source_function == "selected_kernel"
    assert mapping["selected_kernel"] == "scrc_k0"
    assert mapping["helper"] == "scrc_k1"
    assert mapping["X"] == "p0"
    assert mapping["Y"] == "p1"
    assert mapping["N"] == "p2"
    assert mapping["BLOCK_SIZE"] == "c0"
    assert "def scrc_k0(p0, p1, p2, c0: tl.constexpr)" in anonymous
    assert "selected_kernel" not in anonymous


def test_bound_constexpr_is_anonymous_p_and_never_pure_c() -> None:
    source = """import triton
import triton.language as tl

@triton.jit
def selected_kernel(X, Y, N, VALID: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < VALID
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""
    view = SCRCCompilerView(
        schedule_values={"BLOCK": 128, "num_warps": 4},
        bound_constexpr_values={"VALID": 17},
        compiler_slots=("num_warps",),
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (17,)),
            SCRCCompilerArgument("Y", "tensor", "f32", (17,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=17),
        ),
        grid=(1,),
        grid_contract=("1",),
        parameter_order=("X", "Y", "N", "VALID", "BLOCK"),
        constexpr_parameters=("VALID", "BLOCK"),
    )
    request = scrc.prepare_scrc_lift(
        source, compiler_view=view, source_symbol="selected_kernel"
    )
    payload = json.loads(request.user_input)
    compiler_view = payload["harness_compiler_view"]
    assert "def scrc_k0(p0, p1, p2, p3: tl.constexpr, c0: tl.constexpr)" in (
        payload["anonymous_source"]
    )
    assert compiler_view["bound_constexpr_values"] == {"p3": 17}
    assert compiler_view["schedule_values"] == {"c0": 128, "c1": 4}
    assert compiler_view["parameter_order"] == ["p0", "p1", "p2", "p3", "c0"]
    assert compiler_view["constexpr_parameters"] == ["p3", "c0"]


def test_compiler_view_rejects_bound_schedule_overlap() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        SCRCCompilerView(
            schedule_values={"N": 8},
            bound_constexpr_values={"N": 8},
            parameter_order=("N",),
            constexpr_parameters=("N",),
        )


def test_lift_keeps_bound_constexpr_outside_c(monkeypatch) -> None:
    response = make_reduce_artifact().to_json()
    response["j"]["signature"]["parameters"].append(
        {
            "name": "p3",
            "kind": "constexpr",
            "mode": "value",
            "dtype": "i32",
            "rank": 0,
            "logical_ref": "x1",
        },
    )
    response["j"]["boundary"]["domains"]["d0"] = "v1 < p3"

    def fake_call(instructions, user, **kwargs):
        del instructions, kwargs
        view = json.loads(user)["harness_compiler_view"]
        assert view["bound_constexpr_values"] == {"p3": 4}
        assert view["schedule_values"] == {"c0": 128}
        return LLMReply(json.dumps(response), "test-model", {"id": "resp-sem"})

    monkeypatch.setattr(scrc, "call_responses", fake_call)
    compiler_view = SCRCCompilerView(
        schedule_values={"BLOCK": 128},
        bound_constexpr_values={"VALID": 4},
        arguments=(
            SCRCCompilerArgument("X", "tensor", "f32", (2, 4)),
            SCRCCompilerArgument("Y", "tensor", "f32", (2,)),
            SCRCCompilerArgument("N", "scalar", "i32", value=4),
        ),
        grid=(1,),
        grid_contract=("1",),
        parameter_order=("X", "Y", "N", "VALID", "BLOCK"),
        constexpr_parameters=("VALID", "BLOCK"),
    )
    source = """@triton.jit
def private_reduce(X, Y, N, VALID: tl.constexpr, BLOCK: tl.constexpr):
    value = tl.load(X)
    tl.store(Y, value)
"""
    extraction = scrc.extract_scrc_artifact(
        source, compiler_view=compiler_view
    )
    assert extraction.artifact.c.values == {"c0": 128}
    assert [
        item.name for item in extraction.artifact.j.signature.parameters
    ] == ["p0", "p1", "p2", "p3"]
    assert extraction.artifact.j.signature.parameters[3].kind.value == "constexpr"


def test_source_symbol_selection_fails_closed() -> None:
    source = "def present(x):\n    return x\n"
    with pytest.raises(ValueError, match="must resolve exactly once"):
        anonymize_source(source, source_symbol="missing")


def test_lift_llm_directly_returns_complete_pi_in_one_call(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
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
    assert extraction.source == source
    assert len(calls) == 1
    assert extraction.provenance.response_id == "resp0"


def test_lift_preserves_trusted_c_and_public_compiler_role(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    response = artifact.to_json()
    response["c"]["slots"].append(
        {
            "name": "c1",
            "kind": "compiler",
            "role": "num_warps",
            "value_type": "int",
            "required": True,
        }
    )
    response["c"]["values"]["c1"] = 4

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
    assert extraction.artifact.c.values == {"c0": 128, "c1": 4}
    assert extraction.artifact.c.slots[1].role == "num_warps"
