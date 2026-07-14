"""Unit tests for Gate1's strict information-separation model."""

from __future__ import annotations

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    CompileCase,
    Frontend,
    Gate1Artifact,
    Gate1Result,
    Gate1Status,
    KernelInterface,
    KernelParameter,
    LeakageKind,
    LeakagePolicy,
    LocationEntry,
    LocationMap,
    ParameterKind,
    ParameterMode,
    ReductionBoundary,
    ReplayPlan,
    ReplayStep,
    RuleCard,
    RuleRegistry,
    RuntimeArgument,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SchemaError,
    SemanticArchive,
    SourceSpan,
    build_masked_rewriter_prompt,
    build_normal_rewriter_prompt,
    leakage_ratio,
    masked_ablation_passes,
    scan_artifact_leakage,
    scan_static_leakage,
)


def _artifact() -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "row_sum_semantics",
            "inputs": [{"name": "input_tensor", "dtype": "f32", "shape": ["M", "N"]}],
            "outputs": [{"name": "output_tensor", "dtype": "f32", "shape": ["M"]}],
            "body": [
                {
                    "op": "reduce",
                    "result": "row_sum_value",
                    "operands": ["input_tensor"],
                    "attrs": {"fn": "sum", "axes": [1]},
                }
            ],
        }
    )
    j = SemanticArchive(
        kernel_id="row_sum_kernel",
        intent=intent,
        node_ids={"reduce_sum_node": "body[0]"},
        boundary=Boundary(
            domains={"column_domain": "column_offset < N"},
            accesses=[
                BoundaryAccess(
                    access_id="input_masked_load",
                    kind=AccessKind.LOAD,
                    tensor="input_tensor",
                    index_expression="row_offset * N + column_offset",
                    predicate="column_offset < N",
                    fill_value=0,
                    node_ref="reduce_sum_node",
                ),
                BoundaryAccess(
                    access_id="output_guarded_store",
                    kind=AccessKind.STORE,
                    tensor="output_tensor",
                    index_expression="row_offset",
                    predicate="row_offset < M",
                    node_ref="reduce_sum_node",
                ),
            ],
            reductions=[
                ReductionBoundary(
                    node_ref="reduce_sum_node",
                    tile_axis="N",
                    identity=0,
                    masked_contribution=0,
                )
            ],
        ),
    )
    interface = KernelInterface(
        kernel_id="row_sum_kernel",
        frontend=Frontend.TRITON,
        parameters=[
            KernelParameter(
                "input_tensor", ParameterKind.POINTER, ParameterMode.INPUT, "f32", 2,
                tensor_ref="input_tensor",
            ),
            KernelParameter(
                "output_tensor", ParameterKind.POINTER, ParameterMode.OUTPUT, "f32", 1,
                tensor_ref="output_tensor",
            ),
            KernelParameter("M", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
            KernelParameter("N", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
            KernelParameter("BLOCK_N", ParameterKind.CONSTEXPR),
        ],
        grid=["ceil_div(M, 1)"],
        compiler_slots=["num_warps"],
        function_name="row_sum_impl",
        do_not_specialize=["M", "N"],
    )
    schedule = ScheduleSpec(
        slots=[
            ScheduleSlot("BLOCK_N", ScheduleSlotKind.CONSTEXPR),
            ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
        ],
        values={"BLOCK_N": 128, "num_warps": 4},
    )
    plan = ReplayPlan(
        [
            ReplayStep("bind_program_domain", refs=["column_domain"]),
            ReplayStep("masked_load", refs=["input_masked_load"]),
            ReplayStep("native_reduce", node_ref="reduce_sum_node"),
            ReplayStep("masked_store", refs=["output_guarded_store"]),
        ]
    )
    location_map = LocationMap(
        source_name="private_kernel.py",
        entries=[
            LocationEntry(
                SourceSpan(10, 12),
                node_refs=["reduce_sum_node"],
                access_refs=["input_masked_load"],
            )
        ],
    )
    return Gate1Artifact(
        j=j,
        interface=interface,
        schedule=schedule,
        plan=plan,
        location_map=location_map,
        source="TOP_SECRET_SOURCE = tl.sum(tl.load(pointer))",
    )


def _registry() -> RuleRegistry:
    return RuleRegistry(
        [
            RuleCard("bind_program_domain", Frontend.TRITON, example="pid = tl.program_id(0)"),
            RuleCard("masked_load", Frontend.TRITON, api_mapping={"load": "tl.load"}),
            RuleCard(
                "native_reduce", Frontend.TRITON,
                applies_to=["reduce"], reads=["fn", "axes", "operand"],
                api_mapping={"sum": "tl.sum", "max": "tl.max"},
            ),
            RuleCard("masked_store", Frontend.TRITON, api_mapping={"store": "tl.store"}),
        ]
    )


def test_artifact_strict_json_roundtrip() -> None:
    artifact = _artifact()
    assert Gate1Artifact.from_json(artifact.to_json()) == artifact
    assert Gate1Artifact.from_json_str(artifact.to_json_str()) == artifact

    result = Gate1Result(
        Gate1Status.REPLAY_CLOSED,
        artifact=artifact,
        diagnostics=["closed"],
        relift_attempts=1,
        repair_attempts=0,
        leakage_ratio=0.25,
    )
    assert Gate1Result.from_json_str(result.to_json_str()) == result

    case = CompileCase(
        arguments=[RuntimeArgument("M", value=32, dtype="i32")],
        grid=[32],
    )
    assert CompileCase.from_json_str(case.to_json_str()) == case


def test_reduction_boundary_rejects_c_slot_as_tile_axis() -> None:
    payload = _artifact().to_json()
    payload["j"]["boundary"]["reductions"][0]["tile_axis"] = "BLOCK_N"

    with pytest.raises(
        SchemaError, match="tile_axis must not reference C schedule slot"
    ):
        Gate1Artifact.from_json(payload)


def test_schedule_constexpr_must_exist_in_interface_signature() -> None:
    payload = _artifact().to_json()
    payload["interface"]["parameters"] = [
        item
        for item in payload["interface"]["parameters"]
        if item["name"] != "BLOCK_N"
    ]

    with pytest.raises(
        SchemaError, match="constexpr slots are absent from KernelInterface"
    ):
        Gate1Artifact.from_json(payload)


def test_semantic_scalar_tensor_ref_requires_scalar_j_input_and_matching_dtype() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "eps", "dtype": "f32", "shape": []}
    )
    payload["interface"]["parameters"].insert(
        -1,
        {
            "name": "eps",
            "kind": "scalar",
            "mode": "value",
            "dtype": "float32",
            "rank": 0,
            "tensor_ref": "eps",
        },
    )
    parsed = Gate1Artifact.from_json(payload)
    assert parsed.interface.parameters[-2].tensor_ref == "eps"

    bad_shape = _artifact().to_json()
    bad_shape["interface"]["parameters"][2]["tensor_ref"] = "input_tensor"
    with pytest.raises(SchemaError, match=r"tensor_ref must have shape=\[\]"):
        Gate1Artifact.from_json(bad_shape)

    bad_dtype = payload
    bad_dtype["j"]["intent"]["intent_function"]["inputs"][-1]["dtype"] = "f16"
    with pytest.raises(SchemaError, match="tensor_ref dtype must match"):
        Gate1Artifact.from_json(bad_dtype)

    pointer_dtype = _artifact().to_json()
    pointer_dtype["j"]["intent"]["intent_function"]["inputs"][0]["dtype"] = "f16"
    with pytest.raises(SchemaError, match="tensor_ref dtype must match"):
        Gate1Artifact.from_json(pointer_dtype)


def test_unknown_keys_are_rejected_at_every_closed_boundary() -> None:
    payload = _artifact().to_json()
    payload["surprise"] = True
    with pytest.raises(SchemaError, match="unknown key"):
        Gate1Artifact.from_json(payload)

    payload = _artifact().to_json()
    payload["plan"]["steps"][0]["mechanism"] = "program_id"
    with pytest.raises(SchemaError, match="mechanism"):
        Gate1Artifact.from_json(payload)

    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0]["typo_operands"] = []
    with pytest.raises(SchemaError, match="typo_operands"):
        Gate1Artifact.from_json(payload)


def test_pi_schema_has_only_rule_node_ref_and_refs() -> None:
    for step in _artifact().plan.to_json()["steps"]:
        assert set(step) <= {"rule", "node_ref", "refs"}
    with pytest.raises(SchemaError, match="unknown key"):
        ReplayStep.from_json(
            {"rule": "native_reduce", "node_ref": "n0", "action": "tl.sum(x)"}
        )


def test_rewriter_view_excludes_source_locations_and_c_values() -> None:
    artifact = _artifact()
    view = artifact.rewriter_view()
    text = str(view)
    assert "TOP_SECRET_SOURCE" not in text
    assert "private_kernel.py" not in text
    assert "location_map" not in view
    assert "source" not in view
    assert "values" not in view["schedule"]
    assert view["schedule"]["slots"][0]["name"] == "BLOCK_N"
    assert view["interface"]["function_name"] == "row_sum_impl"
    assert "do_not_specialize" not in view["interface"]
    assert view["j"]["boundary"]["accesses"][0]["index_expression"] == (
        "row_offset * N + column_offset"
    )
    assert view["j"]["boundary"]["reductions"][0]["identity"] == 0


def test_masked_view_uses_consistent_opaque_identifiers() -> None:
    view = _artifact().rewriter_view(masked=True)
    masked_j_and_plan = str({"j": view["j"], "plan": view["plan"]})
    for secret in (
        "row_sum_semantics", "input_tensor", "output_tensor",
        "reduce_sum_node", "input_masked_load", "output_guarded_store",
        "column_domain", "column_offset < N",
        "row_offset * N + column_offset",
    ):
        assert secret not in masked_j_and_plan
    assert view["j"]["kernel_id"] == "scrc_k0"
    assert view["plan"]["steps"][2]["node_ref"] == "n0"
    assert view["j"]["boundary"]["accesses"][0]["access_id"] == "a0"
    # ABI/C structure remains reusable, but source spellings are opaque and the
    # replay adapter restores them deterministically before compilation.
    assert view["interface"]["kernel_id"] == "scrc_k0"
    assert view["interface"]["function_name"] == "scrc_k0"
    assert [item["name"] for item in view["interface"]["parameters"]] == [
        "scrc_p0", "scrc_p1", "scrc_p2", "scrc_p3", "scrc_p4",
    ]
    assert view["interface"]["grid"] == ["ceil_div(scrc_p2, 1)"]
    assert view["interface"]["compiler_slots"] == ["scrc_c0"]
    assert "do_not_specialize" not in view["interface"]
    assert [slot["name"] for slot in view["schedule"]["slots"]] == [
        "scrc_p4", "scrc_c0",
    ]
    assert "values" not in view["schedule"]
    assert view["interface"]["parameters"][0]["dtype"] == "f32"
    assert view["interface"]["parameters"][0]["tensor_ref"] == "t0"
    assert view["j"]["intent"]["intent_function"]["body"][0]["op"] == "<masked>"


def test_address_and_reduction_boundary_fields_are_mandatory() -> None:
    with pytest.raises(SchemaError, match="index_expression"):
        BoundaryAccess("a0", AccessKind.LOAD, "x")
    with pytest.raises(SchemaError, match="identity or masked_contribution"):
        ReductionBoundary("n0", 0)


def test_rule_registry_resolves_only_cards_used_by_plan() -> None:
    registry = _registry()
    plan = _artifact().plan
    cards = registry.resolve(plan, frontend=Frontend.TRITON)
    assert [card.id for card in cards] == [
        "bind_program_domain", "masked_load", "native_reduce", "masked_store",
    ]
    retrieval = registry.rewriter_view(plan, frontend=Frontend.TRITON)
    assert [card["id"] for card in retrieval["cards"]] == [
        card.id for card in cards
    ]
    assert [item["rule"] for item in retrieval["selection"]] == [
        step.rule for step in plan.steps
    ]
    assert RuleRegistry.from_json_str(registry.to_json_str()).to_json() == registry.to_json()
    with pytest.raises(SchemaError, match="duplicate"):
        RuleRegistry([cards[0], cards[0]])
    with pytest.raises(SchemaError, match="unknown replay rule"):
        registry.resolve(ReplayPlan([ReplayStep("missing", node_ref="n0")]))


def test_static_leakage_scan_catches_api_source_ngram_and_budget() -> None:
    bad = ReplayPlan(
        [
            ReplayStep("tl.load", node_ref="n0"),
            ReplayStep("alpha = beta + gamma * delta", refs=["n0", "n1"]),
        ]
    )
    report = scan_static_leakage(
        bad,
        source="value = alpha = beta + gamma * delta; return value",
        policy=LeakagePolicy(max_literal_chars=12, max_total_literal_chars=40),
    )
    kinds = {violation.kind for violation in report.violations}
    assert not report.passed
    assert LeakageKind.API_TOKEN in kinds
    assert LeakageKind.SOURCE_NGRAM in kinds
    assert LeakageKind.BUDGET in kinds

    malformed = scan_static_leakage(
        {"steps": [{"rule": "x", "node_ref": "n0", "semantic_hint": "sum"}]}
    )
    assert malformed.violations[0].kind is LeakageKind.SCHEMA


def test_source_ngram_scan_targets_pi_not_semantic_boundary_expressions() -> None:
    artifact = _artifact()
    source = (
        "address = row_offset * N + column_offset\n"
        "valid = column_offset < N"
    )
    clean = scan_static_leakage(
        ReplayPlan([ReplayStep("masked_load", refs=["input_masked_load"])]),
        source=source,
        j=artifact.j,
    )
    assert not any(
        item.kind is LeakageKind.SOURCE_NGRAM for item in clean.violations
    )

    leaked = scan_static_leakage(
        ReplayPlan(
            [
                ReplayStep(
                    "row_offset * N + column_offset",
                    refs=["input_masked_load"],
                )
            ]
        ),
        source=source,
        j=artifact.j,
    )
    assert any(
        item.kind is LeakageKind.SOURCE_NGRAM for item in leaked.violations
    )


def test_artifact_scan_and_prompts_preserve_isolation() -> None:
    artifact = _artifact()
    registry = _registry()
    report = scan_artifact_leakage(artifact, registry=registry)
    assert report.passed

    normal = build_normal_rewriter_prompt(artifact, registry)
    masked = build_masked_rewriter_prompt(artifact, registry)
    assert "TOP_SECRET_SOURCE" not in normal
    assert "private_kernel.py" not in normal
    assert '"values"' not in normal
    assert "BLOCK_N" in normal
    assert "128" not in normal
    assert "tl.sum" in normal  # concrete API names live in selected system rule cards
    assert "row_sum_kernel" not in masked
    assert "row_sum_impl" not in masked
    assert "BLOCK_N" not in masked
    assert '"function_name": "scrc_k0"' in masked
    assert "scrc_p4" in masked
    assert "tl.sum" in masked  # masked-J retains the same pi/rule-card channel
    assert "reduce_sum_node" not in masked
    assert "ABLATION RUN" in masked
    assert "interface.compiler_slots are launch-only" in masked


def test_leakage_ratio_has_fixed_threshold_semantics() -> None:
    assert leakage_ratio(1.0, 0.5) == 0.5
    assert masked_ablation_passes(1.0, 0.5)
    assert not masked_ablation_passes(1.0, 0.5001)
    assert leakage_ratio(0.0, 0.0) == 0.0
