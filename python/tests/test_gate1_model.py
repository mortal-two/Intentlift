"""Unit tests for Gate1's strict information-separation model."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    CompileCase,
    Frontend,
    Gate1Artifact,
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
    RuleSelectionMode,
    RuntimeArgument,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SchemaError,
    SemanticArchive,
    SourceSpan,
    StrategyBinding,
    StrategyChoice,
    StrategyFamily,
    StrategySpec,
    ComposeMismatchError,
    assert_replay_plan_matches,
    assert_source_isolated_payload,
    build_masked_rewriter_view,
    build_masked_rewriter_prompt,
    build_pi_ablations,
    build_normal_rewriter_prompt,
    build_normal_rewriter_view,
    build_rewriter_payload,
    build_rewriter_prompt,
    compose_replay_plan,
    default_triton_rule_registry,
    materialize_replay_plan,
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
            ReplayStep("bind_wide_program_domain", refs=["column_domain"]),
            ReplayStep("bind_runtime_scalar", refs=["M"]),
            ReplayStep("bind_runtime_scalar", refs=["N"]),
            ReplayStep("declare_schedule_slot", refs=["BLOCK_N"]),
            ReplayStep("declare_schedule_slot", refs=["num_warps"]),
            ReplayStep("materialize_index_tile", refs=["input_masked_load"]),
            ReplayStep("realize_masked_access", refs=["input_masked_load"]),
            ReplayStep("realize_reduce", node_ref="reduce_sum_node"),
            ReplayStep("materialize_index_tile", refs=["output_guarded_store"]),
            ReplayStep("realize_masked_access", refs=["output_guarded_store"]),
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
    return default_triton_rule_registry()


def test_artifact_strict_json_roundtrip() -> None:
    artifact = _artifact()
    assert Gate1Artifact.from_json(artifact.to_json()) == artifact
    assert Gate1Artifact.from_json_str(artifact.to_json_str()) == artifact

    case = CompileCase(
        arguments=[RuntimeArgument("M", value=32, dtype="i32")],
        grid=[32],
    )
    assert CompileCase.from_json_str(case.to_json_str()) == case


def test_compile_time_inactive_input_remains_in_complete_static_abi() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["inputs"].append(
        {"name": "optional_label", "dtype": "f32", "shape": [1]}
    )
    payload["interface"]["parameters"].insert(
        1,
        {
            "name": "label_ptr",
            "kind": "pointer",
            "mode": "input",
            "dtype": "f32",
            "rank": 1,
            "tensor_ref": "optional_label",
        },
    )

    artifact = Gate1Artifact.from_json(payload)

    assert artifact.interface.parameters[1].mode is ParameterMode.INPUT
    assert not any(
        access.tensor == "optional_label"
        for access in artifact.j.boundary.accesses
    )


def test_strategy_is_closed_override_only_c_and_legacy_defaults_empty() -> None:
    artifact = _artifact()
    assert artifact.strategy == StrategySpec()
    assert artifact.to_json()["strategy"] == {"bindings": []}

    legacy = artifact.to_json()
    legacy.pop("strategy")
    assert Gate1Artifact.from_json(legacy).strategy == StrategySpec()

    strategy = StrategySpec(
        [
            StrategyBinding(
                "column_domain",
                StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                StrategyChoice.WIDE,
            ),
            StrategyBinding(
                "input_masked_load",
                StrategyFamily.ADDRESS_ASSOCIATION,
                StrategyChoice.POINTER_STAGED,
            ),
            StrategyBinding(
                "input_masked_load",
                StrategyFamily.MASK_MATERIALIZATION,
                StrategyChoice.STAGED_REUSE,
            ),
            StrategyBinding(
                "reduce_sum_node",
                StrategyFamily.REDUCTION_CALL_SHAPE,
                StrategyChoice.FLATTENED,
            ),
        ]
    )
    updated = replace(artifact, strategy=strategy)
    assert Gate1Artifact.from_json(updated.to_json()) == updated

    tiled_strategy = StrategySpec(
        [
            StrategyBinding(
                "reduce_sum_node",
                StrategyFamily.REDUCTION_TILING,
                StrategyChoice.MULTI_TILE_ACCUMULATE,
                tile_slot="BLOCK_N",
                accumulator_dtype="f32",
            )
        ]
    )
    tiled = replace(artifact, strategy=tiled_strategy)
    assert Gate1Artifact.from_json(tiled.to_json()) == tiled
    assert tiled.strategy.bindings[0].to_json() == {
        "target_ref": "reduce_sum_node",
        "family": "reduction_tiling",
        "choice": "multi_tile_accumulate",
        "tile_slot": "BLOCK_N",
        "accumulator_dtype": "f32",
    }

    with pytest.raises(SchemaError, match="requires choice 'wide'"):
        StrategyBinding(
            "column_domain",
            StrategyFamily.PROGRAM_COORDINATE_WIDTH,
            StrategyChoice.POINTER_STAGED,
        )
    with pytest.raises(SchemaError, match="duplicate binding"):
        StrategySpec([strategy.bindings[0], strategy.bindings[0]])
    with pytest.raises(SchemaError, match="tile_slot"):
        StrategyBinding(
            "reduce_sum_node",
            StrategyFamily.REDUCTION_TILING,
            StrategyChoice.MULTI_TILE_ACCUMULATE,
        )
    with pytest.raises(SchemaError, match="canonical IntentIR dtype"):
        replace(
            tiled_strategy.bindings[0],
            accumulator_dtype="source_accumulator_name",
        )
    with pytest.raises(SchemaError, match="valid only for reduction_tiling"):
        StrategyBinding(
            "column_domain",
            StrategyFamily.PROGRAM_COORDINATE_WIDTH,
            StrategyChoice.WIDE,
            tile_slot="BLOCK_N",
            accumulator_dtype="f32",
        )
    with pytest.raises(SchemaError, match="cannot combine"):
        StrategySpec(
            [
                strategy.bindings[-1],
                tiled_strategy.bindings[0],
            ]
        )
    with pytest.raises(SchemaError, match="not declared in C.schedule"):
        replace(
            artifact,
            strategy=StrategySpec(
                [
                    replace(
                        tiled_strategy.bindings[0],
                        tile_slot="UNKNOWN_TILE",
                    )
                ]
            ),
        )
    with pytest.raises(SchemaError, match="must name a constexpr"):
        replace(
            artifact,
            strategy=StrategySpec(
                [
                    replace(
                        tiled_strategy.bindings[0],
                        tile_slot="num_warps",
                    )
                ]
            ),
        )
    with pytest.raises(SchemaError, match="is not valid for family"):
        replace(
            artifact,
            strategy=StrategySpec(
                [
                    StrategyBinding(
                        "column_domain",
                        StrategyFamily.ADDRESS_ASSOCIATION,
                        StrategyChoice.POINTER_STAGED,
                    )
                ]
            ),
        )


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


@pytest.mark.parametrize(
    "predicate",
    [
        "b = grid_axis(0), 0 <= b < 2048",
        "row = grid_coordinate(0) where 0 <= row < 2048",
        "row := coordinate(grid_axis=0) with 0 <= row < 2048",
        "program_id(0) < 1",
        "row + 1",
    ],
)
def test_gate1_artifact_rejects_nonportable_domain_predicates(
    predicate: str,
) -> None:
    payload = _artifact().to_json()
    payload["j"]["boundary"]["domains"]["column_domain"] = predicate

    with pytest.raises(
        SchemaError,
        match=r"Boundary\.domains\.column_domain must be a portable boolean",
    ):
        Gate1Artifact.from_json(payload)

    assert payload["j"]["boundary"]["domains"]["column_domain"] == predicate


def test_gate1_artifact_rejects_nonportable_access_predicate() -> None:
    payload = _artifact().to_json()
    invalid = "row := coordinate(grid_axis=0) with row < M"
    payload["j"]["boundary"]["accesses"][0]["predicate"] = invalid

    with pytest.raises(
        SchemaError,
        match=r"BoundaryAccess\.predicate must be a portable boolean",
    ):
        Gate1Artifact.from_json(payload)

    assert payload["j"]["boundary"]["accesses"][0]["predicate"] == invalid


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
        "v2 * p3 + v1"
    )
    assert view["j"]["boundary"]["reductions"][0]["identity"] == 0
    assert view["reference_naming"] == "opaque_replay_refs_v1"
    assert view["j"]["node_ids"] == {"n0": "body[0]"}
    assert list(view["j"]["boundary"]["domains"]) == ["d0"]
    assert [
        access["access_id"] for access in view["j"]["boundary"]["accesses"]
    ] == ["a0", "a1"]
    assert [step.get("node_ref") or step["refs"][0] for step in view["plan"]["steps"]] == [
        "d0",
        "p2",
        "p3",
        "p4",
        "c0",
        "a0",
        "a0",
        "n0",
        "a1",
        "a1",
    ]
    semantic_view = str({"j": view["j"], "plan": view["plan"]})
    for source_derived_name in (
        "input_tensor",
        "output_tensor",
        "reduce_sum_node",
        "input_masked_load",
        "output_guarded_store",
        "column_domain",
        "row_offset",
        "column_offset",
    ):
        assert source_derived_name not in semantic_view
    p3 = next(
        binding for binding in view["reference_bindings"] if binding["ref"] == "p3"
    )
    assert p3["owners"] == [
        {"kind": "interface_parameter", "path": "interface.parameters[3]"}
    ]


def test_normal_payload_opaque_refs_match_rule_selection_and_fail_closed() -> None:
    payload = build_normal_rewriter_view(
        _artifact(), default_triton_rule_registry()
    )
    plan_targets = [
        step.get("node_ref") or step["refs"][0]
        for step in payload["plan"]["steps"]
    ]
    selection_targets = [
        item["target_ref"] for item in payload["rule_selection"]
    ]

    assert plan_targets == selection_targets
    assert set(plan_targets) <= {
        binding["ref"] for binding in payload["reference_bindings"]
    }
    opaque_channels = str(
        {
            "j": payload["j"],
            "plan": payload["plan"],
            "selection": payload["rule_selection"],
            "bindings": payload["reference_bindings"],
        }
    )
    for leaked_name in (
        "reduce_sum_node",
        "input_masked_load",
        "output_guarded_store",
        "column_domain",
        "row_sum_value",
    ):
        assert leaked_name not in opaque_channels

    payload["plan"]["steps"][0]["refs"] = ["column_domain"]
    with pytest.raises(SchemaError, match="declared opaque replay reference"):
        assert_source_isolated_payload(payload)


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
    assert [step["rule"] for step in view["plan"]["steps"]] == [
        "bind_wide_program_domain",
        "bind_runtime_scalar",
        "bind_runtime_scalar",
        "declare_schedule_slot",
        "declare_schedule_slot",
        "materialize_index_tile",
        "realize_masked_access",
        "realize_reduce",
        "materialize_index_tile",
        "realize_masked_access",
    ]
    assert view["plan"]["steps"][0]["refs"] == ["d0"]
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
        "bind_wide_program_domain",
        "bind_runtime_scalar",
        "declare_schedule_slot",
        "materialize_index_tile",
        "realize_masked_access",
        "realize_reduce",
    ]
    retrieval = registry.rewriter_view(plan, frontend=Frontend.TRITON)
    assert [card["id"] for card in retrieval["cards"]] == [
        "bind_wide_program_domain",
        "bind_runtime_scalar",
        "declare_schedule_slot",
        "materialize_index_tile",
        "realize_masked_access",
        "realize_reduce",
    ]
    assert [item["rule"] for item in retrieval["selection"]] == [
        step.rule for step in plan.steps
    ]
    assert retrieval["selection"][0]["selection_mode"] == "override"
    assert retrieval["selection"][0]["override_family"] == (
        "program_coordinate_width"
    )
    assert retrieval["selection"][0]["target_ref"] == "column_domain"
    assert RuleRegistry.from_json_str(registry.to_json_str()).to_json() == registry.to_json()
    with pytest.raises(SchemaError, match="duplicate"):
        RuleRegistry([cards[0], cards[0]])
    with pytest.raises(SchemaError, match="unknown replay rule"):
        registry.resolve(ReplayPlan([ReplayStep("missing", node_ref="n0")]))


def test_rule_registry_snapshot_is_order_independent_and_detached_from_callers() -> None:
    registry = default_triton_rule_registry()
    reordered = RuleRegistry(list(reversed(registry.cards)))

    assert reordered.digest == registry.digest
    assert reordered.to_json() == registry.to_json()

    fetched = registry.get("realize_elementwise")
    fetched.api_mapping["unreviewed"] = "libdevice.unreviewed"
    exposed = registry.cards[0]
    exposed.pitfalls.append("caller mutation")

    assert "unreviewed" not in registry.get("realize_elementwise").api_mapping
    assert "caller mutation" not in registry.cards[0].pitfalls

    frozen = registry.frozen_snapshot()
    assert frozen.frozen
    with pytest.raises(SchemaError, match="frozen RuleRegistry"):
        frozen.register(
            RuleCard(
                "late_hot_reload",
                Frontend.TRITON,
                target_kinds=["intent_op:late"],
                construction_phase=60,
            )
        )


def test_rule_cards_serialize_executable_composer_metadata_compatibly() -> None:
    registry = default_triton_rule_registry()
    assert all(card.target_kinds for card in registry.cards)
    assert all(card.construction_phase is not None for card in registry.cards)

    alternatives = {
        card.id: (card.strategy_family, card.strategy_choice)
        for card in registry.cards
        if card.strategy_family is not None
    }
    assert alternatives == {
        "bind_wide_program_domain": ("program_coordinate_width", "wide"),
        "materialize_pointer_tile": (
            "address_association",
            "pointer_staged",
        ),
        "materialize_mask_tile": (
            "mask_materialization",
            "staged_reuse",
        ),
        "realize_flattened_reduce": (
            "reduction_call_shape",
            "flattened",
        ),
        "realize_multi_tile_reduce": (
            "reduction_tiling",
            "multi_tile_accumulate",
        ),
        "stage_reduction_preheader": (
            "reduction_preheader_staging",
            "stage_invariant_access_scaffolding",
        ),
    }
    cards = {card.id: card for card in registry.cards}
    assert cards["realize_masked_access"].required_fields == ["predicate"]
    assert cards["realize_unmasked_access"].required_fields == ["!predicate"]
    assert set(registry.to_json()["cards"][0]) >= {
        "target_kinds",
        "required_fields",
        "provides",
        "requires",
        "construction_phase",
    }

    legacy = RuleCard.from_json({"id": "legacy", "frontend": "triton"})
    assert legacy.target_kinds == []
    assert legacy.construction_phase is None
    assert legacy.required_fields == []
    assert legacy.provides == []
    assert legacy.requires == []
    with pytest.raises(SchemaError, match="declared together"):
        RuleCard(
            "broken",
            Frontend.TRITON,
            strategy_family="address_association",
        )


def test_multi_tile_reduction_strategy_selects_loop_accumulation_rule() -> None:
    artifact = replace(
        _artifact(),
        plan=ReplayPlan([]),
        strategy=StrategySpec(
            [
                StrategyBinding(
                    "reduce_sum_node",
                    StrategyFamily.REDUCTION_TILING,
                    StrategyChoice.MULTI_TILE_ACCUMULATE,
                    tile_slot="BLOCK_N",
                    accumulator_dtype="f32",
                )
            ]
        ),
    )
    with pytest.raises(ComposeMismatchError):
        assert_replay_plan_matches(
            replace(artifact, plan=_artifact().plan),
            default_triton_rule_registry(),
        )

    materialized, result = materialize_replay_plan(
        artifact, default_triton_rule_registry()
    )
    reduce_steps = [
        step
        for step in materialized.plan.steps
        if step.target_ref == "reduce_sum_node"
    ]

    assert [step.rule for step in reduce_steps] == ["realize_multi_tile_reduce"]
    assert result.provenance.coverage == 1.0
    changed_dtype = compose_replay_plan(
        replace(
            artifact,
            strategy=StrategySpec(
                [
                    replace(
                        artifact.strategy.bindings[0],
                        accumulator_dtype="f64",
                    )
                ]
            ),
        ),
        default_triton_rule_registry(),
    )
    assert changed_dtype.plan.to_json() == result.plan.to_json()
    assert (
        changed_dtype.provenance.strategy_digest
        != result.provenance.strategy_digest
    )

    dual_slot_payload = artifact.to_json()
    dual_slot_payload["interface"]["parameters"].append(
        {"name": "BLOCK_ALT", "kind": "constexpr", "mode": "value"}
    )
    dual_slot_payload["schedule"]["slots"].append(
        {
            "name": "BLOCK_ALT",
            "kind": "constexpr",
            "value_type": "int",
            "required": True,
        }
    )
    dual_slot_payload["schedule"]["values"]["BLOCK_ALT"] = 64
    dual_slot = Gate1Artifact.from_json(dual_slot_payload)
    alternate_slot = replace(
        dual_slot,
        strategy=StrategySpec(
            [replace(dual_slot.strategy.bindings[0], tile_slot="BLOCK_ALT")]
        ),
    )
    dual_slot_result = compose_replay_plan(
        dual_slot, default_triton_rule_registry()
    )
    alternate_slot_result = compose_replay_plan(
        alternate_slot, default_triton_rule_registry()
    )
    assert alternate_slot_result.plan.to_json() == dual_slot_result.plan.to_json()
    assert (
        alternate_slot_result.provenance.strategy_digest
        != dual_slot_result.provenance.strategy_digest
    )

    renamed_payload = artifact.to_json()
    for parameter in renamed_payload["interface"]["parameters"]:
        if parameter["name"] == "BLOCK_N":
            parameter["name"] = "TILE_WIDTH"
    renamed_payload["schedule"]["slots"][0]["name"] = "TILE_WIDTH"
    renamed_payload["schedule"]["values"]["TILE_WIDTH"] = (
        renamed_payload["schedule"]["values"].pop("BLOCK_N")
    )
    renamed_payload["strategy"]["bindings"][0]["tile_slot"] = "TILE_WIDTH"
    renamed_result = compose_replay_plan(
        Gate1Artifact.from_json(renamed_payload),
        default_triton_rule_registry(),
    )
    assert renamed_result.plan_digest == result.plan_digest
    assert (
        renamed_result.provenance.strategy_digest
        == result.provenance.strategy_digest
    )
    card = default_triton_rule_registry().get("realize_multi_tile_reduce")
    assert "C.strategy.tile_slot" in card.api_mapping["loop"]
    assert "convert each local partial" in card.api_mapping["carry"]
    assert "divide exactly once" in card.api_mapping["mean"]
    assert any("never add an IntentIR loop" in item for item in card.pitfalls)
    assert any("lower=0" in item for item in card.pitfalls)
    assert any(
        "iteration-dependent tile offsets" in item for item in card.pitfalls
    )

    replay_view = build_normal_rewriter_view(
        materialized, default_triton_rule_registry()
    )
    assert replay_view["strategy"] == {
        "bindings": [
            {
                "target_ref": "n0",
                "family": "reduction_tiling",
                "choice": "multi_tile_accumulate",
                "tile_slot": "p4",
                "accumulator_dtype": "f32",
            }
        ]
    }
    known_refs = {
        binding["ref"] for binding in replay_view["reference_bindings"]
    }
    assert {"n0", "p4"} <= known_refs

    masked_view = build_masked_rewriter_view(
        materialized, default_triton_rule_registry()
    )
    assert masked_view["strategy"] == {
        "bindings": [
            {
                "target_ref": "n0",
                "family": "reduction_tiling",
                "choice": "multi_tile_accumulate",
                "tile_slot": "scrc_p4",
                "accumulator_dtype": "f32",
            }
        ]
    }

    replay_view["strategy"]["bindings"][0]["tile_slot"] = "BLOCK_N"
    with pytest.raises(SchemaError, match="declared opaque replay reference"):
        assert_source_isolated_payload(replay_view)


def test_reduction_preheader_staging_is_typed_orthogonal_c_strategy() -> None:
    tiling = StrategyBinding(
        "reduce_sum_node",
        StrategyFamily.REDUCTION_TILING,
        StrategyChoice.MULTI_TILE_ACCUMULATE,
        tile_slot="BLOCK_N",
        accumulator_dtype="f32",
    )
    staging = StrategyBinding(
        "reduce_sum_node",
        StrategyFamily.REDUCTION_PREHEADER_STAGING,
        StrategyChoice.STAGE_INVARIANT_ACCESS_SCAFFOLDING,
    )
    with pytest.raises(SchemaError, match="requires reduction_tiling"):
        StrategySpec([staging])

    artifact = replace(
        _artifact(),
        plan=ReplayPlan([]),
        strategy=StrategySpec([tiling, staging]),
    )
    materialized, _result = materialize_replay_plan(
        artifact, default_triton_rule_registry()
    )
    reduction_steps = [
        step.rule
        for step in materialized.plan.steps
        if step.target_ref == "reduce_sum_node"
    ]

    assert reduction_steps == [
        "stage_reduction_preheader",
        "realize_multi_tile_reduce",
    ]
    replay_view = build_normal_rewriter_view(
        materialized, default_triton_rule_registry()
    )
    assert {
        item["family"] for item in replay_view["strategy"]["bindings"]
    } == {"reduction_preheader_staging", "reduction_tiling"}

    invalid_target = "input_masked_load"
    with pytest.raises(SchemaError, match="is not valid for family"):
        replace(
            _artifact(),
            strategy=StrategySpec(
                [
                    replace(tiling, target_ref=invalid_target),
                    replace(staging, target_ref=invalid_target),
                ]
            ),
        )


def test_strict_pi_accepts_canonical_rules_and_requires_targets() -> None:
    registry = default_triton_rule_registry()
    parsed = registry.validate_strict_plan(
        ReplayPlan(
            [ReplayStep("realize_elementwise", node_ref="reduce_sum_node")]
        ),
        frontend=Frontend.TRITON,
    )
    assert parsed.steps[0].rule == "realize_elementwise"
    with pytest.raises(SchemaError, match="requires a target reference"):
        registry.validate_strict_plan(
            ReplayPlan([ReplayStep("bind_wide_program_domain")]),
            frontend=Frontend.TRITON,
        )


def test_pi_rejects_duplicate_and_conflicting_family_overrides() -> None:
    registry = RuleRegistry(
        [
            RuleCard(
                "wide_i64",
                Frontend.TRITON,
                selection_mode=RuleSelectionMode.OVERRIDE,
                override_family="coordinate_width",
            ),
            RuleCard(
                "wide_i128",
                Frontend.TRITON,
                selection_mode=RuleSelectionMode.OVERRIDE,
                override_family="coordinate_width",
            ),
        ]
    )
    duplicate = ReplayPlan(
        [
            ReplayStep("wide_i64", node_ref="column_domain"),
            ReplayStep("wide_i64", node_ref="column_domain"),
        ]
    )
    with pytest.raises(SchemaError, match="duplicate replay rule"):
        registry.validate_strict_plan(duplicate)

    conflict = ReplayPlan(
        [
            ReplayStep("wide_i64", node_ref="column_domain"),
            ReplayStep("wide_i128", node_ref="column_domain"),
        ]
    )
    with pytest.raises(SchemaError, match="conflicting pi override"):
        registry.validate_strict_plan(conflict)


def test_strict_pi_rejects_wrong_target_and_mutually_exclusive_mechanisms() -> None:
    registry = default_triton_rule_registry()
    artifact = _artifact()

    wrong_target = ReplayPlan(
        [
            ReplayStep("realize_const", node_ref="reduce_sum_node")
            if step.rule == "realize_reduce"
            else step
            for step in artifact.plan.steps
        ]
    )
    with pytest.raises(SchemaError, match="is not applicable to target"):
        registry.validate_strict_plan(
            wrong_target, frontend=Frontend.TRITON, artifact=artifact
        )

    conflicting_domain = ReplayPlan(
        [
            ReplayStep("bind_program_domain", refs=["column_domain"]),
            *artifact.plan.steps,
        ]
    )
    with pytest.raises(SchemaError, match="mutually exclusive replay mechanisms"):
        registry.validate_strict_plan(
            conflicting_domain, frontend=Frontend.TRITON, artifact=artifact
        )


def test_strict_pi_preserves_construction_order_in_rewriter_payload() -> None:
    registry = default_triton_rule_registry()
    steps = [
        ReplayStep("materialize_mask_tile", node_ref="input_masked_load"),
        ReplayStep("bind_wide_program_domain", node_ref="column_domain"),
        ReplayStep("materialize_pointer_tile", node_ref="output_guarded_store"),
    ]
    forward = ReplayPlan(steps)
    reversed_plan = ReplayPlan(list(reversed(steps)))
    assert forward.to_json() != reversed_plan.to_json()
    assert registry.rewriter_view(forward)["plan"] == forward.to_json()
    assert registry.rewriter_view(reversed_plan)["plan"] == reversed_plan.to_json()


def test_empty_pi_has_no_hidden_baseline_and_fails_nontrivial_coverage() -> None:
    registry = default_triton_rule_registry()
    artifact = replace(_artifact(), plan=ReplayPlan([]))
    raw = registry.rewriter_view(ReplayPlan([]), frontend=Frontend.TRITON)
    assert raw["plan"] == {"steps": []}
    assert raw["selection"] == []
    assert raw["cards"] == []
    with pytest.raises(SchemaError, match="strict pi is missing"):
        build_normal_rewriter_view(artifact, registry)
    with pytest.raises(SchemaError, match="strict pi is missing"):
        build_masked_rewriter_prompt(artifact, registry)


def test_pi_ablation_specs_are_deterministic_and_keep_normal_validation() -> None:
    registry = default_triton_rule_registry()
    artifact, _ = materialize_replay_plan(
        replace(
            _artifact(),
            strategy=StrategySpec(
                [
                    StrategyBinding(
                        "column_domain",
                        StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                        StrategyChoice.WIDE,
                    )
                ]
            ),
        ),
        registry,
    )
    specs = build_pi_ablations(artifact, registry, limit=2)

    assert [spec.kind.value for spec in specs] == [
        "plan_step_dropout",
        "plan_mechanism_substitution",
    ]
    assert specs[0].dropped_step_index == 0
    assert specs[0].dropped_rule == "bind_wide_program_domain"
    assert specs[0].dropped_target == "column_domain"
    assert len(specs[0].plan.steps) == len(artifact.plan.steps) - 1

    with pytest.raises(ComposeMismatchError):
        assert_replay_plan_matches(artifact, registry, specs[0].plan)

    dropout_only = build_pi_ablations(
        artifact,
        registry,
        limit=1,
        kinds=(specs[0].kind,),
    )
    assert len(dropout_only) == 1
    assert dropout_only[0].kind.value == "plan_step_dropout"

    substitution = build_pi_ablations(
        artifact,
        registry,
        limit=1,
        kinds=(type(specs[0].kind).PLAN_MECHANISM_SUBSTITUTION,),
    )
    assert len(substitution) == 1
    assert substitution[0].dropped_rule == "materialize_index_tile"
    assert substitution[0].replacement_rule == "materialize_pointer_tile"
    assert substitution[0].dropped_target == "input_masked_load"
    assert substitution[0].artifact is not None
    assert substitution[0].strategy_before != substitution[0].strategy_after
    assert substitution[0].artifact.j == artifact.j
    assert substitution[0].artifact.interface == artifact.interface
    assert substitution[0].artifact.schedule == artifact.schedule
    assert any(
        step.rule == "materialize_pointer_tile"
        for step in substitution[0].artifact.plan.steps
    )


def test_semantic_intervention_changes_j_but_not_the_pi_channel() -> None:
    registry = default_triton_rule_registry()
    baseline = _artifact()
    changed_intent_json = baseline.j.intent.to_json()
    changed_intent_json["intent_function"]["body"][0]["attrs"]["fn"] = "max"
    intervention = replace(
        baseline,
        j=replace(
            baseline.j,
            intent=IntentFunction.from_json(changed_intent_json),
            boundary=replace(
                baseline.j.boundary,
                accesses=[
                    replace(access, fill_value="-inf")
                    if access.kind is AccessKind.LOAD
                    else access
                    for access in baseline.j.boundary.accesses
                ],
                reductions=[
                    replace(
                        reduction,
                        identity="-inf",
                        masked_contribution="-inf",
                    )
                    for reduction in baseline.j.boundary.reductions
                ],
            ),
        ),
    )

    baseline_view = build_normal_rewriter_view(baseline, registry)
    intervention_view = build_normal_rewriter_view(intervention, registry)
    assert baseline_view["j"] != intervention_view["j"]
    for key in ("plan", "rule_selection", "rule_cards", "rule_registry_digest"):
        assert baseline_view[key] == intervention_view[key]


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
    assert "tl.sum" in normal  # concrete API names live in selected system rule cards
    assert "one deterministic syntactic normal form" in normal
    assert "reduction_tiling=multi_tile_accumulate" in normal
    assert (
        "reduction_preheader_staging=stage_invariant_access_scaffolding"
        in normal
    )
    assert "raw_pointer + outer_index" in normal
    assert "tile-dependent address, mask or load" in normal
    assert "never collapse the loop" in normal
    assert "Never emit raise" in normal
    assert "assert, try/except/finally" in normal
    assert "aborting Python branch" in normal
    assert "pre-expand a" in normal
    assert "global axis number as a local index" in normal
    assert "insertion count must equal the local rank" in normal
    assert "reuse an identical" in normal.lower()
    assert '"reference_naming": "opaque_replay_refs_v1"' in normal
    assert "reduce_sum_node" not in normal
    assert "input_masked_load" not in normal
    assert "output_guarded_store" not in normal
    assert "column_domain" not in normal
    assert "row_offset" not in normal
    assert "row_sum_kernel" not in masked
    assert "row_sum_impl" not in masked
    assert "BLOCK_N" not in masked
    assert '"function_name": "scrc_k0"' in masked
    assert "scrc_p4" in masked
    assert "tl.sum" in masked  # masked-J retains the same pi/rule-card channel
    assert "reduce_sum_node" not in masked
    assert "ABLATION RUN" in masked
    assert "interface.compiler_slots are launch-only" in masked


def test_free_form_j_attr_cannot_be_used_as_a_source_ngram_channel() -> None:
    artifact = _artifact()
    intent_json = artifact.j.intent.to_json()
    intent_json["intent_function"]["body"][0]["attrs"]["debug"] = (
        "private_helper secret_accumulator source_lane"
    )
    leaky = replace(
        artifact,
        j=replace(
            artifact.j,
            intent=IntentFunction.from_json(intent_json),
        ),
        source=(
            "private_helper secret_accumulator source_lane = "
            "tl.load(pointer)"
        ),
    )

    report = scan_artifact_leakage(
        leaky, registry=default_triton_rule_registry()
    )
    assert not report.passed
    assert any(
        item.kind is LeakageKind.SOURCE_NGRAM and ".attrs.debug" in item.path
        for item in report.violations
    )
