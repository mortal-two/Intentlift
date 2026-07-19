"""Canonicality and anti-tampering tests for deterministic replay-plan composition."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    BroadcastSpec,
    Frontend,
    Gate1Artifact,
    KernelInterface,
    KernelParameter,
    LocationMap,
    OutputMapping,
    ParameterKind,
    ParameterMode,
    ReplayPlan,
    ReplayStep,
    RuleRegistry,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
    StrategyBinding,
    StrategyChoice,
    StrategyFamily,
    StrategySpec,
    default_triton_rule_registry,
    materialize_replay_plan,
)
from intentir.verify.gate1.composer import (
    AmbiguousRuleMatchError,
    ComposeDependencyError,
    ComposeMismatchError,
    ZeroRuleMatchError,
    assert_replay_plan_matches,
    compose_replay_plan,
)
from intentir.verify.gate1.pi_ablations import (
    PiAblationKind,
    build_pi_ablations,
)


def _artifact() -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "identity_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "out",
                    "operands": ["x"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="kernel",
            intent=intent,
            node_ids={"identity_node": "body[0]"},
            boundary=Boundary(
                domains={"lane_domain": "lane < N"},
                accesses=[
                    BoundaryAccess(
                        access_id="load_x",
                        kind=AccessKind.LOAD,
                        tensor="x",
                        predicate="lane < N",
                        fill_value=0.0,
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                    BoundaryAccess(
                        access_id="store_out",
                        kind=AccessKind.STORE,
                        tensor="out",
                        predicate="lane < N",
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                ],
            ),
        ),
        interface=KernelInterface(
            kernel_id="kernel",
            function_name="kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "x",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    1,
                    tensor_ref="x",
                ),
                KernelParameter(
                    "out",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "f32",
                    1,
                    tensor_ref="out",
                ),
                KernelParameter("N", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
                KernelParameter("BLOCK", ParameterKind.CONSTEXPR),
            ],
            grid=["ceil_div(N, BLOCK)"],
            compiler_slots=["num_warps"],
            do_not_specialize=["N"],
        ),
        schedule=ScheduleSpec(
            slots=[
                ScheduleSlot("BLOCK", ScheduleSlotKind.CONSTEXPR),
                ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
            ],
            values={"BLOCK": 64, "num_warps": 4},
        ),
        plan=ReplayPlan(
            [
                ReplayStep("bind_program_domain", refs=["lane_domain"]),
                ReplayStep("bind_runtime_scalar", refs=["N"]),
                ReplayStep("declare_schedule_slot", refs=["BLOCK"]),
                ReplayStep("declare_schedule_slot", refs=["num_warps"]),
                ReplayStep("materialize_index_tile", refs=["load_x"]),
                ReplayStep("realize_masked_access", refs=["load_x"]),
                ReplayStep("realize_cast", node_ref="identity_node"),
                ReplayStep("materialize_index_tile", refs=["store_out"]),
                ReplayStep("realize_masked_access", refs=["store_out"]),
            ]
        ),
        location_map=LocationMap("private.py", []),
        source="SOURCE_CANARY_91F7",
    )


def _independent_node_artifact(*, reversed_declarations: bool) -> Gate1Artifact:
    first = {
        "op": "const",
        "result": "left_value",
        "attrs": {"value": 1.0, "dtype": "f32"},
    }
    second = {
        "op": "const",
        "result": "right_value",
        "attrs": {"value": 2.0, "dtype": "f32"},
    }
    body = [second, first] if reversed_declarations else [first, second]
    body.append(
        {
            "op": "map_binary",
            "result": "out",
            "operands": ["left_value", "right_value"],
            "attrs": {"fn": "add"},
        }
    )
    node_ids = (
        {"right_node": "body[0]", "left_node": "body[1]", "sum_node": "body[2]"}
        if reversed_declarations
        else {"left_node": "body[0]", "right_node": "body[1]", "sum_node": "body[2]"}
    )
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="independent",
            intent=IntentFunction.from_json(
                {
                    "name": "independent",
                    "inputs": [],
                    "outputs": [{"name": "out", "dtype": "f32", "shape": []}],
                    "body": body,
                }
            ),
            node_ids=node_ids,
            boundary=Boundary(),
        ),
        interface=KernelInterface(
            kernel_id="independent",
            function_name="independent",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "out", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 1, tensor_ref="out",
                )
            ],
            grid=["1"],
        ),
        schedule=ScheduleSpec(),
        plan=ReplayPlan([ReplayStep("realize_const", node_ref="left_node")]),
        location_map=LocationMap("private.py", []),
    )


def _iota_artifact() -> Gate1Artifact:
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="iota_kernel",
            intent=IntentFunction.from_json(
                {
                    "name": "iota_kernel",
                    "inputs": [],
                    "outputs": [
                        {"name": "out", "dtype": "i32", "shape": ["N"]}
                    ],
                    "body": [
                        {
                            "op": "iota",
                            "result": "out",
                            "attrs": {"n": "N", "dtype": "i32"},
                        }
                    ],
                }
            ),
            node_ids={"iota_node": "body[0]"},
            boundary=Boundary(
                domains={"global_ordinal": "global_ordinal in [0, N)"},
                accesses=[
                    BoundaryAccess(
                        access_id="store_out",
                        kind=AccessKind.STORE,
                        tensor="out",
                        predicate="global_ordinal < N",
                        node_ref="iota_node",
                        index_expression="global_ordinal",
                    )
                ],
                output_mappings=[
                    OutputMapping("out", "out", "iota_node")
                ],
            ),
        ),
        interface=KernelInterface(
            kernel_id="iota_kernel",
            function_name="iota_kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "out",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "i32",
                    1,
                    tensor_ref="out",
                ),
                KernelParameter(
                    "N", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
                ),
                KernelParameter("BLOCK", ParameterKind.CONSTEXPR),
            ],
            grid=["ceil_div(N, BLOCK)"],
            compiler_slots=["num_warps"],
        ),
        schedule=ScheduleSpec(
            slots=[
                ScheduleSlot("BLOCK", ScheduleSlotKind.CONSTEXPR),
                ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
            ],
            values={"BLOCK": 32, "num_warps": 4},
        ),
        plan=ReplayPlan([]),
        location_map=LocationMap("private.py", []),
    )


def _transpose_artifact() -> Gate1Artifact:
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="transpose_kernel",
            intent=IntentFunction.from_json(
                {
                    "name": "transpose_kernel",
                    "inputs": [
                        {"name": "x", "dtype": "f32", "shape": ["M", "N"]}
                    ],
                    "outputs": [
                        {"name": "out", "dtype": "f32", "shape": ["N", "M"]}
                    ],
                    "body": [
                        {
                            "op": "transpose",
                            "result": "out",
                            "operands": ["x"],
                            "attrs": {"perm": [1, 0]},
                        }
                    ],
                }
            ),
            node_ids={"transpose_node": "body[0]"},
            boundary=Boundary(domains={"program": "program_id(0) < 1"}),
        ),
        interface=KernelInterface(
            kernel_id="transpose_kernel",
            function_name="transpose_kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "x", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, tensor_ref="x",
                ),
                KernelParameter(
                    "out", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 2, tensor_ref="out",
                ),
            ],
            grid=["1"],
        ),
        schedule=ScheduleSpec(),
        plan=ReplayPlan(
            [ReplayStep("realize_transpose", node_ref="transpose_node")]
        ),
        location_map=LocationMap("private.py", []),
    )


def _concat_artifact() -> Gate1Artifact:
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="concat_kernel",
            intent=IntentFunction.from_json(
                {
                    "name": "concat_kernel",
                    "inputs": [
                        {"name": "left", "dtype": "f32", "shape": ["M", "N"]},
                        {"name": "right", "dtype": "f32", "shape": ["M", "K"]},
                    ],
                    "outputs": [
                        {"name": "out", "dtype": "f32", "shape": ["M", "P"]}
                    ],
                    "body": [
                        {
                            "op": "concat",
                            "result": "out",
                            "operands": ["left", "right"],
                            "attrs": {"axis": 1},
                        }
                    ],
                }
            ),
            node_ids={"concat_node": "body[0]"},
            boundary=Boundary(domains={"program": "program_id(0) < 1"}),
        ),
        interface=KernelInterface(
            kernel_id="concat_kernel",
            function_name="concat_kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "left", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, tensor_ref="left",
                ),
                KernelParameter(
                    "right", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 2, tensor_ref="right",
                ),
                KernelParameter(
                    "out", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 2, tensor_ref="out",
                ),
            ],
            grid=["1"],
        ),
        schedule=ScheduleSpec(),
        plan=ReplayPlan([ReplayStep("realize_concat", node_ref="concat_node")]),
        location_map=LocationMap("private.py", []),
    )


def _multi_broadcast_artifact(
    broadcasts: list[BroadcastSpec] | None = None,
) -> Gate1Artifact:
    return Gate1Artifact(
        j=SemanticArchive(
            kernel_id="multi_broadcast",
            intent=IntentFunction.from_json(
                {
                    "name": "multi_broadcast",
                    "inputs": [],
                    "outputs": [
                        {"name": "out", "dtype": "i1", "shape": ["N", "N"]}
                    ],
                    "body": [
                        {
                            "op": "iota",
                            "result": "row",
                            "attrs": {"n": "N", "dtype": "i32"},
                        },
                        {
                            "op": "iota",
                            "result": "col",
                            "attrs": {"n": "N", "dtype": "i32"},
                        },
                        {
                            "op": "map_binary",
                            "result": "out",
                            "operands": ["row", "col"],
                            "attrs": {"fn": "eq"},
                        }
                    ],
                }
            ),
            node_ids={
                "row_node": "body[0]",
                "col_node": "body[1]",
                "compare_node": "body[2]",
            },
            boundary=Boundary(
                broadcasts=(
                    broadcasts
                    if broadcasts is not None
                    else [
                        BroadcastSpec("compare_node", "row", [1]),
                        BroadcastSpec("compare_node", "col", [0]),
                    ]
                )
            ),
        ),
        interface=KernelInterface(
            kernel_id="multi_broadcast",
            function_name="multi_broadcast",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "out", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "i1", 2, tensor_ref="out",
                ),
                KernelParameter(
                    "N", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
                ),
            ],
            grid=["1"],
        ),
        schedule=ScheduleSpec(),
        plan=ReplayPlan([]),
        location_map=LocationMap("private.py", []),
    )


def _reverse_object_keys(value: Any) -> Any:
    """Recursively rebuild JSON objects with the opposite insertion order."""

    if isinstance(value, dict):
        return {
            key: _reverse_object_keys(value[key])
            for key in reversed(tuple(value.keys()))
        }
    if isinstance(value, list):
        return [_reverse_object_keys(item) for item in value]
    return value


def _leaf_pointers(value: Any, pointer: str = "") -> set[str]:
    if isinstance(value, dict):
        leaves: set[str] = set()
        for key, child in value.items():
            token = key.replace("~", "~0").replace("/", "~1")
            leaves.update(_leaf_pointers(child, f"{pointer}/{token}"))
        return leaves
    if isinstance(value, list):
        leaves = set()
        for index, child in enumerate(value):
            leaves.update(_leaf_pointers(child, f"{pointer}/{index}"))
        return leaves
    return {pointer or "/"}


def _drop_step(plan: ReplayPlan) -> ReplayPlan:
    return ReplayPlan(plan.steps[:-1])


def _add_step(plan: ReplayPlan) -> ReplayPlan:
    return ReplayPlan([*plan.steps, plan.steps[-1]])


def _mutate_ref(plan: ReplayPlan) -> ReplayPlan:
    steps = list(plan.steps)
    for index, step in enumerate(steps):
        occupied = set(step.refs)
        if step.node_ref is not None:
            occupied.add(step.node_ref)
        replacement = next(
            (
                candidate.target_ref
                for candidate in steps
                if candidate.target_ref is not None
                and candidate.target_ref not in occupied
            ),
            None,
        )
        if replacement is None:
            continue
        if step.node_ref is not None:
            steps[index] = replace(step, node_ref=replacement)
        else:
            steps[index] = replace(step, refs=[replacement, *step.refs[1:]])
        return ReplayPlan(steps)
    raise AssertionError("fixture must expose at least two distinct replay targets")


def _swap_step_order(plan: ReplayPlan) -> ReplayPlan:
    steps = list(plan.steps)
    for index in range(len(steps) - 1):
        if steps[index] != steps[index + 1]:
            steps[index], steps[index + 1] = steps[index + 1], steps[index]
            return ReplayPlan(steps)
    raise AssertionError("fixture must expose at least two distinct replay steps")


def test_same_inputs_produce_byte_identical_pi() -> None:
    artifact = _artifact()
    registry = default_triton_rule_registry()

    baseline = compose_replay_plan(artifact, registry)
    assert isinstance(baseline.canonical_bytes, bytes)
    for _ in range(4):
        repeated = compose_replay_plan(artifact, registry)
        assert repeated.canonical_bytes == baseline.canonical_bytes
        assert repeated.plan_digest == baseline.plan_digest
        assert repeated.plan.to_json() == baseline.plan.to_json()


def test_json_object_key_order_does_not_change_canonical_pi() -> None:
    artifact = _artifact()
    registry = default_triton_rule_registry()
    reordered_artifact = Gate1Artifact.from_json(
        _reverse_object_keys(artifact.to_json())
    )
    reordered_registry = RuleRegistry.from_json(
        _reverse_object_keys(registry.to_json())
    )

    baseline = compose_replay_plan(artifact, registry)
    reordered = compose_replay_plan(reordered_artifact, reordered_registry)

    assert reordered.canonical_bytes == baseline.canonical_bytes
    assert reordered.plan_digest == baseline.plan_digest
    assert reordered.plan.to_json() == baseline.plan.to_json()


def test_opaque_handle_alpha_renaming_does_not_change_canonical_pi() -> None:
    artifact = _artifact()
    payload = artifact.to_json()
    payload["j"]["node_ids"] = {"q91": "body[0]"}
    payload["j"]["boundary"]["domains"] = {
        "z17": payload["j"]["boundary"]["domains"].pop("lane_domain")
    }
    rename = {
        "identity_node": "q91",
        "load_x": "u41",
        "store_out": "u42",
        "lane_domain": "z17",
    }
    for access in payload["j"]["boundary"]["accesses"]:
        access["access_id"] = rename[access["access_id"]]
        access["node_ref"] = rename[access["node_ref"]]
    for step in payload["plan"]["steps"]:
        if "node_ref" in step:
            step["node_ref"] = rename.get(step["node_ref"], step["node_ref"])
        if "refs" in step:
            step["refs"] = [rename.get(ref, ref) for ref in step["refs"]]

    renamed = Gate1Artifact.from_json(payload)
    registry = default_triton_rule_registry()
    baseline = compose_replay_plan(artifact, registry)
    alpha = compose_replay_plan(renamed, registry)

    assert alpha.plan.to_json() != baseline.plan.to_json()
    assert alpha.canonical_bytes == baseline.canonical_bytes
    assert alpha.plan_digest == baseline.plan_digest
    assert alpha.provenance.j_digest == baseline.provenance.j_digest
    assert alpha.provenance.interface_digest == baseline.provenance.interface_digest
    assert alpha.provenance.schedule_digest == baseline.provenance.schedule_digest
    assert alpha.provenance.strategy_digest == baseline.provenance.strategy_digest


def test_independent_node_declaration_order_does_not_change_canonical_pi() -> None:
    registry = default_triton_rule_registry()
    baseline = compose_replay_plan(
        _independent_node_artifact(reversed_declarations=False), registry
    )
    reordered = compose_replay_plan(
        _independent_node_artifact(reversed_declarations=True), registry
    )

    assert reordered.canonical_bytes == baseline.canonical_bytes
    assert reordered.plan_digest == baseline.plan_digest
    assert reordered.provenance.j_digest == baseline.provenance.j_digest


def test_distinct_broadcasts_for_one_node_compose_one_node_level_step() -> None:
    result = compose_replay_plan(
        _multi_broadcast_artifact(), default_triton_rule_registry()
    )
    steps = [
        (index, step)
        for index, step in enumerate(result.plan.steps)
        if step.rule == "realize_broadcast"
    ]

    assert [(step.node_ref, step.refs) for _, step in steps] == [
        ("compare_node", [])
    ]
    step_index = steps[0][0]
    row_index = next(
        index
        for index, step in enumerate(result.plan.steps)
        if step.node_ref == "row_node"
    )
    col_index = next(
        index
        for index, step in enumerate(result.plan.steps)
        if step.node_ref == "col_node"
    )
    compare_index = next(
        index
        for index, step in enumerate(result.plan.steps)
        if step.node_ref == "compare_node" and step.rule == "realize_elementwise"
    )
    assert row_index < step_index < compare_index
    assert col_index < step_index < compare_index
    rule_sources = result.provenance.leaf_sources[
        f"/steps/{step_index}/rule"
    ]
    node_sources = result.provenance.leaf_sources[
        f"/steps/{step_index}/node_ref"
    ]
    assert {
        source.pointer for source in rule_sources if source.owner == "j"
    } == {
        "/j/boundary/broadcasts/0",
        "/j/boundary/broadcasts/1",
    }
    assert {source.pointer for source in node_sources} == {
        "/j/boundary/broadcasts/0/node_ref",
        "/j/boundary/broadcasts/1/node_ref",
    }


def test_broadcast_declaration_order_is_canonical_but_all_facts_are_bound() -> None:
    registry = default_triton_rule_registry()
    baseline = compose_replay_plan(_multi_broadcast_artifact(), registry)
    reordered = compose_replay_plan(
        _multi_broadcast_artifact(
            [
                BroadcastSpec("compare_node", "col", [0]),
                BroadcastSpec("compare_node", "row", [1]),
            ]
        ),
        registry,
    )
    changed = compose_replay_plan(
        _multi_broadcast_artifact(
            [
                BroadcastSpec("compare_node", "row", [1]),
                BroadcastSpec("compare_node", "col", [1]),
            ]
        ),
        registry,
    )
    changed_operand = compose_replay_plan(
        _multi_broadcast_artifact(
            [
                BroadcastSpec("compare_node", "row", [1]),
                BroadcastSpec("compare_node", "row", [0]),
            ]
        ),
        registry,
    )

    assert reordered.canonical_bytes == baseline.canonical_bytes
    assert reordered.plan_digest == baseline.plan_digest
    assert reordered.provenance.j_digest == baseline.provenance.j_digest
    assert changed.plan_digest == baseline.plan_digest
    assert changed.provenance.j_digest != baseline.provenance.j_digest
    assert changed_operand.provenance.j_digest != baseline.provenance.j_digest


def test_exact_duplicate_broadcast_declarations_fail_closed() -> None:
    duplicate = BroadcastSpec("compare_node", "row", [1])
    artifact = _multi_broadcast_artifact([duplicate, duplicate])

    with pytest.raises(ComposeDependencyError, match="duplicate broadcast"):
        compose_replay_plan(artifact, default_triton_rule_registry())


def test_provenance_covers_every_pi_leaf_without_orphans() -> None:
    result = compose_replay_plan(_artifact(), default_triton_rule_registry())
    expected_leaves = _leaf_pointers(result.plan.to_json())

    assert set(result.provenance.leaf_sources) == expected_leaves
    assert all(result.provenance.leaf_sources.values())
    assert result.provenance.coverage == 1.0
    assert result.provenance.orphan_leaf_count == 0
    result.provenance.validate(result.plan)


def test_materialized_artifact_persists_exact_compose_provenance() -> None:
    artifact, result = materialize_replay_plan(
        _artifact(), default_triton_rule_registry()
    )
    round_tripped = Gate1Artifact.from_json(artifact.to_json())

    assert artifact.plan == result.plan
    assert artifact.compose_provenance == result.provenance.to_json()
    assert round_tripped.compose_provenance == artifact.compose_provenance


def test_iota_has_one_canonical_rule_and_complete_pi_provenance() -> None:
    registry = default_triton_rule_registry()
    assert len(registry.cards) == 24
    artifact, result = materialize_replay_plan(_iota_artifact(), registry)
    iota_steps = [
        step
        for step in artifact.plan.steps
        if step.node_ref == "iota_node"
    ]

    assert [(step.rule, step.refs) for step in iota_steps] == [
        ("realize_iota", [])
    ]
    assert result.provenance.coverage == 1.0
    assert result.provenance.orphan_leaf_count == 0
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )


def test_iota_fails_closed_when_its_shared_rule_is_absent() -> None:
    registry = default_triton_rule_registry()
    without_iota = RuleRegistry(
        [card for card in registry.cards if card.id != "realize_iota"]
    )

    with pytest.raises(ZeroRuleMatchError, match="intent_op:iota"):
        compose_replay_plan(_iota_artifact(), without_iota)


def test_iota_is_classified_as_compute_for_pi_dropout_diagnostics() -> None:
    registry = default_triton_rule_registry()
    artifact, _ = materialize_replay_plan(_iota_artifact(), registry)

    (dropout,) = build_pi_ablations(
        artifact,
        registry,
        limit=1,
        kinds=(PiAblationKind.PLAN_STEP_DROPOUT,),
    )

    assert dropout.dropped_rule == "realize_iota"
    assert dropout.dropped_target == "iota_node"


def test_transpose_has_one_canonical_rule_and_complete_pi_provenance() -> None:
    registry = default_triton_rule_registry()
    artifact, result = materialize_replay_plan(_transpose_artifact(), registry)
    transpose_steps = [
        step
        for step in artifact.plan.steps
        if step.node_ref == "transpose_node"
    ]

    assert [(step.rule, step.refs) for step in transpose_steps] == [
        ("realize_transpose", [])
    ]
    assert result.provenance.coverage == 1.0
    assert result.provenance.orphan_leaf_count == 0
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )


def test_transpose_fails_closed_when_its_rule_is_absent() -> None:
    registry = default_triton_rule_registry()
    without_transpose = RuleRegistry(
        [card for card in registry.cards if card.id != "realize_transpose"]
    )

    with pytest.raises(ZeroRuleMatchError, match="intent_op:transpose"):
        compose_replay_plan(_transpose_artifact(), without_transpose)


def test_concat_has_one_canonical_rule_and_complete_pi_provenance() -> None:
    registry = default_triton_rule_registry()
    artifact, result = materialize_replay_plan(_concat_artifact(), registry)
    concat_steps = [
        step for step in artifact.plan.steps if step.node_ref == "concat_node"
    ]

    assert [(step.rule, step.refs) for step in concat_steps] == [
        ("realize_concat", [])
    ]
    assert result.provenance.coverage == 1.0
    assert result.provenance.orphan_leaf_count == 0
    registry.validate_strict_plan(
        artifact.plan,
        frontend=artifact.interface.frontend,
        artifact=artifact,
    )


def test_concat_fails_closed_when_its_rule_is_absent() -> None:
    registry = default_triton_rule_registry()
    without_concat = RuleRegistry(
        [card for card in registry.cards if card.id != "realize_concat"]
    )

    with pytest.raises(ZeroRuleMatchError, match="intent_op:concat"):
        compose_replay_plan(_concat_artifact(), without_concat)


def test_override_only_strategy_selects_reviewed_alternative_cards() -> None:
    artifact = replace(
        _artifact(),
        strategy=StrategySpec(
            [
                StrategyBinding(
                    "lane_domain",
                    StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                    StrategyChoice.WIDE,
                ),
                StrategyBinding(
                    "load_x",
                    StrategyFamily.ADDRESS_ASSOCIATION,
                    StrategyChoice.POINTER_STAGED,
                ),
                StrategyBinding(
                    "load_x",
                    StrategyFamily.MASK_MATERIALIZATION,
                    StrategyChoice.STAGED_REUSE,
                ),
            ]
        ),
    )
    result = compose_replay_plan(artifact, default_triton_rule_registry())
    selected = {(step.rule, step.target_ref) for step in result.plan.steps}

    assert ("bind_wide_program_domain", "lane_domain") in selected
    assert ("materialize_pointer_tile", "load_x") in selected
    assert ("materialize_mask_tile", "load_x") in selected
    assert ("bind_program_domain", "lane_domain") not in selected
    assert ("materialize_index_tile", "load_x") not in selected


def test_explicit_metadata_drives_renamed_cards_without_registry_priority() -> None:
    payload = _artifact().to_json()
    payload["j"]["boundary"]["accesses"][1].pop("predicate")
    artifact = replace(
        Gate1Artifact.from_json(payload),
        strategy=StrategySpec(
            [
                StrategyBinding(
                    "lane_domain",
                    StrategyFamily.PROGRAM_COORDINATE_WIDTH,
                    StrategyChoice.WIDE,
                )
            ]
        ),
    )
    renamed = []
    for card in default_triton_rule_registry().cards:
        new_id = {
            "bind_wide_program_domain": "wide_from_metadata",
            "realize_masked_access": "masked_from_metadata",
            "realize_unmasked_access": "unmasked_from_metadata",
        }.get(card.id, card.id)
        renamed.append(replace(card, id=new_id))

    result = compose_replay_plan(artifact, RuleRegistry(renamed))
    selected = {(step.rule, step.target_ref) for step in result.plan.steps}
    assert ("wide_from_metadata", "lane_domain") in selected
    assert ("masked_from_metadata", "load_x") in selected
    assert ("unmasked_from_metadata", "store_out") in selected


def test_missing_rule_for_work_item_fails_closed() -> None:
    registry = default_triton_rule_registry()
    without_cast = RuleRegistry(
        [card for card in registry.cards if card.id != "realize_cast"]
    )

    with pytest.raises(ZeroRuleMatchError):
        compose_replay_plan(_artifact(), without_cast)


def test_ambiguous_rules_for_work_item_fail_closed() -> None:
    registry = default_triton_rule_registry()
    cast_card = next(card for card in registry.cards if card.id == "realize_cast")
    ambiguous = RuleRegistry(
        [*registry.cards, replace(cast_card, id="realize_cast_duplicate")]
    )

    with pytest.raises(AmbiguousRuleMatchError):
        compose_replay_plan(_artifact(), ambiguous)


@pytest.mark.parametrize(
    "mutate",
    [_drop_step, _add_step, _mutate_ref, _swap_step_order],
    ids=["drop-step", "add-step", "reference", "order"],
)
def test_modified_stored_pi_is_rejected_by_canonicality_api(
    mutate: Callable[[ReplayPlan], ReplayPlan],
) -> None:
    artifact = _artifact()
    registry = default_triton_rule_registry()
    canonical = compose_replay_plan(artifact, registry)
    canonical_artifact = replace(artifact, plan=canonical.plan)
    accepted = assert_replay_plan_matches(canonical_artifact, registry)
    assert accepted.canonical_bytes == canonical.canonical_bytes

    tampered = replace(canonical_artifact, plan=mutate(canonical.plan))

    with pytest.raises(ComposeMismatchError):
        assert_replay_plan_matches(tampered, registry)
