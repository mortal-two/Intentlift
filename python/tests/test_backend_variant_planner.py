from __future__ import annotations

import pytest

from intentir.backend.variant_planner import (
    SelectionConstraints,
    VariantPlanningError,
    enumerate_schedule_candidates,
    plan_variant_space,
)


def _backend_legality(candidate):
    if candidate.parameters["tile"] > 64:
        return False, "shared_memory_limit"
    return True, ""


def _constraints() -> SelectionConstraints:
    return SelectionConstraints(
        allowed_parameter_values={
            "tile": (32, 64),
            "vector_width": (2, 4),
        },
        required_tags=("contiguous-last-axis",),
        constraint_references=(
            "canonical.access_witness.output.store0",
            "canonical.layout.output.contiguous_axis_1",
            "intentir.layer_c.source_schedule",
        ),
    )


def test_enumeration_keeps_source_freeze_seed_in_same_space() -> None:
    candidates = enumerate_schedule_candidates(
        family="softmax",
        axes={"vector_width": (1, 2, 4), "tile": (32, 64)},
        source_schedule={"tile": 64, "vector_width": 2},
        tags=("contiguous-last-axis",),
    )

    assert candidates[0].variant_id == "softmax.source-seed"
    assert candidates[0].source_seed is True
    assert candidates[0].parameters == {"tile": 64, "vector_width": 2}
    assert len(candidates) == 6
    assert len({candidate.variant_id for candidate in candidates}) == 6
    assert len({candidate.parameter_key for candidate in candidates}) == 6


def test_constraints_on_prunes_only_after_backend_legality() -> None:
    plan = plan_variant_space(
        family="softmax",
        axes={"tile": (32, 64, 128), "vector_width": (1, 2, 4)},
        source_schedule={"tile": 64, "vector_width": 1},
        backend_legal=_backend_legality,
        constraints_enabled=True,
        constraints=_constraints(),
        tags=("contiguous-last-axis",),
    )

    assert plan.counts == {"raw": 9, "backend_legal": 6, "materialize": 5}
    assert len(plan.backend_rejections) == 3
    assert {item.reasons for item in plan.backend_rejections} == {
        ("shared_memory_limit",)
    }
    assert len(plan.constraint_rejections) == 1
    assert plan.constraint_rejections[0].reasons == (
        "vector_width:not_in:[2,4]",
    )
    # The freeze candidate enters the same measured search even when Layer-C
    # constraints would otherwise prune its vector width.
    assert plan.materialization_candidates[0].source_seed is True
    assert plan.constraint_policy is not None


def test_constraints_off_retains_same_raw_and_legal_spaces() -> None:
    common = dict(
        family="softmax",
        axes={"tile": (32, 64, 128), "vector_width": (1, 2, 4)},
        source_schedule={"tile": 64, "vector_width": 1},
        backend_legal=_backend_legality,
        tags=("contiguous-last-axis",),
    )
    constraints_on = plan_variant_space(
        **common, constraints_enabled=True, constraints=_constraints()
    )
    constraints_off = plan_variant_space(
        **common, constraints_enabled=False, constraints=None
    )

    assert constraints_off.raw_candidates == constraints_on.raw_candidates
    assert constraints_off.legal_candidates == constraints_on.legal_candidates
    assert constraints_off.counts == {
        "raw": 9,
        "backend_legal": 6,
        "materialize": 6,
    }
    assert constraints_off.constraint_rejections == ()
    assert constraints_off.constraint_policy is None


def test_plan_digest_is_deterministic_and_excludes_wall_clock_time() -> None:
    kwargs = dict(
        family="warp",
        axes={"tile": (32, 64), "vector_width": (1, 2)},
        source_schedule={"tile": 32, "vector_width": 1},
        backend_legal=lambda candidate: True,
        constraints_enabled=False,
        constraints=None,
    )

    first = plan_variant_space(**kwargs)
    second = plan_variant_space(**kwargs)

    assert first.deterministic_digest == second.deterministic_digest
    assert first.planning_time_ms >= 0.0
    assert second.planning_time_ms >= 0.0


def test_constraint_policy_requires_real_source_references() -> None:
    with pytest.raises(VariantPlanningError, match="source constraints references"):
        SelectionConstraints(
            allowed_parameter_values={"vector_width": (1, 2)},
            constraint_references=(),
        )


def test_source_schedule_must_be_backend_legal_for_freeze_ablation() -> None:
    with pytest.raises(VariantPlanningError, match="source freeze schedule"):
        plan_variant_space(
            family="matmul",
            axes={"tile": (32, 64)},
            source_schedule={"tile": 64},
            backend_legal=lambda candidate: candidate.parameters["tile"] == 32,
            constraints_enabled=False,
            constraints=None,
        )


def test_constraints_on_requires_explicit_policy() -> None:
    with pytest.raises(VariantPlanningError, match="auditable constraints policy"):
        plan_variant_space(
            family="resize",
            axes={"tile": (32,)},
            source_schedule={"tile": 32},
            backend_legal=lambda candidate: True,
            constraints_enabled=True,
            constraints=None,
        )
