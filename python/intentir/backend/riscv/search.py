"""Candidate-universe and discardable-C planning for the native RVV backend.

The candidate universe is constructed without consulting C.  ``Attach_C`` may
only reorder that fixed universe; erasing C therefore changes neither
admission nor the executable states available to the search.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from intentir.backend.materialize import BackendSearchState, CPrior
from intentir.backend.riscv.compile import (
    RiscvLoweringCandidate,
    RiscvSchedule,
    RiscvTransformationState,
)
from intentir.ir.production_snapshot import content_digest


RVV_SEARCH_PLAN_PROTOCOL = "intentir.backend.riscv.search-plan.v1"

_PARALLEL_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_REDUCTION_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_JOINT_TILES = ((1, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 128))


def _schedule_universe() -> tuple[RiscvSchedule, ...]:
    schedules = [
        RiscvSchedule(opt, vectorize, 128, 0, 0)
        for opt in (2, 3)
        for vectorize in (False, True)
    ]
    schedules.extend(
        RiscvSchedule(3, True, 128, parallel, reduction)
        for parallel, reduction in (
            *((value, 0) for value in _PARALLEL_TILES),
            *((0, value) for value in _REDUCTION_TILES),
            *_JOINT_TILES,
        )
    )
    # Candidate IDs are content-addressed, but preserve one deterministic
    # spelling in case schedule grids are edited later.
    unique = {content_digest(item.to_dict()): item for item in schedules}
    return tuple(unique[key] for key in sorted(unique))


def enumerate_riscv_candidates(
    *, enable_b_transformations: bool = True
) -> tuple[RiscvLoweringCandidate, ...]:
    """Build the target schedule/structure universe without reading C."""

    return tuple(
        RiscvLoweringCandidate(RiscvTransformationState(fuse), schedule)
        for fuse in ((False, True) if enable_b_transformations else (False,))
        for schedule in _schedule_universe()
    )


def _usable_c_priors(
    state: BackendSearchState,
) -> tuple[tuple[CPrior, ...], tuple[dict[str, Any], ...]]:
    usable: list[CPrior] = []
    ignored: list[dict[str, Any]] = []
    for prior in state.priors:
        if prior.role not in {"tile_extent", "reduction_tile_extent"}:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "role_not_an_rvv_tile_ordering_hint",
                }
            )
            continue
        if (
            isinstance(prior.value, bool)
            or not isinstance(prior.value, (int, float))
            or float(prior.value) <= 0
        ):
            ignored.append(
                {**prior.to_record(), "reason": "non_positive_numeric_value"}
            )
            continue
        usable.append(prior)
    return tuple(usable), tuple(ignored)


def _c_score(schedule: RiscvSchedule, priors: tuple[CPrior, ...]) -> float:
    if not priors:
        return 0.0
    distances: list[float] = []
    for prior in priors:
        candidate = (
            schedule.reduction_tile
            if prior.role == "reduction_tile_extent"
            else schedule.parallel_tile
        )
        if candidate <= 0:
            return float("inf")
        distances.append(abs(math.log2(float(candidate) / float(prior.value))))
    return statistics.fmean(distances)


def _erase_schedule_key(schedule: RiscvSchedule) -> tuple[Any, ...]:
    # C-free deterministic order starts at the strong untiled LLVM baseline,
    # then explores small-to-large parallel, reduction and joint tilings.
    return (
        schedule.parallel_tile != 0 or schedule.reduction_tile != 0,
        schedule.parallel_tile == 0,
        schedule.reduction_tile == 0,
        max(schedule.parallel_tile, schedule.reduction_tile),
        schedule.parallel_tile,
        schedule.reduction_tile,
        not schedule.vectorize,
        -schedule.opt_level,
    )


def _interleave_structure_lanes(
    candidates: tuple[RiscvLoweringCandidate, ...],
    *,
    schedule_key: Any,
) -> tuple[RiscvLoweringCandidate, ...]:
    lanes: dict[bool, list[RiscvLoweringCandidate]] = {}
    for candidate in candidates:
        lanes.setdefault(candidate.transformation.fuse_elementwise, []).append(
            candidate
        )
    for lane in lanes.values():
        lane.sort(key=lambda item: schedule_key(item.schedule))
    ordered: list[RiscvLoweringCandidate] = []
    rank = 0
    lane_order = tuple(item for item in (False, True) if item in lanes)
    while any(rank < len(lanes[item]) for item in lane_order):
        for item in lane_order:
            if rank < len(lanes[item]):
                ordered.append(lanes[item][rank])
        rank += 1
    return tuple(ordered)


@dataclass(frozen=True)
class RiscvSearchPlan:
    universe: tuple[RiscvLoweringCandidate, ...]
    erase_c_order: tuple[str, ...]
    c_prior_order: tuple[str, ...]
    frozen_c_candidate_id: str
    c_scores: tuple[tuple[str, float | None], ...]
    consumed_c_priors: tuple[CPrior, ...]
    ignored_c_priors: tuple[dict[str, Any], ...]

    @property
    def universe_digest(self) -> str:
        return content_digest(sorted(item.candidate_id for item in self.universe))

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": RVV_SEARCH_PLAN_PROTOCOL,
            "candidate_universe_digest": self.universe_digest,
            "candidate_universe_count": len(self.universe),
            "erase_c_order": list(self.erase_c_order),
            "c_prior_order": list(self.c_prior_order),
            "frozen_c_candidate_id": self.frozen_c_candidate_id,
            "c_scores": {
                candidate_id: score for candidate_id, score in self.c_scores
            },
            "consumed_c_priors": [
                item.to_record() for item in self.consumed_c_priors
            ],
            "ignored_c_priors": list(self.ignored_c_priors),
            "same_universe_for_erase_c_and_c_prior": True,
            "c_controls_admission": False,
            "c_controls_structure": False,
        }


def plan_riscv_search(
    state: BackendSearchState,
    *,
    enable_b_transformations: bool = True,
) -> RiscvSearchPlan:
    if not isinstance(state, BackendSearchState):
        raise TypeError("RVV search planning requires BackendSearchState")
    universe = enumerate_riscv_candidates(
        enable_b_transformations=enable_b_transformations
    )
    usable_priors, ignored_priors = _usable_c_priors(state)
    scores = {
        item.candidate_id: _c_score(item.schedule, usable_priors)
        for item in universe
    }
    erase_order = _interleave_structure_lanes(
        universe, schedule_key=_erase_schedule_key
    )
    c_order = _interleave_structure_lanes(
        universe,
        schedule_key=lambda schedule: (
            _c_score(schedule, usable_priors),
            _erase_schedule_key(schedule),
        ),
    )
    # Frozen-C projects only onto schedule state; C has no authority to select
    # fusion.  When C has no usable RVV hint, this is the canonical C-free state.
    materialized = tuple(
        item for item in c_order if not item.transformation.fuse_elementwise
    )
    frozen = materialized[0]
    return RiscvSearchPlan(
        universe=universe,
        erase_c_order=tuple(item.candidate_id for item in erase_order),
        c_prior_order=tuple(item.candidate_id for item in c_order),
        frozen_c_candidate_id=frozen.candidate_id,
        c_scores=tuple(
            (
                item.candidate_id,
                None if math.isinf(scores[item.candidate_id]) else scores[item.candidate_id],
            )
            for item in universe
        ),
        consumed_c_priors=usable_priors,
        ignored_c_priors=ignored_priors,
    )


__all__ = [
    "RVV_SEARCH_PLAN_PROTOCOL",
    "RiscvSearchPlan",
    "enumerate_riscv_candidates",
    "plan_riscv_search",
]
