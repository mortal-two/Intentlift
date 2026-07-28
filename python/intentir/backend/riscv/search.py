"""Candidate-universe and discardable-C planning for the native RVV backend.

The candidate universe is constructed without consulting C.  ``Attach_C`` may
only reorder that fixed universe; erasing C therefore changes neither
admission nor the executable states available to the search.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Mapping

from intentir.backend.materialize import BackendSearchState, CPrior
from intentir.backend.riscv.compile import (
    RiscvLoweringCandidate,
    RiscvSchedule,
    RiscvTransformationState,
)
from intentir.ir.production_snapshot import content_digest


RVV_SEARCH_PLAN_PROTOCOL = "intentir.backend.riscv.search-plan.v2"
RVV_C_PROJECTION_PROTOCOL = "intentir.backend.riscv.c-projection.v1"

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


@dataclass(frozen=True)
class _ProjectedCPrior:
    prior: CPrior
    schedule_dimension: str
    target_value: int
    rule: str

    def to_record(self) -> dict[str, Any]:
        return {
            "source_prior": self.prior.to_record(),
            "schedule_dimension": self.schedule_dimension,
            "target_value": self.target_value,
            "rule": self.rule,
        }


def _dtype_bits(dtype: str) -> int | None:
    match = re.fullmatch(r"(?:bf|f)(16|32|64)", dtype.lower())
    return int(match.group(1)) if match is not None else None


def _operation_facts(value: Any) -> tuple[set[tuple[int, ...]], bool, list[int]]:
    reduction_axes: set[tuple[int, ...]] = set()
    has_scan = False
    float_bits: list[int] = []

    def visit(item: Any) -> None:
        nonlocal has_scan
        if isinstance(item, Mapping):
            dtype = item.get("dtype")
            if isinstance(dtype, str):
                bits = _dtype_bits(dtype)
                if bits is not None:
                    float_bits.append(bits)
            op = item.get("op")
            if op in {"reduce", "scan"}:
                has_scan = has_scan or op == "scan"
                attrs = item.get("attrs", {})
                raw_axes = attrs.get("axes", [attrs.get("axis")]) if isinstance(
                    attrs, Mapping
                ) else []
                if isinstance(raw_axes, list) and all(
                    isinstance(axis, int) for axis in raw_axes
                ):
                    reduction_axes.add(tuple(int(axis) for axis in raw_axes))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return reduction_axes, has_scan, float_bits


def _target_vlen_bits(state: BackendSearchState) -> int:
    values = []
    for feature in state.target.features:
        match = re.fullmatch(r"vlen([0-9]+)", feature.lower())
        if match is not None:
            values.append(int(match.group(1)))
    if len(set(values)) > 1:
        raise ValueError("RVV target declares conflicting VLEN features")
    return values[0] if values else 128


def _power_of_two_floor(value: float) -> int:
    return 1 << max(0, math.floor(math.log2(max(1.0, value))))


def _project_c_priors(
    state: BackendSearchState,
) -> tuple[
    tuple[_ProjectedCPrior, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    payload = state.base.j.to_json()
    reduction_axes, has_scan, float_bits = _operation_facts(payload)
    vlen_bits = _target_vlen_bits(state)
    element_bits = max(float_bits, default=32)
    vector_lanes = max(1, vlen_bits // element_bits)
    projected: list[_ProjectedCPrior] = []
    ignored: list[dict[str, Any]] = []
    for prior in state.priors:
        if prior.role not in {
            "tile_extent",
            "reduction_tile_extent",
            "constexpr",
        }:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "role_has_no_registered_single_core_rvv_projection",
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
        axis_ref = prior.axis_ref if isinstance(prior.axis_ref, int) else None
        if prior.role == "reduction_tile_extent" and axis_ref is not None:
            maps_to_reduction = any(axis_ref in axes for axes in reduction_axes)
            if not maps_to_reduction:
                ignored.append(
                    {
                        **prior.to_record(),
                        "reason": "reduction_tile_axis_is_not_a_reduction_axis",
                    }
                )
                continue
        elif prior.role == "reduction_tile_extent" and len(reduction_axes) == 1:
            maps_to_reduction = True
        elif prior.role == "reduction_tile_extent" and not reduction_axes:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "reduction_tile_prior_has_no_reduction_axis",
                }
            )
            continue
        elif prior.role == "reduction_tile_extent":
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "unanchored_reduction_tile_has_multiple_reduction_axes",
                }
            )
            continue
        elif axis_ref is not None:
            maps_to_reduction = any(axis_ref in axes for axes in reduction_axes)
        elif len(reduction_axes) == 1:
            maps_to_reduction = True
        elif not reduction_axes:
            maps_to_reduction = False
        else:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "unanchored_tile_prior_has_multiple_reduction_axes",
                }
            )
            continue
        dimension = "reduction_tile" if maps_to_reduction else "parallel_tile"
        target_cap = 2 * vector_lanes if maps_to_reduction else vector_lanes
        target_value = _power_of_two_floor(
            min(float(prior.value), float(target_cap))
        )
        projected.append(
            _ProjectedCPrior(
                prior,
                dimension,
                target_value,
                (
                    "unique_reduction_axis_to_two_rvv_vectors"
                    if maps_to_reduction
                    else "parallel_axis_to_one_rvv_vector"
                ),
            )
        )
    projection = {
        "protocol": RVV_C_PROJECTION_PROTOCOL,
        "target_vlen_bits": vlen_bits,
        "element_bits": element_bits,
        "vector_lanes": vector_lanes,
        "reduction_axes": [list(item) for item in sorted(reduction_axes)],
        "has_scan": has_scan,
        "projected_priors": [item.to_record() for item in projected],
        "ignored_priors": list(ignored),
        "raw_c_value_used_as_target_tile": False,
    }
    return tuple(projected), tuple(ignored), projection


def _c_score(
    schedule: RiscvSchedule, priors: tuple[_ProjectedCPrior, ...]
) -> float:
    if not priors:
        return 0.0
    distances: list[float] = []
    for prior in priors:
        candidate = int(getattr(schedule, prior.schedule_dimension))
        if candidate <= 0:
            return float("inf")
        distances.append(
            abs(math.log2(float(candidate) / float(prior.target_value)))
        )
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


@dataclass(frozen=True)
class RiscvSearchPlan:
    universe: tuple[RiscvLoweringCandidate, ...]
    schedule_only_erase_c_order: tuple[str, ...]
    schedule_only_c_prior_order: tuple[str, ...]
    erase_c_order: tuple[str, ...]
    c_prior_order: tuple[str, ...]
    c_fixed_candidate_id: str
    c_scores: tuple[tuple[str, float | None], ...]
    consumed_c_priors: tuple[CPrior, ...]
    ignored_c_priors: tuple[dict[str, Any], ...]
    c_projection: Mapping[str, Any]

    @property
    def universe_digest(self) -> str:
        return content_digest(sorted(item.candidate_id for item in self.universe))

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": RVV_SEARCH_PLAN_PROTOCOL,
            "candidate_universe_digest": self.universe_digest,
            "candidate_universe_count": len(self.universe),
            "schedule_only_erase_c_order": list(
                self.schedule_only_erase_c_order
            ),
            "schedule_only_c_prior_order": list(
                self.schedule_only_c_prior_order
            ),
            "erase_c_order": list(self.erase_c_order),
            "c_prior_order": list(self.c_prior_order),
            "c_fixed_candidate_id": self.c_fixed_candidate_id,
            "c_scores": {
                candidate_id: score for candidate_id, score in self.c_scores
            },
            "consumed_c_priors": [
                item.to_record() for item in self.consumed_c_priors
            ],
            "ignored_c_priors": list(self.ignored_c_priors),
            "c_projection": dict(self.c_projection),
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
    projected_priors, ignored_priors, projection = _project_c_priors(state)
    scores = {
        item.candidate_id: _c_score(item.schedule, projected_priors)
        for item in universe
    }
    materialized = tuple(
        item for item in universe if not item.transformation.fuse_producers
    )
    expanded = tuple(
        item for item in universe if item.transformation.fuse_producers
    )
    schedule_erase = tuple(
        sorted(materialized, key=lambda item: _erase_schedule_key(item.schedule))
    )
    schedule_c = tuple(
        sorted(
            materialized,
            key=lambda item: (
                _c_score(item.schedule, projected_priors),
                _erase_schedule_key(item.schedule),
            ),
        )
    )
    expanded_erase = tuple(
        sorted(expanded, key=lambda item: _erase_schedule_key(item.schedule))
    )
    expanded_c = tuple(
        sorted(
            expanded,
            key=lambda item: (
                _c_score(item.schedule, projected_priors),
                _erase_schedule_key(item.schedule),
            ),
        )
    )
    # The expanded arm is nested: it traverses the exact fixed-structure
    # schedule spine before spending its remaining equal total budget on other
    # concrete paths. C-Fixed is the first target-projected schedule on that
    # canonical materialized path and is exactly one candidate.
    erase_order = (*schedule_erase, *expanded_erase)
    c_order = (*schedule_c, *expanded_c)
    c_fixed = schedule_c[0]
    return RiscvSearchPlan(
        universe=universe,
        schedule_only_erase_c_order=tuple(
            item.candidate_id for item in schedule_erase
        ),
        schedule_only_c_prior_order=tuple(
            item.candidate_id for item in schedule_c
        ),
        erase_c_order=tuple(item.candidate_id for item in erase_order),
        c_prior_order=tuple(item.candidate_id for item in c_order),
        c_fixed_candidate_id=c_fixed.candidate_id,
        c_scores=tuple(
            (
                item.candidate_id,
                None if math.isinf(scores[item.candidate_id]) else scores[item.candidate_id],
            )
            for item in universe
        ),
        consumed_c_priors=tuple(item.prior for item in projected_priors),
        ignored_c_priors=ignored_priors,
        c_projection=projection,
    )


__all__ = [
    "RVV_SEARCH_PLAN_PROTOCOL",
    "RVV_C_PROJECTION_PROTOCOL",
    "RiscvSearchPlan",
    "enumerate_riscv_candidates",
    "plan_riscv_search",
]
