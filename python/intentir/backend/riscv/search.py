"""Candidate-universe and discardable-C planning for the native RVV backend.

The candidate universe is constructed without consulting C.  ``Attach_C`` may
only reorder that fixed universe; erasing C therefore changes neither
admission nor the executable states available to the search.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, replace
from typing import Any, Mapping

from intentir.backend.materialize import BackendSearchState, CPrior
from intentir.backend.riscv.compile import (
    RiscvLoweringCandidate,
    RiscvSchedule,
    RiscvTransformationState,
)


RVV_SEARCH_PLAN_PROTOCOL = "intentir.backend.riscv.search-plan.v7"
RVV_C_PROJECTION_PROTOCOL = "intentir.backend.riscv.c-projection.v1"

_PARALLEL_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_REDUCTION_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_JOINT_TILES = ((1, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 128))


def _schedule_universe() -> tuple[RiscvSchedule, ...]:
    schedules = [
        RiscvSchedule(opt, vectorize, 128, 0, 0, thread_count, False)
        for thread_count in (1,)
        for opt in (2, 3)
        for vectorize in (False, True)
    ]
    # RVV's scalable-vector policy is a real LLVM schedule state, not a
    # spelling of an arbitrary VLEN.  Keep the untiled anchors in the complete
    # space so a target with a different implementation VLEN can choose
    # strip-mined loops without changing J or the MLIR graph.
    schedules.append(RiscvSchedule(3, True, 0, 0, 0, 1, True))
    schedules.extend(
        RiscvSchedule(3, True, 128, parallel, reduction, thread_count, False)
        for thread_count in (1,)
        for parallel, reduction in (
            *((value, 0) for value in _PARALLEL_TILES),
            *((0, value) for value in _REDUCTION_TILES),
            *_JOINT_TILES,
        )
    )
    # A small set of tiled scalable states is enough to expose the interaction
    # between LLVM strip-mining and the MLIR retile.  The exhaustive fixed-VLEN
    # tile domain remains intact; scalable candidates are not represented by a
    # fake SEW/LMUL knob.
    schedules.extend(
        RiscvSchedule(3, True, 0, parallel, reduction, 1, True)
        for parallel, reduction in (
            (4, 0),
            (16, 0),
            (64, 0),
            (0, 4),
            (0, 16),
            (0, 64),
            *_JOINT_TILES,
        )
    )
    unique: dict[
        tuple[int, bool, int, int, int, int, bool, int], RiscvSchedule
    ] = {}
    for item in schedules:
        key = (
            item.opt_level,
            item.vectorize,
            item.vector_bits,
            item.parallel_tile,
            item.reduction_tile,
            item.thread_count,
            item.scalable_vectorization,
            item.vector_lanes,
        )
        unique.setdefault(key, item)
    return tuple(unique[key] for key in sorted(unique))


def enumerate_riscv_candidates(
    *,
    enable_b_transformations: bool = True,
    enable_reassociation: bool = True,
) -> tuple[RiscvLoweringCandidate, ...]:
    """Build the target schedule/structure universe without reading C."""

    transformations = [RiscvTransformationState(False, False, False)]
    if enable_b_transformations:
        transformations.append(RiscvTransformationState(True, False, False))
    if enable_reassociation:
        transformations.append(RiscvTransformationState(False, True, False))
        transformations.append(RiscvTransformationState(False, True, True))
        if enable_b_transformations:
            transformations.append(RiscvTransformationState(True, True, False))
            transformations.append(RiscvTransformationState(True, True, True))
            transformations.append(RiscvTransformationState(True, True, True, True))
            transformations.append(
                RiscvTransformationState(False, True, False, False, True)
            )
    schedules = _schedule_universe()
    candidates: list[RiscvLoweringCandidate] = []
    for transformation in transformations:
        for base_schedule in schedules:
            if (
                transformation.explicit_vector_reduction
                and not base_schedule.scalable_vectorization
            ):
                continue
            if transformation.fused_softmax and (
                not base_schedule.vectorize
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            lane_values = (
                (4,)
                if not transformation.fused_softmax
                or base_schedule.scalable_vectorization
                else (2, 4, 8, 16)
            )
            candidates.extend(
                RiscvLoweringCandidate(
                    transformation,
                    replace(base_schedule, vector_lanes=lanes),
                )
                for lanes in lane_values
            )
    return tuple(candidates)


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


def _operation_facts(
    value: Any,
) -> tuple[set[tuple[int, ...]], bool, list[int], bool]:
    reduction_axes: set[tuple[int, ...]] = set()
    has_scan = False
    float_bits: list[int] = []
    has_reassociable_float_reduction = False

    def visit(item: Any) -> None:
        nonlocal has_scan, has_reassociable_float_reduction
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
                if (
                    op == "reduce"
                    and isinstance(attrs, Mapping)
                    and attrs.get("assoc") == "float_nondeterministic"
                ):
                    has_reassociable_float_reduction = True
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
    return (
        reduction_axes,
        has_scan,
        float_bits,
        has_reassociable_float_reduction,
    )


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
    reduction_axes, has_scan, float_bits, _has_reassociation = (
        _operation_facts(payload)
    )
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
        schedule.scalable_vectorization,
        -schedule.opt_level,
        schedule.thread_count,
        abs(schedule.vector_lanes - 4),
    )


def _c_ranked_with_robust_anchors(
    candidates: tuple[RiscvLoweringCandidate, ...],
    priors: tuple[_ProjectedCPrior, ...],
) -> tuple[RiscvLoweringCandidate, ...]:
    """Rank by C while retaining early target-independent anchor schedules.

    C is a discardable search prior rather than an oracle for a different
    target.  The first state therefore remains the strongest projected-C
    candidate (the C-Fixed arm), but a finite search budget must also see the
    two robust untiled O3 anchors before spending all visits around that
    projection.  This preserves the candidate universe and lets C accelerate
    discovery without making a stale cross-target tile actively harmful.
    """

    if not priors:
        return tuple(sorted(candidates, key=lambda item: _erase_schedule_key(item.schedule)))

    ranked_groups: list[tuple[RiscvLoweringCandidate, ...]] = []
    projected_dimensions = {item.schedule_dimension for item in priors}

    def unprojected_tile_count(candidate: RiscvLoweringCandidate) -> int:
        # C-Fixed is the mechanical projection of the authored C, not an
        # opportunity to silently activate another schedule axis.  A tied
        # candidate therefore keeps every unmentioned tile dimension at its
        # canonical zero value.  Schedule-Only still contains the complete
        # target-independent universe and may subsequently explore it.
        schedule = candidate.schedule
        return sum(
            int(getattr(schedule, dimension) != 0)
            for dimension in ("parallel_tile", "reduction_tile")
            if dimension not in projected_dimensions
        )

    for thread_count in sorted({item.schedule.thread_count for item in candidates}):
        group = tuple(
            item for item in candidates if item.schedule.thread_count == thread_count
        )
        ranked = tuple(
            sorted(
                group,
                key=lambda item: (
                    _c_score(item.schedule, priors),
                    unprojected_tile_count(item),
                    _erase_schedule_key(item.schedule),
                ),
            )
        )
        first = ranked[0]
        anchors: list[RiscvLoweringCandidate] = []
        for vectorize, scalable in (
            (True, False),
            (True, True),
            (False, False),
        ):
            anchor = next(
                (
                    item
                    for item in group
                    if item.schedule.opt_level == 3
                    and item.schedule.vectorize is vectorize
                    and item.schedule.scalable_vectorization is scalable
                    and item.schedule.parallel_tile == 0
                    and item.schedule.reduction_tile == 0
                ),
                None,
            )
            if anchor is not None and anchor != first:
                anchors.append(anchor)
        promoted = {first.candidate_id, *(item.candidate_id for item in anchors)}
        ranked_groups.append(
            (
                first,
                *anchors,
                *(item for item in ranked if item.candidate_id not in promoted),
            )
        )
    return tuple(
        group[index]
        for index in range(max(len(group) for group in ranked_groups))
        for group in ranked_groups
        if index < len(group)
    )


def _interleave_lowering_paths(
    canonical: tuple[RiscvLoweringCandidate, ...],
    expanded_paths: tuple[tuple[RiscvLoweringCandidate, ...], ...],
) -> tuple[RiscvLoweringCandidate, ...]:
    """Give each ranked schedule one visit for every admitted MLIR state.

    A state+schedule budget must actually reach both concrete MLIR paths.  The
    previous concatenation spent every practical budget on the canonical path,
    making E3 observationally identical to E2 even when a faster fused state
    existed. Interleaving preserves each state's own valid schedule order while
    preventing fusion or explicit reduction reassociation from being starved;
    different MLIR states need not expose identical schedule domains.
    """

    if not expanded_paths:
        return canonical
    result: list[RiscvLoweringCandidate] = []
    all_paths = (canonical, *expanded_paths)
    for index in range(max(len(path) for path in all_paths)):
        result.extend(path[index] for path in all_paths if index < len(path))
    return tuple(result)


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

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": RVV_SEARCH_PLAN_PROTOCOL,
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
    _axes, _has_scan, _float_bits, has_reassociation = _operation_facts(
        state.base.j.to_json()
    )
    projected_priors, ignored_priors, projection = _project_c_priors(state)
    raw_universe = enumerate_riscv_candidates(
        enable_b_transformations=enable_b_transformations,
        enable_reassociation=has_reassociation,
    )
    minimum_reduction_tile = int(projection["vector_lanes"])
    universe = tuple(
        item
        for item in raw_universe
        if item.schedule.reduction_tile == 0
        or item.schedule.reduction_tile >= minimum_reduction_tile
    )
    projection = {
        **projection,
        "minimum_reduction_tile": minimum_reduction_tile,
        "subvector_reduction_schedules_rejected": len(raw_universe) - len(universe),
        "subvector_reduction_rule": (
            "a vectorized reduction tile must cover at least one target vector"
        ),
    }
    scores = {
        item.candidate_id: _c_score(item.schedule, projected_priors)
        for item in universe
    }
    canonical = tuple(
        item
        for item in universe
        if not item.transformation.fuse_producers
        and not item.transformation.reassociate_reductions
    )
    alternative_states = tuple(
        dict.fromkeys(
            (
                item.transformation.fuse_producers,
                item.transformation.reassociate_reductions,
                item.transformation.explicit_vector_reduction,
                item.transformation.joint_vector_reduction,
                item.transformation.fused_softmax,
            )
            for item in universe
            if item not in canonical
        )
    )
    expanded = tuple(
        tuple(
            item
            for item in universe
            if (
                item.transformation.fuse_producers,
                item.transformation.reassociate_reductions,
                item.transformation.explicit_vector_reduction,
                item.transformation.joint_vector_reduction,
                item.transformation.fused_softmax,
            )
            == state_key
        )
        for state_key in alternative_states
    )
    schedule_erase = tuple(
        sorted(canonical, key=lambda item: _erase_schedule_key(item.schedule))
    )
    schedule_c = _c_ranked_with_robust_anchors(canonical, projected_priors)
    expanded_erase = tuple(
        tuple(sorted(path, key=lambda item: _erase_schedule_key(item.schedule)))
        for path in expanded
    )
    expanded_c = tuple(
        _c_ranked_with_robust_anchors(path, projected_priors)
        for path in expanded
    )
    # E2 retains the complete canonical schedule order. E3 has the same total
    # visit budget but round-robin interleaves canonical and structurally
    # expanded MLIR. Each state owns its valid schedule domain; in particular,
    # the dynamic-VL explicit vector state only admits scalable schedules.
    # C-Fixed remains exactly the first target-projected canonical candidate.
    erase_order = _interleave_lowering_paths(schedule_erase, expanded_erase)
    c_order = _interleave_lowering_paths(schedule_c, expanded_c)
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
