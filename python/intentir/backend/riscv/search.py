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


RVV_SEARCH_PLAN_PROTOCOL = "intentir.backend.riscv.search-plan.v11"
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
    enable_rowmask_zero_fill: bool = False,
    enable_composed_gather: bool = False,
    enable_sibling_output_fusion: bool = False,
    enable_paired_rotation: bool = False,
    enable_streaming_attention: bool = False,
) -> tuple[RiscvLoweringCandidate, ...]:
    """Build the target schedule/structure universe without reading C."""

    transformations = [RiscvTransformationState(False, False, False)]
    if enable_b_transformations:
        transformations.append(RiscvTransformationState(True, False, False))
    if enable_rowmask_zero_fill:
        transformations.append(
            RiscvTransformationState(False, rowmask_zero_fill=True)
        )
    if enable_composed_gather:
        transformations.append(
            RiscvTransformationState(False, composed_indexed_gather=True)
        )
    if enable_sibling_output_fusion and enable_b_transformations:
        transformations.append(
            RiscvTransformationState(True, fuse_sibling_outputs=True)
        )
    if enable_paired_rotation:
        # The contiguous direct loop is the canonical complete consumer of the
        # closed paired-rotation view, not a Full-only optimization.  Full may
        # additionally change loop topology by hoisting angles across heads.
        transformations.append(
            RiscvTransformationState(False, paired_rotation=True)
        )
        if enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    False,
                    paired_rotation=True,
                    paired_rotation_head_angle_reuse=True,
                )
            )
    if enable_reassociation:
        transformations.append(RiscvTransformationState(False, True, False))
        if enable_streaming_attention and enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    False,
                    True,
                    streaming_attention=True,
                )
            )
        if enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    True,
                    True,
                    mixed_precision_dot_reduction=True,
                )
            )
        transformations.append(RiscvTransformationState(False, True, True))
        if enable_b_transformations:
            # The two Softmax states are placed before the generic fused
            # reduction states so a practical round-robin budget observes both
            # the execution-domain checked vector-exp path and the exact-exp
            # fallback.  This is ordering only; C never admits either state.
            transformations.append(
                RiscvTransformationState(
                    False,
                    True,
                    fused_softmax=True,
                    approximate_exp=True,
                )
            )
            transformations.append(
                RiscvTransformationState(False, True, fused_softmax=True)
            )
            transformations.append(RiscvTransformationState(True, True, False))
            transformations.append(RiscvTransformationState(True, True, True))
            transformations.append(RiscvTransformationState(True, True, True, True))
    schedules = _schedule_universe()
    candidates: list[RiscvLoweringCandidate] = []
    for transformation in transformations:
        for base_schedule in schedules:
            if transformation.paired_rotation and (
                base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or (
                    transformation.paired_rotation_head_angle_reuse
                    and base_schedule.vectorize
                )
            ):
                continue
            if transformation.rowmask_zero_fill and (
                base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or base_schedule.scalable_vectorization
            ):
                continue
            if transformation.composed_indexed_gather and (
                not base_schedule.vectorize
                or base_schedule.scalable_vectorization
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if (
                transformation.explicit_vector_reduction
                and not base_schedule.scalable_vectorization
            ):
                continue
            if transformation.mixed_precision_dot_reduction and (
                not base_schedule.vectorize
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if transformation.fused_softmax and (
                not base_schedule.vectorize
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or (
                    transformation.approximate_exp
                    and base_schedule.scalable_vectorization
                )
            ):
                continue
            if transformation.streaming_attention and (
                not base_schedule.vectorize
                or base_schedule.scalable_vectorization
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if transformation.streaming_attention:
                lane_values = (2, 4, 8, 16)
            elif transformation.mixed_precision_dot_reduction:
                lane_values = (
                    (4,) if base_schedule.scalable_vectorization else (16,)
                )
            else:
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
) -> tuple[set[tuple[int, ...]], bool, list[int], bool, bool]:
    reduction_axes: set[tuple[int, ...]] = set()
    has_scan = False
    float_bits: list[int] = []
    has_reassociable_float_reduction = False
    has_contraction = False

    def visit(item: Any) -> None:
        nonlocal has_scan, has_reassociable_float_reduction, has_contraction
        if isinstance(item, Mapping):
            dtype = item.get("dtype")
            if isinstance(dtype, str):
                bits = _dtype_bits(dtype)
                if bits is not None:
                    float_bits.append(bits)
            op = item.get("op")
            has_contraction = has_contraction or op == "contraction"
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
        has_contraction,
    )


def _has_typed_rowmask_graph(value: Mapping[str, Any]) -> bool:
    intent = value.get("intent")
    if not isinstance(intent, Mapping):
        return False
    function = intent.get("intent_function")
    if not isinstance(function, Mapping):
        return False
    inputs = function.get("inputs")
    body = function.get("body")
    outputs = function.get("outputs")
    if not isinstance(inputs, list) or not isinstance(body, list) or not isinstance(outputs, list):
        return False
    input_by_name = {
        item.get("name"): item
        for item in inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(body) != 2 or len(outputs) != 1:
        return False
    cast, product = body
    if not isinstance(cast, Mapping) or not isinstance(product, Mapping):
        return False
    cast_operands = cast.get("operands")
    product_operands = product.get("operands")
    cast_type = cast.get("type")
    product_type = product.get("type")
    attrs = product.get("attrs")
    if (
        cast.get("op") != "cast"
        or not isinstance(cast_operands, list)
        or len(cast_operands) != 1
        or not isinstance(cast_type, Mapping)
        or cast_type.get("dtype") != "f32"
        or cast.get("result") is None
        or product.get("op") != "map_binary"
        or not isinstance(attrs, Mapping)
        or attrs.get("fn", "mul") != "mul"
        or not isinstance(product_operands, list)
        or len(product_operands) != 2
        or cast.get("result") not in product_operands
        or not isinstance(product_type, Mapping)
        or product_type.get("dtype") != "f32"
    ):
        return False
    mask = input_by_name.get(cast_operands[0])
    data_name = next(
        (operand for operand in product_operands if operand != cast.get("result")),
        None,
    )
    data = input_by_name.get(data_name)
    mask_shape = mask.get("shape") if isinstance(mask, Mapping) else None
    data_shape = data.get("shape") if isinstance(data, Mapping) else None
    result_shape = product_type.get("shape")
    output = outputs[0]
    return bool(
        isinstance(mask, Mapping)
        and mask.get("dtype") == "i1"
        and isinstance(mask_shape, list)
        and len(mask_shape) == 1
        and isinstance(data, Mapping)
        and data.get("dtype") == "f32"
        and isinstance(data_shape, list)
        and len(data_shape) == 2
        and data_shape[0] == mask_shape[0]
        and result_shape == data_shape
        and isinstance(output, Mapping)
        and output.get("name") == product.get("result")
        and output.get("dtype") == "f32"
        and output.get("shape") == data_shape
    )


def _has_typed_gather_chain(value: Mapping[str, Any]) -> bool:
    intent = value.get("intent")
    if not isinstance(intent, Mapping):
        return False
    function = intent.get("intent_function")
    if not isinstance(function, Mapping):
        return False
    inputs = function.get("inputs")
    body = function.get("body")
    outputs = function.get("outputs")
    if (
        not isinstance(inputs, list)
        or not isinstance(body, list)
        or not isinstance(outputs, list)
        or len(body) != 2
        or len(outputs) != 1
    ):
        return False
    first, second = body
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    first_operands = first.get("operands")
    second_operands = second.get("operands")
    first_attrs = first.get("attrs")
    second_attrs = second.get("attrs")
    first_type = first.get("type")
    second_type = second.get("type")
    output = outputs[0]
    input_by_name = {
        item.get("name"): item
        for item in inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if (
        first.get("op") != "gather"
        or second.get("op") != "gather"
        or not isinstance(first_operands, list)
        or len(first_operands) != 2
        or not isinstance(second_operands, list)
        or len(second_operands) != 2
        or second_operands[0] != first.get("result")
        or not isinstance(first_attrs, Mapping)
        or first_attrs.get("mode") != "take"
        or first_attrs.get("axis") != 0
        or not isinstance(second_attrs, Mapping)
        or second_attrs.get("mode") != "along_axis"
        or second_attrs.get("axis") != 1
        or not isinstance(first_type, Mapping)
        or first_type.get("dtype") != "f32"
        or not isinstance(second_type, Mapping)
        or second_type.get("dtype") != "f32"
        or not isinstance(output, Mapping)
        or output.get("name") != second.get("result")
    ):
        return False
    data = input_by_name.get(first_operands[0])
    rows = input_by_name.get(first_operands[1])
    columns = input_by_name.get(second_operands[1])
    return bool(
        isinstance(data, Mapping)
        and data.get("dtype") == "f32"
        and isinstance(data.get("shape"), list)
        and len(data["shape"]) == 2
        and isinstance(rows, Mapping)
        and rows.get("dtype") == "i32"
        and isinstance(rows.get("shape"), list)
        and len(rows["shape"]) == 1
        and isinstance(columns, Mapping)
        and columns.get("dtype") == "i32"
        and columns.get("shape") == rows.get("shape")
        and second_type.get("shape") == rows.get("shape")
    )


def _has_two_public_outputs(value: Mapping[str, Any]) -> bool:
    intent = value.get("intent")
    if not isinstance(intent, Mapping):
        return False
    function = intent.get("intent_function")
    if not isinstance(function, Mapping):
        return False
    outputs = function.get("outputs")
    return bool(
        isinstance(outputs, list)
        and len(outputs) == 2
        and all(isinstance(output, Mapping) for output in outputs)
    )


def _has_typed_flash_attention_graph(value: Mapping[str, Any]) -> bool:
    """Match the closed structured stable-attention J without kernel names."""

    intent = value.get("intent")
    boundary = value.get("boundary")
    constraints = value.get("dimension_constraints")
    if (
        not isinstance(intent, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(constraints, list)
        or boundary.get("aliases") not in (None, [])
    ):
        return False
    function = intent.get("intent_function")
    if not isinstance(function, Mapping):
        return False
    inputs = function.get("inputs")
    outputs = function.get("outputs")
    body = function.get("body")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not isinstance(body, list)
        or len(inputs) != 6
        or len(outputs) != 3
    ):
        return False

    input_by_name = {
        item.get("name"): item
        for item in inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    output_by_name = {
        item.get("name"): item
        for item in outputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    operations: list[Mapping[str, Any]] = []

    def visit(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, Mapping):
                continue
            operations.append(item)
            visit(item.get("body"))
            for arm in ("positive", "negative"):
                region = item.get(arm)
                if isinstance(region, Mapping):
                    visit(region.get("body"))

    visit(body)
    contractions = [item for item in operations if item.get("op") == "contraction"]
    loops = [item for item in operations if item.get("op") == "loop"]
    scatters = [item for item in operations if item.get("op") == "scatter"]
    reductions = [item for item in operations if item.get("op") == "reduce"]
    if (
        len(contractions) != 2
        or len(loops) != 1
        or len(scatters) != 3
        or len(reductions) != 2
    ):
        return False
    qk = next(
        (
            item
            for item in contractions
            if isinstance(item.get("attrs"), Mapping)
            and item["attrs"].get("subscripts") == "bqhd,bkhd->bhqk"
        ),
        None,
    )
    pv = next(
        (
            item
            for item in contractions
            if isinstance(item.get("attrs"), Mapping)
            and item["attrs"].get("subscripts") == "bhqk,bkhd->bqhd"
        ),
        None,
    )
    if qk is None or pv is None:
        return False
    qk_operands = qk.get("operands")
    pv_operands = pv.get("operands")
    if (
        not isinstance(qk_operands, list)
        or len(qk_operands) != 2
        or not isinstance(pv_operands, list)
        or len(pv_operands) != 2
    ):
        return False
    scans = loops[0].get("scan")
    scan_sources = {
        item.get("name"): item.get("from")
        for item in scans
        if isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("from"), str)
    } if isinstance(scans, list) else {}

    def boundary_input(name: Any) -> Any:
        return input_by_name.get(scan_sources.get(name, name))

    query = boundary_input(qk_operands[0])
    key = boundary_input(qk_operands[1])
    value_input = boundary_input(pv_operands[1])
    if not all(isinstance(item, Mapping) for item in (query, key, value_input)):
        return False
    query_shape = query.get("shape")
    key_shape = key.get("shape")
    value_shape = value_input.get("shape")
    if (
        query.get("dtype") != "f16"
        or key.get("dtype") != "f16"
        or value_input.get("dtype") != "f16"
        or not isinstance(query_shape, list)
        or not isinstance(key_shape, list)
        or len(query_shape) != 4
        or len(key_shape) != 4
        or value_shape != key_shape
        or query_shape[0] != key_shape[0]
        or query_shape[2:] != key_shape[2:]
    ):
        return False
    rank_four_outputs = [
        item
        for item in output_by_name.values()
        if item.get("dtype") == "f16" and item.get("shape") == query_shape
    ]
    rank_three_outputs = [
        item
        for item in output_by_name.values()
        if item.get("dtype") == "f32"
        and isinstance(item.get("shape"), list)
        and len(item["shape"]) == 3
        and item["shape"][:2] == [query_shape[0], query_shape[2]]
    ]
    scalar_f32_inputs = [
        item
        for item in input_by_name.values()
        if item.get("dtype") == "f32" and item.get("shape") == []
    ]
    scratch_inputs = [
        item
        for item in input_by_name.values()
        if item.get("dtype") == "f32"
        and isinstance(item.get("shape"), list)
        and len(item["shape"]) == 3
        and item["shape"][:2] == [query_shape[0], query_shape[2]]
    ]
    if (
        len(rank_four_outputs) != 1
        or len(rank_three_outputs) != 2
        or len(scalar_f32_inputs) != 1
        or len(scratch_inputs) != 1
    ):
        return False
    loop_trip = loops[0].get("trip")
    if not isinstance(loop_trip, Mapping) or loop_trip.get("axis") != key_shape[1]:
        return False
    reduction_facts = {
        (
            item.get("attrs", {}).get("fn"),
            tuple(item.get("attrs", {}).get("axes", [])),
            item.get("attrs", {}).get("assoc"),
        )
        for item in reductions
        if isinstance(item.get("attrs"), Mapping)
    }
    if reduction_facts != {
        ("max", (3,), "float_nondeterministic"),
        ("sum", (3,), "float_nondeterministic"),
    }:
        return False
    if any(
        not isinstance(item.get("attrs"), Mapping)
        or item["attrs"].get("axis") != 2
        or item["attrs"].get("mode") != "along_axis"
        or item["attrs"].get("reduce") != "none"
        for item in scatters
    ):
        return False
    updates = boundary.get("updates")
    relations = boundary.get("result_relations")
    if (
        not isinstance(updates, list)
        or len(updates) != 3
        or any(
            not isinstance(item, Mapping)
            or item.get("axis") != 2
            or item.get("combiner") != "overwrite"
            or item.get("duplicate_policy") != "forbid"
            or item.get("order") != "input_order"
            for item in updates
        )
        or not isinstance(relations, list)
        or len(relations) != 3
        or any(
            not isinstance(item, Mapping) or item.get("kind") != "tolerance"
            for item in relations
        )
    ):
        return False
    padded = scratch_inputs[0]["shape"][2]
    return any(
        isinstance(item, str)
        and item.replace(" ", "")
        in {
            f"{query_shape[1]}<={padded}",
            f"{padded}>={query_shape[1]}",
        }
        for item in constraints
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
    reduction_axes, has_scan, float_bits, _has_reassociation, _has_contraction = (
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
    j_payload = state.base.j.to_json()
    reduction_axes, _has_scan, _float_bits, has_reassociation, has_contraction = (
        _operation_facts(j_payload)
    )
    has_typed_rowmask = _has_typed_rowmask_graph(j_payload)
    has_typed_gather_chain = _has_typed_gather_chain(j_payload)
    has_paired_rotation = (
        'intent.paired_rotation = "interleaved"' in state.base.intent_mlir
        and "intentir.paired-last-axis-rotation.v1" in state.base.intent_mlir
    )
    has_streaming_attention = _has_typed_flash_attention_graph(j_payload)
    has_sibling_outputs = (
        _has_two_public_outputs(j_payload)
        and not reduction_axes
        and not has_paired_rotation
    )
    projected_priors, ignored_priors, projection = _project_c_priors(state)
    raw_universe = enumerate_riscv_candidates(
        enable_b_transformations=enable_b_transformations,
        enable_reassociation=has_reassociation,
        enable_rowmask_zero_fill=has_typed_rowmask,
        enable_composed_gather=has_typed_gather_chain,
        enable_sibling_output_fusion=has_sibling_outputs,
        enable_paired_rotation=has_paired_rotation,
        enable_streaming_attention=has_streaming_attention,
    )
    minimum_reduction_tile = int(projection["vector_lanes"])
    subvector_reduction_schedules_rejected = sum(
        0 < item.schedule.reduction_tile < minimum_reduction_tile
        for item in raw_universe
    )
    contraction_reduction_schedules_rejected = sum(
        has_contraction and item.schedule.reduction_tile > 0
        for item in raw_universe
    )
    universe = tuple(
        item
        for item in raw_universe
        if item.schedule.reduction_tile == 0
        or (
            not has_contraction
            and item.schedule.reduction_tile >= minimum_reduction_tile
        )
    )
    projection = {
        **projection,
        "minimum_reduction_tile": minimum_reduction_tile,
        "target_schedule_rejections": len(raw_universe) - len(universe),
        "subvector_reduction_schedules_rejected": (
            subvector_reduction_schedules_rejected
        ),
        "subvector_reduction_rule": (
            "a vectorized reduction tile must cover at least one target vector"
        ),
        "generic_contraction_reduction_tiling_admitted": not has_contraction,
        "generic_contraction_reduction_schedules_rejected": (
            contraction_reduction_schedules_rejected
        ),
    }
    scores = {
        item.candidate_id: _c_score(item.schedule, projected_priors)
        for item in universe
    }
    if has_paired_rotation:
        # Once the J has a closed interleaved paired view, carrying the exploded
        # 215-op materialized graph as C-Fixed would recreate the invalid
        # generic-fallback-vs-complete-consumer ablation.  Transport only the
        # complete contiguous spine and its genuine head-reuse alternative.
        universe = tuple(
            item for item in universe if item.transformation.paired_rotation
        )
        canonical = tuple(
            item
            for item in universe
            if not item.transformation.paired_rotation_head_angle_reuse
        )
    else:
        canonical = tuple(
            item
            for item in universe
            if not item.transformation.fuse_producers
            and not item.transformation.reassociate_reductions
            and not item.transformation.rowmask_zero_fill
            and not item.transformation.composed_indexed_gather
            and not item.transformation.fuse_sibling_outputs
            and not item.transformation.paired_rotation
        )
    alternative_states = tuple(
        dict.fromkeys(
            (
                item.transformation.fuse_producers,
                item.transformation.reassociate_reductions,
                item.transformation.explicit_vector_reduction,
                item.transformation.joint_vector_reduction,
                item.transformation.fused_softmax,
                item.transformation.rowmask_zero_fill,
                item.transformation.composed_indexed_gather,
                item.transformation.fuse_sibling_outputs,
                item.transformation.approximate_exp,
                item.transformation.mixed_precision_dot_reduction,
                item.transformation.paired_rotation,
                item.transformation.paired_rotation_head_angle_reuse,
                item.transformation.streaming_attention,
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
                item.transformation.rowmask_zero_fill,
                item.transformation.composed_indexed_gather,
                item.transformation.fuse_sibling_outputs,
                item.transformation.approximate_exp,
                item.transformation.mixed_precision_dot_reduction,
                item.transformation.paired_rotation,
                item.transformation.paired_rotation_head_angle_reuse,
                item.transformation.streaming_attention,
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
