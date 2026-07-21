"""J-owned exact, tolerance, and allowed-update relations for Gate2."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.model import (
    DuplicatePolicy,
    LogicalUpdateSpec,
    ResultRelationKind,
    ResultRelationSpec,
    SemanticArchive,
)

from .model import OutputAgreement, PairwiseAgreement


GATE2_COMPARISON_PROTOCOL = "canonical_j_result_relation_v3"

_DTYPE_ALIASES = {
    "bool": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
}


def _dtype_key(value: str) -> str:
    normalized = value.removeprefix("torch.").lower()
    return _DTYPE_ALIASES.get(normalized, normalized)


def _exact_element_domain(left: np.ndarray, right: np.ndarray) -> bool:
    exact_kinds = {"b", "i", "u"}
    return left.dtype.kind in exact_kinds and right.dtype.kind in exact_kinds


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    if _exact_element_domain(left, right):
        if bool(np.array_equal(left, right)):
            return 0.0
        return float(
            max(
                abs(int(lhs) - int(rhs))
                for lhs, rhs in zip(left.flat, right.flat, strict=True)
            )
        )
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not bool(np.any(finite)):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _canonical_node_ref(j: SemanticArchive, ref: str) -> str:
    return j.node_ids.get(ref, ref)


def _update_for_relation(
    j: SemanticArchive, relation: ResultRelationSpec
) -> LogicalUpdateSpec:
    assert relation.update_node_ref is not None
    target = _canonical_node_ref(j, relation.update_node_ref)
    matches = [
        item
        for item in j.boundary.updates
        if _canonical_node_ref(j, item.node_ref) == target
    ]
    if len(matches) != 1:
        raise ValueError("one_of_updates relation has no unique logical update")
    return matches[0]


def _j_inputs(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    expected = {item.name for item in j.intent.inputs}
    if set(input_bindings) != expected:
        raise ValueError("Gate2 input bindings differ from canonical J inputs")
    try:
        result = {
            name: np.asarray(values[runtime_name])
            for name, runtime_name in input_bindings.items()
        }
    except KeyError as exc:
        raise ValueError(f"Gate2 is missing input {exc.args[0]!r}") from exc
    dimension_extents: dict[str, int] = {}
    for tensor in j.intent.inputs:
        value = result[tensor.name]
        actual_dtype = _dtype_key(value.dtype.name)
        expected_dtype = _dtype_key(tensor.dtype)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Gate2 input {tensor.name!r} dtype {actual_dtype!r} does not "
                f"match J dtype {expected_dtype!r}"
            )
        if value.ndim != len(tensor.shape):
            raise ValueError(
                f"Gate2 input {tensor.name!r} rank differs from canonical J"
            )
        for axis, (declared, actual) in enumerate(
            zip(tensor.shape, value.shape, strict=True)
        ):
            if isinstance(declared, int) and declared != actual:
                raise ValueError(
                    f"Gate2 input {tensor.name!r} extent at axis {axis} "
                    "differs from canonical J"
                )
            if isinstance(declared, str):
                previous = dimension_extents.setdefault(declared, int(actual))
                if previous != actual:
                    raise ValueError(
                        f"Gate2 symbolic extent {declared!r} is inconsistent"
                    )
    return result


def _active_update_groups(
    update: LogicalUpdateSpec,
    j_inputs: Mapping[str, np.ndarray],
    *,
    require_values: bool,
) -> tuple[np.ndarray, dict[tuple[int, ...], list[np.generic]]]:
    missing = [
        ref
        for ref in (update.base, update.indices)
        if ref not in j_inputs
    ]
    if require_values and update.values not in j_inputs:
        missing.append(update.values)
    if missing:
        raise NotImplementedError(
            "Gate2 update relation currently requires external J input ref(s): "
            + ", ".join(sorted(set(missing)))
        )

    base = np.asarray(j_inputs[update.base])
    indices = np.asarray(j_inputs[update.indices])
    if indices.dtype.kind not in {"i", "u"}:
        raise ValueError("logical update indices must be integer-typed")
    if update.axis >= base.ndim or indices.ndim != base.ndim:
        raise ValueError("logical update rank/axis violates canonical J")
    values = (
        np.asarray(j_inputs[update.values])
        if require_values
        else np.zeros(indices.shape, dtype=base.dtype)
    )
    if values.shape != indices.shape:
        raise ValueError("logical update indices and values shapes differ")
    groups: dict[tuple[int, ...], list[np.generic]] = {}
    for coordinate in np.ndindex(indices.shape):
        destination = list(coordinate)
        index = int(indices[coordinate])
        if index < 0 or index >= base.shape[update.axis]:
            raise ValueError("logical update violates its index-bounds relation")
        destination[update.axis] = index
        if any(
            position < 0 or position >= base.shape[axis]
            for axis, position in enumerate(destination)
        ):
            raise ValueError("logical update non-axis coordinate is out of bounds")
        groups.setdefault(tuple(destination), []).append(values[coordinate])
    return base, groups


def validate_j_inputs(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> None:
    """Validate typed input-domain and uniqueness preconditions before running."""

    j_inputs = _j_inputs(j, input_bindings, values)
    for item in j.boundary.index_bounds:
        if item.data not in j_inputs or item.indices not in j_inputs:
            raise NotImplementedError(
                "Gate2 index bounds currently require external data/indices inputs"
            )
        data = np.asarray(j_inputs[item.data])
        indices = np.asarray(j_inputs[item.indices])
        if item.axis >= data.ndim or indices.dtype.kind not in {"i", "u"}:
            raise ValueError("Gate2 indexed-domain input violates canonical J")
        if bool(np.any(indices < 0)) or bool(
            np.any(indices >= data.shape[item.axis])
        ):
            raise ValueError(
                f"Gate2 index-bounds precondition failed at {item.node_ref!r}"
            )
    for update in j.boundary.updates:
        if update.duplicate_policy is not DuplicatePolicy.FORBID:
            continue
        _base, groups = _active_update_groups(
            update, j_inputs, require_values=False
        )
        if any(len(candidates) != 1 for candidates in groups.values()):
            raise ValueError(
                f"Gate2 unique-index precondition failed at {update.node_ref!r}"
            )


def _scalar_equal(left: Any, right: Any) -> bool:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.dtype.kind == "f" and rhs.dtype.kind == "f":
        if bool(np.isnan(lhs)) and bool(np.isnan(rhs)):
            return True
    return bool(lhs == rhs)


def _one_of_membership(
    candidate: np.ndarray,
    relation: ResultRelationSpec,
    j: SemanticArchive,
    j_inputs: Mapping[str, np.ndarray],
) -> tuple[bool, str]:
    update = _update_for_relation(j, relation)
    if update.duplicate_policy is not DuplicatePolicy.ONE_OF:
        return False, "one_of_updates relation targets a non-one-of update"
    try:
        base, groups = _active_update_groups(
            update, j_inputs, require_values=True
        )
    except (NotImplementedError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    candidate = np.asarray(candidate)
    if candidate.shape != base.shape:
        return False, f"shape mismatch against update base: {candidate.shape} != {base.shape}"
    if candidate.dtype != base.dtype:
        return False, f"dtype mismatch against update base: {candidate.dtype} != {base.dtype}"

    touched = np.zeros(base.shape, dtype=np.bool_)
    for destination in groups:
        touched[destination] = True
    if not bool(np.array_equal(candidate[~touched], base[~touched], equal_nan=True)):
        return False, "one_of_updates changed an untouched destination"
    for destination, contenders in groups.items():
        if not any(
            _scalar_equal(candidate[destination], contender)
            for contender in contenders
        ):
            return False, f"value at {destination} is not an active update contender"
    return True, ""


def compare_outputs(
    left_name: str,
    left: Mapping[str, np.ndarray],
    right_name: str,
    right: Mapping[str, np.ndarray],
    *,
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    output_bindings: Mapping[str, str],
    inputs: Mapping[str, Any],
) -> PairwiseAgreement:
    """Compare two observations using only result relations owned by ``j``."""

    expected_outputs = {item.name for item in j.intent.outputs}
    if set(output_bindings) != expected_outputs:
        raise ValueError("Gate2 output bindings differ from canonical J outputs")
    explicit = {item.output: item for item in j.boundary.result_relations}
    j_inputs = _j_inputs(j, input_bindings, inputs)
    observations: list[OutputAgreement] = []

    for output in j.intent.outputs:
        logical_key = output.name
        runtime_key = output_bindings[logical_key]
        relation = explicit.get(
            logical_key,
            ResultRelationSpec(logical_key, ResultRelationKind.EXACT),
        )
        missing = []
        if runtime_key not in left:
            missing.append(left_name)
        if runtime_key not in right:
            missing.append(right_name)
        if missing:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"output {runtime_key!r} missing from {', '.join(missing)}",
                    relation.kind.value,
                    relation.kind is not ResultRelationKind.ONE_OF_UPDATES,
                )
            )
            continue

        lhs = np.asarray(left[runtime_key])
        rhs = np.asarray(right[runtime_key])
        expected_dtype = _dtype_key(output.dtype)
        observed_dtypes = (_dtype_key(lhs.dtype.name), _dtype_key(rhs.dtype.name))
        if observed_dtypes != (expected_dtype, expected_dtype):
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    "output dtype differs from canonical J: "
                    f"{observed_dtypes} != {(expected_dtype, expected_dtype)}",
                    relation.kind.value,
                    relation.kind is not ResultRelationKind.ONE_OF_UPDATES,
                )
            )
            continue
        if relation.kind is ResultRelationKind.ONE_OF_UPDATES:
            lhs_ok, lhs_detail = _one_of_membership(lhs, relation, j, j_inputs)
            rhs_ok, rhs_detail = _one_of_membership(rhs, relation, j, j_inputs)
            details = [
                f"{name}: {detail}"
                for name, ok, detail in (
                    (left_name, lhs_ok, lhs_detail),
                    (right_name, rhs_ok, rhs_detail),
                )
                if not ok
            ]
            observations.append(
                OutputAgreement(
                    runtime_key,
                    lhs_ok and rhs_ok,
                    None,
                    "; ".join(details),
                    relation.kind.value,
                    False,
                )
            )
            continue

        if lhs.shape != rhs.shape:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    float("inf"),
                    f"shape mismatch: {lhs.shape} != {rhs.shape}",
                    relation.kind.value,
                    True,
                )
            )
            continue
        if lhs.dtype != rhs.dtype:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"dtype mismatch: {lhs.dtype} != {rhs.dtype}",
                    relation.kind.value,
                    True,
                )
            )
            continue
        if relation.kind is ResultRelationKind.EXACT or _exact_element_domain(lhs, rhs):
            passed = bool(np.array_equal(lhs, rhs))
        else:
            assert relation.rtol is not None
            assert relation.atol is not None
            assert relation.equal_nan is not None
            passed = bool(
                np.allclose(
                    lhs,
                    rhs,
                    rtol=relation.rtol,
                    atol=relation.atol,
                    equal_nan=relation.equal_nan,
                )
            )
        observations.append(
            OutputAgreement(
                runtime_key,
                passed,
                _max_abs_error(lhs, rhs),
                "",
                relation.kind.value,
                True,
            )
        )

    return PairwiseAgreement(
        left_name,
        right_name,
        all(item.passed for item in observations),
        tuple(observations),
    )


__all__ = [
    "GATE2_COMPARISON_PROTOCOL",
    "compare_outputs",
    "validate_j_inputs",
]
