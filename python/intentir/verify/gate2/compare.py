"""Tolerance-based output comparison for Gate2."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .model import OutputAgreement, PairwiseAgreement, Tolerance


GATE2_COMPARISON_PROTOCOL = "exact_integer_tolerant_float_v2"


def _exact_element_domain(left: np.ndarray, right: np.ndarray) -> bool:
    """Return true when tolerance/float conversion must not participate."""

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
        # Python integers avoid signed overflow (notably around int64 limits),
        # and bool naturally maps to 0/1 for an auditable exact delta.
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


def compare_outputs(
    left_name: str,
    left: Mapping[str, np.ndarray],
    right_name: str,
    right: Mapping[str, np.ndarray],
    output_keys: Sequence[str],
    tolerances: Mapping[str, Tolerance] | None = None,
) -> PairwiseAgreement:
    tolerances = dict(tolerances or {})
    observations: list[OutputAgreement] = []
    for key in output_keys:
        if key not in left or key not in right:
            missing = []
            if key not in left:
                missing.append(left_name)
            if key not in right:
                missing.append(right_name)
            observations.append(
                OutputAgreement(
                    key,
                    False,
                    None,
                    f"output {key!r} missing from {', '.join(missing)}",
                )
            )
            continue
        lhs = np.asarray(left[key])
        rhs = np.asarray(right[key])
        tolerance = tolerances.get(key, Tolerance())
        if lhs.shape != rhs.shape:
            observations.append(
                OutputAgreement(
                    key,
                    False,
                    float("inf"),
                    f"shape mismatch: {lhs.shape} != {rhs.shape}",
                )
            )
            continue
        if _exact_element_domain(lhs, rhs):
            passed = bool(np.array_equal(lhs, rhs))
        else:
            passed = bool(
                np.allclose(
                    lhs,
                    rhs,
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                    equal_nan=tolerance.equal_nan,
                )
            )
        observations.append(
            OutputAgreement(key, passed, _max_abs_error(lhs, rhs))
        )
    return PairwiseAgreement(
        left_name,
        right_name,
        all(item.passed for item in observations),
        tuple(observations),
    )


__all__ = ["GATE2_COMPARISON_PROTOCOL", "compare_outputs"]
