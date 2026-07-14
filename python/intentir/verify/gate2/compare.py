"""Tolerance-based output comparison for Gate2."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .model import OutputAgreement, PairwiseAgreement, Tolerance


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
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


__all__ = ["compare_outputs"]
