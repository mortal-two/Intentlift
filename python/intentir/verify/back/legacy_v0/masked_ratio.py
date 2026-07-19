"""Archived fuzzy masked-J ratio helpers.

Current Gate1 uses actual-pi masked-J hard non-closure and never consults
these functions.  They remain only for reproducing historical reports.
"""

from __future__ import annotations


def leakage_ratio(normal_score: float, masked_score: float) -> float:
    """Return the historical ``masked / normal`` score ratio."""

    for name, value in (
        ("normal_score", normal_score),
        ("masked_score", masked_score),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative number")
    if normal_score == 0:
        return float("inf") if masked_score > 0 else 0.0
    return float(masked_score) / float(normal_score)


def masked_ablation_passes(
    normal_score: float,
    masked_score: float,
    *,
    threshold: float = 0.5,
) -> bool:
    """Apply the removed fuzzy-ratio acceptance rule."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return leakage_ratio(normal_score, masked_score) <= threshold


__all__ = ["leakage_ratio", "masked_ablation_passes"]
