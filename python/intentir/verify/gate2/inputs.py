"""Deterministic value generation for SCRC Gate2 execution cases.

Gate2 input values are evidence, not incidental test fixtures.  This module
keeps the supported distributions explicit and reproducible so live SCRC
runs can record which numerical domain was exercised without falling back to
the archived JSON interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class InputDistribution(str, Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    POSITIVE = "positive"
    NEAR_ONE = "near_one"
    LARGE_DYNAMIC = "large_dynamic"
    IEEE_SPECIAL = "ieee_special"


_DTYPE_ALIASES = {
    "bool": "bool",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "f16": "float16",
    "bf16": "float32",
    "f32": "float32",
    "f64": "float64",
}


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    key = dtype.removeprefix("torch.").lower()
    key = _DTYPE_ALIASES.get(key, key)
    try:
        return np.dtype(key)
    except TypeError as exc:
        raise ValueError(f"unsupported Gate2 input dtype {dtype!r}") from exc


@dataclass(frozen=True)
class TensorInputSpec:
    shape: tuple[int, ...]
    dtype: str
    distribution: InputDistribution = InputDistribution.NORMAL
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, InputDistribution):
            object.__setattr__(
                self, "distribution", InputDistribution(self.distribution)
            )
        if any(int(extent) < 0 for extent in self.shape):
            raise ValueError("Gate2 input extents must be non-negative")
        if not np.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Gate2 input scale must be finite and non-negative")
        _numpy_dtype(self.dtype)


def _floating_values(
    rng: np.random.Generator, spec: TensorInputSpec, dtype: np.dtype[Any]
) -> np.ndarray:
    shape = tuple(int(extent) for extent in spec.shape)
    distribution = spec.distribution
    scale = float(spec.scale)
    if distribution is InputDistribution.NORMAL:
        values = rng.standard_normal(shape) * scale
    elif distribution is InputDistribution.UNIFORM:
        values = rng.uniform(-scale, scale, size=shape)
    elif distribution is InputDistribution.POSITIVE:
        values = np.abs(rng.standard_normal(shape)) * scale
    elif distribution is InputDistribution.NEAR_ONE:
        values = 1.0 + rng.uniform(-scale, scale, size=shape)
    elif distribution is InputDistribution.LARGE_DYNAMIC:
        count = int(np.prod(shape, dtype=np.int64))
        if count == 0:
            values = np.empty(shape, dtype=np.float64)
        else:
            exponents = np.linspace(-12.0, 12.0, count, dtype=np.float64)
            signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
            values = (signs * np.power(2.0, exponents) * scale).reshape(shape)
    elif distribution is InputDistribution.IEEE_SPECIAL:
        pattern = np.asarray(
            [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan],
            dtype=np.float64,
        )
        count = int(np.prod(shape, dtype=np.int64))
        values = np.resize(pattern, count).reshape(shape)
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(distribution)
    return np.asarray(values, dtype=dtype)


def generate_tensor_input(
    spec: TensorInputSpec,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one deterministic tensor value from an explicit input spec."""

    if (seed is None) == (rng is None):
        raise ValueError("provide exactly one of seed or rng")
    generator = rng if rng is not None else np.random.default_rng(seed)
    assert generator is not None
    dtype = _numpy_dtype(spec.dtype)
    shape = tuple(int(extent) for extent in spec.shape)

    if np.issubdtype(dtype, np.floating):
        return _floating_values(generator, spec, dtype)
    if spec.distribution in {
        InputDistribution.LARGE_DYNAMIC,
        InputDistribution.IEEE_SPECIAL,
        InputDistribution.NEAR_ONE,
    }:
        raise ValueError(
            f"distribution {spec.distribution.value!r} requires a floating dtype"
        )
    if np.issubdtype(dtype, np.bool_):
        return generator.integers(0, 2, size=shape, dtype=np.int8).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        bound = max(1, int(spec.scale))
        low = 0 if spec.distribution is InputDistribution.POSITIVE else -bound
        return generator.integers(low, bound + 1, size=shape, dtype=dtype)
    raise ValueError(f"unsupported Gate2 input dtype {spec.dtype!r}")


__all__ = [
    "InputDistribution",
    "TensorInputSpec",
    "generate_tensor_input",
]
