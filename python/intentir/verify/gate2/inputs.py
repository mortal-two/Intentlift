"""Deterministic value generation for SCRC Gate2 execution cases.

Gate2 input values are evidence, not incidental test fixtures.  This module
keeps the supported distributions explicit and reproducible so live SCRC
runs can record which numerical domain was exercised without falling back to
the archived JSON interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class InputDistribution(str, Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    POSITIVE = "positive"
    NEAR_ONE = "near_one"
    LARGE_DYNAMIC = "large_dynamic"
    IEEE_SPECIAL = "ieee_special"


@dataclass(frozen=True)
class BoundaryExtent:
    label: str
    extent: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Gate2 boundary extent label must not be empty")
        if isinstance(self.extent, bool) or int(self.extent) <= 0:
            raise ValueError("Gate2 boundary extent must be positive")


def boundary_1d_extents(block: int) -> tuple[BoundaryExtent, ...]:
    """Canonical SCRC shape axis from section 9.3 of the design."""

    if isinstance(block, bool) or int(block) < 3:
        raise ValueError("Gate2 boundary block must be an integer >= 3")
    block = int(block)
    return (
        BoundaryExtent("one", 1),
        BoundaryExtent("block_minus_one", block - 1),
        BoundaryExtent("block", block),
        BoundaryExtent("block_plus_one", block + 1),
        BoundaryExtent("non_divisible", 2 * block + 3),
        BoundaryExtent("far_multitile", 8 * block + 17),
    )


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


@dataclass(frozen=True)
class InputCaseTemplate:
    template_id: str
    tensors: Mapping[str, TensorInputSpec]
    scalars: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("Gate2 input template_id must not be empty")
        overlap = sorted(set(self.tensors) & set(self.scalars))
        if overlap:
            raise ValueError(
                "Gate2 input template tensor/scalar keys overlap: "
                + ", ".join(overlap)
            )
        for key, spec in self.tensors.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Gate2 tensor input keys must be non-empty strings")
            if not isinstance(spec, TensorInputSpec):
                raise TypeError("Gate2 input template tensors require TensorInputSpec")


@dataclass(frozen=True)
class GeneratedInputCase:
    case_id: str
    seed: int
    values: Mapping[str, Any]
    metadata: Mapping[str, Any]


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


def generate_input_matrix(
    templates: Sequence[InputCaseTemplate], seeds: Sequence[int]
) -> tuple[GeneratedInputCase, ...]:
    """Expand explicit shape/value templates across reproducible seeds."""

    if not templates:
        raise ValueError("Gate2 input matrix requires at least one template")
    if not seeds:
        raise ValueError("Gate2 input matrix requires at least one seed")
    template_ids = [template.template_id for template in templates]
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("Gate2 input matrix template_id values must be unique")
    normalized_seeds: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Gate2 matrix seeds must be integers")
        normalized_seeds.append(seed)
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("Gate2 input matrix seeds must be unique")

    generated: list[GeneratedInputCase] = []
    for template in templates:
        for seed in normalized_seeds:
            rng = np.random.default_rng(seed)
            values: dict[str, Any] = dict(template.scalars)
            tensor_metadata: dict[str, Any] = {}
            for key in sorted(template.tensors):
                spec = template.tensors[key]
                values[key] = generate_tensor_input(spec, rng=rng)
                tensor_metadata[key] = {
                    "shape": list(spec.shape),
                    "dtype": spec.dtype,
                    "distribution": spec.distribution.value,
                    "scale": spec.scale,
                }
            generated.append(
                GeneratedInputCase(
                    f"{template.template_id}.seed{seed}",
                    seed,
                    values,
                    {
                        **dict(template.metadata),
                        "template_id": template.template_id,
                        "seed": seed,
                        "tensors": tensor_metadata,
                    },
                )
            )
    return tuple(generated)


__all__ = [
    "BoundaryExtent",
    "GeneratedInputCase",
    "InputCaseTemplate",
    "InputDistribution",
    "TensorInputSpec",
    "boundary_1d_extents",
    "generate_input_matrix",
    "generate_tensor_input",
]
